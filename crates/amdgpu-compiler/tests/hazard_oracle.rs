//! Hazard oracle: for every (producer, consumer, operand slot) pair of table rows, run the pair
//! back-to-back with a data dependency through the pinned llc tail and record the s_nop /
//! s_waitcnt / s_delay_alu LLVM inserts; compare with the wait states our own hazard pass
//! provides. Writes a markdown matrix to $AMDC_ORACLE_OUT when set.

mod common;

use amdgpu_compiler::backend::llvm20::Table;
use amdgpu_compiler::backend::{llvm_mir, LlvmTail, LlvmTools, ALL_MOPS};
use amdgpu_compiler::hazards;
use amdgpu_compiler::mir::*;
use common::*;

struct Case {
    producer: String,
    consumer: String,
    slot: Option<usize>,
}

fn is_special(op: MOp) -> bool {
    matches!(op, MOp::SNop | MOp::SWaitVm | MOp::SWaitLgkm | MOp::SEndpgm | MOp::SBarrier)
}

#[test]
fn hazard_matrix() {
    let tools = LlvmTools::locate();
    if !std::process::Command::new(tools.tool("llc")).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIPPED: llc not runnable");
        return;
    }
    let t = Table::load();
    let mut keys: Vec<&String> = t.rows.keys().filter(|k| !k.starts_with("T_") && !k.starts_with("X_")).collect();
    keys.sort();
    let variant = |k: &str| -> (MOp, &amdgpu_compiler::backend::llvm20::Row) {
        let name = k.split(':').next().unwrap();
        (*ALL_MOPS.iter().find(|o| format!("{o:?}") == name).unwrap(), &t.rows[k])
    };
    let mut sm = Samples::new();
    let mut blocks = Vec::new();
    let mut cases: Vec<Case> = Vec::new();
    let mut own_nops: Vec<u32> = Vec::new();
    for pk in &keys {
        let (pop, prow) = variant(pk);
        if is_special(pop) || (prow.text.len() == 1 && prow.text[0] == "DROP") {
            continue;
        }
        for ck in &keys {
            let (cop, crow) = variant(ck);
            if is_special(cop) || (crow.text.len() == 1 && crow.text[0] == "DROP") {
                continue;
            }
            // slots to link: every register source whose class equals the producer's dst class;
            // if the producer has no dst (vcc/scc setters) one implicit-dependency case.
            let slots: Vec<Option<usize>> = match prow.dst {
                Some(dc) => crow
                    .srcs
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s, amdgpu_compiler::backend::llvm20::SigClass::Reg(c) if *c == dc))
                    .map(|(i, _)| Some(i))
                    .collect(),
                None => vec![None],
            };
            for slot in slots {
                sm.reset_phys();
                let p = sm.inst(pop, prow, None);
                let link = match (slot, p.dst) {
                    (Some(s), Some(d)) => Some((s, d)),
                    _ => None,
                };
                let c = sm.inst(cop, crow, link);
                let mut b = Block::new(blocks.len() as u32);
                b.insts.push(p);
                b.insts.push(c);
                // own model
                let mut one = func("own", vec![b.clone()], sm.vregs.clone());
                hazards::fix_value_hazards(&mut one);
                let nops: u32 = one.blocks[0]
                    .insts
                    .iter()
                    .filter(|i| i.op == MOp::SNop)
                    .map(|i| i.imm as u32 + 1)
                    .sum();
                own_nops.push(nops);
                blocks.push(b);
                cases.push(Case { producer: pk.to_string(), consumer: ck.to_string(), slot });
            }
        }
    }
    eprintln!("hazard oracle: {} pairs", cases.len());
    let nb = blocks.len();
    for (i, b) in blocks.iter_mut().enumerate() {
        b.label = i as u32;
        b.term = if i + 1 < nb { Term::Goto(i as u32 + 1) } else { Term::Endpgm };
    }
    let m = func("oracle", blocks, sm.vregs.clone());
    let a = sm.alloc();
    // Per-block physical numbering is not kept in `sm.phys` (reset_phys): rebuild per block instead.
    let _ = (&m, &a);
    // `phys` entries are per vreg, so blocks may share physical registers: that is fine because
    // every block ends in s_endpgm and no value flows between blocks.
    let mir = llvm_mir::export_with(&m, &a, true).expect("export").mir;
    let mut tail = LlvmTail::new();
    tail.skip_lock = true;
    let asm = tail.run_llc("oracle", &mir, "gcn-create-vopd").expect("llc");
    if let Ok(p) = std::env::var("AMDC_ORACLE_OUT") {
        if !p.is_empty() {
            let _ = std::fs::write(format!("{p}.s"), &asm);
            let _ = std::fs::write(format!("{p}.mir"), &mir);
        }
    }
    // Split by s_setprio markers.
    let mut per: Vec<Vec<String>> = vec![Vec::new(); cases.len()];
    let mut cur: Option<usize> = None;
    for line in asm.lines() {
        let l = line.split(';').next().unwrap().trim();
        if let Some(n) = l.strip_prefix("s_setprio ") {
            let n = n.trim();
            cur = match n.strip_prefix("0x") {
                Some(h) => usize::from_str_radix(h, 16).ok(),
                None => n.parse::<usize>().ok(),
            };
            continue;
        }
        if l.is_empty() || l.starts_with('.') || l.ends_with(':') {
            continue;
        }
        if let Some(i) = cur {
            if i < per.len() {
                per[i].push(l.to_string());
            }
        }
    }
    let mut rows = Vec::new();
    let (mut missing, mut total_llvm) = (0, 0);
    for (i, c) in cases.iter().enumerate() {
        let body = &per[i];
        let mut nops = 0u32;
        let mut waits = Vec::new();
        for l in body {
            if l == "s_nop 15" {
                continue; // block separator
            }
            if let Some(n) = l.strip_prefix("s_nop ") {
                nops += n.trim().parse::<u32>().unwrap_or(0) + 1;
            } else if l.starts_with("s_waitcnt") || l.starts_with("s_delay_alu") {
                waits.push(l.clone());
            }
        }
        if nops > 0 || !waits.is_empty() {
            total_llvm += 1;
        }
        let own = own_nops[i];
        let verdict = if nops > own { missing += 1; "LLVM>own" } else if nops < own { "own>LLVM" } else { "=" };
        rows.push(format!(
            "| {} | {}{} | {} | {} | {} | {} |",
            c.producer,
            c.consumer,
            c.slot.map(|s| format!(" (src{s})")).unwrap_or_default(),
            nops,
            own,
            verdict,
            waits.join("; ")
        ));
    }
    eprintln!("pairs with LLVM-inserted filler: {total_llvm}; LLVM needs more nops than own for {missing}");
    if let Ok(p) = std::env::var("AMDC_ORACLE_OUT") {
        let mut out = String::from("| producer | consumer | LLVM nop cycles | own nop cycles | verdict | other LLVM filler |\n|---|---|---|---|---|---|\n");
        for r in rows {
            out.push_str(&r);
            out.push('\n');
        }
        std::fs::write(p, out).unwrap();
    }
    assert!(cases.len() > 100);
}