//! Hazard oracle: for every (producer, consumer, operand slot) pair of table rows, run the pair
//! back-to-back with a data dependency through the pinned llc tail and record the s_nop /
//! s_waitcnt / s_delay_alu LLVM inserts; compare with the wait states our own hazard pass
//! provides. Writes a markdown matrix to $AMDC_ORACLE_OUT when set.
//!
//! ENFORCING: our model is the weakest production path (value hazards without the scheduled path's `strict` extras) plus every post-RA pass;
//! a pair where LLVM inserts more wait states than we do fails unless the excess is explained by an entry of `allowance` (each with its
//! evidence: a hardware probe, or the upstream LLVM rule showing the pinned LLVM is over-conservative on gfx950).

mod common;

use amdgpu_codegen::backend::llvm20::Table;
use amdgpu_codegen::backend::{llvm_mir, LlvmTail, LlvmTools, ALL_MOPS};
use amdgpu_codegen::hazards;
use amdgpu_codegen::mir::*;
use common::*;

struct Case {
    producer: String,
    consumer: String,
    slot: Option<usize>,
}

fn is_special(op: MOp) -> bool {
    // the MUBUF ops need an S4 descriptor quad: a 128-bit SGPR live-in is not expressible in the oracle's per-block liveins (they are covered by the one-row-per-op
    // machine-verifier tests and the hardware probes instead)
    matches!(op, MOp::SNop | MOp::SWaitVm | MOp::SWaitLgkm | MOp::SEndpgm | MOp::SBarrier)
        || matches!(
            op,
            MOp::SBufferRsrc | MOp::SBufferRsrcDyn | MOp::BufferLoadDwordX1 | MOp::BufferLoadDwordX2 | MOp::BufferLoadDwordX4 | MOp::BufferStoreDwordX1 | MOp::BufferStoreDwordX2 | MOp::BufferStoreDwordX4
        )
}

// 167k pairs through every hazard pass: ~1.5 min in release, tens of minutes in debug. Debug `cargo test` skips it; the node gate runs it in release
// (python/tests/run_all.sh: `cargo test --release -p amdgpu-codegen --test hazard_oracle -- --include-ignored`).
#[test]
#[cfg_attr(debug_assertions, ignore = "release only: cargo test --release -p amdgpu-codegen --test hazard_oracle (run by python/tests/run_all.sh)")]
fn hazard_matrix() {
    let tools = LlvmTools::locate();
    if !std::process::Command::new(tools.tool("llc")).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIPPED: llc not runnable");
        return;
    }
    let t = Table::load();
    let mut keys: Vec<&String> = t.rows.keys().filter(|k| !k.starts_with("T_") && !k.starts_with("X_")).collect();
    keys.sort();
    let variant = |k: &str| -> (MOp, &amdgpu_codegen::backend::llvm20::Row) {
        let name = k.split(|c| c == ':' || c == '@').next().unwrap();
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
                // v_mov from a tuple dword: any tuple class feeds it (covers MFMA results).
                Some(dc) if cop == MOp::VReadSub && dc.dwords() >= 2 && dc.is_vector() => vec![Some(0)],
                Some(dc) => crow
                    .srcs
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s, amdgpu_codegen::backend::llvm20::SigClass::Reg(c) if *c == dc))
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
                // the post-RA passes of both bodies (pass::SCHEDULED / UNSCHEDULED), on this pair's allocation
                let pa = sm.alloc();
                hazards::pad_inflight_waw(&mut one, &pa);
                hazards::pad_physical_war(&mut one, &pa);
                hazards::pad_valu_sgpr(&mut one, &pa);
                hazards::pad_soft_clause(&mut one, &pa);
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
    if let Ok(p) = std::env::var("AMDC_ORACLE_OUT") {
        if !p.is_empty() {
            let _ = std::fs::write(format!("{p}.mir"), &mir); // written BEFORE llc so a failing llc can be reproduced by hand
        }
    }
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
        // the block label comment (`; %bb.N:`) switches the case; a `s_setprio N` inside a case is an instruction under test, not a marker
        if let Some(n) = line.trim().strip_prefix("; %bb.").and_then(|r| r.strip_suffix(':')) {
            cur = n.trim().parse::<usize>().ok();
            continue;
        }
        let l = line.split(';').next().unwrap().trim();
        if let (Some(n), None) = (l.strip_prefix("s_setprio "), cur) {
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
    let mut explained: std::collections::BTreeMap<&'static str, usize> = Default::default();
    let mut unexplained: Vec<String> = Vec::new();
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
            } else if (l.starts_with("s_waitcnt") && l != "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)") || l.starts_with("s_delay_alu") {
                waits.push(l.clone());
            }
        }
        if nops > 0 || !waits.is_empty() {
            total_llvm += 1;
        }
        let own = own_nops[i];
        let verdict = if nops > own { missing += 1; "LLVM>own" } else if nops < own { "own>LLVM" } else { "=" };
        if nops > own {
            let (slack, why) = allowance(&c.producer, &c.consumer);
            if nops <= own + slack {
                for w in why {
                    *explained.entry(w).or_default() += 1;
                }
            } else {
                unexplained.push(format!("{} -> {}{}: LLVM {} own {} (allowance {})", c.producer, c.consumer, c.slot.map(|s| format!(" (src{s})")).unwrap_or_default(), nops, own, slack));
            }
        }
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
    if let Some(p) = std::env::var("AMDC_ORACLE_OUT").ok().filter(|p| !p.is_empty()) {
        let mut out = String::from("| producer | consumer | LLVM nop cycles | own nop cycles | verdict | other LLVM filler |\n|---|---|---|---|---|---|\n");
        for r in rows {
            out.push_str(&r);
            out.push('\n');
        }
        std::fs::write(p, out).unwrap();
    }
    for (w, n) in &explained {
        eprintln!("explained ({n} pairs): {w}");
    }
    assert!(cases.len() > 100);
    assert!(unexplained.is_empty(), "{} pair(s) where LLVM inserts more wait states than our hazard passes, not explained by `allowance`:\n{}", unexplained.len(), unexplained.iter().take(40).cloned().collect::<Vec<_>>().join("\n"));
}

fn base(k: &str) -> &str {
    k.split(|c| c == ':' || c == '@' || c == ' ').next().unwrap()
}

/// The wait states LLVM may insert beyond ours for a (producer, consumer) row of the table, and why. Every class cites its evidence.
fn allowance(producer: &str, consumer: &str) -> (u32, Vec<&'static str>) {
    let (p, c) = (base(producer), base(consumer));
    let mut slack = 0;
    let mut why = Vec::new();
    // VALU writes VCC -> VALU reads VCC (LLVM checkVALUHazards, hasVDecCoExecHazard: 2): on gfx950 the hardware forwards VCC - tools/vcc_hazard_probe.sh
    // (v_cmp / v_add_co -> v_addc / v_cndmask with 0, 1, 2 wait states: all correct, 3 runs). Our VPtrAdd emits v_add_co + v_addc back to back, so
    // LLVM's `s_nop 1` inside it counts once per VPtrAdd of the pair; a VCC writer feeding a VCC reader counts once.
    const VCC_W: &[&str] = &["VCmpEqF32", "VCmpEqI32", "VCmpEqU32", "VCmpGeF32", "VCmpGeI32", "VCmpGeU32", "VCmpGtF32", "VCmpGtI32", "VCmpGtU32", "VCmpLeF32", "VCmpLeI32", "VCmpLeU32", "VCmpLgF32", "VCmpLtF32", "VCmpLtI32", "VCmpLtU32", "VCmpNeF32", "VCmpNeI32", "VCmpOF32", "VCmpUF32", "VAddCoU32", "VAddcCoU32", "VMadU64U32", "VPtrAdd"];
    const VCC_R: &[&str] = &["VAddcCoU32", "VCndmask", "VPtrAdd"];
    let vcc_why = "VALU VCC write -> VALU VCC read: interlocked on gfx950 (tools/vcc_hazard_probe.sh); LLVM hasVDecCoExecHazard is over-conservative for VCC";
    for x in [p, c] {
        if x == "VPtrAdd" {
            slack += 2;
            why.push(vcc_why);
        }
    }
    if (VCC_W.contains(&p) || p.starts_with("VCmp")) && VCC_R.contains(&c) && !(p == "VPtrAdd" || c == "VPtrAdd") {
        slack += 2;
        why.push(vcc_why);
    }
    // VALU lane-mask write (VOP3 compare into an SGPR pair) -> v_cndmask_b32_e64 reading it as the mask (LLVM VALUWriteSGPRVALURead: 2; our
    // target row `valu-sgpr-mask`: 1) - tools/lanemask_hazard_probe.sh: correct at 0 / 1 / 2 wait states, 512 threads, 5 runs
    if (p == "VCmpMaskLtI32" || p == "VCmpMaskG") && c == "VCndmaskM" {
        slack += 1;
        why.push("VALU lane-mask write -> v_cndmask mask read: 1 wait state suffices on gfx950 (tools/lanemask_hazard_probe.sh); LLVM asks 2");
    }
    // VOP3P producers: the pinned LLVM reads the packed instruction's op_sel_hi bit as the VOP3 destination op_sel (SISrcMods::DST_OP_SEL is the same
    // bit) and inserts the dst-sel forwarding nop; upstream GCNHazardRecognizer::getDstSelForwardingOperand excludes VOP3P ("VOP3P uses this bit for
    // source op_sel_hi"). A packed instruction writes the whole dword.
    if p.starts_with("VPk") {
        slack += 1;
        why.push("VOP3P producer: pinned-LLVM dst-sel forwarding false positive (upstream getDstSelForwardingOperand excludes VOP3P)");
    }
    // 16x16x128 f8f6f4 with FP4 / FP6 operands: LLVM uses the 8-pass count for every format (12 wait states); the hardware needs 7 before a VALU reader
    // and 1 before a VMEM reader (tools/mfma_lat_probe.sh, 3 runs); our table uses 8.
    if (p.starts_with("VMfmaPlain16x16x128A") || p.starts_with("VMfmaScale16x16x128A")) && (p.contains("A4") || p.contains("A6")) {
        slack += 4;
        why.push("16x16x128 FP4/FP6 MFMA: 4-pass latency (hardware 7 / 1 wait states, tools/mfma_lat_probe.sh); LLVM counts 8 passes");
    }
    (slack, why)
}