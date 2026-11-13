//! Exact authored mode (docs/exact-mode.md), step E1: `amdc exact-report` - per IR statement, the machine instructions its lowering
//! produced in the final code, with the ones the back end added on its own marked (the instructions an exact region may not contain).

use crate::mir::*;

/// Why an instruction is not the author's (None = it is the statement's own instruction).
pub fn synthetic_kind(m: &MFunction, f: &amdgpu_dsl::Function, i: &MInst) -> Option<&'static str> {
    use amdgpu_dsl::{Op, Stmt};
    if i.stmt == 0 {
        return Some(match i.op {
            MOp::SWaitVm | MOp::SWaitLgkm => "wait (added after lowering)",
            MOp::SNop => "s_nop (hazard pass)",
            _ => "added after lowering",
        });
    }
    let st = f.stmts().get(i.stmt as usize - 1);
    let op = match st {
        Some(Stmt::Let { op, .. }) => Some(op),
        _ => None,
    };
    if i.auto {
        return Some("automatic wait");
    }
    match i.op {
        MOp::SNop if !matches!(op, Some(Op::Asm { .. })) => Some("s_nop"),
        MOp::VMovB32 | MOp::SMovB32 if i.srcs.iter().all(|s| matches!(s, Src::Imm(_))) && !matches!(op, Some(Op::ConstI32(_) | Op::ConstF32(_))) => {
            Some("constant materialization")
        }
        MOp::VMovB32 | MOp::SMovB32 if i.srcs.iter().any(|s| matches!(s, Src::Reg(_))) && !matches!(st, Some(Stmt::SetVar { .. })) => Some("register copy"),
        MOp::VMovB32 | MOp::SMovB32 if matches!(st, Some(Stmt::SetVar { .. })) => Some("loop-variable copy"),
        MOp::VMovFromS | MOp::VMovFromSPair => Some("SGPR -> VGPR copy"),
        MOp::VReadFirstLane if !matches!(op, Some(Op::ReadFirstLane(_))) => Some("readfirstlane"),
        _ if !i.skip_m0 && is_dma(i.op) => {
            Some("implicit M0 write (+ s_nop) of an LDS-DMA")
        }
        _ => {
            let _ = m;
            None
        }
    }
}

fn is_dma(op: MOp) -> bool {
    format!("{op:?}").contains("Lds") && (format!("{op:?}").starts_with("Buffer") || format!("{op:?}").starts_with("Global"))
}

/// The report text: per statement, its instructions in final order (an instruction of an earlier statement emitted later, e.g. a hoisted
/// or cached value, starts a new group).
pub fn report(f: &amdgpu_dsl::Function, m: &MFunction, a: &crate::regalloc::Alloc, region: Option<&str>) -> String {
    let mut out = String::new();
    let region_id = region.and_then(|r| m.reg_hints.regions.iter().position(|(n, _)| n == r).map(|i| i as u16 + 1));
    if region.is_some() && region_id.is_none() {
        return format!("no region `{}` in {}\n", region.unwrap(), f.name);
    }
    let (mut total, mut synth) = (0usize, 0usize);
    let mut kinds: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    let mut last_stmt = u32::MAX;
    for b in &m.blocks {
        out.push_str(&format!("BB{}:\n", b.label));
        for i in &b.insts {
            if let Some(r) = region_id {
                if i.region != r {
                    continue;
                }
            }
            if i.stmt != last_stmt {
                last_stmt = i.stmt;
                let s = if i.stmt == 0 { "(added after lowering)".to_string() } else { f.stmts().get(i.stmt as usize - 1).map(|s| format!("{s:?}")).unwrap_or_default() };
                let s: String = s.chars().take(140).collect();
                out.push_str(&format!("  stmt {}: {}\n", if i.stmt == 0 { "-".to_string() } else { (i.stmt - 1).to_string() }, s));
            }
            let text = crate::emit::inst_lines(a, i).join(" ; ");
            total += 1;
            match synthetic_kind(m, f, i) {
                Some(k) => {
                    synth += 1;
                    *kinds.entry(k).or_default() += 1;
                    out.push_str(&format!("      {text:<70} <- {k}\n"));
                }
                None => out.push_str(&format!("      {text}\n")),
            }
        }
    }
    out.push_str(&format!("{}: {total} instructions, {synth} not from the author's statements", f.name));
    for (k, n) in &kinds {
        out.push_str(&format!("; {k}: {n}"));
    }
    out.push('\n');
    out
}
