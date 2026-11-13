//! Exact authored mode (docs/exact-mode.md), step E1: `amdc exact-report` - per IR statement, the machine instructions its lowering
//! produced in the final code, with the ones the back end added on its own marked (the instructions an exact region may not contain).

use crate::mir::*;

/// Why an instruction is not the author's (None = it is the statement's own instruction).
pub fn synthetic_kind(m: &MFunction, stmts: &[amdgpu_dsl::Stmt], i: &MInst) -> Option<&'static str> {
    use amdgpu_dsl::{Op, Stmt};
    if i.stmt == 0 {
        return Some(match i.op {
            MOp::SWaitVm | MOp::SWaitLgkm => "wait (added after lowering)",
            MOp::SNop => "s_nop (hazard pass)",
            _ => "added after lowering",
        });
    }
    let st = stmts.get(i.stmt as usize - 1);
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
pub fn report(f: &amdgpu_dsl::Function, stmts: &[amdgpu_dsl::Stmt], m: &MFunction, a: &crate::regalloc::Alloc, region: Option<&str>) -> String {
    let mut out = String::new();
    let in_region = |id: u16| region.map_or(true, |r| id > 0 && m.reg_hints.regions.get(id as usize - 1).map_or(false, |(n, _)| n == r));
    if region.is_some() && !m.reg_hints.regions.iter().any(|(n, _)| Some(n.as_str()) == region) {
        return format!("no region `{}` in {}\n", region.unwrap(), f.name);
    }
    let (mut total, mut synth) = (0usize, 0usize);
    let mut kinds: std::collections::BTreeMap<&'static str, usize> = std::collections::BTreeMap::new();
    let mut last_stmt = u32::MAX;
    let mut inside = region.is_none();
    for b in &m.blocks {
        out.push_str(&format!("BB{}:\n", b.label));
        for i in &b.insts {
            // an instruction added after lowering (region 0) belongs to the region it sits in
            if i.stmt != 0 {
                inside = in_region(i.region);
            } else if region.is_none() {
                inside = true;
            }
            if !inside {
                continue;
            }
            if i.stmt != last_stmt {
                last_stmt = i.stmt;
                let s = if i.stmt == 0 { "(added after lowering)".to_string() } else { stmts.get(i.stmt as usize - 1).map(|s| format!("{s:?}")).unwrap_or_default() };
                let s: String = s.chars().take(140).collect();
                out.push_str(&format!("  stmt {}: {}\n", if i.stmt == 0 { "-".to_string() } else { (i.stmt - 1).to_string() }, s));
            }
            let text = crate::emit::inst_lines(a, i).join(" ; ");
            total += 1;
            match synthetic_kind(m, stmts, i) {
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

// ------------------------------------------------------------------------------------------------ E2: verification by check-mode passes

/// One violation found by a check-mode pass: the pass, where it would insert, and what.
#[derive(Clone, Debug, PartialEq)]
pub struct Violation {
    pub rule: &'static str,
    pub block: u32,
    /// the instruction (final text) the insertion would precede (None = at the block end)
    pub before: Option<String>,
    pub after: Option<String>,
    pub needs: String,
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] block {}: `{}` needed between `{}` and `{}`",
            self.rule,
            self.block,
            self.needs,
            self.after.as_deref().unwrap_or("<block start>"),
            self.before.as_deref().unwrap_or("<block end>")
        )
    }
}

fn same(a: &MInst, b: &MInst) -> bool {
    a.op == b.op && a.dst == b.dst && a.imm == b.imm && format!("{:?}", a.srcs) == format!("{:?}", b.srcs) && a.skip_m0 == b.skip_m0
}

/// The instructions `after` has and `before` lacks (the insertions of a fixing pass), as violations of `rule`.
fn diff(rule: &'static str, before: &MFunction, after: &MFunction, a: &crate::regalloc::Alloc, out: &mut Vec<Violation>) {
    let txt = |i: &MInst| crate::emit::inst_lines(a, i).join(" ; ");
    for (bb, ba) in before.blocks.iter().zip(after.blocks.iter()) {
        let (x, y) = (&bb.insts, &ba.insts);
        let (mut i, mut j) = (0usize, 0usize);
        while j < y.len() {
            if i < x.len() && same(&x[i], &y[j]) {
                i += 1;
                j += 1;
                continue;
            }
            // y[j] was inserted (fixing passes only insert, or change an instruction in place)
            out.push(Violation {
                rule,
                block: ba.label,
                before: x.get(i).map(txt),
                after: if i > 0 { x.get(i - 1).map(txt) } else { None },
                needs: txt(&y[j]),
            });
            if i < x.len() && j + 1 < y.len() && same(&x[i], &y[j + 1]) {
                j += 1;
            } else {
                // changed in place: skip both
                i += 1;
                j += 1;
            }
        }
    }
    if before.blocks.len() != after.blocks.len() {
        out.push(Violation { rule, block: 0, before: None, after: None, needs: format!("{} blocks added", after.blocks.len() as i64 - before.blocks.len() as i64) });
    }
}

/// Verifies a final (allocated) function without changing it: every fixing pass runs on a copy, and whatever it would insert is a
/// violation (docs/exact-mode.md section 4). `strict` = the scheduled path's block-end MFMA padding.
pub fn verify_final(m: &MFunction, a: &crate::regalloc::Alloc, strict: bool) -> Vec<Violation> {
    let mut out = Vec::new();
    let run = |rule: &'static str, f: &dyn Fn(&mut MFunction), out: &mut Vec<Violation>| {
        let mut c = m.clone();
        f(&mut c);
        diff(rule, m, &c, a, out);
    };
    run("waits", &|c| {
        crate::sched::waits::insert_waits(c);
    }, &mut out);
    run("hazard-value", &|c| crate::hazards::fix_value_hazards_opts(c, strict), &mut out);
    run("hazard-inflight-waw", &|c| {
        crate::hazards::pad_inflight_waw(c, a);
    }, &mut out);
    run("hazard-phys-war", &|c| crate::hazards::pad_physical_war(c, a), &mut out);
    run("m0-pad", &|c| { crate::sched::m0::pad(c); }, &mut out);
    run("hazard-valu-sgpr", &|c| {
        crate::hazards::pad_valu_sgpr(c, a);
    }, &mut out);
    run("hazard-soft-clause", &|c| {
        crate::hazards::pad_soft_clause(c, a);
    }, &mut out);
    out
}
