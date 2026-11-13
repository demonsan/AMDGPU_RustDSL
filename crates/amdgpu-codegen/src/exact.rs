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
        MOp::SNop if !matches!(op, Some(Op::Asm { .. })) && !matches!(st, Some(Stmt::SNop { .. })) => Some("s_nop"),
        // an authored move (`mov_v` / `mov_s`) is the author's instruction
        MOp::VMovB32 | MOp::SMovB32 | MOp::VMovFromS if matches!(op, Some(Op::MovV { .. } | Op::MovS { .. })) => None,
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
    a.op == b.op && a.dst == b.dst && a.imm == b.imm && a.neg == b.neg && format!("{:?}", a.srcs) == format!("{:?}", b.srcs) && a.skip_m0 == b.skip_m0
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
            // the next original instruction further on: everything before it was inserted (passes insert runs of nops / waits)
            match (j + 1..y.len().min(j + 24)).find(|&k| i < x.len() && same(&x[i], &y[k])) {
                Some(k) => {
                    for q in j + 1..k {
                        out.push(Violation {
                            rule,
                            block: ba.label,
                            before: x.get(i).map(txt),
                            after: if i > 0 { x.get(i - 1).map(txt) } else { None },
                            needs: txt(&y[q]),
                        });
                    }
                    j = k;
                }
                None => {
                    // changed in place: skip both
                    i += 1;
                    j += 1;
                }
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

// ------------------------------------------------------------------------------------------------ E2: register legality of an allocation

/// No value is clobbered: for every instruction, each physical register dword it writes is not holding another value that is still live
/// (read later on some path). Per dword (tuple parts, in-place sub-register writes), across the CFG (loop back edges included). Works on
/// any allocation: the compiler's, or the author's pins in exact mode. Returns one message per clobber (instruction, register, the
/// value whose live range it cuts).
pub fn check_registers(m: &MFunction, a: &crate::regalloc::Alloc) -> Vec<String> {
    check_registers_with(m, a, &[])
}

/// `check_registers` with values that hold a hardware-initialised register at entry (the workitem id in v0, the kernarg pointer, the
/// workgroup ids): they count as written from the start.
pub fn check_registers_with(m: &MFunction, a: &crate::regalloc::Alloc, entry: &[VReg]) -> Vec<String> {
    use crate::sched::effects::{effects, Slot};
    use std::collections::{HashMap, HashSet};
    let nb = m.blocks.len();
    // physical key of a dword slot (vreg tuple * 16 + dword)
    // physical key of a half slot: (scalar file, register, half)
    let phys_of = |s: Slot| -> Option<(bool, u32)> {
        let d = s / 2;
        let (t, off) = (d / 16, d % 16);
        let p = *a.phys.get(t as usize)?;
        if p == u32::MAX {
            return None;
        }
        Some((a.scalar.get(t as usize).copied().unwrap_or(false), (p + off) * 2 + s % 2))
    };
    let succ = crate::sched::liveness::successors(m);
    let effs: Vec<Vec<(Vec<Slot>, Vec<Slot>)>> = m
        .blocks
        .iter()
        .map(|b| b.insts.iter().map(|i| {
            // 16-bit halves: slot * 2 + half. A 16-bit half writer writes its half only; the dword read the effects model adds for the
            // surviving half is not a use of the old value
            let e = effects(m, i);
            let at = crate::op_attrs::attrs(i.op);
            let halves = |v: &Vec<Slot>| v.iter().copied().filter(|s| *s < (u32::MAX - 16_384)).flat_map(|s| [s * 2, s * 2 + 1]).collect::<Vec<_>>();
            let mut reads = halves(&e.reads);
            let mut writes = halves(&e.writes);
            if at.regfx == crate::op_attrs::RegFx::PartialWrite {
                if let Some(h) = at.half {
                    let hb = if h == crate::op_attrs::Half::Lo { 0 } else { 1 };
                    let dst: Vec<Slot> = i.dst.map(|d| crate::sched::effects::reg_slots(m, d)).unwrap_or_default();
                    writes.retain(|x| !dst.contains(&(x / 2)) || x % 2 == hb);
                    // the destination read is the preserved half, unless the destination is also a real source
                    reads.retain(|x| !dst.contains(&(x / 2)));
                }
            }
            (reads, writes)
        }).collect())
        .collect();
    // live-in per block by slot
    let mut live_in: Vec<HashSet<Slot>> = vec![HashSet::new(); nb];
    loop {
        let mut changed = false;
        for bi in (0..nb).rev() {
            let mut live: HashSet<Slot> = succ[bi].iter().flat_map(|&s| live_in[s].iter().copied()).collect();
            for (r, w) in effs[bi].iter().rev() {
                for s in w {
                    live.remove(s);
                }
                live.extend(r.iter().copied());
            }
            if live != live_in[bi] {
                live_in[bi] = live;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // a dword only holds a value once something wrote it: reads of never-written dwords (unused lanes of a tuple, an accumulator read
    // before its first write on the first iteration) do not make the register busy. Forward "may be defined" sets per block entry.
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (b, ss) in succ.iter().enumerate() {
        for &s in ss {
            preds[s].push(b);
        }
    }
    let mut def_in: Vec<HashSet<Slot>> = vec![HashSet::new(); nb];
    if nb > 0 {
        for &r in entry {
            for s in crate::sched::effects::reg_slots(m, r) {
                def_in[0].insert(s * 2);
                def_in[0].insert(s * 2 + 1);
            }
        }
    }
    loop {
        let mut changed = false;
        for bi in 0..nb {
            let mut d: HashSet<Slot> = preds[bi].iter().flat_map(|&p| {
                let mut x = def_in[p].clone();
                for (_, w) in &effs[p] {
                    x.extend(w.iter().copied());
                }
                x
            }).collect();
            d.extend(def_in[bi].iter().copied());
            if d.len() != def_in[bi].len() {
                def_in[bi] = d;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let first_def: Vec<HashMap<Slot, usize>> = effs
        .iter()
        .map(|b| {
            let mut fd = HashMap::new();
            for (ii, (_, w)) in b.iter().enumerate() {
                for s in w {
                    fd.entry(*s).or_insert(ii);
                }
            }
            fd
        })
        .collect();
    let name = |s: Slot| -> String {
        let d = s / 2;
        let t = d / 16;
        let n = m.reg_hints.names.get(&t).cloned().unwrap_or_else(|| format!("r{t}"));
        format!("{n}[{}]{}", d % 16, if s % 2 == 0 { ".lo" } else { ".hi" })
    };
    let mut out = Vec::new();
    for bi in 0..nb {
        let mut live: HashSet<Slot> = succ[bi].iter().flat_map(|&s| live_in[s].iter().copied()).collect();
        for (ii, (r, w)) in effs[bi].iter().enumerate().rev() {
            // after this instruction: `live`; a written dword may not hold another live value
            let mut by_phys: HashMap<(bool, u32), Vec<Slot>> = HashMap::new();
            for s in &live {
                if let Some(p) = phys_of(*s) {
                    by_phys.entry(p).or_default().push(*s);
                }
            }
            for s in w {
                if let Some(p) = phys_of(*s) {
                    for o in by_phys.get(&p).map(|v| v.as_slice()).unwrap_or(&[]) {
                        let defined = def_in[bi].contains(o) || first_def[bi].get(o).map_or(false, |&d| d < ii);
                        if o != s && !w.contains(o) && defined {
                            let inst = &m.blocks[bi].insts[ii];
                            out.push(format!(
                                "block {} inst {ii} ({:?}) writes {}{} (as {}) while {} still lives there",
                                m.blocks[bi].label, inst.op, if p.0 { "s" } else { "v" }, (p.1 / 2) & !AGPR_TAG, name(*s), name(*o)
                            ));
                        }
                    }
                }
            }
            for s in w {
                live.remove(s);
            }
            live.extend(r.iter().copied());
        }
    }
    out.sort();
    out.dedup();
    out
}

// ------------------------------------------------------------------------------------------------ E3: the exact pass body

/// The exact back end: the lowered function as the author wrote it, registers from the pins, then verification only. Every problem is
/// collected and reported together (an `Err` with one line per violation).
pub fn finalize(f: &amdgpu_dsl::Function, mut m: MFunction) -> Result<(MFunction, crate::regalloc::Alloc), String> {
    let mut errs: Vec<String> = Vec::new();
    let stmts = crate::lower::last_stmts();
    // 1. lowering's automatic waits are not the author's: the author places the waits, the check below verifies them
    crate::sched::waits::strip_auto_waits(&mut m);
    // 2. LDS-DMAs use the M0 the author set (`set_m0`); they never write it themselves
    for b in m.blocks.iter_mut() {
        for i in b.insts.iter_mut() {
            if is_dma(i.op) {
                i.skip_m0 = true;
            }
        }
    }
    // 3. registers: every value from its pin
    // a pin of a view (an alias at dword offset k, possibly of another alias) pins its tuple at base - k; two pins of one tuple must agree
    let mut pin: std::collections::HashMap<VReg, (bool, u32)> = std::collections::HashMap::new();
    let mut pin_by: std::collections::HashMap<VReg, VReg> = std::collections::HashMap::new();
    let view_of = |m: &MFunction, r: VReg| -> (VReg, u32) {
        let (mut t, mut off) = (r, 0u32);
        for _ in 0..16 {
            match m.alias.get(&t) {
                Some(&(u, k)) => {
                    off += k as u32;
                    t = u;
                }
                None => break,
            }
        }
        (t, off)
    };
    for (s, list) in [(false, m.reg_hints.pins_v.clone()), (true, m.reg_hints.pins_s.clone())] {
        for (r, b) in list {
            let (t, off) = view_of(&m, r);
            if b < off {
                errs.push(format!("exact: {} pinned at {}{b} lies at dword {off} of its tuple: the tuple would start below register 0", name_of(&m, r), if s { "s" } else { "v" }));
                continue;
            }
            let base = (s, b - off);
            match pin.get(&t) {
                Some(&old) if old != base => errs.push(format!(
                    "exact: the pins of {} and {} disagree: they are views of one register tuple ({}{} vs {}{} for its start)",
                    name_of(&m, r), name_of(&m, pin_by[&t]), if old.0 { "s" } else { "v" }, old.1, if s { "s" } else { "v" }, base.1
                )),
                _ => {
                    pin.insert(t, base);
                    pin_by.insert(t, r);
                }
            }
        }
    }
    // a copy of a hardware register into a value pinned at that same register is no instruction
    let alias = m.alias.clone();
    let canon = |r: VReg| alias.get(&r).map(|a| a.0).unwrap_or(r);
    let mut entry: Vec<VReg> = Vec::new();
    for b in m.blocks.iter_mut() {
        b.insts.retain(|i| {
            let same = match (i.op, i.dst) {
                (MOp::VReadPhys, Some(d)) => pin.get(&canon(d)) == Some(&(false, i.imm as u32)),
                (MOp::SReadPhys, Some(d)) => pin.get(&canon(d)) == Some(&(true, i.imm as u32)),
                _ => false,
            };
            if same {
                entry.push(canon(i.dst.unwrap()));
            }
            !same
        });
    }
    let n = m.vregs.len();
    let mut phys = vec![u32::MAX; n];
    let mut scalar = vec![false; n];
    let mut used: std::collections::HashSet<VReg> = std::collections::HashSet::new();
    for b in &m.blocks {
        for i in &b.insts {
            for r in i.dst.iter().copied().chain(i.uses()) {
                used.insert(m.canon(r));
            }
        }
    }
    let (mut max_v, mut max_s) = (0u32, 0u32);
    let mut missing: Vec<String> = Vec::new();
    for r in 0..n as u32 {
        if m.canon(r) != r {
            continue;
        }
        let c = m.vreg_class(r);
        scalar[r as usize] = !c.is_vector();
        match pin.get(&r) {
            Some(&(s, b)) => {
                if s == c.is_vector() {
                    errs.push(format!("exact: {} ({c:?}) is pinned to the {} file", name_of(&m, r), if s { "SGPR" } else { "VGPR" }));
                }
                phys[r as usize] = b;
                if s {
                    max_s = max_s.max(b + c.dwords());
                } else {
                    max_v = max_v.max(b + c.dwords());
                }
            }
            None if used.contains(&r) => {
                // name it by the statement that defines it (a lowering temporary has no DSL name)
                let def = m.blocks.iter().flat_map(|b| b.insts.iter()).find(|i| i.dst.map_or(false, |d| m.canon(d) == r));
                let at = def.map(|i| {
                    let s = if i.stmt == 0 { "-".to_string() } else { format!("{:?}", stmts.get(i.stmt as usize - 1)).chars().take(80).collect() };
                    format!(" ({:?} at stmt {}: {s})", i.op, i.stmt as i64 - 1)
                });
                missing.push(format!("{}{}", name_of(&m, r), at.unwrap_or_default()));
            }
            None => {}
        }
    }
    // aliases (views) sit inside their tuple
    // views (aliases, possibly of aliases) sit inside their tuple
    fn resolve(m: &MFunction, phys: &[u32], r: VReg, depth: u32) -> u32 {
        if depth > 16 {
            return u32::MAX;
        }
        match m.alias.get(&r) {
            Some(&(t, off)) => {
                let b = resolve(m, phys, t, depth + 1);
                if b == u32::MAX { u32::MAX } else { b + off as u32 }
            }
            None => phys[r as usize],
        }
    }
    let aliases: Vec<VReg> = m.alias.keys().copied().collect();
    for v in aliases {
        let p = resolve(&m, &phys, v, 0);
        if p != u32::MAX {
            phys[v as usize] = p;
            let t = m.canon(v);
            scalar[v as usize] = scalar[t as usize];
        }
    }
    // a copy between two values the author pinned to the same register is no instruction (tuple assembly from consecutively pinned
    // parts, a view or bitcast at the same register)
    {
        let ph = |r: VReg, phys: &Vec<u32>| -> Option<u32> {
            let p = phys[r as usize];
            (p != u32::MAX).then_some(p)
        };
        for b in m.blocks.iter_mut() {
            b.insts.retain(|i| {
                let copy = matches!(i.op, MOp::VMovB32 | MOp::SMovB32) && i.srcs.len() == 1;
                match (copy, i.dst, i.srcs.first()) {
                    (true, Some(d), Some(Src::Reg(s))) => !(ph(d, &phys).is_some() && ph(d, &phys) == ph(*s, &phys) && scalar[d as usize] == scalar[*s as usize]),
                    _ => true,
                }
            });
        }
    }
    // 4. no instruction the author did not write
    for b in &m.blocks {
        for i in &b.insts {
            if let Some(k) = synthetic_kind(&m, &stmts, i) {
                let st = if i.stmt == 0 { "-".to_string() } else { format!("{} ({})", i.stmt - 1, format!("{:?}", stmts.get(i.stmt as usize - 1)).chars().take(90).collect::<String>()) };
                let text = if i.dst.map_or(true, |d| phys[d as usize] != u32::MAX) && i.srcs.iter().all(|s| !matches!(s, Src::Reg(r) if phys[*r as usize] == u32::MAX)) {
                    let a0 = crate::regalloc::Alloc { phys: phys.clone(), max_v: 0, max_s: 0, scalar: scalar.clone() };
                    crate::emit::inst_lines(&a0, i).join(" ; ")
                } else {
                    format!("{:?} (unpinned operands)", i.op)
                };
                errs.push(format!("exact: block {} `{text}`: {k}, at stmt {st}", b.label));
            }
        }
    }
    if !missing.is_empty() {
        errs.push(format!("exact: {} value(s) without a pin: {}", missing.len(), missing.iter().take(40).cloned().collect::<Vec<_>>().join("\n    ")));
    }
    let a = crate::regalloc::Alloc { phys, max_v, max_s, scalar };
    if errs.is_empty() {
        // 5. verification: waits, hazards, registers (only meaningful once every value has a register)
        for v in verify_final(&m, &a, false) {
            errs.push(format!("exact: {v}"));
        }
        for c in check_registers_with(&m, &a, &entry) {
            errs.push(format!("exact: register clobber: {c}"));
        }
    }
    let _ = f;
    if errs.is_empty() {
        Ok((m, a))
    } else {
        Err(format!("exact mode: {} violation(s)\n{}", errs.len(), errs.join("\n")))
    }
}

fn name_of(m: &MFunction, r: VReg) -> String {
    m.reg_hints.names.get(&r).cloned().unwrap_or_else(|| format!("r{r}"))
}
