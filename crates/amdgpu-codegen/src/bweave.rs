//! The back-end Weaver's machine-instruction unit graph (docs/two-weavers.md section 3), step B1: an ANALYSIS of a lowered /
//! final MIR function. Instructions are grouped by their plan unit (`MInst::unit`, from `Stmt::UnitMark`); instructions outside
//! any unit form synthetic units, one per maximal run of the same weave region. Nothing here changes the MIR.

use crate::mir::*;
use crate::sched::model::{Pipe, NPIPES};
use std::collections::HashMap;

/// Where a unit comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnitKey {
    /// a plan unit (`plan::UnitDecl::id`)
    Plan(u32),
    /// a run of untagged instructions in weave region `region` (0 = outside every region); `seq` numbers the runs
    Glue { region: u16, seq: u32 },
}

/// Per-unit resource use (instruction counts and issue cycles per pipe, from the target's instruction table).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResVector {
    pub insts: [u32; NPIPES],
    pub issue: [u32; NPIPES],
}

/// One unit of the graph.
#[derive(Clone, Debug)]
pub struct MUnit {
    pub key: UnitKey,
    /// the weave region of the unit's first instruction
    pub region: u16,
    /// (block index, instruction index) of every instruction, in program order
    pub insts: Vec<(usize, usize)>,
    pub live_in: Vec<VReg>,
    pub live_out: Vec<VReg>,
    pub res: ResVector,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MDepKind {
    /// a register value defined in `from`, read in `to`
    Reg(VReg),
    /// `from` writes memory (LDS / global) that `to` reads or writes, or `from` reads what `to` writes
    Memory,
    /// `from` holds a barrier or a wait that `to` must stay behind
    Order,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MDep {
    pub from: usize,
    pub to: usize,
    pub kind: MDepKind,
}

/// The unit graph of a function.
#[derive(Clone, Debug, Default)]
pub struct MUnitGraph {
    pub units: Vec<MUnit>,
    pub deps: Vec<MDep>,
}

fn canon(m: &MFunction, r: VReg) -> VReg {
    m.canon(r)
}

/// Builds the graph of `m` (program order = block order, instruction order).
pub fn build(m: &MFunction) -> MUnitGraph {
    let mut g = MUnitGraph::default();
    let mut plan_idx: HashMap<u32, usize> = HashMap::new();
    let mut glue_seq = 0u32;
    // the unit of every instruction
    let mut owner: Vec<Vec<usize>> = Vec::with_capacity(m.blocks.len());
    let mut last_glue: Option<(usize, u16)> = None;
    for (bi, b) in m.blocks.iter().enumerate() {
        let mut row = Vec::with_capacity(b.insts.len());
        for (ii, inst) in b.insts.iter().enumerate() {
            let u = if inst.unit > 0 {
                last_glue = None;
                let id = inst.unit - 1;
                *plan_idx.entry(id).or_insert_with(|| {
                    g.units.push(MUnit { key: UnitKey::Plan(id), region: inst.region, insts: vec![], live_in: vec![], live_out: vec![], res: Default::default() });
                    g.units.len() - 1
                })
            } else {
                match last_glue {
                    Some((u, r)) if r == inst.region => u,
                    _ => {
                        g.units.push(MUnit { key: UnitKey::Glue { region: inst.region, seq: glue_seq }, region: inst.region, insts: vec![], live_in: vec![], live_out: vec![], res: Default::default() });
                        glue_seq += 1;
                        let u = g.units.len() - 1;
                        last_glue = Some((u, inst.region));
                        u
                    }
                }
            };
            g.units[u].insts.push((bi, ii));
            let c = crate::op_attrs::attrs(inst.op).cost;
            g.units[u].res.insts[c.pipe.index()] += 1;
            g.units[u].res.issue[c.pipe.index()] += c.issue;
            row.push(u);
        }
        owner.push(row);
    }
    // register dependences and liveness across units (program order; a use reads the latest earlier def, or a def from another unit)
    let mut def_unit: HashMap<VReg, usize> = HashMap::new();
    let mut seen_dep: std::collections::HashSet<(usize, usize, VReg)> = Default::default();
    // memory / order state: the last unit that wrote LDS, wrote global, held a barrier / wait
    let (mut lds_w, mut glb_w, mut fence): (Option<usize>, Option<usize>, Option<usize>) = (None, None, None);
    let mut mem_seen: std::collections::HashSet<(usize, usize, u8)> = Default::default();
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            let u = owner[bi][ii];
            for s in &inst.srcs {
                if let Src::Reg(r) = s {
                    let r = canon(m, *r);
                    if let Some(&d) = def_unit.get(&r) {
                        if d != u && seen_dep.insert((d, u, r)) {
                            g.deps.push(MDep { from: d, to: u, kind: MDepKind::Reg(r) });
                            if !g.units[d].live_out.contains(&r) {
                                g.units[d].live_out.push(r);
                            }
                            if !g.units[u].live_in.contains(&r) {
                                g.units[u].live_in.push(r);
                            }
                        }
                    } else if !g.units[u].live_in.contains(&r) {
                        g.units[u].live_in.push(r);
                    }
                }
            }
            if let Some(d) = inst.dst {
                def_unit.insert(canon(m, d), u);
            }
            let at = crate::op_attrs::attrs(inst.op);
            let mut dep = |g: &mut MUnitGraph, from: Option<usize>, kind: MDepKind, tag: u8| {
                if let Some(f) = from {
                    if f != u && mem_seen.insert((f, u, tag)) {
                        g.deps.push(MDep { from: f, to: u, kind });
                    }
                }
            };
            dep(&mut g, fence, MDepKind::Order, 0);
            if let Some(mm) = at.mem {
                let lds = matches!(mm.space, crate::sched::effects::Space::Lds);
                if lds {
                    dep(&mut g, lds_w, MDepKind::Memory, 1);
                } else {
                    dep(&mut g, glb_w, MDepKind::Memory, 2);
                }
                if mm.write {
                    if lds {
                        lds_w = Some(u);
                    } else {
                        glb_w = Some(u);
                    }
                }
            }
            if matches!(inst.op, MOp::SBarrier | MOp::SWaitVm | MOp::SWaitLgkm) {
                fence = Some(u);
            }
        }
    }
    g
}

/// The unit's name for reports: the plan unit's name when a plan is given, else the region name.
pub fn unit_name(m: &MFunction, u: &MUnit, plan: Option<&amdgpu_dsl::plan::KernelPlan>) -> String {
    let region = |r: u16| if r == 0 { "-".to_string() } else { m.reg_hints.regions.get(r as usize - 1).map(|x| x.0.clone()).unwrap_or_default() };
    match &u.key {
        UnitKey::Plan(id) => plan.and_then(|p| p.units.iter().find(|d| d.id == *id)).map(|d| d.name.clone()).unwrap_or_else(|| format!("unit{id}")),
        UnitKey::Glue { region: r, seq } => format!("glue{seq}@{}", region(*r)),
    }
}

/// Checks a plan against the graph's current order: every plan dependence goes forward (its `from` unit starts before its `to`
/// unit) and every plan unit of a phase lies in that phase's region.
pub fn check_plan(m: &MFunction, g: &MUnitGraph, plan: &amdgpu_dsl::plan::KernelPlan) -> Vec<String> {
    let mut errs = Vec::new();
    let first = |id: u32| g.units.iter().find(|u| u.key == UnitKey::Plan(id)).and_then(|u| u.insts.first().copied());
    for d in &plan.deps {
        if d.distance > 0 {
            continue;
        }
        if let (Some(a), Some(b)) = (first(d.from), first(d.to)) {
            if a > b {
                errs.push(format!("plan dependence {} -> {} runs backwards in the emitted order", d.from, d.to));
            }
        }
    }
    for u in &g.units {
        if let UnitKey::Plan(id) = u.key {
            let Some(decl) = plan.units.iter().find(|d| d.id == id) else {
                errs.push(format!("unit {id} has no plan entry"));
                continue;
            };
            let Some(ph) = plan.phases.iter().find(|p| p.id == decl.phase) else { continue };
            let rname = if u.region == 0 { String::new() } else { m.reg_hints.regions.get(u.region as usize - 1).map(|x| x.0.clone()).unwrap_or_default() };
            if rname != ph.name {
                errs.push(format!("unit `{}` lies in region `{rname}`, its phase is `{}`", decl.name, ph.name));
            }
        }
    }
    errs
}

/// The text report of `amdc weave-report`.
pub fn report(m: &MFunction, g: &MUnitGraph, plan: Option<&amdgpu_dsl::plan::KernelPlan>) -> String {
    let pipes = [Pipe::Valu, Pipe::Trans, Pipe::Mfma, Pipe::Lds, Pipe::Vmem, Pipe::Salu, Pipe::Smem, Pipe::Ctrl];
    let mut s = format!("unit graph of {}: {} units, {} dependences ({} plan units)\n", m.name, g.units.len(), g.deps.len(), g.units.iter().filter(|u| matches!(u.key, UnitKey::Plan(_))).count());
    s.push_str(&format!("  {:>4}  {:<28} {:>6} {:>5} {:>5}  {}\n", "#", "unit", "insts", "in", "out", "per pipe (insts / issue cycles)"));
    for (i, u) in g.units.iter().enumerate() {
        let per: Vec<String> = pipes
            .iter()
            .filter(|p| u.res.insts[p.index()] > 0)
            .map(|p| format!("{:?} {}/{}", p, u.res.insts[p.index()], u.res.issue[p.index()]))
            .collect();
        s.push_str(&format!("  {:>4}  {:<28} {:>6} {:>5} {:>5}  {}\n", i, unit_name(m, u, plan), u.insts.len(), u.live_in.len(), u.live_out.len(), per.join(", ")));
    }
    let (r, mm, o) = g.deps.iter().fold((0, 0, 0), |(a, b, c), d| match d.kind {
        MDepKind::Reg(_) => (a + 1, b, c),
        MDepKind::Memory => (a, b + 1, c),
        MDepKind::Order => (a, b, c + 1),
    });
    s.push_str(&format!("dependences: {r} register, {mm} memory, {o} order\n"));
    if let Some(p) = plan {
        let e = check_plan(m, g, p);
        s.push_str(&format!("plan check: {}\n", if e.is_empty() { "ok".to_string() } else { e.join("; ") }));
    }
    s
}

// ------------------------------------------------------------------------------------------------ B3: ordering on the unit graph

/// A maximal run of one block's instructions with the same unit tag (`MInst::unit`; 0 = glue).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Seg {
    pub unit: u32,
    pub start: usize,
    pub end: usize,
}

/// The segments of a block, in program order.
pub fn block_segments(b: &Block) -> Vec<Seg> {
    let mut out: Vec<Seg> = Vec::new();
    for (i, inst) in b.insts.iter().enumerate() {
        match out.last_mut() {
            Some(s) if s.unit == inst.unit => s.end = i + 1,
            _ => out.push(Seg { unit: inst.unit, start: i, end: i + 1 }),
        }
    }
    out
}

/// Segment-level predecessors of a block: segment `j` depends on `i < j` when any instruction of `j` depends on one of `i` in the
/// scheduler's dependence graph (registers incl. vcc / scc and partial writes, memory, fences; `sched::depgraph`).
pub fn seg_preds(m: &MFunction, b: &Block, segs: &[Seg]) -> Vec<Vec<usize>> {
    let g = crate::sched::depgraph::build(m, &b.insts, &b.term, false);
    let mut seg_of = vec![0usize; b.insts.len()];
    for (k, s) in segs.iter().enumerate() {
        for i in s.start..s.end {
            seg_of[i] = k;
        }
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); segs.len()];
    for i in 0..g.n {
        for &(j, _) in &g.succs[i] {
            let (a, b2) = (seg_of[i], seg_of[j]);
            if a != b2 && !preds[b2].contains(&a) {
                preds[b2].push(a);
            }
        }
    }
    preds
}

/// How the back-end Weaver orders the segments of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderPolicy {
    /// today's order (B3: every phase is reproduced exactly)
    Identity,
}

/// Is `ord` a permutation of the segments that keeps every dependence?
pub fn is_valid_order(ord: &[usize], preds: &[Vec<usize>]) -> bool {
    let mut pos = vec![usize::MAX; preds.len()];
    for (p, &s) in ord.iter().enumerate() {
        if s >= preds.len() || pos[s] != usize::MAX {
            return false;
        }
        pos[s] = p;
    }
    if pos.iter().any(|p| *p == usize::MAX) {
        return false;
    }
    preds.iter().enumerate().all(|(s, ps)| ps.iter().all(|&q| pos[q] < pos[s]))
}

/// Rewrites a block's instructions in segment order `ord`.
pub fn apply_order(b: &mut Block, segs: &[Seg], ord: &[usize]) {
    let old = std::mem::take(&mut b.insts);
    let mut out = Vec::with_capacity(old.len());
    for &k in ord {
        out.extend_from_slice(&old[segs[k].start..segs[k].end]);
    }
    b.insts = out;
}

/// B4: the plan's interleave hints inside `Free` phases. For a hint (skeleton unit, filler units) whose units all lie in one block and
/// whose phase is `Free`, the filler instructions are spread over the skeleton: after each MFMA of the skeleton the next
/// ceil(fillers / MFMAs) filler instructions. The merged block order is applied only when it keeps every edge of the instruction
/// dependence graph (`sched::depgraph`); otherwise the block keeps its order. Returns (blocks merged, hints that fell back).
pub fn interleave(m: &mut MFunction, plan: &amdgpu_dsl::plan::KernelPlan) -> (usize, usize) {
    interleave_with(m, plan, crate::sched::waits::RingModel::of(plan))
}

/// `interleave` with the ring model of the function (tile-marker membership, `RingModel::of_function`).
pub fn interleave_with(m: &mut MFunction, plan: &amdgpu_dsl::plan::KernelPlan, rings: Option<crate::sched::waits::RingModel>) -> (usize, usize) {
    use amdgpu_dsl::plan::OrderSpec;
    let free_unit = |id: u32| -> bool {
        plan.units.iter().find(|u| u.id == id).and_then(|u| plan.phases.iter().find(|p| p.id == u.phase)).map_or(false, |p| !matches!(p.order, OrderSpec::Authored))
    };
    let (mut merged, mut fell_back) = (0usize, 0usize);
    for bi in 0..m.blocks.len() {
        for (skel, fills) in &plan.hints.interleave {
            if !free_unit(*skel) || fills.iter().any(|f| !free_unit(*f)) {
                continue;
            }
            let b = &m.blocks[bi];
            let tag = |u: u32| u + 1;
            let in_skel: Vec<usize> = (0..b.insts.len()).filter(|&i| b.insts[i].unit == tag(*skel)).collect();
            let in_fill: Vec<usize> = (0..b.insts.len()).filter(|&i| fills.iter().any(|f| b.insts[i].unit == tag(*f))).collect();
            if in_skel.is_empty() || in_fill.is_empty() {
                continue;
            }
            let mfma: Vec<usize> = in_skel.iter().copied().filter(|&i| crate::op_attrs::attrs(b.insts[i].op).cost.pipe == Pipe::Mfma).collect();
            if mfma.is_empty() {
                continue;
            }
            let per = in_fill.len().div_ceil(mfma.len());
            // ALU may cross the LDS-DMA fences (they order memory, not ALU), and a ring DMA is independent of the LDS reads of other
            // slots / rings (the wait pass keeps the slot dependences); every other edge of the dependence graph stays
            let g = {
                let mut g = crate::sched::depgraph::build(m, &b.insts, &b.term, true);
                if let Some(rm) = &rings {
                    let eff: Vec<crate::sched::effects::Effects> = b.insts.iter().map(|i| crate::sched::effects::effects(m, i)).collect();
                    for i in 0..g.n {
                        let keep: Vec<(usize, u32)> = g.succs[i]
                            .iter()
                            .copied()
                            .filter(|&(j, _)| !crate::sched::waits::ring_independent(rm, &eff[i], &b.insts[i], &eff[j], &b.insts[j]))
                            .collect();
                        g.succs[i] = keep;
                    }
                    for j in 0..g.n {
                        g.preds[j].clear();
                    }
                    for i in 0..g.n {
                        for &(j, l) in &g.succs[i].clone() {
                            g.preds[j].push((i, l));
                        }
                    }
                }
                g
            };
            // the new order: every instruction outside the two sets keeps its place relative to the first of them; the skeleton is
            // emitted in order, each skeleton instruction preceded by the fillers it depends on (transitively, in their order) and each
            // MFMA followed by up to `per` fillers whose predecessors are all placed
            let first = in_skel[0].min(in_fill[0]);
            let mut ord: Vec<usize> = Vec::with_capacity(b.insts.len());
            let moved: std::collections::HashSet<usize> = in_skel.iter().chain(in_fill.iter()).copied().collect();
            let is_fill: std::collections::HashSet<usize> = in_fill.iter().copied().collect();
            let mut done = vec![false; b.insts.len()];
            // filler ancestors of instruction x (within the filler set), oldest first
            fn fill_ancestors(x: usize, g: &crate::sched::depgraph::Graph, is_fill: &std::collections::HashSet<usize>, done: &[bool], out: &mut Vec<usize>) {
                for &(p, _) in &g.preds[x] {
                    if is_fill.contains(&p) && !done[p] && !out.contains(&p) {
                        fill_ancestors(p, g, is_fill, done, out);
                        out.push(p);
                    }
                }
            }
            let mut placed = false;
            for i in 0..b.insts.len() {
                if moved.contains(&i) {
                    if !placed && i == first {
                        placed = true;
                        for &s in &in_skel {
                            let mut need = Vec::new();
                            fill_ancestors(s, &g, &is_fill, &done, &mut need);
                            need.sort_unstable();
                            for f in need {
                                done[f] = true;
                                ord.push(f);
                            }
                            done[s] = true;
                            ord.push(s);
                            if crate::op_attrs::attrs(b.insts[s].op).cost.pipe == Pipe::Mfma {
                                let mut k = 0;
                                for &f in &in_fill {
                                    if k == per {
                                        break;
                                    }
                                    if !done[f] && g.preds[f].iter().all(|&(p, _)| done[p] || !moved.contains(&p)) {
                                        done[f] = true;
                                        ord.push(f);
                                        k += 1;
                                    }
                                }
                            }
                        }
                        for &f in &in_fill {
                            if !done[f] {
                                done[f] = true;
                                ord.push(f);
                            }
                        }
                    }
                    continue;
                }
                ord.push(i);
            }
            let mut pos = vec![0usize; ord.len()];
            for (k, &i) in ord.iter().enumerate() {
                pos[i] = k;
            }
            let ok = ord.len() == b.insts.len() && (0..g.n).all(|i| g.succs[i].iter().all(|&(j, _)| pos[i] < pos[j]));
            if ok {
                let old = m.blocks[bi].insts.clone();
                m.blocks[bi].insts = ord.iter().map(|&i| old[i].clone()).collect();
                merged += 1;
            } else {
                fell_back += 1;
                if std::env::var("AMDC_SCHED_VERBOSE").is_ok() {
                    let bad = (0..g.n).flat_map(|i| g.succs[i].iter().map(move |&(j, _)| (i, j))).find(|&(i, j)| pos[i] >= pos[j]);
                    if let Some((i, j)) = bad {
                        eprintln!(
                            "bweave interleave (skeleton unit {skel}): block {} keeps its order: {:?} (unit {}) must precede {:?} (unit {})",
                            b.label, b.insts[i].op, b.insts[i].unit, b.insts[j].op, b.insts[j].unit
                        );
                    }
                }
            }
        }
    }
    (merged, fell_back)
}

/// The back-end Weaver's ordering pass: per block, the segments are ordered by `policy` and the order is checked against the
/// segment dependences before it is applied. Returns the number of blocks whose order changed.
pub fn order(m: &mut MFunction, policy: OrderPolicy) -> Result<usize, String> {
    let mut changed = 0;
    for bi in 0..m.blocks.len() {
        let segs = block_segments(&m.blocks[bi]);
        if segs.len() < 2 {
            continue;
        }
        let ord: Vec<usize> = match policy {
            OrderPolicy::Identity => (0..segs.len()).collect(),
        };
        if ord.iter().enumerate().any(|(i, &k)| i != k) {
            let preds = seg_preds(m, &m.blocks[bi], &segs);
            if !is_valid_order(&ord, &preds) {
                return Err(format!("bweave: block {} order {:?} breaks a dependence", m.blocks[bi].label, ord));
            }
            let mut b = std::mem::replace(&mut m.blocks[bi], Block::new(0));
            apply_order(&mut b, &segs, &ord);
            m.blocks[bi] = b;
            changed += 1;
        }
    }
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mf(blocks: Vec<Block>, vregs: usize) -> MFunction {
        MFunction {
            name: "g".into(),
            args: Vec::new(),
            kernarg_size: 0,
            workgroup_size: 64,
            lds_bytes: 0,
            blocks,
            vregs: vec![RClass::V1; vregs],
            loops: Vec::new(),
            alias: Default::default(),
            elide_fallthrough: false,
            edge_pad: false,
            agpr_pin: Default::default(),
            reg_hints: Default::default(),
        }
    }

    #[test]
    fn independent_segments_may_swap_dependent_ones_may_not() {
        let mut b = Block::new(0);
        let tag = |mut i: MInst, u: u32| {
            i.unit = u;
            i
        };
        // unit 1: v0 = 1; unit 2: v1 = 2 (independent); unit 3: v2 = v0 + v1 (depends on both)
        b.insts = vec![
            tag(MInst::new(MOp::VMovB32).def(0).imm(1), 1),
            tag(MInst::new(MOp::VMovB32).def(1).imm(2), 2),
            tag(MInst::new(MOp::VAddU32).def(2).srcreg(0).srcreg(1), 3),
        ];
        let m = mf(vec![b.clone()], 3);
        let segs = block_segments(&m.blocks[0]);
        assert_eq!(segs.len(), 3);
        let preds = seg_preds(&m, &m.blocks[0], &segs);
        assert!(is_valid_order(&[1, 0, 2], &preds), "units 1 and 2 are independent");
        assert!(!is_valid_order(&[0, 2, 1], &preds), "unit 3 reads unit 2's result");
        let mut b2 = b.clone();
        apply_order(&mut b2, &segs, &[1, 0, 2]);
        assert_eq!(b2.insts[0].unit, 2);
        let mut m2 = m.clone();
        assert_eq!(order(&mut m2, OrderPolicy::Identity).unwrap(), 0, "identity changes nothing");
    }

    #[test]
    fn units_group_by_tag_and_untagged_runs_become_glue() {
        let mut b = Block::new(0);
        let mut a = MInst::new(MOp::VMovB32).def(0).imm(1);
        a.unit = 1;
        let mut c = MInst::new(MOp::VAddU32).def(1).srcreg(0).srcreg(0);
        c.unit = 2;
        b.insts = vec![MInst::new(MOp::VMovB32).def(2).imm(0), a, c, MInst::new(MOp::VMovB32).def(3).srcreg(1)];
        let g = build(&mf(vec![b], 4));
        assert_eq!(g.units.len(), 4);
        assert_eq!(g.units[1].key, UnitKey::Plan(0));
        assert_eq!(g.units[2].key, UnitKey::Plan(1));
        assert!(g.deps.contains(&MDep { from: 1, to: 2, kind: MDepKind::Reg(0) }));
        assert!(g.deps.contains(&MDep { from: 2, to: 3, kind: MDepKind::Reg(1) }));
        assert_eq!(g.units[2].live_in, vec![0]);
        assert_eq!(g.units[2].live_out, vec![1]);
    }
}

#[cfg(test)]
mod interleave_tests {
    use super::*;
    use amdgpu_dsl::plan::*;

    fn mf(blocks: Vec<Block>, n: usize) -> MFunction {
        MFunction {
            name: "g".into(),
            args: Vec::new(),
            kernarg_size: 0,
            workgroup_size: 64,
            lds_bytes: 0,
            blocks,
            vregs: (0..n).map(|i| if i < 4 { RClass::V4 } else { RClass::V1 }).collect(),
            loops: Vec::new(),
            alias: Default::default(),
            elide_fallthrough: false,
            edge_pad: false,
            agpr_pin: Default::default(),
            reg_hints: Default::default(),
        }
    }

    fn plan(order: OrderSpec) -> KernelPlan {
        let u = |id, name: &str, class| UnitDecl { id, name: name.into(), phase: 0, class, anchor: false, iter: 0, latency: 0, roles: 0 };
        KernelPlan {
            phases: vec![Phase { id: 0, name: "p".into(), parent: None, kind: PhaseKind::Straight, order, budget: Budget::default() }],
            units: vec![u(0, "exp", UnitClass::Valu), u(1, "qk", UnitClass::Mfma)],
            hints: Hints { interleave: vec![(1, vec![0])], ..Default::default() },
            ..Default::default()
        }
    }

    /// unit 0 (fillers): four independent VALU ops; unit 1 (skeleton): two independent MFMAs.
    fn block() -> Block {
        let mut b = Block::new(0);
        let tag = |mut i: MInst, u: u32| {
            i.unit = u;
            i
        };
        for k in 0..4u32 {
            b.insts.push(tag(MInst::new(MOp::VMovB32).def(4 + k).imm(k as i64), 1));
        }
        b.insts.push(tag(MInst::new(MOp::VMfma16x16x32Bf16Z).def(0).srcreg(1).srcreg(1), 2));
        b.insts.push(tag(MInst::new(MOp::VMfma16x16x32Bf16Z).def(2).srcreg(3).srcreg(3), 2));
        b
    }

    #[test]
    fn a_free_phase_spreads_the_fillers_over_the_mfmas() {
        let mut m = mf(vec![block()], 8);
        assert_eq!(interleave(&mut m, &plan(OrderSpec::Free)), (1, 0));
        let ops: Vec<MOp> = m.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops, vec![MOp::VMfma16x16x32Bf16Z, MOp::VMovB32, MOp::VMovB32, MOp::VMfma16x16x32Bf16Z, MOp::VMovB32, MOp::VMovB32]);
    }

    #[test]
    fn an_authored_phase_keeps_its_order() {
        let mut m = mf(vec![block()], 8);
        let before: Vec<MOp> = m.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(interleave(&mut m, &plan(OrderSpec::Authored)), (0, 0));
        assert_eq!(m.blocks[0].insts.iter().map(|i| i.op).collect::<Vec<_>>(), before);
    }

    #[test]
    fn a_filler_an_mfma_needs_is_pulled_ahead_of_it() {
        let mut b = block();
        // the first MFMA reads filler 0's result (the PV step needs its packed P words)
        b.insts[4].srcs[0] = Src::Reg(4);
        let mut m = mf(vec![b], 8);
        m.vregs[4] = RClass::V4;
        assert_eq!(interleave(&mut m, &plan(OrderSpec::Free)), (1, 0));
        let defs: Vec<Option<u32>> = m.blocks[0].insts.iter().map(|i| i.dst).collect();
        assert_eq!(defs, vec![Some(4), Some(0), Some(5), Some(6), Some(2), Some(7)]);
    }

    /// The replica's B5 shape: the QK skeleton reads K (ring slot t), the fillers are the DMAs of tile t + 2 (iteration offset 2). With a
    /// 3-slot ring they touch another slot, so the DMAs may move between the QK reads and MFMAs (the LDS-DMA fence orders memory only).
    #[test]
    fn ring_dmas_of_another_slot_merge_into_the_qk_window() {
        let mk = |slots: u32| {
            let u = |id, name: &str, class, iter| UnitDecl { id, name: name.into(), phase: 0, class, anchor: false, iter, latency: 0, roles: 0 };
            KernelPlan {
                phases: vec![Phase { id: 0, name: "m".into(), parent: None, kind: PhaseKind::Straight, order: OrderSpec::Free, budget: Budget::default() }],
                rings: vec![RingDecl { id: 0, name: "k".into(), slots, bytes_per_slot: 8192, cadence: Cadence::OneWindow, producer_phase: 0, consumer_phase: 0, owner: RingOwner::Cta }],
                units: vec![u(0, "dma", UnitClass::Dma, 2), u(1, "qk", UnitClass::Mfma, 0)],
                hints: Hints { interleave: vec![(1, vec![0])], ..Default::default() },
                ..Default::default()
            }
        };
        let tag = |mut i: MInst, u: u32| {
            i.unit = u;
            i
        };
        let mut b = Block::new(0);
        // QK: read K (into v1), MFMA, read K (into v3), MFMA; then two DMAs of tile t + 2
        b.insts.push(tag(MInst::new(MOp::LdsReadB128).def(1).srcreg(4).imm(0), 2));
        b.insts.push(tag(MInst::new(MOp::VMfma16x16x32Bf16Z).def(0).srcreg(1).srcreg(1), 2));
        b.insts.push(tag(MInst::new(MOp::LdsReadB128).def(3).srcreg(4).imm(64), 2));
        b.insts.push(tag(MInst::new(MOp::VMfma16x16x32Bf16Z).def(2).srcreg(3).srcreg(3), 2));
        for _ in 0..2 {
            b.insts.push(tag(MInst::new(MOp::BufferLoadDwordX4Lds).srcreg(5).srcreg(4).srcreg(6).imm(0), 1));
        }
        let mut m = mf(vec![b.clone()], 8);
        m.vregs[5] = RClass::S4;
        m.vregs[6] = RClass::S1;
        let mut m2 = m.clone();
        assert_eq!(interleave(&mut m, &mk(3)), (1, 0), "3 slots: the DMAs write another slot");
        let ops: Vec<MOp> = m.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops[2], MOp::BufferLoadDwordX4Lds, "a DMA right after the first MFMA: {ops:?}");
        // 2 slots, distance 2: the DMAs overwrite the slot QK reads: they stay behind the reads
        interleave(&mut m2, &mk(2));
        let ops2: Vec<MOp> = m2.blocks[0].insts.iter().map(|i| i.op).collect();
        assert!(ops2[..4].iter().all(|o| *o != MOp::BufferLoadDwordX4Lds), "the DMAs stay after both K reads: {ops2:?}");
    }

    #[test]
    fn an_unmoved_instruction_in_between_blocks_the_merge() {
        let mut b = block();
        // a glue instruction (no unit) between the fillers and the skeleton defines what the first MFMA reads
        b.insts.insert(4, MInst::new(MOp::VMovB32).def(1).imm(9));
        let mut m = mf(vec![b], 8);
        let before: Vec<MOp> = m.blocks[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(interleave(&mut m, &plan(OrderSpec::Free)), (0, 1));
        assert_eq!(m.blocks[0].insts.iter().map(|i| i.op).collect::<Vec<_>>(), before);
    }
}
