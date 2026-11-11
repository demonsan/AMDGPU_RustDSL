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
