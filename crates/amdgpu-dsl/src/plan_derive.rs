//! Derivation of the rest of the kernel plan from what the kernel and the Weaver already declared (docs/two-weavers.md 6.1, step W2c).
//!
//! The Weaver (`amdgpu_stage::weave`) records phases, units, dependences and interleave hints while it emits. At `FnBuilder::finish` the plan of a kernel that wove
//! anything is completed from the function's own declarations, so the back end reads ONE structure instead of region tokens, register marks, stage contracts and tile
//! declarations:
//! - `RegionDecl { authored, tokens }` -> `Phase::order` (`Authored` / `Free`) and `Hints::tokens`; `RegMark::Cap` -> `Phase::budget`;
//! - `RegMark::Share` / `Separate` / `PinV` / `Ring` -> `reg_constraints` (`Share`, `Separate`, `Pin`, `RegRing`: value-level allocator constraints);
//! - phases whose units sit inside a loop -> `PhaseKind::LoopBody` (with the peeled prologue / epilogue counts of `rotate`);
//! - the stage contracts (`live_out` of one stage matching `live_in` of a later one) -> `HandOff`s (registers or an LDS tile, with the recorded fragment layout);
//! - LDS rings of the contract (`Contract::tiles`, more than one slot) -> `RingDecl`s, producer / consumer phases from the units inside the tile's produce / consume regions
//!   (private tiles: one ring per wave); the cadence is inferred from the slot count (3 slots: one window, else the two-phase offset form);
//! - the contract's last-arriver patterns -> `SyncRule::CrossCta` (`FlagRef::Kernarg` counter, the guarded global region, AcqRel);
//! - the CTA shape from the workgroup size.
//! Nothing here changes any instruction: the plan is data (the md5 manifest of the emitted assembly is the gate).

use crate::plan::*;
use crate::{Function, Node, RegMark, RegRef, RegionKind, Stmt};
use std::collections::HashMap;

#[derive(Clone, Default)]
struct Ctx {
    in_loop: bool,
    /// tile name + produce / consume of the enclosing protocol regions
    tiles: Vec<(String, RegionKind)>,
    stage: Option<usize>,
}

struct Walk {
    /// unit id -> context of its begin mark
    units: HashMap<u32, Ctx>,
    /// the units whose begin mark is open at the current position
    open: Vec<u32>,
    /// unit id -> the tile protocol regions (tile, produce / consume) that lie inside its marks
    inner: HashMap<u32, Vec<(String, RegionKind)>>,
    /// the stages in program order: (contract index into `stages`, units inside)
    stages: Vec<(String, crate::StageContract, Vec<u32>)>,
}

fn walk(nodes: &[Node], ctx: &Ctx, w: &mut Walk) {
    for n in nodes {
        match n {
            Node::S(Stmt::UnitMark { id, begin: true }) => {
                w.open.push(*id);
                w.units.insert(*id, ctx.clone());
                if let Some(s) = ctx.stage {
                    w.stages[s].2.push(*id);
                }
            }
            Node::S(Stmt::UnitMark { begin: false, .. }) => {
                w.open.pop();
            }
            Node::S(_) => {}
            Node::Loop { body, .. } => walk(body, &Ctx { in_loop: true, ..ctx.clone() }, w),
            Node::If { then, els, .. } => {
                walk(then, ctx, w);
                if let Some(e) = els {
                    walk(e, ctx, w);
                }
            }
            Node::Region { kind, tile, body, .. } => {
                let mut c = ctx.clone();
                c.tiles.push((tile.clone(), *kind));
                for u in w.open.clone() {
                    w.inner.entry(u).or_default().push((tile.clone(), *kind));
                }
                walk(body, &c, w);
            }
            Node::Stage { name, contract, body } => {
                w.stages.push((name.clone(), (**contract).clone(), Vec::new()));
                let mut c = ctx.clone();
                c.stage = Some(w.stages.len() - 1);
                walk(body, &c, w);
            }
        }
    }
}

/// Completes `f.plan` (when the kernel has one) from the function's declarations; called once, by `FnBuilder::finish`.
pub fn derive(f: &mut Function) {
    let Some(mut plan) = f.plan.take() else { return };
    let mut w = Walk { units: HashMap::new(), open: Vec::new(), inner: HashMap::new(), stages: Vec::new() };
    walk(&f.body, &Ctx::default(), &mut w);
    let phase_of = |id: u32, plan: &KernelPlan| plan.units.iter().find(|u| u.id == id).map(|u| u.phase);

    // CTA shape
    if plan.cta.is_none() && f.workgroup_size > 0 {
        plan.cta = Some(CtaShape { threads: f.workgroup_size, waves: f.workgroup_size.div_ceil(64), occupancy: None });
    }

    // order, tokens, budgets (by region / stage name)
    for ph in plan.phases.iter_mut() {
        if let Some(d) = f.regions.iter().find(|d| d.name == ph.name) {
            ph.order = if d.authored { OrderSpec::Authored } else { OrderSpec::Free };
        }
    }
    for d in &f.regions {
        if d.tokens.is_empty() {
            continue;
        }
        for ph in plan.phases.iter().filter(|p| p.name == d.name) {
            if !plan.hints.tokens.iter().any(|(id, t)| *id == ph.id && *t == d.tokens) {
                plan.hints.tokens.push((ph.id, d.tokens.clone()));
            }
        }
    }
    for m in &f.reg_plan {
        match m {
            RegMark::Cap { region, vgprs, agprs } => {
                for ph in plan.phases.iter_mut().filter(|p| p.name == *region) {
                    if let Some(v) = vgprs {
                        ph.budget.vgpr = Some(ph.budget.vgpr.map_or(*v, |o| o.min(*v)));
                    }
                    if let Some(a) = agprs {
                        ph.budget.agpr = Some(ph.budget.agpr.map_or(*a, |o| o.min(*a)));
                    }
                }
            }
            RegMark::Share(rs) => {
                let c = RegConstraint::Share(rs.clone());
                if !plan.reg_constraints.contains(&c) {
                    plan.reg_constraints.push(c);
                }
            }
            RegMark::Separate(rs) => {
                let c = RegConstraint::Separate(rs.clone());
                if !plan.reg_constraints.contains(&c) {
                    plan.reg_constraints.push(c);
                }
            }
            RegMark::PinV { r, base } => {
                let c = RegConstraint::Pin { value: *r, base: *base };
                if !plan.reg_constraints.contains(&c) {
                    plan.reg_constraints.push(c);
                }
            }
            RegMark::PinS { .. } => {}
            RegMark::Ring { name, bufs } => {
                let c = RegConstraint::RegRing { name: name.clone(), bufs: bufs.clone() };
                if !plan.reg_constraints.contains(&c) {
                    plan.reg_constraints.push(c);
                }
            }
        }
    }

    // loop bodies: a phase all of whose units sit in a loop (rotate's steady phase, the stages of a pipelined loop)
    let nphase = plan.phases.len();
    let mut in_loop = vec![false; nphase];
    let mut any = vec![false; nphase];
    for u in &plan.units {
        if let (Some(ctx), Some(pi)) = (w.units.get(&u.id), plan.phases.iter().position(|p| p.id == u.phase)) {
            any[pi] = true;
            if ctx.in_loop {
                in_loop[pi] = true;
            }
        }
    }
    let has_prologue = plan.phases.iter().any(|p| p.name == "rotate.prologue");
    let has_epilogue = plan.phases.iter().any(|p| p.name == "rotate.epilogue");
    for (i, ph) in plan.phases.iter_mut().enumerate() {
        if any[i] && in_loop[i] && matches!(ph.kind, PhaseKind::Straight) {
            let peel = if ph.name == "rotate.steady" { (has_prologue as u32, has_epilogue as u32) } else { (0, 0) };
            ph.kind = PhaseKind::LoopBody { var: None, trip: None, peel };
        }
    }

    // hand-offs between stages: a live_out value of an earlier stage that a later stage takes as live_in
    let stage_phase: Vec<Option<PhaseId>> = w.stages.iter().map(|(_, _, us)| us.first().and_then(|u| phase_of(*u, &plan))).collect();
    for (a, (_, ca, _)) in w.stages.iter().enumerate() {
        let Some(pa) = stage_phase[a] else { continue };
        for out in &ca.live_out {
            if out.values.is_empty() {
                continue;
            }
            for (b, (_, cb, _)) in w.stages.iter().enumerate().skip(a + 1) {
                let Some(pb) = stage_phase[b] else { continue };
                if pb == pa {
                    continue;
                }
                let hit = cb.live_in.iter().any(|i| i.name == out.name || i.values.iter().any(|v| out.values.contains(v)));
                if !hit {
                    continue;
                }
                let via = if let Some(tile) = out.storage.strip_prefix("lds:") {
                    match f.contract.tiles.iter().find(|t| t.name == tile) {
                        Some(t) => Via::Lds { offset: t.base.max(0) as u32, bytes: (t.slot_size * t.slots as i64).max(0) as u32 },
                        None => Via::Lds { offset: 0, bytes: 0 },
                    }
                } else if out.storage == "global" {
                    Via::Global
                } else {
                    Via::Reg { file: if out.storage == "agpr" { RegFileKind::Agpr } else { RegFileKind::Vgpr }, pin: None, dwords: out.values.len() as u32, align: 1 }
                };
                let h = HandOff { value: RegRef::Val(out.values[0]), from: pa, to: pb, via, layout: if out.layout.is_empty() { None } else { Some(out.layout.clone()) } };
                if !plan.handoffs.contains(&h) {
                    plan.handoffs.push(h);
                }
                break;
            }
        }
    }

    // LDS rings: the contract's multi-slot tiles; producer / consumer phases from the units inside their produce / consume regions
    if !plan.phases.is_empty() && plan.rings.is_empty() {
        let first_phase = plan.phases[0].id;
        for t in f.contract.tiles.iter().filter(|t| !t.global && (t.slots > 1 || t.private)) {
            let find = |kind: RegionKind| -> Option<PhaseId> {
                plan.units
                    .iter()
                    .find(|u| {
                        let hit = |v: &[(String, RegionKind)]| v.iter().any(|(n, k)| *n == t.name && *k == kind);
                        w.units.get(&u.id).map_or(false, |c| hit(&c.tiles)) || w.inner.get(&u.id).map_or(false, |v| hit(v))
                    })
                    .map(|u| u.phase)
            };
            let (prod, cons) = (find(RegionKind::Produce), find(RegionKind::Consume));
            let (producer_phase, consumer_phase) = (prod.or(cons).unwrap_or(first_phase), cons.or(prod).unwrap_or(first_phase));
            let id = plan.rings.len() as u32;
            plan.rings.push(RingDecl {
                id,
                name: t.name.clone(),
                slots: t.slots.max(1),
                bytes_per_slot: t.slot_size.max(0) as u32,
                cadence: if t.slots == 3 { Cadence::OneWindow } else { Cadence::TwoPhaseOffset },
                producer_phase,
                consumer_phase,
                owner: if t.private { RingOwner::Wave } else { RingOwner::Cta },
            });
        }
    }
    // cross-CTA protocols: the opt-in last-arriver patterns of the contract (counter by kernarg name, the global region it guards)
    for la in &f.contract.last_arrivers {
        let flag = FlagRef::Kernarg { arg: la.counter_arg.clone(), region: Some(la.region.clone()) };
        let r = SyncRule::CrossCta {
            flag: flag.clone(),
            scope: "device".into(),
            protocol: CrossCtaProtocol::LastArriver { counter: flag, partials: vec![la.region.clone()], splits: Some(la.splits.clone()) },
            order: MemOrder::AcqRel,
        };
        if !plan.sync.contains(&r) {
            plan.sync.push(r);
        }
    }
    f.plan = Some(plan);
}
