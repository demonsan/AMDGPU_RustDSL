//! The kernel plan: the interface between the front-end Weaver and the back-end Weaver (docs/two-weavers.md section 2).
//!
//! Plain data that travels with the `Function` (printed / parsed by irv1 when present). CONSTRAINTS (phases with their order
//! rule and budget, unit dependences, hand-offs, sync rules) are binding: a back end that cannot meet one reports a compile error
//! naming it. HINTS may be ignored. A function without a plan compiles exactly as before (every weave region `Authored`, no
//! budget). Unit ids are the ids of the `Stmt::UnitMark` markers around each unit's statements.

use crate::{RegRef, VarId};

pub type PhaseId = u32;
pub type UnitId = u32;

/// The kernel plan.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelPlan {
    pub phases: Vec<Phase>,
    pub units: Vec<UnitDecl>,
    pub deps: Vec<Dep>,
    pub handoffs: Vec<HandOff>,
    pub sync: Vec<SyncRule>,
    pub hints: Hints,
}

/// A phase: a weave region (today's `RegionMark` name) with its order rule and resource budget.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase {
    pub id: PhaseId,
    /// the weave region name the phase's statements are marked with
    pub name: String,
    pub parent: Option<PhaseId>,
    pub kind: PhaseKind,
    pub order: OrderSpec,
    pub budget: Budget,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PhaseKind {
    Straight,
    /// the body of a loop over `var` (`trip`: the iteration count when known)
    LoopBody { var: Option<VarId>, trip: Option<u32> },
    Branch,
}

/// The order rule of a phase.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderSpec {
    /// the units' statement order is the order
    Authored,
    /// any order that respects the dependences, sync rules and budget
    Free,
    /// `Free` plus these extra (before, after) edges
    Partial(Vec<(UnitId, UnitId)>),
}

/// Per-phase resource caps (None: the kernel's limit).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Budget {
    pub vgpr: Option<u32>,
    pub agpr: Option<u32>,
    pub sgpr: Option<u32>,
    pub lds_bytes: Option<u32>,
    /// at most this many outstanding operations on the named counter inside the phase
    pub inflight: Vec<(String, u32)>,
    pub m0: M0Use,
}

/// Who may change M0 inside a phase.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum M0Use {
    #[default]
    Free,
    /// the phase owns M0 (its LDS-DMA bases); other phases restore it
    Owned,
}

/// What a unit is (the front end's classification; today's `weave::UnitClass`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitClass {
    Mfma,
    Valu,
    LdsRead,
    Dma,
    Salu,
    Branch,
    Setup,
}

/// One unit of the plan.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitDecl {
    pub id: UnitId,
    pub name: String,
    pub phase: PhaseId,
    pub class: UnitClass,
    /// keeps its position relative to the phase boundaries (waits, barriers, setup)
    pub anchor: bool,
    /// iteration offset of the unit's work (0 = this iteration, +1 = the next one)
    pub iter: i32,
    /// nominal issue -> result latency (cycles), for priorities only
    pub latency: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DepKind {
    Data,
    /// through memory (`lds` / `global`)
    Memory(String),
    Order,
}

/// `from` must complete (data) or issue (order) before `to`; `distance` = iterations between them (0 = the same iteration).
#[derive(Clone, Debug, PartialEq)]
pub struct Dep {
    pub from: UnitId,
    pub to: UnitId,
    pub kind: DepKind,
    pub distance: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegFileKind {
    Vgpr,
    Agpr,
    Sgpr,
}

/// How a value crosses phases.
#[derive(Clone, Debug, PartialEq)]
pub enum Via {
    /// in registers of `file`, at `pin` when fixed
    Reg { file: RegFileKind, pin: Option<u32> },
    Lds { offset: u32, bytes: u32 },
    Global,
}

/// A value handed from one phase to another.
#[derive(Clone, Debug, PartialEq)]
pub struct HandOff {
    pub value: RegRef,
    pub from: PhaseId,
    pub to: PhaseId,
    pub via: Via,
    /// the tile layout the consumer expects (checked by the stage contracts)
    pub layout: Option<String>,
}

/// Synchronization the order must keep.
#[derive(Clone, Debug, PartialEq)]
pub enum SyncRule {
    /// a workgroup barrier between the two units
    Barrier { after: UnitId, before: UnitId },
    /// the consumer waits on `counter` for the producer, with at most `slack` newer operations outstanding (ring slots)
    Counted { producer: UnitId, consumer: UnitId, counter: String, slack: u32 },
    /// an inter-CTA protocol on a flag (kept and checked by gpu_protocol)
    CrossCta { flag: RegRef, scope: String },
}

/// Optional guidance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hints {
    /// skeleton unit -> filler units to place in its shadow
    pub interleave: Vec<(UnitId, Vec<UnitId>)>,
    pub priority: Vec<(UnitId, i32)>,
    pub prefetch_depth: Vec<(PhaseId, u32)>,
    /// transitional: today's scheduler option tokens per phase (removed when the back-end Weaver orders Free phases itself)
    pub tokens: Vec<(PhaseId, String)>,
}

/// Structural checks of a plan (ids unique and resolvable, deps between known units, parents known).
pub fn verify(p: &KernelPlan) -> Vec<String> {
    let mut errs = Vec::new();
    let mut phase_ids: Vec<PhaseId> = Vec::new();
    for ph in &p.phases {
        if phase_ids.contains(&ph.id) {
            errs.push(format!("plan: phase id {} twice", ph.id));
        }
        phase_ids.push(ph.id);
    }
    for ph in &p.phases {
        if let Some(par) = ph.parent {
            if !phase_ids.contains(&par) || par == ph.id {
                errs.push(format!("plan: phase `{}` has unknown parent {par}", ph.name));
            }
        }
    }
    let mut unit_ids: Vec<UnitId> = Vec::new();
    for u in &p.units {
        if unit_ids.contains(&u.id) {
            errs.push(format!("plan: unit id {} twice", u.id));
        }
        unit_ids.push(u.id);
        if !phase_ids.contains(&u.phase) {
            errs.push(format!("plan: unit `{}` in unknown phase {}", u.name, u.phase));
        }
    }
    let known = |u: UnitId| unit_ids.contains(&u);
    for d in &p.deps {
        if !known(d.from) || !known(d.to) {
            errs.push(format!("plan: dependence {} -> {} names an unknown unit", d.from, d.to));
        }
        if d.from == d.to && d.distance == 0 {
            errs.push(format!("plan: unit {} depends on itself in the same iteration", d.from));
        }
    }
    for ph in &p.phases {
        if let OrderSpec::Partial(edges) = &ph.order {
            for (a, b) in edges {
                if !known(*a) || !known(*b) {
                    errs.push(format!("plan: phase `{}` order edge {a} -> {b} names an unknown unit", ph.name));
                }
            }
        }
    }
    for h in &p.handoffs {
        if !phase_ids.contains(&h.from) || !phase_ids.contains(&h.to) {
            errs.push(format!("plan: hand-off names an unknown phase ({} -> {})", h.from, h.to));
        }
    }
    for s in &p.sync {
        match s {
            SyncRule::Barrier { after, before } if !known(*after) || !known(*before) => errs.push(format!("plan: barrier {after} -> {before} names an unknown unit")),
            SyncRule::Counted { producer, consumer, .. } if !known(*producer) || !known(*consumer) => {
                errs.push(format!("plan: counted sync {producer} -> {consumer} names an unknown unit"))
            }
            _ => {}
        }
    }
    errs
}
