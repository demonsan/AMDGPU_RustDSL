//! The front-end Weaver (docs/two-weavers.md, docs/fusion-stage-library-design.md "Stage / Weaver / backend"): the [`Weaver`] takes the units of one or more stages (the captured IR pieces of
//! `amdgpu_dsl::units`: dependences, resource class, latency, register profile, interleave hint) and emits them in an order: the kernel author's explicit order is authoritative
//! ([`Weaver::order`] / [`Weaver::seq`]); [`Weaver::spread`] is an optional automatic strategy (skeleton units with filler units spread between them). The weave is checked to be a
//! topological order of the dependencies before anything is emitted. The output is ordinary region IR - verified (gpu_verify, GV-STAGE) and compiled as before. Per-shape decisions
//! (which order, which spread) stay at the call site: nothing here is a scheduler. [`rotate`] is the software-pipelined emission by iteration offsets.
//! (The unit capture container itself - `Unit`, `Units`, `Hint`, `UnitClass` - lives in `amdgpu_dsl::units`; it is re-exported here.)

use amdgpu_dsl::plan::{self, Budget, Dep, DepKind, OrderSpec, Phase, PhaseKind, UnitDecl};
use amdgpu_dsl::FnBuilder;
pub use amdgpu_dsl::units::{Hint, RegProfile, Unit, UnitClass, UnitId, Units};

/// Places the units of a phase and emits them.
pub struct Weaver {
    units: Units,
    order: Option<Vec<UnitId>>,
    label: String,
    /// the kernel gave the order (`order` / `spread`): never overridden by the experiment switches (`weave_auto`)
    explicit: bool,
    /// the name of the plan phase of an unlabelled weave (no weave region is emitted for it; `rotate` names its loop parts)
    plan_label: String,
    /// (skeleton unit, the fillers placed in its shadow) of a `spread`
    interleave: Vec<(UnitId, Vec<UnitId>)>,
    /// the recorded phase is `OrderSpec::Free` (the back-end Weaver may reorder its units within the dependences, sync rules and budget)
    free: bool,
    /// the recorded phase's register budget (B6a: the scheduler's pressure limit in the phase)
    budget: Budget,
}

impl Weaver {
    pub fn new(units: Units) -> Weaver {
        Weaver { units, order: None, label: String::new(), explicit: false, plan_label: String::new(), interleave: Vec::new(), free: false, budget: Budget::default() }
    }
    /// A plan-only phase name (no region is emitted): the units still get plan entries and unit marks.
    /// Records the phase as `OrderSpec::Free` (docs/w3-b5-b6.md): the emitted order stays the authored one; the back-end Weaver may reorder.
    pub fn free(mut self) -> Weaver {
        self.free = true;
        self
    }
    /// The phase's arch-VGPR budget (B6a).
    pub fn budget_vgpr(mut self, vgpr: u32) -> Weaver {
        self.budget.vgpr = Some(vgpr);
        self
    }
    /// An interleave hint without changing the emission order: `fillers` are to be placed in the shadow of `skeleton` (the back-end merge, B4).
    pub fn interleave(mut self, skeleton: UnitId, fillers: Vec<UnitId>) -> Weaver {
        self.interleave.push((skeleton, fillers));
        self
    }
    pub fn plan_label(mut self, label: impl Into<String>) -> Weaver {
        self.plan_label = label.into();
        self
    }
    /// A label for reports / errors (the phase name).
    pub fn phase(mut self, label: impl Into<String>) -> Weaver {
        self.label = label.into();
        self
    }
    /// Statement granularity: every unit split into its node groups (`Units::split`); an explicit order is dropped (re-order the
    /// sub-units with `order` / a strategy afterwards).
    pub fn fine(self) -> Weaver {
        Weaver { units: self.units.split(), order: None, label: self.label, explicit: false, plan_label: self.plan_label, interleave: Vec::new(), free: self.free, budget: self.budget }
    }
    /// The units of the weave (strategies read their facts).
    pub fn units(&self) -> &Units {
        &self.units
    }
    /// Declaration order.
    pub fn seq(mut self) -> Weaver {
        self.order = Some((0..self.units.len()).collect());
        self
    }
    /// The explicit order (authoritative): every unit exactly once.
    pub fn order(mut self, ids: Vec<UnitId>) -> Weaver {
        self.explicit = true;
        self.order = Some(ids);
        self
    }
    /// Strategy: `skeleton` in its given order, `fillers` distributed round-robin after the skeleton units, at most `per` after each (the rest
    /// after the last skeleton unit); every other unit keeps its declaration position before the first skeleton unit.
    pub fn spread(mut self, skeleton: &[UnitId], fillers: &[UnitId], per: usize) -> Weaver {
        let n = self.units.len();
        let mut placed = vec![false; n];
        for &i in skeleton.iter().chain(fillers) {
            placed[i] = true;
        }
        let mut ord: Vec<UnitId> = (0..n).filter(|&i| !placed[i]).collect();
        let mut fi = fillers.iter().copied().peekable();
        self.interleave.clear();
        for &s in skeleton {
            ord.push(s);
            let mut shadow = Vec::new();
            for _ in 0..per {
                match fi.next() {
                    Some(f) => {
                        ord.push(f);
                        shadow.push(f);
                    }
                    None => break,
                }
            }
            if !shadow.is_empty() {
                self.interleave.push((s, shadow));
            }
        }
        if let (Some(last), true) = (skeleton.last(), fi.peek().is_some()) {
            let rest: Vec<UnitId> = fi.clone().collect();
            match self.interleave.iter_mut().find(|(s, _)| s == last) {
                Some((_, sh)) => sh.extend(rest),
                None => self.interleave.push((*last, rest)),
            }
        }
        ord.extend(fi);
        self.explicit = true;
        self.order = Some(ord);
        self
    }
    /// The order that `emit` would use (after the checks).
    pub fn plan(&self) -> Vec<UnitId> {
        let mut ord = self.order.clone().unwrap_or_else(|| (0..self.units.len()).collect());
        // experiment switch (A/B without editing the kernel): `AMDC_WEAVE_AUTO=<label>` replaces a declaration-order weave by the cost-based order
        if !self.explicit && !self.label.is_empty() && crate::weave_auto::env_enabled(&self.label) {
            ord = crate::weave_auto::auto_order(&self.units, &crate::weave_auto::Model::gfx950());
        }
        check_order(&self.units, &ord, &self.label);
        ord
    }
    /// Checks the order (a permutation of the units and a topological order of their dependencies) and emits.
    /// A labelled weave is wrapped in a weave region of that name (no code): the backend's register caps / reports and the scheduler's per-region
    /// strategies key on it (docs/weaver-regs.md). Units with internal control flow (class `Branch`) stay inside the region.
    pub fn emit(self, b: &mut FnBuilder) {
        self.emit_phase(b);
    }
    /// `emit`, returning the id of the recorded plan phase (None for an unnamed weave): for ring declarations that name their phases.
    pub fn emit_phase(mut self, b: &mut FnBuilder) -> Option<plan::PhaseId> {
        // statement-granularity experiment switch: split into statement groups (derived dependences), then the cost-based order
        if !self.explicit && !self.label.is_empty() && crate::weave_auto::env_stmt_enabled(&self.label) {
            self = self.fine();
            let o = crate::weave_auto::auto_order(&self.units, &crate::weave_auto::Model::gfx950());
            self.order = Some(o);
            self.explicit = true;
        }
        let ord = self.plan();
        let label = self.label.clone();
        // the plan (docs/two-weavers.md): one phase per weave, its units in emission order with their dependences, the interleave hints; each unit's statements are
        // bracketed by `UnitMark`s (no code) with the plan unit id. An unnamed weave (no label, no plan label) records nothing.
        let phase_name = if label.is_empty() { self.plan_label.clone() } else { label.clone() };
        let phase_id: Option<plan::PhaseId> = if phase_name.is_empty() { None } else { Some(b.plan_mut().phases.len() as u32) };
        let unit_ids: Vec<u32> = if phase_name.is_empty() {
            Vec::new()
        } else {
            let p = b.plan_mut();
            let phase = p.phases.len() as u32;
            let base = p.units.len() as u32;
            let mut ids = vec![0u32; ord.len()];
            for (pos, &i) in ord.iter().enumerate() {
                ids[i] = base + pos as u32;
            }
            p.phases.push(Phase { id: phase, name: phase_name, parent: None, kind: PhaseKind::Straight, order: if self.free { OrderSpec::Free } else { OrderSpec::Authored }, budget: self.budget.clone() });
            for &i in &ord {
                let u = &self.units.v[i];
                p.units.push(UnitDecl { id: ids[i], name: u.name.clone(), phase, class: plan_class(u.class), anchor: u.hint == Hint::Anchor, iter: u.iter, latency: u.latency, roles: 0 });
            }
            for &i in &ord {
                for &d in &self.units.v[i].deps {
                    let kind = if crate::weave_auto::data_dep(&self.units.v[d].summary, &self.units.v[i].summary) { DepKind::Data } else { DepKind::Order };
                    p.deps.push(Dep { from: ids[d], to: ids[i], kind, distance: 0 });
                }
            }
            for (s, fillers) in &self.interleave {
                p.hints.interleave.push((ids[*s], fillers.iter().map(|&f| ids[f]).collect()));
            }
            ids
        };
        let mut slots: Vec<Option<Unit>> = self.units.v.into_iter().map(Some).collect();
        if !label.is_empty() {
            b.weave_region_begin(&label);
        }
        for i in ord {
            let u = slots[i].take().expect("weave: unit emitted twice");
            if unit_ids.is_empty() {
                b.splice(u.cap);
            } else {
                b.unit_begin(unit_ids[i]);
                b.splice(u.cap);
                b.unit_end(unit_ids[i]);
            }
        }
        if !label.is_empty() {
            b.weave_region_end();
        }
        phase_id
    }
}

/// Declares a ring of the kernel plan (docs/w3-b5-b6.md: B5 reads its depth; the slot of a unit with iteration offset `d` is `d mod slots`).
/// Returns the ring id.
pub fn declare_ring(b: &mut FnBuilder, name: &str, slots: u32, bytes_per_slot: u32, cadence: plan::Cadence, producer: plan::PhaseId, consumer: plan::PhaseId) -> u32 {
    let p = b.plan_mut();
    let id = p.rings.len() as u32;
    p.rings.push(plan::RingDecl { id, name: name.to_string(), slots, bytes_per_slot, cadence, producer_phase: producer, consumer_phase: consumer, owner: plan::RingOwner::Cta });
    id
}

fn plan_class(c: UnitClass) -> plan::UnitClass {
    match c {
        UnitClass::Mfma => plan::UnitClass::Mfma,
        UnitClass::Valu => plan::UnitClass::Valu,
        UnitClass::LdsRead => plan::UnitClass::LdsRead,
        UnitClass::Dma => plan::UnitClass::Dma,
        UnitClass::Salu => plan::UnitClass::Salu,
        UnitClass::Branch => plan::UnitClass::Branch,
        UnitClass::Setup => plan::UnitClass::Setup,
    }
}

/// Panics unless `ord` lists every unit exactly once and every dependency before its consumer.
pub fn check_order(units: &Units, ord: &[UnitId], label: &str) {
    let n = units.len();
    assert_eq!(ord.len(), n, "weave `{label}`: the order lists {} of {n} units", ord.len());
    let mut pos = vec![usize::MAX; n];
    for (p, &i) in ord.iter().enumerate() {
        assert!(i < n, "weave `{label}`: unknown unit {i}");
        assert_eq!(pos[i], usize::MAX, "weave `{label}`: unit {i} (`{}`) listed twice", units.v[i].name);
        pos[i] = p;
    }
    for (i, u) in units.v.iter().enumerate() {
        for &d in &u.deps {
            assert!(pos[d] < pos[i], "weave `{label}`: `{}` placed before its dependency `{}`", u.name, units.v[d].name);
        }
    }
}

/// Which part of a rotated loop is emitted (software pipelining by iteration offsets): the prologue keeps the units that run AHEAD of the first
/// iteration (offset >= 1), the steady body keeps every unit, the epilogue (the peeled last iteration) keeps the units of the current tile
/// (offset <= 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotStage {
    Prologue,
    Steady,
    Epilogue,
}

impl RotStage {
    pub fn keeps(self, iter: i32) -> bool {
        match self {
            RotStage::Prologue => iter >= 1,
            RotStage::Steady => true,
            RotStage::Epilogue => iter <= 0,
        }
    }
}

/// One part of a loop body: a name, the iteration offset of its work, and whether it is an ordering anchor (waits / barriers).
#[derive(Clone, Copy, Debug)]
pub struct Part {
    pub name: &'static str,
    pub iter: i32,
    pub anchor: bool,
}

/// Emits the parts of a loop body that `stage` keeps, as captured units in part order (`build(b, i)` emits part `i`; parts that are not kept
/// are never built, so they allocate no IR). `order`: an explicit order of the kept units (ids in kept order), or `None` for the part order.
pub fn rotate(b: &mut FnBuilder, parts: &[Part], stage: RotStage, order: Option<Vec<UnitId>>, build: &mut dyn FnMut(&mut FnBuilder, usize)) {
    let mut u = Units::new();
    for (i, p) in parts.iter().enumerate() {
        if !stage.keeps(p.iter) {
            continue;
        }
        let (id, ()) = if p.anchor { u.anchor(b, p.name, |b| build(b, i)) } else { u.capture(b, p.name, |b| build(b, i)) };
        u.set_iter(id, p.iter);
    }
    let w = Weaver::new(u).plan_label(match stage {
        RotStage::Prologue => "rotate.prologue",
        RotStage::Steady => "rotate.steady",
        RotStage::Epilogue => "rotate.epilogue",
    });
    match order {
        Some(o) => w.order(o).emit(b),
        None => w.seq().emit(b),
    }
}
