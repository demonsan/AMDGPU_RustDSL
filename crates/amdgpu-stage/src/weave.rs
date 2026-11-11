//! The front-end Weaver (docs/two-weavers.md, docs/fusion-stage-library-design.md "Stage / Weaver / backend"): the [`Weaver`] takes the units of one or more stages (the captured IR pieces of
//! `amdgpu_dsl::units`: dependences, resource class, latency, register profile, interleave hint) and emits them in an order: the kernel author's explicit order is authoritative
//! ([`Weaver::order`] / [`Weaver::seq`]); [`Weaver::spread`] is an optional automatic strategy (skeleton units with filler units spread between them). The weave is checked to be a
//! topological order of the dependencies before anything is emitted. The output is ordinary region IR - verified (gpu_verify, GV-STAGE) and compiled as before. Per-shape decisions
//! (which order, which spread) stay at the call site: nothing here is a scheduler. [`rotate`] is the software-pipelined emission by iteration offsets.
//! (The unit capture container itself - `Unit`, `Units`, `Hint`, `UnitClass` - lives in `amdgpu_dsl::units`; it is re-exported here.)

use amdgpu_dsl::FnBuilder;
pub use amdgpu_dsl::units::{Hint, RegProfile, Unit, UnitClass, UnitId, Units};

/// Places the units of a phase and emits them.
pub struct Weaver {
    units: Units,
    order: Option<Vec<UnitId>>,
    label: String,
    /// the kernel gave the order (`order` / `spread`): never overridden by the experiment switches (`weave_auto`)
    explicit: bool,
}

impl Weaver {
    pub fn new(units: Units) -> Weaver {
        Weaver { units, order: None, label: String::new(), explicit: false }
    }
    /// A label for reports / errors (the phase name).
    pub fn phase(mut self, label: impl Into<String>) -> Weaver {
        self.label = label.into();
        self
    }
    /// Statement granularity: every unit split into its node groups (`Units::split`); an explicit order is dropped (re-order the
    /// sub-units with `order` / a strategy afterwards).
    pub fn fine(self) -> Weaver {
        Weaver { units: self.units.split(), order: None, label: self.label, explicit: false }
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
        for &s in skeleton {
            ord.push(s);
            for _ in 0..per {
                match fi.next() {
                    Some(f) => ord.push(f),
                    None => break,
                }
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
    pub fn emit(mut self, b: &mut FnBuilder) {
        // statement-granularity experiment switch: split into statement groups (derived dependences), then the cost-based order
        if !self.explicit && !self.label.is_empty() && crate::weave_auto::env_stmt_enabled(&self.label) {
            self = self.fine();
            let o = crate::weave_auto::auto_order(&self.units, &crate::weave_auto::Model::gfx950());
            self.order = Some(o);
            self.explicit = true;
        }
        let ord = self.plan();
        let label = self.label.clone();
        let mut slots: Vec<Option<Unit>> = self.units.v.into_iter().map(Some).collect();
        if !label.is_empty() {
            b.weave_region_begin(&label);
        }
        for i in ord {
            let u = slots[i].take().expect("weave: unit emitted twice");
            b.splice(u.cap);
        }
        if !label.is_empty() {
            b.weave_region_end();
        }
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
    let w = Weaver::new(u);
    match order {
        Some(o) => w.order(o).emit(b),
        None => w.seq().emit(b),
    }
}
