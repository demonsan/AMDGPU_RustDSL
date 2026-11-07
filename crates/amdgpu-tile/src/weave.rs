//! Emission units and the weaver (docs/fusion-stage-library-design.md, "Stage / Weaver / backend").
//!
//! A stage instantiated for one shape returns [`Unit`]s instead of emitting: each unit is a closure that emits a small, self-contained piece of IR
//! (one QK block, one PV block, one DMA instruction, ...) plus the facts a weaver needs to place it - dependencies on earlier units, a resource
//! class, a nominal latency, a register profile and an interleave hint. The [`Weaver`] takes the units of one or more stages and emits them in an
//! order: the kernel author's explicit order is authoritative ([`Weaver::order`] / [`Weaver::seq`]); [`Weaver::spread`] is an optional automatic
//! strategy (skeleton units with filler units spread between them). The weave is checked to be a topological order of the dependencies before
//! anything is emitted. The output is ordinary region IR - verified (gpu_verify, GV-STAGE) and compiled as before. Per-shape decisions (which
//! order, which spread) stay at the call site: nothing here is a scheduler.

use amdgpu_dsl::FnBuilder;
use std::cell::RefCell;
use std::rc::Rc;

pub type UnitId = usize;

/// What the unit occupies (for the weaver's strategies and reports).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnitClass {
    /// matrix instructions (and the operand reads they own)
    Mfma,
    Valu,
    LdsRead,
    /// global -> LDS copies
    Dma,
    Salu,
    /// a region with control flow (e.g. the lazy rollback branch)
    Branch,
    /// address / slot setup bound once per instance
    Setup,
}

/// How a weaver strategy may move the unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hint {
    /// the backbone of a phase (MFMAs): fillers are spread between skeleton units
    Skeleton,
    /// independent work that may go into the gaps of the skeleton
    Filler,
    /// should be issued early relative to its consumer (reads, copies)
    LatencySensitive,
    /// must keep its position relative to the phase boundaries (setup, branches)
    Anchor,
}

/// Register facts of a unit (dwords per lane; estimates for planning - the allocator and gpu_verify remain the judges).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegProfile {
    pub live_in: u32,
    pub live_out: u32,
    pub temps: u32,
    /// results may live in AGPRs
    pub agpr_ok: bool,
}

/// A value produced by one unit and consumed by later ones (set when the producer is emitted).
pub struct Out<T>(Rc<RefCell<Option<T>>>);

impl<T> Clone for Out<T> {
    fn clone(&self) -> Self {
        Out(self.0.clone())
    }
}

impl<T> Default for Out<T> {
    fn default() -> Self {
        Out(Rc::new(RefCell::new(None)))
    }
}

impl<T> Out<T> {
    pub fn new() -> Out<T> {
        Out::default()
    }
    pub fn set(&self, v: T) {
        *self.0.borrow_mut() = Some(v);
    }
    /// Borrow the produced value (panics when the producer has not been emitted: the weave is not a topological order).
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        let g = self.0.borrow();
        f(g.as_ref().expect("weave: a unit consumed a value before its producer was emitted"))
    }
}

impl<T: Clone> Out<T> {
    pub fn get(&self) -> T {
        self.with(|v| v.clone())
    }
}

/// One emission unit.
pub struct Unit<'a> {
    pub name: String,
    pub class: UnitClass,
    pub hint: Hint,
    /// nominal issue -> result latency (cycles)
    pub latency: u32,
    /// units that must be emitted before this one (indices in the owning [`Units`])
    pub deps: Vec<UnitId>,
    pub regs: RegProfile,
    emit: Box<dyn FnOnce(&mut FnBuilder) + 'a>,
}

/// The units of one or more stage instances.
#[derive(Default)]
pub struct Units<'a> {
    v: Vec<Unit<'a>>,
}

impl<'a> Units<'a> {
    pub fn new() -> Units<'a> {
        Units { v: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.v.len()
    }
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    /// Adds a unit; returns its id.
    pub fn add(&mut self, name: impl Into<String>, class: UnitClass, hint: Hint, deps: &[UnitId], emit: impl FnOnce(&mut FnBuilder) + 'a) -> UnitId {
        for &d in deps {
            assert!(d < self.v.len(), "Units::add: dependency {d} of `{}` is not an earlier unit", self.v.len());
        }
        self.v.push(Unit { name: name.into(), class, hint, latency: 0, deps: deps.to_vec(), regs: RegProfile::default(), emit: Box::new(emit) });
        self.v.len() - 1
    }
    /// Sets the nominal latency and register profile of unit `id`.
    pub fn facts(&mut self, id: UnitId, latency: u32, regs: RegProfile) {
        self.v[id].latency = latency;
        self.v[id].regs = regs;
    }
    /// Appends another unit list (its ids are shifted by the current length); returns the shift. `after`: extra dependencies of every appended
    /// unit (e.g. the last unit of a phase that must come first).
    pub fn append(&mut self, other: Units<'a>, after: &[UnitId]) -> usize {
        let base = self.v.len();
        for mut u in other.v {
            for d in u.deps.iter_mut() {
                *d += base;
            }
            u.deps.extend_from_slice(after);
            self.v.push(u);
        }
        base
    }
    pub fn unit(&self, id: UnitId) -> &Unit<'a> {
        &self.v[id]
    }
    /// Ids of the units with hint `h`, in declaration order.
    pub fn with_hint(&self, h: Hint) -> Vec<UnitId> {
        (0..self.v.len()).filter(|&i| self.v[i].hint == h).collect()
    }
    /// Emits every unit in declaration order (the stage's own order: what the stage emitted before it returned units).
    pub fn emit_all(self, b: &mut FnBuilder) {
        Weaver::new(self).seq().emit(b);
    }
}

/// Places the units of a phase and emits them.
pub struct Weaver<'a> {
    units: Units<'a>,
    order: Option<Vec<UnitId>>,
    label: String,
}

impl<'a> Weaver<'a> {
    pub fn new(units: Units<'a>) -> Weaver<'a> {
        Weaver { units, order: None, label: String::new() }
    }
    /// A label for reports / errors (the phase name).
    pub fn phase(mut self, label: impl Into<String>) -> Weaver<'a> {
        self.label = label.into();
        self
    }
    pub fn units(&self) -> &Units<'a> {
        &self.units
    }
    /// Declaration order.
    pub fn seq(mut self) -> Weaver<'a> {
        self.order = Some((0..self.units.len()).collect());
        self
    }
    /// The explicit order (authoritative): every unit exactly once.
    pub fn order(mut self, ids: Vec<UnitId>) -> Weaver<'a> {
        self.order = Some(ids);
        self
    }
    /// Strategy: `skeleton` in its given order, `fillers` distributed round-robin after the skeleton units, at most `per` after each (the rest
    /// after the last skeleton unit); every other unit keeps its declaration position before the first skeleton unit.
    pub fn spread(mut self, skeleton: &[UnitId], fillers: &[UnitId], per: usize) -> Weaver<'a> {
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
        self.order = Some(ord);
        self
    }
    /// The order that `emit` would use (after the checks).
    pub fn plan(&self) -> Vec<UnitId> {
        let ord = self.order.clone().unwrap_or_else(|| (0..self.units.len()).collect());
        check_order(&self.units, &ord, &self.label);
        ord
    }
    /// Checks the order (a permutation of the units and a topological order of their dependencies) and emits.
    pub fn emit(self, b: &mut FnBuilder) {
        let ord = self.plan();
        let mut slots: Vec<Option<Unit<'a>>> = self.units.v.into_iter().map(Some).collect();
        for i in ord {
            let u = slots[i].take().expect("weave: unit emitted twice");
            (u.emit)(b);
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
