//! Emission units (docs/fusion-stage-library-design.md, docs/weave-capture.md, docs/two-weavers.md): a stage instantiated for one shape returns [`Unit`]s instead of emitting. Each unit holds a small,
//! self-contained piece of IR captured from the builder (one QK block, one PV block, one DMA instruction, ...; `Units::capture`) plus the facts a weaver needs to place it - dependences on
//! earlier units (derived from the statement facts), a resource class, a nominal latency, a register profile and an interleave hint. This is the capture container only (IR level, no
//! ordering policy); the front-end Weaver that orders and emits units is `amdgpu_stage::weave`.

use crate::facts::{must_order, Facts, ResClass};
use crate::{Captured, FnBuilder};

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

/// One emission unit.
pub struct Unit {
    pub name: String,
    pub class: UnitClass,
    pub hint: Hint,
    /// nominal issue -> result latency (cycles)
    pub latency: u32,
    /// units that must be emitted before this one (indices in the owning [`Units`])
    pub deps: Vec<UnitId>,
    pub regs: RegProfile,
    /// iteration offset of the unit's work relative to the loop iteration it is emitted in (0 = this tile, +1 = the next one): `rotate`
    pub iter: i32,
    /// statement facts (one per top-level node, in order)
    pub stmts: Vec<Facts>,
    /// the facts of the whole unit
    pub summary: Facts,
    /// unsplittable node ranges (`Captured::groups`)
    pub groups: Vec<std::ops::Range<usize>>,
    pub cap: Captured,
}

/// The union of the facts of several nodes (a unit's summary): defs, uses not defined inside, variable reads / writes, effects, footprint.
fn merge_facts(fs: &[Facts]) -> Facts {
    let mut s = Facts { class: ResClass::Region, defs: vec![], uses: vec![], var_reads: vec![], var_writes: vec![], eff: Default::default(), tile: None, footprint: 0 };
    for f in fs {
        s.defs.extend(f.defs.iter().copied());
        for u in &f.uses {
            if !s.uses.contains(u) {
                s.uses.push(*u);
            }
        }
        for v in &f.var_reads {
            if !s.var_reads.contains(v) {
                s.var_reads.push(*v);
            }
        }
        for v in &f.var_writes {
            if !s.var_writes.contains(v) {
                s.var_writes.push(*v);
            }
        }
        let (e, o) = (&mut s.eff, &f.eff);
        e.lds_read |= o.lds_read;
        e.lds_write |= o.lds_write;
        e.global_read |= o.global_read;
        e.global_write |= o.global_write;
        e.dma |= o.dma;
        e.barrier |= o.barrier;
        e.wait_lds = match (e.wait_lds, o.wait_lds) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        e.wait_vm = match (e.wait_vm, o.wait_vm) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        e.fence |= o.fence;
        e.marker |= o.marker;
        e.directive |= o.directive;
        e.branch |= o.branch;
        e.exec_change |= o.exec_change;
        e.asm |= o.asm;
        s.footprint += f.footprint;
    }
    let defs = s.defs.clone();
    s.uses.retain(|u| !defs.contains(u));
    s
}

/// The unit class / hint / latency derived from statement facts.
fn classify(fs: &[Facts], sum: &Facts) -> (UnitClass, Hint, u32) {
    let any = |p: &dyn Fn(&ResClass) -> bool| fs.iter().any(|f| p(&f.class));
    let class = if sum.eff.branch {
        UnitClass::Branch
    } else if any(&|c| matches!(c, ResClass::Mfma { .. })) {
        UnitClass::Mfma
    } else if sum.eff.dma {
        UnitClass::Dma
    } else if any(&|c| matches!(c, ResClass::LdsRead)) {
        UnitClass::LdsRead
    } else if any(&|c| matches!(c, ResClass::Valu | ResClass::Trans)) {
        UnitClass::Valu
    } else if any(&|c| matches!(c, ResClass::Salu)) {
        UnitClass::Salu
    } else {
        UnitClass::Setup
    };
    let hint = if sum.eff.anchor() || sum.eff.branch {
        Hint::Anchor
    } else {
        match class {
            UnitClass::Mfma => Hint::Skeleton,
            UnitClass::LdsRead | UnitClass::Dma => Hint::LatencySensitive,
            _ => Hint::Filler,
        }
    };
    let latency = fs.iter().map(|f| f.class.latency()).max().unwrap_or(0);
    (class, hint, latency)
}

/// The units of one or more stage instances.
#[derive(Default)]
pub struct Units {
    pub v: Vec<Unit>,
}

impl Units {
    pub fn new() -> Units {
        Units { v: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.v.len()
    }
    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }
    /// Adds a unit: `f` runs NOW into a capture buffer (its values are usable by the code that follows, e.g. later units), and the statements
    /// are spliced where the weave places the unit. Dependencies on the earlier units are derived from the statement facts (values, loop
    /// variables, effects: `crate::facts::must_order`).
    /// Class, hint, latency and register footprint are filled from the facts. Captured in order and woven in declaration order, the program
    /// is identical to building it directly.
    pub fn capture<R>(&mut self, b: &mut FnBuilder, name: impl Into<String>, f: impl FnOnce(&mut FnBuilder) -> R) -> (UnitId, R) {
        let (cap, r) = b.capture(f);
        let stmts = b.facts(&cap);
        let groups = cap.groups();
        let sum = merge_facts(&stmts);
        let deps: Vec<UnitId> = (0..self.v.len())
            .filter(|&j| must_order(&self.v[j].summary, &sum))
            .collect();
        let (class, hint, latency) = classify(&stmts, &sum);
        let regs = RegProfile { live_in: 0, live_out: sum.footprint, temps: sum.footprint, agpr_ok: false };
        self.v.push(Unit { name: name.into(), class, hint, latency, deps, regs, iter: 0, stmts, summary: sum, groups, cap });
        (self.v.len() - 1, r)
    }
    /// `capture` for an ordering anchor (waits, barriers, markers): hint `Anchor` regardless of the facts.
    pub fn anchor<R>(&mut self, b: &mut FnBuilder, name: impl Into<String>, f: impl FnOnce(&mut FnBuilder) -> R) -> (UnitId, R) {
        let (id, r) = self.capture(b, name, f);
        self.v[id].hint = Hint::Anchor;
        (id, r)
    }
    /// Sets the iteration offset of unit `id` (see `rotate`).
    pub fn set_iter(&mut self, id: UnitId, iter: i32) {
        self.v[id].iter = iter;
    }
    /// Splits every unit into its unsplittable node groups (statement granularity; a region / tile region / stage / weave-region mark pair
    /// stays whole). Dependencies between all the groups are re-derived from the statement facts. Sub-unit names are `<unit>#<k>`; declaration
    /// order = the original statement order.
    pub fn split(self) -> Units {
        let mut out = Units::new();
        for u in self.v {
            let mut nodes = u.cap.nodes.into_iter();
            for (k, g) in u.groups.iter().enumerate() {
                let part: Vec<crate::Node> = nodes.by_ref().take(g.len()).collect();
                let fs: Vec<Facts> = u.stmts[g.clone()].to_vec();
                let sum = merge_facts(&fs);
                let deps: Vec<UnitId> = (0..out.v.len()).filter(|&j| must_order(&out.v[j].summary, &sum)).collect();
                let (class, hint, latency) = classify(&fs, &sum);
                let regs = RegProfile { live_in: 0, live_out: sum.footprint, temps: sum.footprint, agpr_ok: false };
                let groups = vec![0..part.len()];
                out.v.push(Unit { name: format!("{}#{k}", u.name), class, hint, latency, deps, regs, iter: u.iter, stmts: fs, summary: sum, groups, cap: Captured { nodes: part } });
            }
        }
        out
    }
    /// Appends another unit list (its ids are shifted by the current length); returns the shift. `after`: extra dependencies of every appended
    /// unit (e.g. the last unit of a phase that must come first).
    pub fn append(&mut self, other: Units, after: &[UnitId]) -> usize {
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
    pub fn unit(&self, id: UnitId) -> &Unit {
        &self.v[id]
    }
    /// Ids of the units with iteration offset `iter`.
    pub fn with_iter(&self, iter: i32) -> Vec<UnitId> {
        (0..self.v.len()).filter(|&i| self.v[i].iter == iter).collect()
    }
    /// Emits every unit in declaration order (the stage's own order: what the stage emitted before it returned units; the weaver's `seq` without a label).
    pub fn emit_all(self, b: &mut FnBuilder) {
        for u in self.v {
            b.splice(u.cap);
        }
    }
}

