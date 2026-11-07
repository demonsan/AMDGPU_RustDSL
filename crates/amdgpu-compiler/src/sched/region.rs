//! Weave regions: per-region scheduler strategies (docs/fusion-stage-library-design.md, "Weaver").
//!
//! A weave region is a contiguous span of the emitted program (it may cross basic blocks: each block carries its own runs). Every instruction lowered from a region carries the
//! region id (`MInst::region`, 0 = none; stamped by lowering from `Stmt::RegionMark` / stages, ids into `MFunction::reg_hints.regions`, one id per scope INSTANCE). The strategy of a region
//! is keyed by its NAME: the kernel declares `Function::regions` (name, order policy, tokens); `Options::regions[k - 1]` is declaration `k` and `Options::region_of[id]` maps a lowered
//! region id to its strategy (the nearest declared region on the way up the region nesting; 0 = the kernel default). A strategy has
//!  * an ORDER policy: `authored` = the order the weaver emitted is authoritative: the list scheduler, `slotfill` and `ldshoist` never touch it; the scoreboard waits are recomputed (`rewait`)
//!    and the order-preserving post-passes (`ldsink`, `wsink`, `ldahead`, `ldahead32`) still apply. Free (`authored = false`): the region's tokens decide.
//!  * `tokens`: scheduler option tokens (the `AMDC_SCHED_OPTS` vocabulary) applied ON TOP of the kernel's options for the instructions of the region only.
//!
//! Boundaries are never crossed: the list scheduler schedules each run of one region by itself (live sets at the run's ends from the block's liveness), the post-passes stop at a region
//! change. `ldshoist` / `slotfill` (whole-block passes) apply only to a block whose instructions all belong to regions that enable them. Without regions nothing here runs: the scheduler is
//! byte-identical.

use super::Options;
use crate::mir::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionSpec {
    pub name: String,
    /// The emitted order is authoritative (no list scheduling / slotfill / ldshoist inside the region).
    pub authored: bool,
    /// Scheduler option tokens for this region (applied on top of the kernel's options).
    pub tokens: String,
}

impl Options {
    /// The options that apply to the instructions of region `id` (0 = the kernel default): the kernel's options, with the region's tokens on top; an `authored` region never reorders,
    /// slot-fills or hoists. The returned options carry no region table.
    pub fn for_region(&self, id: u16) -> Options {
        let mut o = self.clone();
        o.regions = Vec::new();
        if id == 0 {
            return o;
        }
        let Some(r) = self.regions.get(id as usize - 1) else { return o };
        let lock = |o: &mut Options| {
            if r.authored {
                o.reorder = false;
                o.slotfill = false;
                o.ldshoist = false;
            }
        };
        lock(&mut o);
        o.apply_tokens(&r.tokens);
        lock(&mut o); // tokens cannot re-enable what an authored region forbids
        o
    }

    /// `for_region` for every id 0..=regions.len(), indexable by `region_index`.
    pub fn region_options(&self) -> Vec<Options> {
        (0..=self.regions.len() as u16).map(|i| self.for_region(i)).collect()
    }

    /// Does the kernel default or any region satisfy `f` (the gate of a pass)?
    pub fn any_region(&self, f: impl Fn(&Options) -> bool) -> bool {
        f(&self.for_region(0)) || (1..=self.regions.len() as u16).any(|i| f(&self.for_region(i)))
    }
}

/// The strategy id (`Options::regions[id - 1]`, 0 = the kernel default) of lowered region id `r`: through `region_of` when the compiler filled it, else the id itself.
pub fn strategy_of(o: &Options, r: u16) -> u16 {
    let s = if o.region_of.is_empty() { r } else { o.region_of.get(r as usize).copied().unwrap_or(0) };
    if (s as usize) <= o.regions.len() {
        s
    } else {
        0
    }
}

/// Index into `region_options()` of an instruction's strategy (an unknown id falls back to the default).
pub fn region_index(o: &Options, r: u16) -> usize {
    strategy_of(o, r) as usize
}

/// Maximal runs `[lo, hi)` of equal STRATEGY (adjacent instances of the same declared region are one run); the third field is the strategy id.
pub fn runs(o: &Options, insts: &[MInst]) -> Vec<(usize, usize, u16)> {
    let mut out = Vec::new();
    let mut lo = 0;
    for i in 1..=insts.len() {
        if i == insts.len() || strategy_of(o, insts[i].region) != strategy_of(o, insts[lo].region) {
            if lo < i {
                out.push((lo, i, strategy_of(o, insts[lo].region)));
            }
            lo = i;
        }
    }
    out
}

/// Does this block use regions (and does the kernel declare any)?
pub fn has_regions(o: &Options, insts: &[MInst]) -> bool {
    !o.regions.is_empty() && insts.iter().any(|i| strategy_of(o, i.region) != 0)
}

/// Whole-block passes (`ldshoist`, `slotfill`): the block is processed when the kernel default or - with regions - every instruction's region enables `f`.
pub fn block_allows(o: &Options, insts: &[MInst], f: impl Fn(&Options) -> bool) -> bool {
    if !has_regions(o, insts) {
        return true; // the caller's gate (the kernel option) already decided
    }
    let ro = o.region_options();
    insts.iter().all(|i| f(&ro[region_index(o, i.region)]))
}