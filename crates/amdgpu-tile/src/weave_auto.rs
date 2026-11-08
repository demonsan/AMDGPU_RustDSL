//! `weave_auto`: a COST-BASED automatic weave strategy (docs/weaver-auto.md). Given the units of a phase it computes an order - a topological order of their dependencies - that minimises
//! the cycles of an in-order, single-wave issue model (one instruction per `issue_gap` cycles whatever the pipe, the MFMA pipe busy `mfma_cycles` per scaled 32x32x64, in-order LDS returns with a
//! queue-depth penalty, result latencies of MFMA / LDS / VALU), under a register-pressure guard. The author's explicit order stays authoritative: this is an optional strategy
//! (`Weaver::order(auto_order(..))`, or the experiment switch `AMDC_WEAVE_AUTO=<phase label>[,..]` for an A/B without editing the kernel).
//!
//! The model is the one calibrated on the warm ATT of the 4-wave FMHA (docs/lgkm-lookahead.md): issue gap 4, MFMA result 84 cycles after issue, `ds_read` about 125-145 cycles to its consumer and much
//! later behind a deep queue, LDS-DMA issue 24. It is NOT a performance oracle below ~3800 simulated cycles per iteration (the hardware is LDS-bandwidth bound there); the strategy is
//! therefore reported with measured numbers, not adopted by default.
//!
//! Instruction counts per unit come from `Unit::cost` (set by the stage with `Units::cost`, or captured); units without one get `estimate`, an interim name / class based guess for the shipped FMHA and
//! GEMM units.

use crate::weave::{Hint, Unit, UnitClass, UnitId, Units};

/// What a unit issues (counts per instruction kind) - the input of the issue model.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnitCost {
    pub ds_b128: u32,
    /// `ds_read_b64_tr_b8`
    pub ds_tr: u32,
    pub ds_b32: u32,
    pub valu: u32,
    /// VALU ops that occupy the issue slot twice as long (`v_cvt_scalef32_*`, trans)
    pub valu8: u32,
    pub salu: u32,
    /// scaled 32x32x64 MFMAs (the MFMA pipe is busy `Model::mfma_cycles` each)
    pub mfma: u32,
    /// 16x16 MFMAs / `smfmac` (busy `Model::mfma16_cycles` each)
    pub mfma16: u32,
    /// LDS-DMA issues
    pub dma: u32,
    /// the unit waits for everything outstanding first (barrier / counted wait / a branch with its own control flow)
    pub fence: bool,
}

/// The machine model (gfx950, one wave per SIMD).
#[derive(Clone, Debug)]
pub struct Model {
    pub issue_gap: i64,
    pub mfma_cycles: i64,
    pub mfma16_cycles: i64,
    /// issue -> result readable
    pub mfma_latency: i64,
    pub mfma16_latency: i64,
    pub valu_latency: i64,
    pub ds_b128_latency: i64,
    pub ds_tr_latency: i64,
    pub ds_b32_latency: i64,
    /// LDS ops outstanding at issue from which the return slows down
    pub lds_queue_depth: usize,
    pub lds_queue_penalty: i64,
    pub dma_issue: i64,
    pub branch_cost: i64,
    /// extra live VGPR dwords the order may use above the declaration order's peak
    pub reg_slack: i64,
    /// improvement passes of the local search
    pub passes: usize,
}

impl Model {
    pub fn gfx950() -> Model {
        Model { issue_gap: 4, mfma_cycles: 64, mfma16_cycles: 32, mfma_latency: 84, mfma16_latency: 48, valu_latency: 8, ds_b128_latency: 144, ds_tr_latency: 125, ds_b32_latency: 64, lds_queue_depth: 8, lds_queue_penalty: 200, dma_issue: 24, branch_cost: 24, reg_slack: 16, passes: 24 }
    }
}

/// Interim cost guess for a unit that has none (the shipped FMHA / GEMM units by name, otherwise by class and latency).
pub fn estimate(u: &Unit<'_>) -> UnitCost {
    if let Some(c) = u.cost {
        return c;
    }
    let n = u.name.as_str();
    let mut c = UnitCost::default();
    if n == "qk.setup" || n == "window.qk.setup" {
        c.salu = 4;
        c.valu = 2;
        c.mfma16 = 1;
    } else if n.starts_with("qk.read") {
        c.ds_b128 = 4;
        c.ds_b32 = 1;
    } else if n.starts_with("qk.mma") || n.starts_with("window.qk.block") {
        c.valu = 1;
        c.mfma = 2;
    } else if n == "pv.setup" {
        c.valu = 4;
    } else if n == "pv.rowsum" {
        c.mfma16 = 2;
    } else if n.starts_with("pv.read") {
        c.ds_tr = 4;
    } else if n.starts_with("pv.mma") {
        c.mfma = 1;
    } else if n == "softmax.decide" {
        c.valu = 24;
        c.fence = true;
    } else if n == "softmax.convert" {
        c.valu = 40;
        c.valu8 = 32;
    } else if n.starts_with("window.convert") {
        c.valu = 10;
        c.valu8 = 8;
    } else if n.starts_with("mma.i") {
        c.mfma16 = 1;
    } else if n.starts_with("reload.") {
        c.ds_b128 = 4;
    } else if n.starts_with("ab.") && u.class == UnitClass::Dma {
        c.dma = 1;
        c.salu = 2;
    } else {
        match u.class {
            UnitClass::Mfma => c.mfma = 1,
            UnitClass::LdsRead => c.ds_b128 = 2,
            UnitClass::Valu => c.valu = (u.latency / 4).max(4),
            UnitClass::Dma => c.dma = 1,
            UnitClass::Salu | UnitClass::Setup => c.salu = 2,
            UnitClass::Branch => {
                c.valu = 8;
                c.fence = true;
            }
        }
    }
    c
}

/// Live dwords a unit adds (its outputs are consumed by later units) and its own peak temporaries: (live_out - live_in, temps). Interim guess for units without a `RegProfile`.
fn reg_facts(u: &Unit<'_>) -> (i64, i64) {
    if u.regs != Default::default() {
        return (u.regs.live_out as i64 - u.regs.live_in as i64, u.regs.temps as i64);
    }
    let n = u.name.as_str();
    if n.starts_with("qk.read") {
        (17, 0)
    } else if n.starts_with("pv.read") {
        (8, 0)
    } else if n.starts_with("qk.mma") || n.starts_with("window.qk.block") {
        (-17, 4)
    } else if n.starts_with("pv.mma") {
        (-8, 2)
    } else {
        (0, 0)
    }
}

#[derive(Clone)]
struct State {
    t: i64,
    mfma_free: i64,
    /// completion times of the LDS ops in flight (in-order)
    lds: Vec<i64>,
    last_lds: i64,
    /// the cycle everything issued so far is readable
    all_ready: i64,
    ready: Vec<i64>,
}

impl State {
    fn new(n: usize) -> State {
        State { t: 0, mfma_free: 0, lds: Vec::new(), last_lds: 0, all_ready: 0, ready: vec![0; n] }
    }

    /// Issues unit `i` (its ops in the order: LDS, DMA, SALU, VALU, MFMA) and records when its result is readable.
    fn step(&mut self, i: usize, u: &Unit<'_>, c: &UnitCost, m: &Model) {
        let mut gate = u.deps.iter().map(|&d| self.ready[d]).max().unwrap_or(0);
        if c.fence {
            gate = gate.max(self.all_ready).max(self.last_lds).max(self.mfma_free);
        }
        let mut t = self.t.max(gate);
        let mut last_kind = 0; // 0 = other, 1 = valu, 2 = lds, 3 = mfma, 4 = mfma16
        let mut result = t;
        macro_rules! issue {
            ($gap:expr) => {{
                t += $gap;
                t
            }};
        }
        let mut lds_op = |this: &mut State, t: i64, lat: i64| -> i64 {
            this.lds.retain(|&x| x > t);
            let pen = if this.lds.len() >= m.lds_queue_depth { m.lds_queue_penalty } else { 0 };
            let comp = (t + lat + pen).max(this.last_lds);
            this.last_lds = comp;
            this.lds.push(comp);
            comp
        };
        for _ in 0..c.ds_b128 {
            let ti = issue!(m.issue_gap);
            result = lds_op(self, ti, m.ds_b128_latency);
            last_kind = 2;
        }
        for _ in 0..c.ds_tr {
            let ti = issue!(m.issue_gap);
            result = lds_op(self, ti, m.ds_tr_latency);
            last_kind = 2;
        }
        for _ in 0..c.ds_b32 {
            let ti = issue!(m.issue_gap);
            result = lds_op(self, ti, m.ds_b32_latency);
            last_kind = 2;
        }
        for _ in 0..c.dma {
            let ti = issue!(m.dma_issue);
            self.lds.retain(|&x| x > ti);
            result = ti;
            last_kind = 0;
        }
        for _ in 0..c.salu {
            result = issue!(m.issue_gap);
            last_kind = 0;
        }
        for _ in 0..c.valu {
            result = issue!(m.issue_gap) + m.valu_latency;
            last_kind = 1;
        }
        for _ in 0..c.valu8 {
            result = issue!(2 * m.issue_gap) + m.valu_latency;
            last_kind = 1;
        }
        for _ in 0..c.mfma {
            t = (t + m.issue_gap).max(self.mfma_free);
            self.mfma_free = t + m.mfma_cycles;
            result = t + m.mfma_latency;
            last_kind = 3;
        }
        for _ in 0..c.mfma16 {
            t = (t + m.issue_gap).max(self.mfma_free);
            self.mfma_free = t + m.mfma16_cycles;
            result = t + m.mfma16_latency;
            last_kind = 4;
        }
        if c.fence {
            t += m.branch_cost;
            result = result.max(t);
        }
        let _ = last_kind;
        self.t = t;
        self.ready[i] = result.max(t);
        self.all_ready = self.all_ready.max(self.ready[i]);
    }
}

/// The modelled cycles of the units in `order` (from phase entry until every result is readable) and the peak extra live dwords.
pub fn simulate(units: &Units<'_>, order: &[UnitId], m: &Model) -> (i64, i64) {
    let mut s = State::new(units.len());
    let (mut live, mut peak) = (0i64, 0i64);
    for &i in order {
        let u = units.unit(i);
        s.step(i, u, &estimate(u), m);
        let (d, t) = reg_facts(u);
        peak = peak.max(live + d.max(0) + t);
        live += d;
        peak = peak.max(live);
    }
    (s.all_ready.max(s.t), peak)
}

fn objective(units: &Units<'_>, order: &[UnitId], m: &Model, peak_cap: i64) -> i64 {
    let (cyc, peak) = simulate(units, order, m);
    cyc * 16 + (peak - peak_cap).max(0) * 4096
}

/// A topological order of `units` minimising the modelled cycles: a greedy list schedule (the ready unit that finishes issuing earliest), then a local search that moves single units to other legal positions
/// while the objective improves. `Anchor` units stay where the greedy pass put them and nothing crosses a `fence` unit; the declaration order is the fallback and the register guard's reference.
pub fn auto_order(units: &Units<'_>, m: &Model) -> Vec<UnitId> {
    let n = units.len();
    if n < 3 {
        return (0..n).collect();
    }
    let decl: Vec<UnitId> = (0..n).collect();
    let (_, decl_peak) = simulate(units, &decl, m);
    let cap = decl_peak + m.reg_slack;
    // greedy list schedule
    let mut done = vec![false; n];
    let mut st = State::new(n);
    let mut order: Vec<UnitId> = Vec::with_capacity(n);
    while order.len() < n {
        let mut best: Option<(i64, usize)> = None;
        for i in 0..n {
            if done[i] || units.unit(i).deps.iter().any(|&d| !done[d]) {
                continue;
            }
            // anchors keep their declaration position relative to other anchors: only pick an anchor when it is the first unfinished anchor
            if units.unit(i).hint == Hint::Anchor && (0..i).any(|j| !done[j] && units.unit(j).hint == Hint::Anchor) {
                continue;
            }
            let u = units.unit(i);
            let mut t = st.clone();
            t.step(i, u, &estimate(u), m);
            let key = t.t * 4096 + t.ready[i].min(4095);
            if best.map_or(true, |(k, _)| key < k) {
                best = Some((key, i));
            }
        }
        let (_, i) = best.expect("a ready unit (the dependences are acyclic by construction)");
        let u = units.unit(i);
        st.step(i, u, &estimate(u), m);
        done[i] = true;
        order.push(i);
    }
    let mut best_cost = objective(units, &order, m, cap);
    // the declaration order may already be better than the greedy one
    let decl_cost = objective(units, &decl, m, cap);
    if decl_cost <= best_cost {
        order = decl;
        best_cost = decl_cost;
    }
    let fence = |i: usize| estimate(units.unit(i)).fence;
    for _ in 0..m.passes {
        let mut improved = false;
        for i in 0..n {
            if units.unit(i).hint == Hint::Anchor {
                continue;
            }
            let p = order.iter().position(|&x| x == i).unwrap();
            // legal window: after the last dependency, before the first consumer, not across a fence unit
            let lo = units.unit(i).deps.iter().map(|&d| order.iter().position(|&x| x == d).unwrap() + 1).max().unwrap_or(0);
            let hi = (0..n).filter(|&j| units.unit(j).deps.contains(&i)).map(|j| order.iter().position(|&x| x == j).unwrap()).min().unwrap_or(n);
            let mut lo = lo;
            let mut hi_cap = hi;
            for (q, &x) in order.iter().enumerate() {
                if x != i && fence(x) {
                    if q < p {
                        lo = lo.max(q + 1);
                    } else {
                        hi_cap = hi_cap.min(q);
                    }
                }
            }
            for q in lo..hi_cap {
                if q == p {
                    continue;
                }
                let mut cand = order.clone();
                cand.remove(p);
                let ins = if q > p { q - 1 } else { q };
                cand.insert(ins.min(cand.len()), i);
                let c = objective(units, &cand, m, cap);
                if c < best_cost {
                    best_cost = c;
                    order = cand;
                    improved = true;
                    break;
                }
            }
        }
        if !improved {
            break;
        }
    }
    order
}

/// `AMDC_WEAVE_AUTO=<label>[,<label>..]` (or `*`): the phase labels whose weave uses `auto_order` instead of the declaration order (an A/B switch; an explicit `Weaver::order` is never overridden).
pub fn env_enabled(label: &str) -> bool {
    match std::env::var("AMDC_WEAVE_AUTO") {
        Ok(v) if !v.is_empty() => v.split(',').any(|x| x.trim() == "*" || x.trim() == label),
        _ => false,
    }
}