//! `weave_auto`: a COST-BASED automatic weave strategy (docs/weaver-auto.md). Given the units of a phase (every unit is a captured piece of IR with statement facts, `Unit::stmts` /
//! `Unit::summary`, and dependences derived from `facts::must_order`) it computes an order - a topological order of the dependences - that minimises the cycles of an in-order, single-wave issue
//! model (one instruction per `issue_gap` cycles whatever the pipe, the MFMA pipe busy 4 x passes, in-order LDS returns with a queue-depth penalty, result latencies of MFMA / LDS / VALU), under a
//! register-pressure guard. The author's explicit order stays authoritative: this is an optional strategy (`Weaver::order(auto_order(..))`, or the experiment switches `AMDC_WEAVE_AUTO=<phase label>[,..]`
//! (unit granularity) and `AMDC_WEAVE_AUTO_STMT=<label>[,..]` (statement granularity: `Weaver::fine()` first) for an A/B without editing the kernel).
//!
//! The model is the one calibrated on the warm ATT of the 4-wave FMHA (docs/lgkm-lookahead.md): issue gap 4, MFMA result 84 cycles after issue, `ds_read` about 125-145 cycles to its consumer and much
//! later behind a deep queue, LDS-DMA issue 24. It is NOT a performance oracle below ~3800 simulated cycles per iteration (the hardware is LDS-bandwidth bound there); the strategy is
//! therefore reported with measured numbers, not adopted by default.

use amdgpu_dsl::units::{Hint, Units};
use amdgpu_dsl::facts::{Facts, ResClass};
use std::collections::HashMap;

/// What an item issues (counts per instruction kind) - the input of the issue model.
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
    /// generic `ds_read` (statement facts do not know the width)
    pub lds: u32,
    /// generic matrix instructions (statement facts): count and busy cycles each
    pub xdl: u32,
    pub xdl_busy: u32,
    /// the item waits for everything outstanding first (barrier / counted wait / a branch with its own control flow)
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
    /// how far (positions) the local search moves an item
    pub window: usize,
}

impl Model {
    pub fn gfx950() -> Model {
        // the machine numbers are the target's (amdgpu_target::gfx950 `latency.weave`); reg_slack / passes / window are the search's own
        let w = &amdgpu_target::gfx950::GFX950.latency.weave;
        Model {
            issue_gap: w.issue_gap,
            mfma_cycles: w.mfma_cycles,
            mfma16_cycles: w.mfma16_cycles,
            mfma_latency: w.mfma_latency,
            mfma16_latency: w.mfma16_latency,
            valu_latency: w.valu_latency,
            ds_b128_latency: w.ds_b128_latency,
            ds_tr_latency: w.ds_tr_latency,
            ds_b32_latency: w.ds_b32_latency,
            lds_queue_depth: w.lds_queue_depth,
            lds_queue_penalty: w.lds_queue_penalty,
            dma_issue: w.dma_issue,
            branch_cost: w.branch_cost,
            reg_slack: 16,
            passes: 24,
            window: 40,
        }
    }
}
/// One schedulable item of the planner: a weave unit or a captured statement (or a group of statements that must not be split).
#[derive(Clone, Debug, Default)]
pub struct Item {
    pub name: String,
    pub cost: UnitCost,
    /// items that must precede this one (ordering: data, memory, anchors, fences)
    pub deps: Vec<usize>,
    /// the subset of `deps` whose RESULT this item waits for (data dependences; the timing gate)
    pub tdeps: Vec<usize>,
    /// anchors and branches: the local search does not move them
    pub fixed: bool,
    /// waits for everything outstanding before it issues (barrier, counted wait, branch)
    pub drain: bool,
    /// value-level register model: (value, dwords) defined / values read; both empty = the `delta` / `temps` model
    pub defs: Vec<(u32, i64)>,
    pub uses: Vec<u32>,
    pub delta: i64,
    pub temps: i64,
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

    fn lds_op(&mut self, t: i64, lat: i64, m: &Model) -> i64 {
        self.lds.retain(|&x| x > t);
        let pen = if self.lds.len() >= m.lds_queue_depth { m.lds_queue_penalty } else { 0 };
        let comp = (t + lat + pen).max(self.last_lds);
        self.last_lds = comp;
        self.lds.push(comp);
        comp
    }

    /// Issues item `i` (its ops in the order: LDS, DMA, SALU, VALU, MFMA) and records when its result is readable.
    fn step(&mut self, i: usize, it: &Item, m: &Model) {
        let c = &it.cost;
        let mut gate = it.tdeps.iter().map(|&d| self.ready[d]).max().unwrap_or(0);
        if it.drain || c.fence {
            gate = gate.max(self.all_ready).max(self.last_lds).max(self.mfma_free);
        }
        let mut t = self.t.max(gate);
        let mut result = t;
        for _ in 0..c.ds_b128 {
            t += m.issue_gap;
            result = self.lds_op(t, m.ds_b128_latency, m);
        }
        for _ in 0..c.lds {
            t += m.issue_gap;
            result = self.lds_op(t, m.ds_b128_latency, m);
        }
        for _ in 0..c.ds_tr {
            t += m.issue_gap;
            result = self.lds_op(t, m.ds_tr_latency, m);
        }
        for _ in 0..c.ds_b32 {
            t += m.issue_gap;
            result = self.lds_op(t, m.ds_b32_latency, m);
        }
        for _ in 0..c.dma {
            t += m.dma_issue;
            self.lds.retain(|&x| x > t);
            result = t;
        }
        for _ in 0..c.salu {
            t += m.issue_gap;
            result = t;
        }
        for _ in 0..c.valu {
            t += m.issue_gap;
            result = t + m.valu_latency;
        }
        for _ in 0..c.valu8 {
            t += 2 * m.issue_gap;
            result = t + m.valu_latency;
        }
        for _ in 0..c.mfma {
            t = (t + m.issue_gap).max(self.mfma_free);
            self.mfma_free = t + m.mfma_cycles;
            result = t + m.mfma_latency;
        }
        for _ in 0..c.mfma16 {
            t = (t + m.issue_gap).max(self.mfma_free);
            self.mfma_free = t + m.mfma16_cycles;
            result = t + m.mfma16_latency;
        }
        for _ in 0..c.xdl {
            t = (t + m.issue_gap).max(self.mfma_free);
            self.mfma_free = t + c.xdl_busy as i64;
            result = t + c.xdl_busy as i64 + (m.mfma_latency - m.mfma_cycles);
        }
        if it.drain || c.fence {
            t += m.branch_cost;
            result = result.max(t);
        }
        self.t = t;
        self.ready[i] = result.max(t);
        self.all_ready = self.all_ready.max(self.ready[i]);
    }
}

/// The modelled cycles of `items` issued in `order` (from phase entry until every result is readable) and the peak live dwords the order adds.
pub fn simulate_items(items: &[Item], order: &[usize], m: &Model) -> (i64, i64) {
    let mut s = State::new(items.len());
    let value_level = items.iter().any(|it| !it.defs.is_empty() || !it.uses.is_empty());
    let (mut live, mut peak) = (0i64, 0i64);
    let mut dw: HashMap<u32, i64> = HashMap::new();
    let mut remaining: HashMap<u32, u32> = HashMap::new();
    if value_level {
        for it in items {
            for &(v, d) in &it.defs {
                dw.insert(v, d);
            }
        }
        for it in items {
            for &u in &it.uses {
                if dw.contains_key(&u) {
                    *remaining.entry(u).or_insert(0) += 1;
                }
            }
        }
    }
    for &i in order {
        let it = &items[i];
        s.step(i, it, m);
        if value_level {
            for &(_, d) in &it.defs {
                live += d;
            }
            peak = peak.max(live + it.temps);
            for u in &it.uses {
                if let Some(r) = remaining.get_mut(u) {
                    *r -= 1;
                    if *r == 0 {
                        live -= dw[u];
                    }
                }
            }
        } else {
            peak = peak.max(live + it.delta.max(0) + it.temps);
            live += it.delta;
            peak = peak.max(live);
        }
    }
    (s.all_ready.max(s.t), peak)
}

fn objective(items: &[Item], order: &[usize], m: &Model, peak_cap: i64) -> i64 {
    let (cyc, peak) = simulate_items(items, order, m);
    cyc * 16 + (peak - peak_cap).max(0) * 4096
}

/// A topological order of `items` minimising the modelled cycles: a greedy list schedule (the ready item that finishes issuing earliest), then a local search that moves single items to other legal
/// positions (within `m.window`) while the objective improves. `fixed` items stay where the greedy pass put them; the declaration order is the fallback and the register guard's reference.
pub fn plan_items(items: &[Item], m: &Model) -> Vec<usize> {
    let n = items.len();
    if n < 3 {
        return (0..n).collect();
    }
    let decl: Vec<usize> = (0..n).collect();
    let (_, decl_peak) = simulate_items(items, &decl, m);
    let cap = decl_peak + m.reg_slack;
    let mut done = vec![false; n];
    let mut st = State::new(n);
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut first_fixed = 0usize;
    while order.len() < n {
        while first_fixed < n && (done[first_fixed] || !items[first_fixed].fixed) {
            first_fixed += 1;
        }
        let mut best: Option<(i64, usize)> = None;
        for i in 0..n {
            if done[i] || items[i].deps.iter().any(|&d| !done[d]) {
                continue;
            }
            // fixed items keep their declaration order among themselves
            if items[i].fixed && i != first_fixed {
                continue;
            }
            let mut t = st.clone();
            t.step(i, &items[i], m);
            let key = t.t * 4096 + t.ready[i].min(4095);
            if best.map_or(true, |(k, _)| key < k) {
                best = Some((key, i));
            }
        }
        let (_, i) = best.expect("a ready item (the dependences are acyclic by construction)");
        st.step(i, &items[i], m);
        done[i] = true;
        order.push(i);
    }
    let mut best_cost = objective(items, &order, m, cap);
    let decl_cost = objective(items, &decl, m, cap);
    if decl_cost <= best_cost {
        order = decl;
        best_cost = decl_cost;
    }
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (j, it) in items.iter().enumerate() {
        for &d in &it.deps {
            consumers[d].push(j);
        }
    }
    for _ in 0..m.passes {
        let mut improved = false;
        let mut pos = vec![0usize; n];
        for i in 0..n {
            for (q, &x) in order.iter().enumerate() {
                pos[x] = q;
            }
            if items[i].fixed {
                continue;
            }
            let p = pos[i];
            let lo = items[i].deps.iter().map(|&d| pos[d] + 1).max().unwrap_or(0).max(p.saturating_sub(m.window));
            let hi = consumers[i].iter().map(|&j| pos[j]).min().unwrap_or(n).min(p + m.window + 1);
            for q in lo..hi {
                if q == p {
                    continue;
                }
                let mut cand = order.clone();
                cand.remove(p);
                let ins = if q > p { q - 1 } else { q };
                cand.insert(ins.min(cand.len()), i);
                let c = objective(items, &cand, m, cap);
                if c < best_cost {
                    best_cost = c;
                    order = cand;
                    improved = true;
                    for (q2, &x) in order.iter().enumerate() {
                        pos[x] = q2;
                    }
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

fn class_cost(c: ResClass) -> UnitCost {
    let mut k = UnitCost::default();
    match c {
        ResClass::Mfma { passes } => {
            k.xdl = 1;
            k.xdl_busy = 4 * passes as u32;
        }
        ResClass::Valu => k.valu = 1,
        ResClass::Trans => k.valu8 = 1,
        ResClass::Salu | ResClass::LdsWrite | ResClass::VmemStore => k.salu = 1,
        ResClass::LdsRead => k.lds = 1,
        ResClass::VmemLoad | ResClass::Dma => k.dma = 1,
        ResClass::Sync | ResClass::Asm => k.salu = 1,
        ResClass::Marker => {}
        ResClass::Region => k.valu = 8,
    }
    k
}

fn add_cost(a: &mut UnitCost, b: &UnitCost) {
    a.ds_b128 += b.ds_b128;
    a.ds_tr += b.ds_tr;
    a.ds_b32 += b.ds_b32;
    a.valu += b.valu;
    a.valu8 += b.valu8;
    a.salu += b.salu;
    a.mfma += b.mfma;
    a.mfma16 += b.mfma16;
    a.dma += b.dma;
    a.lds += b.lds;
    a.xdl += b.xdl;
    a.xdl_busy = a.xdl_busy.max(b.xdl_busy);
}

pub fn data_dep(a: &Facts, b: &Facts) -> bool {
    let hits = |x: &[amdgpu_dsl::VarId], y: &[amdgpu_dsl::VarId]| x.iter().any(|v| y.contains(v));
    b.uses.iter().any(|u| a.defs.contains(u)) || hits(&a.var_writes, &b.var_reads) || hits(&a.var_reads, &b.var_writes) || hits(&a.var_writes, &b.var_writes)
}

// ---------------------------------------------------------------- weave units

/// The planner items of a phase's units: costs from the statement classes, ordering dependences = the units' derived `deps`, timing dependences = their data dependences, the register model at value level
/// (a unit's defs share its footprint).
pub fn items_from_units(units: &Units) -> Vec<Item> {
    (0..units.len())
        .map(|i| {
            let u = units.unit(i);
            let mut cost = UnitCost::default();
            for st in &u.stmts {
                add_cost(&mut cost, &class_cost(st.class));
            }
            let e = &u.summary.eff;
            let tdeps: Vec<usize> = u.deps.iter().copied().filter(|&d| data_dep(&units.unit(d).summary, &u.summary)).collect();
            let per = if u.summary.defs.is_empty() { 0 } else { (u.summary.footprint as i64 / u.summary.defs.len() as i64).max(if u.summary.footprint > 0 { 1 } else { 0 }) };
            let defs: Vec<(u32, i64)> = u.summary.defs.iter().map(|&v| (v as u32, per)).filter(|&(_, d)| d > 0).collect();
            let uses: Vec<u32> = u.summary.uses.iter().map(|&v| v as u32).collect();
            Item {
                name: u.name.clone(),
                cost,
                deps: u.deps.clone(),
                tdeps,
                fixed: u.hint == Hint::Anchor || e.anchor() || e.branch,
                drain: e.barrier || e.wait_lds.is_some() || e.wait_vm.is_some() || e.branch || e.asm,
                defs,
                uses,
                delta: 0,
                temps: 0,
            }
        })
        .collect()
}

/// The modelled cycles of the units in `order` and the peak live dwords.
pub fn simulate(units: &Units, order: &[usize], m: &Model) -> (i64, i64) {
    simulate_items(&items_from_units(units), order, m)
}

/// A topological order of the units of a phase (or, after `Weaver::fine()`, of its statement groups) minimising the modelled cycles (see [`plan_items`]); anchors and branches stay where the greedy pass put
/// them. Debug aids: `AMDC_WEAVE_AUTO_VERBOSE=1` prints the plan, `AMDC_WEAVE_AUTO_PREFIX=N` keeps only the first N planned items (the rest in declaration order; `0` = the identity).
pub fn auto_order(units: &Units, m: &Model) -> Vec<usize> {
    let items = items_from_units(units);
    let mut order = plan_items(&items, m);
    if let Some(n) = std::env::var("AMDC_WEAVE_AUTO_PREFIX").ok().and_then(|v| v.parse::<usize>().ok()) {
        let head: Vec<usize> = order.iter().copied().take(n).collect();
        let mut rest: Vec<usize> = (0..items.len()).filter(|i| !head.contains(i)).collect();
        rest.sort_unstable();
        order = head.into_iter().chain(rest).collect();
    }
    if std::env::var("AMDC_WEAVE_AUTO_VERBOSE").map_or(false, |v| !v.is_empty()) {
        let (c0, p0) = simulate_items(&items, &(0..items.len()).collect::<Vec<_>>(), m);
        let (c1, p1) = simulate_items(&items, &order, m);
        eprintln!("weave_auto: {} items; modelled cycles {c0} -> {c1}, peak live {p0} -> {p1}", items.len());
        for (pos, &g) in order.iter().enumerate() {
            let u = units.unit(g);
            eprintln!("  {pos:3} <- {g:3} {} {:?} {:?}", u.name, u.class, u.hint);
        }
    }
    order
}

/// `AMDC_WEAVE_AUTO=<label>[,<label>..]` (or `*`): the phase labels whose weave uses `auto_order` instead of the declaration order (an A/B switch; an explicit `Weaver::order` is never overridden).
pub fn env_enabled(label: &str) -> bool {
    env_label("AMDC_WEAVE_AUTO", label)
}

/// `AMDC_WEAVE_AUTO_STMT=<label>[,..]`: the same at STATEMENT granularity - the weave is split with `Weaver::fine()` and its statement groups ordered by `auto_order` (legality from the derived dependences).
pub fn env_stmt_enabled(label: &str) -> bool {
    env_label("AMDC_WEAVE_AUTO_STMT", label)
}

fn env_label(var: &str, label: &str) -> bool {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => v.split(',').any(|x| x.trim() == "*" || x.trim() == label),
        _ => false,
    }
}