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
use amdgpu_dsl::facts::{self, Facts, ResClass};
use amdgpu_dsl::{Captured, Function, Node};
use std::collections::HashMap;

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
    /// generic ds_read (width unknown: statement facts)
    pub lds: u32,
    /// generic matrix instructions (statement facts): count and busy cycles each
    pub xdl: u32,
    pub xdl_busy: u32,
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
    /// how far (positions) the local search moves an item
    pub window: usize,
}

impl Model {
    pub fn gfx950() -> Model {
        Model { issue_gap: 4, mfma_cycles: 64, mfma16_cycles: 32, mfma_latency: 84, mfma16_latency: 48, valu_latency: 8, ds_b128_latency: 144, ds_tr_latency: 125, ds_b32_latency: 64, lds_queue_depth: 8, lds_queue_penalty: 200, dma_issue: 24, branch_cost: 24, reg_slack: 16, passes: 24, window: 40 }
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

// ---------------------------------------------------------------- weave units

fn items_from_units(units: &Units<'_>) -> Vec<Item> {
    let n = units.len();
    let is_fence = |i: usize| estimate(units.unit(i)).fence;
    (0..n)
        .map(|i| {
            let u = units.unit(i);
            // effective ordering dependences: the real ones plus the fences' - a fence unit follows every unit declared before it and precedes every unit declared after it
            let mut d = u.deps.clone();
            if is_fence(i) {
                d.extend(0..i);
            }
            d.extend((0..i).filter(|&f| is_fence(f)));
            d.sort_unstable();
            d.dedup();
            let (delta, temps) = reg_facts(u);
            Item { name: u.name.clone(), cost: estimate(u), deps: d, tdeps: u.deps.clone(), fixed: u.hint == Hint::Anchor, drain: false, delta, temps, ..Default::default() }
        })
        .collect()
}

/// The modelled cycles of the units in `order` and the peak extra live dwords.
pub fn simulate(units: &Units<'_>, order: &[UnitId], m: &Model) -> (i64, i64) {
    simulate_items(&items_from_units(units), order, m)
}

/// A topological order of the units of a phase minimising the modelled cycles (see [`plan_items`]); `Anchor` units stay where the greedy pass put them and nothing crosses a `fence` unit.
pub fn auto_order(units: &Units<'_>, m: &Model) -> Vec<UnitId> {
    plan_items(&items_from_units(units), m)
}

// ---------------------------------------------------------------- captured statements

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

fn data_dep(a: &Facts, b: &Facts) -> bool {
    let hits = |x: &[amdgpu_dsl::VarId], y: &[amdgpu_dsl::VarId]| x.iter().any(|v| y.contains(v));
    b.uses.iter().any(|u| a.defs.contains(u)) || hits(&a.var_writes, &b.var_reads) || hits(&a.var_reads, &b.var_writes) || hits(&a.var_writes, &b.var_writes)
}

/// Items of a capture: the top-level nodes (weave-region mark pairs stay together as one item), their dependences from `facts::must_order`, their costs from the resource classes, the register model at value
/// level. Returns the items and, per item, the node range it stands for.
pub fn items_from_nodes(f: &Function, cap: &Captured) -> (Vec<Item>, Vec<std::ops::Range<usize>>) {
    let groups = cap.groups();
    let node_facts: Vec<Facts> = cap.facts(f);
    let facts_of = |r: &std::ops::Range<usize>| -> Facts {
        if r.len() == 1 {
            node_facts[r.start].clone()
        } else {
            Captured { nodes: cap.nodes[r.clone()].to_vec() }.summary(f)
        }
    };
    let gf: Vec<Facts> = groups.iter().map(facts_of).collect();
    let mut items = Vec::with_capacity(groups.len());
    for (gi, r) in groups.iter().enumerate() {
        let fa = &gf[gi];
        let mut cost = UnitCost::default();
        for k in r.clone() {
            add_cost(&mut cost, &class_cost(node_facts[k].class));
        }
        let mut deps = Vec::new();
        let mut tdeps = Vec::new();
        for gj in 0..gi {
            if facts::must_order(&gf[gj], fa) {
                deps.push(gj);
                if data_dep(&gf[gj], fa) {
                    tdeps.push(gj);
                }
            }
        }
        let drain = fa.eff.barrier || fa.eff.wait_lds.is_some() || fa.eff.wait_vm.is_some() || fa.eff.branch || fa.eff.asm;
        let defs: Vec<(u32, i64)> = fa.defs.iter().map(|&v| (v as u32, facts::dwords(f.value_tys[v as usize]) as i64)).filter(|&(_, d)| d > 0).collect();
        let uses: Vec<u32> = fa.uses.iter().map(|&v| v as u32).collect();
        items.push(Item { name: format!("n{}", r.start), cost, deps, tdeps, fixed: fa.eff.anchor() || fa.eff.branch, drain, defs, uses, delta: 0, temps: 0 });
    }
    (items, groups)
}

/// The cost-based order of the top-level nodes of a capture (a permutation of `0..cap.nodes.len()`; every legal reorder by `facts::must_order`, mark-pair groups kept contiguous).
pub fn auto_order_nodes(f: &Function, cap: &Captured, m: &Model) -> Vec<usize> {
    let (items, groups) = items_from_nodes(f, cap);
    let mut order = plan_items(&items, m);
    // bisect aids: `AMDC_WEAVE_AUTO_PREFIX=N` keeps the first N planned items, the rest in declaration order (`0` = the identity)
    if let Some(n) = std::env::var("AMDC_WEAVE_AUTO_PREFIX").ok().and_then(|v| v.parse::<usize>().ok()) {
        let head: Vec<usize> = order.iter().copied().take(n).collect();
        let mut rest: Vec<usize> = (0..items.len()).filter(|i| !head.contains(i)).collect();
        rest.sort_unstable();
        order = head.into_iter().chain(rest).collect();
    }
    if std::env::var("AMDC_WEAVE_AUTO_VERBOSE").map_or(false, |v| !v.is_empty()) {
        let fs = cap.facts(f);
        let (c0, p0) = simulate_items(&items, &(0..items.len()).collect::<Vec<_>>(), m);
        let (c1, p1) = simulate_items(&items, &order, m);
        eprintln!("weave_auto: {} items; modelled cycles {c0} -> {c1}, peak live {p0} -> {p1}", items.len());
        for (pos, &g) in order.iter().enumerate() {
            let k = groups[g].start;
            let e = &fs[k].eff;
            let flags = format!("{}{}{}{}{}{}", if e.lds_read { "Lr " } else { "" }, if e.lds_write { "Lw " } else { "" }, if e.global_read || e.global_write { "G " } else { "" }, if e.anchor() { "ANCHOR " } else { "" }, if e.branch { "BRANCH " } else { "" }, if !fs[k].defs.is_empty() { "def" } else { "" });
            let mut det = format!("{:?}", cap.nodes[k]);
            det.truncate(170);
            eprintln!("  {pos:3} <- {g:3} {:?} {flags} defs={:?} uses={:?} vr={:?} vw={:?} | {det}", fs[k].class, fs[k].defs, fs[k].uses, fs[k].var_reads, fs[k].var_writes);
        }
    }
    order.into_iter().flat_map(|g| groups[g].clone()).collect()
}

/// The modelled cycles / peak live dwords of the capture's nodes in the order `order` (node indices).
pub fn simulate_nodes(f: &Function, cap: &Captured, order: &[usize], m: &Model) -> (i64, i64) {
    let (items, groups) = items_from_nodes(f, cap);
    let gorder: Vec<usize> = order.iter().filter_map(|&k| groups.iter().position(|g| g.start == k)).collect();
    simulate_items(&items, &gorder, m)
}

/// Reorders the nodes of `cap` (by `auto_order_nodes`) ready to be spliced.
pub fn reorder(f: &Function, cap: Captured, m: &Model) -> Vec<Node> {
    let ord = auto_order_nodes(f, &cap, m);
    let mut slots: Vec<Option<Node>> = cap.nodes.into_iter().map(Some).collect();
    ord.into_iter().map(|k| slots[k].take().expect("a permutation")).collect()
}

/// `AMDC_WEAVE_AUTO=<label>[,<label>..]` (or `*`): the phase labels whose weave uses `auto_order` instead of the declaration order (an A/B switch; an explicit `Weaver::order` is never overridden).
pub fn env_enabled(label: &str) -> bool {
    env_label("AMDC_WEAVE_AUTO", label)
}

/// `AMDC_WEAVE_AUTO_STMT=<label>[,..]`: the same at STATEMENT granularity - the weave's units are captured and their statements reordered by `auto_order_nodes` (legality from the statement facts).
pub fn env_stmt_enabled(label: &str) -> bool {
    env_label("AMDC_WEAVE_AUTO_STMT", label)
}

fn env_label(var: &str, label: &str) -> bool {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => v.split(',').any(|x| x.trim() == "*" || x.trim() == label),
        _ => false,
    }
}