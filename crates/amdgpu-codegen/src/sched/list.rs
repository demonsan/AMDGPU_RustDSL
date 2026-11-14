//! Register-pressure-aware list scheduler for one region.
//!
//! Forward, cycle-driven: a wave issues one instruction per cycle in order; an instruction starts
//! when its producers' latencies have elapsed and its pipe is free. Among the instructions that
//! can start earliest, the one with the longest latency-weighted path to the end of the region
//! (critical-path height) wins, ties broken by original order. A pressure model (live VGPR
//! dwords, tuples counted at full width) vetoes candidates that would exceed the limit while a
//! pressure-neutral alternative exists.

use std::collections::{HashMap, HashSet};

use super::depgraph::Graph;
use super::effects::*;
use super::model::{self, OpCost, NPIPES};
use crate::mir::*;

struct Pressure {
    /// vreg -> (remaining reading instructions, currently live); dense by vreg number (only the vregs some instruction names are ever queried)
    st: Vec<(u32, bool)>,
    live_out: Vec<bool>,
    width: Vec<i64>,
    cur: i64,
}

fn vregs_of(m: &MFunction, slots: &[Slot]) -> Vec<VReg> {
    let mut v: Vec<VReg> = Vec::new();
    for &s in slots {
        if s >= SCC {
            continue;
        }
        let r = s / 16;
        if (r as usize) < m.vregs.len() && m.vreg_class(r).is_vector() && !v.contains(&r) {
            v.push(r);
        }
    }
    v
}

impl Pressure {
    fn new(m: &MFunction, rd: &[Vec<VReg>], wr: &[Vec<VReg>], live_in: &HashSet<VReg>, live_out: &HashSet<VReg>) -> Pressure {
        let nv = m.vregs.len();
        let mut st: Vec<(u32, bool)> = vec![(0, false); nv];
        let mut seen = vec![false; nv];
        let mut width = vec![0i64; nv];
        for i in 0..rd.len() {
            for &v in rd[i].iter().chain(wr[i].iter()) {
                let u = v as usize;
                if !seen[u] {
                    seen[u] = true;
                    st[u] = (0, live_in.contains(&v));
                }
                width[u] = m.vreg_class(v).dwords() as i64;
            }
            for &v in &rd[i] {
                st[v as usize].0 += 1;
            }
        }
        let mut cur = 0i64;
        let mut lo = vec![false; nv];
        for &v in live_out {
            if (v as usize) < nv {
                lo[v as usize] = true;
            }
            if m.vreg_class(v).is_vector() && !seen[v as usize] {
                cur += m.vreg_class(v).dwords() as i64;
            }
        }
        for u in 0..nv {
            if seen[u] && st[u].1 {
                cur += width[u];
            }
        }
        Pressure { st, live_out: lo, width, cur }
    }

    #[inline]
    fn after(&self, v: VReg, reads: bool) -> (u32, bool, bool) {
        let (rr, live) = self.st[v as usize];
        let rr_after = rr - if reads { 1 } else { 0 };
        let live_after = self.live_out[v as usize] || rr_after > 0;
        (rr_after, live, live_after)
    }

    fn delta(&self, rd: &[VReg], wr: &[VReg]) -> i64 {
        let mut d = 0i64;
        for &v in rd.iter().chain(wr.iter().filter(|w| !rd.contains(w))) {
            let (_, live, live_after) = self.after(v, rd.contains(&v));
            d += self.width[v as usize] * (live_after as i64 - live as i64);
        }
        d
    }

    fn commit(&mut self, rd: &[VReg], wr: &[VReg]) {
        for &v in rd.iter().chain(wr.iter().filter(|w| !rd.contains(w))) {
            let (rr_after, live, live_after) = self.after(v, rd.contains(&v));
            self.cur += self.width[v as usize] * (live_after as i64 - live as i64);
            self.st[v as usize] = (rr_after, live_after);
        }
    }
}

pub struct ListResult {
    pub perm: Vec<usize>,
    pub est_before: u32,
    pub est_after: u32,
    pub peak_before: i64,
    pub peak_after: i64,
    pub limit_eff: i64,
    pub window: usize,
}

/// Cycle estimate of issuing `order` in order on the machine model.
pub fn estimate(g: &Graph, costs: &[OpCost], order: &[usize]) -> u32 {
    let n = g.n;
    let mut start = vec![0u32; n];
    let mut pipe_free = [0u32; NPIPES];
    let mut cycle = 0u32;
    let mut end = 0u32;
    for &i in order {
        let mut t = cycle;
        for &(p, lat) in &g.preds[i] {
            t = t.max(start[p] + lat);
        }
        t = t.max(pipe_free[costs[i].pipe.index()]);
        start[i] = t;
        pipe_free[costs[i].pipe.index()] = t + costs[i].issue;
        cycle = t + 1;
        end = end.max(t + costs[i].issue);
    }
    end
}

/// What every window of one block shares: the cost table, the vreg reads / writes per instruction, the heights and the original order's estimate and pressure peak.
pub struct Pre {
    costs: Vec<OpCost>,
    rd: Vec<Vec<VReg>>,
    wr: Vec<Vec<VReg>>,
    height: Vec<u32>,
    est_before: u32,
    peak_before: i64,
}

pub fn prepare(m: &MFunction, insts: &[MInst], g: &Graph, live_in: &HashSet<VReg>, live_out: &HashSet<VReg>) -> Pre {
    let n = g.n;
    let costs: Vec<OpCost> = insts.iter().map(|i| model::cost(i.op)).collect();
    let rd: Vec<Vec<VReg>> = g.eff.iter().map(|e| vregs_of(m, &e.reads)).collect();
    let wr: Vec<Vec<VReg>> = g.eff.iter().map(|e| vregs_of(m, &e.writes)).collect();
    // original order: estimate and pressure peak
    let ident: Vec<usize> = (0..n).collect();
    let est_before = estimate(g, &costs, &ident);
    let mut pr0 = Pressure::new(m, &rd, &wr, live_in, live_out);
    let mut peak_before = pr0.cur;
    for i in 0..n {
        pr0.commit(&rd[i], &wr[i]);
        peak_before = peak_before.max(pr0.cur);
    }
    // heights
    let mut height = vec![0u32; n];
    for i in (0..n).rev() {
        let mut h = costs[i].latency;
        if g.term_vcc_writer == Some(i) {
            h = h.max(model::VCC_TO_BRANCH + 8);
        }
        for &(s, lat) in &g.succs[i] {
            h = h.max(lat + height[s]);
        }
        height[i] = h;
    }
    Pre { costs, rd, wr, height, est_before, peak_before }
}

pub fn schedule(
    m: &MFunction,
    insts: &[MInst],
    g: &Graph,
    live_in: &HashSet<VReg>,
    live_out: &HashSet<VReg>,
    limit: i64,
    window: usize,
) -> ListResult {
    let pre = prepare(m, insts, g, live_in, live_out);
    schedule_with(m, g, live_in, live_out, &pre, limit, window)
}

fn schedule_with(m: &MFunction, g: &Graph, live_in: &HashSet<VReg>, live_out: &HashSet<VReg>, pre: &Pre, limit: i64, window: usize) -> ListResult {
    let n = g.n;
    let (costs, rd, wr, height, est_before, peak_before) = (&pre.costs, &pre.rd, &pre.wr, &pre.height, pre.est_before, pre.peak_before);
    let limit_eff = limit.max(peak_before);

    let mut pr = Pressure::new(m, rd, wr, live_in, live_out);
    let mut peak_after = pr.cur;
    let mut indeg: Vec<usize> = g.preds.iter().map(|p| p.len()).collect();
    let mut ready_time = vec![0u32; n];
    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut pipe_free = [0u32; NPIPES];
    let mut cycle = 0u32;
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut done = vec![false; n];
    let mut lowest = 0usize;
    let mut start = vec![0u32; n];
    // candidate pool: (index in ready, start time, delta, fits); the buffer is reused across the steps
    let mut pool: Vec<(usize, u32, i64, bool)> = Vec::new();
    while order.len() < n {
        // earliest start among the ready instructions inside the window
        let mut tmin = u32::MAX;
        pool.clear();
        for (k, &c) in ready.iter().enumerate() {
            if c >= lowest.saturating_add(window) {
                continue;
            }
            let t = ready_time[c].max(pipe_free[costs[c].pipe.index()]).max(cycle);
            tmin = tmin.min(t);
            let d = pr.delta(&rd[c], &wr[c]);
            let fits = pr.cur + d <= limit_eff || d <= 0;
            pool.push((k, t, d, fits));
        }
        let pick_best = |sel: &dyn Fn(&(usize, u32, i64, bool)) -> bool| -> Option<usize> {
            let mut best: Option<(u32, std::cmp::Reverse<usize>, usize)> = None;
            for p in pool.iter().filter(|p| sel(p)) {
                let c = ready[p.0];
                let key = (height[c], std::cmp::Reverse(c), p.0);
                if best.map_or(true, |b| (key.0, key.1) > (b.0, b.1)) {
                    best = Some(key);
                }
            }
            best.map(|b| b.2)
        };
        let k = if let Some(k) = pick_best(&|p| p.1 == tmin && p.3) {
            k
        } else if let Some(k) = {
            // nothing fits at the earliest start: take the earliest fitting one
            let tf = pool.iter().filter(|p| p.3).map(|p| p.1).min();
            tf.and_then(|tf| pick_best(&|p| p.3 && p.1 == tf))
        } {
            k
        } else {
            // nothing fits: least pressure growth
            let dmin = pool.iter().map(|p| p.2).min().unwrap();
            pick_best(&|p| p.2 == dmin).unwrap()
        };        let c = ready.swap_remove(k);
        let t = ready_time[c].max(pipe_free[costs[c].pipe.index()]).max(cycle);
        start[c] = t;
        pipe_free[costs[c].pipe.index()] = t + costs[c].issue;
        cycle = t + 1;
        pr.commit(&rd[c], &wr[c]);
        peak_after = peak_after.max(pr.cur);
        order.push(c);
        done[c] = true;
        while lowest < n && done[lowest] {
            lowest += 1;
        }
        for &(s, lat) in &g.succs[c] {
            ready_time[s] = ready_time[s].max(t + lat);
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }
    let est_after = estimate(g, costs, &order);
    ListResult { perm: order, est_before, est_after, peak_before, peak_after, limit_eff, window }
}
/// Try several scheduling windows (how far past the oldest unscheduled instruction candidates may
/// come from); keep the best estimate that stays within the pressure limit, else the original order.
pub fn schedule_best(
    m: &MFunction,
    insts: &[MInst],
    g: &Graph,
    live_in: &HashSet<VReg>,
    live_out: &HashSet<VReg>,
    limit: i64,
    fast: bool,
) -> ListResult {
    let mut best: Option<ListResult> = None;
    let mut first: Option<ListResult> = None;
    let windows: &[usize] = if fast { &[usize::MAX, 100] } else { &[usize::MAX, 400, 200, 100, 50, 25] };
    let pre = prepare(m, insts, g, live_in, live_out);
    for &w in windows {
        let r = schedule_with(m, g, live_in, live_out, &pre, limit, w);
        if r.peak_after <= r.limit_eff && best.as_ref().map_or(true, |b| r.est_after < b.est_after) {
            best = Some(ListResult { perm: r.perm.clone(), ..r });
        } else if first.is_none() {
            first = Some(r);
        }
    }
    match best {
        Some(b) => b,
        None => {
            let f = first.unwrap();
            ListResult { perm: (0..g.n).collect(), est_after: f.est_before, peak_after: f.peak_before, ..f }
        }
    }
}