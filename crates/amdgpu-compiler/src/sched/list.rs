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
    /// vreg -> (remaining reading instructions, currently live)
    st: HashMap<VReg, (u32, bool)>,
    live_out: HashSet<VReg>,
    width: HashMap<VReg, i64>,
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
        let mut st: HashMap<VReg, (u32, bool)> = HashMap::new();
        let mut width = HashMap::new();
        for i in 0..rd.len() {
            for &v in rd[i].iter().chain(wr[i].iter()) {
                st.entry(v).or_insert((0, live_in.contains(&v)));
                width.insert(v, m.vreg_class(v).dwords() as i64);
            }
            for &v in &rd[i] {
                st.get_mut(&v).unwrap().0 += 1;
            }
        }
        let mut cur = 0i64;
        for &v in live_out {
            if m.vreg_class(v).is_vector() && !st.contains_key(&v) {
                cur += m.vreg_class(v).dwords() as i64;
            }
        }
        for (v, (_, live)) in &st {
            if *live {
                cur += width[v];
            }
        }
        Pressure { st, live_out: live_out.clone(), width, cur }
    }

    fn after(&self, v: VReg, reads: bool) -> (u32, bool, bool) {
        let (rr, live) = self.st[&v];
        let rr_after = rr - if reads { 1 } else { 0 };
        let live_after = self.live_out.contains(&v) || rr_after > 0;
        (rr_after, live, live_after)
    }

    fn delta(&self, rd: &[VReg], wr: &[VReg]) -> i64 {
        let mut d = 0i64;
        for &v in rd.iter().chain(wr.iter().filter(|w| !rd.contains(w))) {
            let (_, live, live_after) = self.after(v, rd.contains(&v));
            d += self.width[&v] * (live_after as i64 - live as i64);
        }
        d
    }

    fn commit(&mut self, rd: &[VReg], wr: &[VReg]) {
        let all: Vec<VReg> = rd.iter().chain(wr.iter().filter(|w| !rd.contains(w))).copied().collect();
        for v in all {
            let (rr_after, live, live_after) = self.after(v, rd.contains(&v));
            self.cur += self.width[&v] * (live_after as i64 - live as i64);
            self.st.insert(v, (rr_after, live_after));
        }
    }
}

pub struct ListResult {
    pub perm: Vec<usize>,
    pub est_before: u32,
    pub est_after: u32,
    pub peak_before: i64,
    pub peak_after: i64,
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

pub fn schedule(
    m: &MFunction,
    insts: &[MInst],
    g: &Graph,
    live_in: &HashSet<VReg>,
    live_out: &HashSet<VReg>,
    limit: i64,
) -> ListResult {
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
    let limit_eff = limit.max(peak_before);

    // heights
    let mut height = vec![0u32; n];
    for i in (0..n).rev() {
        let mut h = costs[i].latency;
        for &(s, lat) in &g.succs[i] {
            h = h.max(lat + height[s]);
        }
        height[i] = h;
    }

    let mut pr = Pressure::new(m, &rd, &wr, live_in, live_out);
    let mut peak_after = pr.cur;
    let mut indeg: Vec<usize> = g.preds.iter().map(|p| p.len()).collect();
    let mut ready_time = vec![0u32; n];
    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut pipe_free = [0u32; NPIPES];
    let mut cycle = 0u32;
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut start = vec![0u32; n];
    while order.len() < n {
        // earliest start among the ready instructions
        let mut tmin = u32::MAX;
        for &c in &ready {
            let t = ready_time[c].max(pipe_free[costs[c].pipe.index()]).max(cycle);
            tmin = tmin.min(t);
        }
        // candidates that can start at tmin: pick by (fits pressure, height, original order)
        let mut best: Option<(bool, u32, std::cmp::Reverse<usize>, usize)> = None;
        for (k, &c) in ready.iter().enumerate() {
            let t = ready_time[c].max(pipe_free[costs[c].pipe.index()]).max(cycle);
            if t != tmin {
                continue;
            }
            let d = pr.delta(&rd[c], &wr[c]);
            let fits = pr.cur + d <= limit_eff || d <= 0;
            let key = (fits, height[c], std::cmp::Reverse(c), k);
            if best.map_or(true, |b| (key.0, key.1, key.2) > (b.0, b.1, b.2)) {
                best = Some(key);
            }
        }
        let (_, _, _, k) = best.expect("ready list is never empty while instructions remain");
        let c = ready.swap_remove(k);
        let t = tmin;
        start[c] = t;
        pipe_free[costs[c].pipe.index()] = t + costs[c].issue;
        cycle = t + 1;
        pr.commit(&rd[c], &wr[c]);
        peak_after = peak_after.max(pr.cur);
        order.push(c);
        for &(s, lat) in &g.succs[c] {
            ready_time[s] = ready_time[s].max(t + lat);
            indeg[s] -= 1;
            if indeg[s] == 0 {
                ready.push(s);
            }
        }
    }
    let est_after = estimate(g, &costs, &order);
    ListResult { perm: order, est_before, est_after, peak_before, peak_after }
}