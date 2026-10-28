//! Multi-wave SIMD simulator for the machine model: validates the model against hardware
//! (docs/scheduling.md) and reports estimated cycles per loop iteration for a final
//! (post-scheduling, post-hazard) MIR function.
//!
//! One SIMD is simulated with `waves` identical waves running the same loop path. Per wave:
//! in-order issue, at most one instruction per cycle, `issue` cycles between instructions of the
//! same pipe, operands ready after the producer's latency, `s_waitcnt` counting in-flight
//! vm/lgkm operations, `s_nop N` = (N+1)*4 cycles, taken branches cost `BRANCH_TAKEN`. Shared by
//! the waves of the SIMD: each unit is busy `occ` cycles per instruction (plain fp32 ops 2 so two
//! waves dual-issue, everything else 4+), `s_barrier` releases when all waves arrived.

use std::collections::HashMap;

use super::effects::*;
use super::model::*;
use crate::mir::*;

#[derive(Clone)]
struct SInst {
    cost: OpCost,
    reads: Vec<Slot>,
    writes: Vec<Slot>,
    fence: Fence,
    is_vmem: bool,
    is_lds: bool,
    extra: i64, // cycles added after issue (branch cost)
}

struct Wave {
    pc: usize,
    iter: usize,
    last_issue: i64,
    pipe_free: [i64; NPIPES],
    ready: HashMap<Slot, i64>,
    vm: Vec<i64>,
    lg: Vec<i64>,
    at_barrier: Option<i64>,
    iter_end: Vec<i64>,
}

/// The hot path through a loop: from the header follow the fall-through side of conditionals that
/// stay in the loop, the exit-test is "not taken", until the back edge. Returns block indices.
pub fn hot_path(m: &MFunction, header: usize, latch: usize) -> Vec<usize> {
    let label_to_block: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut path = Vec::new();
    let mut cur = header;
    for _ in 0..m.blocks.len() * 2 {
        path.push(cur);
        if cur == latch {
            break;
        }
        let next = match &m.blocks[cur].term {
            Term::Goto(t) => label_to_block.get(t).copied(),
            Term::CondGoto { target, fall, .. } => {
                let ti = label_to_block.get(target).copied();
                let fi = label_to_block.get(fall).copied();
                // exit edges (target outside [header, latch]) are not taken
                let inside = |x: Option<usize>| x.map_or(false, |i| i >= header && i <= latch);
                if inside(fi) {
                    fi
                } else {
                    ti
                }
            }
            Term::Endpgm => None,
        };
        match next {
            Some(n) if n > cur || n == header => cur = n,
            _ => break,
        }
    }
    path
}

fn build_prog(m: &MFunction, path: &[usize], header: usize) -> Vec<SInst> {
    let label_to_block: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut prog = Vec::new();
    for (pi, &bi) in path.iter().enumerate() {
        let b = &m.blocks[bi];
        for inst in &b.insts {
            let e = effects(m, inst);
            let cost = cost(inst.op);
            let mut c = cost;
            if inst.op == MOp::SNop {
                c.issue = ((inst.imm + 1) * 4) as u32;
                c.occ = 0;
            }
            prog.push(SInst {
                cost: c,
                reads: e.reads.clone(),
                writes: e.writes.clone(),
                fence: e.fence,
                is_vmem: matches!(&e.mem, Some(a) if a.space == Space::Global),
                is_lds: matches!(&e.mem, Some(a) if a.space == Space::Lds),
                extra: 0,
            });
        }
        // terminator cost
        let next_in_path = path.get(pi + 1).copied().unwrap_or(header);
        let next_layout = bi + 1;
        let mut cost_t = 0i64;
        match &b.term {
            Term::Goto(t) => {
                let ti = label_to_block.get(t).copied().unwrap_or(usize::MAX);
                if !(m.elide_fallthrough && ti == next_layout) {
                    cost_t += BRANCH_TAKEN as i64;
                }
            }
            Term::CondGoto { target, fall, .. } => {
                let ti = label_to_block.get(target).copied().unwrap_or(usize::MAX);
                let fi = label_to_block.get(fall).copied().unwrap_or(usize::MAX);
                if ti == next_in_path && fi != next_in_path {
                    cost_t += BRANCH_TAKEN as i64;
                } else {
                    cost_t += BRANCH_NOT_TAKEN as i64;
                    if !(m.elide_fallthrough && fi == next_layout) {
                        cost_t += BRANCH_TAKEN as i64;
                    }
                }
            }
            Term::Endpgm => {}
        }
        if cost_t > 0 {
            prog.push(SInst {
                cost: OpCost { pipe: Pipe::Ctrl, issue: 1, occ: 0, latency: 0, grade: Grade::Experimental },
                reads: term_reads(&b.term),
                writes: vec![],
                fence: Fence::None,
                is_vmem: false,
                is_lds: false,
                extra: cost_t,
            });
        }
    }
    prog
}

/// Estimated steady-state cycles per iteration of one loop with `waves` waves on the SIMD.
pub fn simulate_prog(prog: &[SInst], waves: usize, iters: usize) -> f64 {
    if prog.is_empty() || waves == 0 {
        return 0.0;
    }
    let mut ws: Vec<Wave> = (0..waves)
        .map(|_| Wave {
            pc: 0,
            iter: 0,
            last_issue: -1,
            pipe_free: [0; NPIPES],
            ready: HashMap::new(),
            vm: Vec::new(),
            lg: Vec::new(),
            at_barrier: None,
            iter_end: Vec::new(),
        })
        .collect();
    let mut unit_free = [0i64; NPIPES];
    loop {
        // earliest feasible issue time per wave
        let mut best: Option<(i64, usize)> = None;
        for (wi, w) in ws.iter().enumerate() {
            if w.iter >= iters || w.at_barrier.is_some() {
                continue;
            }
            let ins = &prog[w.pc];
            let mut t = w.last_issue + 1;
            t = t.max(w.pipe_free[ins.cost.pipe.index()]);
            if ins.cost.occ > 0 {
                t = t.max(unit_free[ins.cost.pipe.index()]);
            }
            for s in &ins.reads {
                if let Some(r) = w.ready.get(s) {
                    t = t.max(*r);
                }
            }
            match ins.fence {
                Fence::WaitVm(n) => {
                    let n = n.max(0) as usize;
                    if w.vm.len() > n {
                        t = t.max(w.vm[w.vm.len() - 1 - n]);
                    }
                }
                Fence::WaitLgkm(n) => {
                    let n = n.max(0) as usize;
                    if w.lg.len() > n {
                        t = t.max(w.lg[w.lg.len() - 1 - n]);
                    }
                }
                _ => {}
            }
            if best.map_or(true, |b| t < b.0) {
                best = Some((t, wi));
            }
        }
        let Some((t, wi)) = best else {
            // everyone finished or at a barrier
            let waiting = ws.iter().filter(|w| w.at_barrier.is_some()).count();
            if waiting == 0 {
                break;
            }
            let release = ws.iter().filter_map(|w| w.at_barrier).max().unwrap() + BARRIER_COST as i64;
            for w in ws.iter_mut() {
                if w.at_barrier.is_some() {
                    w.at_barrier = None;
                    w.last_issue = release;
                    w.pc += 1;
                    if w.pc == prog.len() {
                        w.pc = 0;
                        w.iter += 1;
                        w.iter_end.push(release);
                    }
                }
            }
            continue;
        };
        let ins = prog[ws[wi].pc].clone();
        if ins.fence == Fence::Barrier {
            ws[wi].at_barrier = Some(t);
            continue;
        }
        let w = &mut ws[wi];
        w.last_issue = t + ins.extra;
        w.pipe_free[ins.cost.pipe.index()] = t + ins.cost.issue as i64;
        if ins.cost.occ > 0 {
            unit_free[ins.cost.pipe.index()] = t + ins.cost.occ as i64;
        }
        let comp = t + ins.cost.latency as i64;
        for s in &ins.writes {
            w.ready.insert(*s, comp);
        }
        if ins.is_vmem {
            let c = comp.max(w.vm.last().copied().unwrap_or(0));
            w.vm.retain(|x| *x > t);
            w.vm.push(c);
        }
        if ins.is_lds {
            let c = comp.max(w.lg.last().copied().unwrap_or(0));
            w.lg.retain(|x| *x > t);
            w.lg.push(c);
        }
        w.pc += 1;
        if w.pc == prog.len() {
            w.pc = 0;
            w.iter += 1;
            w.iter_end.push(w.last_issue + 1);
        }
    }
    // time at which the slowest wave finishes iteration k\n    let end_at = |k: usize| ws.iter().map(|w| w.iter_end[k]).max().unwrap_or(0);
    let k0 = (iters / 2).max(1).min(iters - 1);
    if iters < 4 {
        return end_at(iters - 1) as f64 / iters as f64;
    }
    (end_at(iters - 1) - end_at(k0 - 1)) as f64 / (iters - k0) as f64
}

#[derive(Debug, Clone)]
pub struct LoopReport {
    pub header_label: u32,
    pub blocks: usize,
    pub insts: usize,
    pub cycles_per_iter: Vec<(usize, f64)>,
}

/// Simulate every loop of `m` for the given waves-per-SIMD counts.
pub fn report(m: &MFunction, waves: &[usize]) -> Vec<LoopReport> {
    let label_to_block: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut out = Vec::new();
    for &(h, l) in &m.loops {
        let (Some(&hi), Some(&li)) = (label_to_block.get(&h), label_to_block.get(&l)) else { continue };
        let path = hot_path(m, hi, li);
        let prog = build_prog(m, &path, hi);
        let insts: usize = path.iter().map(|&b| m.blocks[b].insts.len()).sum();
        let cycles = waves.iter().map(|&w| (w, simulate_prog(&prog, w, 24))).collect();
        out.push(LoopReport { header_label: h, blocks: path.len(), insts, cycles_per_iter: cycles });
    }
    out
}