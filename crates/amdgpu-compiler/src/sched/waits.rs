//! Wait re-insertion: a scoreboard over the *final* instruction order, whole function.
//!
//! Replaces the `s_waitcnt` that lowering inserted (`MInst::auto`). Two in-flight queues:
//! * `vm`: every global load/store, in issue order (`vmcnt`, in-order completion);
//! * `lg`: every LDS op and scalar load (`lgkmcnt`; partial waits only while no SMEM is queued,
//!   because scalar loads may return out of order).
//! An entry carries `after` = the number of same-queue operations issued after it (minimum over
//! all control-flow paths at joins), so `s_waitcnt vmcnt(n)` with `n = after` of the newest
//! needed entry guarantees that entry (and every older one) completed. An instruction that reads
//! or writes a register with a load in flight (RAW / WAW) gets a wait first. Because the state
//! flows across blocks, loads issued before an `if` stay in flight across it (lowering used to
//! drain them at every control-flow edge).
//!
//! Counter limits: vmcnt 6 bits (63), lgkmcnt 4 bits (15): an entry with that many younger
//! operations has necessarily completed.

use std::collections::BTreeMap;

use super::effects::*;
use super::liveness::successors;
use crate::mir::*;

/// Test-only switch (AMDC_WAITS_SKIP_FENCE_OPERANDS=1): restore the old behaviour in which fence-class instructions (LDS-DMA loads) never wait for their register operands - used by the
/// regression probe probe_dma_voff_spill to prove that it fails without the fix.
fn skip_fence_operands() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("AMDC_WAITS_SKIP_FENCE_OPERANDS").map_or(false, |v| v == "1"))
}

/// Debug: only emit full drains (mcnt(0) / lgkmcnt(0)), never partial waits.
pub static DRAIN_ONLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const VM_MAX: u32 = 63;
const LG_MAX: u32 = 15;

#[derive(Clone, PartialEq, Eq, Debug)]
struct Ent {
    after: u32,
    slots: Vec<Slot>,
    smem: bool,
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct Queue {
    e: BTreeMap<u64, Ent>,
}

impl Queue {
    fn issue(&mut self, id: u64, slots: Vec<Slot>, smem: bool, max: u32) {
        for v in self.e.values_mut() {
            v.after += 1;
        }
        // scalar loads return out of order: never assume they completed because of the count
        self.e.retain(|_, v| v.after < max || v.smem);
        self.e.insert(id, Ent { after: 0, slots, smem });
    }

    /// Smallest `after` among the entries touching `slots`, if any.
    fn need(&self, slots: &[Slot]) -> Option<u32> {
        self.e
            .values()
            .filter(|v| v.slots.iter().any(|s| slots.contains(s)))
            .map(|v| v.after)
            .min()
    }

    fn complete(&mut self, n: u32) {
        self.e.retain(|_, v| v.after < n);
    }

    fn merge(&mut self, o: &Queue) -> bool {
        let mut changed = false;
        for (k, v) in &o.e {
            match self.e.get_mut(k) {
                Some(x) => {
                    if v.after < x.after {
                        x.after = v.after;
                        changed = true;
                    }
                }
                None => {
                    self.e.insert(*k, v.clone());
                    changed = true;
                }
            }
        }
        changed
    }
}

#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct State {
    vm: Queue,
    lg: Queue,
}

fn step(st: &mut State, e: &Effects, inst: &MInst, id: u64, out: &mut Option<&mut Vec<MInst>>) -> usize {
    let mut inserted = 0;
    match e.fence {
        Fence::WaitVm(n) => {
            st.vm.complete(n.max(0) as u32);
            if let Some(o) = out {
                o.push(inst.clone());
            }
            return 0;
        }
        Fence::WaitLgkm(n) => {
            let any_smem = st.lg.e.values().any(|v| v.smem);
            if n <= 0 {
                st.lg.e.clear();
            } else if !any_smem {
                st.lg.complete(n as u32);
            }
            if let Some(o) = out {
                o.push(inst.clone());
            }
            return 0;
        }
        // Barrier-class instructions (s_barrier, s_setprio, s_nop, the LDS-DMA loads, cache maintenance) are ordering fences for the list scheduler, but the ones that read
        // registers (a DMA's voffset / soffset) still need their operands to have arrived: they fall through to the operand check below.
        Fence::Barrier | Fence::SetPrio | Fence::Nop if e.reads.is_empty() || skip_fence_operands() => {
            if let Some(o) = out {
                o.push(inst.clone());
            }
            return 0;
        }
        Fence::Barrier | Fence::SetPrio | Fence::Nop | Fence::None => {}
    }
    let mut touched: Vec<Slot> = e.reads.clone();
    for s in &e.writes {
        if !touched.contains(s) {
            touched.push(*s);
        }
    }
    if let Some(n) = st.vm.need(&touched) {
        let n = if DRAIN_ONLY.load(std::sync::atomic::Ordering::Relaxed) { 0 } else { n };
        if let Some(o) = out {
            o.push(MInst::new(MOp::SWaitVm).imm(n as i64).auto());
        }
        st.vm.complete(n);
        inserted += 1;
    }
    if let Some(n) = st.lg.need(&touched) {
        let any_smem = st.lg.e.values().any(|v| v.smem);
        let n = if any_smem || DRAIN_ONLY.load(std::sync::atomic::Ordering::Relaxed) { 0 } else { n };
        if let Some(o) = out {
            o.push(MInst::new(MOp::SWaitLgkm).imm(n as i64).auto());
        }
        st.lg.complete(n);
        inserted += 1;
    }
    if e.fence != Fence::None {
        // (not tracked in the queues: a DMA writes no register, and counting it would only loosen the partial waits of the shipped kernels)
        if let Some(o) = out {
            o.push(inst.clone());
        }
        return inserted;
    }
    if let Some(a) = &e.mem {
        match a.space {
            Space::Global => st.vm.issue(id, e.writes.clone(), false, VM_MAX),
            Space::Lds => st.lg.issue(id, e.writes.clone(), false, LG_MAX),
        }
    } else if e.smem {
        st.lg.issue(id, e.writes.clone(), true, LG_MAX);
    }
    if let Some(o) = out {
        o.push(inst.clone());
    }
    inserted
}

fn run_block(m: &MFunction, b: usize, st: &mut State, mut out: Option<&mut Vec<MInst>>) -> usize {
    let mut n = 0;
    for (ii, inst) in m.blocks[b].insts.iter().enumerate() {
        let e = effects(m, inst);
        n += step(st, &e, inst, ((b as u64) << 32) | ii as u64, &mut out);
    }
    n
}

/// Insert the waits (`auto = true`) into every reachable block. Returns the number inserted.
pub fn insert_waits(m: &mut MFunction) -> usize {
    let nb = m.blocks.len();
    let succs = successors(m);
    let mut in_st: Vec<Option<State>> = vec![None; nb];
    if nb == 0 {
        return 0;
    }
    in_st[0] = Some(State::default());
    let mut guard = 0;
    loop {
        guard += 1;
        let mut changed = false;
        for b in 0..nb {
            let Some(s) = in_st[b].clone() else { continue };
            let mut st = s;
            run_block(m, b, &mut st, None);
            for &t in &succs[b] {
                match &mut in_st[t] {
                    None => {
                        in_st[t] = Some(st.clone());
                        changed = true;
                    }
                    Some(x) => {
                        if x.vm.merge(&st.vm) | x.lg.merge(&st.lg) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed || guard > 1000 {
            break;
        }
    }
    let mut total = 0;
    for b in 0..nb {
        let Some(s) = in_st[b].clone() else { continue };
        let mut st = s;
        let mut out: Vec<MInst> = Vec::with_capacity(m.blocks[b].insts.len() + 16);
        total += run_block(m, b, &mut st, Some(&mut out));
        m.blocks[b].insts = out;
    }
    total
}

/// Remove the waits inserted by lowering (or by an earlier `insert_waits`).
pub fn strip_auto_waits(m: &mut MFunction) -> usize {
    let mut n = 0;
    for b in &mut m.blocks {
        let before = b.insts.len();
        b.insts.retain(|i| !(i.auto && matches!(i.op, MOp::SWaitVm | MOp::SWaitLgkm)));
        n += before - b.insts.len();
    }
    n
}

/// Independent check of the final order: re-simulate the scoreboard *with the waits present* and
/// confirm that no instruction reads or writes a register that still has a load in flight.
pub fn verify_waits(m: &MFunction) -> Result<(), String> {
    let nb = m.blocks.len();
    let succs = successors(m);
    let mut in_st: Vec<Option<State>> = vec![None; nb];
    if nb == 0 {
        return Ok(());
    }
    in_st[0] = Some(State::default());
    let mut guard = 0;
    loop {
        guard += 1;
        let mut changed = false;
        for b in 0..nb {
            let Some(s) = in_st[b].clone() else { continue };
            let mut st = s;
            for (ii, inst) in m.blocks[b].insts.iter().enumerate() {
                let e = effects(m, inst);
                let mut touched: Vec<Slot> = e.reads.clone();
                touched.extend(e.writes.iter().copied());
                if !matches!(e.fence, Fence::WaitVm(_) | Fence::WaitLgkm(_)) && !(skip_fence_operands() && e.fence != Fence::None) {
                    if st.vm.need(&touched).is_some() || st.lg.need(&touched).is_some() {
                        return Err(format!("block {b} inst {ii} ({:?}) touches a register with a load in flight", inst.op));
                    }
                }
                step(&mut st, &e, inst, ((b as u64) << 32) | ii as u64, &mut None);
            }
            for &t in &succs[b] {
                match &mut in_st[t] {
                    None => {
                        in_st[t] = Some(st.clone());
                        changed = true;
                    }
                    Some(x) => {
                        if x.vm.merge(&st.vm) | x.lg.merge(&st.lg) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed || guard > 1000 {
            break;
        }
    }
    Ok(())
}