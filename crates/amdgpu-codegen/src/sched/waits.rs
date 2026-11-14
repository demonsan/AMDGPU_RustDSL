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


// ------------------------------------------------------------------------------------------------ W3 B5: ring slots as memory tokens
//
// An LDS-DMA writes LDS, not a register, so neither of its two dependences is visible to the register scoreboard:
//   RAW  the DMA of a ring slot must complete (vmcnt) before the first LDS read of that slot;
//   WAR  every LDS read of a slot must complete (lgkmcnt) before a DMA that overwrites the slot is issued.
// Today's kernels state both with explicit waits in program order. Once B5 moves a DMA across iterations (a producer unit with
// `iter != 0` in a Free phase), the plan's rings carry them instead: a producer unit of ring R with iteration offset d writes the
// slot (t + d) mod S, a consumer unit with offset c reads (t + c) mod S, so the two touch the same slot iff d = c (mod S). The
// scoreboard tracks one token per (ring, relative slot); at a loop back edge the tokens shift by the iterations one trip advances.
// Active only while a plan with such a phase is installed (`with_ring_model`), so every other kernel keeps its waits.

/// The rings of a plan as seen by the wait pass.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RingModel {
    /// unit tag (`MInst::unit`) -> (ring, slots, iteration offset) for the DMA producers of a ring
    pub producers: std::collections::HashMap<u32, Vec<(usize, u32, i64)>>,
    /// unit tag -> (ring, slots, iteration offset) for the units reading a ring's slots
    pub consumers: std::collections::HashMap<u32, Vec<(usize, u32, i64)>>,
    /// slots of each ring
    pub slots: Vec<u32>,
    /// iterations one trip of the loop body advances (the token shift at a back edge)
    pub body_iters: i64,
    /// unit tag -> the LDS tiles the unit's tile markers produce / consume (from the kernel's `region Produce / Consume tile`)
    pub tiles_prod: std::collections::HashMap<u32, Vec<String>>,
    pub tiles_cons: std::collections::HashMap<u32, Vec<String>>,
    /// declared tile aliases (`lds_alias`): such tiles share bytes
    pub aliases: Vec<(String, String)>,
}

impl RingModel {
    /// The model of `plan`, when it has a non-authored phase holding a DMA unit with a non-zero iteration offset (B5 active).
    pub fn of(plan: &amdgpu_dsl::plan::KernelPlan) -> Option<RingModel> {
        use amdgpu_dsl::plan::{OrderSpec, UnitClass};
        let free = |ph: u32| plan.phases.iter().any(|p| p.id == ph && !matches!(p.order, OrderSpec::Authored));
        if !plan.units.iter().any(|u| u.class == UnitClass::Dma && u.iter != 0 && free(u.phase)) || plan.rings.is_empty() {
            return None;
        }
        let mut m = RingModel { slots: plan.rings.iter().map(|r| r.slots.max(1)).collect(), ..Default::default() };
        let mut max_c = 0i64;
        let min_p = plan.units.iter().filter(|u| u.class == UnitClass::Dma && u.iter != 0).map(|u| u.iter).min().unwrap_or(i32::MAX);
        for (ri, r) in plan.rings.iter().enumerate() {
            for u in &plan.units {
                let e = (ri, r.slots.max(1), u.iter as i64);
                if u.class == UnitClass::Dma && u.phase == r.producer_phase {
                    m.producers.entry(u.id + 1).or_default().push(e);
                } else if u.phase == r.consumer_phase {
                    m.consumers.entry(u.id + 1).or_default().push(e);
                    // the iterations a trip covers: the reading units' offsets (a DMA's setup units carry the DMA's offset)
                    if u.iter < min_p {
                        max_c = max_c.max(u.iter as i64);
                    }
                }
            }
        }
        // a moved DMA unit no ring names as its producer (the ring's producer phase is another copy of the body): it is taken as a
        // producer of every ring its own phase consumes (conservative: more slot tokens, never fewer)
        for u in &plan.units {
            if u.class == UnitClass::Dma && u.iter != 0 && !m.producers.contains_key(&(u.id + 1)) {
                let rings: Vec<usize> = plan.rings.iter().enumerate().filter(|(_, r)| r.consumer_phase == u.phase).map(|(i, _)| i).collect();
                for ri in rings {
                    m.producers.entry(u.id + 1).or_default().push((ri, plan.rings[ri].slots.max(1), u.iter as i64));
                }
            }
        }
        m.body_iters = max_c + 1;
        Some(m)
    }
}

impl RingModel {
    /// The model of a function: the plan's rings, with each unit's ring membership taken from the tile markers inside it (exact per unit;
    /// the plan's producer / consumer phases cannot name every copy of a body). Falls back to the plan-only rules for rings no marker names.
    pub fn of_function(f: &amdgpu_dsl::Function) -> Option<RingModel> {
        use amdgpu_dsl::plan::UnitClass;
        use amdgpu_dsl::Stmt;
        let plan = f.plan.as_ref()?;
        let mut m = RingModel::of(plan)?;
        let mut stack: Vec<u32> = Vec::new();
        for s in f.stmts() {
            match s {
                Stmt::UnitMark { id, begin: true } => stack.push(*id + 1),
                Stmt::UnitMark { begin: false, .. } => {
                    stack.pop();
                }
                Stmt::TileProduce { tile, .. } | Stmt::TileConsume { tile, .. } => {
                    if let Some(&u) = stack.last() {
                        let map = if matches!(s, Stmt::TileProduce { .. }) { &mut m.tiles_prod } else { &mut m.tiles_cons };
                        let e = map.entry(u).or_default();
                        if !e.contains(tile) {
                            e.push(tile.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        m.aliases = f.contract.tile_aliases.clone();
        let iter_of = |tag: u32| plan.units.iter().find(|u| u.id + 1 == tag).map(|u| u.iter as i64).unwrap_or(0);
        let is_dma = |tag: u32| plan.units.iter().find(|u| u.id + 1 == tag).map_or(false, |u| u.class == UnitClass::Dma);
        for (ri, r) in plan.rings.iter().enumerate() {
            let s = r.slots.max(1);
            let prods: Vec<u32> = m.tiles_prod.iter().filter(|(t, v)| is_dma(**t) && v.contains(&r.name)).map(|(t, _)| *t).collect();
            let cons: Vec<u32> = m.tiles_cons.iter().filter(|(_, v)| v.contains(&r.name)).map(|(t, _)| *t).collect();
            if prods.is_empty() && cons.is_empty() {
                continue;
            }
            // the markers name this ring: they replace the phase-derived membership
            for v in m.producers.values_mut() {
                v.retain(|e| e.0 != ri);
            }
            for v in m.consumers.values_mut() {
                v.retain(|e| e.0 != ri);
            }
            for t in prods {
                m.producers.entry(t).or_default().push((ri, s, iter_of(t)));
            }
            for t in cons {
                m.consumers.entry(t).or_default().push((ri, s, iter_of(t)));
            }
        }
        m.producers.retain(|_, v| !v.is_empty());
        m.consumers.retain(|_, v| !v.is_empty());
        Some(m)
    }

    fn tiles_meet(&self, a: &[String], b: &[String]) -> bool {
        a.iter().any(|x| b.iter().any(|y| x == y || self.aliases.iter().any(|(p, q)| (p == x && q == y) || (p == y && q == x))))
    }
}

thread_local! {
    static RINGS: std::cell::RefCell<Option<RingModel>> = const { std::cell::RefCell::new(None) };
}

/// Runs `f` with `model` installed for the wait pass and its verifier on this thread.
pub fn with_ring_model<R>(model: Option<RingModel>, f: impl FnOnce() -> R) -> R {
    let old = RINGS.with(|c| c.replace(model));
    let r = f();
    RINGS.with(|c| *c.borrow_mut() = old);
    r
}

const TOK_BASE: u32 = u32::MAX - 8192;

/// kind 0 = a DMA write of the slot, 1 = an LDS read of the slot
fn tok(kind: u32, ring: usize, rel: i64, slots: u32) -> Slot {
    TOK_BASE + kind * 4096 + ring as u32 * 64 + rel.rem_euclid(slots as i64) as u32
}

fn tok_parts(s: Slot) -> Option<(u32, usize, u32)> {
    if (TOK_BASE..TOK_BASE + 8192).contains(&s) {
        let x = s - TOK_BASE;
        Some((x / 4096, ((x % 4096) / 64) as usize, x % 64))
    } else {
        None
    }
}

/// (tokens whose vm entries must complete first, tokens whose lg entries must complete first, tokens this instruction issues)
fn ring_tokens(e: &Effects, inst: &MInst) -> (Vec<Slot>, Vec<Slot>, Vec<Slot>) {
    RINGS.with(|c| {
        let c = c.borrow();
        let Some(rm) = c.as_ref() else { return (vec![], vec![], vec![]) };
        ring_tokens_of(rm, e, inst)
    })
}

/// A ring DMA and an LDS read of a ring that touch different slots (or different rings): the plan's rings are disjoint LDS
/// allocations and the slot tokens say which slot each access touches, so the two need no order (the wait pass keeps the
/// completion dependences of the slots that do meet). False whenever either access is not a ring access.
pub fn ring_independent(rm: &RingModel, ea: &Effects, a: &MInst, eb: &Effects, b: &MInst) -> bool {
    // a DMA of one tile and an LDS read of other tiles (the units' markers name both sides; declared tiles are disjoint unless aliased)
    let is_dma = |e: &Effects| e.fence == Fence::Barrier && e.mem.is_some();
    let is_read = |e: &Effects| e.fence == Fence::None && matches!(&e.mem, Some(m) if m.space == Space::Lds && !m.write);
    for (d, de, r, re) in [(a, ea, b, eb), (b, eb, a, ea)] {
        if is_dma(de) && is_read(re) {
            if let (Some(p), Some(c)) = (rm.tiles_prod.get(&d.unit), rm.tiles_cons.get(&r.unit)) {
                if !p.is_empty() && !c.is_empty() && !rm.tiles_meet(p, c) {
                    return true;
                }
            }
        }
    }
    let (ra, la, ia) = ring_tokens_of(rm, ea, a);
    let (rb, lb, ib) = ring_tokens_of(rm, eb, b);
    // (DMA, read): the DMA issues kind-0 tokens, the read needs kind-0 tokens (raw); (read, DMA): the read issues kind-1, the DMA needs kind-1 (war)
    let dma_read = !ia.is_empty() && !la.is_empty() && !rb.is_empty() && ra.is_empty();
    let read_dma = !ib.is_empty() && !lb.is_empty() && !ra.is_empty() && rb.is_empty();
    if dma_read {
        return !ia.iter().any(|x| rb.contains(x));
    }
    if read_dma {
        return !ia.iter().any(|x| lb.contains(x));
    }
    false
}

fn ring_tokens_of(rm: &RingModel, e: &Effects, inst: &MInst) -> (Vec<Slot>, Vec<Slot>, Vec<Slot>) {
    {
        let dma = e.fence == Fence::Barrier && e.mem.is_some();
        let lds_read = e.fence == Fence::None && matches!(&e.mem, Some(a) if a.space == Space::Lds && !a.write);
        if dma {
            if let Some(ps) = rm.producers.get(&inst.unit) {
                let war = ps.iter().map(|&(r, s, d)| tok(1, r, d, s)).collect();
                let iss = ps.iter().map(|&(r, s, d)| tok(0, r, d, s)).collect();
                return (vec![], war, iss);
            }
        } else if lds_read {
            if let Some(cs) = rm.consumers.get(&inst.unit) {
                let raw = cs.iter().map(|&(r, s, d)| tok(0, r, d, s)).collect();
                let iss = cs.iter().map(|&(r, s, d)| tok(1, r, d, s)).collect();
                return (raw, vec![], iss);
            }
        }
        (vec![], vec![], vec![])
    }
}

/// The state carried over a loop back edge: ring tokens move back by the iterations one trip advanced.
fn shift_tokens(st: &State) -> State {
    RINGS.with(|c| {
        let c = c.borrow();
        let Some(rm) = c.as_ref() else { return st.clone() };
        let mut out = st.clone();
        for q in [&mut out.vm, &mut out.lg] {
            let old = std::mem::take(&mut q.e);
            for (key, mut v) in old {
                let mut rel_key = None;
                for s in v.slots.iter_mut() {
                    if let Some((k, r, rel)) = tok_parts(*s) {
                        let n = rm.slots.get(r).copied().unwrap_or(1);
                        *s = tok(k, r, rel as i64 - rm.body_iters, n);
                        rel_key = tok_parts(*s).map(|x| x.2);
                    }
                }
                // an access issued on an earlier trip is a different entry from the same instruction's next issue: key it by its age
                let key = match rel_key {
                    Some(rel) => ring_key(key, rel),
                    None => key,
                };
                q.e.insert(key, v);
            }
        }
        out
    })
}

/// Queue key of a ring access: the instruction id plus its current relative slot (top bits).
fn ring_key(id: u64, rel: u32) -> u64 {
    (id & ((1u64 << 48) - 1)) | (1u64 << 63) | ((rel as u64) << 48)
}

fn is_back_edge(m: &MFunction, from: usize, to: usize) -> bool {
    let (fl, tl) = (m.blocks[from].label, m.blocks[to].label);
    m.loops.iter().any(|&(h, l)| h == tl && l == fl)
}

const VM_MAX: u32 = crate::target::T.counters.of(amdgpu_target::MemClass::VmemLoad).max;
const LG_MAX: u32 = crate::target::T.counters.of(amdgpu_target::MemClass::Lds).max;

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
    let (ring_vm, ring_lg, ring_issue) = ring_tokens(e, inst);
    if !ring_vm.is_empty() || !ring_lg.is_empty() || !ring_issue.is_empty() {
        if let Some(n) = st.vm.need(&ring_vm) {
            if let Some(o) = out {
                o.push(MInst::new(MOp::SWaitVm).imm(n as i64).auto());
            }
            st.vm.complete(n);
            inserted += 1;
        }
        if let Some(n) = st.lg.need(&ring_lg) {
            let n = if st.lg.e.values().any(|v| v.smem) { 0 } else { n };
            if let Some(o) = out {
                o.push(MInst::new(MOp::SWaitLgkm).imm(n as i64).auto());
            }
            st.lg.complete(n);
            inserted += 1;
        }
        if e.fence == Fence::Barrier {
            // a ring DMA: operand waits as usual, then it enters the vm queue with its slot token
            let touched: Vec<Slot> = e.reads.clone();
            if let Some(n) = st.vm.need(&touched) {
                if let Some(o) = out {
                    o.push(MInst::new(MOp::SWaitVm).imm(n as i64).auto());
                }
                st.vm.complete(n);
                inserted += 1;
            }
            if let Some(n) = st.lg.need(&touched) {
                let n = if st.lg.e.values().any(|v| v.smem) { 0 } else { n };
                if let Some(o) = out {
                    o.push(MInst::new(MOp::SWaitLgkm).imm(n as i64).auto());
                }
                st.lg.complete(n);
                inserted += 1;
            }
            let key = ring_issue.first().and_then(|s| tok_parts(*s)).map_or(id, |x| ring_key(id, x.2));
            st.vm.issue(key, ring_issue, false, VM_MAX);
            if let Some(o) = out {
                o.push(inst.clone());
            }
            return inserted;
        }
    }
    match e.fence {
        Fence::WaitBoth(v, n) => {
            st.vm.complete(v.max(0) as u32);
            let any_smem = st.lg.e.values().any(|x| x.smem);
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
        // (a fenced SMEM op, the timing stamp `s_memtime`, is the exception: its result must still be tracked, so it falls through)
        Fence::Barrier | Fence::SetPrio | Fence::Nop if e.reads.is_empty() && !e.smem => {
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
    // a global load overwriting a register an older global load still writes needs no vmcnt wait: loads complete in issue order, so the
    // younger data lands last (the physical-register check of the default path, `with_in_order_vm_waw`; the value-level pass never sees
    // two values in one register)
    let vm_touched: Vec<Slot> = if IN_ORDER_VM_WAW.with(|c| c.get()) && matches!(&e.mem, Some(a) if a.space == Space::Global) && !e.writes.is_empty() {
        let mut v: Vec<Slot> = e.reads.clone();
        v.retain(|s| !e.writes.contains(s) || e.reads.contains(s));
        v
    } else {
        touched.clone()
    };
    if let Some(n) = st.vm.need(&vm_touched) {
        if let Some(o) = out {
            o.push(MInst::new(MOp::SWaitVm).imm(n as i64).auto());
        }
        st.vm.complete(n);
        inserted += 1;
    }
    if let Some(n) = st.lg.need(&touched) {
        let any_smem = st.lg.e.values().any(|v| v.smem);
        let n = if any_smem { 0 } else { n };
        if let Some(o) = out {
            o.push(MInst::new(MOp::SWaitLgkm).imm(n as i64).auto());
        }
        st.lg.complete(n);
        inserted += 1;
    }
    if e.fence != Fence::None {
        // (not tracked in the queues: a DMA writes no register, and counting it would only loosen the partial waits of the shipped kernels)
        if e.smem {
            st.lg.issue(id, e.writes.clone(), true, LG_MAX); // the s_memtime stamp: an SMEM result that the next reader waits for
        }
        if let Some(o) = out {
            o.push(inst.clone());
        }
        return inserted;
    }
    if let Some(a) = &e.mem {
        let key = ring_issue.first().and_then(|s| tok_parts(*s)).map_or(id, |x| ring_key(id, x.2));
        let mut slots = e.writes.clone();
        slots.extend(ring_issue);
        match a.space {
            Space::Global => st.vm.issue(key, slots, false, VM_MAX),
            Space::Lds => st.lg.issue(key, slots, false, LG_MAX),
        }
    } else if e.smem {
        st.lg.issue(id, e.writes.clone(), true, LG_MAX);
    }
    if let Some(o) = out {
        o.push(inst.clone());
    }
    inserted
}

thread_local! {
    static IN_ORDER_VM_WAW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with global-load WAW against older global loads exempt (in-order completion); see `insert_waits_physical`.
pub fn with_in_order_vm_waw<R>(f: impl FnOnce() -> R) -> R {
    IN_ORDER_VM_WAW.with(|c| c.set(true));
    let r = f();
    IN_ORDER_VM_WAW.with(|c| c.set(false));
    r
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
                let st = if is_back_edge(m, b, t) { shift_tokens(&st) } else { st.clone() };
                match &mut in_st[t] {
                    None => {
                        in_st[t] = Some(st);
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
                if !matches!(e.fence, Fence::WaitVm(_) | Fence::WaitLgkm(_) | Fence::WaitBoth(..)) {
                    if st.vm.need(&touched).is_some() || st.lg.need(&touched).is_some() {
                        return Err(format!("block {b} inst {ii} ({:?}) touches a register with a load in flight", inst.op));
                    }
                    let (rv, rl, _) = ring_tokens(&e, inst);
                    if st.vm.need(&rv).is_some() {
                        return Err(format!("block {b} inst {ii} ({:?}) reads a ring slot whose DMA is still in flight", inst.op));
                    }
                    if st.lg.need(&rl).is_some() {
                        return Err(format!("block {b} inst {ii} ({:?}) overwrites a ring slot whose LDS reads are still in flight", inst.op));
                    }
                }
                step(&mut st, &e, inst, ((b as u64) << 32) | ii as u64, &mut None);
            }
            for &t in &succs[b] {
                let st = if is_back_edge(m, b, t) { shift_tokens(&st) } else { st.clone() };
                match &mut in_st[t] {
                    None => {
                        in_st[t] = Some(st);
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