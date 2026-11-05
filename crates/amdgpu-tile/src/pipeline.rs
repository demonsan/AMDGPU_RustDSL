//! L3b: the tile pipeline driver. The cadence (who stages which tile when, the waits, the barriers, the phase offset of the two wave groups, the peeled
//! last tile, the tile-counter contract) is data (`Cadence`); the kernel supplies only the work of each phase (`PipeBody`). Every (slot, generation) a body
//! sees comes from `TileIt`, so the protocol markers are consistent by construction.
//!
//! `Cadence::TwoPhaseOffset` (the `_vs128` / `_kvsplit` structure, kernels.rs `emit_fmha_vs128_role`):
//! ```text
//! prologue: stage(0) wait_vm(0) | stores | wait_vm(0) wait_lds(0) barrier | [B: barrier] | compute(0) wait_lds(0) barrier
//! loop t:   stage(t+1) | v_phase(t) | wait_vm(0) | wait_lds(0) barrier | m_phase(t) | wait_lds(0) barrier
//! tail:     if partial { iter(masked) } else { iter }  (no stage; the last barrier only in group A: equal barrier counts in both groups)
//! ```
//! `Cadence::OneWindow` (the `_t128` structure, `emit_fmha_t128_role`): a 3-slot ring, one group, one barrier per tile:
//! ```text
//! prologue: stage(0) wait_vm(0) | stage(1) wait_vm(0) | stores | wait_vm(0) wait_lds(0) barrier | compute(0) wait_lds(0)
//! loop t:   stage(t+2) | window(t) | wait_vm(0) wait_lds(0) barrier
//! tail:     if partial { window(masked) } else { window }   (no stage, no barrier)
//! ```

use crate::lane::Role;
use crate::shared::SlotIdx;
use amdgpu_dsl::{ev, varx};
use amdgpu_dsl::{DType, Ext, FnBuilder, Opnd, ValueId, VarId, I32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cadence {
    TwoPhaseOffset,
    OneWindow,
    /// 2-slot ring with the operands staged through REGISTERS (the aiter 4-wave GEMM): the body keeps tile t in registers (loop state) and, while
    /// computing it, reloads them in place from the slot of tile t+1; one wait + barrier per tile publishes t+1 and frees the slot of t, which then
    /// receives tile t+2 (one tile of DMA latency hidden behind the compute, the LDS reads of t+1 behind the MMAs of t):
    /// ```text
    /// prologue: stage(0) | stage(1) | stores | wait_vm(0) wait_lds(0) barrier | compute(0) = load tile 0 into registers
    /// loop t:   wait_vm(0) wait_lds(0) barrier | stage(t+2) -> slot(t) | m_phase(t): MMAs of t + in-place loads of t+1 (`t.next`)
    /// tail:     m_phase (no loads: `staged` = false)
    /// ```
    RegStaged,
    /// 2-slot ring, one group, ONE barrier per tile (prefetch distance 1, for tiles whose 3 slots do not fit the LDS):
    /// ```text
    /// prologue: stage(0) wait_vm(0) | stores | wait_vm(0) wait_lds(0) barrier | compute(0) wait_lds(0)
    /// loop t:   stage(t+1) | window(t) | wait_vm(0) wait_lds(0) barrier
    /// ```
    /// The barrier both publishes tile t+1 and retires the reads of tile t before stage(t+2) overwrites its slot.
    TwoSlot,
}

/// One iteration of the pipeline as the body sees it.
#[derive(Clone, Debug)]
pub struct TileIt {
    /// tile counter (0 in the prologue, the loop variable inside the loop and in the tail)
    pub it: Opnd,
    /// the tile being computed: slot `idx`, generation `it`
    pub cur: SlotIdx,
    /// tile t+1: slot / generation (`_vs128`: staged in this iteration; `_t128`: staged in the previous one)
    pub next: SlotIdx,
    /// tile t+2 (`OneWindow` only)
    pub n2: Option<SlotIdx>,
    /// the masked copy of the last (partial) tile
    pub masked: bool,
    /// a tile is staged in this iteration (false in the peeled tail)
    pub staged: bool,
    /// rows per tile
    pub tile_rows: i32,
}

impl TileIt {
    /// First row (key) of the current tile: `it * tile_rows`.
    pub fn key0(&self) -> I32 {
        amdgpu_dsl::ox(self.it) << (self.tile_rows.trailing_zeros() as i32)
    }
}

/// What a kernel supplies to the pipeline. `role` is the compile-time group tag of the branch being emitted.
pub trait PipeBody {
    /// Issue the copies of tile `key0` (first row) into `slot` for the stages this role produces (e.g. `RowsCopy::issue` on `buf.at(slot)`).
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx);
    /// Called right after the pipeline's `wait_vm(0)` that completes the stage of `slot` (e.g. the VGPR path of the K scales: `ds_write` them now).
    fn stage_complete(&mut self, _b: &mut FnBuilder, _role: Role, _slot: &SlotIdx) {}
    /// `Pipeline::late_stage_b`: the stage of tile t+1 handed to the M phase of tile t (issued after the mid barrier, completed before the end
    /// barrier); the default issues it at once.
    fn defer_stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
        self.stage(b, role, key0, slot)
    }
    /// Constant LDS data written once before the first barrier (Q staging, ones block).
    fn prologue_stores(&mut self, _b: &mut FnBuilder, _role: Role) {}
    /// Work on tile 0 before the loop (e.g. QK(0)); its results usually become loop-state initial values.
    fn prologue_compute(&mut self, b: &mut FnBuilder, role: Role, t0: &TileIt);
    /// Initial values of the body's loop-carried state (called after `prologue_compute`).
    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)>;
    /// The loop variables of that state, in the same order (called once, right after `begin_loop`).
    fn bind_state(&mut self, vars: &[VarId]);
    /// `TwoPhaseOffset`: the VALU phase of tile t (between the stage and the first barrier).
    fn v_phase(&mut self, _b: &mut FnBuilder, _role: Role, _t: &TileIt) {}
    /// `TwoPhaseOffset`: the matrix phase (between the two barriers).
    fn m_phase(&mut self, _b: &mut FnBuilder, _role: Role, _t: &TileIt) {}
    /// `OneWindow`: the whole window of tile t.
    fn window(&mut self, _b: &mut FnBuilder, _role: Role, _t: &TileIt) {}
    /// `RegStaged`: register-only work of tile t issued BEFORE the iteration's wait + barrier (e.g. the first MMAs, which then run while the wave
    /// waits for the DMA / the other waves); it must not touch the LDS slots.
    fn pre_sync(&mut self, _b: &mut FnBuilder, _role: Role, _t: &TileIt) {}
    /// `RegStaged`: between the top-of-tile waits and the barrier (work that touches no LDS / VMEM, e.g. MMAs of the register tile)
    fn post_wait(&mut self, _b: &mut FnBuilder, _role: Role, _t: &TileIt) {}
}

/// The pipeline over `nkv` tiles of `tile_rows` rows (the last one possibly partial: `kv_len` rows in total).
pub struct Pipeline {
    pub cadence: Cadence,
    /// number of tiles (>= 1), wave-uniform
    pub nkv: ValueId,
    /// valid rows in total (the last tile is partial when `kv_len < nkv * tile_rows`)
    pub kv_len: ValueId,
    pub tile_rows: i32,
    /// contract upper bound of the tile counter (e.g. `ceil(SK / 128)`)
    pub nkv_bound: Ext,
    /// loop-variable order: [it, state.., idx] (the manual kernels' order; DEFAULT, measured +2.3% per-CU on the _vs128 port vs [it, idx, state..]) or,
    /// when false, [it, idx, state..]; a register-allocation knob only
    pub ring_var_last: bool,
    /// ring slot strides whose `slot * stride` offsets (cur, next, n2) are bound at the TOP of every iteration, like the manual `ring()` (otherwise the
    /// first consumer binds them, possibly late in the window: `_t128` measured 4 `v_mul_lo_u32` at the window head). E.g. `vec![32768, 512]`.
    pub prebind: Vec<i64>,
    /// `OneWindow`: vector-memory operations allowed in flight at the end of a window (`s_waitcnt vmcnt(n)` instead of 0): the number of VMEM ops ONE
    /// stage issues per wave, so that the wait completes tile t+1 (published by the barrier) while tile t+2, issued at this window's start, keeps
    /// loading across the next window. Only valid when the body issues no other VMEM in the window and does not use `stage_complete` (the
    /// verifier checks the completion with the counted-vmcnt rule).
    pub vm_inflight: u8,
    /// `TwoPhaseOffset`: drain this wave's LDS reads (`s_waitcnt lgkmcnt(0)`) before the mid-tile barrier (default). Off when the V phase issues
    /// read-ahead `ds_read`s for the M phase (the vendor keeps them in flight across the barrier; the V phase writes no LDS, its DMA is
    /// completed by the `wait_vm(0)`).
    pub mid_lds_drain: bool,
    /// `TwoPhaseOffset`: group B stages tile t+1 inside its M phase of tile t (`PipeBody::defer_stage`) instead of before the V phase: the slot it
    /// writes (tile t-1's) is free after the end barrier of t-1, and the stage completes (`wait_vm(0)`) before the end barrier of t, i.e. before
    /// group A, one phase ahead, reads it in its M phase of t+1.
    pub late_stage_b: bool,
    /// `RegStaged`: two tiles per loop pass with STATIC ring slots (0 then 1: the tile counter is even at the top), so the slot offsets are
    /// constants (the vendor `u2`); the loop may exit between the two halves.
    pub unroll2: bool,
    /// `RegStaged`: the vmcnt allowed at the top-of-tile wait (default 0). A body that publishes part of the stage later (its own wait + barrier
    /// before the first read of that part) sets this to the number of DMAs issued after the part needed first.
    pub top_vm: u8,
    /// `RegStaged`: the DMAs of tile 1 (issued after tile 0) that may still be in flight at the prologue barrier (default 0)
    pub prologue_vm: u8,
    /// `RegStaged`: the body issued tiles 0 and 1 itself before `run` (as early as their addresses exist); the prologue skips those stages
    pub pre_staged: bool,
    /// DIAGNOSTIC (wrong results unless the body has no LDS traffic): no top-of-tile wait + barrier in RegStaged
    pub ablate_no_sync: bool,
    /// loop-exit bound through read_first_lane (scalar compare)
    pub scalar_exit: bool,
    /// `RegStaged`: tile 1 staged after `prologue_stores`; the prologue wait then leaves `late_tile1_vm` (its DMAs) in flight
    pub late_tile1: bool,
    pub late_tile1_vm: u8,
    /// with `late_tile1`: issue tile 1 after the prologue barrier and the tile-0 reads
    pub tile1_after_barrier: bool,
    /// compute the t+2 key after the barrier (the old order; A/B only)
    pub key_late: bool,
    /// `TwoPhaseOffset`: the next tile's key from a `read_first_lane` of the tile counter (wave-uniform: SALU instead of VALU for the clamp and
    /// the DMA address math)
    pub scalar_keys: bool,
}

impl Pipeline {
    pub fn new(cadence: Cadence, nkv: ValueId, kv_len: ValueId, tile_rows: i32, nkv_bound: Ext) -> Pipeline {
        assert!(tile_rows > 0 && (tile_rows as u32).is_power_of_two(), "Pipeline: tile rows must be a power of two");
        Pipeline { cadence, nkv, kv_len, tile_rows, nkv_bound, ring_var_last: true, prebind: Vec::new(), vm_inflight: 0, unroll2: false, top_vm: 0, prologue_vm: 0, pre_staged: false, ablate_no_sync: false, scalar_exit: false, late_tile1: false, late_tile1_vm: 0, tile1_after_barrier: false, key_late: false, mid_lds_drain: true, late_stage_b: false, scalar_keys: false }
    }
    pub fn slots(&self) -> u32 {
        match self.cadence {
            Cadence::TwoPhaseOffset => 2,
            Cadence::OneWindow => 3,
            Cadence::RegStaged => 2,
            Cadence::TwoSlot => 2,
        }
    }
    fn lg(&self) -> i32 {
        self.tile_rows.trailing_zeros() as i32
    }

    /// Emits the whole pipeline for one role (call it inside `Roles::each`, or once with `Role::All`).
    pub fn run(&self, b: &mut FnBuilder, role: Role, body: &mut dyn PipeBody) {
        let t0 = TileIt {
            it: Opnd::I32(0),
            cur: SlotIdx::new(0, 0),
            next: SlotIdx::new(1, 1),
            n2: match self.cadence {
                Cadence::OneWindow => Some(SlotIdx::new(2, 2)),
                Cadence::RegStaged => Some(SlotIdx::new(0, 2)),
                Cadence::TwoPhaseOffset | Cadence::TwoSlot => None,
            },
            masked: false,
            staged: true,
            tile_rows: self.tile_rows,
        };
        // ---- prologue ----
        let s0 = SlotIdx::new(0, 0);
        if !self.pre_staged {
            body.stage(b, role, &I32::c(0), &s0);
        }
        if self.cadence != Cadence::RegStaged {
            // (RegStaged bodies do not use stage_complete: tile 0 completes at the prologue wait, overlapped with the stores)
            b.wait_vm(0);
            body.stage_complete(b, role, &s0);
        }
        if self.cadence == Cadence::RegStaged && !self.pre_staged && !self.late_tile1 {
            // tile 1 into slot 1 (clamped as below); both complete at the prologue wait
            let key1 = b.let_i32(ev(self.nkv).gt(1).select_i32(1, ev(self.nkv) - 1) << self.lg());
            body.stage(b, role, &key1, &SlotIdx::new(1, 1));
        }
        if self.cadence == Cadence::OneWindow {
            // tile 1 (clamped to the last tile: a one-tile sequence re-stages tile 0 as generation 1)
            let key1 = b.let_i32(ev(self.nkv).gt(1).select_i32(1, ev(self.nkv) - 1) << self.lg());
            let s1 = SlotIdx::new(1, 1);
            body.stage(b, role, &key1, &s1);
            b.wait_vm(0);
            body.stage_complete(b, role, &s1);
        }
        body.prologue_stores(b, role);
        if self.cadence == Cadence::RegStaged && !self.pre_staged && self.late_tile1 && !self.tile1_after_barrier {
            // tile 1 after the prologue stores; it lands behind the first tile and is published by the loop's first top barrier
            let key1 = b.let_i32(ev(self.nkv).gt(1).select_i32(1, ev(self.nkv) - 1) << self.lg());
            body.stage(b, role, &key1, &SlotIdx::new(1, 1));
        }
        // RegStaged: tile 1 is first read after the loop's top-of-tile wait + barrier, so its DMAs may stay in flight here (`prologue_vm`)
        b.wait_vm(if self.cadence == Cadence::RegStaged { if self.late_tile1 { self.late_tile1_vm } else { self.prologue_vm } } else { 0 });
        b.wait_lds(0);
        b.barrier();
        if self.cadence == Cadence::TwoPhaseOffset && role == Role::B {
            b.barrier(); // group B runs one phase behind group A (A computes tile 0 before this barrier)
        }
        body.prologue_compute(b, role, &t0);
        if self.cadence == Cadence::RegStaged && !self.pre_staged && self.late_tile1 && self.tile1_after_barrier {
            // tile 1 behind the tile-0 fragment reads (slot 1 has no reader before the loop's first top barrier)
            let key1 = b.let_i32(ev(self.nkv).gt(1).select_i32(1, ev(self.nkv) - 1) << self.lg());
            body.stage(b, role, &key1, &SlotIdx::new(1, 1));
        }
        b.wait_lds(0);
        if self.cadence == Cadence::TwoPhaseOffset {
            b.barrier();
        }
        // ---- steady loop ----
        let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::I32, Opnd::I32(0))];
        // the variable inits must be values (materialised before the loop)
        let z = b.c_i32(0);
        inits[0].1 = Opnd::Val(z);
        inits[1].1 = Opnd::Val(z);
        let state = body.loop_state(b);
        let ns = state.len();
        if self.ring_var_last {
            let idx0 = inits.pop().expect("idx");
            inits.extend(state);
            inits.push(idx0);
        } else {
            inits.extend(state);
        }
        let vars = b.begin_loop(&inits);
        let (it, idx) = if self.ring_var_last { (vars[0], vars[1 + ns]) } else { (vars[0], vars[1]) };
        body.bind_state(if self.ring_var_last { &vars[1..1 + ns] } else { &vars[2..] });
        b.assume_trusted(
            it,
            Ext::c(0),
            self.nkv_bound.clone(),
            "tile counter (tile::Pipeline): starts at 0 and the loop exits at it >= nkv - 1, so the body only runs with it <= nkv - 2 <= ceil(rows / tile_rows); the guard compares against a derived value the interval engine cannot relate to the bound",
        );
        // wave-uniform: read_first_lane keeps the loop-exit compare scalar (s_cmp + s_cbranch_scc instead of v_cmp -> vcc + hazard nops)
        let nkv_m1 = if self.scalar_exit {
            let v = b.bind_i32(ev(self.nkv) - 1);
            let s = b.read_first_lane(v);
            ev(s)
        } else {
            b.let_i32(ev(self.nkv) - 1)
        };
        let stop = b.bind_cond(!varx(it).lt(&nkv_m1));
        b.break_if(stop);
        if self.unroll2 && self.cadence == Cadence::RegStaged {
            // `it` steps by 2 per pass (an affine counter for the protocol model); `idx` = which half the loop left from (0: after the pass, 1:
            // between the halves) - the last tile is it + idx, in slot idx
            // first half: tile `it` (even) in slot 0
            self.iter_rs(b, role, body, Opnd::Var(it), Opnd::I32(0), false, true);
            let one = b.c_i32(1);
            b.set_var(idx, one);
            let mid = b.bind_i32(varx(it) + 1);
            let stop1 = b.bind_cond(!ev(mid).lt(&nkv_m1));
            b.break_if(stop1);
            // second half: tile it + 1 (odd) in slot 1
            self.iter_rs(b, role, body, Opnd::Val(mid), Opnd::I32(1), false, true);
            let zero = b.c_i32(0);
            b.set_var(idx, zero);
            let it2 = b.bind_i32(varx(it) + 2);
            b.set_var(it, it2);
            let stop2 = b.bind_cond(!ev(it2).lt(&nkv_m1));
            b.break_if(stop2);
            b.end_loop();
            let it_t = b.bind_i32(varx(it) + varx(idx));
            let is_tail = b.bind_cond(ev(self.kv_len).lt((ev(it_t) << self.lg()) + self.tile_rows));
            b.begin_if(is_tail);
            self.iter_rs(b, role, body, Opnd::Val(it_t), Opnd::Var(idx), true, false);
            b.else_();
            self.iter_rs(b, role, body, Opnd::Val(it_t), Opnd::Var(idx), false, false);
            b.end_if();
            return;
        }
        let i_next = self.iter(b, role, body, it, idx, false, true, true);
        b.set_var(idx, i_next);
        let it_next = b.bind_i32(varx(it) + 1);
        b.set_var(it, it_next);
        let stop2 = b.bind_cond(!ev(it_next).lt(&nkv_m1));
        b.break_if(stop2);
        b.end_loop();
        // ---- peeled last tile ----
        let is_tail = b.bind_cond(ev(self.kv_len).lt((varx(it) << self.lg()) + self.tile_rows));
        let last_barrier = role != Role::B;
        b.begin_if(is_tail);
        self.iter(b, role, body, it, idx, true, false, last_barrier);
        b.else_();
        self.iter(b, role, body, it, idx, false, false, last_barrier);
        b.end_if();
    }

    fn prebind(&self, b: &mut FnBuilder, t: &TileIt) {
        for s in &self.prebind {
            t.cur.off_v(b, *s);
            t.next.off_v(b, *s);
            if let Some(n2) = &t.n2 {
                n2.off_v(b, *s);
            }
        }
    }

    /// One `RegStaged` iteration of tile `it` (a bound value) in the static ring slot `slot` (unroll2).
    fn iter_rs(&self, b: &mut FnBuilder, role: Role, body: &mut dyn PipeBody, it: Opnd, slot: Opnd, masked: bool, staged: bool) {
        let nkv = ev(self.nkv);
        let it_p1 = b.let_i32(amdgpu_dsl::ox(it) + 1);
        let it_p2 = b.let_i32(amdgpu_dsl::ox(it) + 2);
        let other = match slot {
            Opnd::I32(k) => Opnd::I32(1 - k),
            o => b.bind_opnd(&(amdgpu_dsl::ox(o) ^ 1)),
        };
        let t = TileIt {
            it,
            cur: SlotIdx::new(slot, it),
            next: SlotIdx::new(other, b.bind_opnd(&it_p1)),
            n2: Some(SlotIdx::new(slot, b.bind_opnd(&it_p2))),
            masked,
            staged,
            tile_rows: self.tile_rows,
        };
        self.prebind(b, &t);
        // the t+2 key (scalar) before the waits: right after the barrier the stage's first DMA issues at once (the vendor order)
        let key_n2 = if staged && !self.key_late { Some(b.let_i32(it_p2.lt(&nkv).select_i32(&it_p2, &nkv - 1) << self.lg())) } else { None };
        body.pre_sync(b, role, &t);
        if staged {
            if !self.ablate_no_sync {
                b.wait_vm(self.top_vm);
                b.wait_lds(0);
                body.post_wait(b, role, &t);
                b.barrier();
            }
            let key_n2 = key_n2.unwrap_or_else(|| b.let_i32(it_p2.lt(&nkv).select_i32(&it_p2, &nkv - 1) << self.lg()));
            body.stage(b, role, &key_n2, t.n2.as_ref().expect("n2"));
        }
        body.m_phase(b, role, &t);
    }

    /// One iteration; returns the next ring index.
    #[allow(clippy::too_many_arguments)]
    fn iter(&self, b: &mut FnBuilder, role: Role, body: &mut dyn PipeBody, it: VarId, idx: VarId, masked: bool, staged: bool, last_barrier: bool) -> ValueId {
        let nkv = ev(self.nkv);
        match self.cadence {
            Cadence::TwoPhaseOffset => {
                let i1 = b.bind_i32(varx(idx) ^ 1);
                let it_p1 = if self.scalar_keys {
                    let su = b.read_first_lane(Opnd::Var(it));
                    b.let_i32(ev(su) + 1)
                } else {
                    b.let_i32(varx(it) + 1)
                };
                let nn = it_p1.lt(&nkv).select_i32(&it_p1, &nkv - 1);
                let t = TileIt {
                    it: Opnd::Var(it),
                    cur: SlotIdx::new(idx, it),
                    next: SlotIdx::new(i1, b.bind_opnd(&it_p1)),
                    n2: None,
                    masked,
                    staged,
                    tile_rows: self.tile_rows,
                };
                self.prebind(b, &t);
                let late = self.late_stage_b && role == Role::B;
                let key_n = staged.then(|| b.let_i32(nn << self.lg()));
                if staged && !late {
                    body.stage(b, role, key_n.as_ref().expect("key"), &t.next);
                }
                body.v_phase(b, role, &t);
                if staged && !late {
                    b.wait_vm(0);
                    body.stage_complete(b, role, &t.next);
                }
                if self.mid_lds_drain {
                    b.wait_lds(0);
                }
                b.barrier();
                if staged && late {
                    body.defer_stage(b, role, key_n.as_ref().expect("key"), &t.next);
                }
                body.m_phase(b, role, &t);
                if staged && late {
                    b.wait_vm(0);
                    body.stage_complete(b, role, &t.next);
                }
                b.wait_lds(0);
                if last_barrier {
                    b.barrier();
                }
                i1
            }
            Cadence::TwoSlot => {
                let i1 = b.bind_i32(varx(idx) ^ 1);
                let it_p1 = b.let_i32(varx(it) + 1);
                let nn = it_p1.lt(&nkv).select_i32(&it_p1, &nkv - 1);
                let t = TileIt {
                    it: Opnd::Var(it),
                    cur: SlotIdx::new(idx, it),
                    next: SlotIdx::new(i1, b.bind_opnd(&it_p1)),
                    n2: None,
                    masked,
                    staged,
                    tile_rows: self.tile_rows,
                };
                self.prebind(b, &t);
                if staged {
                    let key_n = b.let_i32(nn << self.lg());
                    body.stage(b, role, &key_n, &t.next);
                }
                body.window(b, role, &t);
                if staged {
                    b.wait_vm(0);
                    b.wait_lds(0);
                    b.barrier();
                }
                i1
            }
            Cadence::RegStaged => {
                let i1 = b.bind_i32(varx(idx) ^ 1);
                let it_p1 = b.let_i32(varx(it) + 1);
                let it_p2 = b.let_i32(varx(it) + 2);
                let t = TileIt {
                    it: Opnd::Var(it),
                    cur: SlotIdx::new(idx, it),
                    next: SlotIdx::new(i1, b.bind_opnd(&it_p1)),
                    // tile t+2 reuses the slot of tile t (its registers were loaded in the previous iteration)
                    n2: Some(SlotIdx::new(idx, b.bind_opnd(&it_p2))),
                    masked,
                    staged,
                    tile_rows: self.tile_rows,
                };
                self.prebind(b, &t);
                body.pre_sync(b, role, &t);
                if staged {
                    if !self.ablate_no_sync {
                        b.wait_vm(self.top_vm);
                        b.wait_lds(0);
                        body.post_wait(b, role, &t);
                        b.barrier();
                    }
                    let key_n2 = b.let_i32(it_p2.lt(&nkv).select_i32(&it_p2, &nkv - 1) << self.lg());
                    body.stage(b, role, &key_n2, t.n2.as_ref().expect("n2"));
                }
                body.m_phase(b, role, &t);
                i1
            }
            Cadence::OneWindow => {
                let i1 = b.bind_i32(varx(idx).lt(2).select_i32(varx(idx) + 1, 0));
                let i2 = b.bind_i32(I32::c(0).lt(varx(idx)).select_i32(varx(idx) - 1, 2));
                let it_p1 = b.let_i32(varx(it) + 1);
                let it_p2 = b.let_i32(varx(it) + 2);
                let t = TileIt {
                    it: Opnd::Var(it),
                    cur: SlotIdx::new(idx, it),
                    next: SlotIdx::new(i1, b.bind_opnd(&it_p1)),
                    n2: Some(SlotIdx::new(i2, b.bind_opnd(&it_p2))),
                    masked,
                    staged,
                    tile_rows: self.tile_rows,
                };
                self.prebind(b, &t);
                let n2 = t.n2.clone().expect("n2");
                if staged {
                    let key_n2 = b.let_i32(it_p2.lt(&nkv).select_i32(&it_p2, &nkv - 1) << self.lg());
                    body.stage(b, role, &key_n2, &n2);
                }
                body.window(b, role, &t);
                if staged {
                    b.wait_vm(self.vm_inflight);
                    body.stage_complete(b, role, &n2);
                    b.wait_lds(0);
                    b.barrier();
                }
                i1
            }
        }
    }
}
