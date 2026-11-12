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

pub use amdgpu_tile::lane::TileIt;
use amdgpu_tile::lane::Role;
use amdgpu_tile::shared::SlotIdx;
use crate::weave::{rotate, Part, RotStage};

/// The `OneWindow` iteration: stage(t+2) | window(t) | complete(t+2) + wait + barrier.
const ONE_WINDOW_PARTS: [Part; 3] = [
    Part { name: "stage", iter: 2, anchor: false },
    Part { name: "window", iter: 0, anchor: false },
    Part { name: "complete_sync", iter: 2, anchor: true },
];

/// The `RegStaged` iteration: key(t+2) | pre_sync(t) | waits + post_wait(t) + barrier | stage(t+2) | M phase(t).
const REG_STAGED_PARTS: [Part; 5] = [
    Part { name: "key", iter: 2, anchor: false },
    Part { name: "pre_sync", iter: 0, anchor: false },
    Part { name: "sync", iter: 2, anchor: true },
    Part { name: "stage", iter: 2, anchor: false },
    Part { name: "m_phase", iter: 0, anchor: false },
];

/// The `TwoPhaseOffset` iteration as rotated parts: stage(t+1) | V phase(t) | complete(t+1) | wait + barrier | M phase(t) | wait (+ barrier).
const TWO_PHASE_PARTS: [Part; 6] = [
    Part { name: "stage", iter: 1, anchor: false },
    Part { name: "v_phase", iter: 0, anchor: false },
    Part { name: "complete", iter: 1, anchor: true },
    Part { name: "mid_sync", iter: 0, anchor: true },
    Part { name: "m_phase", iter: 0, anchor: false },
    Part { name: "end_sync", iter: 0, anchor: true },
];
use amdgpu_dsl::{ev, varx};
use amdgpu_dsl::{DType, Ext, FnBuilder, Opnd, ValueId, VarId, I32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cadence {
    TwoPhaseOffset,
    OneWindow,
    /// 2-slot ring with the operands staged through REGISTERS (the aiter 4-wave GEMM): the body keeps tile t in registers (loop state) and, while
    /// computing it, reloads them in place from the slot of tile t+1; one wait + barrier per tile publishes t+1 and frees the slot of t, which then
    /// receives tile t+2. Two tiles per loop pass with STATIC ring slots (0 then 1: the tile counter is even at the top; the vendor `u2`), the loop
    /// may exit between the two halves:
    /// ```text
    /// prologue: stage(0) | stores | stage(1) | wait_vm(tile1_vm) wait_lds(0) barrier | compute(0) = load tile 0 into registers
    /// loop t:   pre_sync(t) | wait_vm(0) wait_lds(0) post_wait(t) barrier | stage(t+2) -> slot(t) | m_phase(t): MMAs of t + in-place loads of t+1
    /// tail:     m_phase (no loads: `staged` = false)
    /// ```
    RegStaged,
}

/// What a kernel supplies to the pipeline. `role` is the compile-time group tag of the branch being emitted.
pub trait PipeBody {
    /// Issue the copies of tile `key0` (first row) into `slot` for the stages this role produces (e.g. `RowsCopy::issue` on `buf.at(slot)`).
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx);
    /// Called right after the pipeline's `wait_vm(0)` that completes the stage of `slot` (e.g. the VGPR path of the K scales: `ds_write` them now).
    fn stage_complete(&mut self, _b: &mut FnBuilder, _role: Role, _slot: &SlotIdx) {}
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
    /// ring slot strides whose `slot * stride` offsets (cur, next, n2) are bound at the TOP of every iteration, like the manual `ring()` (otherwise the
    /// first consumer binds them, possibly late in the window: `_t128` measured 4 `v_mul_lo_u32` at the window head). E.g. `vec![32768, 512]`.
    pub prebind: Vec<i64>,
    /// `OneWindow`: vector-memory operations allowed in flight at the end of a window (`s_waitcnt vmcnt(n)` instead of 0): the number of VMEM ops ONE
    /// stage issues per wave, so that the wait completes tile t+1 (published by the barrier) while tile t+2, issued at this window's start, keeps
    /// loading across the next window. Only valid when the body issues no other VMEM in the window and does not use `stage_complete` (the
    /// verifier checks the completion with the counted-vmcnt rule).
    pub vm_inflight: u8,
    /// `TwoPhaseOffset` + `unroll2`: the body stages tile t + 2 itself in its matrix phase (`TileIt::n2` = the static slot of t, generation
    /// t + 2; docs/fmha-replica.md R3 step 5, the vendor's copy distance); the prologue stages tiles 0 and 1, part 0 stages nothing
    pub ahead2: bool,
    /// `RegStaged`: the DMAs of tile 1 (staged after `prologue_stores`) that may still be in flight at the prologue barrier: tile 1 is first read after
    /// the loop's first top-of-tile wait + barrier
    pub late_tile1_vm: u8,
    /// loop-exit bound through read_first_lane (scalar compare)
    pub scalar_exit: bool,
    /// `RegStaged` ring depth: 2 (default) or 3. Depth 3 = a 3-slot ring, three tiles per loop pass with static slots 0 / 1 / 2: iteration t
    /// stages tile t + 3 into the slot of t and its top wait keeps the DMAs of tile t + 2 in flight (`s_waitcnt vmcnt(late_tile1_vm)`: the body
    /// issues exactly `late_tile1_vm` VMEM ops per stage and none elsewhere in the loop); for small tiles whose compute per tile is shorter than
    /// the memory latency.
    pub rs_depth: u8,
    /// `TwoPhaseOffset`: the next tile's key without the clamp to the last tile (inside the loop t + 1 <= nkv - 1 holds; valid only when
    /// every stage reads through a range-checked buffer descriptor, so the verifier needs no clamp for the bounds)
    pub key_unclamped: bool,
    /// `TwoPhaseOffset`: two tiles per loop pass with STATIC ring slots (0 then 1, the vendor kernel's unrolled pair; docs/fmha-replica.md R3 step
    /// 2): an exit after each tile, `idx` records which half the loop left from and the peeled last tile runs in slot `idx`
    pub unroll2: bool,
}

impl Pipeline {
    pub fn new(cadence: Cadence, nkv: ValueId, kv_len: ValueId, tile_rows: i32, nkv_bound: Ext) -> Pipeline {
        assert!(tile_rows > 0 && (tile_rows as u32).is_power_of_two(), "Pipeline: tile rows must be a power of two");
        Pipeline { cadence, nkv, kv_len, tile_rows, nkv_bound, prebind: Vec::new(), vm_inflight: 0, ahead2: false, late_tile1_vm: 0, scalar_exit: false, key_unclamped: false, rs_depth: 2, unroll2: false }
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
                Cadence::RegStaged => Some(SlotIdx::new(0, self.rs_depth as i32)),
                Cadence::TwoPhaseOffset => None,
            },
            masked: false,
            staged: true,
            tile_rows: self.tile_rows,
        };
        // ---- prologue ----
        let s0 = SlotIdx::new(0, 0);
        body.stage(b, role, &I32::c(0), &s0);
        if self.cadence != Cadence::RegStaged {
            // (RegStaged bodies do not use stage_complete: tile 0 completes at the prologue wait, overlapped with the stores)
            b.wait_vm(0);
            body.stage_complete(b, role, &s0);
        }
        if self.cadence == Cadence::OneWindow {
            // tile 1 (clamped to the last tile: a one-tile sequence re-stages tile 0 as generation 1)
            let key1 = b.let_i32(ev(self.nkv).gt(1).select_i32(1, ev(self.nkv) - 1) << self.lg());
            let s1 = SlotIdx::new(1, 1);
            body.stage(b, role, &key1, &s1);
            b.wait_vm(0);
            body.stage_complete(b, role, &s1);
        }
        if self.ahead2 {
            assert!(self.cadence == Cadence::TwoPhaseOffset && self.unroll2, "ahead2: TwoPhaseOffset with unroll2");
            // tile 1 as well (completed by the first iteration's wait: it is the first `next`)
            let key1 = b.let_i32(ev(self.nkv).gt(1).select_i32(1, ev(self.nkv) - 1) << self.lg());
            body.stage(b, role, &key1, &SlotIdx::new(1, 1));
        }
        body.prologue_stores(b, role);
        if self.cadence == Cadence::RegStaged {
            // tile 1 after the prologue stores; it lands behind the first tile and is published by the loop's first top barrier
            let key1 = b.let_i32(ev(self.nkv).gt(1).select_i32(1, ev(self.nkv) - 1) << self.lg());
            body.stage(b, role, &key1, &SlotIdx::new(1, 1));
            // depth D >= 3: tiles 2 .. D - 1 as well (clamped)
            for k in 2..self.rs_depth.max(2) as i32 {
                let key2 = b.let_i32(ev(self.nkv).gt(k).select_i32(k, ev(self.nkv) - 1) << self.lg());
                body.stage(b, role, &key2, &SlotIdx::new(k, k));
            }
        }
        // RegStaged: tile 1 (depth 3: and tile 2) is first read after the loop's top-of-tile wait + barrier, so its DMAs may stay in flight here
        let rs_late = if self.rs_depth >= 3 { (self.rs_depth - 1) * self.late_tile1_vm } else { self.late_tile1_vm };
        b.wait_vm(if self.cadence == Cadence::RegStaged { rs_late } else { 0 });
        b.wait_lds(0);
        b.barrier();
        if self.cadence == Cadence::TwoPhaseOffset && role == Role::B {
            b.barrier(); // group B runs one phase behind group A (A computes tile 0 before this barrier)
        }
        body.prologue_compute(b, role, &t0);
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
        // loop-variable order [it, state.., idx] (the manual kernels' order: measured +2.3 % per-CU on the _vs128 port vs [it, idx, state..])
        let idx0 = inits.pop().expect("idx");
        inits.extend(state);
        inits.push(idx0);
        let vars = b.begin_loop(&inits);
        let (it, idx) = (vars[0], vars[1 + ns]);
        body.bind_state(&vars[1..1 + ns]);
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
        if self.cadence == Cadence::RegStaged && self.rs_depth >= 3 {
            // depth D >= 3: tiles it .. it + D - 1 in slots 0 .. D - 1; `idx` = the slot the loop left from
            let d = self.rs_depth as i32;
            let mut cur = Opnd::Var(it);
            for k in 0..d {
                self.iter_rs(b, role, body, cur.clone(), Opnd::I32(k), false, true);
                if k + 1 < d {
                    let kn = b.c_i32(k + 1);
                    b.set_var(idx, kn);
                    let mid = b.bind_i32(varx(it) + (k + 1));
                    let stop1 = b.bind_cond(!ev(mid).lt(&nkv_m1));
                    b.break_if(stop1);
                    cur = Opnd::Val(mid);
                } else {
                    let zero = b.c_i32(0);
                    b.set_var(idx, zero);
                    let itd = b.bind_i32(varx(it) + d);
                    b.set_var(it, itd);
                    let stopd = b.bind_cond(!ev(itd).lt(&nkv_m1));
                    b.break_if(stopd);
                }
            }
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
        if self.cadence == Cadence::RegStaged {
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
        if self.cadence == Cadence::TwoPhaseOffset && self.unroll2 {
            // first half: tile `it` (even) in slot 0
            self.iter_tp2(b, role, body, Opnd::Var(it), Opnd::I32(0), false, true, true);
            let one = b.c_i32(1);
            b.set_var(idx, one);
            let mid = b.bind_i32(varx(it) + 1);
            let stop1 = b.bind_cond(!ev(mid).lt(&nkv_m1));
            b.break_if(stop1);
            // second half: tile it + 1 (odd) in slot 1
            self.iter_tp2(b, role, body, Opnd::Val(mid), Opnd::I32(1), false, true, true);
            let zero = b.c_i32(0);
            b.set_var(idx, zero);
            let it2 = b.bind_i32(varx(it) + 2);
            b.set_var(it, it2);
            let stop2 = b.bind_cond(!ev(it2).lt(&nkv_m1));
            b.break_if(stop2);
            b.end_loop();
            // the last tile: it + idx, in slot idx. One drain per exit with its STATIC slot (the vendor's drain1 / drain0): `if (idx == 1)
            // {drain in slot 1} else {drain in slot 0}`; the back-end Weaver threads each exit edge (known `idx`) straight to its drain
            let last_barrier = role != Role::B;
            // `0 < idx` (idx is 0 / 1): a less-than compare, which the back-end jump threading folds per edge (`jthread`)
            let which = b.bind_cond(I32::c(0).lt(varx(idx)));
            b.begin_if(which);
            for k in [1, 0] {
                let it_t = b.bind_i32(varx(it) + k);
                let is_tail = b.bind_cond(ev(self.kv_len).lt((ev(it_t) << self.lg()) + self.tile_rows));
                b.begin_if(is_tail);
                self.iter_tp2(b, role, body, Opnd::Val(it_t), Opnd::I32(k), true, false, last_barrier);
                b.else_();
                self.iter_tp2(b, role, body, Opnd::Val(it_t), Opnd::I32(k), false, false, last_barrier);
                b.end_if();
                if k == 1 {
                    b.else_();
                }
            }
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

    /// One `TwoPhaseOffset` iteration of tile `it` (a bound value) in the static ring slot `slot` (`unroll2`); the same rotated weave as `iter`.
    #[allow(clippy::too_many_arguments)]
    fn iter_tp2(&self, b: &mut FnBuilder, role: Role, body: &mut dyn PipeBody, it: Opnd, slot: Opnd, masked: bool, staged: bool, last_barrier: bool) {
        let nkv = ev(self.nkv);
        let other = match slot {
            Opnd::I32(k) => Opnd::I32(1 - k),
            o => b.bind_opnd(&(amdgpu_dsl::ox(o) ^ 1)),
        };
        let it_p1 = b.let_i32(amdgpu_dsl::ox(it) + 1);
        let nn = it_p1.lt(&nkv).select_i32(&it_p1, &nkv - 1);
        let n2 = if self.ahead2 { let it_p2 = b.let_i32(amdgpu_dsl::ox(it) + 2); Some(SlotIdx::new(slot, b.bind_opnd(&it_p2))) } else { None };
        let t = TileIt { it, cur: SlotIdx::new(slot, it), next: SlotIdx::new(other, b.bind_opnd(&it_p1)), n2, masked, staged, tile_rows: self.tile_rows };
        self.prebind(b, &t);
        let stage_sel = if staged { RotStage::Steady } else { RotStage::Epilogue };
        let (lg, unclamped, ahead2) = (self.lg(), self.key_unclamped, self.ahead2);
        rotate(b, &TWO_PHASE_PARTS, stage_sel, None, &mut |b, i| match i {
            0 if ahead2 => {}
            0 => {
                let k = if unclamped { b.let_i32((amdgpu_dsl::ox(it) + 1) << lg) } else { b.let_i32(nn.clone() << lg) };
                body.stage(b, role, &k, &t.next);
            }
            1 => body.v_phase(b, role, &t),
            2 => {
                b.wait_vm(0);
                body.stage_complete(b, role, &t.next);
            }
            3 => {
                b.wait_lds(0);
                b.barrier();
            }
            4 => body.m_phase(b, role, &t),
            5 => {
                b.wait_lds(0);
                if last_barrier {
                    b.barrier();
                }
            }
            _ => unreachable!(),
        });
    }

    /// One `RegStaged` iteration of tile `it` (a bound value) in the static ring slot `slot`.
    fn iter_rs(&self, b: &mut FnBuilder, role: Role, body: &mut dyn PipeBody, it: Opnd, slot: Opnd, masked: bool, staged: bool) {
        let nkv = ev(self.nkv);
        let it_p1 = b.let_i32(amdgpu_dsl::ox(it) + 1);
        let d3 = self.rs_depth >= 3;
        let dd = self.rs_depth.max(2) as i32;
        // the tile staged in this iteration: t + depth, into the slot of t
        let it_p2 = b.let_i32(amdgpu_dsl::ox(it) + if d3 { dd } else { 2 });
        let other = match (slot, d3) {
            (Opnd::I32(k), false) => Opnd::I32(1 - k),
            (Opnd::I32(k), true) => Opnd::I32((k + 1) % dd),
            (o, false) => b.bind_opnd(&(amdgpu_dsl::ox(o) ^ 1)),
            (o, true) => b.bind_opnd(&amdgpu_dsl::ox(o).lt(dd - 1).select_i32(amdgpu_dsl::ox(o) + 1, 0)),
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
        // rotated weave: the t+2 key (scalar, before the waits: right after the barrier the stage's first DMA issues at once, the vendor order),
        // the sync with the work issued after the waits, and the staging of t+2 run two iterations ahead (`iter` +2)
        let stage_sel = if staged { RotStage::Steady } else { RotStage::Epilogue };
        let lg = self.lg();
        let mut key_n2: Option<I32> = None;
        rotate(b, &REG_STAGED_PARTS, stage_sel, None, &mut |b, i| match i {
            0 => key_n2 = Some(b.let_i32(it_p2.lt(&nkv).select_i32(&it_p2, &nkv - 1) << lg)),
            1 => body.pre_sync(b, role, &t),
            2 => {
                // (depth 3: the DMAs of t + 2, issued in the previous iteration, stay in flight)
                b.wait_vm(if d3 { (self.rs_depth - 2) * self.late_tile1_vm } else { 0 });
                b.wait_lds(0);
                body.post_wait(b, role, &t);
                b.barrier();
            }
            3 => body.stage(b, role, key_n2.as_ref().expect("key"), t.n2.as_ref().expect("n2")),
            4 => body.m_phase(b, role, &t),
            _ => unreachable!(),
        });
    }

    /// One iteration; returns the next ring index.
    #[allow(clippy::too_many_arguments)]
    fn iter(&self, b: &mut FnBuilder, role: Role, body: &mut dyn PipeBody, it: VarId, idx: VarId, masked: bool, staged: bool, last_barrier: bool) -> ValueId {
        let nkv = ev(self.nkv);
        match self.cadence {
            Cadence::TwoPhaseOffset => {
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
                // the iteration as a rotated weave: the staging of tile t+1 (and its completion) runs one iteration ahead (`iter` +1), so the
                // peeled last iteration (`staged` = false) is the epilogue of the rotation; waits and barriers are anchors
                let stage_sel = if staged { RotStage::Steady } else { RotStage::Epilogue };
                let (lg, unclamped) = (self.lg(), self.key_unclamped);
                rotate(b, &TWO_PHASE_PARTS, stage_sel, None, &mut |b, i| match i {
                    0 => {
                        let k = if unclamped { b.let_i32((varx(it) + 1) << lg) } else { b.let_i32(nn.clone() << lg) };
                        body.stage(b, role, &k, &t.next);
                    }
                    1 => body.v_phase(b, role, &t),
                    2 => {
                        b.wait_vm(0);
                        body.stage_complete(b, role, &t.next);
                    }
                    3 => {
                        b.wait_lds(0);
                        b.barrier();
                    }
                    4 => body.m_phase(b, role, &t),
                    5 => {
                        b.wait_lds(0);
                        if last_barrier {
                            b.barrier();
                        }
                    }
                    _ => unreachable!(),
                });
                i1
            }
            // RegStaged runs two tiles per pass with static slots (`iter_rs`)
            Cadence::RegStaged => unreachable!("Pipeline: RegStaged iterations are emitted by iter_rs"),
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
                // rotated weave: the staging of tile t+2 and its completion run two iterations ahead (`iter` +2)
                let stage_sel = if staged { RotStage::Steady } else { RotStage::Epilogue };
                let (lg, vm) = (self.lg(), self.vm_inflight);
                rotate(b, &ONE_WINDOW_PARTS, stage_sel, None, &mut |b, i| match i {
                    0 => {
                        let key_n2 = b.let_i32(it_p2.lt(&nkv).select_i32(&it_p2, &nkv - 1) << lg);
                        body.stage(b, role, &key_n2, &n2);
                    }
                    1 => body.window(b, role, &t),
                    2 => {
                        b.wait_vm(vm);
                        body.stage_complete(b, role, &n2);
                        b.wait_lds(0);
                        b.barrier();
                    }
                    _ => unreachable!(),
                });
                i1
            }
        }
    }
}
