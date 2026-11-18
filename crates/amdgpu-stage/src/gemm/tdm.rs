//! The GEMM stage template with a TDM producer (gfx1250, G3 step 5 of docs/gfx1250-plan.md): the register-free window of `Cadence::OneWindow` (3-slot K ring, stage(t + 2) | window(t) |
//! complete + barrier(t + 2)) where `stage` issues `tensor_load_to_lds` (the tensor data mover) instead of an LDS-DMA copy, and the window reads the operand fragments from the slot and
//! feeds a WMMA.
//!
//! The pipeline driver (`crate::pipeline`) owns the cadence: the loop, the slot / generation arithmetic (`TileIt`), the rotation by iteration offsets (the producer part runs at
//! `iter = 2`, the plan unit class `Dma`), the ring markers and the barrier. This body supplies only the work of the phases:
//!
//! * `stage`  : per-wave descriptors (group 1 shared by every load, group 0 per load: the global tile address, the LDS slot) and the two tensor loads of A and B, inside the ring's
//!   produce markers;
//! * `stage_complete`: the wait for the tensor counter (the driver's `wait_vm` is a no-op for a TDM producer); the back end may relax it to the exact count from the slot tokens;
//! * `window` : the fragments of the current slot (consume markers) and one WMMA per K tile into the loop-carried accumulator.
//!
//! Every wave issues the same loads (no cross-wave tensor completion): the barrier of the pipeline is the slot-reuse release.

use amdgpu_dsl::{ev, DType, FnBuilder, Op, Opnd, TdmCpol, ValueId, VarId, WmmaKind};
use amdgpu_tile::lane::{Role, TileIt};
use amdgpu_tile::shared::{Shared, SlotIdx};

use crate::PipeBody;

/// Group 0 of a tensor descriptor: valid, the LDS address `lds`, the 57-bit global address of `base + byte_off`, type 2 (the words `FnBuilder::tensor_desc_2d` builds for its group 0).
pub fn tdm_group0(b: &mut FnBuilder, base: ValueId, byte_off: ValueId, lds: Opnd) -> ValueId {
    let p = b.let_(Op::PtrAdd { base: Opnd::Val(base), byte_off: Opnd::Val(byte_off) });
    let lo = b.let_(Op::PtrLo(Opnd::Val(p)));
    let hi = b.let_(Op::PtrHi(Opnd::Val(p)));
    let hm = b.let_(Op::AndI32(Opnd::Val(hi), Opnd::I32(0x01ff_ffff)));
    let w3 = b.let_(Op::AddI32(Opnd::Val(hm), Opnd::I32(0x8000_0000u32 as i32)));
    b.let_(Op::RsrcCat { words: [Opnd::I32(1), lds, Opnd::Val(lo), Opnd::Val(w3)] })
}

/// The GEMM window of one wave: C(16 x 16) += A(16 x 64 fp8) x B(16 x 64 fp8)^T per K tile of 64 bytes.
pub struct TdmGemmBody {
    /// the A / B rings (one slot = the tile of 32 rows x 64 bytes of A / B)
    pub a: Shared,
    pub b: Shared,
    /// the kernarg pointers of A / B
    pub pa: ValueId,
    pub pb: ValueId,
    /// byte offsets of this work-group's first A / B row (row0 * K, col0 * K)
    pub a_org: ValueId,
    pub b_org: ValueId,
    /// group 1 of the descriptor, shared by every load (extents, strides, the tile)
    pub g1: ValueId,
    /// this lane's byte offset of its A / B fragment inside a slot's tile (row m of the wave's 16 rows, K byte 8 g)
    pub abase: ValueId,
    pub bbase: ValueId,
    /// the loop-carried accumulator (bound by `bind_state`)
    pub acc: Option<VarId>,
}

impl PipeBody for TdmGemmBody {
    fn stage(&mut self, b: &mut FnBuilder, _role: Role, key0: &amdgpu_dsl::I32, slot: &SlotIdx) {
        // `key0` is the K byte offset of the tile (tile_rows = 64: tile << 6)
        let koff = b.bind_i32(key0.clone());
        let oa = b.add_i32(self.a_org, koff);
        let ob = b.add_i32(self.b_org, koff);
        let (ra, rb) = (self.a.at(slot), self.b.at(slot));
        let la = b.bind_i32(ra.base());
        let lb = b.bind_i32(rb.base());
        let g0a = tdm_group0(b, self.pa, oa, Opnd::Val(la));
        let g0b = tdm_group0(b, self.pb, ob, Opnd::Val(lb));
        let g1 = self.g1;
        ra.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0a), Opnd::Val(g1), None, TdmCpol::default()));
        rb.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0b), Opnd::Val(g1), None, TdmCpol::default()));
    }

    fn finish(&mut self, b: &mut FnBuilder, _role: Role) {
        b.wait_tensor(0);   // the last steady iteration's clamped stage is still in flight
    }

    fn stage_complete(&mut self, b: &mut FnBuilder, _role: Role, _slot: &SlotIdx) {
        // the loads of the slot just staged are the only ones outstanding (the driver completes every stage before it issues the next)
        b.wait_tensor(0);
    }

    fn prologue_compute(&mut self, _b: &mut FnBuilder, _role: Role, _t0: &TileIt) {}

    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let zero = b.frag_cat(&[Opnd::I32(0); 8]);
        vec![(WmmaKind::Fp8_16x16x64.acc_ty(), Opnd::Val(zero))]
    }

    fn bind_state(&mut self, vars: &[VarId]) {
        self.acc = Some(vars[0]);
    }

    fn window(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let acc = self.acc.expect("loop state bound");
        let (ra, rb) = (self.a.at(&t.cur), self.b.at(&t.cur));
        let (abase, bbase) = (self.abase, self.bbase);
        let pa: Vec<Opnd> = ra.read(b, |b| {
            (0..4).map(|j| {
                let addr = b.bind_i32(ra.base() + ev(abase) + 16 * j);
                Opnd::Val(b.load_lds_b64(addr))
            }).collect()
        });
        let pb: Vec<Opnd> = rb.read(b, |b| {
            (0..4).map(|j| {
                let addr = b.bind_i32(rb.base() + ev(bbase) + 16 * j);
                Opnd::Val(b.load_lds_b64(addr))
            }).collect()
        });
        let fa = b.frag_cat(&pa);
        let fb = b.frag_cat(&pb);
        let d = b.wmma(WmmaKind::Fp8_16x16x64, fa, fb, Some(Opnd::Var(acc)));
        b.set_var(acc, Opnd::Val(d));
    }
}

