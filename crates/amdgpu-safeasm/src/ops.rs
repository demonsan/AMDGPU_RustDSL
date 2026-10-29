//! Opcode templates. The operand types encode the encoding rules of gfx9/gfx950 VALU and matrix instructions:
//!
//! * `e32` forms: src0 may be a VGPR, SGPR, inline constant or literal (`Operand`); src1 must be a VGPR (`VgprOnly`).
//! * VOP3 / `e64` forms: any operand except a literal, and the constant bus allows ONE scalar source in total (checked at compile
//!   time through an associated constant: a violation is a readable compile error, not an assembler error).
//! * `v_cndmask_b32_e32` reads `vcc` through the constant bus, so it takes VGPR sources only (type-enforced by `VgprOnly`); the
//!   `_e64` form reads an SGPR pair as its mask, which uses the single bus slot, so its data operands must be bus-free.
//! * matrix instructions take `Ready` operands (the 2-wait VALU->matrix rule), keep their operands borrowed while in flight and return
//!   a token that must be retired (20 wait states) before the result can be touched.

use crate::regs::*;
use crate::seq::*;
use amdgpu_compiler::mir::{MInst, MOp, Src};
use std::marker::PhantomData;

struct Bus1<A, B>(PhantomData<(A, B)>);
impl<A: Operand, B: Operand> Bus1<A, B> {
    const OK: () = assert!(
        A::BUS + B::BUS <= 1 && A::VOP3_OK && B::VOP3_OK,
        "gfx9 VOP3: at most one SGPR/literal source per instruction (constant bus limit 1) and no 32-bit literal"
    );
}
struct Bus2<A, B>(PhantomData<(A, B)>);
impl<A: Operand, B: Operand> Bus2<A, B> {
    const OK: () = assert!(
        A::BUS + B::BUS == 0 && A::VOP3_OK && B::VOP3_OK,
        "v_cndmask_b32_e64 reads its SGPR-pair mask through the constant bus: the data operands must be VGPRs or inline constants"
    );
}

impl Seq {
    // ---- VALU, e32 -------------------------------------------------------------------------------------------------------------
    /// `v_mov_b32 dst, src0`.
    pub fn v_mov_b32<D: Dst<1>, A: Operand>(&mut self, dst: D, a: A) {
        let i = MInst::new(MOp::VMovB32).def(dst.dst_id()).src(a.to_src());
        // a plain VGPR source is a copy; an SGPR source is `v_mov_b32 v, s`
        let i = match a.to_src() {
            Src::Reg(_) => i,
            Src::Imm(k) => MInst::new(MOp::VMovB32).def(dst.dst_id()).imm(k),
        };
        self.push(i);
        self.note_valu_write(dst.dst_id());
    }

    /// `v_add_u32_e32 dst, src0, vsrc1`.
    pub fn v_add_u32<D: Dst<1>, A: Operand, B: VgprOnly>(&mut self, dst: D, a: A, b: B) {
        self.push(MInst::new(MOp::VAddU32).def(dst.dst_id()).src(a.to_src()).src(b.to_src()));
        self.note_valu_write(dst.dst_id());
    }

    /// `v_add_f32_e32 dst, src0, vsrc1`.
    pub fn v_add_f32<D: Dst<1>, A: Operand, B: VgprOnly>(&mut self, dst: D, a: A, b: B) {
        self.push(MInst::new(MOp::VAddF32).def(dst.dst_id()).src(a.to_src()).src(b.to_src()));
        self.note_valu_write(dst.dst_id());
    }

    /// `v_add_f32_e32 acc, src0, acc` (accumulate in place).
    pub fn v_add_f32_acc<A: Operand>(&mut self, acc: &mut Vgpr<1>, a: A) {
        self.push(MInst::new(MOp::VAddF32).def(acc.id).src(a.to_src()).srcreg(acc.id));
        self.note_valu_write(acc.id);
    }

    // ---- VALU, VOP3 -------------------------------------------------------------------------------------------------------------
    /// `v_mul_lo_u32 dst, a, b` (VOP3: one scalar source at most, no literal).
    pub fn v_mul_lo_u32<D: Dst<1>, A: Operand, B: Operand>(&mut self, dst: D, a: A, b: B) {
        let () = Bus1::<A, B>::OK;
        self.push(MInst::new(MOp::VMulLoU32).def(dst.dst_id()).src(a.to_src()).src(b.to_src()));
        self.note_valu_write(dst.dst_id());
    }

    // ---- compare / select through vcc ---------------------------------------------------------------------------------------------
    /// `v_cmp_lt_i32_e32 vcc, src0, vsrc1`: `Vcc<Free>` in, `Vcc<Live>` out.
    pub fn v_cmp_lt_i32<A: Operand, B: VgprOnly>(&mut self, _vcc: Vcc<Free>, a: A, b: B) -> Vcc<Live> {
        self.push(MInst::new(MOp::VCmpLtI32).src(a.to_src()).src(b.to_src()));
        self.vcc_written();
        Vcc { _s: PhantomData }
    }

    /// `v_cndmask_b32_e32 dst, if_false, if_true, vcc`: VGPR sources only (vcc already uses the constant bus). Consumes the mask.
    pub fn v_cndmask_b32<D: Dst<1>, F: VgprOnly, T: VgprOnly>(&mut self, _vcc: Vcc<Live>, dst: D, if_false: F, if_true: T) -> Vcc<Free> {
        self.vcc_ready();
        self.push(MInst::new(MOp::VCndmask).def(dst.dst_id()).src(if_false.to_src()).src(if_true.to_src()));
        self.note_valu_write(dst.dst_id());
        Vcc { _s: PhantomData }
    }

    // ---- compare / select through an SGPR pair (several masks can be live) ---------------------------------------------------------
    /// `v_cmp_lt_i32_e64 s[a:b], x, y` (VOP3).
    pub fn v_cmp_lt_i32_e64<A: Operand, B: Operand>(&mut self, mask: &mut Sgpr<2>, a: A, b: B) {
        let () = Bus1::<A, B>::OK;
        self.push(MInst::new(MOp::VCmpMaskLtI32).def(mask.id).src(a.to_src()).src(b.to_src()));
        self.note_valu_write(mask.id);
    }

    /// `v_cndmask_b32_e64 dst, if_false, if_true, s[a:b]`.
    pub fn v_cndmask_b32_e64<D: Dst<1>, F: Operand, T: Operand>(&mut self, dst: D, if_false: F, if_true: T, mask: &Sgpr<2>) {
        let () = Bus2::<F, T>::OK;
        // the mask was written by a VALU compare: same 2 wait states as vcc
        if let Some(w) = self.write_of(mask.id) {
            self.pad_after(w.0, VCC_TO_CNDMASK);
        }
        self.push(MInst::new(MOp::VCndmaskM).def(dst.dst_id()).src(if_false.to_src()).src(if_true.to_src()).srcreg(mask.id));
        self.note_valu_write(dst.dst_id());
    }

    // ---- hazard-sensitive reads -----------------------------------------------------------------------------------------------------
    /// `v_readfirstlane_b32 sdst, vsrc`: the source must be `Ready` (>= 2 wait states after its VALU write).
    pub fn v_readfirstlane_b32(&mut self, dst: &mut Sgpr<1>, src: Ready<'_, 1>) {
        self.push(MInst::new(MOp::VReadFirstLane).def(dst.id).srcreg(src.id));
    }

    // ---- matrix ----------------------------------------------------------------------------------------------------------------------
    /// `v_mfma_scale_f32_32x32x64_f8f6f4 acc, a, b, acc, sa, sb` (accumulate in place). `acc` stays mutably borrowed, and every
    /// operand borrowed, until the returned token is retired.
    pub fn v_mfma_scale_f32_32x32x64_f8f6f4<'a>(
        &mut self,
        acc: &'a mut Vgpr<16>,
        a: Ready<'a, 8>,
        b: Ready<'a, 8>,
        sa: Ready<'a, 1>,
        sb: Ready<'a, 1>,
    ) -> Mfma<'a> {
        self.make_ready_pub(acc.id); // the accumulator is read as C: a recent VALU write of it needs the 2 wait states too
        let issued = self.clock;
        self.push(MInst::new(MOp::VMfmaScale32x32x64Fp8).def(acc.id).srcreg(a.id).srcreg(b.id).srcreg(acc.id).srcreg(sa.id).srcreg(sb.id));
        self.note_write(acc.id, MATRIX_RESULT);
        Mfma { issued, acc: acc.id, _p: PhantomData }
    }

    /// The same with an implicit zero accumulator (`D = A x B`): `acc` is only written.
    pub fn v_mfma_scale_f32_32x32x64_f8f6f4_zero<'a>(&mut self, acc: &'a mut Vgpr<16>, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a> {
        let issued = self.clock;
        self.push(MInst::new(MOp::VMfmaScale32x32x64Fp8Z).def(acc.id).srcreg(a.id).srcreg(b.id).srcreg(sa.id).srcreg(sb.id));
        self.note_write(acc.id, MATRIX_RESULT);
        Mfma { issued, acc: acc.id, _p: PhantomData }
    }

    /// `v_smfmac_f32_16x16x128_fp8_fp8 acc, a_sparse, b, idx`: the accumulator C is `acc` itself (implicit in the ISA, explicit here).
    pub fn v_smfmac_f32_16x16x128_fp8_fp8<'a>(&mut self, acc: &'a mut Vgpr<4>, a: Ready<'a, 4>, b: Ready<'a, 8>, idx: Ready<'a, 1>) -> Smfmac<'a> {
        self.make_ready_pub(acc.id);
        let issued = self.clock;
        self.push(MInst::new(MOp::VSmfmac16x16x128Fp8).def(acc.id).srcreg(a.id).srcreg(b.id).srcreg(idx.id).srcreg(acc.id));
        self.note_write(acc.id, MATRIX_RESULT);
        Smfmac { issued, acc: acc.id, _p: PhantomData }
    }
}

impl<'a> Mfma<'a> {
    /// Dependent MFMA of the same opcode accumulating in place into the same register: hardware-interlocked, no wait needed.
    pub fn chain(self, s: &mut Seq, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a> {
        let issued = s.clock;
        let acc = self.acc;
        s.push(MInst::new(MOp::VMfmaScale32x32x64Fp8).def(acc).srcreg(a.id).srcreg(b.id).srcreg(acc).srcreg(sa.id).srcreg(sb.id));
        s.note_write(acc, MATRIX_RESULT);
        Mfma { issued, acc, _p: PhantomData }
    }
}

impl<'a> Smfmac<'a> {
    pub fn chain(self, s: &mut Seq, a: Ready<'a, 4>, b: Ready<'a, 8>, idx: Ready<'a, 1>) -> Smfmac<'a> {
        let issued = s.clock;
        let acc = self.acc;
        s.push(MInst::new(MOp::VSmfmac16x16x128Fp8).def(acc).srcreg(a.id).srcreg(b.id).srcreg(idx.id).srcreg(acc));
        s.note_write(acc, MATRIX_RESULT);
        Smfmac { issued, acc, _p: PhantomData }
    }
}