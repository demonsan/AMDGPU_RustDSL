//! Opcode templates. The operand types encode the encoding rules of gfx9/gfx950 VALU and matrix instructions:
//!
//! * `e32` forms: src0 may be a VGPR, SGPR, inline constant or literal (`Operand`); src1 must be a VGPR (`VgprOnly`).
//! * VOP3 / `e64` forms: any operand except a literal, and the constant bus allows ONE scalar source in total (checked at compile
//!   time through an associated constant: a violation is a readable compile error, not an assembler error).
//! * `v_cndmask_b32_e32` reads `vcc` through the constant bus, so it takes VGPR sources only (type-enforced by `VgprOnly`); the
//!   `_e64` form reads an SGPR pair as its mask, which uses the single bus slot, so its data operands must be bus-free.
//! * matrix instructions take `Ready` operands (the 2-wait VALU->matrix rule), keep their operands borrowed while in flight, OWN their
//!   accumulator and return a token that must be retired (20 wait states) or exited before the result can be touched.
//! * loads (`ds_read_*`, `global_load_*`) own their destination and return a `Pending` token; `wait` emits the partial `s_waitcnt`.

use crate::block::*;
use crate::regs::*;
use crate::seq::*;
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

fn inst(op: &'static str, dst: Option<u32>, srcs: Vec<ASrc>) -> AInst {
    AInst { op, dst, srcs, imm: 0 }
}

impl Seq {
    // ---- VALU, e32 -------------------------------------------------------------------------------------------------------------
    /// `v_mov_b32 dst, src0`.
    pub fn v_mov_b32<D: Dst<1>, A: Operand>(&mut self, dst: D, a: A) {
        let i = match a.to_src() {
            ASrc::Reg(r) => inst("VMovB32", Some(dst.dst_id()), vec![ASrc::Reg(r)]),
            ASrc::Imm(k) => AInst { op: "VMovB32", dst: Some(dst.dst_id()), srcs: vec![], imm: k },
        };
        self.push(i);
        self.note_valu_write(dst.dst_id());
    }

    /// `v_add_u32_e32 dst, src0, vsrc1`.
    pub fn v_add_u32<D: Dst<1>, A: Operand, B: VgprOnly>(&mut self, dst: D, a: A, b: B) {
        self.push(inst("VAddU32", Some(dst.dst_id()), vec![a.to_src(), b.to_src()]));
        self.note_valu_write(dst.dst_id());
    }

    /// `v_add_f32_e32 dst, src0, vsrc1`.
    pub fn v_add_f32<D: Dst<1>, A: Operand, B: VgprOnly>(&mut self, dst: D, a: A, b: B) {
        self.push(inst("VAddF32", Some(dst.dst_id()), vec![a.to_src(), b.to_src()]));
        self.note_valu_write(dst.dst_id());
    }

    /// `v_add_f32_e32 acc, src0, acc` (accumulate in place).
    pub fn v_add_f32_acc<A: Operand>(&mut self, acc: &mut Vgpr<1>, a: A) {
        self.push(inst("VAddF32", Some(acc.id), vec![a.to_src(), ASrc::Reg(acc.id)]));
        self.note_valu_write(acc.id);
    }

    // ---- VALU, VOP3 -------------------------------------------------------------------------------------------------------------
    /// `v_mul_lo_u32 dst, a, b` (VOP3: one scalar source at most, no literal).
    pub fn v_mul_lo_u32<D: Dst<1>, A: Operand, B: Operand>(&mut self, dst: D, a: A, b: B) {
        let () = Bus1::<A, B>::OK;
        self.push(inst("VMulLoU32", Some(dst.dst_id()), vec![a.to_src(), b.to_src()]));
        self.note_valu_write(dst.dst_id());
    }

    // ---- compare / select through vcc ---------------------------------------------------------------------------------------------
    /// `v_cmp_lt_i32_e32 vcc, src0, vsrc1`: `Vcc<Free>` in, `Vcc<Live>` out.
    pub fn v_cmp_lt_i32<A: Operand, B: VgprOnly>(&mut self, _vcc: Vcc<Free>, a: A, b: B) -> Vcc<Live> {
        self.push(inst("VCmpLtI32", None, vec![a.to_src(), b.to_src()]));
        self.vcc_written();
        Vcc { _s: PhantomData }
    }

    /// `v_cndmask_b32_e32 dst, if_false, if_true, vcc`: VGPR sources only (vcc already uses the constant bus). Consumes the mask.
    pub fn v_cndmask_b32<D: Dst<1>, F: VgprOnly, T: VgprOnly>(&mut self, _vcc: Vcc<Live>, dst: D, if_false: F, if_true: T) -> Vcc<Free> {
        self.vcc_ready();
        self.push(inst("VCndmask", Some(dst.dst_id()), vec![if_false.to_src(), if_true.to_src()]));
        self.note_valu_write(dst.dst_id());
        Vcc { _s: PhantomData }
    }

    // ---- compare / select through an SGPR pair (several masks can be live) ---------------------------------------------------------
    /// `v_cmp_lt_i32_e64 s[a:b], x, y` (VOP3).
    pub fn v_cmp_lt_i32_e64<A: Operand, B: Operand>(&mut self, mask: &mut Sgpr<2>, a: A, b: B) {
        let () = Bus1::<A, B>::OK;
        self.push(inst("VCmpMaskLtI32", Some(mask.id), vec![a.to_src(), b.to_src()]));
        self.note_valu_write(mask.id);
    }

    /// `v_cndmask_b32_e64 dst, if_false, if_true, s[a:b]`.
    pub fn v_cndmask_b32_e64<D: Dst<1>, F: Operand, T: Operand>(&mut self, dst: D, if_false: F, if_true: T, mask: &Sgpr<2>) {
        let () = Bus2::<F, T>::OK;
        if let Some(w) = self.write_of(mask.id) {
            self.pad_after(w.0, VCC_TO_CNDMASK);
        }
        self.push(inst("VCndmaskM", Some(dst.dst_id()), vec![if_false.to_src(), if_true.to_src(), ASrc::Reg(mask.id)]));
        self.note_valu_write(dst.dst_id());
    }

    // ---- hazard-sensitive reads -----------------------------------------------------------------------------------------------------
    /// `v_readfirstlane_b32 sdst, vsrc`: the source must be `Ready` (>= 2 wait states after its VALU write).
    pub fn v_readfirstlane_b32(&mut self, dst: &mut Sgpr<1>, src: Ready<'_, 1>) {
        self.push(inst("VReadFirstLane", Some(dst.id), vec![ASrc::Reg(src.id)]));
    }

    // ---- matrix ----------------------------------------------------------------------------------------------------------------------
    /// `v_mfma_scale_f32_32x32x64_f8f6f4 acc, a, b, acc, sa, sb` (accumulate in place). The accumulator moves into the returned token and
    /// every operand stays borrowed until the token is retired / exited.
    pub fn v_mfma_scale_f32_32x32x64_f8f6f4<'a>(&mut self, acc: Vgpr<16>, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a, OP_MFMA_SCALE> {
        self.make_ready_pub(acc.id); // the accumulator is read as C: a recent VALU write of it needs the 2 wait states too
        let issued = self.clock;
        self.push(inst("VMfmaScale32x32x64Fp8", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(acc.id), ASrc::Reg(sa.id), ASrc::Reg(sb.id)]));
        self.note_write(acc.id, MATRIX_RESULT);
        Mfma { issued, acc: Some(acc), poison: self.poison.clone(), _p: PhantomData }
    }

    /// The same with an implicit zero accumulator (`D = A x B`): `acc` is only written.
    pub fn v_mfma_scale_f32_32x32x64_f8f6f4_zero<'a>(&mut self, acc: Vgpr<16>, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a, OP_MFMA_SCALE_ZERO> {
        let issued = self.clock;
        self.push(inst("VMfmaScale32x32x64Fp8Z", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(sa.id), ASrc::Reg(sb.id)]));
        self.note_write(acc.id, MATRIX_RESULT);
        Mfma { issued, acc: Some(acc), poison: self.poison.clone(), _p: PhantomData }
    }

    /// `v_smfmac_f32_16x16x128_fp8_fp8 acc, a_sparse, b, idx`: the accumulator C is `acc` itself (implicit in the ISA, explicit here).
    pub fn v_smfmac_f32_16x16x128_fp8_fp8<'a>(&mut self, acc: Vgpr<4>, a: Ready<'a, 4>, b: Ready<'a, 8>, idx: Ready<'a, 1>) -> Smfmac<'a, OP_SMFMAC_FP8> {
        self.make_ready_pub(acc.id);
        let issued = self.clock;
        self.push(inst("VSmfmac16x16x128Fp8", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(idx.id), ASrc::Reg(acc.id)]));
        self.note_write(acc.id, MATRIX_RESULT);
        Smfmac { issued, acc: Some(acc), poison: self.poison.clone(), _p: PhantomData }
    }

    // ---- loads (Pending tokens) -----------------------------------------------------------------------------------------------------
    /// `ds_read_b128 dst, vaddr offset:N`: LDS read (lgkmcnt queue). The destination moves into the token.
    pub fn ds_read_b128(&mut self, dst: Vgpr<4>, addr: &Vgpr<1>, offset: u16) -> Pending<Vgpr<4>> {
        self.push(AInst { op: "LdsReadB128", dst: Some(dst.id), srcs: vec![ASrc::Reg(addr.id)], imm: offset as i64 });
        self.clear_write(dst.id);
        self.issue_load(Queue::Lgkm, dst)
    }

    /// `ds_read_b32 dst, vaddr offset:N`.
    pub fn ds_read_b32(&mut self, dst: Vgpr<1>, addr: &Vgpr<1>, offset: u16) -> Pending<Vgpr<1>> {
        self.push(AInst { op: "LdsReadB32", dst: Some(dst.id), srcs: vec![ASrc::Reg(addr.id)], imm: offset as i64 });
        self.clear_write(dst.id);
        self.issue_load(Queue::Lgkm, dst)
    }

    /// `global_load_dwordx4 dst, vaddr[2], off offset:N` (vmcnt queue).
    pub fn global_load_dwordx4(&mut self, dst: Vgpr<4>, addr: &Vgpr<2>, offset: i32) -> Pending<Vgpr<4>> {
        self.push(AInst { op: "GlobalLoadB128", dst: Some(dst.id), srcs: vec![ASrc::Reg(addr.id)], imm: offset as i64 });
        self.clear_write(dst.id);
        self.issue_load(Queue::Vm, dst)
    }

    /// `ds_write_b32 vaddr, vdata offset:N`: occupies a lgkmcnt slot (no value comes back).
    pub fn ds_write_b32(&mut self, addr: &Vgpr<1>, data: &Vgpr<1>, offset: u16) {
        self.push(AInst { op: "LdsWriteB32", dst: None, srcs: vec![ASrc::Reg(addr.id), ASrc::Reg(data.id)], imm: offset as i64 });
        self.issue_store(Queue::Lgkm);
    }
}

impl<'a> Mfma<'a, OP_MFMA_SCALE> {
    /// Dependent MFMA of the SAME accumulate opcode accumulating in place into the same register: hardware-interlocked, no wait needed.
    /// (Not defined for the `_zero` form: a different opcode needs the full result latency, i.e. `retire`.)
    pub fn chain(mut self, s: &mut Seq, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a, OP_MFMA_SCALE> {
        let issued = s.clock;
        let acc = self.acc.take().expect("accumulator");
        s.push(inst("VMfmaScale32x32x64Fp8", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(acc.id), ASrc::Reg(sa.id), ASrc::Reg(sb.id)]));
        s.note_write(acc.id, MATRIX_RESULT);
        Mfma { issued, acc: Some(acc), poison: s.poison.clone(), _p: PhantomData }
    }
}

impl<'a> Smfmac<'a, OP_SMFMAC_FP8> {
    pub fn chain(mut self, s: &mut Seq, a: Ready<'a, 4>, b: Ready<'a, 8>, idx: Ready<'a, 1>) -> Smfmac<'a, OP_SMFMAC_FP8> {
        let issued = s.clock;
        let acc = self.acc.take().expect("accumulator");
        s.push(inst("VSmfmac16x16x128Fp8", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(idx.id), ASrc::Reg(acc.id)]));
        s.note_write(acc.id, MATRIX_RESULT);
        Smfmac { issued, acc: Some(acc), poison: s.poison.clone(), _p: PhantomData }
    }
}