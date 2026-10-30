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

struct Bus3<A, B, C>(PhantomData<(A, B, C)>);
impl<A: Operand, B: Operand, C: Operand> Bus3<A, B, C> {
    const OK: () = assert!(
        A::BUS + B::BUS + C::BUS <= 1 && A::VOP3_OK && B::VOP3_OK && C::VOP3_OK,
        "gfx9 VOP3: at most one SGPR/literal source per instruction (constant bus limit 1) and no 32-bit literal"
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

    // ---- compare / select through vcc ---------------------------------------------------------------------------------------------
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
        Mfma { issued, lat: MATRIX_RESULT, acc: Some(acc), poison: self.poison.clone(), _p: PhantomData }
    }

    /// Scaled MFMA with an EXPLICIT accumulator input: `dst = A x B + C` (`v_mfma_scale ... dst, a, b, c, sa, sb`, the persistent-bias form: C is a
    /// register the sequence only READS, dst is a different register that the token owns). Same opcode as the accumulate form, so `.chain()` (in place
    /// on dst) works afterwards. `c` is a `Ready` proof: it is borrowed for the life of the token (cannot be overwritten while the MFMA reads it), and a C
    /// produced by a matrix op is only reachable after `retire` (20 wait states): the verifier rejects a C read less than 20 states after its producer
    /// (the interlocked same-opcode exemption applies only to the in-place chain dst == C).
    pub fn v_mfma_scale_f32_32x32x64_f8f6f4_c<'a>(&mut self, dst: Vgpr<16>, c: Ready<'a, 16>, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a, OP_MFMA_SCALE> {
        let issued = self.clock;
        self.push(inst("VMfmaScale32x32x64Fp8", Some(dst.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(c.id), ASrc::Reg(sa.id), ASrc::Reg(sb.id)]));
        self.note_write(dst.id, MATRIX_RESULT);
        Mfma { issued, lat: MATRIX_RESULT, acc: Some(dst), poison: self.poison.clone(), _p: PhantomData }
    }

    /// The same with an implicit zero accumulator (`D = A x B`): `acc` is only written.
    pub fn v_mfma_scale_f32_32x32x64_f8f6f4_zero<'a>(&mut self, acc: Vgpr<16>, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a, OP_MFMA_SCALE_ZERO> {
        let issued = self.clock;
        self.push(inst("VMfmaScale32x32x64Fp8Z", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(sa.id), ASrc::Reg(sb.id)]));
        self.note_write(acc.id, MATRIX_RESULT);
        Mfma { issued, lat: MATRIX_RESULT, acc: Some(acc), poison: self.poison.clone(), _p: PhantomData }
    }

    /// `v_mfma_f32_32x32x8_f16 acc, a, b, 0` (the bias MFMA): implicit zero accumulator, `A` and `B` are 2-dword f16 fragments. A different opcode from
    /// the scaled MFMA: a scaled MFMA that accumulates into this result needs the full result latency (`retire`: 12 wait states), not the in-place chain.
    pub fn v_mfma_f32_32x32x8_f16_zero<'a>(&mut self, acc: Vgpr<16>, a: Ready<'a, 2>, b: Ready<'a, 2>) -> Mfma<'a, OP_MFMA_F16_ZERO> {
        let issued = self.clock;
        self.push(inst("VMfma32x32x8F16Z", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id)]));
        self.note_write(acc.id, MATRIX_RESULT_F16);
        Mfma { issued, lat: MATRIX_RESULT_F16, acc: Some(acc), poison: self.poison.clone(), _p: PhantomData }
    }

    /// `v_smfmac_f32_16x16x128_fp8_fp8 acc, a_sparse, b, idx`: the accumulator C is `acc` itself (implicit in the ISA, explicit here).
    pub fn v_smfmac_f32_16x16x128_fp8_fp8<'a>(&mut self, acc: Vgpr<4>, a: Ready<'a, 4>, b: Ready<'a, 8>, idx: Ready<'a, 1>) -> Smfmac<'a, OP_SMFMAC_FP8> {
        self.make_ready_pub(acc.id);
        let issued = self.clock;
        self.push(inst("VSmfmac16x16x128Fp8", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(idx.id), ASrc::Reg(acc.id)]));
        self.note_write(acc.id, MATRIX_RESULT);
        Smfmac { issued, acc: Some(acc), poison: self.poison.clone(), _p: PhantomData }
    }

    // ---- loads / DS (Pending tokens; offsets are compile-time constants with alignment checks) ---------------------------------------
    /// `ds_read_b128 dst, vaddr offset:OFF`: LDS read (lgkmcnt queue). OFF is 16-byte aligned (compile-time). The destination moves into the token.
    pub fn ds_read_b128<const OFF: u32, D: Dst<4>>(&mut self, dst: D, addr: &impl Readable<1>) -> Pending<D> {
        let () = DsOff::<OFF, 16>::OK;
        self.push(AInst { op: "LdsReadB128", dst: Some(dst.dst_id()), srcs: vec![ASrc::Reg(addr.reg_id())], imm: OFF as i64 });
        self.clear_write(dst.dst_id());
        self.issue_load(Queue::Lgkm, dst)
    }

    /// `ds_read_b64_tr_b8 dst, vaddr offset:OFF`: transposed 8-bit read (2 dwords). OFF is 8-byte aligned.
    pub fn ds_read_b64_tr_b8<const OFF: u32, D: Dst<2>>(&mut self, dst: D, addr: &impl Readable<1>) -> Pending<D> {
        let () = DsOff::<OFF, 8>::OK;
        self.push(AInst { op: "LdsReadB64TrB8", dst: Some(dst.dst_id()), srcs: vec![ASrc::Reg(addr.reg_id())], imm: OFF as i64 });
        self.clear_write(dst.dst_id());
        self.issue_load(Queue::Lgkm, dst)
    }

    /// `ds_read_b32 dst, vaddr offset:OFF` (OFF 4-byte aligned).
    pub fn ds_read_b32<const OFF: u32, D: Dst<1>>(&mut self, dst: D, addr: &impl Readable<1>) -> Pending<D> {
        let () = DsOff::<OFF, 4>::OK;
        self.push(AInst { op: "LdsReadB32", dst: Some(dst.dst_id()), srcs: vec![ASrc::Reg(addr.reg_id())], imm: OFF as i64 });
        self.clear_write(dst.dst_id());
        self.issue_load(Queue::Lgkm, dst)
    }

    /// `ds_write_b32 vaddr, vdata offset:OFF`: occupies a lgkmcnt slot (no value comes back).
    pub fn ds_write_b32<const OFF: u32>(&mut self, addr: &impl Readable<1>, data: &impl Readable<1>) {
        let () = DsOff::<OFF, 4>::OK;
        self.push(AInst { op: "LdsWriteB32", dst: None, srcs: vec![ASrc::Reg(addr.reg_id()), ASrc::Reg(data.reg_id())], imm: OFF as i64 });
        self.issue_store(Queue::Lgkm);
    }

    /// `ds_write_b128 vaddr, vdata[4] offset:OFF` (OFF 16-byte aligned).
    pub fn ds_write_b128<const OFF: u32>(&mut self, addr: &impl Readable<1>, data: &impl Readable<4>) {
        let () = DsOff::<OFF, 16>::OK;
        self.push(AInst { op: "LdsWriteB128", dst: None, srcs: vec![ASrc::Reg(addr.reg_id()), ASrc::Reg(data.reg_id())], imm: OFF as i64 });
        self.issue_store(Queue::Lgkm);
    }

    /// `global_load_dwordx4 dst, vaddr[2], off offset:N` (vmcnt queue).
    pub fn global_load_dwordx4(&mut self, dst: Vgpr<4>, addr: &impl Readable<2>, offset: i32) -> Pending<Vgpr<4>> {
        self.push(AInst { op: "GlobalLoadB128", dst: Some(dst.id), srcs: vec![ASrc::Reg(addr.reg_id())], imm: offset as i64 });
        self.clear_write(dst.id);
        self.issue_load(Queue::Vm, dst)
    }

    // ---- transcendental ops: the result needs 4 wait states (TRANS_LATENCY 5 slots) before any reader ------------------------------------
    /// `v_exp_f32 dst, src`: returns a token; `settle` pads the missing wait states and gives the register back.
    pub fn v_exp_f32<A: Operand>(&mut self, dst: Vgpr<1>, a: A) -> Trans {
        self.trans("VExpF32", dst, a)
    }
    pub fn v_rcp_f32<A: Operand>(&mut self, dst: Vgpr<1>, a: A) -> Trans {
        self.trans("VRcpF32", dst, a)
    }
    pub fn v_log_f32<A: Operand>(&mut self, dst: Vgpr<1>, a: A) -> Trans {
        self.trans("VLogF32", dst, a)
    }

    fn trans<A: Operand>(&mut self, op: &'static str, dst: Vgpr<1>, a: A) -> Trans {
        // a packed-fp32 result feeding a transcendental op needs 2 slots first (hazards.rs `pkw`)
        if let ASrc::Reg(r) = a.to_src() {
            self.pad_for_pk(r);
        }
        let issued = self.clock;
        self.push(inst(op, Some(dst.id), vec![a.to_src()]));
        Trans { issued, reg: Some(dst), poison: self.poison.clone() }
    }

    // ---- packed (VOP3P) fp32 ops on register pairs ------------------------------------------------------------------------------------
    /// `v_pk_add_f32 dst[2], a[2], b[2]`.
    pub fn v_pk_add_f32<D: Dst<2>, A: Src<2>, B: Src<2>>(&mut self, dst: D, a: A, b: B) {
        self.push(inst("VPkAddF32", Some(dst.dst_id()), vec![ASrc::Reg(a.src_id()), ASrc::Reg(b.src_id())]));
        self.note_valu_write(dst.dst_id());
        self.note_pk_write(dst.dst_id());
    }
    /// `v_pk_mul_f32 dst[2], a[2], b[2]`.
    pub fn v_pk_mul_f32<D: Dst<2>, A: Src<2>, B: Src<2>>(&mut self, dst: D, a: A, b: B) {
        self.push(inst("VPkMulF32", Some(dst.dst_id()), vec![ASrc::Reg(a.src_id()), ASrc::Reg(b.src_id())]));
        self.note_valu_write(dst.dst_id());
        self.note_pk_write(dst.dst_id());
    }

    // ---- permlane32 swap: swaps the upper half of `a` with the lower half of `b` across the wave; both registers are rewritten ------------
    pub fn v_permlane32_swap_b32(&mut self, a: &mut Vgpr<1>, b: &mut Vgpr<1>) {
        // both operands are VALU-written inputs of a hazard-sensitive reader: 2 wait states first
        self.make_ready_pub(a.id);
        self.make_ready_pub(b.id);
        self.push(inst("VPermlane32Swap", Some(a.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id)]));
        self.note_valu_write(a.id);
        self.note_valu_write(b.id);
    }
}

/// A transcendental result in flight (5-slot latency). Owns the destination register; `settle` pads exactly the missing wait states.
#[must_use = "a transcendental result must be settled before it is used"]
pub struct Trans {
    issued: u32,
    reg: Option<Vgpr<1>>,
    poison: Poison,
}

impl Trans {
    pub fn settle(mut self, s: &mut Seq) -> Vgpr<1> {
        let r = self.reg.take().expect("register");
        s.pad_after(self.issued, TRANS_RESULT);
        r
    }
}

impl Drop for Trans {
    fn drop(&mut self) {
        if self.reg.is_some() {
            self.poison.borrow_mut().push("a transcendental result was abandoned without settle()".to_string());
        }
    }
}

struct DsOff<const OFF: u32, const A: u32>;
impl<const OFF: u32, const A: u32> DsOff<OFF, A> {
    const OK: () = assert!(OFF % A == 0 && OFF <= 65535, "ds offset must be aligned to the access size and fit the 16-bit offset field");
}
impl<'a> Mfma<'a, OP_MFMA_SCALE> {
    /// Dependent MFMA of the SAME accumulate opcode accumulating in place into the same register: hardware-interlocked, no wait needed.
    /// (Not defined for the `_zero` form: a different opcode needs the full result latency, i.e. `retire`.)
    pub fn chain(mut self, s: &mut Seq, a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>) -> Mfma<'a, OP_MFMA_SCALE> {
        let issued = s.clock;
        let acc = self.acc.take().expect("accumulator");
        s.push(inst("VMfmaScale32x32x64Fp8", Some(acc.id), vec![ASrc::Reg(a.id), ASrc::Reg(b.id), ASrc::Reg(acc.id), ASrc::Reg(sa.id), ASrc::Reg(sb.id)]));
        s.note_write(acc.id, MATRIX_RESULT);
        Mfma { issued, lat: MATRIX_RESULT, acc: Some(acc), poison: s.poison.clone(), _p: PhantomData }
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

include!(concat!(env!("OUT_DIR"), "/generated_ops.rs"));

/// Hand-written templates (they carry tokens or type-state): (method, MOp key, kind).
pub const HAND_WRITTEN_OPS: &[(&str, &str, &str)] = &[
    ("v_mov_b32", "VMovB32", "vop1 (immediate form)"),
    ("v_cmp_lt_i32_e64", "VCmpMaskLtI32", "vop3 compare into an SGPR pair (own-backend op)"),
    ("v_cndmask_b32", "VCndmask", "vcc type-state"),
    ("v_cndmask_b32_e64", "VCndmaskM", "SGPR-pair mask"),
    ("v_readfirstlane_b32", "VReadFirstLane", "Ready operand"),
    ("v_mfma_scale_f32_32x32x64_f8f6f4", "VMfmaScale32x32x64Fp8", "matrix token (chainable)"),
    ("v_mfma_scale_f32_32x32x64_f8f6f4_zero", "VMfmaScale32x32x64Fp8Z", "matrix token"),
    ("v_mfma_f32_32x32x8_f16_zero", "VMfma32x32x8F16Z", "matrix token (12-state result)"),
    ("v_mfma_scale_f32_32x32x64_f8f6f4_c", "VMfmaScale32x32x64Fp8", "matrix token with explicit C operand (dst != C)"),
    ("v_smfmac_f32_16x16x128_fp8_fp8", "VSmfmac16x16x128Fp8", "matrix token (chainable)"),
    ("ds_read_b128", "LdsReadB128", "Pending (lgkmcnt)"),
    ("ds_read_b32", "LdsReadB32", "Pending (lgkmcnt)"),
    ("ds_write_b32", "LdsWriteB32", "lgkm slot"),
    ("global_load_dwordx4", "GlobalLoadB128", "Pending (vmcnt)"),
    ("v_add_f32_acc", "VAddF32", "in-place accumulate"),
    ("v_writesub (zeroed)", "VWriteSub", "register init"),
    ("ds_read_b64_tr_b8", "LdsReadB64TrB8", "Pending (lgkmcnt), 8-byte aligned offset"),
    ("ds_write_b128", "LdsWriteB128", "lgkm slot, 16-byte aligned offset"),
    ("v_exp_f32 / v_rcp_f32 / v_log_f32", "VExpF32", "Trans token (4 wait states)"),
    ("v_rcp_f32", "VRcpF32", "Trans token"),
    ("v_log_f32", "VLogF32", "Trans token"),
    ("v_pk_add_f32", "VPkAddF32", "VOP3P pair, pk->trans rule"),
    ("v_pk_mul_f32", "VPkMulF32", "VOP3P pair, pk->trans rule"),
    ("v_permlane32_swap_b32", "VPermlane32Swap", "in-place pair, 2-state reader rule"),
    ("v_cvt_scalef32_pk_fp8_bf16 lo", "VCvtPkFp8Lo", "partial-write type-state Pw"),
    ("v_cvt_scalef32_pk_fp8_bf16 hi", "VCvtPkFp8Hi", "partial-write type-state Pw"),
    ("v_cvt_pk_fp8_f32 lo", "VCvtPkFp8F32", "partial-write type-state Pw"),
    ("v_cvt_pk_fp8_f32 hi", "VCvtPkFp8F32Hi", "partial-write type-state Pw"),
];
