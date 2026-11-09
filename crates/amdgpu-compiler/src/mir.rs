//! Machine IR for the gfx950 backend.
//!
//! Registers are still virtual here: allocation happens in `regalloc`. Control
//! flow is a flat list of blocks with explicit terminators so that layout never
//! changes meaning. 64-bit pointers are a single `V2`/`S2` virtual register
//! whose physical base must be even.

use std::collections::HashMap;

pub type VReg = u32;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum RClass {
    V1,
    V2,
    V4,
    V8,
    V16,
    S1,
    S2,
    /// Four consecutive SGPRs (a MUBUF buffer resource descriptor), 4-aligned.
    S4,
    /// Accumulation VGPRs (gfx90a+/gfx950 unified file: arch VGPRs v0..accum_offset-1, AGPRs a0.. above them). Written by MFMA with
    /// cc_cd, by ds_read/loads, and by _accvgpr_write; read by MFMA (any operand), stores and _accvgpr_read, never by VALU.
    A1,
    A2,
    A4,
    A8,
    A16,
}

/// Allocated AGPR registers are stored in Alloc::phys as AGPR_TAG + index so one table serves both files.
pub const AGPR_TAG: u32 = 0x4000_0000;

pub fn is_agpr_phys(p: u32) -> bool {
    p >= AGPR_TAG
}

impl RClass {
    pub fn dwords(self) -> u32 {
        match self {
            RClass::V2 | RClass::S2 | RClass::A2 => 2,
            RClass::V4 | RClass::S4 | RClass::A4 => 4,
            RClass::V8 | RClass::A8 => 8,
            RClass::V16 | RClass::A16 => 16,
            _ => 1,
        }
    }

    /// Arch-VGPR register classes share one physical file (AGPR classes are a separate file: is_agpr).
    pub fn is_vector(self) -> bool {
        matches!(self, RClass::V1 | RClass::V2 | RClass::V4 | RClass::V8 | RClass::V16)
    }

    pub fn is_agpr(self) -> bool {
        matches!(self, RClass::A1 | RClass::A2 | RClass::A4 | RClass::A8 | RClass::A16)
    }

    pub fn is_scalar(self) -> bool {
        matches!(self, RClass::S1 | RClass::S2 | RClass::S4)
    }

    /// The vector-register class of the same width (V* <-> A*).
    pub fn to_agpr(self) -> Option<RClass> {
        Some(match self {
            RClass::V1 | RClass::A1 => RClass::A1,
            RClass::V2 | RClass::A2 => RClass::A2,
            RClass::V4 | RClass::A4 => RClass::A4,
            RClass::V8 | RClass::A8 => RClass::A8,
            RClass::V16 | RClass::A16 => RClass::A16,
            _ => return None,
        })
    }

    pub fn to_vgpr(self) -> Option<RClass> {
        Some(match self {
            RClass::V1 | RClass::A1 => RClass::V1,
            RClass::V2 | RClass::A2 => RClass::V2,
            RClass::V4 | RClass::A4 => RClass::V4,
            RClass::V8 | RClass::A8 => RClass::V8,
            RClass::V16 | RClass::A16 => RClass::V16,
            _ => return None,
        })
    }

    /// Same width, ignoring the register file (V4 ~ A4).
    pub fn same_width_vector(self, o: RClass) -> bool {
        (self.is_vector() || self.is_agpr()) && (o.is_vector() || o.is_agpr()) && self.dwords() == o.dwords()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    Reg(VReg),
    Imm(i64),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MOp {
    // vector ALU
    VMovB32,
    VMovFromS,
    VMovFromSPair,
    VPtrAdd,
    VReadPhys,
    VReadFirstLane,
    /// `v_mov_b32_dpp d, s quad_perm:[..] row_mask:0xf bank_mask:0xf` (imm = the 8-bit quad_perm control)
    VMovDpp,
    VCvtPknormU16,
    VCvtPkFp8F32,
    VCvtPkBf16F32,
    VCvtPkFp8Lo,
    VCvtPkFp8Hi,
    /// `v_cvt_pk_fp8_f32` into the high 16 bits of its destination (low half preserved); srcs: a, b, old.
    VCvtPkFp8F32Hi,
    VPkAddF32,
    /// `v_permlane32_swap_b32` dst, src: swaps dst lanes 32..63 with src lanes 0..31 (both registers change); srcs: dst, src.
    VPermlane32Swap,
    VPkMulF32,
    VAddU32,
    VAddCoU32,
    VAddcCoU32,
    VAddF32,
    VSubF32,
    VMulF32,
    VMaxF32,
    VMax3F32,
    VPermB32,
    VPkSubU16,
    VFmaF32,
    VExpF32,
    VLogF32,
    VRcpF32,
    VCvtF32F16,
    VCvtF16F32,
    VAndB32,
    VMinI32,
    VPkFmaF32,
    // B1 64-bit address arithmetic
    SMulHiU32,
    VMadU64U32,
    // B2 cross-lane / wave-mask ops (readlane / writelane: imm = lane; ds_* lane permutes: imm = offset / swizzle pattern)
    VReadLane,
    VWriteLane,
    VMbcntLo,
    VMbcntHi,
    VBcntU32,
    VFfblB32,
    VFfbhU32,
    VCmpMaskNeU32,
    SMaskLo,
    SMaskHi,
    LdsBpermute,
    LdsPermute,
    LdsSwizzle,
    // P1 A4: DPP with a full control word (imm: dpp_ctrl | row_mask^0xf << 12 | bank_mask^0xf << 16 | bound_zero << 20; srcs end with the old value),
    // permlane16_swap
    VMovDppX,
    VAddF32Dpp,
    VSubF32Dpp,
    VMulF32Dpp,
    VMaxF32Dpp,
    VMinF32Dpp,
    VAddU32Dpp,
    VMaxI32Dpp,
    VMinI32Dpp,
    VMaxU32Dpp,
    VMinU32Dpp,
    VAndB32Dpp,
    VOrB32Dpp,
    VXorB32Dpp,
    VPermlane16Swap,
    // P1 A2 conversions (imm: word / byte select; partial writes keep the destination's other bits: srcs end with the old value)
    VCvtPkF32Fp8,
    VCvtF32Fp8,
    VCvtPkF32Bf8,
    VCvtF32Bf8,
    VCvtPkBf8F32,
    VCvtScalePkBf16Fp8,
    VCvtScalePkF16Fp8,
    VCvtScalePkBf16Bf8,
    VCvtScalePkF16Bf8,
    VCvtScalePkF32Fp8,
    VCvtScalePkF32Bf8,
    VCvtScalePkF32Fp4,
    VCvtScale2xPk16Fp6F32,
    VCvtScale2xPk16Bf6F32,
    VCvtScalePk32Fp6F16,
    VCvtScalePk32Fp6Bf16,
    VCvtScalePk32Bf6F16,
    VCvtScalePk32Bf6Bf16,
    VCvtScalePk32F16Fp6,
    VCvtScalePk32Bf16Fp6,
    VCvtScalePk32F16Bf6,
    VCvtScalePk32Bf16Bf6,
    VCvtScalePkBf16Fp4,
    VCvtScalePkF16Fp4,
    VCvtScalePkFp8F32,
    VCvtScalePkBf8F32,
    VCvtScalePkFp4F32,
    VCvtPkU8F32,
    VCvtPkI16I32,
    VCvtPkU16U32,
    // P1 A1 VALU (docs/op-coverage-tileops.md)
    VFloorF32,
    VCeilF32,
    VTruncF32,
    VRndneF32,
    VFractF32,
    VRsqF32,
    VSqrtF32,
    VSinF32,
    VCosF32,
    VCvtI32F32,
    VCvtU32F32,
    VCvtF32I32,
    VCvtF32U32,
    VCvtF32Ubyte0,
    VCvtF32Ubyte1,
    VCvtF32Ubyte2,
    VCvtF32Ubyte3,
    VMinF32,
    VMaxI32,
    VMaxU32,
    VMinU32,
    VSubU32,
    VMulU32U24,
    VAshrI32,
    VMulHiU32,
    VMulHiI32,
    VMin3F32,
    VMed3F32,
    VMin3I32,
    VMin3U32,
    VMax3I32,
    VMax3U32,
    VMed3I32,
    VMed3U32,
    VMadU32U24,
    VLshlAddU32,
    VAdd3U32,
    VBfeU32,
    VBfeI32,
    VBfiB32,
    VPkAddF16,
    VPkMulF16,
    VPkMaxF16,
    VPkMinF16,
    VPkFmaF16,
    VCmpLeF32,
    VCmpGtF32,
    VCmpGeF32,
    VCmpEqF32,
    VCmpNeF32,
    VCmpLgF32,
    VCmpUF32,
    VCmpOF32,
    VCmpLeI32,
    VCmpEqI32,
    VCmpLtU32,
    VCmpLeU32,
    VCmpGtU32,
    VCmpGeU32,
    VCmpEqU32,
    VOrB32,
    VXorB32,
    VLshrB32,
    VLshlB32,
    VMulLoU32,
    VCmpLtI32,
    VCmpGeI32,
    /// `v_cmp_gt_i32 vcc, a, b` (a > b): lets a scalar bound sit in src0 for `x < bound` (`sgpr`).
    VCmpGtI32,
    /// Compare into an SGPR PAIR instead of vcc (`v_cmp_*_e64 s[a:b], x, y`): several masks can be in flight, so the 2 wait states
    /// between a compare and its `v_cndmask` can be covered by the next compares (`maskreg`).
    VCmpMaskLtI32,
    /// `v_cndmask_b32_e64 d, a, b, s[mask]`: srcs = [false value, true value, mask pair].
    VCndmaskM,
    VCmpNeI32,
    VCmpLtF32,
    VCndmask,
    // fragments / matrix cores
    VWriteSub,
    VReadSub,
    VMfma16x16x16F16,
    VMfma32x32x8F16,
    /// 32x32x8 f16 MFMA with an inline-zero accumulator: D = A x B (srcs: a, b).
    VMfma32x32x8F16Z,
    VMfma16x16x32Fp8,
    VMfma32x32x16Fp8,
    VMfmaScale32x32x64Fp8,
    /// `v_mfma_scale_f32_16x16x128_f8f6f4`: dst V4 <- a (V8), b (V8), c (V4), scale_a (V1), scale_b (V1); `imm` = op_sel byte selects (bits 0..1 A, 4..5 B).
    VMfmaScale16x16x128Fp8,
    /// Scaled MFMA with an inline-zero accumulator (no C operand): D = A x B.
    VMfmaScale32x32x64Fp8Z,
    /// UNSCALED fp8 MFMA (_mfma_f32_32x32x64_f8f6f4): dst V16 <- a (V8), b (V8), c (V16); the Z form has no C (inline zero).
    VMfma32x32x64Fp8,
    VMfma32x32x64Fp8Z,
    /// Sparse fp8 MFMA `v_smfmac_f32_16x16x128_fp8_fp8`: the accumulator D (4 dwords) is also
    /// the input C, A (4 dwords) is the 2:4-compressed sparse operand, B (8 dwords) is dense,
    /// srcs[2] is the SRC2 sparse-index register (one dword) and srcs[3] is the accumulator C, which is the same register as D (implicit in the
    /// ISA, explicit here so every pass sees the read). Semantics:
    /// D += A_sparse(16x128, 64 nonzeros/row) x B(128x16).
    VSmfmac16x16x128Fp8,
    // unscaled f8f6f4 with format select (imm = cbsz << 8 | blgp << 12)
    VMfmaPlain16x16x128A4B4,
    VMfmaPlain16x16x128A4B4Z,
    VMfmaPlain16x16x128A4B6,
    VMfmaPlain16x16x128A4B6Z,
    VMfmaPlain16x16x128A4B8,
    VMfmaPlain16x16x128A4B8Z,
    VMfmaPlain16x16x128A6B4,
    VMfmaPlain16x16x128A6B4Z,
    VMfmaPlain16x16x128A6B6,
    VMfmaPlain16x16x128A6B6Z,
    VMfmaPlain16x16x128A6B8,
    VMfmaPlain16x16x128A6B8Z,
    VMfmaPlain16x16x128A8B4,
    VMfmaPlain16x16x128A8B4Z,
    VMfmaPlain16x16x128A8B6,
    VMfmaPlain16x16x128A8B6Z,
    VMfmaPlain16x16x128A8B8,
    VMfmaPlain16x16x128A8B8Z,
    VMfmaPlain32x32x64A4B4,
    VMfmaPlain32x32x64A4B4Z,
    VMfmaPlain32x32x64A4B6,
    VMfmaPlain32x32x64A4B6Z,
    VMfmaPlain32x32x64A4B8,
    VMfmaPlain32x32x64A4B8Z,
    VMfmaPlain32x32x64A6B4,
    VMfmaPlain32x32x64A6B4Z,
    VMfmaPlain32x32x64A6B6,
    VMfmaPlain32x32x64A6B6Z,
    VMfmaPlain32x32x64A6B8,
    VMfmaPlain32x32x64A6B8Z,
    VMfmaPlain32x32x64A8B4,
    VMfmaPlain32x32x64A8B4Z,
    VMfmaPlain32x32x64A8B6,
    VMfmaPlain32x32x64A8B6Z,
    VMfmaPlain32x32x64A8B8,
    VMfmaPlain32x32x64A8B8Z,
    // P1-d scaled f8f6f4 with format select: A / B register sizes 4 (fp4), 6 (fp6 / bf6), 8 (fp8 / bf8); imm = opsel | cbsz << 8 | blgp << 12
    VMfmaScale16x16x128A4B4,
    VMfmaScale16x16x128A4B4Z,
    VMfmaScale16x16x128A4B6,
    VMfmaScale16x16x128A4B6Z,
    VMfmaScale16x16x128A4B8,
    VMfmaScale16x16x128A4B8Z,
    VMfmaScale16x16x128A6B4,
    VMfmaScale16x16x128A6B4Z,
    VMfmaScale16x16x128A6B6,
    VMfmaScale16x16x128A6B6Z,
    VMfmaScale16x16x128A6B8,
    VMfmaScale16x16x128A6B8Z,
    VMfmaScale16x16x128A8B4,
    VMfmaScale16x16x128A8B4Z,
    VMfmaScale16x16x128A8B6,
    VMfmaScale16x16x128A8B6Z,
    VMfmaScale16x16x128A8B8,
    VMfmaScale16x16x128A8B8Z,
    VMfmaScale32x32x64A4B4,
    VMfmaScale32x32x64A4B4Z,
    VMfmaScale32x32x64A4B6,
    VMfmaScale32x32x64A4B6Z,
    VMfmaScale32x32x64A4B8,
    VMfmaScale32x32x64A4B8Z,
    VMfmaScale32x32x64A6B4,
    VMfmaScale32x32x64A6B4Z,
    VMfmaScale32x32x64A6B6,
    VMfmaScale32x32x64A6B6Z,
    VMfmaScale32x32x64A6B8,
    VMfmaScale32x32x64A6B8Z,
    VMfmaScale32x32x64A8B4,
    VMfmaScale32x32x64A8B4Z,
    VMfmaScale32x32x64A8B6,
    VMfmaScale32x32x64A8B6Z,
    VMfmaScale32x32x64A8B8,
    VMfmaScale32x32x64A8B8Z,
    // P1-d gfx950 double-rate bf16 / f16 and i8 MFMA (docs/op-coverage-tileops.md); `Z`: zero accumulator
    VMfma16x16x32Bf16,
    VMfma16x16x32Bf16Z,
    VMfma32x32x16Bf16,
    VMfma32x32x16Bf16Z,
    VMfma16x16x32F16,
    VMfma16x16x32F16Z,
    VMfma32x32x16F16,
    VMfma32x32x16F16Z,
    VMfma16x16x64I8,
    VMfma16x16x64I8Z,
    VMfma32x32x32I8,
    VMfma32x32x32I8Z,
    /// _accvgpr_read_b32 vD, aS: dst V1 <- src A1.
    VAccvgprRead,
    /// _accvgpr_write_b32 aD, src: dst A1 <- V1 / S1 / inline constant (Imm).
    VAccvgprWrite,
    /// _accvgpr_mov_b32 aD, aS: dst A1 <- A1.
    VAccvgprMov,
    // workgroup LDS
    LdsWriteB32,
    LdsReadB32,
    LdsWriteB128,
    LdsReadB128,
    /// `ds_read_b64`: dst V2 <- [addr + imm]
    LdsReadB64,
    /// `ds_write_b64`: srcs [addr, value V2], imm offset
    LdsWriteB64,
    LdsReadB64TrB8,
    SBarrier,
    /// `s_setprio` imm.
    SSetPrio,
    /// Scheduler directive (no instruction): `sched_group_barrier(mask, size, sync)`, imm = mask | size << 16 | sync << 32. Consumed (and removed) by `sched::phase`.
    SchedGroup,
    // global memory
    GlobalLoadU16,
    GlobalLoadF32,
    GlobalLoadB128,
    /// `global_load_dwordx4 v[d], v_off, s[base]`: scalar base + 32-bit vector offset (no 64-bit address arithmetic).
    /// Global -> LDS DMA (M0 = LDS base from srcs[1], an S1 register): srcs: ptr (V2), lds base (S1); imm = byte offset.
    GlobalLoadLdsB128,
    GlobalLoadLdsB32,
    /// scalar-base (SADDR) forms: srcs = [saddr pair, 32-bit vector offset, M0 source]
    GlobalLoadLdsB128S,
    GlobalLoadLdsB32S,
    // ---- MUBUF (buffer) memory ops. Loads: srcs = [rsrc (S4), voffset (V1 reg, or Imm = the `off` form), soffset (S1 reg or Imm)], imm = 12-bit offset.
    // Stores: srcs = [data, rsrc, voffset, soffset]. LDS DMA forms: srcs = [rsrc, voffset, soffset, M0 source (S1)] (data lands in LDS, no VGPR destination).
    BufferLoadDwordX1,
    BufferLoadDwordX2,
    BufferLoadDwordX4,
    BufferStoreDwordX1,
    BufferStoreDwordX2,
    BufferStoreDwordX4,
    BufferLoadDwordX4Lds,
    BufferLoadDwordX1Lds,
    /// Buffer resource descriptor: dst S4 <- base pointer (S2); imm = num_records. dword0 = base lo, dword1 = base hi & 0xffff, dword2 = num_records, dword3 = flags.
    SBufferRsrc,
    /// The same with a RUNTIME num_records: srcs = [base (S2), num_records (S1)].
    SBufferRsrcDyn,
    GlobalStoreU16,
    GlobalStoreF32,
    GlobalStoreF32x4,
    GlobalStoreF32x2,
    // P1 A3 memory (docs/op-coverage-tileops.md)
    GlobalLoadU8,
    GlobalLoadI8,
    GlobalLoadI16,
    GlobalLoadB64,
    GlobalLoadB96,
    GlobalLoadD16U8,
    GlobalLoadD16U8Hi,
    GlobalLoadD16I8,
    GlobalLoadD16I8Hi,
    GlobalLoadD16B16,
    GlobalLoadD16B16Hi,
    GlobalStoreB8,
    GlobalStoreB8Hi,
    GlobalStoreB16Hi,
    GlobalStoreB96,
    BufferLoadU8,
    BufferLoadI8,
    BufferLoadU16,
    BufferLoadI16,
    BufferLoadB96,
    BufferStoreB8,
    BufferStoreB16,
    BufferStoreB8Hi,
    BufferStoreB16Hi,
    BufferStoreB96,
    LdsReadU8,
    LdsReadI8,
    LdsReadU16,
    LdsReadI16,
    LdsReadB96,
    LdsReadB64TrB16,
    LdsReadD16U8,
    LdsReadD16U8Hi,
    LdsReadD16I8,
    LdsReadD16I8Hi,
    LdsReadD16B16,
    LdsReadD16B16Hi,
    LdsWriteB8,
    LdsWriteB16,
    LdsWriteB8Hi,
    LdsWriteB16Hi,
    LdsWriteB96,
    LdsRead2B32,
    LdsRead2B64,
    LdsRead2St64B32,
    LdsRead2St64B64,
    LdsWrite2B32,
    LdsWrite2B64,
    LdsWrite2St64B32,
    LdsWrite2St64B64,
    /// Device-scope (`sc1`) coherent loads / write-through stores, the returning device-scope atomic add (`sc0` = return) and the L2 fences
    /// of the last-arriver pattern (`buffer_wbl2 sc1` / `buffer_inv sc1`).
    GlobalLoadF32Sc1,
    GlobalStoreF32Sc1,
    GlobalAtomicAddRet,
    // buffer atomics (srcs: data, rsrc, voffset, soffset; imm offset) and packed 16-bit global atomics
    BufferAtomicAddU32,
    /// returning form (`sc0`): the old value lands in the data register (tied: srcs[0] == dst)
    BufferAtomicAddU32Ret,
    BufferAtomicAddF32,
    BufferAtomicPkAddBf16,
    BufferAtomicPkAddF16,
    GlobalAtomicPkAddBf16,
    GlobalAtomicPkAddBf16Rtn,
    GlobalAtomicPkAddF16,
    GlobalAtomicPkAddF16Rtn,
    // B3 atomics (LDS: srcs addr, data, imm = offset; global: srcs ptr, data, imm = offset; `Rtn`: returns the old value)
    LdsAtomicAddU32,
    LdsAtomicAddU32Rtn,
    LdsAtomicMaxU32,
    LdsAtomicMaxU32Rtn,
    LdsAtomicMinU32,
    LdsAtomicMinU32Rtn,
    LdsAtomicMaxI32,
    LdsAtomicMaxI32Rtn,
    LdsAtomicMinI32,
    LdsAtomicMinI32Rtn,
    LdsAtomicIncU32,
    LdsAtomicIncU32Rtn,
    LdsAtomicAddF32,
    LdsAtomicAddF32Rtn,
    GlobalAtomicAddU32,
    GlobalAtomicAddU32Rtn,
    GlobalAtomicMaxU32,
    GlobalAtomicMaxU32Rtn,
    GlobalAtomicMinU32,
    GlobalAtomicMinU32Rtn,
    GlobalAtomicMaxI32,
    GlobalAtomicMaxI32Rtn,
    GlobalAtomicMinI32,
    GlobalAtomicMinI32Rtn,
    GlobalAtomicIncU32,
    GlobalAtomicIncU32Rtn,
    GlobalAtomicAddF32,
    GlobalAtomicAddF32Rtn,
    BufferWbl2,
    BufferInv,
    // scalar ALU / memory
    SLoadDword,
    SLoadDwordx2,
    /// `s_load_dwordx4` of the kernarg segment into an SGPR quad (`kabatch`)
    SLoadDwordx4,
    /// `s_memtime s[d:d+1]` (64-bit core-clock counter; dst S2). SMEM: its result needs `lgkmcnt(0)`.
    SMemTime,
    /// `s_mov_b32 d, lo(src)`: the low dword of an S2 pair (dst S1, src S2).
    SLoPart,
    SReadPhys,
    SMovB32,
    /// `s_mov_b32 m0, src`: the explicit M0 write of the LDS-DMA loads that follow (sched::m0::explicit; own-only)
    SMovM0,
    /// `s_add_u32 m0, s0, s1`: the M0 write with the LDS base's add folded in (clobbers SCC)
    SAddM0,
    SAddU32,
    SAddcU32,
    SMulI32,
    SShlB32,
    SLshrB32,
    SAndB32,
    SXorB32,
    SCmpLtI32,
    /// `s_min_i32` (writes SCC)
    SMinI32,
    /// `s_cselect_b32 d, a, b`: d = SCC ? a : b (reads SCC; lowered right after its `s_cmp`)
    SCselectB32,
    /// `s_cselect_b64 d, a, b` on SGPR pairs (a wave-uniform pointer select; reads SCC)
    SCselectB64,
    /// two `s_cselect_b64` on the halves of SGPR quads (a wave-uniform buffer-descriptor select; reads SCC)
    SCselectB128,
    // misc
    SNop,
    SWaitVm,
    SWaitLgkm,
    SEndpgm,
}

impl MOp {
    /// Does this instruction consume `vcc` from the immediately preceding
    /// instruction? Used by the verifier to guarantee adjacency. (`op_attrs`: `consumes_vcc`)
    pub fn consumes_vcc(self) -> bool {
        crate::op_attrs::attrs(self).consumes_vcc
    }

    /// Does this instruction define `vcc`? (`op_attrs`: `defines_vcc`)
    pub fn defines_vcc(self) -> bool {
        crate::op_attrs::attrs(self).defines_vcc
    }
}

#[derive(Clone, Debug)]
pub struct MInst {
    pub op: MOp,
    pub dst: Option<VReg>,
    pub srcs: Vec<Src>,
    /// Offset, physical register index or label, depending on `op`.
    pub imm: i64,
    /// True for s_waitcnt inserted automatically by lowering (touch/flush_pending), false for
    /// explicit DSL waits. Only the LLVM-tail exporter looks at it (--llvm-tail-drop-auto-waits).
    pub auto: bool,
    /// Pinned instruction (safe-asm blocks built with `Seq::pin`): the scheduler keeps it in program order and treats a run of pinned
    /// instructions as a barrier (nothing moves into, out of or across it).
    pub pin: bool,
    /// LDS-DMA load whose M0 already holds its LDS base (the previous DMA of the block used the same source register): emit neither the s_mov_b32 m0 nor the
    /// s_nop 0 that must follow an M0 write (set by sched::m0::reuse).
    pub skip_m0: bool,
    /// The weave / stage region the instruction was lowered in (`MFunction::reg_hints.regions`, 1-based; 0 = none). No codegen effect.
    pub region: u16,
    /// Cache policy of a global / buffer access, LLVM CPol bits on gfx950: 1 = sc0, 2 = nt, 16 = sc1 (0 = default).
    pub cpol: u8,
    /// Diagnostics: the IR value (`ValueId`) whose lowering emitted this instruction (the last value-defining statement for the others);
    /// `u32::MAX` for instructions added by later passes (waits, nops, copies). Not used by any pass.
    pub origin: u32,
}

impl MInst {
    pub fn new(op: MOp) -> Self {
        MInst { op, dst: None, srcs: Vec::new(), imm: 0, auto: false, pin: false, skip_m0: false, region: 0, cpol: 0, origin: u32::MAX }
    }

    pub fn def(mut self, r: VReg) -> Self {
        self.dst = Some(r);
        self
    }

    pub fn src(mut self, s: Src) -> Self {
        self.srcs.push(s);
        self
    }

    pub fn srcreg(mut self, r: VReg) -> Self {
        self.srcs.push(Src::Reg(r));
        self
    }

    pub fn auto(mut self) -> Self {
        self.auto = true;
        self
    }

    pub fn cpol(mut self, c: u8) -> Self {
        self.cpol = c;
        self
    }
    pub fn imm(mut self, v: i64) -> Self {
        self.imm = v;
        self
    }

    pub fn uses(&self) -> Vec<VReg> {
        let mut v = Vec::new();
        for s in &self.srcs {
            if let Src::Reg(r) = s {
                v.push(*r);
            }
        }
        v
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pred {
    Scc0,
    Scc1,
    Vccnz,
    Vccz,
}

#[derive(Clone, Debug)]
pub enum Term {
    Goto(u32),
    CondGoto { on: Pred, target: u32, fall: u32 },
    Endpgm,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub label: u32,
    pub insts: Vec<MInst>,
    pub term: Term,
    pub pred: Option<Pred>,
}

impl Block {
    pub fn new(label: u32) -> Self {
        Block { label, insts: Vec::new(), term: Term::Endpgm, pred: None }
    }
}

#[derive(Clone, Debug)]
pub struct MArg {
    pub name: String,
    pub dsl_ty: amdgpu_dsl::DType,
    pub offset: u32,
    pub size: u32,
}

#[derive(Clone, Debug)]
pub struct MFunction {
    pub name: String,
    pub args: Vec<MArg>,
    pub kernarg_size: u32,
    pub workgroup_size: u32,
    /// Workgroup LDS bytes declared by the kernel.
    pub lds_bytes: u32,
    pub blocks: Vec<Block>,
    pub vregs: Vec<RClass>,
    /// (header label, latch label) pairs, used to extend live ranges that cross
    /// a loop backedge.
    pub loops: Vec<(u32, u32)>,
    /// Sub-register views: alias vreg -> (tuple vreg, dword offset). An alias owns no register;
    /// after allocation it is the tuple's physical register plus the offset, so extracting or
    /// inserting a lane of a fragment needs no move.
    pub alias: HashMap<VReg, (VReg, u8)>,
    /// Emit no s_branch to the immediately following block (set by the scheduler pipeline only).
    pub elide_fallthrough: bool,
    /// `edgepad` lowering option: block-end MFMA padding follows untouched successors and goes on the edges that need it (hazards.rs)
    pub edge_pad: bool,
    /// Explicit AGPR placement (mdgpu_pin_agpr): AGPR-class vreg -> first AGPR index. Honoured exactly by the allocator.
    pub agpr_pin: HashMap<VReg, u32>,
    /// Register planning (DSL `RegMark`s resolved to vregs) and the weave / stage regions instructions are tagged with (`MInst::region`).
    pub reg_hints: RegHints,
}

/// The kernel's register plan at vreg level (regalloc.rs honours and checks it). Empty for kernels without a plan.
#[derive(Clone, Debug, Default)]
pub struct RegHints {
    /// regions: (name, parent region id; 0 = none); `MInst::region` = 1-based index into this list
    pub regions: Vec<(String, u16)>,
    /// exact arch-VGPR placements (canonical vreg, base)
    pub pins_v: Vec<(VReg, u32)>,
    pub share: Vec<Vec<VReg>>,
    pub separate: Vec<Vec<VReg>>,
    /// (region name, arch VGPR cap, AGPR cap)
    pub caps: Vec<(String, Option<u32>, Option<u32>)>,
    pub rings: Vec<(String, Vec<VReg>)>,
    /// annotations whose target has no register (unused / folded value): kept for the report
    pub unresolved: Vec<String>,
    /// canonical vreg -> the DSL name of the value / variable it holds (`%v12`, `%r3`), for reports
    pub names: HashMap<VReg, String>,
    /// DSL name (`%v12`, `%r3`) -> the canonical vreg holding it (several names can share one register)
    pub by_name: HashMap<String, VReg>,
}

impl RegHints {
    /// Does the plan constrain the allocator at all?
    pub fn active(&self) -> bool {
        !(self.pins_v.is_empty() && self.share.is_empty() && self.separate.is_empty() && self.caps.is_empty() && self.rings.is_empty())
    }
}

impl MFunction {
    /// The vreg that owns the register behind `r` (itself unless `r` is an alias).
    pub fn canon(&self, r: VReg) -> VReg {
        self.alias.get(&r).map(|a| a.0).unwrap_or(r)
    }

    pub fn vreg_class(&self, r: VReg) -> RClass {
        self.vregs[r as usize]
    }

}

/// Light static checks that do not need physical registers.
pub fn verify(m: &MFunction) -> Vec<String> {
    let mut errs = Vec::new();
    let labels: Vec<u32> = m.blocks.iter().map(|b| b.label).collect();
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            for s in &inst.srcs {
                if let Src::Reg(r) = s {
                    if (*r as usize) >= m.vregs.len() {
                        errs.push(format!("block {bi} inst {ii}: use of unknown vreg r{r}"));
                    }
                }
            }
            if let Some(d) = inst.dst {
                if (d as usize) >= m.vregs.len() {
                    errs.push(format!("block {bi} inst {ii}: def of unknown vreg r{d}"));
                }
            }
            // AGPR legality: only the instructions listed in `agpr::legal_*` may touch the accumulation-register file, and an MFMA needs
            // vdst and srcC in the same file.
            for (k, s) in inst.srcs.iter().enumerate() {
                if let Src::Reg(r) = s {
                    if (*r as usize) < m.vregs.len() && m.vreg_class(*r).is_agpr() && !crate::agpr::legal_src(inst.op, k) {
                        errs.push(format!("block {bi} inst {ii} ({:?}): source {k} is an AGPR but this instruction cannot read one", inst.op));
                    }
                }
            }
            if let Some(d) = inst.dst {
                if (d as usize) < m.vregs.len() && m.vreg_class(d).is_agpr() && !crate::agpr::legal_dst(inst.op) {
                    errs.push(format!("block {bi} inst {ii} ({:?}): destination is an AGPR but this instruction cannot write one", inst.op));
                }
            }
            if crate::agpr::is_mfma(inst.op) && !matches!(inst.op, MOp::VMfmaScale32x32x64Fp8Z | MOp::VMfma32x32x8F16Z | MOp::VMfma32x32x64Fp8Z) {
                if let (Some(d), Some(Src::Reg(c))) = (inst.dst, inst.srcs.get(2)) {
                    if (d as usize) < m.vregs.len() && (*c as usize) < m.vregs.len() && m.vreg_class(d).is_agpr() != m.vreg_class(*c).is_agpr() {
                        errs.push(format!("block {bi} inst {ii} ({:?}): MFMA vdst and srcC are in different register files", inst.op));
                    }
                }
            }
            if matches!(inst.op, MOp::VAccvgprRead) && !(matches!(inst.srcs.first(), Some(Src::Reg(r)) if m.vreg_class(*r) == RClass::A1) && inst.dst.map_or(false, |d| m.vreg_class(d) == RClass::V1)) {
                errs.push(format!("block {bi} inst {ii}: v_accvgpr_read_b32 reads one A1 into one V1"));
            }
            if matches!(inst.op, MOp::VAccvgprWrite) && !inst.dst.map_or(false, |d| m.vreg_class(d) == RClass::A1) {
                errs.push(format!("block {bi} inst {ii}: v_accvgpr_write_b32 writes one A1"));
            }
            if matches!(inst.op, MOp::VAccvgprMov) && !(matches!(inst.srcs.first(), Some(Src::Reg(r)) if m.vreg_class(*r) == RClass::A1) && inst.dst.map_or(false, |d| m.vreg_class(d) == RClass::A1)) {
                errs.push(format!("block {bi} inst {ii}: v_accvgpr_mov_b32 moves one A1 to one A1"));
            }            // Class sanity for the operations with fixed register shapes.
            match inst.op {
                MOp::GlobalLoadU16 | MOp::GlobalLoadF32 => {
                    if let Some(d) = inst.dst {
                        if !m.vreg_class(d).same_width_vector(RClass::V1) {
                            errs.push(format!("block {bi} inst {ii}: load dst must be V1"));
                        }
                    }
                    if let Some(Src::Reg(a)) = inst.srcs.first() {
                        if m.vreg_class(*a) != RClass::V2 {
                            errs.push(format!("block {bi} inst {ii}: load base must be V2"));
                        }
                    } else {
                        errs.push(format!("block {bi} inst {ii}: load needs a base register"));
                    }
                }
                MOp::GlobalStoreU16 | MOp::GlobalStoreF32 | MOp::GlobalStoreF32x4 | MOp::GlobalStoreF32x2 => {
                    if let Some(Src::Reg(a)) = inst.srcs.first() {
                        if m.vreg_class(*a) != RClass::V2 {
                            errs.push(format!("block {bi} inst {ii}: store base must be V2"));
                        }
                    }
                }
                MOp::BufferLoadDwordX1 | MOp::BufferLoadDwordX2 | MOp::BufferLoadDwordX4 | MOp::BufferStoreDwordX1 | MOp::BufferStoreDwordX2 | MOp::BufferStoreDwordX4 | MOp::BufferLoadDwordX4Lds | MOp::BufferLoadDwordX1Lds => {
                    let ri = match inst.op {
                        MOp::BufferStoreDwordX1 | MOp::BufferStoreDwordX2 | MOp::BufferStoreDwordX4 => 1,
                        _ => 0,
                    };
                    match inst.srcs.get(ri) {
                        Some(Src::Reg(r)) if m.vreg_class(*r) == RClass::S4 => {}
                        _ => errs.push(format!("block {bi} inst {ii}: buffer access needs an S4 resource descriptor as source {ri}")),
                    }
                    if inst.imm < 0 || inst.imm > 4095 {
                        errs.push(format!("block {bi} inst {ii}: buffer immediate offset {} does not fit 12 bits", inst.imm));
                    }
                    if let Some(Src::Reg(v)) = inst.srcs.get(ri + 1) {
                        if m.vreg_class(*v) != RClass::V1 {
                            errs.push(format!("block {bi} inst {ii}: buffer voffset must be V1"));
                        }
                    }
                    if let Some(Src::Reg(s)) = inst.srcs.get(ri + 2) {
                        if m.vreg_class(*s) != RClass::S1 {
                            errs.push(format!("block {bi} inst {ii}: buffer soffset must be S1"));
                        }
                    }
                }
                MOp::SBufferRsrcDyn => {
                    if inst.dst.map_or(true, |d| m.vreg_class(d) != RClass::S4)
                        || !matches!(inst.srcs.first(), Some(Src::Reg(b)) if m.vreg_class(*b) == RClass::S2)
                        || !matches!(inst.srcs.get(1), Some(Src::Reg(n)) if m.vreg_class(*n) == RClass::S1)
                    {
                        errs.push(format!("block {bi} inst {ii}: s_buffer_rsrc_dyn builds an S4 from an S2 base and an S1 size"));
                    }
                }
                MOp::SBufferRsrc => {
                    if inst.dst.map_or(true, |d| m.vreg_class(d) != RClass::S4) || !matches!(inst.srcs.first(), Some(Src::Reg(b)) if m.vreg_class(*b) == RClass::S2) {
                        errs.push(format!("block {bi} inst {ii}: s_buffer_rsrc builds an S4 from an S2 base"));
                    }
                }
                MOp::SLoadDwordx2 => {
                    if let Some(d) = inst.dst {
                        if m.vreg_class(d) != RClass::S2 {
                            errs.push(format!("block {bi} inst {ii}: s_load_dwordx2 dst must be S2"));
                        }
                    }
                }
                MOp::SLoadDword => {
                    if let Some(d) = inst.dst {
                        if m.vreg_class(d) != RClass::S1 {
                            errs.push(format!("block {bi} inst {ii}: s_load_dword dst must be S1"));
                        }
                    }
                }
                MOp::VPtrAdd => {
                    if let Some(d) = inst.dst {
                        if m.vreg_class(d) != RClass::V2 {
                            errs.push(format!("block {bi} inst {ii}: ptradd dst must be V2"));
                        }
                    }
                }
                MOp::VMfma16x16x16F16 => {
                    if let Some(d) = inst.dst {
                        if !m.vreg_class(d).same_width_vector(RClass::V4) {
                            errs.push(format!("block {bi} inst {ii}: mfma dst must be V4"));
                        }
                    }
                    let want = [RClass::V2, RClass::V2, RClass::V4];
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r).same_width_vector(*w) => {}
                            Some(Src::Reg(r)) => errs.push(format!(
                                "block {bi} inst {ii}: mfma operand {k} is {:?}, expected {w:?}",
                                m.vreg_class(*r)
                            )),
                            _ => errs.push(format!(
                                "block {bi} inst {ii}: mfma operand {k} is not a register"
                            )),
                        }
                    }
                }
                MOp::VMfmaScale32x32x64Fp8 => {
                    // D/A/B/C are 16/8/8/16 dwords, plus two scalar scales.
                    let want = [RClass::V16, RClass::V8, RClass::V8, RClass::V16, RClass::V1, RClass::V1];
                    if let Some(d) = inst.dst {
                        if !m.vreg_class(d).same_width_vector(RClass::V16) {
                            errs.push(format!("block {bi} inst {ii}: scaled mfma dst must be V16"));
                        }
                    }
                    for (k, w) in want.iter().enumerate().skip(1) {
                        match inst.srcs.get(k - 1) {
                            Some(Src::Reg(r)) if m.vreg_class(*r).same_width_vector(*w) => {}
                            Some(Src::Reg(r)) => errs.push(format!(
                                "block {bi} inst {ii}: scaled mfma operand {k} is {:?}, expected {w:?}",
                                m.vreg_class(*r)
                            )),
                            _ => errs.push(format!(
                                "block {bi} inst {ii}: scaled mfma operand {k} is not a register"
                            )),
                        }
                    }
                }
                MOp::VMfmaScale16x16x128Fp8 => {
                    let want = [RClass::V8, RClass::V8, RClass::V4, RClass::V1, RClass::V1];
                    if !inst.dst.map_or(false, |d| m.vreg_class(d).same_width_vector(RClass::V4)) {
                        errs.push(format!("block {bi} inst {ii}: 16x16x128 scaled mfma dst must be V4"));
                    }
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if (k < 3 && m.vreg_class(*r).same_width_vector(*w)) || m.vreg_class(*r) == *w => {}
                            _ => errs.push(format!("block {bi} inst {ii}: 16x16x128 scaled mfma operand {k} must be a {w:?} register")),
                        }
                    }
                }
                MOp::VMfma32x32x64Fp8 | MOp::VMfma32x32x64Fp8Z => {
                    // unscaled fp8 MFMA: D/A/B/C are 16/8/8/16 dwords (the Z form has no C)
                    let want: &[RClass] = if inst.op == MOp::VMfma32x32x64Fp8 { &[RClass::V8, RClass::V8, RClass::V16] } else { &[RClass::V8, RClass::V8] };
                    if inst.dst.map_or(true, |d| !m.vreg_class(d).same_width_vector(RClass::V16)) {
                        errs.push(format!("block {bi} inst {ii}: fp8 mfma dst must be V16"));
                    }
                    if inst.srcs.len() != want.len() {
                        errs.push(format!("block {bi} inst {ii}: fp8 mfma takes {} register operands", want.len()));
                    }
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r).same_width_vector(*w) => {}
                            _ => errs.push(format!("block {bi} inst {ii}: fp8 mfma operand {k} must be a {w:?} register")),
                        }
                    }
                }
                MOp::VSmfmac16x16x128Fp8 => {
                    // D/A/B are 4/4/8 dwords; SRC2 is a single dword sparse-index register.
                    if let Some(d) = inst.dst {
                        if m.vreg_class(d) != RClass::V4 {
                            errs.push(format!("block {bi} inst {ii}: smfmac dst must be V4"));
                        }
                    }
                    if inst.srcs.len() != 4 || !matches!(inst.srcs.get(3), Some(Src::Reg(c)) if Some(*c) == inst.dst) {
                        errs.push(format!("block {bi} inst {ii}: smfmac needs the accumulator C (== dst) as srcs[3]"));
                    }
                    let want = [RClass::V4, RClass::V8, RClass::V1];
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r) == *w => {}
                            Some(Src::Reg(r)) => errs.push(format!(
                                "block {bi} inst {ii}: smfmac operand {k} is {:?}, expected {w:?}",
                                m.vreg_class(*r)
                            )),
                            _ => errs.push(format!(
                                "block {bi} inst {ii}: smfmac operand {k} is not a register"
                            )),
                        }
                    }
                }
                MOp::VMfma16x16x32Fp8 | MOp::VMfma32x32x16Fp8 => {
                    let wide = inst.op == MOp::VMfma32x32x16Fp8;
                    let acc = if wide { RClass::V16 } else { RClass::V4 };
                    if let Some(d) = inst.dst {
                        if !m.vreg_class(d).same_width_vector(acc) {
                            errs.push(format!("block {bi} inst {ii}: fp8 mfma dst class mismatch"));
                        }
                    }
                    let want = [RClass::V2, RClass::V2, acc];
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r).same_width_vector(*w) => {}
                            Some(Src::Reg(r)) => errs.push(format!(
                                "block {bi} inst {ii}: fp8 mfma operand {k} is {:?}, expected {w:?}",
                                m.vreg_class(*r)
                            )),
                            _ => errs.push(format!(
                                "block {bi} inst {ii}: fp8 mfma operand {k} is not a register"
                            )),
                        }
                    }
                }
                MOp::VMfma32x32x8F16Z => {
                    if let Some(d) = inst.dst {
                        if !m.vreg_class(d).same_width_vector(RClass::V16) {
                            errs.push(format!("block {bi} inst {ii}: f16 mfma dst must be V16"));
                        }
                    }
                }
                MOp::VMfma32x32x8F16 => {
                    if let Some(d) = inst.dst {
                        if !m.vreg_class(d).same_width_vector(RClass::V16) {
                            errs.push(format!("block {bi} inst {ii}: wide mfma dst must be V16"));
                        }
                    }
                    let want = [RClass::V2, RClass::V2, RClass::V16];
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r).same_width_vector(*w) => {}
                            Some(Src::Reg(r)) => errs.push(format!(
                                "block {bi} inst {ii}: wide mfma operand {k} is {:?}, expected {w:?}",
                                m.vreg_class(*r)
                            )),
                            _ => errs.push(format!(
                                "block {bi} inst {ii}: wide mfma operand {k} is not a register"
                            )),
                        }
                    }
                }
                MOp::VWriteSub | MOp::VReadSub => {
                    if inst.op == MOp::VWriteSub {
                        if let Some(d) = inst.dst {
                            if !matches!(m.vreg_class(d), RClass::V2 | RClass::V4 | RClass::V8 | RClass::V16) {
                                errs.push(format!(
                                    "block {bi} inst {ii}: tuple write dst must be V2 or V4"
                                ));
                            }
                        }
                    }
                    if let Some(Src::Reg(r)) = inst.srcs.first() {
                        let class = m.vreg_class(*r);
                        if inst.op == MOp::VReadSub && !matches!(class, RClass::V2 | RClass::V4 | RClass::V8 | RClass::V16) {
                            errs.push(format!(
                                "block {bi} inst {ii}: tuple read base must be V2 or V4"
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        // Branch targets must exist.
        match &b.term {
            Term::Goto(t) => {
                if !labels.contains(t) {
                    errs.push(format!("block {bi}: branch to unknown label {t}"));
                }
            }
            Term::CondGoto { target, fall, .. } => {
                if !labels.contains(target) {
                    errs.push(format!("block {bi}: branch to unknown label {target}"));
                }
                if !labels.contains(fall) {
                    errs.push(format!("block {bi}: branch to unknown label {fall}"));
                }
            }
            Term::Endpgm => {}
        }
    }
    errs
}
