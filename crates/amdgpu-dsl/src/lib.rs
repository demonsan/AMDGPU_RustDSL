//! Typed, structured IR and builder for AMDGPU kernels.
//!
//! Design notes
//! ------------
//! * Pure values are SSA (`ValueId`); values defined inside a loop or an `if`
//!   body are scoped to that region.
//! * Loop-carried state uses explicit mutable variables (`VarId`) rather than
//!   phi nodes. Lowering maps each variable to a virtual register, which keeps
//!   the backend free of phi elimination and makes the ownership rules
//!   ("a variable is written only through `SetVar`") checkable.
//! * Every operation is typed and target independent: no physical register
//!   numbers or machine encodings appear at this level.
//! * Effects are explicit: only `Load*`/`Store*`/`Kernarg*` touch memory.

#![forbid(unsafe_code)]

use std::collections::HashSet;
use std::fmt::Write as _;

pub type ValueId = u32;

/// Loop-carried mutable variable. A distinct type so it cannot be confused
/// with an SSA value id.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct VarId(pub u32);

impl std::fmt::Display for VarId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Element types understood by the IR.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DType {
    /// 16-bit float bit pattern held in a 32-bit register's low half.
    F16,
    F32,
    I32,
    U32,
    /// Predicate, materialised as 0/1 in a 32-bit register.
    I1,
    /// 64-bit global pointer.
    Ptr,
    /// Two-dword opaque fragment (an MFMA A/B operand).
    Frag2,
    /// Four-dword opaque fragment (an MFMA 16x16x16 accumulator).
    Frag4,
    /// Eight-dword opaque fragment (an MFMA 32x32x64 A/B operand).
    Frag8,
    /// Sixteen-dword opaque fragment (an MFMA 32x32 accumulator).
    Frag16,
    /// MUBUF buffer resource descriptor: 4 consecutive SGPRs (base, size, flags). Only exists in scalar registers; built by `Op::BufferRsrc`.
    Rsrc,
}

impl DType {
    pub fn is_int(self) -> bool {
        matches!(self, DType::I32 | DType::U32 | DType::I1)
    }

    pub fn is_float(self) -> bool {
        matches!(self, DType::F32)
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::I1 => "i1",
            DType::Ptr => "ptr",
            DType::Frag2 => "frag2",
            DType::Frag4 => "frag4",
            DType::Frag8 => "frag8",
            DType::Frag16 => "frag16",
            DType::Rsrc => "rsrc",
        }
    }

    pub fn dwords(self) -> u32 {
        match self {
            DType::Ptr | DType::Frag2 => 2,
            DType::Frag4 => 4,
            DType::Frag8 => 8,
            DType::Frag16 => 16,
            DType::Rsrc => 4,
            _ => 1,
        }
    }
}

/// Operand of an operation or statement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Opnd {
    Val(ValueId),
    Var(VarId),
    I32(i32),
    F32(f32),
}

impl From<ValueId> for Opnd {
    fn from(v: ValueId) -> Self {
        Opnd::Val(v)
    }
}

impl From<VarId> for Opnd {
    fn from(v: VarId) -> Self {
        Opnd::Var(v)
    }
}

impl From<i32> for Opnd {
    fn from(v: i32) -> Self {
        Opnd::I32(v)
    }
}

impl From<f32> for Opnd {
    fn from(v: f32) -> Self {
        Opnd::F32(v)
    }
}

/// Pure operations. All operands are consumed by value.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    ConstI32(i32),
    ConstF32(f32),

    KernargI32 { offset: u32 },
    KernargF32 { offset: u32 },
    KernargPtr { offset: u32 },

    /// Low 32 bits of the wave's `s_memtime` (core-clock counter) taken here, wave-uniform. `id` only keeps two stamps from comparing equal. The lowering pins the
    /// stamp in program order (it is a fence for the scheduler) and drains `lgkmcnt` right after it (SMEM result): place stamps where the kernel already waits.
    MemTimeLo { id: u32 },
    WorkitemId,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,

    AddI32(Opnd, Opnd),
    MulI32(Opnd, Opnd),
    ShlI32(Opnd, Opnd),
    LshrI32(Opnd, Opnd),
    AndI32(Opnd, Opnd),
    /// Signed 32-bit minimum (`v_min_i32`).
    MinI32(Opnd, Opnd),
    XorI32(Opnd, Opnd),

    AddF32(Opnd, Opnd),
    SubF32(Opnd, Opnd),
    MulF32(Opnd, Opnd),
    MaxF32(Opnd, Opnd),
    /// Three-way maximum (`v_max3_f32`).
    Max3F32(Opnd, Opnd, Opnd),
    FmaF32(Opnd, Opnd, Opnd),
    ExpF32(Opnd),
    RcpF32(Opnd),

    CvtF16ToF32(Opnd),
    CvtF32ToF16(Opnd),

    CmpLtI32(Opnd, Opnd),
    CmpGeI32(Opnd, Opnd),
    /// `a < b` on f32 (`v_cmp_lt_f32`).
    CmpLtF32(Opnd, Opnd),
    NotI1(Opnd),
    Select { ty: DType, cond: Opnd, a: Opnd, b: Opnd },

    /// `base + byte_off` for a 64-bit pointer; offset is a 32-bit value.
    PtrAdd { base: Opnd, byte_off: Opnd },

    LoadU16 { base: Opnd, byte_off: u32 },
    LoadF32 { base: Opnd, byte_off: u32 },
    /// 16-byte global load (`global_load_dwordx4`) into a `Frag4`.
    LoadGlobalB128 { base: Opnd, byte_off: u32 },
    /// Device-scope coherent loads (`sc1`): bypass the non-coherent cache levels, see the other XCDs' write-through stores.
    LoadF32Coherent { base: Opnd, byte_off: u32 },
    /// Buffer resource descriptor over the scalar (kernarg) pointer `base`: 4 SGPRs = [base lo, base hi (bits 47:32), num_records, flags]. Bounds checking
    /// is off with `num_records = 0xFFFF_FFFF`; the descriptor is allocated and reused by the compiler like any other SGPR value.
    BufferRsrc { base: Opnd, num_records: u32 },
    /// Descriptor with a RUNTIME `num_records` (a wave-uniform integer value or loop variable, in bytes): dword2 of the descriptor is an SGPR copy.
    BufferRsrcDyn { base: Opnd, num_records: Opnd },
    /// MUBUF load (`buffer_load_dword[x2|x4]`): address = resource base + `voff` (per-lane byte offset, absent = `off` form) + `soff` (wave-uniform byte
    /// offset, SGPR or 0) + `imm` (12-bit). `dwords` = 1 (f32), 2 (frag2), 4 (frag4).
    BufferLoad { rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm: u32, dwords: u8 },
    /// Two MUBUF `buffer_load_dwordx4` (same rsrc / voff / soff, immediates `imm_lo` / `imm_hi`) straight into the low / high 4 dwords of ONE
    /// 8-dword fragment (Frag8, no copy; each half is its own vmcnt entry).
    BufferLoadB128x2 { rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm_lo: u32, imm_hi: u32 },
    /// `global_atomic_add` (device scope, returns the OLD value): every active lane adds `value` at `base + byte_off`.
    AtomicAddU32 { base: Opnd, byte_off: u32, value: Opnd },
    /// Read one dword from workgroup LDS; `addr` is a byte offset.
    LoadLdsB32 { addr: Opnd },
    /// Read four dwords (16 B) from workgroup LDS.
    LoadLdsB128 { addr: Opnd },
    /// Read two dwords (8 B, `ds_read_b64`) from workgroup LDS: a `Frag2`.
    LoadLdsB64 { addr: Opnd },
    /// 8-bit transposing LDS read (ds_read_b64_tr_b8), two dwords per lane.
    LoadLdsTrB8 { addr: Opnd },
    /// Four `ds_read_b64_tr_b8` reads written straight into the four quarters of one `Frag8`.
    LoadLdsTr4B8 { addrs: [Opnd; 4] },
    /// Two `ds_read_b128` reads written straight into the two halves of one `Frag8`.
    LoadLdsB128x2 { lo: Opnd, hi: Opnd },
    /// Broadcast lane 0 of a vector value into a scalar register.
    ReadFirstLane(Opnd),
    /// Cross-lane quad permutation (`v_mov_b32_dpp quad_perm`): lane `l` receives lane `4 (l / 4) + perm[l % 4]` of `x` (all rows / banks written).
    DppQuadPerm { x: Opnd, perm: [u8; 4] },

    /// Build a two-dword fragment from two 32-bit values.
    Frag2(Opnd, Opnd),
    /// Read dword `index` (0 or 1) of a two-dword fragment.
    Frag2Lane { frag: Opnd, index: u8 },
    /// Dwords 8 half .. 8 half + 7 of the Frag16 variable `var` as a Frag8 value WITHOUT a copy (a view of the variable's registers): valid only while
    /// the variable is not written (checked by `verify`: no in-place write of `var` between the slice and any use of it).
    VarSlice8 { var: VarId, half: u8 },
    /// `v_permlane32_swap`: exchanges lanes 32..63 of `a` with lanes 0..31 of `b`. Result: a Frag2 whose dword 0 is the new `a` and dword 1
    /// the new `b` (use `permlane32_swap`, which returns the two values).
    Permlane32Swap { a: Opnd, b: Opnd },
    /// Build a four-dword fragment from four 32-bit values.
    Frag4(Opnd, Opnd, Opnd, Opnd),
    /// Read dword `index` (0..3) of a four-dword fragment.
    Frag4Lane { frag: Opnd, index: u8 },
    /// `v_mfma_f32_16x16x16_f16` with an existing accumulator.
    MfmaF16 { a: Opnd, b: Opnd, c: Opnd },
    /// `v_mfma_f32_32x32x8_f16` with an existing accumulator.
    MfmaF16Wide { a: Opnd, b: Opnd, c: Opnd },
    /// Build an eight-dword fragment from eight 32-bit values.
    Frag8(Opnd, Opnd, Opnd, Opnd, Opnd, Opnd, Opnd, Opnd),
    /// Concatenate two four-dword fragments into an eight-dword fragment.
    CatFrag8(Opnd, Opnd),
    /// Pack two f32 values into two unsigned-normalized 16-bit values (low = a).
    CvtPknormU16(Opnd, Opnd),
    /// `v_mfma_scale_f32_32x32x64_f8f6f4`: block-scaled fp8 MFMA. The two scale
    /// operands are E8M0 exponents in scalar registers.
    MfmaScaleFp8_32x32x64 { a: Opnd, b: Opnd, c: Opnd, scale_a: Opnd, scale_b: Opnd },
    /// `v_mfma_f32_32x32x64_f8f6f4`: the UNSCALED fp8 (e4m3 x e4m3) MFMA (no scale operands; identical to the scaled form with both scales = 127).
    MfmaFp8_32x32x64 { a: Opnd, b: Opnd, c: Opnd },
    /// `v_smfmac_f32_16x16x128_fp8_fp8`: sparse (2:4) fp8 A times dense fp8 B, f32 accumulate.
    /// `a` is a Frag4 (the 2:4-compressed A operand), `b` a Frag8 (dense B, 128 K x 16 N),
    /// `idx` the SRC2 dword of sixteen 2-bit sparse indices, `c` the Frag4 accumulator.
    SmfmacFp8_16x16x128 { a: Opnd, b: Opnd, idx: Opnd, c: Opnd },
    /// A safe-asm instruction sequence (`amdgpu-safeasm`): its input ports are bound to `inputs` (in port order), its single output port is
    /// the result. Lowered into the same MIR as every other op, so register allocation, scheduling and the hazard passes apply to it.
    Asm { block: std::sync::Arc<amdgpu_safeasm::AsmBlock>, inputs: Vec<Opnd> },
    /// Read dword `index` (0..15) of a sixteen-dword fragment.
    Frag16Lane { frag: Opnd, index: u8 },
    /// A fragment of `ty` zeroed out.
    FragZero { ty: DType },
    /// Reinterpret the bits of an integer as `f32` (register renaming only).
    BitcastF32(Opnd),
    /// Register reinterpretation to i32 (no instruction).
    BitcastI32(Opnd),
    /// Two f32 -> two bf16 (round to nearest even) packed into one dword (_cvt_pk_bf16_f32).
    CvtPkBf16F32(Opnd, Opnd),
    /// Base-2 exponential: the hardware primitive (_exp_f32), no log2(e) scaling.
    Exp2F32(Opnd),
    /// Base-2 logarithm: the hardware primitive (v_log_f32).
    Log2F32(Opnd),
    /// A `Frag2` with both dwords set to one 32-bit value.
    Frag2Dup(Opnd),
    /// `v_pk_add_f32` on two `Frag2` operands.
    PkAddF32(Opnd, Opnd),
    /// `v_pk_mul_f32` on two `Frag2` operands.
    PkMulF32(Opnd, Opnd),
    /// Four f32 -> four E4M3 bytes of one dword (a, b low half; c, d high half): two `v_cvt_pk_fp8_f32`.
    CvtPkFp8x4F32(Opnd, Opnd, Opnd, Opnd),
    /// max(x, x of lane l ^ 32) via `v_permlane32_swap_b32` (every lane gets the max of the two halves).
    PartnerMaxF32(Opnd),
    /// Schraudolph exp2 pack: `v_cvt_pknorm_u16_f32` of (a,b) and (c,d), each u16 pair reinterpreted as bf16 and packed to E4M3
    /// (a,b low half, c,d high half) with `v_cvt_scalef32_pk_fp8_bf16` (scale operand last).
    CvtPknormP8x4(Opnd, Opnd, Opnd, Opnd, Opnd),
    /// `v_perm_b32 d, a, b, sel` (byte select).
    PermB32(Opnd, Opnd, Opnd),
    /// `v_pk_sub_u16`: per-half 16-bit subtraction (no borrow across the halves).
    PkSubU16(Opnd, Opnd),
    /// `v_cvt_pknorm_u16_f32`: two values in [0, 1] -> one dword of two u16 (65535 x).
    CvtPknormPair(Opnd, Opnd),
    /// Second half of `CvtPknormP8x4`: two u16-pair dwords (bf16 reinterpretation) -> four fp8 with a scale.
    CvtP8x4FromU16(Opnd, Opnd, Opnd),
}

impl Op {
    pub fn result_ty(&self) -> DType {
        match self {
            Op::ConstI32(_) | Op::KernargI32 { .. } | Op::MemTimeLo { .. } | Op::AddI32(..) | Op::MulI32(..) => DType::I32,
            Op::ShlI32(..) | Op::LshrI32(..) | Op::AndI32(..) | Op::MinI32(..) | Op::XorI32(..) | Op::LoadU16 { .. } => DType::I32,
            Op::ConstF32(_) | Op::KernargF32 { .. } => DType::F32,
            Op::AddF32(..) | Op::SubF32(..) | Op::MulF32(..) | Op::MaxF32(..) | Op::FmaF32(..) => {
                DType::F32
            }
            Op::ExpF32(..) | Op::Exp2F32(..) | Op::Log2F32(..) | Op::RcpF32(..) | Op::CvtF16ToF32(..) | Op::LoadF32 { .. } | Op::LoadF32Coherent { .. } => DType::F32,
            Op::AtomicAddU32 { .. } => DType::U32,
            Op::BufferRsrc { .. } | Op::BufferRsrcDyn { .. } => DType::Rsrc,
            Op::BufferLoadB128x2 { .. } => DType::Frag8,
            Op::BufferLoad { dwords, .. } => match dwords {
                1 => DType::F32,
                2 => DType::Frag2,
                _ => DType::Frag4,
            },
            Op::CvtF32ToF16(..) => DType::I32,
            Op::WorkitemId | Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => DType::U32,
            Op::CmpLtI32(..) | Op::CmpGeI32(..) | Op::CmpLtF32(..) | Op::NotI1(..) => DType::I1,
            Op::Max3F32(..) => DType::F32,
            Op::Select { ty, .. } => *ty,
            Op::PtrAdd { .. } | Op::KernargPtr { .. } => DType::Ptr,
            Op::Frag2(..) => DType::Frag2,
            Op::Frag2Lane { .. } | Op::Frag4Lane { .. } => DType::U32,
            Op::VarSlice8 { .. } => DType::Frag8,
            Op::Permlane32Swap { .. } => DType::Frag2,
            Op::Frag4(..) | Op::MfmaF16 { .. } => DType::Frag4,
            Op::MfmaF16Wide { .. } => DType::Frag16,
            Op::Frag8(..) => DType::Frag8,
            Op::CatFrag8(..) => DType::Frag8,
            Op::CvtPknormU16(..) | Op::CvtPkBf16F32(..) => DType::U32,
            Op::MfmaScaleFp8_32x32x64 { .. } | Op::MfmaFp8_32x32x64 { .. } => DType::Frag16,
            Op::SmfmacFp8_16x16x128 { .. } => DType::Frag4,
            Op::Asm { block, .. } => asm_out_ty(block),
            Op::FragZero { ty } => *ty,
            Op::Frag16Lane { .. } => DType::U32,
            Op::Frag2Dup(..) | Op::PkAddF32(..) | Op::PkMulF32(..) => DType::Frag2,
            Op::CvtPkFp8x4F32(..) => DType::U32,
            Op::PartnerMaxF32(..) => DType::F32,
            Op::CvtPknormP8x4(..) => DType::U32,
        Op::PermB32(..) | Op::CvtPknormPair(..) | Op::PkSubU16(..) => DType::I32,
        Op::CvtP8x4FromU16(..) => DType::U32,
            Op::BitcastF32(..) => DType::F32,
            Op::BitcastI32(..) => DType::I32,
            Op::LoadLdsB32 { .. } => DType::U32,
            Op::LoadLdsB128 { .. } => DType::Frag4,
            Op::LoadLdsTrB8 { .. } | Op::LoadLdsB64 { .. } => DType::Frag2,
            Op::LoadLdsTr4B8 { .. } | Op::LoadLdsB128x2 { .. } => DType::Frag8,
            Op::LoadGlobalB128 { .. } => DType::Frag4,
            Op::ReadFirstLane(..) => DType::U32,
            Op::DppQuadPerm { .. } => DType::U32,
        }
    }

    pub fn operands(&self) -> Vec<Opnd> {
        match self {
            Op::ConstI32(_) | Op::ConstF32(_) | Op::KernargI32 { .. } | Op::KernargF32 { .. }
            | Op::KernargPtr { .. } | Op::MemTimeLo { .. } | Op::WorkitemId | Op::WorkgroupIdX | Op::WorkgroupIdY
            | Op::WorkgroupIdZ => vec![],
            Op::AddI32(a, b)
            | Op::MulI32(a, b)
            | Op::ShlI32(a, b)
            | Op::LshrI32(a, b)
            | Op::AndI32(a, b)
            | Op::MinI32(a, b)
            | Op::XorI32(a, b)
            | Op::AddF32(a, b)
            | Op::SubF32(a, b)
            | Op::MulF32(a, b)
            | Op::MaxF32(a, b)
            | Op::CmpLtI32(a, b)
            | Op::CmpGeI32(a, b)
            | Op::CmpLtF32(a, b)
            | Op::PtrAdd { base: a, byte_off: b } => vec![*a, *b],
            Op::FmaF32(a, b, c) => vec![*a, *b, *c],
            Op::Max3F32(a, b, c) => vec![*a, *b, *c],
            Op::ExpF32(a) | Op::Exp2F32(a) | Op::Log2F32(a) | Op::RcpF32(a) | Op::CvtF16ToF32(a) | Op::CvtF32ToF16(a) | Op::NotI1(a) => {
                vec![*a]
            }
            Op::Select { cond, a, b, .. } => vec![*cond, *a, *b],
            Op::LoadU16 { base, .. } | Op::LoadF32 { base, .. } => vec![*base],
            Op::Frag2(a, b) => vec![*a, *b],
            Op::Frag2Lane { frag, .. } => vec![*frag],
            Op::VarSlice8 { var, .. } => vec![Opnd::Var(*var)],
            Op::Permlane32Swap { a, b } => vec![*a, *b],
            Op::Frag4(a, b, c, d) => vec![*a, *b, *c, *d],
            Op::Frag4Lane { frag, .. } => vec![*frag],
            Op::MfmaF16 { a, b, c } => vec![*a, *b, *c],
            Op::MfmaF16Wide { a, b, c } => vec![*a, *b, *c],
            Op::Frag8(a, b, c, d, e, f, g, h) => vec![*a, *b, *c, *d, *e, *f, *g, *h],
            Op::CatFrag8(a, b) => vec![*a, *b],
            Op::CvtPknormU16(a, b) | Op::CvtPkBf16F32(a, b) => vec![*a, *b],
            Op::MfmaScaleFp8_32x32x64 { a, b, c, scale_a, scale_b } => {
                vec![*a, *b, *c, *scale_a, *scale_b]
            }
            Op::MfmaFp8_32x32x64 { a, b, c } => vec![*a, *b, *c],
            Op::SmfmacFp8_16x16x128 { a, b, idx, c } => vec![*a, *b, *idx, *c],
            Op::Asm { inputs, .. } => inputs.clone(),
            Op::Frag16Lane { frag, .. } => vec![*frag],
            Op::Frag2Dup(a) | Op::PartnerMaxF32(a) => vec![*a],
            Op::PkAddF32(a, b) | Op::PkMulF32(a, b) => vec![*a, *b],
            Op::CvtPkFp8x4F32(a, b, c, d) => vec![*a, *b, *c, *d],
            Op::CvtPknormP8x4(a, b, c, d, e) => vec![*a, *b, *c, *d, *e],
            Op::PermB32(a, b, c) | Op::CvtP8x4FromU16(a, b, c) => vec![*a, *b, *c],
            Op::CvtPknormPair(a, b) | Op::PkSubU16(a, b) => vec![*a, *b],
            Op::FragZero { .. } => vec![],
            Op::BitcastF32(a) | Op::BitcastI32(a) => vec![*a],
            Op::LoadLdsB32 { addr } => vec![*addr],
            Op::LoadLdsB128 { addr } => vec![*addr],
            Op::LoadLdsB64 { addr } => vec![*addr],
            Op::LoadLdsTrB8 { addr } => vec![*addr],
            Op::LoadLdsTr4B8 { addrs } => addrs.to_vec(),
            Op::LoadGlobalB128 { base, .. } | Op::LoadF32Coherent { base, .. } => vec![*base],
        Op::AtomicAddU32 { base, value, .. } => vec![*base, *value],
            Op::LoadLdsB128x2 { lo, hi } => vec![*lo, *hi],
            Op::BufferRsrc { base, .. } => vec![*base],
            Op::BufferRsrcDyn { base, num_records } => vec![*base, *num_records],
            Op::BufferLoad { rsrc, voff, soff, .. } | Op::BufferLoadB128x2 { rsrc, voff, soff, .. } => {
                let mut v = vec![*rsrc];
                v.extend(voff.iter().copied());
                v.push(*soff);
                v
            }
            Op::ReadFirstLane(a) => vec![*a],
            Op::DppQuadPerm { x, .. } => vec![*x],
        }
    }
}

/// Contract expressions in the shared symbolic arithmetic (`amdgpu-sym`).
impl Ext {
    /// Exact symbolic value (None for division, which is bounded by `to_itv`).
    pub fn to_poly(&self, cx: &amdgpu_sym::Ctx) -> Option<amdgpu_sym::Poly> {
        use amdgpu_sym::{Poly, Sym};
        Some(match self {
            Ext::Const(c) => Poly::int(*c as i128),
            Ext::Ka(o) => cx.sym_poly(Sym::Ka(*o)),
            Ext::Grid(a) => cx.sym_poly(Sym::Grid(*a)),
            Ext::Sym(n) => cx.sym_poly(Sym::Named(n.clone())),
            Ext::Add(a, b) => a.to_poly(cx)?.add(&b.to_poly(cx)?),
            Ext::Sub(a, b) => a.to_poly(cx)?.sub(&b.to_poly(cx)?),
            Ext::Mul(a, b) => a.to_poly(cx)?.mul(&b.to_poly(cx)?)?,
            Ext::CeilDiv(..) | Ext::FloorDiv(..) => return None,
        })
    }
    /// Candidate-bound interval (divisions bounded by shifting the dividend's bounds).
    pub fn to_itv(&self, cx: &amdgpu_sym::Ctx) -> amdgpu_sym::Itv {
        use amdgpu_sym::{cap, Itv, Poly, Q};
        match self {
            Ext::Const(_) | Ext::Ka(_) | Ext::Grid(_) | Ext::Sym(_) => match self.to_poly(cx) {
                Some(p) => Itv::point(p),
                None => Itv::top(),
            },
            Ext::Add(a, b) => a.to_itv(cx).add(&b.to_itv(cx)),
            Ext::Sub(a, b) => a.to_itv(cx).add(&b.to_itv(cx).scale(Q::int(-1))),
            Ext::Mul(a, b) => a.to_itv(cx).mul(&b.to_itv(cx), cx),
            Ext::CeilDiv(a, k) => {
                let x = a.to_itv(cx);
                let k = (*k).max(1) as i128;
                Itv { lo: cap(x.lo.iter().map(|p| p.scale(Q::new(1, k))).collect()), hi: cap(x.hi.iter().map(|p| p.add(&Poly::int(k - 1)).scale(Q::new(1, k))).collect()) }
            }
            Ext::FloorDiv(a, k) => {
                let x = a.to_itv(cx);
                let k = (*k).max(1) as i128;
                Itv { lo: cap(x.lo.iter().map(|p| p.add(&Poly::int(-(k - 1))).scale(Q::new(1, k))).collect()), hi: cap(x.hi.iter().map(|p| p.scale(Q::new(1, k))).collect()) }
            }
        }
    }
}

/// Symbolic integer expression for the GPU-semantics contract (extents, ranges): constants, 32-bit kernarg scalars (by kernarg byte offset) and the launch grid
/// dimensions (in workgroups). Non-negative quantities; `Sub` is allowed (the analyzer works with intervals, see docs/gpu-semantics-verifier.md).
#[derive(Clone, Debug, PartialEq)]
pub enum Ext {
    Const(i64),
    /// The 32-bit unsigned kernarg scalar at this kernarg byte offset.
    Ka(u32),
    /// Launch grid dimension in workgroups (0 = x, 1 = y, 2 = z).
    Grid(u8),
    /// A free named non-negative symbol (>= 1 unless `sym_range` says otherwise), e.g. the batch count `B` that is not a kernarg; related to the grid by `grid_is`.
    Sym(String),
    Add(Box<Ext>, Box<Ext>),
    Sub(Box<Ext>, Box<Ext>),
    Mul(Box<Ext>, Box<Ext>),
    /// ceil(e / k) and floor(e / k) for a positive constant k.
    CeilDiv(Box<Ext>, i64),
    FloorDiv(Box<Ext>, i64),
}

impl Ext {
    pub fn c(v: i64) -> Ext {
        Ext::Const(v)
    }
    pub fn ka(offset: u32) -> Ext {
        Ext::Ka(offset)
    }
    pub fn sym(name: &str) -> Ext {
        Ext::Sym(name.to_string())
    }
    pub fn grid_x() -> Ext {
        Ext::Grid(0)
    }
    pub fn grid_y() -> Ext {
        Ext::Grid(1)
    }
    pub fn grid_z() -> Ext {
        Ext::Grid(2)
    }
    pub fn ceil_div(self, k: i64) -> Ext {
        Ext::CeilDiv(Box::new(self), k)
    }
    pub fn floor_div(self, k: i64) -> Ext {
        Ext::FloorDiv(Box::new(self), k)
    }
}

impl From<i64> for Ext {
    fn from(v: i64) -> Ext {
        Ext::Const(v)
    }
}

impl std::ops::Add for Ext {
    type Output = Ext;
    fn add(self, o: Ext) -> Ext {
        Ext::Add(Box::new(self), Box::new(o))
    }
}
impl std::ops::Sub for Ext {
    type Output = Ext;
    fn sub(self, o: Ext) -> Ext {
        Ext::Sub(Box::new(self), Box::new(o))
    }
}
impl std::ops::Mul for Ext {
    type Output = Ext;
    fn mul(self, o: Ext) -> Ext {
        Ext::Mul(Box::new(self), Box::new(o))
    }
}

/// Declared extent of a global buffer reachable from a pointer kernarg (the HOST CONTRACT: the caller guarantees the allocation).
/// Loads are checked against `alloc` (the full allocation, including any zeroed slack the kernel is allowed to read); stores and atomics against `logical`
/// when given (the slack must not be written), else against `alloc`.
#[derive(Clone, Debug)]
pub struct ArgExtent {
    pub arg: String,
    pub alloc: Ext,
    pub logical: Option<Ext>,
}

/// A named LDS tile (or ring of equal slots): slot k occupies `[base + k*slot_size, base + (k+1)*slot_size)`.
#[derive(Clone, Debug)]
pub struct TileDecl {
    pub name: String,
    pub base: i64,
    pub slot_size: i64,
    pub slots: u32,
    /// Per-WAVE private region: slot k (= wave key k) is the scratch of one wave only (`lds_private`): its events never interact with other waves.
    pub private: bool,
    /// A global-memory region (`buf_region`): not LDS; its protocol (produce / release / complete / barrier-publish / acquire / consume) is checked by gpu_protocol.
    pub global: bool,
}

/// A global-memory region of a pointer kernarg: bytes `[base, base + size)` of `arg` (v4 protocol model, intra-workgroup + the opt-in last-arriver pattern).
#[derive(Clone, Debug)]
pub struct BufRegion {
    pub name: String,
    pub arg: String,
    pub base: Ext,
    pub size: Ext,
}

/// Opt-in fused-combine pattern: every block stores its slice of `region`, releases, takes a ticket on `counter_arg`; the last arriver acquires and reads all `splits` slices.
#[derive(Clone, Debug)]
pub struct LastArriver {
    pub region: String,
    pub counter_arg: String,
    pub splits: Ext,
}

/// The kernel's GPU-semantics contract, checked by `amdgpu_compiler::gpu_verify`.
#[derive(Clone, Debug, Default)]
pub struct Contract {
    pub extents: Vec<ArgExtent>,
    /// (kernarg offset, lo, hi): the 32-bit scalar at that offset lies in [lo, hi] (host contract; `hi` = i64::MAX = unbounded).
    pub ka_ranges: Vec<(u32, i64, i64)>,
    /// (axis, expr): the launch grid dimension equals `expr` (host contract), e.g. grid z = `sym("B") * ka(0x280)` for a split launch; the analyzer substitutes it, so
    /// products and quotients by the same factor simplify exactly without a division operator.
    pub grid_defs: Vec<(u8, Ext)>,
    /// (kernarg offset, expr): the 32-bit kernarg scalar at that offset EQUALS `expr` (host contract), e.g. a row stride `ka(0x70) == grid_y * ka(0x90)`; substituted by
    /// the analyzer like the grid definitions.
    pub ka_defs: Vec<(u32, Ext)>,
    /// (kernarg offset, allowed values): the scalar takes one of these values (host contract; the analyzer uses their min/max as the range).
    pub ka_sets: Vec<(u32, Vec<i64>)>,
    /// LDS tiles / rings (v1.1 lifetime model) and the explicitly aliased pairs (allowed to overlap).
    pub tiles: Vec<TileDecl>,
    pub tile_aliases: Vec<(String, String)>,
    /// Explicit trust that the tile protocol is periodic in the loop iteration (see `tile_trust_periodic`): skips the affine-counter check.
    pub tile_trust: Option<String>,
    /// Global-memory regions, the last-arriver patterns and the per-region trust entries (name, reason) of the v4 protocol model.
    pub buf_regions: Vec<BufRegion>,
    pub last_arrivers: Vec<LastArriver>,
    pub buf_trusts: Vec<(String, String)>,
    /// (symbol name, lo, hi) bounds of free symbols.
    pub sym_ranges: Vec<(String, i64, i64)>,
}

/// One value set crossing a stage boundary (`StageContract`): the IR values (fragment registers), the element table they hold (`[t + threads v]`
/// -> tensor element, after any K permutation: a layout fingerprint computed by the tile layer) and storage facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageVal {
    pub name: String,
    pub values: Vec<ValueId>,
    pub table: Vec<i64>,
    pub elem_bits: u32,
    pub storage: String,
}

/// The resource contract of a pipeline STAGE (fusion hand-off, docs/fusion-layout-handoff.md §3): what it consumes from earlier stages and
/// hands to later ones, the LDS tiles it frees / reuses, the barrier phase at its entry (barriers before it in program order) and the wave
/// roles. Checked by `gpu_verify` at the region edge (GV-STAGE); emits no code and is not a scheduling barrier.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StageContract {
    pub live_in: Vec<StageVal>,
    pub live_out: Vec<StageVal>,
    pub lds_freed: Vec<String>,
    pub lds_reused: Vec<String>,
    pub barrier_phase: u32,
    pub roles: Vec<String>,
}

/// Statements: a flat sequence with explicit structured markers.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// Opens a pipeline stage (`Node::Stage`) with its resource contract. Emits no code.
    StageBegin { name: String, contract: Box<StageContract> },
    /// Closes the innermost stage.
    StageEnd,
    /// LDS tile protocol marker (v1.1): from here until the next tile marker / `TileEnd`, the LDS writes and DMA issued by this wave are version `version` of slot
    /// `slot` of tile `tile`. Emits no code.
    TileProduce { tile: String, slot: Opnd, version: Opnd },
    /// ... the LDS reads issued by this wave expect version `version` of slot `slot` of tile `tile`.
    TileConsume { tile: String, slot: Opnd, version: Opnd },
    /// Ends the current tile marker scope.
    TileEnd,
    /// Contract fact: from here on `target` (a value or a loop variable) lies in [lo, hi]. For loop variables the analyzer PROVES it inductively when it can
    /// (initial value and every `set_var` stay inside) and otherwise records it as a TRUSTED assumption in the report. Lowers to nothing.
    Assume { target: Opnd, lo: Ext, hi: Ext, trusted: Option<String> },
    Let { dst: ValueId, op: Op },
    StoreU16 { base: Opnd, byte_off: u32, value: Opnd },
    StoreF32 { base: Opnd, byte_off: u32, value: Opnd },
    /// 16-byte global store of a Frag4 (`global_store_dwordx4`).
    StoreF32x4 { base: Opnd, byte_off: u32, value: Opnd },
    /// 8-byte global store of a Frag2 (`global_store_dwordx2`).
    StoreF32x2 { base: Opnd, byte_off: u32, value: Opnd },
    /// MUBUF store (`buffer_store_dword[x2|x4]`), same addressing as `Op::BufferLoad`; `value` is f32 (1), frag2 (2) or frag4 (4).
    BufferStore { rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm: u32, dwords: u8, value: Opnd },
    /// MUBUF global -> LDS DMA (`buffer_load_dword[x4] ... lds`, the analogue of `LoadGlobalLds`): the LDS destination is `M0 = lds_addr` (wave-uniform,
    /// first lane's value) + imm + 4*dwords*lane; `dwords` = 1 or 4.
    BufferLoadLds { rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, lds_addr: Opnd, imm: u32, dwords: u8 },
    /// Device-scope write-through stores (`sc1`): visible to coherent loads / atomics of other workgroups on other XCDs.
    StoreF32Wt { base: Opnd, byte_off: u32, value: Opnd },
    /// Release fence at device scope: `s_waitcnt vmcnt(0) lgkmcnt(0)` [+ `buffer_wbl2 sc1` + `s_waitcnt vmcnt(0)` when `wb`].
    FenceRelease { wb: bool },
    /// Acquire fence at device scope: `s_waitcnt vmcnt(0)` [+ `buffer_inv sc1` when `inv`]; later loads cannot move above it.
    FenceAcquire { inv: bool },
    /// Write one dword into workgroup LDS; `addr` is a byte offset.
    StoreLdsB32 { addr: Opnd, value: Opnd },
    /// Write four dwords (16 B) into workgroup LDS.
    StoreLdsB128 { addr: Opnd, value: Opnd },
    /// Write two dwords (8 B, `ds_write_b64`; the value is a `Frag2`) into workgroup LDS.
    StoreLdsB64 { addr: Opnd, value: Opnd },
    /// `s_waitcnt lgkmcnt(count)`.
    WaitLds { count: u8 },
    /// `s_barrier`: synchronise every wave in the workgroup.
    Barrier,
    /// `s_setprio`: wave issue priority 0..3.
    SetPrio { prio: u8 },
    /// Scheduler directive (`sched_group_barrier`): see `FnBuilder::sched_group_barrier`. No instruction is emitted.
    SchedGroup { mask: u32, size: u32, sync: u32 },
    /// `s_waitcnt vmcnt(count)`: wait until at most `count` global/LDS-DMA loads are outstanding.
    WaitVm { count: u8 },
    /// Global -> LDS DMA load (`global_load_lds_dwordx4` or `_dword`): lane i of the wave writes its 16 (4) bytes at
    /// `lds_addr + 16 i` (`+ 4 i`), `lds_addr` being wave-uniform; completion is tracked by `vmcnt` only.
    LoadGlobalLds { base: Opnd, lds_addr: Opnd, byte_off: i32, dwords: u8 },
    SetVar { var: VarId, value: Opnd },
    /// Multiply all sixteen dwords of a `Frag16` loop variable by an f32 scalar, in place.
    VarScaleF32 { var: VarId, scalar: Opnd },
    /// Multiply all sixteen dwords of a Frag16 loop variable by the (duplicated) scalar held in a Frag2, in place, with _pk_mul_f32.
    VarScalePk { var: VarId, pair: Opnd },
    /// Add the (duplicated) scalar held in a `Frag2` to all sixteen dwords of a `Frag16` loop variable, in place (`v_pk_add_f32`).
    VarAddPk { var: VarId, pair: Opnd },
    /// `var = mfma_f32_32x32x8_f16(a, b)` with a zero accumulator (a, b are `Frag2`).
    VarMfmaF16Z { var: VarId, a: Opnd, b: Opnd },
    /// Multiply one dword of a `Frag16` loop variable by an f32 scalar, in place.
    VarScaleLane { var: VarId, index: u8, scalar: Opnd },
    /// `var.lane[index] = value` (a 32-bit value into one dword of a fragment variable; the lowering writes the producer straight into that
    /// dword when the value has no other use).
    VarSetLane { var: VarId, index: u8, value: Opnd },
    /// `var = scaled_mfma_32x32x64_fp8(a, b, var)`: accumulate into a loop variable in place.
    /// `opsel`: E8M0 byte select of the scale dwords (bits 0..1 = byte of `scale_a`, bits 4..5 = byte of `scale_b`; 0 = byte 0), as `VarMfmaScale16`.
    VarMfmaScale { var: VarId, a: Opnd, b: Opnd, scale_a: Opnd, scale_b: Opnd, opsel: u8 },
    /// `var = a x b + c` (c a Frag16 VALUE, not the variable itself: the accumulator input and the destination are different registers)
    VarMfmaScaleC { var: VarId, a: Opnd, b: Opnd, c: Opnd, scale_a: Opnd, scale_b: Opnd },
    /// UNSCALED fp8 MFMA (`v_mfma_f32_32x32x64_f8f6f4`) into a loop variable: in place `var += a x b`
    VarMfmaFp8 { var: VarId, a: Opnd, b: Opnd },
    /// ... with a zero accumulator (`var = a x b`)
    VarMfmaFp8Z { var: VarId, a: Opnd, b: Opnd },
    /// ... with a separate accumulator input (`var = a x b + c`, c a Frag16 value, dst != c)
    VarMfmaFp8C { var: VarId, a: Opnd, b: Opnd, c: Opnd },
    /// `var += smfmac_16x16x128_fp8(a, b, idx)`: sparse-A MFMA accumulating into a `Frag4` loop variable in place.
    VarSmfmac { var: VarId, a: Opnd, b: Opnd, idx: Opnd },
    /// `var += v_mfma_scale_f32_16x16x128_f8f6f4(a, b; scale_a, scale_b)` in place on a `Frag4` loop variable (E4M3 operands, 8 dwords each).
    /// `opsel` selects the E8M0 byte of each scale dword: bits 0..1 = byte of `scale_a`, bits 4..5 = byte of `scale_b` (`op_sel` / `op_sel_hi`).
    VarMfmaScale16 { var: VarId, a: Opnd, b: Opnd, scale_a: Opnd, scale_b: Opnd, opsel: u8 },
    /// A safe-asm sequence updating the loop variable `var` in place through its in/out port (the first input port, `Seq::inout`);
    /// ``inputs`` bind to the remaining input ports in order.
    VarAsm { var: VarId, block: std::sync::Arc<amdgpu_safeasm::AsmBlock>, inputs: Vec<Opnd> },
    /// A safe-asm sequence updating SEVERAL loop variables in place: the i-th in/out port (declaration order) is bound to `vars[i]`.
    VarAsmN { vars: Vec<VarId>, block: std::sync::Arc<amdgpu_safeasm::AsmBlock>, inputs: Vec<Opnd> },
    BeginLoop { vars: Vec<VarId>, init: Vec<Opnd> },
    BreakIf { cond: Opnd },
    EndLoop,
    BeginIf { cond: Opnd },
    Else,
    EndIf,
}

#[derive(Clone, Debug)]
pub struct Arg {
    pub name: String,
    pub ty: DType,
    pub kernarg_offset: u32,
    pub size: u32,
}

pub const SCHED_ALU: u32 = 0x1;
pub const SCHED_VALU: u32 = 0x2;
pub const SCHED_SALU: u32 = 0x4;
pub const SCHED_MFMA: u32 = 0x8;
pub const SCHED_VMEM: u32 = 0x10;
pub const SCHED_VMEM_READ: u32 = 0x20;
pub const SCHED_VMEM_WRITE: u32 = 0x40;
pub const SCHED_DS: u32 = 0x80;
pub const SCHED_DS_READ: u32 = 0x100;
pub const SCHED_DS_WRITE: u32 = 0x200;

/// What an AGPR request applies to (see FnBuilder::pin_agpr / FnBuilder::agpr).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgprTarget {
    Var(VarId),
    Val(ValueId),
}

/// One request to keep a loop variable / value in the accumulation-register file (gfx90a+/gfx950), optionally at an exact AGPR index
/// (the analogue of clang's mdgpu_pin_agpr).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AgprMark {
    pub target: AgprTarget,
    pub pin: Option<u32>,
}

/// A node of the structured kernel body (region IR): straight-line statements and nested regions. Loops, ifs and tile / buffer
/// protocol regions are trees, so a begin without its end (or an end without its begin) cannot be represented.
#[derive(Clone, Debug)]
pub enum Node {
    /// a straight-line statement (never a structure marker)
    S(Stmt),
    /// `begin_loop(vars = init) ... end_loop` (do-while; `Stmt::BreakIf` inside the body leaves it)
    Loop { vars: Vec<VarId>, init: Vec<Opnd>, body: Vec<Node> },
    /// wave-level `if cond { then } else { els }` (`els = None`: no else branch)
    If { cond: Opnd, then: Vec<Node>, els: Option<Vec<Node>> },
    /// tile / buffer protocol region: `tile_produce` / `tile_consume` (and `buf_produce` / `buf_consume`) ... `tile_end`
    Region { kind: RegionKind, tile: String, slot: Opnd, version: Opnd, body: Vec<Node> },
    /// a pipeline stage with its boundary contract (`stage_begin` ... `stage_end`): checked at the edge, transparent to lowering / scheduling
    Stage { name: String, contract: Box<StageContract>, body: Vec<Node> },
}

/// Kind of a protocol region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Produce,
    Consume,
}

fn flatten_into(nodes: &[Node], out: &mut Vec<Stmt>) {
    for n in nodes {
        match n {
            Node::S(s) => out.push(s.clone()),
            Node::Loop { vars, init, body } => {
                out.push(Stmt::BeginLoop { vars: vars.clone(), init: init.clone() });
                flatten_into(body, out);
                out.push(Stmt::EndLoop);
            }
            Node::If { cond, then, els } => {
                out.push(Stmt::BeginIf { cond: *cond });
                flatten_into(then, out);
                if let Some(e) = els {
                    out.push(Stmt::Else);
                    flatten_into(e, out);
                }
                out.push(Stmt::EndIf);
            }
            Node::Region { kind, tile, slot, version, body } => {
                out.push(match kind {
                    RegionKind::Produce => Stmt::TileProduce { tile: tile.clone(), slot: *slot, version: *version },
                    RegionKind::Consume => Stmt::TileConsume { tile: tile.clone(), slot: *slot, version: *version },
                });
                flatten_into(body, out);
                out.push(Stmt::TileEnd);
            }
            Node::Stage { name, contract, body } => {
                out.push(Stmt::StageBegin { name: name.clone(), contract: contract.clone() });
                flatten_into(body, out);
                out.push(Stmt::StageEnd);
            }
        }
    }
}

/// Builder / parser frame of an open region.
#[derive(Clone, Debug)]
enum Frame {
    Loop { vars: Vec<VarId>, init: Vec<Opnd>, body: Vec<Node> },
    If { cond: Opnd, then: Vec<Node>, els: Option<Vec<Node>> },
    Region { kind: RegionKind, tile: String, slot: Opnd, version: Opnd, body: Vec<Node> },
    Stage { name: String, contract: Box<StageContract>, body: Vec<Node> },
}

impl Frame {
    fn name(&self) -> &'static str {
        match self {
            Frame::Loop { .. } => "loop",
            Frame::If { .. } => "if",
            Frame::Region { .. } => "tile region",
            Frame::Stage { .. } => "stage",
        }
    }
    fn nodes(&mut self) -> &mut Vec<Node> {
        match self {
            Frame::Loop { body, .. } | Frame::Region { body, .. } | Frame::Stage { body, .. } => body,
            Frame::If { then, els, .. } => els.as_mut().unwrap_or(then),
        }
    }
    fn close(self) -> Node {
        match self {
            Frame::Loop { vars, init, body } => Node::Loop { vars, init, body },
            Frame::If { cond, then, els } => Node::If { cond, then, els },
            Frame::Region { kind, tile, slot, version, body } => Node::Region { kind, tile, slot, version, body },
            Frame::Stage { name, contract, body } => Node::Stage { name, contract, body },
        }
    }
}

/// Region-tree construction shared by the builder and `Function::parse_flat`.
#[derive(Clone, Debug, Default)]
struct TreeBuilder {
    root: Vec<Node>,
    open: Vec<Frame>,
    /// statements of the flat view emitted so far (markers included)
    count: usize,
}

impl TreeBuilder {
    fn cur(&mut self) -> &mut Vec<Node> {
        match self.open.last_mut() {
            Some(f) => f.nodes(),
            None => &mut self.root,
        }
    }
    /// Appends one flat statement: a marker opens / switches / closes a region, anything else is a leaf.
    fn push(&mut self, st: Stmt) -> Result<(), String> {
        self.count += 1;
        match st {
            Stmt::BeginLoop { vars, init } => self.open.push(Frame::Loop { vars, init, body: Vec::new() }),
            Stmt::BeginIf { cond } => self.open.push(Frame::If { cond, then: Vec::new(), els: None }),
            Stmt::TileProduce { tile, slot, version } => {
                if self.open.iter().any(|f| matches!(f, Frame::Region { .. })) {
                    return Err(format!("tile_produce({tile}) inside an open tile region (close it with tile_end first)"));
                }
                self.open.push(Frame::Region { kind: RegionKind::Produce, tile, slot, version, body: Vec::new() })
            }
            Stmt::TileConsume { tile, slot, version } => {
                if self.open.iter().any(|f| matches!(f, Frame::Region { .. })) {
                    return Err(format!("tile_consume({tile}) inside an open tile region (close it with tile_end first)"));
                }
                self.open.push(Frame::Region { kind: RegionKind::Consume, tile, slot, version, body: Vec::new() })
            }
            Stmt::Else => match self.open.last_mut() {
                Some(Frame::If { els, .. }) if els.is_none() => *els = Some(Vec::new()),
                Some(Frame::If { .. }) => return Err("second else in one if".into()),
                other => return Err(format!("else without an open if (innermost open region: {})", other.map_or("none", |f| f.name()))),
            },
            Stmt::EndLoop => self.close(|f| matches!(f, Frame::Loop { .. }), "end_loop")?,
            Stmt::EndIf => self.close(|f| matches!(f, Frame::If { .. }), "end_if")?,
            Stmt::TileEnd => self.close(|f| matches!(f, Frame::Region { .. }), "tile_end")?,
            Stmt::StageBegin { name, contract } => {
                if self.open.iter().any(|f| matches!(f, Frame::Region { .. })) {
                    return Err(format!("stage {name} inside an open tile region (close it with tile_end first)"));
                }
                self.open.push(Frame::Stage { name, contract, body: Vec::new() })
            }
            Stmt::StageEnd => self.close(|f| matches!(f, Frame::Stage { .. }), "stage_end")?,
            other => self.cur().push(Node::S(other)),
        }
        Ok(())
    }
    fn close(&mut self, is: impl Fn(&Frame) -> bool, what: &str) -> Result<(), String> {
        match self.open.last() {
            Some(f) if is(f) => {
                let n = self.open.pop().unwrap().close();
                self.cur().push(n);
                Ok(())
            }
            other => Err(format!("{what} without its opening statement (innermost open region: {})", other.map_or("none", |f| f.name()))),
        }
    }
    fn finish(self) -> Result<Vec<Node>, String> {
        if let Some(f) = self.open.last() {
            return Err(format!("{} open regions at the end of the kernel (innermost: {})", self.open.len(), f.name()));
        }
        Ok(self.root)
    }
}

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub args: Vec<Arg>,
    pub kernarg_size: u32,
    pub workgroup_size: u32,
    /// Bytes of workgroup LDS this kernel declares.
    pub lds_bytes: u32,
    pub contract: Contract,
    /// the structured body (region IR, canonical)
    body: Vec<Node>,
    /// its linearization with structure markers (derived from `body`; what the passes iterate)
    flat: Vec<Stmt>,
    pub var_tys: Vec<DType>,
    pub value_tys: Vec<DType>,
    /// Opt-out of the loop-variable re-read lint (see [`lint`]).
    pub allow_var_rereads: bool,
    /// AGPR placement requests (empty for every existing kernel).
    pub agpr: Vec<AgprMark>,
    /// Explicit compile policy (lowering options + scheduler preset). `None` = the compiler derives it from the kernel name (the historical behaviour,
    /// which every existing kernel keeps); kernels written on the tile library set it so their policy does not depend on their name.
    pub compile_profile: Option<CompileProfile>,
    /// Extra lowering option tokens for this kernel (the AMDC_LOWER_OPTS vocabulary, e.g. lgkmn), on top of its profile - so nothing is name-keyed.
    pub lower_opts: Vec<String>,
    /// Scheduler option tokens (the AMDC_SCHED_OPTS vocabulary: rewait,ldshoist,xbar,hoist=N,limit=N,...); Some turns the scheduler on for this kernel (unless AMDC_SCHED=0).
    pub sched_opts: Option<String>,
}

/// Compile policy of a kernel (see `amdgpu-compiler/src/lib.rs`): which lowering options and which scheduler preset it gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompileProfile {
    /// The policy of `fmha_fwd_hd128_mxfp8_vs128` (scalarize/retarget2/imm/sgpr/zeroinit lowering, full reordering list scheduler).
    Vs128,
    /// The policy of `fmha_fwd_hd128_mxfp8_t128` (as Vs128 + addrfold; scheduler: ALU across the barrier, VGPR pressure limit 248).
    T128,
    /// The policy of `fmha_fwd_hd128_mxfp8_kvsplit` (same as Vs128).
    KvSplit,
    /// The policy of the KV-split combine kernels (FMHA zeroinit lowering, no scheduler preset).
    Combine,
    /// The MXFP8 GEMM kernels (mdgpu-kernels::gemm_mxfp8): lowering dce,cmpfold,lgkmn,zeroinit, scheduler = the scoreboard wait pass (partial mcnt/lgkmcnt,
    /// including across the loop back edge) + ds_read hoisting, no instruction reordering. Kernel-specific additions go through Function::lower_opts / Function::sched_opts.
    Gemm,
    /// No FMHA-specific policy (plain lowering, scheduler only when requested by the environment).
    Default,
}

impl Function {
    /// The structured body (region IR).
    pub fn body(&self) -> &[Node] {
        &self.body
    }
    /// The body linearized with structure markers (`BeginLoop .. EndLoop`, `BeginIf .. Else .. EndIf`, `TileProduce/TileConsume .. TileEnd`); always
    /// balanced because it is derived from the tree.
    pub fn stmts(&self) -> &[Stmt] {
        &self.flat
    }
    /// Replaces the body.
    pub fn set_body(&mut self, body: Vec<Node>) {
        let mut flat = Vec::new();
        flatten_into(&body, &mut flat);
        self.body = body;
        self.flat = flat;
    }
    /// Builds the region tree of a flat statement sequence (fails on an unbalanced sequence).
    pub fn parse_flat(stmts: Vec<Stmt>) -> Result<Vec<Node>, String> {
        let mut tb = TreeBuilder::default();
        for (i, s) in stmts.into_iter().enumerate() {
            tb.push(s).map_err(|e| format!("stmt {i}: {e}"))?;
        }
        tb.finish()
    }
    /// Replaces the body by the tree of a flat statement sequence; panics if it is unbalanced.
    pub fn set_stmts(&mut self, stmts: Vec<Stmt>) {
        let body = Function::parse_flat(stmts).unwrap_or_else(|e| panic!("set_stmts: {e}"));
        self.set_body(body);
    }
    /// Edits the linearized body (tests: fault injection) and rebuilds the tree; panics if the edit unbalances it.
    pub fn edit_stmts(&mut self, edit: impl FnOnce(&mut Vec<Stmt>)) {
        let mut flat = self.flat.clone();
        edit(&mut flat);
        let body = Function::parse_flat(flat).unwrap_or_else(|e| panic!("edit_stmts: {e}"));
        self.set_body(body);
    }
    pub fn value_ty(&self, v: ValueId) -> Option<DType> {
        self.value_tys.get(v as usize).copied()
    }

    pub fn var_ty(&self, v: VarId) -> Option<DType> {
        self.var_tys.get(v.0 as usize).copied()
    }

    /// Human readable dump used by tests and `amdc dump`.
    pub fn dump(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "fn {} (wg={}, kernarg={})", self.name, self.workgroup_size, self.kernarg_size);
        for a in &self.args {
            let _ = writeln!(s, "  arg {} : {} @ {} (+{})", a.name, a.ty.name(), a.kernarg_offset, a.size);
        }
        let mut depth = 0usize;
        for st in self.stmts() {
            match st {
                Stmt::EndLoop | Stmt::EndIf | Stmt::Else => depth = depth.saturating_sub(1),
                _ => {}
            }
            let pad = "  ".repeat(depth + 1);
            match st {
                Stmt::Let { dst, op } => {
                    let _ = writeln!(s, "{pad}v{dst}: {} = {op:?}", self.value_ty(*dst).map(|t| t.name()).unwrap_or("?"));
                }
                Stmt::StoreU16 { base, byte_off, value } => {
                    let _ = writeln!(s, "{pad}store u16 [{base:?}+{byte_off}] <- {value:?}");
                }
                Stmt::StoreF32 { base, byte_off, value } => {
                    let _ = writeln!(s, "{pad}store f32 [{base:?}+{byte_off}] <- {value:?}");
                }
                Stmt::StoreF32x4 { base, byte_off, value } => {
                    let _ = writeln!(s, "{pad}store f32x4 [{base:?}+{byte_off}] <- {value:?}");
                }
                Stmt::StoreF32x2 { base, byte_off, value } => {
                    let _ = writeln!(s, "{pad}store f32x2 [{base:?}+{byte_off}] <- {value:?}");
                }
                Stmt::BufferStore { rsrc, voff, soff, imm, dwords, value } => {
                    let _ = writeln!(s, "{pad}buffer_store x{dwords} [{rsrc:?} voff={voff:?} soff={soff:?} +{imm}] <- {value:?}");
                }
                Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, imm, dwords } => {
                    let _ = writeln!(s, "{pad}buffer_load_lds x{dwords} [{rsrc:?} voff={voff:?} soff={soff:?} +{imm}] -> lds[{lds_addr:?}]");
                }
                Stmt::StoreF32Wt { base, byte_off, value } => {
                    let _ = writeln!(s, "{pad}store.wt f32 [{base:?}+{byte_off}] <- {value:?}");
                }
                Stmt::FenceRelease { wb } => {
                    let _ = writeln!(s, "{pad}fence.release(wb={wb})");
                }
                Stmt::FenceAcquire { inv } => {
                    let _ = writeln!(s, "{pad}fence.acquire(inv={inv})");
                }
                Stmt::StoreLdsB32 { addr, value } => {
                    let _ = writeln!(s, "{pad}lds[{addr:?}] <- {value:?}");
                }
                Stmt::StoreLdsB128 { addr, value } => {
                    let _ = writeln!(s, "{pad}lds128[{addr:?}] <- {value:?}");
                }
                Stmt::StoreLdsB64 { addr, value } => {
                    let _ = writeln!(s, "{pad}lds64[{addr:?}] <- {value:?}");
                }
                Stmt::WaitLds { count } => {
                    let _ = writeln!(s, "{pad}waitlds({count})");
                }
                Stmt::Assume { target, lo, hi, trusted } => {
                    let _ = writeln!(s, "{pad}assume{} {target:?} in [{lo:?}, {hi:?}]", if trusted.is_some() { "_trusted" } else { "" });
                }
                Stmt::TileProduce { tile, slot, version } => {
                    let _ = writeln!(s, "{pad}tile_produce {tile}[{slot:?}] v={version:?}");
                }
                Stmt::TileConsume { tile, slot, version } => {
                    let _ = writeln!(s, "{pad}tile_consume {tile}[{slot:?}] v={version:?}");
                }
                Stmt::TileEnd => {
                    let _ = writeln!(s, "{pad}tile_end");
                }
                Stmt::StageBegin { name, contract } => {
                    let _ = writeln!(s, "{pad}stage {name} in={:?} out={:?}", contract.live_in.iter().map(|x| &x.name).collect::<Vec<_>>(), contract.live_out.iter().map(|x| &x.name).collect::<Vec<_>>());
                }
                Stmt::StageEnd => {
                    let _ = writeln!(s, "{pad}stage_end");
                }
                Stmt::Barrier => {
                    let _ = writeln!(s, "{pad}barrier");
                }
                Stmt::SetPrio { prio } => {
                    let _ = writeln!(s, "{pad}setprio {prio}");
                }
                Stmt::SchedGroup { mask, size, sync } => {
                    let _ = writeln!(s, "{pad}sched_group_barrier mask=0x{mask:x} size={size} sync={sync}");
                }
                Stmt::WaitVm { count } => {
                    let _ = writeln!(s, "{pad}waitvm({count})");
                }
                Stmt::LoadGlobalLds { base, lds_addr, byte_off, dwords } => {
                    let _ = writeln!(s, "{pad}dma{} lds[{lds_addr:?}] <- [{base:?}+{byte_off}]", dwords * 4);
                }
                Stmt::SetVar { var, value } => {
                    let _ = writeln!(s, "{pad}var{} <- {value:?}", var.0);
                }
                Stmt::VarScaleF32 { var, scalar } => {
                    let _ = writeln!(s, "{pad}var{} *= {scalar:?}", var.0);
                }
                Stmt::VarSetLane { var, index, value } => {
                    let _ = writeln!(s, "{pad}var{}.lane[{index}] = {value:?}", var.0);
                }
                Stmt::VarScaleLane { var, index, scalar } => {
                    let _ = writeln!(s, "{pad}var{}[{index}] *= {scalar:?}", var.0);
                }
                Stmt::VarAddPk { var, pair } => {
                    let _ = writeln!(s, "{pad}var{} +=pk {pair:?}", var.0);
                }
                Stmt::VarMfmaF16Z { var, a, b } => {
                    let _ = writeln!(s, "{pad}var{} = mfma_f16({a:?}, {b:?})", var.0);
                }
                Stmt::VarScalePk { var, pair } => {
                    let _ = writeln!(s, "{pad}var{} *=pk {pair:?}", var.0);
                }
                Stmt::VarMfmaFp8 { var, a, b } => {
                    let _ = writeln!(s, "{pad}var{} += mfma_fp8({a:?}, {b:?})", var.0);
                }
                Stmt::VarMfmaFp8Z { var, a, b } => {
                    let _ = writeln!(s, "{pad}var{} = mfma_fp8({a:?}, {b:?})", var.0);
                }
                Stmt::VarMfmaFp8C { var, a, b, c } => {
                    let _ = writeln!(s, "{pad}var{} = mfma_fp8({a:?}, {b:?}, c={c:?})", var.0);
                }
                Stmt::VarMfmaScaleC { var, a, b, c, scale_a, scale_b } => {
                    let _ = writeln!(s, "{pad}var{} = mfma_scale({a:?}, {b:?}, c={c:?}; {scale_a:?}, {scale_b:?})", var.0);
                }
                Stmt::VarMfmaScale { var, a, b, scale_a, scale_b, opsel } => {
                    if *opsel == 0 {
                        let _ = writeln!(s, "{pad}var{} += mfma_scale({a:?}, {b:?}; {scale_a:?}, {scale_b:?})", var.0);
                    } else {
                        let _ = writeln!(s, "{pad}var{} += mfma_scale({a:?}, {b:?}; {scale_a:?}, {scale_b:?}; opsel 0x{opsel:02x})", var.0);
                    }
                }
                Stmt::VarMfmaScale16 { var, a, b, scale_a, scale_b, opsel } => {
                    let _ = writeln!(s, "{pad}var{} += mfma_scale16({a:?}, {b:?}; {scale_a:?}, {scale_b:?}; opsel 0x{opsel:02x})", var.0);
                }
                Stmt::VarSmfmac { var, a, b, idx } => {
                    let _ = writeln!(s, "{pad}var{} += smfmac({a:?}, {b:?}; {idx:?})", var.0);
                }
                Stmt::VarAsm { var, block, inputs } => {
                    let _ = writeln!(s, "{pad}var{} <- asm `{}` ({} insts; {inputs:?})", var.0, block.name(), block.insts().len());
                }
                Stmt::VarAsmN { vars, block, inputs } => {
                    let ids: Vec<u32> = vars.iter().map(|v| v.0 as u32).collect();
                    let _ = writeln!(s, "{pad}vars{ids:?} <- asm `{}` ({} insts; {inputs:?})", block.name(), block.insts().len());
                }
                Stmt::BeginLoop { vars, init } => {
                    let _ = writeln!(s, "{pad}loop vars={vars:?} init={init:?}");
                    depth += 1;
                }
                Stmt::BreakIf { cond } => {
                    let _ = writeln!(s, "{pad}break if {cond:?}");
                }
                Stmt::EndLoop => {
                    let _ = writeln!(s, "{pad}endloop");
                }
                Stmt::BeginIf { cond } => {
                    let _ = writeln!(s, "{pad}if {cond:?}");
                    depth += 1;
                }
                Stmt::Else => {
                    let _ = writeln!(s, "{pad}else");
                    depth += 1;
                }
                Stmt::EndIf => {
                    let _ = writeln!(s, "{pad}endif");
                }
            }
        }
        s
    }
}

/// Builder that allocates ids and keeps the statement stream balanced.
pub struct FnBuilder {
    pub f: Function,
    /// the region tree under construction (open loops / ifs / tile regions)
    tree: TreeBuilder,
    /// constants materialised by the expression layer (`expr.rs`), keyed by (is_float, bits): reused by later `bind`s in the same or an inner scope, cleared when a scope closes
    pub(crate) expr_consts: std::collections::HashMap<(bool, u32), ValueId>,
}

impl FnBuilder {
    pub fn new(name: &str, workgroup_size: u32) -> Self {
        FnBuilder {
            f: Function {
                name: name.to_string(),
                args: Vec::new(),
                kernarg_size: 0,
                workgroup_size,
                lds_bytes: 0,
                contract: Contract::default(),
                body: Vec::new(),
                flat: Vec::new(),
                var_tys: Vec::new(),
                value_tys: Vec::new(),
                allow_var_rereads: false,
            agpr: Vec::new(),
                compile_profile: None,
            lower_opts: Vec::new(),
            sched_opts: None,
            },
            expr_consts: Default::default(),
            tree: TreeBuilder::default(),
        }
    }

    /// Declares that this kernel intentionally re-reads loop variables after `set_var` in the same iteration.
    pub fn allow_var_rereads(&mut self) {
        self.f.allow_var_rereads = true;
    }

    /// Sets the explicit compile policy (see [`CompileProfile`]); without it the compiler keeps deriving the policy from the kernel name.
    pub fn compile_profile(&mut self, p: CompileProfile) {
        self.f.compile_profile = Some(p);
    }

    /// Extra lowering option tokens for this kernel (see Function::lower_opts).
    pub fn lower_opts(&mut self, opts: &[&str]) {
        self.f.lower_opts.extend(opts.iter().map(|s| s.to_string()));
    }

    /// Scheduler option tokens for this kernel (see Function::sched_opts).
    pub fn sched_opts(&mut self, opts: &str) {
        self.f.sched_opts = Some(opts.to_string());
    }

    /// Declares how many bytes of workgroup LDS the kernel uses.
    pub fn lds(&mut self, bytes: u32) {
        self.f.lds_bytes = bytes;
    }

    /// GPU-semantics contract: the pointer kernarg `name` points to an allocation of `alloc_bytes` bytes (an expression over kernarg scalars and the grid).
    pub fn arg_tensor(&mut self, name: &str, alloc_bytes: Ext) {
        self.f.contract.extents.push(ArgExtent { arg: name.to_string(), alloc: alloc_bytes, logical: None });
    }
    /// Like `arg_tensor` with a zeroed slack at the end: reads may use `alloc_bytes`, writes must stay within `logical_bytes`.
    pub fn arg_tensor_slack(&mut self, name: &str, logical_bytes: Ext, alloc_bytes: Ext) {
        self.f.contract.extents.push(ArgExtent { arg: name.to_string(), alloc: alloc_bytes, logical: Some(logical_bytes) });
    }
    /// Host contract: the 32-bit kernarg scalar at `offset` lies in [lo, hi] (use `i64::MAX` for no upper bound).
    pub fn ka_range(&mut self, offset: u32, lo: i64, hi: i64) {
        self.f.contract.ka_ranges.push((offset, lo, hi));
    }
    /// Host contract: launch grid axis (0 = x, 1 = y, 2 = z) equals `expr`, e.g. `b.grid_is(2, Ext::sym("B") * Ext::ka(0x280))` for B batches x splits.
    pub fn grid_is(&mut self, axis: u8, expr: Ext) {
        self.f.contract.grid_defs.push((axis, expr));
    }
    /// Host contract: the u32 kernarg scalar at `offset` is one of `values` (e.g. the number of KV splits in {2, 4, 8}); the min/max become its range.
    pub fn ka_one_of(&mut self, offset: u32, values: &[i64]) {
        self.f.contract.ka_sets.push((offset, values.to_vec()));
    }
    /// Host contract: the u32 kernarg scalar at `offset` equals `expr` (e.g. strides derived from the shape: `b.ka_is(0x70, Ext::grid_y() * Ext::ka(0x90))`).
    pub fn ka_is(&mut self, offset: u32, expr: Ext) {
        self.f.contract.ka_defs.push((offset, expr));
    }
    /// Bounds of a free `Ext::sym` (default [1, +inf)).
    pub fn sym_range(&mut self, name: &str, lo: i64, hi: i64) {
        self.f.contract.sym_ranges.push((name.to_string(), lo, hi));
    }
    /// Declares an LDS tile `[base, base+size)` (one slot). Tiles must not overlap each other unless aliased with `lds_alias`.
    pub fn lds_tile(&mut self, name: &str, base: i64, size: i64) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base, slot_size: size, slots: 1, private: false, global: false });
    }
    /// Declares a ring of `slots` equal slots starting at `base`.
    pub fn lds_ring(&mut self, name: &str, base: i64, slot_size: i64, slots: u32) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base, slot_size, slots, private: false, global: false });
    }
    /// Declares a per-WAVE private LDS region: `waves` slices of `size_per_wave` bytes at `base + wave * size_per_wave`. Mark accesses with `tile_produce/tile_consume(name,
    /// wave_key, version)` where `wave_key` is THE value the address is computed from (e.g. `base + wave_key * size + lane * 16`): the verifier proves the access lies in the
    /// slice of that very key (own wave) and that the wave reads only versions it produced and completed itself (no cross-wave publish is needed or assumed).
    pub fn lds_private(&mut self, name: &str, base: i64, size_per_wave: i64, waves: u32) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base, slot_size: size_per_wave, slots: waves, private: true, global: false });
    }
    /// The two tiles are allowed to overlap (time-multiplexed storage): the lifetime check then requires their events to be ordered like versions of one tile.
    pub fn lds_alias(&mut self, a: &str, b: &str) {
        self.f.contract.tile_aliases.push((a.to_string(), b.to_string()));
    }
    /// Trust entry for the tile simulator: the slot / version expressions are NOT plain affine counters (xor/add/and/mul by constants) but the protocol is periodic in
    /// the loop iteration by construction, so the simulated iterations (>= 10 and >= 2 x ring slots + 4) cover every launch. Listed as TRUSTED in the report.
    pub fn tile_trust_periodic(&mut self, reason: &str) {
        self.f.contract.tile_trust = Some(reason.to_string());
    }
    /// Declares a GLOBAL-memory region: bytes `[base, base + size)` of the pointer kernarg `arg`. Mark the writes with `buf_produce(name, gen)` and the reads with
    /// `buf_consume(name, gen)`: the verifier then checks the four protocol obligations (complete before publish: `wait_vm(0)` / `fence_release`; publish before read:
    /// release -> barrier -> acquire inside the workgroup; read before overwrite; generation consistency). `gen` follows the same affine/periodic rules as tile versions.
    pub fn buf_region(&mut self, name: &str, arg: &str, base: Ext, size: Ext) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base: 0, slot_size: 1, slots: 1, private: false, global: true });
        self.f.contract.buf_regions.push(BufRegion { name: name.to_string(), arg: arg.to_string(), base, size });
    }
    /// The global writes issued from here (until `tile_end` / the next marker / barrier / control boundary) produce generation `gen` of the region.
    pub fn buf_produce(&mut self, name: &str, generation: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileProduce { tile: name.to_string(), slot: Opnd::I32(0), version: generation.into() });
    }
    /// The global reads issued from here expect generation `gen` of the region.
    pub fn buf_consume(&mut self, name: &str, generation: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileConsume { tile: name.to_string(), slot: Opnd::I32(0), version: generation.into() });
    }
    /// Declares the fused-combine (last-arriver) pattern over `region`: the stores marked `buf_produce` (generation = the block's split id, derived from the workgroup id)
    /// must be followed by `fence_release`, then an atomic ticket on the pointer kernarg `counter_arg`; the reads marked `buf_consume` must sit in the branch guarded by the
    /// ticket result, after `fence_acquire` (or with `sc1` loads for the lite variant), and cover more than one slice.
    pub fn buf_last_arriver(&mut self, region: &str, counter_arg: &str, splits: Ext) {
        self.f.contract.last_arrivers.push(LastArriver { region: region.to_string(), counter_arg: counter_arg.to_string(), splits });
    }
    /// Counted trust escape for one region: the verifier does not check it (e.g. a visibility argument it cannot make); listed in the certificate.
    pub fn buf_trust(&mut self, region: &str, reason: &str) {
        self.f.contract.buf_trusts.push((region.to_string(), reason.to_string()));
    }
    /// See `Stmt::TileProduce`.
    pub fn tile_produce(&mut self, tile: &str, slot: impl Into<Opnd>, version: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileProduce { tile: tile.to_string(), slot: slot.into(), version: version.into() });
    }
    /// See `Stmt::TileConsume`.
    pub fn tile_consume(&mut self, tile: &str, slot: impl Into<Opnd>, version: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileConsume { tile: tile.to_string(), slot: slot.into(), version: version.into() });
    }
    pub fn tile_end(&mut self) {
        self.push_stmt(Stmt::TileEnd);
    }
    /// Opens a pipeline stage with its boundary contract (see `Stmt::StageBegin`).
    pub fn stage_begin(&mut self, name: &str, contract: StageContract) {
        self.push_stmt(Stmt::StageBegin { name: name.to_string(), contract: Box::new(contract) });
    }
    pub fn stage_end(&mut self) {
        self.push_stmt(Stmt::StageEnd);
    }
    /// Adds a hand-over set to the innermost open stage's contract (its values are usually computed inside the stage).
    pub fn stage_live_out(&mut self, v: StageVal) {
        match self.tree.open.iter_mut().rev().find(|f| matches!(f, Frame::Stage { .. })) {
            Some(Frame::Stage { contract, .. }) => contract.live_out.push(v),
            _ => panic!("{}: stage_live_out outside a stage", self.f.name),
        }
    }
    /// Closure form of `stage_begin` / `stage_end`.
    pub fn stage<R>(&mut self, name: &str, contract: StageContract, f: impl FnOnce(&mut Self) -> R) -> R {
        self.stage_begin(name, contract);
        let r = f(self);
        self.stage_end();
        r
    }
    /// Contract fact `lo <= target <= hi` from this point on (see `Stmt::Assume`). Emits no code.
    pub fn assume(&mut self, target: impl Into<Opnd>, lo: Ext, hi: Ext) {
        self.push_stmt(Stmt::Assume { target: target.into(), lo, hi, trusted: None });
    }
    /// Like `assume`, but for a loop variable whose range holds by a guard the analyzer cannot see (e.g. a counter bounded by the loop exit test): the induction
    /// check failing is NOT an error, the assumption is listed as TRUSTED with `reason`.
    pub fn assume_trusted(&mut self, target: impl Into<Opnd>, lo: Ext, hi: Ext, reason: &str) {
        self.push_stmt(Stmt::Assume { target: target.into(), lo, hi, trusted: Some(reason.to_string()) });
    }

    pub fn arg(&mut self, name: &str, ty: DType, offset: u32, size: u32) {
        self.f.args.push(Arg { name: name.to_string(), ty, kernarg_offset: offset, size });
        self.f.kernarg_size = self.f.kernarg_size.max(offset + size);
    }

    /// Embed a safe-asm sequence. `inputs` bind to the block's input ports in order (types are checked against the port classes by `verify`).
    pub fn asm(&mut self, block: amdgpu_safeasm::AsmBlock, inputs: &[Opnd]) -> ValueId {
        self.let_(Op::Asm { block: std::sync::Arc::new(block), inputs: inputs.to_vec() })
    }

    pub fn let_(&mut self, op: Op) -> ValueId {
        let ty = op.result_ty();
        let dst = self.f.value_tys.len() as ValueId;
        self.f.value_tys.push(ty);
        self.push_stmt(Stmt::Let { dst, op });
        dst
    }

    pub fn store_u16(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreU16 { base: base.into(), byte_off, value: value.into() });
    }

    pub fn store_f32(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32 { base: base.into(), byte_off, value: value.into() });
    }

    // ---- MUBUF (buffer_load/buffer_store): SGPR resource descriptor + per-lane 32-bit byte offset + SGPR soffset + 12-bit immediate ----
    /// Resource descriptor over the kernarg pointer `base` (bounds checking off: num_records = 0xFFFFFFFF). The descriptor lives in 4 SGPRs.
    pub fn buffer_rsrc(&mut self, base: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BufferRsrc { base: base.into(), num_records: 0xFFFF_FFFF })
    }
    /// Descriptor with an explicit `num_records` (byte range; out-of-range accesses read 0 / are dropped).
    pub fn buffer_rsrc_sized(&mut self, base: impl Into<Opnd>, num_records: u32) -> ValueId {
        self.let_(Op::BufferRsrc { base: base.into(), num_records })
    }
    pub fn buffer_load_f32(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32) -> ValueId {
        self.let_(Op::BufferLoad { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 1 })
    }
    pub fn buffer_load_b64(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32) -> ValueId {
        self.let_(Op::BufferLoad { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 2 })
    }
    pub fn buffer_load_b128(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32) -> ValueId {
        self.let_(Op::BufferLoad { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 4 })
    }
    /// `off` forms (no per-lane vaddr): every lane reads the same address (resource base + soff + imm).
    /// Two `buffer_load_dwordx4` (offsets `imm_lo` / `imm_hi`) into the two halves of one Frag8 (an MFMA operand), without the copy of `cat_frag8`.
    pub fn buffer_load_b128x2(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm_lo: u32, imm_hi: u32) -> ValueId {
        self.let_(Op::BufferLoadB128x2 { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm_lo, imm_hi })
    }

    pub fn buffer_store_f32(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::BufferStore { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 1, value: value.into() });
    }
    pub fn buffer_store_b64(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::BufferStore { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 2, value: value.into() });
    }
    pub fn buffer_store_b128(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::BufferStore { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 4, value: value.into() });
    }
    /// MUBUF -> LDS DMA of 16 bytes per lane (`buffer_load_dwordx4 ... lds`): LDS destination = M0 (`lds_addr`) + imm + 16 * lane.
    pub fn buffer_load_lds_b128(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, lds_addr: impl Into<Opnd>, imm: u32) {
        self.push_stmt(Stmt::BufferLoadLds { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), lds_addr: lds_addr.into(), imm, dwords: 4 });
    }
    /// Descriptor with a RUNTIME byte range: `num_records` is a wave-uniform integer (kernarg value, scalar arithmetic or loop variable). Rebuild it where the
    /// value changes (the descriptor is an ordinary SGPR-quad value). Reads at offset >= num_records return 0, writes are dropped.
    pub fn buffer_desc_dyn(&mut self, ptr: impl Into<Opnd>, num_records_bytes: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BufferRsrcDyn { base: ptr.into(), num_records: num_records_bytes.into() })
    }
    /// Alias names used by the kernels: descriptor over a kernarg pointer with `num_records` bytes (reads at offset >= num_records return 0, writes are dropped).
    pub fn buffer_desc(&mut self, ptr: impl Into<Opnd>, num_records_bytes: u32) -> ValueId {
        self.buffer_rsrc_sized(ptr, num_records_bytes)
    }
    /// `buffer_load_dword[x4] ... lds` with the LDS slot base `lds_addr` (M0) and `dwords` = 1 or 4.
    pub fn load_buffer_lds(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, lds_addr: impl Into<Opnd>, imm: u32, dwords: u8) {
        self.push_stmt(Stmt::BufferLoadLds { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), lds_addr: lds_addr.into(), imm, dwords });
    }
    /// 4 bytes per lane (`buffer_load_dword ... lds`).
    pub fn buffer_load_lds_b32(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, lds_addr: impl Into<Opnd>, imm: u32) {
        self.push_stmt(Stmt::BufferLoadLds { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), lds_addr: lds_addr.into(), imm, dwords: 1 });
    }

    pub fn store_f32x2(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32x2 { base: base.into(), byte_off, value: value.into() });
    }

    pub fn store_f32x4(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32x4 { base: base.into(), byte_off, value: value.into() });
    }

    pub fn set_var(&mut self, var: VarId, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::SetVar { var, value: value.into() });
    }

    /// Declares `vars` initialised from `init`, then opens the loop body.
    pub fn begin_loop(&mut self, inits: &[(DType, Opnd)]) -> Vec<VarId> {
        let mut vars = Vec::with_capacity(inits.len());
        let mut init = Vec::with_capacity(inits.len());
        for (ty, val) in inits {
            let id = VarId(self.f.var_tys.len() as u32);
            self.f.var_tys.push(*ty);
            vars.push(id);
            init.push(*val);
        }
        self.push_stmt(Stmt::BeginLoop { vars: vars.clone(), init });
        vars
    }

    pub fn break_if(&mut self, cond: impl Into<Opnd>) {
        self.push_stmt(Stmt::BreakIf { cond: cond.into() });
    }

    pub fn end_loop(&mut self) {
        self.expr_consts.clear();
        self.push_stmt(Stmt::EndLoop);
    }

    pub fn begin_if(&mut self, cond: impl Into<Opnd>) {
        self.push_stmt(Stmt::BeginIf { cond: cond.into() });
    }

    pub fn else_(&mut self) {
        self.expr_consts.clear();
        self.push_stmt(Stmt::Else);
    }

    pub fn end_if(&mut self) {
        self.expr_consts.clear();
        self.push_stmt(Stmt::EndIf);
    }

    /// Keep the loop variable `var` in the AGPR file (any placement). MFMAs that accumulate into it use the `acc_cd` form; VALU access is
    /// done through `v_accvgpr_read/write` inserted by the compiler (docs/agpr.md).
    pub fn pin_agpr(&mut self, var: VarId) {
        self.f.agpr.push(AgprMark { target: AgprTarget::Var(var), pin: None });
    }

    /// As `pin_agpr`, at the exact AGPR index `base` (aligned to the register width).
    pub fn pin_agpr_at(&mut self, var: VarId, base: u32) {
        self.f.agpr.push(AgprMark { target: AgprTarget::Var(var), pin: Some(base) });
    }

    /// Mark the register tuple that holds `v` (an LDS/global load result, an MFMA result, a fragment) as AGPR-resident; returns `v`.
    pub fn agpr(&mut self, v: ValueId) -> ValueId {
        self.f.agpr.push(AgprMark { target: AgprTarget::Val(v), pin: None });
        v
    }

    pub fn finish(mut self) -> Function {
        let tree = std::mem::take(&mut self.tree);
        let body = tree.finish().unwrap_or_else(|e| panic!("kernel `{}`: {e}", self.f.name));
        self.f.set_body(body);
        self.f
    }

    /// Statements of the flat view emitted so far (markers included).
    pub(crate) fn tree_count(&self) -> usize {
        self.tree.count
    }

    /// Appends a statement to the innermost open region (markers open / close regions); an unbalanced end panics here, at build time.
    fn push_stmt(&mut self, st: Stmt) {
        if let Err(e) = self.tree.push(st) {
            panic!("kernel `{}`: {e}", self.f.name);
        }
    }

    /// Closure form of `begin_loop` / `end_loop`: `body` receives the loop variables.
    pub fn loop_<R>(&mut self, inits: &[(DType, Opnd)], body: impl FnOnce(&mut Self, &[VarId]) -> R) -> R {
        let vars = self.begin_loop(inits);
        let r = body(self, &vars);
        self.end_loop();
        r
    }
    /// Closure form of `begin_if` / `end_if`.
    pub fn if_<R>(&mut self, cond: impl Into<Opnd>, then: impl FnOnce(&mut Self) -> R) -> R {
        self.begin_if(cond);
        let r = then(self);
        self.end_if();
        r
    }
    /// Closure form of `begin_if` / `else_` / `end_if`.
    pub fn if_else(&mut self, cond: impl Into<Opnd>, then: impl FnOnce(&mut Self), els: impl FnOnce(&mut Self)) {
        self.begin_if(cond);
        then(self);
        self.else_();
        els(self);
        self.end_if();
    }
    /// Closure form of `tile_produce` / `tile_end`.
    pub fn produce_region<R>(&mut self, tile: &str, slot: impl Into<Opnd>, version: impl Into<Opnd>, body: impl FnOnce(&mut Self) -> R) -> R {
        self.tile_produce(tile, slot, version);
        let r = body(self);
        self.tile_end();
        r
    }

    // ---- convenience constructors -------------------------------------

    pub fn c_i32(&mut self, v: i32) -> ValueId {
        self.let_(Op::ConstI32(v))
    }

    pub fn c_f32(&mut self, v: f32) -> ValueId {
        self.let_(Op::ConstF32(v))
    }

    pub fn add_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::AddI32(a.into(), b.into()))
    }

    pub fn min_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MinI32(a.into(), b.into()))
    }

    pub fn mul_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MulI32(a.into(), b.into()))
    }

    pub fn shl_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ShlI32(a.into(), b.into()))
    }

    pub fn add_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::AddF32(a.into(), b.into()))
    }

    pub fn sub_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::SubF32(a.into(), b.into()))
    }

    pub fn mul_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MulF32(a.into(), b.into()))
    }

    pub fn max_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MaxF32(a.into(), b.into()))
    }

    pub fn xor_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::XorI32(a.into(), b.into()))
    }

    pub fn max3_f32(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::Max3F32(a.into(), b.into(), c.into()))
    }

    pub fn exp_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ExpF32(a.into()))
    }

    pub fn rcp_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::RcpF32(a.into()))
    }

    pub fn cvt_f32_to_f16(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtF32ToF16(a.into()))
    }

    pub fn cmp_lt_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CmpLtF32(a.into(), b.into()))
    }

    pub fn cmp_lt_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CmpLtI32(a.into(), b.into()))
    }

    pub fn load_u16(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadU16 { base: base.into(), byte_off })
    }

    /// Coherent (`sc1`) f32 load: reads through to the device-coherent level.
    pub fn load_f32_coherent(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadF32Coherent { base: base.into(), byte_off })
    }

    /// `global_atomic_add` returning the OLD value, device scope. Per-lane semantics: every active lane adds ITS `value`; pass `1` in lane 0 and `0`
    /// elsewhere for a one-ticket-per-wave counter, then `read_first_lane` the result.
    pub fn atomic_add_u32(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) -> ValueId {
        self.let_(Op::AtomicAddU32 { base: base.into(), byte_off, value: value.into() })
    }

    pub fn store_f32_wt(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32Wt { base: base.into(), byte_off, value: value.into() });
    }

    /// Device-scope release: wait for all earlier loads/stores, write back L2 (`buffer_wbl2 sc1`), wait again.
    pub fn fence_release(&mut self) {
        self.push_stmt(Stmt::FenceRelease { wb: true });
    }

    /// Release without the L2 write-back (valid when every shared store used `store_*_wt`).
    pub fn fence_release_lite(&mut self) {
        self.push_stmt(Stmt::FenceRelease { wb: false });
    }

    /// Device-scope acquire: wait for the ticket, invalidate (`buffer_inv sc1`).
    pub fn fence_acquire(&mut self) {
        self.push_stmt(Stmt::FenceAcquire { inv: true });
    }

    pub fn fence_acquire_lite(&mut self) {
        self.push_stmt(Stmt::FenceAcquire { inv: false });
    }

    pub fn load_f32(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadF32 { base: base.into(), byte_off })
    }

    pub fn ptr_add(&mut self, base: impl Into<Opnd>, byte_off: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PtrAdd { base: base.into(), byte_off: byte_off.into() })
    }

    pub fn frag2(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Frag2(a.into(), b.into()))
    }

    /// `v_permlane32_swap`: `(a', b')` with a' = [a lanes 0..31, b lanes 0..31] and b' = [a lanes 32..63, b lanes 32..63] (lanes 32..63 of `a`
    /// exchanged with lanes 0..31 of `b`). 32-bit values.
    pub fn permlane32_swap(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> (ValueId, ValueId) {
        let p = self.let_(Op::Permlane32Swap { a: a.into(), b: b.into() });
        (self.frag2_lane(p, 0), self.frag2_lane(p, 1))
    }

    pub fn frag2_lane(&mut self, frag: impl Into<Opnd>, index: u8) -> ValueId {
        self.let_(Op::Frag2Lane { frag: frag.into(), index })
    }

    pub fn frag4(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
        d: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::Frag4(a.into(), b.into(), c.into(), d.into()))
    }

    pub fn frag4_lane(&mut self, frag: impl Into<Opnd>, index: u8) -> ValueId {
        self.let_(Op::Frag4Lane { frag: frag.into(), index })
    }

    pub fn mfma_f16(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::MfmaF16 { a: a.into(), b: b.into(), c: c.into() })
    }

    pub fn bitcast_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BitcastF32(a.into()))
    }

    pub fn bitcast_i32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BitcastI32(a.into()))
    }

    pub fn mfma_f16_wide(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::MfmaF16Wide { a: a.into(), b: b.into(), c: c.into() })
    }

    #[allow(clippy::too_many_arguments)]
    /// Concatenate two Frag4 values into a Frag8 (low half first).
    /// Pack two f32 into two unsigned-normalized 16-bit values.
    pub fn cvt_pknorm_u16_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPknormU16(a.into(), b.into()))
    }

    /// Pack two f32 into two E4M3 fp8 bytes (upper 16 bits are zero).
    pub fn exp2_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Exp2F32(a.into()))
    }

    pub fn log2_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Log2F32(a.into()))
    }

    pub fn cvt_pk_bf16_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPkBf16F32(a.into(), b.into()))
    }

    pub fn cat_frag8(&mut self, lo: impl Into<Opnd>, hi: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CatFrag8(lo.into(), hi.into()))
    }

    pub fn frag8(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
        d: impl Into<Opnd>,
        e: impl Into<Opnd>,
        f: impl Into<Opnd>,
        g: impl Into<Opnd>,
        h: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::Frag8(
            a.into(),
            b.into(),
            c.into(),
            d.into(),
            e.into(),
            f.into(),
            g.into(),
            h.into(),
        ))
    }

    pub fn mfma_scale_fp8(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
        scale_a: impl Into<Opnd>,
        scale_b: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::MfmaScaleFp8_32x32x64 {
            a: a.into(),
            b: b.into(),
            c: c.into(),
            scale_a: scale_a.into(),
            scale_b: scale_b.into(),
        })
    }

    pub fn frag16_lane(&mut self, frag: impl Into<Opnd>, index: u8) -> ValueId {
        self.let_(Op::Frag16Lane { frag: frag.into(), index })
    }

    /// Sparse fp8 MFMA (`v_smfmac_f32_16x16x128_fp8_fp8`): D = compressed A x dense B + C.
    pub fn smfmac_fp8_16x16x128(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        idx: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::SmfmacFp8_16x16x128 {
            a: a.into(),
            b: b.into(),
            idx: idx.into(),
            c: c.into(),
        })
    }

    /// A zero-initialised fragment of the given type.
    pub fn frag_zero(&mut self, ty: DType) -> ValueId {
        self.let_(Op::FragZero { ty })
    }

    /// One dword read from workgroup LDS.
    pub fn load_lds_b32(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB32 { addr: addr.into() })
    }

    /// One dword write into workgroup LDS.
    pub fn store_lds_b32(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreLdsB32 { addr: addr.into(), value: value.into() });
    }

    /// Four dwords (16 B) read from workgroup LDS.
    pub fn load_lds_b128(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB128 { addr: addr.into() })
    }

    /// `ds_read_b64`: two dwords from LDS (8-byte aligned) as a `Frag2`.
    /// Wave-uniform select of two buffer descriptors (`s_cmp` + 2 x `s_cselect_b64`, no branch): `cond` must be a (possibly negated) signed compare
    /// of wave-uniform integers. The GPU-semantics verifier proves every access through the result against BOTH descriptors.
    pub fn select_uniform_rsrc(&mut self, cond: impl Into<Opnd>, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Select { ty: DType::Rsrc, cond: cond.into(), a: a.into(), b: b.into() })
    }
    /// Wave-uniform select of two SGPR pointers (kernarg pointers or uniform pointer arithmetic; `s_cmp` + `s_cselect_b64`). Accesses through the
    /// result are proven against BOTH pointers' extents.
    pub fn select_uniform_ptr(&mut self, cond: impl Into<Opnd>, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Select { ty: DType::Ptr, cond: cond.into(), a: a.into(), b: b.into() })
    }

    pub fn load_lds_b64(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB64 { addr: addr.into() })
    }

    /// Four dwords (16 B) write into workgroup LDS.
    /// Transposing 8-bit LDS read into two dwords per lane.
    /// 16-byte global load into a `Frag4`.
    pub fn load_b128(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadGlobalB128 { base: base.into(), byte_off })
    }

    /// Four transposing reads into one `Frag8` (bytes 8r..8r+7 from the r-th address).
    pub fn load_lds_tr_b8x4(
        &mut self,
        a0: impl Into<Opnd>,
        a1: impl Into<Opnd>,
        a2: impl Into<Opnd>,
        a3: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::LoadLdsTr4B8 { addrs: [a0.into(), a1.into(), a2.into(), a3.into()] })
    }

    /// Two 16-byte LDS reads into the low and high half of one `Frag8`.
    pub fn load_lds_b128x2(&mut self, lo: impl Into<Opnd>, hi: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB128x2 { lo: lo.into(), hi: hi.into() })
    }

    pub fn partner_max_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PartnerMaxF32(a.into()))
    }
    pub fn frag2_dup(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Frag2Dup(a.into()))
    }
    pub fn pk_add_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PkAddF32(a.into(), b.into()))
    }
    pub fn pk_mul_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PkMulF32(a.into(), b.into()))
    }
    pub fn cvt_pk_fp8x4_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>, d: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPkFp8x4F32(a.into(), b.into(), c.into(), d.into()))
    }
    pub fn var_mfma_scale_c(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>, scale_a: impl Into<Opnd>, scale_b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaScaleC { var, a: a.into(), b: b.into(), c: c.into(), scale_a: scale_a.into(), scale_b: scale_b.into() });
    }

    /// Unscaled fp8 MFMA `a x b + c` (value form; c = frag_zero gives the zero-accumulator instruction).
    pub fn mfma_fp8_32x32x64(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MfmaFp8_32x32x64 { a: a.into(), b: b.into(), c: c.into() })
    }
    /// `var += a x b` (unscaled fp8, in place).
    pub fn var_mfma_fp8_32x32x64(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaFp8 { var, a: a.into(), b: b.into() });
    }
    /// `var = a x b` (unscaled fp8, zero accumulator).
    pub fn var_mfma_fp8_32x32x64_z(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaFp8Z { var, a: a.into(), b: b.into() });
    }
    /// `var = a x b + c` (unscaled fp8; c a separate Frag16 value).
    pub fn var_mfma_fp8_32x32x64_c(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaFp8C { var, a: a.into(), b: b.into(), c: c.into() });
    }
    pub fn var_scale_lane(&mut self, var: VarId, index: u8, scalar: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarScaleLane { var, index, scalar: scalar.into() });
    }

    /// `var.lane[index] = value` (32-bit value into one dword of a fragment variable).
    pub fn var_set_lane(&mut self, var: VarId, index: u8, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarSetLane { var, index, value: value.into() });
    }

    /// Dwords 8 half .. 8 half + 7 of the Frag16 variable `var` as a Frag8 operand without a copy (see `Op::VarSlice8`).
    pub fn var_slice8(&mut self, var: VarId, half: u8) -> ValueId {
        assert!(half < 2, "var_slice8: half is 0 or 1");
        self.let_(Op::VarSlice8 { var, half })
    }
    pub fn perm_b32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, sel: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PermB32(a.into(), b.into(), sel.into()))
    }
    pub fn pk_sub_u16(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PkSubU16(a.into(), b.into()))
    }
    pub fn cvt_pknorm_pair(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPknormPair(a.into(), b.into()))
    }
    pub fn cvt_p8x4_from_u16(&mut self, u0: impl Into<Opnd>, u1: impl Into<Opnd>, scale: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtP8x4FromU16(u0.into(), u1.into(), scale.into()))
    }
    pub fn cvt_pknorm_p8x4(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>, d: impl Into<Opnd>, scale: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPknormP8x4(a.into(), b.into(), c.into(), d.into(), scale.into()))
    }
    pub fn var_add_pk(&mut self, var: VarId, pair: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarAddPk { var, pair: pair.into() });
    }
    pub fn var_mfma_f16_z(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaF16Z { var, a: a.into(), b: b.into() });
    }
    pub fn var_scale_pk(&mut self, var: VarId, pair: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarScalePk { var, pair: pair.into() });
    }

    /// `var = scaled_mfma(a, b, var)` in place.
    pub fn var_mfma_scale(
        &mut self,
        var: VarId,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        scale_a: impl Into<Opnd>,
        scale_b: impl Into<Opnd>,
    ) {
        self.push_stmt(Stmt::VarMfmaScale {
            var,
            a: a.into(),
            b: b.into(),
            scale_a: scale_a.into(),
            scale_b: scale_b.into(),
            opsel: 0,
        });
    }

    /// `var += v_mfma_scale_f32_32x32x64_f8f6f4(a, b; scale_a.byte[byte_a], scale_b.byte[byte_b])` in place: the scale operands are whole dwords and
    /// the instruction's op_sel / op_sel_hi pick the E8M0 byte (no VALU byte extraction; the vendor's form).
    #[allow(clippy::too_many_arguments)]
    pub fn var_mfma_scale_sel(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, scale_a: impl Into<Opnd>, scale_b: impl Into<Opnd>, byte_a: u8, byte_b: u8) {
        assert!(byte_a < 4 && byte_b < 4, "scale byte select is 0..3");
        self.push_stmt(Stmt::VarMfmaScale { var, a: a.into(), b: b.into(), scale_a: scale_a.into(), scale_b: scale_b.into(), opsel: byte_a | (byte_b << 4) });
    }

    /// `var += smfmac_16x16x128_fp8_fp8(a, b, idx)`: accumulate into a `Frag4` loop variable in place.
    pub fn var_smfmac_fp8(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, idx: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarSmfmac { var, a: a.into(), b: b.into(), idx: idx.into() });
    }

    /// `var += v_mfma_scale_f32_16x16x128_f8f6f4(a, b; scale_a.byte[byte_a], scale_b.byte[byte_b])` on a `Frag4` loop variable.
    #[allow(clippy::too_many_arguments)]
    pub fn var_mfma_scale16(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, scale_a: impl Into<Opnd>, scale_b: impl Into<Opnd>, byte_a: u8, byte_b: u8) {
        assert!(byte_a < 4 && byte_b < 4, "scale byte select is 0..3");
        self.push_stmt(Stmt::VarMfmaScale16 { var, a: a.into(), b: b.into(), scale_a: scale_a.into(), scale_b: scale_b.into(), opsel: byte_a | (byte_b << 4) });
    }

    /// Update several loop variables in place: the i-th `Seq::inout` port is bound to `vars[i]`.
    pub fn var_asm_n(&mut self, vars: &[VarId], block: amdgpu_safeasm::AsmBlock, inputs: &[Opnd]) {
        self.push_stmt(Stmt::VarAsmN { vars: vars.to_vec(), block: std::sync::Arc::new(block), inputs: inputs.to_vec() });
    }

    pub fn load_lds_tr_b8(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsTrB8 { addr: addr.into() })
    }

    pub fn store_lds_b128(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreLdsB128 { addr: addr.into(), value: value.into() });
    }

    /// `ds_write_b64`: a `Frag2` into LDS (8-byte aligned).
    pub fn store_lds_b64(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreLdsB64 { addr: addr.into(), value: value.into() });
    }

    /// Broadcast lane 0 of a vector value into a scalar register.
    pub fn read_first_lane(&mut self, v: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ReadFirstLane(v.into()))
    }

    /// `v_mov_b32_dpp quad_perm:[p0,p1,p2,p3]`: lane `l` gets lane `4 (l / 4) + perm[l % 4]` of `x` (a 32-bit value; e.g. `[1,0,3,2]` swaps
    /// neighbours, `[2,3,0,1]` swaps pairs). The result is a U32 bit pattern.
    pub fn dpp_quad_perm(&mut self, x: impl Into<Opnd>, perm: [u8; 4]) -> ValueId {
        assert!(perm.iter().all(|p| *p < 4), "dpp_quad_perm: lane selectors are 0..3");
        self.let_(Op::DppQuadPerm { x: x.into(), perm })
    }

    /// `s_waitcnt lgkmcnt(count)`.
    pub fn wait_lds(&mut self, count: u8) {
        self.push_stmt(Stmt::WaitLds { count });
    }

    /// `s_barrier`.
    pub fn wait_vm(&mut self, count: u8) {
        self.push_stmt(Stmt::WaitVm { count });
    }
    /// Scheduler directive, the analogue of ``__builtin_amdgcn_sched_group_barrier(mask, size, sync)``. Consecutive directives of one basic block form the issue
    /// pipeline of that block: group 1 = the next ``size`` instructions of the ``mask`` class (in program order, among those not placed yet), then group 2, ...;
    /// everything a selected instruction depends on is pulled in just before it, every register / memory / barrier dependence is respected, and the instructions
    /// no group selects follow in program order. Honoured by the scheduler of ``CompileProfile::Gemm`` (or explicit ``sched_opts``); otherwise ignored.
    /// Mask bits as in LLVM: ``SCHED_ALU`` 0x1 (any ALU incl. MFMA), ``SCHED_VALU`` 0x2, ``SCHED_SALU`` 0x4, ``SCHED_MFMA`` 0x8, ``SCHED_VMEM`` 0x10,
    /// ``SCHED_VMEM_READ`` 0x20 (buffer/global loads, LDS-DMA included), ``SCHED_VMEM_WRITE`` 0x40, ``SCHED_DS`` 0x80, ``SCHED_DS_READ`` 0x100, ``SCHED_DS_WRITE`` 0x200.
    pub fn sched_group_barrier(&mut self, mask: u32, size: u32, sync: u32) {
        self.push_stmt(Stmt::SchedGroup { mask, size, sync });
    }

    /// ``repeat`` times the group sequence ``groups`` (``(mask, size)`` pairs) as ``sched_group_barrier`` directives, e.g. one MFMA then two ``ds_read``s then one DMA:
    /// ``b.sched_pattern(&[(SCHED_MFMA, 1), (SCHED_DS_READ, 2), (SCHED_VMEM_READ, 1)], 64)``.
    pub fn sched_pattern(&mut self, groups: &[(u32, u32)], repeat: u32) {
        for _ in 0..repeat {
            for &(mask, size) in groups {
                self.sched_group_barrier(mask, size, 0);
            }
        }
    }
    /// 16-byte (dwords = 4) or 4-byte (dwords = 1) global -> LDS DMA load; see `Stmt::LoadGlobalLds`.
    pub fn load_global_lds(&mut self, base: impl Into<Opnd>, lds_addr: impl Into<Opnd>, byte_off: i32, dwords: u8) {
        self.push_stmt(Stmt::LoadGlobalLds { base: base.into(), lds_addr: lds_addr.into(), byte_off, dwords });
    }
    pub fn set_prio(&mut self, prio: u8) {
        self.push_stmt(Stmt::SetPrio { prio });
    }
    pub fn barrier(&mut self) {
        self.push_stmt(Stmt::Barrier);
    }
    /// Timing stamp: the low 32 bits of this wave's `s_memtime` (core-clock ticks), as a wave-uniform i32 value. Subtract two stamps (wrapping `sub`) for cycles;
    /// store them through a debug-buffer argument (`store_f32` of the value / bitcast) to read per-wave stamps on the host (`tools/stamp_host.py`, `docs/stamps.md`).
    /// Opt-in: a kernel without stamps compiles to the same instructions as before. Cost: one `s_memtime` + a full `s_waitcnt lgkmcnt(0)` + one `s_mov_b32`; put
    /// stamps at points where the kernel already drains (e.g. right after a tile's `vmcnt(0) lgkmcnt(0)` + barrier). Pinned in program order against other memory ops.
    /// Byte offset of this wave's stamp record in a debug buffer laid out `[workgroup][wave][slot]` of u32: `((wg * waves_per_wg + tid / 64) * slots) * 4`.
    /// `wg` is the linear workgroup index (for a 2-D grid pass `bx + by * grid_x`).
    pub fn stamp_rec_off(&mut self, wg: impl Into<Opnd>, slots: u32, waves_per_wg: u32) -> ValueId {
        let tid = self.let_(Op::WorkitemId);
        let c6 = self.c_i32(6);
        let wave = self.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(c6)));
        let cw = self.c_i32(waves_per_wg as i32);
        let wgw = self.mul_i32(wg.into(), cw);
        let idx = self.add_i32(wgw, wave);
        let cb = self.c_i32((slots * 4) as i32);
        self.mul_i32(idx, cb)
    }
    /// Store a stamp (or any i32) into slot `slot` of the record at `rec_off` (from `stamp_rec_off`) of the debug buffer `buf` (a pointer argument). All 64 lanes
    /// store the same dword to the same address (one coalesced write).
    pub fn stamp_store(&mut self, buf: impl Into<Opnd>, rec_off: impl Into<Opnd>, slot: u32, stamp: ValueId) {
        let p = self.ptr_add(buf, rec_off);
        let f = self.bitcast_f32(stamp);
        self.store_f32(p, slot * 4, f);
    }
    pub fn memtime_lo(&mut self) -> ValueId {
        let id = self.f.value_tys.len() as u32;
        self.let_(Op::MemTimeLo { id })
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

fn opnd_ty(f: &Function, o: Opnd) -> Result<DType, String> {
    match o {
        Opnd::Val(v) => f.value_ty(v).ok_or_else(|| format!("use of undefined value v{v}")),
        Opnd::Var(v) => f.var_ty(v).ok_or_else(|| format!("use of undefined var v{v}")),
        Opnd::I32(_) => Ok(DType::I32),
        Opnd::F32(_) => Ok(DType::F32),
    }
}

fn expect_int(f: &Function, o: Opnd, what: &str) -> Result<(), String> {
    let t = opnd_ty(f, o)?;
    if t.is_int() || t == DType::U32 {
        Ok(())
    } else {
        Err(format!("{what}: expected integer operand, found {}", t.name()))
    }
}

fn expect_float(f: &Function, o: Opnd, what: &str) -> Result<(), String> {
    let t = opnd_ty(f, o)?;
    if t.is_float() {
        Ok(())
    } else {
        Err(format!("{what}: expected float operand, found {}", t.name()))
    }
}

/// Shared checks of a MUBUF access: descriptor operand type, integer offsets, 12-bit immediate, supported width.
fn check_buffer_access(f: &Function, rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm: u32, dwords: u8) -> Vec<String> {
    let mut errs = Vec::new();
    match opnd_ty(f, rsrc) {
        Ok(DType::Rsrc) => {}
        other => errs.push(format!("buffer access: descriptor operand is {other:?}, expected rsrc (use buffer_rsrc)")),
    }
    if let Some(v) = voff {
        if let Err(e) = expect_int(f, v, "buffer voffset") {
            errs.push(e);
        }
    }
    if let Err(e) = expect_int(f, soff, "buffer soffset") {
        errs.push(e);
    }
    if imm > 4095 {
        errs.push(format!("buffer immediate offset {imm} does not fit the 12-bit field"));
    }
    if !matches!(dwords, 1 | 2 | 4) {
        errs.push(format!("buffer access of {dwords} dwords is not supported (1, 2 or 4)"));
    }
    errs
}
fn expect_ptr(f: &Function, o: Opnd, what: &str) -> Result<(), String> {
    let t = opnd_ty(f, o)?;
    if t == DType::Ptr {
        Ok(())
    } else {
        Err(format!("{what}: expected pointer operand, found {}", t.name()))
    }
}

/// Structural, scope and type validation. Returns every problem found.
pub fn verify(f: &Function) -> Vec<String> {
    let mut errs = Vec::new();
    check_var_slices(f, &mut errs);

    // Region stack: values visible in the current scope.
    let mut scopes: Vec<HashSet<ValueId>> = vec![HashSet::new()];
    let mut loop_depth = 0usize;
    let mut if_depth = 0usize;
    let mut loop_vars: Vec<Vec<VarId>> = Vec::new();
    // variables of loops that already ended: their final values may be updated by straight-line code after the loop (peeled iterations)
    let mut done_vars: Vec<VarId> = Vec::new();
    // Vars stay readable after their loop ends: the register holds the final
    // value carried by the last iteration.
    let mut vars_defined: HashSet<VarId> = HashSet::new();

    let visible = |scopes: &Vec<HashSet<ValueId>>, v: ValueId| {
        scopes.iter().any(|s| s.contains(&v))
    };

    for (idx, st) in f.stmts().iter().enumerate() {
        let here = |e: String| format!("stmt {idx}: {e}");
        match st {
            Stmt::Let { dst, op } => {
                for o in op.operands() {
                    if let Opnd::Val(v) = o {
                        if !visible(&scopes, v) {
                            errs.push(here(format!("operand v{v} is not visible here")));
                        }
                    }
                    if let Opnd::Var(v) = o {
                        if !vars_defined.contains(&v) {
                            errs.push(here(format!("var{v} used before its loop")));
                        }
                    }
                }
                match op {
                    Op::Asm { block, inputs } => {
                        if inputs.len() != block.inputs().len() {
                            errs.push(here(format!("asm block `{}`: {} inputs bound, {} ports", block.name(), inputs.len(), block.inputs().len())));
                        } else {
                            for (k, (p, o)) in block.inputs().iter().zip(inputs).enumerate() {
                                let ty = match o {
                                    Opnd::Val(v) => f.value_ty(*v),
                                    Opnd::Var(v) => f.var_ty(*v),
                                    Opnd::I32(_) => Some(DType::I32),
                                    Opnd::F32(_) => Some(DType::F32),
                                };
                                match ty {
                                    Some(t) if asm_port_accepts(p.class, t) => {}
                                    other => errs.push(here(format!("asm block `{}`: input {k} is {other:?}, the port is {:?}", block.name(), p.class))),
                                }
                            }
                        }
                        if block.outputs().is_empty() {
                            errs.push(here(format!("asm block `{}` has no output port", block.name())));
                        }
                    }
                    Op::AddI32(a, b)
                    | Op::MulI32(a, b)
                    | Op::ShlI32(a, b)
                    | Op::LshrI32(a, b)
                    | Op::AndI32(a, b)
                    | Op::MinI32(a, b)
                    | Op::XorI32(a, b)
                    | Op::CmpLtI32(a, b)
                    | Op::CmpGeI32(a, b) => {
                        if let Err(e) = expect_int(f, *a, "int op") {
                            errs.push(here(e));
                        }
                        if let Err(e) = expect_int(f, *b, "int op") {
                            errs.push(here(e));
                        }
                    }
                    Op::AddF32(a, b) | Op::SubF32(a, b) | Op::MulF32(a, b) | Op::MaxF32(a, b) | Op::CmpLtF32(a, b) => {
                        if let Err(e) = expect_float(f, *a, "float op") {
                            errs.push(here(e));
                        }
                        if let Err(e) = expect_float(f, *b, "float op") {
                            errs.push(here(e));
                        }
                    }
                    Op::FmaF32(a, b, c) => {
                        for o in [a, b, c] {
                            if let Err(e) = expect_float(f, *o, "fma") {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::Max3F32(a, b, c) => {
                        for o in [a, b, c] {
                            if let Err(e) = expect_float(f, *o, "max3") {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::ExpF32(a) | Op::Exp2F32(a) | Op::Log2F32(a) | Op::RcpF32(a) => {
                        if let Err(e) = expect_float(f, *a, "unary float") {
                            errs.push(here(e));
                        }
                    }
                    Op::CvtF32ToF16(a) => {
                        if let Err(e) = expect_float(f, *a, "cvt") {
                            errs.push(here(e));
                        }
                    }
                    Op::CvtF16ToF32(a) => {
                        if let Err(e) = expect_int(f, *a, "cvt") {
                            errs.push(here(e));
                        }
                    }
                    Op::NotI1(a) => {
                        if let Ok(t) = opnd_ty(f, *a) {
                            if t != DType::I1 {
                                errs.push(here(format!("not: expected i1, found {}", t.name())));
                            }
                        }
                    }
                    Op::Select { cond, a, b, ty } => {
                        match opnd_ty(f, *cond) {
                            Ok(DType::I1) => {}
                            Ok(t) => errs.push(here(format!("select: cond must be i1, found {}", t.name()))),
                            Err(e) => errs.push(here(e)),
                        }
                        for o in [a, b] {
                            if let Ok(t) = opnd_ty(f, *o) {
                                if t != *ty {
                                    errs.push(here(format!("select: branch type {} != {}", t.name(), ty.name())));
                                }
                            }
                        }
                    }
                    Op::PtrAdd { base, byte_off } => {
                        if let Err(e) = expect_ptr(f, *base, "ptradd") {
                            errs.push(here(e));
                        }
                        if let Err(e) = expect_int(f, *byte_off, "ptradd") {
                            errs.push(here(e));
                        }
                    }
                    Op::AtomicAddU32 { base, value, .. } => {
                        if let Err(e) = expect_ptr(f, *base, "atomic add") {
                            errs.push(here(e));
                        }
                        if let Err(e) = expect_int(f, *value, "atomic add value") {
                            errs.push(here(e));
                        }
                    }
                    Op::LoadU16 { base, .. } | Op::LoadF32 { base, .. } | Op::LoadF32Coherent { base, .. } => {
                        if let Err(e) = expect_ptr(f, *base, "load") {
                            errs.push(here(e));
                        }
                    }
                    Op::BufferRsrc { base, .. } => {
                        if let Err(e) = expect_ptr(f, *base, "buffer descriptor base") {
                            errs.push(here(e));
                        }
                    }
                    Op::BufferRsrcDyn { base, num_records } => {
                        if let Err(e) = expect_ptr(f, *base, "buffer descriptor base") {
                            errs.push(here(e));
                        }
                        if let Err(e) = expect_int(f, *num_records, "buffer descriptor num_records") {
                            errs.push(here(e));
                        }
                    }
                    Op::BufferLoad { rsrc, voff, soff, imm, dwords } => {
                        for e in check_buffer_access(f, *rsrc, *voff, *soff, *imm, *dwords) {
                            errs.push(here(e));
                        }
                    }
                    Op::BufferLoadB128x2 { rsrc, voff, soff, imm_lo, imm_hi } => {
                        for imm in [*imm_lo, *imm_hi] {
                            for e in check_buffer_access(f, *rsrc, *voff, *soff, imm, 4) {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::Frag2(a, b) => {
                        for o in [a, b] {
                            if let Err(e) = expect_int(f, *o, "frag2") {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::Frag4(a, b, c, d) => {
                        for o in [a, b, c, d] {
                            if let Err(e) = expect_int(f, *o, "frag4") {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::Frag2Lane { frag, index } => {
                        match opnd_ty(f, *frag) {
                            Ok(DType::Frag2) => {}
                            Ok(t) => errs.push(here(format!("frag2lane: expected frag2, found {}", t.name()))),
                            Err(e) => errs.push(here(e)),
                        }
                        if *index > 1 {
                            errs.push(here("frag2lane: index out of range".to_string()));
                        }
                    }
                    Op::Frag4Lane { frag, index } => {
                        match opnd_ty(f, *frag) {
                            Ok(DType::Frag4) => {}
                            Ok(t) => errs.push(here(format!("frag4lane: expected frag4, found {}", t.name()))),
                            Err(e) => errs.push(here(e)),
                        }
                        if *index > 3 {
                            errs.push(here("frag4lane: index out of range".to_string()));
                        }
                    }
                    Op::MfmaF16 { a, b, c } => {
                        for (o, want, what) in [(*a, DType::Frag2, "a"), (*b, DType::Frag2, "b"), (*c, DType::Frag4, "c")] {
                            if let Ok(t) = opnd_ty(f, o) {
                                if t != want {
                                    errs.push(here(format!("mfma: operand {what} is {}, expected {}", t.name(), want.name())));
                                }
                            }
                        }
                    }
                    Op::BitcastI32(_) => {}
                    Op::BitcastF32(a) => {
                        if let Err(e) = expect_int(f, *a, "bitcast") {
                            errs.push(here(e));
                        }
                    }
                    Op::MfmaF16Wide { a, b, c } => {
                        for (o, want, what) in [
                            (*a, DType::Frag2, "a"),
                            (*b, DType::Frag2, "b"),
                            (*c, DType::Frag16, "c"),
                        ] {
                            if let Ok(t) = opnd_ty(f, o) {
                                if t != want {
                                    errs.push(here(format!(
                                        "wide mfma: operand {what} is {}, expected {}",
                                        t.name(),
                                        want.name()
                                    )));
                                }
                            }
                        }
                    }
                    Op::Frag16Lane { frag, index } => {
                        match opnd_ty(f, *frag) {
                            Ok(DType::Frag16) | Ok(DType::Frag8) | Ok(DType::Frag4) | Ok(DType::Frag2) => {}
                            Ok(t) => errs.push(here(format!(
                                "frag16lane: expected a fragment, found {}",
                                t.name()
                            ))),
                            Err(e) => errs.push(here(e)),
                        }
                        if *index > 15 {
                            errs.push(here("frag16lane: index out of range".to_string()));
                        }
                    }
                    Op::FragZero { .. } => {}
                    Op::CvtPknormU16(a, b) => {
                        for o in [a, b] {
                            if let Err(e) = expect_float(f, *o, "pknorm") {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::CvtPkBf16F32(a, b) => {
                        for o in [a, b] {
                            if let Err(e) = expect_float(f, *o, "cvt_pk_bf16") {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::CatFrag8(a, b) => {
                        for o in [a, b] {
                            match opnd_ty(f, *o) {
                                Ok(DType::Frag4) => {}
                                Ok(t) => errs.push(here(format!(
                                    "catfrag8: expected frag4, found {}",
                                    t.name()
                                ))),
                                Err(e) => errs.push(here(e)),
                            }
                        }
                    }
                    Op::Frag8(a, b, c, d, e, ff, g, h) => {
                        for o in [a, b, c, d, e, ff, g, h] {
                            if let Err(err) = expect_int(f, *o, "frag8") {
                                errs.push(here(err));
                            }
                        }
                    }
                    Op::MfmaFp8_32x32x64 { a, b, c } => {
                        for (o, want, what) in [(*a, DType::Frag8, "a"), (*b, DType::Frag8, "b"), (*c, DType::Frag16, "c")] {
                            if let Ok(t) = opnd_ty(f, o) {
                                if t != want {
                                    errs.push(here(format!("fp8 mfma: operand {what} is {}, expected {}", t.name(), want.name())));
                                }
                            }
                        }
                    }
                    Op::MfmaScaleFp8_32x32x64 { a, b, c, scale_a, scale_b } => {
                        for (o, want, what) in [
                            (*a, DType::Frag8, "a"),
                            (*b, DType::Frag8, "b"),
                            (*c, DType::Frag16, "c"),
                        ] {
                            if let Ok(t) = opnd_ty(f, o) {
                                if t != want {
                                    errs.push(here(format!(
                                        "scaled mfma: operand {what} is {}, expected {}",
                                        t.name(),
                                        want.name()
                                    )));
                                }
                            }
                        }
                        for (o, what) in [(*scale_a, "scale_a"), (*scale_b, "scale_b")] {
                            if let Err(err) = expect_int(f, o, what) {
                                errs.push(here(err));
                            }
                        }
                    }
                    Op::LoadLdsTrB8 { addr } => {
                        if let Err(e) = expect_int(f, *addr, "lds tr_b8 load address") {
                            errs.push(here(e));
                        }
                    }
                    Op::LoadLdsB128 { addr } => {
                        if let Err(e) = expect_int(f, *addr, "lds128 load address") {
                            errs.push(here(e));
                        }
                    }
                    Op::LoadLdsB64 { addr } => {
                        if let Err(e) = expect_int(f, *addr, "lds64 load address") {
                            errs.push(here(e));
                        }
                    }
                    Op::LoadLdsB32 { addr } => {
                        if let Err(e) = expect_int(f, *addr, "lds load address") {
                            errs.push(here(e));
                        }
                    }
                    Op::ReadFirstLane(a) => {
                        if let Err(e) = expect_int(f, *a, "readfirstlane") {
                            errs.push(here(e));
                        }
                    }
                    _ => {}
                }
                // Scope: value becomes visible here.
                scopes.last_mut().unwrap().insert(*dst);
            }
            Stmt::StoreU16 { base, value, .. } => {
                if let Err(e) = expect_ptr(f, *base, "store") {
                    errs.push(here(e));
                }
                if let Err(e) = expect_int(f, *value, "store u16") {
                    errs.push(here(e));
                }
                if let Opnd::Var(v) = value {
                    if !vars_defined.contains(v) {
                        errs.push(here(format!("var{v} used before its loop")));
                    }
                }
            }
            Stmt::StoreF32x4 { base, .. } | Stmt::StoreF32x2 { base, .. } => {
                if let Err(e) = expect_ptr(f, *base, "store") {
                    errs.push(here(e));
                }
            }
            Stmt::BufferStore { rsrc, voff, soff, imm, dwords, value } => {
                for e in check_buffer_access(f, *rsrc, *voff, *soff, *imm, *dwords) {
                    errs.push(here(e));
                }
                let want = match dwords {
                    1 => DType::F32,
                    2 => DType::Frag2,
                    _ => DType::Frag4,
                };
                match opnd_ty(f, *value) {
                    Ok(t) if t == want || (*dwords == 1 && t == DType::U32) => {}
                    other => errs.push(here(format!("buffer_store x{dwords} value is {other:?}, expected {}", want.name()))),
                }
            }
            Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, imm, dwords } => {
                if !matches!(dwords, 1 | 4) {
                    errs.push(here(format!("buffer_load lds supports 1 or 4 dwords, not {dwords}")));
                }
                for e in check_buffer_access(f, *rsrc, *voff, *soff, *imm, *dwords) {
                    errs.push(here(e));
                }
                if let Err(e) = expect_int(f, *lds_addr, "buffer_load lds address") {
                    errs.push(here(e));
                }
            }
            Stmt::FenceRelease { .. } | Stmt::FenceAcquire { .. } => {}
            Stmt::StoreF32Wt { base, value, .. } => {
                if let Err(e) = expect_ptr(f, *base, "store") {
                    errs.push(here(e));
                }
                if let Err(e) = expect_float(f, *value, "store.wt f32") {
                    errs.push(here(e));
                }
            }
            Stmt::StoreF32 { base, value, .. } => {
                if let Err(e) = expect_ptr(f, *base, "store") {
                    errs.push(here(e));
                }
                if let Err(e) = expect_float(f, *value, "store f32") {
                    errs.push(here(e));
                }
                if let Opnd::Var(v) = value {
                    if !vars_defined.contains(v) {
                        errs.push(here(format!("var{v} used before its loop")));
                    }
                }
            }
            Stmt::StoreLdsB32 { addr, value } => {
                if let Err(e) = expect_int(f, *addr, "lds store address") {
                    errs.push(here(e));
                }
                if let Err(e) = expect_int(f, *value, "lds store value") {
                    errs.push(here(e));
                }
                for o in [addr, value] {
                    if let Opnd::Val(v) = o {
                        if !visible(&scopes, *v) {
                            errs.push(here(format!("operand v{v} is not visible here")));
                        }
                    }
                }
            }
            Stmt::StoreLdsB64 { addr, value } => {
                if let Err(e) = expect_int(f, *addr, "lds64 store address") {
                    errs.push(here(e));
                }
                match opnd_ty(f, *value) {
                    Ok(DType::Frag2) => {}
                    Ok(t) => errs.push(here(format!("lds64 store: expected frag2, found {}", t.name()))),
                    Err(e) => errs.push(here(e)),
                }
            }
            Stmt::StoreLdsB128 { addr, value } => {
                if let Err(e) = expect_int(f, *addr, "lds128 store address") {
                    errs.push(here(e));
                }
                match opnd_ty(f, *value) {
                    Ok(DType::Frag4) => {}
                    Ok(t) => errs.push(here(format!("lds128 store: expected frag4, found {}", t.name()))),
                    Err(e) => errs.push(here(e)),
                }
            }
            Stmt::WaitLds { .. } | Stmt::Barrier | Stmt::SetPrio { .. } | Stmt::SchedGroup { .. } | Stmt::WaitVm { .. } | Stmt::StageEnd => {}
            Stmt::StageBegin { name, contract } => {
                for v in contract.live_in.iter().chain(contract.live_out.iter()).flat_map(|x| x.values.iter()) {
                    if let Err(e) = opnd_ty(f, Opnd::Val(*v)) {
                        errs.push(here(format!("stage {name}: contract value v{v}: {e}")));
                    }
                }
            }
            Stmt::Assume { target, .. } => {
                if let Err(e) = expect_int(f, *target, "assume target") {
                    errs.push(here(e));
                }
            }
            Stmt::TileProduce { tile, slot, version } | Stmt::TileConsume { tile, slot, version } => {
                if !f.contract.tiles.iter().any(|t| &t.name == tile) {
                    errs.push(here(format!("tile marker names undeclared tile `{tile}` (lds_tile / lds_ring)")));
                }
                for (o, what) in [(*slot, "tile slot"), (*version, "tile version")] {
                    if let Err(e) = expect_int(f, o, what) {
                        errs.push(here(e));
                    }
                }
            }
            Stmt::TileEnd => {}
            Stmt::LoadGlobalLds { base, lds_addr, dwords, .. } => {
                if let Err(e) = expect_ptr(f, *base, "dma base") {
                    errs.push(here(e));
                }
                if let Err(e) = expect_int(f, *lds_addr, "dma lds address") {
                    errs.push(here(e));
                }
                if *dwords != 1 && *dwords != 4 {
                    errs.push(here("dma: dwords must be 1 or 4".to_string()));
                }
            }
            Stmt::SetVar { var, value } => {
                // Any enclosing loop's variable may be updated; an inner loop can
                // therefore update an outer loop's accumulator.
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} assigned outside its loop")));
                }
                if let (Some(vt), Ok(t)) = (f.var_ty(*var), opnd_ty(f, *value)) {
                    if vt != t && !(vt.is_int() && t.is_int()) {
                        errs.push(here(format!("var{var} type {} != {}", vt.name(), t.name())));
                    }
                }
                if let Opnd::Val(v) = value {
                    if !visible(&scopes, *v) {
                        errs.push(here(format!("operand v{v} is not visible here")));
                    }
                }
            }
            Stmt::VarScaleF32 { var, scalar } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} scaled outside its loop")));
                }
                if f.var_ty(*var) != Some(DType::Frag16) {
                    errs.push(here(format!("var{var} must be a frag16 to be scaled in place")));
                }
                if let Err(e) = expect_float(f, *scalar, "var scale") {
                    errs.push(here(e));
                }
            }
            Stmt::VarSetLane { var, index, value } => {
                let lanes = match f.var_ty(*var) {
                    Some(DType::Frag16) => 16,
                    Some(DType::Frag8) => 8,
                    Some(DType::Frag4) => 4,
                    Some(DType::Frag2) => 2,
                    _ => 0,
                };
                if *index as usize >= lanes {
                    errs.push(here(format!("var{var}.lane[{index}]: not a lane of a fragment variable")));
                }
                match opnd_ty(f, *value) {
                    Ok(DType::I32 | DType::U32 | DType::F32) => {}
                    Ok(t) => errs.push(here(format!("var set lane: expected a 32-bit value, found {}", t.name()))),
                    Err(e) => errs.push(here(e)),
                }
            }
            Stmt::VarScaleLane { var, .. } => {
                if !matches!(f.var_ty(*var), Some(DType::Frag16) | Some(DType::Frag4)) {
                    errs.push(here(format!("var{var} must be a frag16 or frag4")));
                }
            }
            Stmt::VarAddPk { var, .. } | Stmt::VarMfmaF16Z { var, .. } => {
                if f.var_ty(*var) != Some(DType::Frag16) {
                    errs.push(here(format!("var{var} must be a frag16")));
                }
            }
            Stmt::VarScalePk { var, pair } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} scaled outside its loop")));
                }
                if f.var_ty(*var) != Some(DType::Frag16) {
                    errs.push(here(format!("var{var} must be a frag16 to be scaled in place")));
                }
                if let Ok(t) = opnd_ty(f, *pair) {
                    if t != DType::Frag2 {
                        errs.push(here(format!("var scale pair: expected frag2, found {}", t.name())));
                    }
                }
            }
            Stmt::VarMfmaScaleC { var, .. } | Stmt::VarMfmaFp8Z { var, .. } | Stmt::VarMfmaFp8C { var, .. } => {
                if f.var_ty(*var) != Some(DType::Frag16) {
                    errs.push(here(format!("var{var} must be a frag16 accumulator")));
                }
            }
            Stmt::VarMfmaFp8 { var, a, b } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} accumulated outside its loop")));
                }
                if f.var_ty(*var) != Some(DType::Frag16) {
                    errs.push(here(format!("var{var} must be a frag16 accumulator")));
                }
                for o in [a, b] {
                    match opnd_ty(f, *o) {
                        Ok(DType::Frag8) => {}
                        other => errs.push(here(format!("fp8 mfma operand is {other:?}, expected frag8"))),
                    }
                }
            }
            Stmt::VarMfmaScale { var, a, b, scale_a, scale_b, .. } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} accumulated outside its loop")));
                }
                if f.var_ty(*var) != Some(DType::Frag16) {
                    errs.push(here(format!("var{var} must be a frag16 accumulator")));
                }
                for (o, ty) in [(a, DType::Frag8), (b, DType::Frag8)] {
                    match opnd_ty(f, *o) {
                        Ok(t) if t == ty => {}
                        Ok(t) => errs.push(here(format!("mfma operand is {}, expected {}", t.name(), ty.name()))),
                        Err(e) => errs.push(here(e)),
                    }
                }
                for o in [scale_a, scale_b] {
                    if let Err(e) = expect_int(f, *o, "mfma scale") {
                        errs.push(here(e));
                    }
                }
            }
            Stmt::VarAsm { var, block, inputs } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} updated by an asm block outside its loop")));
                }
                let ports: Vec<_> = block.inputs().iter().collect();
                match ports.first() {
                    Some(p0) if p0.mutable => {
                        if !f.var_ty(*var).map_or(false, |t| asm_port_accepts(p0.class, t)) {
                            errs.push(here(format!("asm block `{}`: in/out port {:?} does not match var{var}", block.name(), p0.class)));
                        }
                    }
                    _ => errs.push(here(format!("asm block `{}` has no leading in/out port for var{var}", block.name()))),
                }
                if ports.len() != inputs.len() + 1 || ports.iter().skip(1).any(|p| p.mutable) {
                    errs.push(here(format!("asm block `{}`: {} inputs bound for {} remaining ports", block.name(), inputs.len(), ports.len().saturating_sub(1))));
                } else {
                    for (k, (p, o)) in ports.iter().skip(1).zip(inputs).enumerate() {
                        match opnd_ty(f, *o) {
                            Ok(t) if asm_port_accepts(p.class, t) => {}
                            other => errs.push(here(format!("asm block `{}`: input {k} is {other:?}, the port is {:?}", block.name(), p.class))),
                        }
                    }
                }
            }
            Stmt::VarAsmN { vars, block, inputs } => {
                let muts: Vec<_> = block.inputs().iter().filter(|p| p.mutable).collect();
                if muts.len() != vars.len() {
                    errs.push(here(format!("asm block `{}` has {} in/out ports, {} variables bound", block.name(), muts.len(), vars.len())));
                } else {
                    for (k, (p, v)) in muts.iter().zip(vars).enumerate() {
                        let ok = loop_vars.iter().rev().any(|vs| vs.contains(v)) || done_vars.contains(v);
                        if !ok {
                            errs.push(here(format!("var{v} updated by an asm block outside its loop")));
                        }
                        if !f.var_ty(*v).map_or(false, |t| asm_port_accepts(p.class, t)) {
                            errs.push(here(format!("asm block `{}`: in/out port {k} ({:?}) does not match var{v}", block.name(), p.class)));
                        }
                    }
                }
                let ro: Vec<_> = block.inputs().iter().filter(|p| !p.mutable).collect();
                if ro.len() != inputs.len() {
                    errs.push(here(format!("asm block `{}`: {} inputs bound for {} read-only ports", block.name(), inputs.len(), ro.len())));
                } else {
                    for (k, (p, o)) in ro.iter().zip(inputs).enumerate() {
                        match opnd_ty(f, *o) {
                            Ok(t) if asm_port_accepts(p.class, t) => {}
                            other => errs.push(here(format!("asm block `{}`: input {k} is {other:?}, the port is {:?}", block.name(), p.class))),
                        }
                    }
                }
            }
            Stmt::VarMfmaScale16 { var, a, b, scale_a, scale_b, .. } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} accumulated outside its loop")));
                }
                if f.var_ty(*var) != Some(DType::Frag4) {
                    errs.push(here(format!("var{var} must be a frag4 accumulator (16x16 MFMA)")));
                }
                for (o, ty) in [(a, DType::Frag8), (b, DType::Frag8)] {
                    match opnd_ty(f, *o) {
                        Ok(t) if t == ty => {}
                        Ok(t) => errs.push(here(format!("mfma 16x16x128 operand is {}, expected {}", t.name(), ty.name()))),
                        Err(e) => errs.push(here(e)),
                    }
                }
                for (o, what) in [(scale_a, "scale_a"), (scale_b, "scale_b")] {
                    if let Err(e) = expect_int(f, *o, what) {
                        errs.push(here(e));
                    }
                }
            }
            Stmt::VarSmfmac { var, a, b, idx } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} accumulated outside its loop")));
                }
                if f.var_ty(*var) != Some(DType::Frag4) {
                    errs.push(here(format!("var{var} must be a frag4 accumulator")));
                }
                for (o, ty) in [(a, DType::Frag4), (b, DType::Frag8)] {
                    match opnd_ty(f, *o) {
                        Ok(t) if t == ty => {}
                        Ok(t) => errs.push(here(format!("smfmac operand is {}, expected {}", t.name(), ty.name()))),
                        Err(e) => errs.push(here(e)),
                    }
                }
                if let Err(e) = expect_int(f, *idx, "smfmac index") {
                    errs.push(here(e));
                }
            }
            Stmt::BeginLoop { vars, init } => {
                if vars.len() != init.len() {
                    errs.push(here("begin_loop: vars/init length mismatch".into()));
                }
                for (i, var) in vars.iter().enumerate() {
                    if let Some(Opnd::Val(v)) = init.get(i) {
                        if !visible(&scopes, *v) {
                            errs.push(here(format!("loop init v{v} is not visible")));
                        }
                    }
                    if let (Some(vt), Some(e)) = (f.var_ty(*var), init.get(i)) {
                        if let Ok(t) = opnd_ty(f, *e) {
                            if vt != t && !(vt.is_int() && t.is_int()) {
                                errs.push(here(format!("loop var{var} type {} != init {}", vt.name(), t.name())));
                            }
                        }
                    }
                }
                loop_vars.push(vars.clone());
                vars_defined.extend(vars.iter().copied());
                loop_depth += 1;
                scopes.push(HashSet::new());
            }
            Stmt::BreakIf { cond } => {
                if loop_depth == 0 {
                    errs.push(here("break outside loop".into()));
                }
                if let Ok(t) = opnd_ty(f, *cond) {
                    if t != DType::I1 {
                        errs.push(here(format!("break: expected i1, found {}", t.name())));
                    }
                }
                if let Opnd::Val(v) = cond {
                    if !visible(&scopes, *v) {
                        errs.push(here(format!("operand v{v} is not visible here")));
                    }
                }
            }
            Stmt::EndLoop => {
                if loop_depth == 0 {
                    errs.push(here("endloop without loop".into()));
                } else {
                    loop_depth -= 1;
                    if let Some(vs) = loop_vars.pop() {
                    done_vars.extend(vs);
                }
                    // Values (including loop variables) stay visible after the
                    // loop: a loop variable then holds its last value, which is
                    // what the lowering leaves in its register.
                    let inner = scopes.pop().unwrap_or_default();
                    if let Some(parent) = scopes.last_mut() {
                        parent.extend(inner);
                    } else {
                        errs.push(here("endloop without enclosing scope".into()));
                    }
                }
            }
            Stmt::BeginIf { cond } => {
                match opnd_ty(f, *cond) {
                    Ok(DType::I1) => {}
                    Ok(t) => errs.push(here(format!("if: expected i1, found {}", t.name()))),
                    Err(e) => errs.push(here(e)),
                }
                if_depth += 1;
                scopes.push(HashSet::new());
            }
            Stmt::Else => {
                if if_depth == 0 {
                    errs.push(here("else without if".into()));
                }
                scopes.pop();
                scopes.push(HashSet::new());
            }
            Stmt::EndIf => {
                if if_depth == 0 {
                    errs.push(here("endif without if".into()));
                } else {
                    if_depth -= 1;
                    scopes.pop();
                }
            }
        }
    }

    if loop_depth != 0 {
        errs.push("unterminated loop".to_string());
    }
    if if_depth != 0 {
        errs.push("unterminated if".to_string());
    }
    errs
}
/// Kernel definitions built with this DSL.
pub mod expr;
pub mod print;
pub use expr::{ev, ox, varx};
pub use expr::{Cond, Ptr, F32, I32};

/// Every `Opnd` a statement reads (loop variables read implicitly by in-place `Var*` statements are not listed).
/// The value / variable operands a statement reads (structure and contract markers: none).
pub fn stmt_operands(st: &Stmt) -> Vec<Opnd> {
    match st {
        Stmt::Let { op, .. } => op.operands(),
        Stmt::StoreU16 { base, value, .. } | Stmt::StoreF32 { base, value, .. } | Stmt::StoreF32x4 { base, value, .. } | Stmt::StoreF32Wt { base, value, .. } | Stmt::StoreF32x2 { base, value, .. } => vec![*base, *value],
        Stmt::FenceRelease { .. } | Stmt::FenceAcquire { .. } => vec![],
        Stmt::StoreLdsB32 { addr, value } | Stmt::StoreLdsB128 { addr, value } | Stmt::StoreLdsB64 { addr, value } => vec![*addr, *value],
        Stmt::LoadGlobalLds { base, lds_addr, .. } => vec![*base, *lds_addr],
        Stmt::BufferStore { rsrc, voff, soff, value, .. } => {
            let mut v = vec![*rsrc];
            v.extend(voff.iter().copied());
            v.push(*soff);
            v.push(*value);
            v
        }
        Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, .. } => {
            let mut v = vec![*rsrc];
            v.extend(voff.iter().copied());
            v.push(*soff);
            v.push(*lds_addr);
            v
        }
        Stmt::SetVar { value, .. } => vec![*value],
        Stmt::VarScaleF32 { scalar, .. } | Stmt::VarScaleLane { scalar, .. } => vec![*scalar],
        Stmt::VarSetLane { var, value, .. } => vec![Opnd::Var(*var), *value],
        Stmt::VarScalePk { pair, .. } | Stmt::VarAddPk { pair, .. } => vec![*pair],
        Stmt::VarMfmaF16Z { a, b, .. } => vec![*a, *b],
        Stmt::VarMfmaScale { a, b, scale_a, scale_b, .. } => {
            vec![*a, *b, *scale_a, *scale_b]
        }
        Stmt::VarMfmaScaleC { a, b, c, scale_a, scale_b, .. } => vec![*a, *b, *c, *scale_a, *scale_b],
        Stmt::VarMfmaFp8 { a, b, .. } | Stmt::VarMfmaFp8Z { a, b, .. } => vec![*a, *b],
        Stmt::VarMfmaFp8C { a, b, c, .. } => vec![*a, *b, *c],
        Stmt::VarSmfmac { a, b, idx, .. } => vec![*a, *b, *idx],
        Stmt::VarMfmaScale16 { a, b, scale_a, scale_b, .. } => vec![*a, *b, *scale_a, *scale_b],
        Stmt::VarAsm { inputs, .. } | Stmt::VarAsmN { inputs, .. } => inputs.clone(),
        Stmt::BeginLoop { init, .. } => init.clone(),
        Stmt::BreakIf { cond } | Stmt::BeginIf { cond } => vec![*cond],
        _ => Vec::new(),
    }
}

/// Semantics of loop variables: a read of a loop variable (`Opnd::Var`) **after a `set_var` of the same variable in
/// the same iteration** is easy to get wrong - the author usually expects the value just written, but the variable is
/// loop-carried state whose value after a `set_var` is not guaranteed by the contract (docs/archive/verification.md); use a
/// plain SSA value (the operand you passed to `set_var`) for intra-iteration producer/consumer pairs.
/// This lint reports such reads conservatively (a `set_var` inside an `if` counts). Cross-iteration reads (read before
/// the `set_var`, the common case), reads of the loop condition value held in an SSA value, and reads after the loop are
/// not reported. Kernels that really intend this can call `FnBuilder::allow_var_rereads()`.
pub fn lint(f: &Function) -> Vec<String> {
    if f.allow_var_rereads {
        return Vec::new();
    }
    struct L {
        vars: HashSet<VarId>,
        set_so_far: HashSet<VarId>,
    }
    let mut stack: Vec<L> = Vec::new();
    let mut out = Vec::new();
    for (idx, st) in f.stmts().iter().enumerate() {
        for o in stmt_operands(st) {
            if let Opnd::Var(v) = o {
                if let Some(l) = stack.iter().rev().find(|l| l.vars.contains(&v)) {
                    if l.set_so_far.contains(&v) {
                        out.push(format!(
                            "stmt {idx}: loop variable var{} is read after set_var(var{}) in the same iteration; the value of such a read is not defined by the DSL contract (the lowering usually returns the value just written, other paths can return the iteration-start value) - use the SSA value passed to set_var",
                            v.0, v.0
                        ));
                    }
                }
            }
        }
        match st {
            Stmt::BeginLoop { vars, .. } => stack.push(L { vars: vars.iter().copied().collect(), set_so_far: HashSet::new() }),
            Stmt::EndLoop => {
                stack.pop();
            }
            Stmt::SetVar { var, .. } => {
                if let Some(l) = stack.iter_mut().rev().find(|l| l.vars.contains(var)) {
                    l.set_so_far.insert(*var);
                }
            }
            _ => {}
        }
    }
    out
}
/// Value type of a safe-asm block's single output port (`I32` for one dword: callers bitcast as needed).
pub fn asm_out_ty(b: &amdgpu_safeasm::AsmBlock) -> DType {
    use amdgpu_safeasm::AClass;
    match b.outputs().first().map(|p| p.class) {
        Some(AClass::V2) => DType::Frag2,
        Some(AClass::V4) => DType::Frag4,
        Some(AClass::V8) => DType::Frag8,
        Some(AClass::V16) => DType::Frag16,
        _ => DType::I32,
    }
}

/// The DSL type a safe-asm input port of class `c` accepts (one dword: any 32-bit value).
pub fn asm_port_accepts(c: amdgpu_safeasm::AClass, ty: DType) -> bool {
    use amdgpu_safeasm::AClass;
    match c {
        AClass::V1 => matches!(ty, DType::I32 | DType::U32 | DType::F32 | DType::F16),
        AClass::V2 => matches!(ty, DType::Frag2 | DType::Ptr),
        AClass::V4 => ty == DType::Frag4,
        AClass::V8 => ty == DType::Frag8,
        AClass::V16 => ty == DType::Frag16,
        AClass::S1 | AClass::S2 => false,
    }
}

/// The variables a statement writes in place.
pub fn stmt_var_writes(st: &Stmt) -> Vec<VarId> {
    match st {
        Stmt::SetVar { var, .. }
        | Stmt::VarScaleF32 { var, .. }
        | Stmt::VarScalePk { var, .. }
        | Stmt::VarAddPk { var, .. }
        | Stmt::VarMfmaF16Z { var, .. }
        | Stmt::VarScaleLane { var, .. }
        | Stmt::VarSetLane { var, .. }
        | Stmt::VarMfmaScale { var, .. }
        | Stmt::VarMfmaScale16 { var, .. }
        | Stmt::VarMfmaScaleC { var, .. }
        | Stmt::VarMfmaFp8 { var, .. }
        | Stmt::VarMfmaFp8Z { var, .. }
        | Stmt::VarMfmaFp8C { var, .. }
        | Stmt::VarSmfmac { var, .. }
        | Stmt::VarAsm { var, .. } => vec![*var],
        Stmt::VarAsmN { vars, .. } | Stmt::BeginLoop { vars, .. } => vars.clone(),
        _ => Vec::new(),
    }
}

/// `VarSlice8` views: a slice of `var` must not be used after an in-place write of `var` (it would read the new registers). Over the linear
/// statement order (a write inside a loop body that precedes the slice's definition in the next iteration is a new slice value).
fn check_var_slices(f: &Function, errs: &mut Vec<String>) {
    let mut live: std::collections::HashMap<ValueId, (VarId, bool)> = std::collections::HashMap::new();
    for (idx, st) in f.stmts().iter().enumerate() {
        for o in stmt_operands(st) {
            if let Opnd::Val(v) = o {
                if let Some((var, stale)) = live.get(&v) {
                    if *stale {
                        errs.push(format!("stmt {idx}: v{v} (a slice of var{var}) is used after var{var} was written"));
                    }
                }
            }
        }
        if let Stmt::Let { dst, op: Op::VarSlice8 { var, half } } = st {
            if f.var_ty(*var) != Some(DType::Frag16) || *half > 1 {
                errs.push(format!("stmt {idx}: var_slice8 needs a frag16 variable and half 0 / 1"));
            }
            live.insert(*dst, (*var, false));
        }
        for w in stmt_var_writes(st) {
            for (_, (var, stale)) in live.iter_mut() {
                if *var == w {
                    *stale = true;
                }
            }
        }
    }
}
