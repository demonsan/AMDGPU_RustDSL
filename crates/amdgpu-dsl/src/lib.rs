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

/// Unary VALU ops (P1, docs/op-coverage-tileops.md). `SinF32` / `CosF32` take the angle in REVOLUTIONS (sin(2 pi x)), as the hardware does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOp { FloorF32, CeilF32, TruncF32, RndneF32, FractF32, RsqF32, SqrtF32, SinF32, CosF32, CvtI32F32, CvtU32F32, CvtF32I32, CvtF32U32, CvtF32Ubyte0, CvtF32Ubyte1, CvtF32Ubyte2, CvtF32Ubyte3, Ffbl, Ffbh }

/// Binary VALU ops. `AshrI32(a, b)` = a >> b (arithmetic); `MulHi*` = high 32 bits of the 64-bit product; `MulU32U24` = low 24 bits x low 24 bits;
/// `Pk*F16` operate on two f16 halves of a dword (a `U32` container).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinOp { MinF32, MaxI32, MaxU32, MinU32, SubU32, MulU32U24, AshrI32, MulHiU32, MulHiI32, OrB32, PkAddF16, PkMulF16, PkMaxF16, PkMinF16, MbcntLo, MbcntHi, BcntAdd }

/// Ternary VALU ops. `MadU32U24(a, b, c)` = a[23:0] * b[23:0] + c; `LshlAddU32(a, s, c)` = (a << s) + c; `Add3U32` = a + b + c;
/// `BfeU32 / BfeI32(x, off, width)` = bit-field extract; `BfiB32(mask, a, b)` = (mask & a) | (!mask & b); `PkFmaF16` on f16 pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TriOp { Min3F32, Med3F32, Min3I32, Max3I32, Med3I32, Min3U32, Max3U32, Med3U32, MadU32U24, LshlAddU32, Add3U32, BfeU32, BfeI32, BfiB32, PkFmaF16 }

/// Comparison predicates. Float: `Lt/Le/Gt/Ge/Eq/Lg` are ORDERED (false when either is NaN), `Ne` is unordered-or-not-equal (IEEE `!=`, true for NaN),
/// `Uno` = either is NaN, `Ord` = neither is NaN. `Lg`, `Uno`, `Ord` are float-only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpPred { Lt, Le, Gt, Ge, Eq, Ne, Lg, Uno, Ord }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CmpTy { F32, I32, U32 }

/// Width / type of a memory access (P1 A3). Loads: `U8` / `I8` / `U16` / `I16` zero- / sign-extend into an `I32`; `B32` -> `I32`; `B64` -> `Frag2`;
/// `B96` -> `Frag4` (lane 3 unspecified); `B128` -> `Frag4`. Stores write the low 8 / 16 bits (`U8` / `U16`; `I8` / `I16` are the same store).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemTy { U8, I8, U16, I16, B32, B64, B96, B128 }

/// The data of a `d16` load: one byte (zero- or sign-extended to 16 bits) or a 16-bit half, placed in the low (`hi: false`) or high half of
/// the result dword. The other half is ZERO: on gfx950 (SRAM-ECC enabled) the d16 loads do not preserve the unused half (hardware probe).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum D16Ty { U8, I8, B16 }

/// MFMA block shape M x N x K (gfx950).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MmaShape { M16N16K32, M32N32K16, M16N16K64, M32N32K32, M16N16K128, M32N32K64 }

/// MFMA operand element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MmaTy { F16, Bf16, I8, Fp8, Bf8, Fp6, Bf6, Fp4 }

/// One MFMA instruction family member: shape, A / B element types (f8f6f4 families may mix them), block-scaled or not
/// (docs/op-coverage-tileops.md). Supported combinations: `MmaKind::lowerable`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MmaKind {
    pub shape: MmaShape,
    pub a: MmaTy,
    pub b: MmaTy,
    pub scaled: bool,
}

/// E8M0 scales of a block-scaled MFMA: one dword per lane each; `opsel` bits 0..1 = byte of `sa`, bits 4..5 = byte of `sb`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MmaScale {
    pub sa: Opnd,
    pub sb: Opnd,
    pub opsel: u8,
}

impl MmaKind {
    pub const fn new(shape: MmaShape, ty: MmaTy) -> MmaKind {
        MmaKind { shape, a: ty, b: ty, scaled: false }
    }
    /// A block-scaled f8f6f4 kind (`M16N16K128` / `M32N32K64`) with A / B formats `a` / `b` (fp8, bf8, fp6, bf6, fp4; may differ).
    pub const fn scaled_f8f6f4(shape: MmaShape, a: MmaTy, b: MmaTy) -> MmaKind {
        MmaKind { shape, a, b, scaled: true }
    }
    /// An unscaled f8f6f4 kind (`v_mfma_f32_{16x16x128,32x32x64}_f8f6f4`): no scale operands.
    pub const fn f8f6f4(shape: MmaShape, a: MmaTy, b: MmaTy) -> MmaKind {
        MmaKind { shape, a, b, scaled: false }
    }
    /// Accumulator / result dwords per lane (16x16: 4, 32x32: 16).
    pub fn acc_dwords(self) -> u32 {
        match self.shape {
            MmaShape::M16N16K32 | MmaShape::M16N16K64 | MmaShape::M16N16K128 => 4,
            _ => 16,
        }
    }
    /// A (or B) operand dwords per lane.
    pub fn operand_dwords(self, t: MmaTy) -> u32 {
        match t {
            MmaTy::F16 | MmaTy::Bf16 | MmaTy::I8 | MmaTy::Fp4 => 4,
            MmaTy::Fp6 | MmaTy::Bf6 => 6,
            MmaTy::Fp8 | MmaTy::Bf8 => 8,
        }
    }
    pub fn acc_ty(self) -> DType {
        if self.acc_dwords() == 4 { DType::Frag4 } else { DType::Frag16 }
    }
    pub fn operand_ty(self, t: MmaTy) -> DType {
        match self.operand_dwords(t) {
            4 => DType::Frag4,
            _ => DType::Frag8,
        }
    }
    /// The combinations lowering supports today (others are rejected by the IR verifier).
    pub fn lowerable(self) -> bool {
        use MmaShape as S;
        use MmaTy as T;
        let f8f6f4 = |t: MmaTy| matches!(t, T::Fp8 | T::Bf8 | T::Fp6 | T::Bf6 | T::Fp4);
        if matches!(self.shape, S::M16N16K128 | S::M32N32K64) {
            // f8f6f4 with A / B format select (mixed formats allowed), block-scaled or not (unscaled: no scale operands, no unit-scale constants)
            return f8f6f4(self.a) && f8f6f4(self.b);
        }
        if self.scaled {
            return false;
        }
        self.a == self.b && matches!((self.shape, self.a), (S::M16N16K32 | S::M32N32K16, T::Bf16 | T::F16) | (S::M16N16K64 | S::M32N32K32, T::I8))
    }
    /// The f8f6f4 format code (`cbsz` for A, `blgp` for B): fp8 0, bf8 1, fp6 2, bf6 3, fp4 4.
    pub fn f8f6f4_code(t: MmaTy) -> Option<u8> {
        Some(match t {
            MmaTy::Fp8 => 0,
            MmaTy::Bf8 => 1,
            MmaTy::Fp6 => 2,
            MmaTy::Bf6 => 3,
            MmaTy::Fp4 => 4,
            _ => return None,
        })
    }
    /// XDL passes (cost model; issue cycles = 4 x passes).
    pub fn passes(self) -> u8 {
        if self.acc_dwords() == 4 { 8 } else { 16 }
    }
}

/// Cache policy of a global / buffer access (gfx950 `sc0`, `sc1`, `nt` modifiers). Default: none.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct CachePolicy {
    pub sc0: bool,
    pub sc1: bool,
    pub nt: bool,
}

impl CachePolicy {
    pub const NONE: CachePolicy = CachePolicy { sc0: false, sc1: false, nt: false };
    /// Non-temporal (streaming) access.
    pub const NT: CachePolicy = CachePolicy { sc0: false, sc1: false, nt: true };
    /// LLVM CPol bits (1 = sc0, 2 = nt, 16 = sc1).
    pub fn bits(self) -> u8 {
        (self.sc0 as u8) | ((self.nt as u8) << 1) | ((self.sc1 as u8) << 4)
    }
}

/// Conversions (P1 A2; `Op::Cvt { op, srcs, sel }`). Operands per op (unused slots: anything, conventionally `Opnd::I32(0)`).
///
/// SCALE operands (`ScalePk*`) are f32 values of which the hardware uses ONLY THE EXPONENT (E8M0 semantics, measured: a scale of 1.63 acts as
/// 1.0, 3.0 as 2.0): the effective scale is 2^floor(log2 |scale|). Pass powers of two (e.g. an E8M0 byte shifted into an f32 exponent); a
/// constant scale that is not a power of two is rejected by the IR verifier.
/// * `PkF32Fp8` / `PkF32Bf8`: srcs[0] packed bytes, `sel` = word (0 / 1) -> the two values as a Frag2 of f32.
/// * `F32Fp8` / `F32Bf8`: srcs[0], `sel` = byte (0..3) -> F32.
/// * `PkBf8F32`: srcs[0], srcs[1] f32, srcs[2] old dword; the two bf8 bytes replace word `sel` of the old dword -> I32.
/// * `ScalePk{Bf16,F16}{Fp8,Bf8}`: srcs[0] packed bytes, srcs[1] f32 scale (exponent only); word `sel` -> two 16-bit values in a U32 (value · 2^e).
/// * `ScalePkF32{Fp8,Bf8}`: as above -> Frag2 of f32.
/// * `ScalePk{F32,Bf16,F16}Fp4`: srcs[0] packed fp4, srcs[1] scale; byte `sel` (two fp4 values) -> Frag2 f32 / U32.
/// * `ScalePk{Fp8,Bf8}F32`: srcs[0], srcs[1] f32, srcs[2] scale (exponent only), srcs[3] old; word `sel` replaced -> I32 (value / 2^e).
/// * `ScalePkFp4F32`: srcs[0], srcs[1] f32, srcs[2] scale, srcs[3] old; byte `sel` replaced by the two fp4 values -> I32.
/// * `PkU8F32`: srcs[0] f32, srcs[1] old; byte `sel` replaced by the value rounded and clamped to u8 -> I32.
/// * `PkI16I32` / `PkU16U32`: srcs[0], srcs[1] -> two saturated 16-bit values -> U32.
/// * `Scale2xPk16{Fp6,Bf6}F32`: srcs[0], srcs[1] Frag16 of f32, srcs[2] scale -> Frag8 whose dwords 0..5 hold 32 6-bit codes, code j at
///   bits [6 j, 6 j + 6) of the 192 bits; the sources INTERLEAVE (measured, probe_fp6): code 2 k = srcs[0][k], code 2 k + 1 = srcs[1][k].
///   Dwords 6, 7 unspecified. fp6 = e2m3, bf6 = e3m2 (OCP MX); the value stored is x / 2^e (scale exponent e).
/// * `ScalePk32{Fp6,Bf6}{F16,Bf16}`: srcs[0] Frag16 of 32 16-bit values (element i = half i % 2 of dword i / 2), srcs[1] scale -> as above.
/// * `ScalePk32{F16,Bf16}{Fp6,Bf6}`: srcs[0] a Frag8 holding 32 codes in dwords 0..5, srcs[1] scale -> Frag16 of 32 16-bit values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CvtOp {
    PkF32Fp8, PkF32Bf8, F32Fp8, F32Bf8, PkBf8F32,
    ScalePkBf16Fp8, ScalePkF16Fp8, ScalePkBf16Bf8, ScalePkF16Bf8, ScalePkF32Fp8, ScalePkF32Bf8,
    ScalePkF32Fp4, ScalePkBf16Fp4, ScalePkF16Fp4,
    ScalePkFp8F32, ScalePkBf8F32, ScalePkFp4F32,
    PkU8F32, PkI16I32, PkU16U32,
    Scale2xPk16Fp6F32, Scale2xPk16Bf6F32, ScalePk32Fp6F16, ScalePk32Fp6Bf16, ScalePk32Bf6F16, ScalePk32Bf6Bf16, ScalePk32F16Fp6, ScalePk32Bf16Fp6, ScalePk32F16Bf6, ScalePk32Bf16Bf6,
}

impl CvtOp {
    /// (number of operands used, result type, largest `sel`)
    pub fn sig(self) -> (usize, DType, u8) {
        use CvtOp::*;
        match self {
            PkF32Fp8 | PkF32Bf8 => (1, DType::Frag2, 1),
            F32Fp8 | F32Bf8 => (1, DType::F32, 3),
            PkBf8F32 => (3, DType::I32, 1),
            ScalePkBf16Fp8 | ScalePkF16Fp8 | ScalePkBf16Bf8 | ScalePkF16Bf8 => (2, DType::U32, 1),
            ScalePkF32Fp8 | ScalePkF32Bf8 => (2, DType::Frag2, 1),
            ScalePkF32Fp4 => (2, DType::Frag2, 3),
            ScalePkBf16Fp4 | ScalePkF16Fp4 => (2, DType::U32, 3),
            ScalePkFp8F32 | ScalePkBf8F32 => (4, DType::I32, 1),
            ScalePkFp4F32 => (4, DType::I32, 3),
            PkU8F32 => (2, DType::I32, 3),
            PkI16I32 | PkU16U32 => (2, DType::U32, 0),
            Scale2xPk16Fp6F32 | Scale2xPk16Bf6F32 => (3, DType::Frag8, 0),
            ScalePk32Fp6F16 | ScalePk32Fp6Bf16 | ScalePk32Bf6F16 | ScalePk32Bf6Bf16 => (2, DType::Frag8, 0),
            ScalePk32F16Fp6 | ScalePk32Bf16Fp6 | ScalePk32F16Bf6 | ScalePk32Bf16Bf6 => (2, DType::Frag16, 0),
        }
    }
}

/// DPP lane control (gfx9). Rows are 16 lanes. `QuadPerm(p)`: lane i reads lane 4 (i / 4) + p[i % 4]; `RowShl(n)` / `RowShr(n)` / `RowRor(n)`
/// (1..=15): within the row, read lane + n / lane - n / (lane - n) mod 16; `WaveShl1` / `WaveShr1` / `WaveRol1` / `WaveRor1`: whole-wave shift /
/// rotate by one; `RowMirror` (15 - i), `RowHalfMirror` (7 - i within 8); `RowBcast15` (lane 15 of the previous row), `RowBcast31` (lane 31, for
/// lanes 32..63; the rows a broadcast does not target read their own lane — measured). Out-of-range sources are INVALID: with `bound_zero` they\n/// read 0, else the lane keeps `old`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DppCtrl {
    QuadPerm([u8; 4]),
    RowShl(u8),
    RowShr(u8),
    RowRor(u8),
    WaveShl1,
    WaveRol1,
    WaveShr1,
    WaveRor1,
    RowMirror,
    RowHalfMirror,
    RowBcast15,
    RowBcast31,
}

impl DppCtrl {
    /// The 9-bit dpp_ctrl encoding.
    pub fn bits(self) -> u32 {
        match self {
            DppCtrl::QuadPerm(p) => p[0] as u32 & 3 | (p[1] as u32 & 3) << 2 | (p[2] as u32 & 3) << 4 | (p[3] as u32 & 3) << 6,
            DppCtrl::RowShl(n) => 0x100 + n as u32,
            DppCtrl::RowShr(n) => 0x110 + n as u32,
            DppCtrl::RowRor(n) => 0x120 + n as u32,
            DppCtrl::WaveShl1 => 0x130,
            DppCtrl::WaveRol1 => 0x134,
            DppCtrl::WaveShr1 => 0x138,
            DppCtrl::WaveRor1 => 0x13c,
            DppCtrl::RowMirror => 0x140,
            DppCtrl::RowHalfMirror => 0x141,
            DppCtrl::RowBcast15 => 0x142,
            DppCtrl::RowBcast31 => 0x143,
        }
    }
}

/// The DPP part of a `Op::Dpp` / `Op::DppAlu`: control, row / bank write masks (bit r: row r; bit b: lanes 4 b .. 4 b + 3 of every row),
/// and bound_ctrl (invalid sources read 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dpp {
    pub ctrl: DppCtrl,
    pub row_mask: u8,
    pub bank_mask: u8,
    pub bound_zero: bool,
}

impl Dpp {
    /// All rows / banks enabled, invalid sources keep `old`.
    pub const fn new(ctrl: DppCtrl) -> Dpp {
        Dpp { ctrl, row_mask: 0xf, bank_mask: 0xf, bound_zero: false }
    }
    pub fn imm(self) -> i64 {
        self.ctrl.bits() as i64 | (((self.row_mask & 0xf) ^ 0xf) as i64) << 12 | (((self.bank_mask & 0xf) ^ 0xf) as i64) << 16 | (self.bound_zero as i64) << 20
    }
}

/// VALU ops with a DPP-modified first operand: `D = dpp(a) op b` (lanes not written keep `old`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DppAluOp { AddF32, SubF32, MulF32, MaxF32, MinF32, AddU32, MaxI32, MinI32, MaxU32, MinU32, AndB32, OrB32, XorB32 }

/// Atomic read-modify-write operations (B3). `IncU32`: old >= value ? 0 : old + 1 (a wrapping counter).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicOp { AddU32, MaxU32, MinU32, MaxI32, MinI32, IncU32, AddF32, PkAddBf16, PkAddF16 }

impl AtomicOp {
    pub fn value_ty(self) -> DType {
        if self == AtomicOp::AddF32 { DType::F32 } else { DType::U32 }
    }
}

impl MemTy {
    pub fn bytes(self) -> u32 {
        match self {
            MemTy::U8 | MemTy::I8 => 1,
            MemTy::U16 | MemTy::I16 => 2,
            MemTy::B32 => 4,
            MemTy::B64 => 8,
            MemTy::B96 => 12,
            MemTy::B128 => 16,
        }
    }
    pub fn value_ty(self) -> DType {
        match self {
            MemTy::B64 => DType::Frag2,
            MemTy::B96 | MemTy::B128 => DType::Frag4,
            _ => DType::I32,
        }
    }
}
impl D16Ty {
    pub fn bytes(self) -> u32 {
        if self == D16Ty::B16 { 2 } else { 1 }
    }
}

impl UnOp {
    pub fn float_in(self) -> bool {
        !matches!(self, UnOp::CvtF32I32 | UnOp::CvtF32U32 | UnOp::CvtF32Ubyte0 | UnOp::CvtF32Ubyte1 | UnOp::CvtF32Ubyte2 | UnOp::CvtF32Ubyte3 | UnOp::Ffbl | UnOp::Ffbh)
    }
    pub fn result_ty(self) -> DType {
        match self {
            UnOp::CvtI32F32 => DType::I32,
            UnOp::CvtU32F32 => DType::U32,
            UnOp::Ffbl | UnOp::Ffbh => DType::I32,
            _ => DType::F32,
        }
    }
    /// transcendental unit (longer latency, trans hazards)
    pub fn trans(self) -> bool {
        matches!(self, UnOp::RsqF32 | UnOp::SqrtF32 | UnOp::SinF32 | UnOp::CosF32)
    }
}
impl BinOp {
    pub fn packed(self) -> bool {
        matches!(self, BinOp::PkAddF16 | BinOp::PkMulF16 | BinOp::PkMaxF16 | BinOp::PkMinF16)
    }
    pub fn result_ty(self) -> DType {
        match self {
            BinOp::MinF32 => DType::F32,
            b if b.packed() => DType::U32,
            _ => DType::I32,
        }
    }
}
impl TriOp {
    pub fn result_ty(self) -> DType {
        match self {
            TriOp::Min3F32 | TriOp::Med3F32 => DType::F32,
            TriOp::PkFmaF16 => DType::U32,
            _ => DType::I32,
        }
    }
}

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
    /// Dwords 4 idx .. 4 idx + 3 of a Frag8 (idx 0..1) or Frag16 (idx 0..3) variable as a Frag4 value WITHOUT a copy (a register view, as `VarSlice8`).
    VarSlice4 { var: VarId, idx: u8 },
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
    /// unary VALU op (`UnOp`)
    Un(UnOp, Opnd),
    /// binary VALU op (`BinOp`)
    Bin(BinOp, Opnd, Opnd),
    /// ternary VALU op (`TriOp`)
    Tri(TriOp, Opnd, Opnd, Opnd),
    /// comparison -> I1
    Cmp { pred: CmpPred, ty: CmpTy, a: Opnd, b: Opnd },
    /// global load of any width (`MemTy`); result type `ty.value_ty()`
    LoadGlobalN { base: Opnd, byte_off: u32, ty: MemTy, cache: CachePolicy },
    /// `global_load_*_d16[_hi]`: the loaded 8 / 16 bits in one half of a dword, the other half zero
    LoadGlobalD16 { base: Opnd, byte_off: u32, ty: D16Ty, hi: bool, cache: CachePolicy },
    /// MUBUF load of any width
    BufferLoadN { rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm: u32, ty: MemTy, cache: CachePolicy },
    /// LDS load of any width
    LoadLdsN { addr: Opnd, ty: MemTy },
    /// `ds_read_*_d16[_hi]`
    LoadLdsD16 { addr: Opnd, ty: D16Ty, hi: bool },
    /// `ds_read2[st64]_b32 / _b64`: two elements at `addr + off0 * s` and `addr + off1 * s` (s = element size, x 64 for `st64`) -> `Frag2` / `Frag4`
    LoadLds2 { addr: Opnd, off0: u8, off1: u8, b64: bool, st64: bool },
    /// `ds_read_b64_tr_b16`: 16-bit transpose read across the 16-lane groups (as `tr_b8` for bytes) -> `Frag2`
    LoadLdsTrB16 { addr: Opnd },
    /// 64-bit address arithmetic: `base + (u64) a * (u64) b` with `a`, `b` read as UNSIGNED 32-bit (> 4 GiB offsets; a wave-uniform result
    /// stays scalar, so it can be the base of a buffer descriptor)
    PtrMadU32 { base: Opnd, a: Opnd, b: Opnd },
    /// `v_pk_fma_f32`: a * b + c on two f32 lanes (Frag2 each)
    PkFmaF32(Opnd, Opnd, Opnd),
    /// "Any lane": the wave-level reading of a per-lane condition (true for the whole wave when any lane's `cond` holds) — the explicit marker
    /// for branching on a per-lane condition (`if` / `break_if` are wave-level). No code: the branch itself has this semantics.
    AnyLane(Opnd),
    /// Dwords `start .. start + n` of a fragment value as a value of its own WITHOUT a copy (a register view; n = 1 -> I32, 2 / 4 / 8 -> FragN)
    FragSlice { frag: Opnd, start: u8, n: u8 },
    /// The concatenation of one-dword values and fragments into one fragment of the total width (2, 4, 8 or 16 dwords; copies)
    FragCat { parts: Vec<Opnd>, dwords: u8 },
    /// LDS atomic returning the old value (`ds_<op>_rtn_*`)
    AtomicLds { op: AtomicOp, addr: Opnd, value: Opnd },
    /// global atomic returning the old value (`global_atomic_<op> ... sc0`)
    AtomicGlobal { op: AtomicOp, base: Opnd, byte_off: u32, value: Opnd },
    /// returning buffer atomic (`buffer_atomic_add ... sc0`, AddU32): the old value. Lanes whose offset is out of range (`>= num_records`) are
    /// dropped by the range check before reaching L2 and return 0 - a per-lane mask: e.g. only lane 0 takes a ticket (no same-address storm)
    BufferAtomicRet { op: AtomicOp, rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm: u32, value: Opnd },
    /// `v_readlane_b32`: lane `lane`'s value of `x` (wave-uniform result)
    ReadLane { x: Opnd, lane: u8 },
    /// `v_writelane_b32`: `old` with lane `lane` replaced by the (uniform) `x` (a non-uniform `x` is read from the first active lane)
    WriteLane { x: Opnd, lane: u8, old: Opnd },
    /// the 32-bit half (`hi`) of the wave mask of lanes where `cond` (I1) holds (`v_cmp_ne_u32_e64` into an SGPR pair); wave-uniform
    Ballot { cond: Opnd, hi: bool },
    /// `ds_bpermute_b32`: lane k gets `x` of lane (addr[k] + offset) / 4 % 64 (a pull; no LDS memory is touched)
    Bpermute { addr: Opnd, x: Opnd, offset: u16 },
    /// `ds_permute_b32`: lane k sends its `x` to lane (addr[k] + offset) / 4 % 64 (a push; lanes nobody writes get 0)
    Permute { addr: Opnd, x: Opnd, offset: u16 },
    /// `ds_swizzle_b32 offset:pattern` (bit 15: quad-perm mode, else bit-mask mode and_mask / or_mask / xor_mask within 32 lanes)
    Swizzle { x: Opnd, pattern: u16 },
    /// `v_mov_b32_dpp`: lane i gets `x` of the lane `dpp.ctrl` names (lanes not written keep `old`)
    Dpp { dpp: Dpp, x: Opnd, old: Opnd },
    /// `v_<op>_dpp`: dpp(a) op b, lanes not written keep `old`
    DppAlu { op: DppAluOp, dpp: Dpp, a: Opnd, b: Opnd, old: Opnd },
    /// `v_permlane16_swap`: exchanges the odd 16-lane rows of `a` with the even rows of `b`; Frag2 (new a, new b)
    Permlane16Swap { a: Opnd, b: Opnd },
    /// conversion (`CvtOp`; operands and `sel` per op)
    Cvt { op: CvtOp, srcs: [Opnd; 4], sel: u8 },
    /// D = A x B + C (`c: None`: zero accumulator) of the MFMA `kind`; result `kind.acc_ty()`
    Mfma { kind: MmaKind, a: Opnd, b: Opnd, c: Option<Opnd>, scale: Option<MmaScale> },
}

impl Op {
    pub fn result_ty(&self) -> DType {
        match self {
            Op::Un(o, _) => o.result_ty(),
            Op::Bin(o, ..) => o.result_ty(),
            Op::Tri(o, ..) => o.result_ty(),
            Op::Cmp { .. } => DType::I1,
            Op::LoadGlobalN { ty, .. } | Op::BufferLoadN { ty, .. } | Op::LoadLdsN { ty, .. } => ty.value_ty(),
            Op::LoadGlobalD16 { .. } | Op::LoadLdsD16 { .. } => DType::I32,
            Op::LoadLds2 { b64, .. } => if *b64 { DType::Frag4 } else { DType::Frag2 },
            Op::LoadLdsTrB16 { .. } => DType::Frag2,
            Op::Mfma { kind, .. } => kind.acc_ty(),
            Op::Cvt { op, .. } => op.sig().1,
            Op::Dpp { .. } => DType::U32,
            Op::AnyLane(_) => DType::I1,
            Op::PkFmaF32(..) => DType::Frag2,
            Op::PtrMadU32 { .. } => DType::Ptr,
            Op::FragSlice { n, .. } => match n {
                1 => DType::I32,
                2 => DType::Frag2,
                4 => DType::Frag4,
                _ => DType::Frag8,
            },
            Op::FragCat { dwords, .. } => match dwords {
                2 => DType::Frag2,
                4 => DType::Frag4,
                8 => DType::Frag8,
                _ => DType::Frag16,
            },
            Op::ReadLane { .. } | Op::WriteLane { .. } | Op::Ballot { .. } | Op::Bpermute { .. } | Op::Permute { .. } | Op::Swizzle { .. } => DType::I32,
            Op::AtomicLds { op, .. } | Op::AtomicGlobal { op, .. } => op.value_ty(),
            Op::BufferAtomicRet { .. } => DType::I32,
            Op::DppAlu { op, .. } => match op {
                DppAluOp::AddF32 | DppAluOp::SubF32 | DppAluOp::MulF32 | DppAluOp::MaxF32 | DppAluOp::MinF32 => DType::F32,
                _ => DType::I32,
            },
            Op::Permlane16Swap { .. } => DType::Frag2,
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
            Op::VarSlice4 { .. } => DType::Frag4,
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
            Op::Un(_, a) => vec![*a],
            Op::Bin(_, a, b) | Op::Cmp { a, b, .. } => vec![*a, *b],
            Op::Tri(_, a, b, c) => vec![*a, *b, *c],
            Op::LoadGlobalN { base, .. } => vec![*base],
            Op::LoadGlobalD16 { base, .. } => vec![*base],
            Op::BufferLoadN { rsrc, voff, soff, .. } => {
                let mut v = vec![*rsrc];
                v.extend(voff.iter().copied());
                v.push(*soff);
                v
            }
            Op::LoadLdsN { addr, .. } | Op::LoadLds2 { addr, .. } | Op::LoadLdsTrB16 { addr } | Op::LoadLdsD16 { addr, .. } => vec![*addr],
            Op::Cvt { op, srcs, .. } => srcs[..op.sig().0].to_vec(),
            Op::Dpp { x, old, .. } => vec![*x, *old],
            Op::AnyLane(c) => vec![*c],
            Op::PkFmaF32(a, b, c) => vec![*a, *b, *c],
            Op::PtrMadU32 { base, a, b } => vec![*base, *a, *b],
            Op::FragSlice { frag, .. } => vec![*frag],
            Op::FragCat { parts, .. } => parts.clone(),
            Op::ReadLane { x, .. } | Op::Swizzle { x, .. } => vec![*x],
            Op::WriteLane { x, old, .. } => vec![*x, *old],
            Op::Ballot { cond, .. } => vec![*cond],
            Op::Bpermute { addr, x, .. } | Op::Permute { addr, x, .. } => vec![*addr, *x],
            Op::AtomicLds { addr, value, .. } => vec![*addr, *value],
            Op::AtomicGlobal { base, value, .. } => vec![*base, *value],
            Op::BufferAtomicRet { rsrc, voff, soff, value, .. } => { let mut v = vec![*rsrc, *soff, *value]; if let Some(x) = voff { v.push(*x); } v }
            Op::DppAlu { a, b, old, .. } => vec![*a, *b, *old],
            Op::Permlane16Swap { a, b } => vec![*a, *b],
            Op::Mfma { a, b, c, scale, .. } => {
                let mut v = vec![*a, *b];
                v.extend(c.iter().copied());
                if let Some(s) = scale {
                    v.push(s.sa);
                    v.push(s.sb);
                }
                v
            }
            Op::Max3F32(a, b, c) => vec![*a, *b, *c],
            Op::ExpF32(a) | Op::Exp2F32(a) | Op::Log2F32(a) | Op::RcpF32(a) | Op::CvtF16ToF32(a) | Op::CvtF32ToF16(a) | Op::NotI1(a) => {
                vec![*a]
            }
            Op::Select { cond, a, b, .. } => vec![*cond, *a, *b],
            Op::LoadU16 { base, .. } | Op::LoadF32 { base, .. } => vec![*base],
            Op::Frag2(a, b) => vec![*a, *b],
            Op::Frag2Lane { frag, .. } => vec![*frag],
            Op::VarSlice8 { var, .. } | Op::VarSlice4 { var, .. } => vec![Opnd::Var(*var)],
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
    /// Thread-OWNED words (`lds_owned`): (col_stride, row_stride) - thread t owns the bytes `base + t col_stride + r row_stride + [0, col_stride)`.
    /// A marker whose every access is proven to lie in the issuing thread's own words is an OWNER event: owner events of different waves never
    /// touch the same bytes, and a thread's own write -> read is ordered by program order (no barrier needed between them).
    pub owned: Option<(i64, i64)>,
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
    /// (new, old): `new` REUSES the storage of `old` (time multiplexing: `LdsPlan::tile_over` / `private(.., overlays)`). Unlike a plain alias, the
    /// protocol engine checks it: every write of one must be separated by a barrier from the other's accesses by any other wave, or provably stay
    /// inside the writer's own slice of a private `old` (GV-TILE-ALIAS-RACE).
    pub tile_reuses: Vec<(String, String)>,
    /// Explicit trust that the tile protocol is periodic in the loop iteration (see `tile_trust_periodic`): skips the affine-counter check.
    pub tile_trust: Option<String>,
    /// Global-memory regions, the last-arriver patterns and the per-region trust entries (name, reason) of the v4 protocol model.
    pub buf_regions: Vec<BufRegion>,
    pub last_arrivers: Vec<LastArriver>,
    pub buf_trusts: Vec<(String, String)>,
    /// (symbol name, lo, hi) bounds of free symbols.
    pub sym_ranges: Vec<(String, i64, i64)>,
    /// Trusted facts: a tuple of values is an injective function of the workgroup id over the listed grid axes (e.g. an XCD raster, a
    /// permutation of the tile grid). Used by the cross-workgroup store proof; counted in the trust ledger (`wg_bijection`).
    pub wg_maps: Vec<WgMap>,
    /// Trusted facts: the `then` branch of an `if` on `cond` runs in exactly one workgroup of every group of workgroups that differ only along
    /// `axes` (a last-arriver ticket). Used by the cross-workgroup store proof; counted in the trust ledger (`wg_winner`).
    pub wg_winners: Vec<WgWinner>,
}

/// See `Contract::wg_winners`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct WgWinner {
    pub cond: ValueId,
    pub axes: Vec<u8>,
    pub reason: String,
}

/// See `Contract::wg_maps`.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct WgMap {
    pub outs: Vec<ValueId>,
    pub axes: Vec<u8>,
    pub reason: String,
    /// `wg_union`: the outputs are ONE index computed differently per store class (e.g. the paired causal q tiles x and n - 1 - x), injective
    /// over (workgroup, class); otherwise the tuple of outputs is injective over the workgroup
    pub union: bool,
    /// `wg_union`: the index range [lo, hi] (part of the fact)
    pub range: Option<(Ext, Ext)>,
}

/// One value set crossing a stage boundary (`StageContract`): the IR values (fragment registers), the element table they hold (`[t + threads v]`
/// -> tensor element, after any K permutation: a layout fingerprint computed by the tile layer) and storage facts.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StageVal {
    pub name: String,
    pub values: Vec<ValueId>,
    pub table: Vec<i64>,
    pub elem_bits: u32,
    /// `vgpr` | `agpr` | `lds:<tile>` | `global` (VGPR <-> AGPR hand-overs are register moves inside the consumer and allowed)
    pub storage: String,
    /// the thread-value layout of the fragment over its tensor tile (`amdgpu_layout::Layout` text, e.g. `((32,2),(4,4,2)):((64,4),(1,8,32))`);
    /// empty = not recorded. Must match between producer and consumer when both record it.
    pub layout: String,
}

/// A scalar / per-row state crossing a stage boundary (row statistics, scales, counters): its values and which lanes own it
/// (`lane` = one per lane, `row` = per accumulator row (the lanes holding that row), `wave` = wave-uniform, `block` = workgroup-uniform).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StageScalar {
    pub name: String,
    pub values: Vec<ValueId>,
    pub owner: String,
}

/// Memory operations allowed in flight across the stage edges (program order, linear view): at the entry the stage tolerates up to `*_in`
/// outstanding operations of the earlier code, at the exit it leaves at most `*_out` of its own. `None` = not constrained.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StageOutstanding {
    pub vm_in: Option<u32>,
    pub lgkm_in: Option<u32>,
    pub vm_out: Option<u32>,
    pub lgkm_out: Option<u32>,
}

/// The stage's resource needs (estimates for planning; `barriers` is checked exactly, the register numbers against `amdc regplan`).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StageResources {
    pub vgprs: Option<u32>,
    pub agprs: Option<u32>,
    pub lds_bytes: Option<u32>,
    /// barriers inside the stage (linear view, every instance)
    pub barriers: Option<u32>,
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
    /// scalar state consumed / handed over (checked like `live_in` / `live_out`, plus the owner)
    pub scalars_in: Vec<StageScalar>,
    pub scalars_out: Vec<StageScalar>,
    pub outstanding: StageOutstanding,
    pub resources: StageResources,
}

impl StageContract {
    /// `barrier_phase` value for a stage whose phase is not constrained: stages inside versioned / role branches and pipelined loops, where the
    /// linear barrier count depends on which alternatives a transformation (e.g. the JIT specializer) keeps; their cross-wave ordering is proven
    /// by the LDS tile protocol instead.
    pub const ANY_PHASE: u32 = u32::MAX;
}

/// Statements: a flat sequence with explicit structured markers.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// Opens a pipeline stage (`Node::Stage`) with its resource contract. Emits no code.
    StageBegin { name: String, contract: Box<StageContract> },
    /// Closes the innermost stage.
    StageEnd,
    /// global store of any width (`U8` / `U16` store the low bits; `hi` with `U8` / `U16`: the `_d16_hi` forms store bits 16..)
    StoreGlobalN { base: Opnd, byte_off: u32, value: Opnd, ty: MemTy, hi: bool, cache: CachePolicy },
    /// MUBUF store of any width (same `hi` rule)
    BufferStoreN { rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm: u32, value: Opnd, ty: MemTy, hi: bool, cache: CachePolicy },
    /// LDS store of any width (same `hi` rule)
    StoreLdsN { addr: Opnd, value: Opnd, ty: MemTy, hi: bool },
    /// In place: `var[start + 2 i .. + 2] = t[2 i .. + 2] * s + var[...]` for every dword pair of `t` (`v_pk_fma_f32`; `s` a Frag2, e.g. a
    /// `frag2_dup` scale)
    VarPkFma { var: VarId, start: u8, t: Opnd, s: Opnd },
    /// LDS atomic without return
    AtomicLdsNoRet { op: AtomicOp, addr: Opnd, value: Opnd },
    /// MUBUF atomic without return (`buffer_atomic_<op>`; AddU32, AddF32, PkAddBf16, PkAddF16): range-checked against the descriptor like a store
    BufferAtomicNoRet { op: AtomicOp, rsrc: Opnd, voff: Option<Opnd>, soff: Opnd, imm: u32, value: Opnd },
    /// global atomic without return
    AtomicGlobalNoRet { op: AtomicOp, base: Opnd, byte_off: u32, value: Opnd },
    /// `ds_write2[st64]_b32 / _b64`: `v0` at `addr + off0 * s`, `v1` at `addr + off1 * s`
    StoreLds2 { addr: Opnd, v0: Opnd, v1: Opnd, off0: u8, off1: u8, b64: bool, st64: bool },
    /// In-place MFMA accumulate into a loop variable: var = A x B + var (`zero`: var = A x B)
    VarMfma { var: VarId, kind: MmaKind, a: Opnd, b: Opnd, scale: Option<MmaScale>, zero: bool },
    /// Register-planning region marker (`FnBuilder::weave_region_begin/end`): the instructions lowered between a begin and its end belong to the
    /// named region (per-region pressure caps and reports, `RegMark::Cap`). Regions nest; a begin and its end sit in the same structural block.
    /// Emits no code.
    RegionMark { name: String, begin: bool },
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

/// A value or a loop variable, as the target of a register-planning annotation (`RegMark`). Tuples are planned as a whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RegRef {
    Val(ValueId),
    Var(VarId),
}

/// Register-planning annotations (the weaver's step 3, docs/weaver-regs.md): emitted by the kernel / weaver, honoured by the register allocator
/// and checked after allocation. An empty plan changes nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegMark {
    /// HARD: the tuple of `r` sits at arch VGPR `base` (aligned to its width).
    PinV { r: RegRef, base: u32 },
    /// HINT: place the later-defined members in the registers of an earlier one when their lifetimes are disjoint (reported when not honoured).
    Share(Vec<RegRef>),
    /// HARD: the members never use the same physical registers, even with disjoint lifetimes.
    Separate(Vec<RegRef>),
    /// HARD: at every instruction of region `region` (a weave region or a stage, nested regions included) the allocated arch-VGPR footprint
    /// (highest live physical VGPR + 1) is at most `vgprs`, and the AGPR footprint at most `agprs` when given.
    Cap { region: String, vgprs: Option<u32>, agprs: Option<u32> },
    /// The `bufs.len()` register buffers of one fragment (double / triple buffering): pairwise separate, placement reported.
    Ring { name: String, bufs: Vec<RegRef> },
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

/// Does the node contain a tile region?
fn contains_region(n: &Node) -> bool {
    match n {
        Node::S(_) => false,
        Node::Region { .. } => true,
        Node::Loop { body, .. } | Node::Stage { body, .. } => body.iter().any(contains_region),
        Node::If { then, els, .. } => then.iter().any(contains_region) || els.iter().flatten().any(contains_region),
    }
}

/// Region-tree construction shared by the builder and `Function::parse_flat`.
#[derive(Clone, Debug, Default)]
struct TreeBuilder {
    root: Vec<Node>,
    open: Vec<Frame>,
    /// statements of the flat view emitted so far (markers included)
    count: usize,
    /// `Stmt::Barrier`s emitted so far (program order: a stage's `barrier_phase`)
    barriers: u32,
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
        if matches!(st, Stmt::Barrier) {
            self.barriers += 1;
        }
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
    /// AGPR placement requests (empty for every existing kernel).
    pub agpr: Vec<AgprMark>,
    /// Where each value was created: the Rust source location of the builder call (`#[track_caller]`), indexed by `ValueId`. A side table
    /// for diagnostics only - not part of the IR (no equality, codec or printing; `Debug` prints nothing).
    pub src_locs: SrcLocs,
    /// Explicit compile policy (lowering options + scheduler preset). `None` = the compiler derives it from the kernel name (the historical behaviour,
    /// which every existing kernel keeps); kernels written on the tile library set it so their policy does not depend on their name.
    pub compile_profile: Option<CompileProfile>,
    /// Extra lowering option tokens for this kernel (the AMDC_LOWER_OPTS vocabulary, e.g. lgkmn), on top of its profile - so nothing is name-keyed.
    pub lower_opts: Vec<String>,
    /// Scheduler option tokens (the AMDC_SCHED_OPTS vocabulary: rewait,ldshoist,xbar,hoist=N,limit=N,...); Some turns the scheduler on for this kernel (unless AMDC_SCHED=0).
    pub sched_opts: Option<String>,
    /// Register-planning annotations ([`RegMark`]); empty for every kernel that does not plan its registers.
    pub reg_plan: Vec<RegMark>,
    /// Scheduler strategies of weave regions, keyed by region NAME (the labels of `weave_region_begin` / the Weaver / stage names): `declare_region(name, authored, tokens)`. A region without a
    /// declaration inherits its enclosing region's strategy (the kernel default at the top). Empty for every kernel that does not ask for per-region strategies (nothing changes then).
    pub regions: Vec<RegionDecl>,
}

/// Builder call sites of the values of a function (see `Function::src_locs`).
#[derive(Clone, Default)]
pub struct SrcLocs(pub Vec<Option<&'static std::panic::Location<'static>>>);

impl std::fmt::Debug for SrcLocs {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl Function {
    /// The declared compile policy (`CompileProfile::policy`; the default policy without a profile).
    pub fn compile_policy(&self) -> CompilePolicy {
        self.compile_profile.map_or(DEFAULT_POLICY, |p| p.policy())
    }

    /// `file:line:col` of the builder call that created value `v`.
    pub fn value_loc(&self, v: ValueId) -> Option<String> {
        self.src_locs.0.get(v as usize).copied().flatten().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
    }
    /// The source location of statement `idx` of `stmts()`: its own value's creation site for a `Let` (exact), else the nearest preceding
    /// `Let`'s (approximate). Returns (location, exact).
    pub fn stmt_loc(&self, idx: usize) -> Option<(String, bool)> {
        let st = self.stmts();
        if let Some(Stmt::Let { dst, .. }) = st.get(idx) {
            if let Some(l) = self.value_loc(*dst) {
                return Some((l, true));
            }
        }
        (0..idx.min(st.len())).rev().find_map(|k| match &st[k] {
            Stmt::Let { dst, .. } => self.value_loc(*dst).map(|l| (l, false)),
            _ => None,
        })
    }
    /// The innermost weave region (`RegionMark`) enclosing statement `idx`.
    pub fn region_of(&self, idx: usize) -> Option<String> {
        let mut stack: Vec<String> = Vec::new();
        for (k, s) in self.stmts().iter().enumerate() {
            if k >= idx {
                break;
            }
            if let Stmt::RegionMark { name, begin } = s {
                if *begin {
                    stack.push(name.clone());
                } else {
                    stack.pop();
                }
            }
        }
        stack.pop()
    }
}

/// A weave region's scheduler strategy (declared on the function by region name; see `Function::regions`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionDecl {
    pub name: String,
    /// The order the weaver emitted is authoritative: `rewait` + the order-preserving post-passes only (no list scheduling / slotfill / ldshoist inside the region).
    pub authored: bool,
    /// Scheduler option tokens applied to the region's instructions on top of the kernel's options.
    pub tokens: String,
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

/// How a policy's lowering tokens combine with the lowering options already in effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LowerBase {
    /// on top of the options in effect (the default `dce,cmpfold`, or a caller's `lower::with_opts` override)
    Inherit,
    /// exactly these tokens (plus the retry ladder's and the kernel's own `lower_opts`)
    Replace,
}

/// The scheduler preset a policy selects (the kernel's `sched_opts` tokens are applied on top).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedPreset {
    /// no preset: the scheduler runs only for `sched_opts`, declared regions or the environment
    None,
    /// scoreboard (partial) waits + ds_read hoisting, no instruction reordering (docs/scheduling.md; measured best on the GEMM)
    Gemm,
    /// ALU may cross the single barrier per tile, VGPR pressure limit 248 (measured 489 -> 505 TFLOP/s on B1H8S2048; limit 246 / 248 only)
    T128,
}

/// The declared compile policy of a kernel: the data the compiler's pipeline reads (it no longer looks at kernel names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompilePolicy {
    pub lower_base: LowerBase,
    pub lower: &'static [&'static str],
    /// M0 reuse for LDS-DMA loads (sched::m0::reuse)
    pub m0_reuse: bool,
    pub sched_preset: SchedPreset,
}

/// The vendor-structure FMHA lowering: `scalarize` (wave-uniform integer chains in SGPRs) for the register savings the svscale / 3-slot swizzled ring
/// need; measured on vs128 (15/15, 301.9-304.0 vs 280.6 TFLOP/s): retarget2 (cvt results written straight into the MFMA fragment lanes), imm and sgpr
/// cut ~230 copies/VALU and free 12 VGPRs; addrfold and shl are slower and stay off (docs/archive/status.md rounds 47-50).
const VS_LOWER: &[&str] = &["dce", "cmpfold", "scalarize", "retarget2", "imm", "sgpr", "zeroinit"];

impl CompileProfile {
    /// The policy table.
    pub const fn policy(self) -> CompilePolicy {
        match self {
            CompileProfile::Vs128 | CompileProfile::KvSplit => CompilePolicy { lower_base: LowerBase::Replace, lower: VS_LOWER, m0_reuse: false, sched_preset: SchedPreset::None },
            // `_t128` (4-wave blocks, one wave per SIMD): addrfold frees the DMA address temporaries so the full scheduler fits (416 -> 494 TFLOP/s)
            CompileProfile::T128 => CompilePolicy {
                lower_base: LowerBase::Replace,
                lower: &["dce", "cmpfold", "scalarize", "retarget2", "imm", "sgpr", "zeroinit", "addrfold"],
                m0_reuse: false,
                sched_preset: SchedPreset::T128,
            },
            // the FMHA kernels zero their accumulators in place (bit-exact on all 15 shapes, -160 v_mov on _t, docs/archive/status.md)
            CompileProfile::Combine => CompilePolicy { lower_base: LowerBase::Inherit, lower: &["zeroinit"], m0_reuse: false, sched_preset: SchedPreset::None },
            // partial lgkmcnt waits keep the prefetched fragment reads in flight (`lgkmn`); scalarize / scalarize2 / waveuni: the wave-uniform DMA address
            // chains (tile origins with clamps/selects, slot bases, M0, soffsets) in SALU; kabatch (kernarg loads batched into s_load_dwordx4) and imm
            // (inline constants / literals as VALU operands instead of v_mov materialisations); M0 reuse for the LDS-DMA loads
            CompileProfile::Gemm => CompilePolicy {
                lower_base: LowerBase::Replace,
                lower: &["dce", "cmpfold", "lgkmn", "zeroinit", "scalarize", "scalarize2", "waveuni", "edgepad", "kabatch", "imm"],
                m0_reuse: true,
                sched_preset: SchedPreset::Gemm,
            },
            CompileProfile::Default => DEFAULT_POLICY,
        }
    }
}

/// The policy of a kernel without a declared profile.
pub const DEFAULT_POLICY: CompilePolicy = CompilePolicy { lower_base: LowerBase::Inherit, lower: &[], m0_reuse: false, sched_preset: SchedPreset::None };

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
    /// A function from all of its parts (the IR text parser, `ir_text::parse`); the linear view is derived from `body`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        name: String,
        args: Vec<Arg>,
        kernarg_size: u32,
        workgroup_size: u32,
        lds_bytes: u32,
        contract: Contract,
        body: Vec<Node>,
        var_tys: Vec<DType>,
        value_tys: Vec<DType>,
        agpr: Vec<AgprMark>,
        compile_profile: Option<CompileProfile>,
        lower_opts: Vec<String>,
        sched_opts: Option<String>,
        reg_plan: Vec<RegMark>,
        regions: Vec<RegionDecl>,
    ) -> Function {
        let mut f = Function { name, args, kernarg_size, workgroup_size, lds_bytes, contract, body: Vec::new(), flat: Vec::new(), var_tys, value_tys, agpr, src_locs: SrcLocs::default(), compile_profile, lower_opts, sched_opts, reg_plan, regions };
        f.set_body(body);
        f
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

}

/// Builder that allocates ids and keeps the statement stream balanced.
pub struct FnBuilder {
    pub f: Function,
    /// the region tree under construction (open loops / ifs / tile regions)
    tree: TreeBuilder,
    /// constants materialised by the expression layer (`expr.rs`), keyed by (is_float, bits): reused by later `bind`s in the same or an inner scope, cleared when a scope closes
    pub(crate) expr_consts: std::collections::HashMap<(bool, u32), ValueId>,
    /// capture mode (`capture_begin`): the region trees the open captures replaced (innermost last)
    captures: Vec<TreeBuilder>,
    /// register views of loop variables: a value defined by a lane / slice / bitcast of a variable (transitively) lowers to an alias of the
    /// variable's registers (no instruction), so every user of the value reads the variable at the user's position (`facts`)
    views: std::collections::HashMap<ValueId, Vec<VarId>>,
}

/// Handle of an open capture (`FnBuilder::capture_begin`); closed by `capture_end` in LIFO order.
#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub struct CaptureId(usize);

impl FnBuilder {
    #[track_caller]
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
            agpr: Vec::new(),
                src_locs: SrcLocs::default(),
                compile_profile: None,
            lower_opts: Vec::new(),
            sched_opts: None,
            regions: Vec::new(),
            reg_plan: Vec::new(),
            },
            expr_consts: Default::default(),
            captures: Vec::new(),
            views: Default::default(),
            tree: TreeBuilder::default(),
        }
    }

    /// Sets the explicit compile policy (see [`CompileProfile`]); without it the compiler keeps deriving the policy from the kernel name.
    #[track_caller]
    pub fn compile_profile(&mut self, p: CompileProfile) {
        self.f.compile_profile = Some(p);
    }

    /// Extra lowering option tokens for this kernel (see Function::lower_opts).
    #[track_caller]
    pub fn lower_opts(&mut self, opts: &[&str]) {
        self.f.lower_opts.extend(opts.iter().map(|s| s.to_string()));
    }

    /// Declares the scheduler strategy of the weave region named `name` (the label of `weave_region_begin` / `Weaver::phase`; a stage name works too): `authored` = the emitted order is
    /// authoritative (no list scheduling / slotfill / ldshoist inside; waits are recomputed and the order-preserving post-passes still apply), `tokens` = scheduler option tokens (the
    /// `AMDC_SCHED_OPTS` vocabulary) applied to the region's instructions only. Undeclared regions inherit the strategy of the region they are nested in.
    #[track_caller]
    pub fn declare_region(&mut self, name: &str, authored: bool, tokens: &str) {
        self.f.regions.retain(|r| r.name != name);
        self.f.regions.push(RegionDecl { name: name.to_string(), authored, tokens: tokens.to_string() });
    }

    /// Scheduler option tokens for this kernel (see Function::sched_opts).
    #[track_caller]
    pub fn sched_opts(&mut self, opts: &str) {
        self.f.sched_opts = Some(opts.to_string());
    }

    /// Declares how many bytes of workgroup LDS the kernel uses.
    #[track_caller]
    pub fn lds(&mut self, bytes: u32) {
        self.f.lds_bytes = bytes;
    }

    /// GPU-semantics contract: the pointer kernarg `name` points to an allocation of `alloc_bytes` bytes (an expression over kernarg scalars and the grid).
    #[track_caller]
    pub fn arg_tensor(&mut self, name: &str, alloc_bytes: Ext) {
        self.f.contract.extents.push(ArgExtent { arg: name.to_string(), alloc: alloc_bytes, logical: None });
    }
    /// Like `arg_tensor` with a zeroed slack at the end: reads may use `alloc_bytes`, writes must stay within `logical_bytes`.
    #[track_caller]
    pub fn arg_tensor_slack(&mut self, name: &str, logical_bytes: Ext, alloc_bytes: Ext) {
        self.f.contract.extents.push(ArgExtent { arg: name.to_string(), alloc: alloc_bytes, logical: Some(logical_bytes) });
    }
    /// Host contract: the 32-bit kernarg scalar at `offset` lies in [lo, hi] (use `i64::MAX` for no upper bound).
    #[track_caller]
    pub fn ka_range(&mut self, offset: u32, lo: i64, hi: i64) {
        self.f.contract.ka_ranges.push((offset, lo, hi));
    }
    /// Host contract: launch grid axis (0 = x, 1 = y, 2 = z) equals `expr`, e.g. `b.grid_is(2, Ext::sym("B") * Ext::ka(0x280))` for B batches x splits.
    #[track_caller]
    pub fn grid_is(&mut self, axis: u8, expr: Ext) {
        self.f.contract.grid_defs.push((axis, expr));
    }
    /// Host contract: the u32 kernarg scalar at `offset` is one of `values` (e.g. the number of KV splits in {2, 4, 8}); the min/max become its range.
    #[track_caller]
    pub fn ka_one_of(&mut self, offset: u32, values: &[i64]) {
        self.f.contract.ka_sets.push((offset, values.to_vec()));
    }
    /// Host contract: the u32 kernarg scalar at `offset` equals `expr` (e.g. strides derived from the shape: `b.ka_is(0x70, Ext::grid_y() * Ext::ka(0x90))`).
    #[track_caller]
    pub fn ka_is(&mut self, offset: u32, expr: Ext) {
        self.f.contract.ka_defs.push((offset, expr));
    }
    /// Bounds of a free `Ext::sym` (default [1, +inf)).
    #[track_caller]
    pub fn sym_range(&mut self, name: &str, lo: i64, hi: i64) {
        self.f.contract.sym_ranges.push((name.to_string(), lo, hi));
    }
    /// Declares an LDS tile `[base, base+size)` (one slot). Tiles must not overlap each other unless aliased with `lds_alias`.
    #[track_caller]
    pub fn lds_tile(&mut self, name: &str, base: i64, size: i64) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base, slot_size: size, slots: 1, private: false, global: false, owned: None });
    }
    /// Declares a ring of `slots` equal slots starting at `base`.
    #[track_caller]
    pub fn lds_ring(&mut self, name: &str, base: i64, slot_size: i64, slots: u32) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base, slot_size, slots, private: false, global: false, owned: None });
    }
    /// Declares a tile of thread-OWNED columns: `bytes` at `base`, thread t owning the bytes `base + t col_stride + r row_stride + [0, col_stride)`
    /// for every row r (e.g. a [rows][threads] u32 table: col_stride 4, row_stride 4 x threads). Mark accesses with the usual tile markers; a
    /// marker scope whose every access the verifier proves to be in the issuing thread's own words needs no barrier against the same thread's
    /// other owner accesses and never races with other threads' owner accesses (accesses to other threads' words are ordinary shared events).
    #[track_caller]
    pub fn lds_owned(&mut self, name: &str, base: i64, bytes: i64, col_stride: i64, row_stride: i64) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base, slot_size: bytes, slots: 1, private: false, global: false, owned: Some((col_stride, row_stride)) });
    }
    /// Declares a per-WAVE private LDS region: `waves` slices of `size_per_wave` bytes at `base + wave * size_per_wave`. Mark accesses with `tile_produce/tile_consume(name,
    /// wave_key, version)` where `wave_key` is THE value the address is computed from (e.g. `base + wave_key * size + lane * 16`): the verifier proves the access lies in the
    /// slice of that very key (own wave) and that the wave reads only versions it produced and completed itself (no cross-wave publish is needed or assumed).
    #[track_caller]
    pub fn lds_private(&mut self, name: &str, base: i64, size_per_wave: i64, waves: u32) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base, slot_size: size_per_wave, slots: waves, private: true, global: false, owned: None });
    }
    /// The two tiles are allowed to overlap (time-multiplexed storage): the lifetime check then requires their events to be ordered like versions of one tile.
    #[track_caller]
    pub fn lds_alias(&mut self, a: &str, b: &str) {
        self.f.contract.tile_aliases.push((a.to_string(), b.to_string()));
    }
    /// `new` reuses the storage of `old` (it may overlap it, like `lds_alias`), and the protocol engine CHECKS the reuse (see `Contract::tile_reuses`).
    #[track_caller]
    pub fn lds_reuse(&mut self, new: &str, old: &str) {
        self.f.contract.tile_aliases.push((new.to_string(), old.to_string()));
        self.f.contract.tile_reuses.push((new.to_string(), old.to_string()));
    }
    /// Trust entry for the tile simulator: the slot / version expressions are NOT plain affine counters (xor/add/and/mul by constants) but the protocol is periodic in
    /// the loop iteration by construction, so the simulated iterations (>= 10 and >= 2 x ring slots + 4) cover every launch. Listed as TRUSTED in the report.
    #[track_caller]
    pub fn tile_trust_periodic(&mut self, reason: &str) {
        self.f.contract.tile_trust = Some(reason.to_string());
    }
    /// Declares a GLOBAL-memory region: bytes `[base, base + size)` of the pointer kernarg `arg`. Mark the writes with `buf_produce(name, gen)` and the reads with
    /// `buf_consume(name, gen)`: the verifier then checks the four protocol obligations (complete before publish: `wait_vm(0)` / `fence_release`; publish before read:
    /// release -> barrier -> acquire inside the workgroup; read before overwrite; generation consistency). `gen` follows the same affine/periodic rules as tile versions.
    #[track_caller]
    pub fn buf_region(&mut self, name: &str, arg: &str, base: Ext, size: Ext) {
        self.f.contract.tiles.push(TileDecl { name: name.to_string(), base: 0, slot_size: 1, slots: 1, private: false, global: true, owned: None });
        self.f.contract.buf_regions.push(BufRegion { name: name.to_string(), arg: arg.to_string(), base, size });
    }
    /// The global writes issued from here (until `tile_end` / the next marker / barrier / control boundary) produce generation `gen` of the region.
    #[track_caller]
    pub fn buf_produce(&mut self, name: &str, generation: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileProduce { tile: name.to_string(), slot: Opnd::I32(0), version: generation.into() });
    }
    /// The global reads issued from here expect generation `gen` of the region.
    #[track_caller]
    pub fn buf_consume(&mut self, name: &str, generation: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileConsume { tile: name.to_string(), slot: Opnd::I32(0), version: generation.into() });
    }
    /// Declares the fused-combine (last-arriver) pattern over `region`: the stores marked `buf_produce` (generation = the block's split id, derived from the workgroup id)
    /// must be followed by `fence_release`, then an atomic ticket on the pointer kernarg `counter_arg`; the reads marked `buf_consume` must sit in the branch guarded by the
    /// ticket result, after `fence_acquire` (or with `sc1` loads for the lite variant), and cover more than one slice.
    #[track_caller]
    pub fn buf_last_arriver(&mut self, region: &str, counter_arg: &str, splits: Ext) {
        self.f.contract.last_arrivers.push(LastArriver { region: region.to_string(), counter_arg: counter_arg.to_string(), splits });
    }
    /// Counted trust escape for one region: the verifier does not check it (e.g. a visibility argument it cannot make); listed in the certificate.
    #[track_caller]
    pub fn buf_trust(&mut self, region: &str, reason: &str) {
        self.f.contract.buf_trusts.push((region.to_string(), reason.to_string()));
    }
    /// See `Stmt::TileProduce`.
    #[track_caller]
    pub fn tile_produce(&mut self, tile: &str, slot: impl Into<Opnd>, version: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileProduce { tile: tile.to_string(), slot: slot.into(), version: version.into() });
    }
    /// See `Stmt::TileConsume`.
    #[track_caller]
    pub fn tile_consume(&mut self, tile: &str, slot: impl Into<Opnd>, version: impl Into<Opnd>) {
        self.push_stmt(Stmt::TileConsume { tile: tile.to_string(), slot: slot.into(), version: version.into() });
    }
    #[track_caller]
    pub fn tile_end(&mut self) {
        self.push_stmt(Stmt::TileEnd);
    }
    /// Opens a pipeline stage with its boundary contract (see `Stmt::StageBegin`).
    #[track_caller]
    pub fn stage_begin(&mut self, name: &str, contract: StageContract) {
        self.push_stmt(Stmt::StageBegin { name: name.to_string(), contract: Box::new(contract) });
    }
    #[track_caller]
    pub fn stage_end(&mut self) {
        self.push_stmt(Stmt::StageEnd);
    }
    /// Adds a hand-over set to the innermost open stage's contract (its values are usually computed inside the stage).
    #[track_caller]
    pub fn stage_live_out(&mut self, v: StageVal) {
        match self.tree.open.iter_mut().rev().find(|f| matches!(f, Frame::Stage { .. })) {
            Some(Frame::Stage { contract, .. }) => contract.live_out.push(v),
            _ => panic!("{}: stage_live_out outside a stage", self.f.name),
        }
    }
    /// Adds a scalar hand-over to the innermost open stage's contract (see `StageContract::scalars_out`).
    #[track_caller]
    pub fn stage_scalar_out(&mut self, s: StageScalar) {
        match self.tree.open.iter_mut().rev().find(|f| matches!(f, Frame::Stage { .. })) {
            Some(Frame::Stage { contract, .. }) => contract.scalars_out.push(s),
            _ => panic!("{}: stage_scalar_out outside a stage", self.f.name),
        }
    }
    /// Closure form of `stage_begin` / `stage_end`.
    #[track_caller]
    pub fn stage<R>(&mut self, name: &str, contract: StageContract, f: impl FnOnce(&mut Self) -> R) -> R {
        self.stage_begin(name, contract);
        let r = f(self);
        self.stage_end();
        r
    }
    /// Contract fact `lo <= target <= hi` from this point on (see `Stmt::Assume`). Emits no code.
    #[track_caller]
    pub fn assume(&mut self, target: impl Into<Opnd>, lo: Ext, hi: Ext) {
        self.push_stmt(Stmt::Assume { target: target.into(), lo, hi, trusted: None });
    }
    /// Like `assume`, but for a loop variable whose range holds by a guard the analyzer cannot see (e.g. a counter bounded by the loop exit test): the induction
    /// check failing is NOT an error, the assumption is listed as TRUSTED with `reason`.
    /// Trusted fact: `outs` (as a tuple) is an injective function of the workgroup id over grid `axes` - two different workgroups never get the
    /// same tuple (an XCD raster permuting the tile grid, a block-order bijection). The cross-workgroup store proof may then use `outs` as the
    /// workgroup digits of those axes. Counted in the trust ledger.
    #[track_caller]
    pub fn wg_bijection(&mut self, outs: &[ValueId], axes: &[u8], reason: &str) {
        self.f.contract.wg_maps.push(WgMap { outs: outs.to_vec(), axes: axes.to_vec(), reason: reason.to_string(), union: false, range: None });
    }

    /// Trusted fact: `outs` are the same index (e.g. a q tile) as computed by different store classes of one workgroup (the paired causal
    /// tiles x and n - 1 - x, the second guarded by n - 1 - x > x), and the index is injective over (workgroup, class). The cross-workgroup
    /// store proof treats them as one digit covering `axes`. Counted in the trust ledger.
    /// Trusted fact: the `then` branch of `if cond` runs in exactly ONE workgroup of each group of workgroups that differ only along grid `axes`
    /// (the last arriver of a ticket counter shared by the group). Stores in that branch need no digit for those axes. Counted in the ledger.
    #[track_caller]
    pub fn wg_winner(&mut self, cond: ValueId, axes: &[u8], reason: &str) {
        self.f.contract.wg_winners.push(WgWinner { cond, axes: axes.to_vec(), reason: reason.to_string() });
    }

    #[track_caller]
    pub fn wg_union(&mut self, outs: &[ValueId], axes: &[u8], lo: Ext, hi: Ext, reason: &str) {
        self.f.contract.wg_maps.push(WgMap { outs: outs.to_vec(), axes: axes.to_vec(), reason: reason.to_string(), union: true, range: Some((lo, hi)) });
    }

    #[track_caller]
    pub fn assume_trusted(&mut self, target: impl Into<Opnd>, lo: Ext, hi: Ext, reason: &str) {
        self.push_stmt(Stmt::Assume { target: target.into(), lo, hi, trusted: Some(reason.to_string()) });
    }

    #[track_caller]
    pub fn arg(&mut self, name: &str, ty: DType, offset: u32, size: u32) {
        self.f.args.push(Arg { name: name.to_string(), ty, kernarg_offset: offset, size });
        self.f.kernarg_size = self.f.kernarg_size.max(offset + size);
    }

    /// Embed a safe-asm sequence. `inputs` bind to the block's input ports in order (types are checked against the port classes by `verify`).
    #[track_caller]
    pub fn asm(&mut self, block: amdgpu_safeasm::AsmBlock, inputs: &[Opnd]) -> ValueId {
        self.let_(Op::Asm { block: std::sync::Arc::new(block), inputs: inputs.to_vec() })
    }

    #[track_caller]
    pub fn let_(&mut self, op: Op) -> ValueId {
        let ty = op.result_ty();
        let dst = self.f.value_tys.len() as ValueId;
        let of = |o: &Opnd, views: &std::collections::HashMap<ValueId, Vec<VarId>>| match o {
            Opnd::Var(v) => vec![*v],
            Opnd::Val(x) => views.get(x).cloned().unwrap_or_default(),
            _ => Vec::new(),
        };
        let vv = match &op {
            Op::Frag2Lane { frag, .. } | Op::Frag4Lane { frag, .. } | Op::Frag16Lane { frag, .. } => of(frag, &self.views),
            Op::BitcastF32(a) | Op::BitcastI32(a) => of(a, &self.views),
            Op::VarSlice8 { var, .. } | Op::VarSlice4 { var, .. } => vec![*var],
            _ => Vec::new(),
        };
        if !vv.is_empty() {
            self.views.insert(dst, vv);
        }
        self.f.value_tys.push(ty);
        // diagnostics side table: where this value was built (the caller through the #[track_caller] builder chain)
        let loc = std::panic::Location::caller();
        let n = self.f.value_tys.len();
        if self.f.src_locs.0.len() < n {
            self.f.src_locs.0.resize(n, None);
        }
        self.f.src_locs.0[dst as usize] = Some(loc);
        self.push_stmt(Stmt::Let { dst, op });
        dst
    }

    #[track_caller]
    pub fn store_u16(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreU16 { base: base.into(), byte_off, value: value.into() });
    }

    #[track_caller]
    pub fn store_f32(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32 { base: base.into(), byte_off, value: value.into() });
    }

    // ---- MUBUF (buffer_load/buffer_store): SGPR resource descriptor + per-lane 32-bit byte offset + SGPR soffset + 12-bit immediate ----
    /// Resource descriptor over the kernarg pointer `base` (bounds checking off: num_records = 0xFFFFFFFF). The descriptor lives in 4 SGPRs.
    #[track_caller]
    pub fn buffer_rsrc(&mut self, base: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BufferRsrc { base: base.into(), num_records: 0xFFFF_FFFF })
    }
    /// Descriptor with an explicit `num_records` (byte range; out-of-range accesses read 0 / are dropped).
    #[track_caller]
    pub fn buffer_rsrc_sized(&mut self, base: impl Into<Opnd>, num_records: u32) -> ValueId {
        self.let_(Op::BufferRsrc { base: base.into(), num_records })
    }
    #[track_caller]
    pub fn buffer_load_f32(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32) -> ValueId {
        self.let_(Op::BufferLoad { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 1 })
    }
    #[track_caller]
    pub fn buffer_load_b64(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32) -> ValueId {
        self.let_(Op::BufferLoad { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 2 })
    }
    #[track_caller]
    pub fn buffer_load_b128(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32) -> ValueId {
        self.let_(Op::BufferLoad { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 4 })
    }
    /// `off` forms (no per-lane vaddr): every lane reads the same address (resource base + soff + imm).
    /// Two `buffer_load_dwordx4` (offsets `imm_lo` / `imm_hi`) into the two halves of one Frag8 (an MFMA operand), without the copy of `cat_frag8`.
    #[track_caller]
    pub fn buffer_load_b128x2(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm_lo: u32, imm_hi: u32) -> ValueId {
        self.let_(Op::BufferLoadB128x2 { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm_lo, imm_hi })
    }

    #[track_caller]
    pub fn buffer_store_f32(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::BufferStore { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 1, value: value.into() });
    }
    #[track_caller]
    pub fn buffer_store_b64(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::BufferStore { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 2, value: value.into() });
    }
    #[track_caller]
    pub fn buffer_store_b128(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::BufferStore { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, dwords: 4, value: value.into() });
    }
    /// MUBUF -> LDS DMA of 16 bytes per lane (`buffer_load_dwordx4 ... lds`): LDS destination = M0 (`lds_addr`) + imm + 16 * lane.
    #[track_caller]
    pub fn buffer_load_lds_b128(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, lds_addr: impl Into<Opnd>, imm: u32) {
        self.push_stmt(Stmt::BufferLoadLds { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), lds_addr: lds_addr.into(), imm, dwords: 4 });
    }
    /// Descriptor with a RUNTIME byte range: `num_records` is a wave-uniform integer (kernarg value, scalar arithmetic or loop variable). Rebuild it where the
    /// value changes (the descriptor is an ordinary SGPR-quad value). Reads at offset >= num_records return 0, writes are dropped.
    #[track_caller]
    pub fn buffer_desc_dyn(&mut self, ptr: impl Into<Opnd>, num_records_bytes: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BufferRsrcDyn { base: ptr.into(), num_records: num_records_bytes.into() })
    }
    /// Alias names used by the kernels: descriptor over a kernarg pointer with `num_records` bytes (reads at offset >= num_records return 0, writes are dropped).
    #[track_caller]
    pub fn buffer_desc(&mut self, ptr: impl Into<Opnd>, num_records_bytes: u32) -> ValueId {
        self.buffer_rsrc_sized(ptr, num_records_bytes)
    }
    /// `buffer_load_dword[x4] ... lds` with the LDS slot base `lds_addr` (M0) and `dwords` = 1 or 4.
    #[track_caller]
    pub fn load_buffer_lds(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, lds_addr: impl Into<Opnd>, imm: u32, dwords: u8) {
        self.push_stmt(Stmt::BufferLoadLds { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), lds_addr: lds_addr.into(), imm, dwords });
    }
    /// 4 bytes per lane (`buffer_load_dword ... lds`).
    #[track_caller]
    pub fn buffer_load_lds_b32(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, lds_addr: impl Into<Opnd>, imm: u32) {
        self.push_stmt(Stmt::BufferLoadLds { rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), lds_addr: lds_addr.into(), imm, dwords: 1 });
    }

    #[track_caller]
    pub fn store_f32x2(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32x2 { base: base.into(), byte_off, value: value.into() });
    }

    #[track_caller]
    pub fn store_f32x4(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32x4 { base: base.into(), byte_off, value: value.into() });
    }

    #[track_caller]
    pub fn set_var(&mut self, var: VarId, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::SetVar { var, value: value.into() });
    }

    /// Declares `vars` initialised from `init`, then opens the loop body.
    #[track_caller]
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

    #[track_caller]
    pub fn break_if(&mut self, cond: impl Into<Opnd>) {
        self.push_stmt(Stmt::BreakIf { cond: cond.into() });
    }

    #[track_caller]
    pub fn end_loop(&mut self) {
        self.expr_consts.clear();
        self.push_stmt(Stmt::EndLoop);
    }

    #[track_caller]
    pub fn begin_if(&mut self, cond: impl Into<Opnd>) {
        self.push_stmt(Stmt::BeginIf { cond: cond.into() });
    }

    #[track_caller]
    pub fn else_(&mut self) {
        self.expr_consts.clear();
        self.push_stmt(Stmt::Else);
    }

    #[track_caller]
    pub fn end_if(&mut self) {
        self.expr_consts.clear();
        self.push_stmt(Stmt::EndIf);
    }

    /// Keep the loop variable `var` in the AGPR file (any placement). MFMAs that accumulate into it use the `acc_cd` form; VALU access is
    /// done through `v_accvgpr_read/write` inserted by the compiler (docs/agpr.md).
    #[track_caller]
    pub fn pin_agpr(&mut self, var: VarId) {
        self.f.agpr.push(AgprMark { target: AgprTarget::Var(var), pin: None });
    }

    /// As `pin_agpr`, at the exact AGPR index `base` (aligned to the register width).
    #[track_caller]
    pub fn pin_agpr_at(&mut self, var: VarId, base: u32) {
        self.f.agpr.push(AgprMark { target: AgprTarget::Var(var), pin: Some(base) });
    }

    /// Mark the register tuple that holds `v` (an LDS/global load result, an MFMA result, a fragment) as AGPR-resident; returns `v`.
    #[track_caller]
    pub fn agpr(&mut self, v: ValueId) -> ValueId {
        self.f.agpr.push(AgprMark { target: AgprTarget::Val(v), pin: None });
        v
    }

    // ---- register planning (RegMark; docs/weaver-regs.md) ----

    /// HARD: keep `r` (its whole tuple) at arch VGPR `base`.
    #[track_caller]
    pub fn pin_vgpr_at(&mut self, r: RegRef, base: u32) {
        self.f.reg_plan.push(RegMark::PinV { r, base });
    }
    /// HINT: let the members share registers when their lifetimes are disjoint.
    #[track_caller]
    pub fn reg_share(&mut self, rs: &[RegRef]) {
        self.f.reg_plan.push(RegMark::Share(rs.to_vec()));
    }
    /// HARD: the members never share physical registers.
    #[track_caller]
    pub fn reg_separate(&mut self, rs: &[RegRef]) {
        self.f.reg_plan.push(RegMark::Separate(rs.to_vec()));
    }
    /// HARD: arch-VGPR footprint cap of region `region` (a weave region or a stage name).
    #[track_caller]
    pub fn reg_cap(&mut self, region: &str, vgprs: u32) {
        self.f.reg_plan.push(RegMark::Cap { region: region.to_string(), vgprs: Some(vgprs), agprs: None });
    }
    /// HARD: AGPR footprint cap of region `region`.
    #[track_caller]
    pub fn reg_cap_agpr(&mut self, region: &str, agprs: u32) {
        self.f.reg_plan.push(RegMark::Cap { region: region.to_string(), vgprs: None, agprs: Some(agprs) });
    }
    /// The register buffers of one fragment (pairwise separate, placement reported).
    #[track_caller]
    pub fn reg_ring(&mut self, name: &str, bufs: &[RegRef]) {
        self.f.reg_plan.push(RegMark::Ring { name: name.to_string(), bufs: bufs.to_vec() });
    }
    /// Opens a register-planning region (no code). Must be closed by `weave_region_end` in the same structural block.
    #[track_caller]
    pub fn weave_region_begin(&mut self, name: &str) {
        self.push_stmt(Stmt::RegionMark { name: name.to_string(), begin: true });
    }
    /// Closes the innermost register-planning region.
    #[track_caller]
    pub fn weave_region_end(&mut self) {
        self.push_stmt(Stmt::RegionMark { name: String::new(), begin: false });
    }
    /// `weave_region_begin(name)`, `f`, `weave_region_end()`.
    #[track_caller]
    pub fn weave_region<R>(&mut self, name: &str, f: impl FnOnce(&mut Self) -> R) -> R {
        self.weave_region_begin(name);
        let r = f(self);
        self.weave_region_end();
        r
    }

    #[track_caller]
    pub fn finish(mut self) -> Function {
        assert!(self.captures.is_empty(), "kernel `{}`: {} capture(s) still open at finish()", self.f.name, self.captures.len());
        let tree = std::mem::take(&mut self.tree);
        let body = tree.finish().unwrap_or_else(|e| panic!("kernel `{}`: {e}", self.f.name));
        self.f.set_body(body);
        self.f
    }

    // ---- P1 VALU ops (docs/op-coverage-tileops.md) ----
    #[track_caller]
    pub fn un(&mut self, op: UnOp, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Un(op, a.into()))
    }
    #[track_caller]
    pub fn bin(&mut self, op: BinOp, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Bin(op, a.into(), b.into()))
    }
    #[track_caller]
    pub fn tri(&mut self, op: TriOp, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Tri(op, a.into(), b.into(), c.into()))
    }
    #[track_caller]
    pub fn cmp(&mut self, pred: CmpPred, ty: CmpTy, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Cmp { pred, ty, a: a.into(), b: b.into() })
    }

    // ---- P1 A3 memory ----
    #[track_caller]
    pub fn load_global(&mut self, base: impl Into<Opnd>, byte_off: u32, ty: MemTy) -> ValueId {
        self.load_global_c(base, byte_off, ty, CachePolicy::NONE)
    }
    /// `load_global` with a cache policy (`sc0` / `sc1` / `nt`).
    #[track_caller]
    pub fn load_global_c(&mut self, base: impl Into<Opnd>, byte_off: u32, ty: MemTy, cache: CachePolicy) -> ValueId {
        self.let_(Op::LoadGlobalN { base: base.into(), byte_off, ty, cache })
    }
    #[track_caller]
    pub fn load_global_d16(&mut self, base: impl Into<Opnd>, byte_off: u32, ty: D16Ty, hi: bool) -> ValueId {
        self.let_(Op::LoadGlobalD16 { base: base.into(), byte_off, ty, hi, cache: CachePolicy::NONE })
    }
    #[track_caller]
    pub fn store_global(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>, ty: MemTy, hi: bool) {
        self.store_global_c(base, byte_off, value, ty, hi, CachePolicy::NONE)
    }
    /// `store_global` with a cache policy.
    #[track_caller]
    pub fn store_global_c(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>, ty: MemTy, hi: bool, cache: CachePolicy) {
        self.push_stmt(Stmt::StoreGlobalN { base: base.into(), byte_off, value: value.into(), ty, hi, cache });
    }
    #[track_caller]
    pub fn buffer_load_n(&mut self, rsrc: impl Into<Opnd>, voff: Option<Opnd>, soff: impl Into<Opnd>, imm: u32, ty: MemTy) -> ValueId {
        self.buffer_load_c(rsrc, voff, soff, imm, ty, CachePolicy::NONE)
    }
    /// `buffer_load_n` with a cache policy.
    #[track_caller]
    pub fn buffer_load_c(&mut self, rsrc: impl Into<Opnd>, voff: Option<Opnd>, soff: impl Into<Opnd>, imm: u32, ty: MemTy, cache: CachePolicy) -> ValueId {
        self.let_(Op::BufferLoadN { rsrc: rsrc.into(), voff, soff: soff.into(), imm, ty, cache })
    }
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn buffer_store_n(&mut self, rsrc: impl Into<Opnd>, voff: Option<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>, ty: MemTy, hi: bool) {
        self.buffer_store_c(rsrc, voff, soff, imm, value, ty, hi, CachePolicy::NONE)
    }
    /// `buffer_store_n` with a cache policy.
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn buffer_store_c(&mut self, rsrc: impl Into<Opnd>, voff: Option<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>, ty: MemTy, hi: bool, cache: CachePolicy) {
        self.push_stmt(Stmt::BufferStoreN { rsrc: rsrc.into(), voff, soff: soff.into(), imm, value: value.into(), ty, hi, cache });
    }
    #[track_caller]
    pub fn load_lds(&mut self, addr: impl Into<Opnd>, ty: MemTy) -> ValueId {
        self.let_(Op::LoadLdsN { addr: addr.into(), ty })
    }
    #[track_caller]
    pub fn load_lds_d16(&mut self, addr: impl Into<Opnd>, ty: D16Ty, hi: bool) -> ValueId {
        self.let_(Op::LoadLdsD16 { addr: addr.into(), ty, hi })
    }
    #[track_caller]
    pub fn store_lds(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>, ty: MemTy, hi: bool) {
        self.push_stmt(Stmt::StoreLdsN { addr: addr.into(), value: value.into(), ty, hi });
    }
    #[track_caller]
    pub fn load_lds2(&mut self, addr: impl Into<Opnd>, off0: u8, off1: u8, b64: bool, st64: bool) -> ValueId {
        self.let_(Op::LoadLds2 { addr: addr.into(), off0, off1, b64, st64 })
    }
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn store_lds2(&mut self, addr: impl Into<Opnd>, v0: impl Into<Opnd>, v1: impl Into<Opnd>, off0: u8, off1: u8, b64: bool, st64: bool) {
        self.push_stmt(Stmt::StoreLds2 { addr: addr.into(), v0: v0.into(), v1: v1.into(), off0, off1, b64, st64 });
    }
    #[track_caller]
    pub fn load_lds_tr_b16(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsTrB16 { addr: addr.into() })
    }

    /// LDS atomic: returns the old value (`ret`) or `None` (no-return form, cheaper).
    #[track_caller]
    pub fn atomic_lds(&mut self, op: AtomicOp, addr: impl Into<Opnd>, value: impl Into<Opnd>, ret: bool) -> Option<ValueId> {
        let (addr, value) = (addr.into(), value.into());
        if ret {
            Some(self.let_(Op::AtomicLds { op, addr, value }))
        } else {
            self.push_stmt(Stmt::AtomicLdsNoRet { op, addr, value });
            None
        }
    }
    /// Global atomic: returns the old value (`ret`) or `None`.
    #[track_caller]
    pub fn atomic_global(&mut self, op: AtomicOp, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>, ret: bool) -> Option<ValueId> {
        let (base, value) = (base.into(), value.into());
        if ret {
            Some(self.let_(Op::AtomicGlobal { op, base, byte_off, value }))
        } else {
            self.push_stmt(Stmt::AtomicGlobalNoRet { op, base, byte_off, value });
            None
        }
    }
    /// `base + (u64) a * (u64) b` (unsigned 32-bit factors, 64-bit product): addresses beyond 4 GiB (`index * stride`). Wave-uniform operands
    /// give a scalar pointer (usable as a buffer-descriptor base: move the descriptor to the row / block, keep per-lane offsets small).
    #[track_caller]
    pub fn ptr_mad(&mut self, base: impl Into<Opnd>, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PtrMadU32 { base: base.into(), a: a.into(), b: b.into() })
    }
    /// `v_pk_fma_f32`: a * b + c, two f32 lanes per instruction (Frag2 operands).
    #[track_caller]
    pub fn pk_fma_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PkFmaF32(a.into(), b.into(), c.into()))
    }
    /// In-place block update `var[start + k] += t[k] * s` (k over the dwords of `t`, two per `v_pk_fma_f32`; `s` a Frag2 such as
    /// `frag2_dup(scale)`), e.g. the per-tile block-scale accumulation `acc += t * (sa sb)` of the blockscale GEMMs.
    #[track_caller]
    pub fn var_pk_fma(&mut self, var: VarId, start: u8, t: impl Into<Opnd>, s: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarPkFma { var, start, t: t.into(), s: s.into() });
    }
    /// Mark a per-lane condition for a wave-level branch: the wave takes the branch when ANY lane's `cond` holds (`if` / `break_if` conditions must
    /// be wave-uniform; this is the explicit way to branch on a per-lane one).
    #[track_caller]
    pub fn any_lane(&mut self, cond: impl Into<Opnd>) -> ValueId {
        self.let_(Op::AnyLane(cond.into()))
    }
    /// Dwords `start .. start + n` of fragment `frag` as a view (no instruction).
    #[track_caller]
    pub fn frag_slice(&mut self, frag: impl Into<Opnd>, start: u8, n: u8) -> ValueId {
        assert!(matches!(n, 1 | 2 | 4 | 8), "frag_slice: n is 1, 2, 4 or 8");
        self.let_(Op::FragSlice { frag: frag.into(), start, n })
    }
    /// `lo ++ hi` as one Frag4 (e.g. two `ds_read_b64_tr_b16` results = a bf16 MFMA operand). Fresh single-use parts are written IN PLACE (the
    /// defining instruction targets its half of the result: no copies, no early wait).
    #[track_caller]
    pub fn cat_frag4(&mut self, lo: impl Into<Opnd>, hi: impl Into<Opnd>) -> ValueId {
        let (lo, hi) = (lo.into(), hi.into());
        self.frag_cat(&[lo, hi])
    }
    /// Two `ds_read_b64_tr_b16` (at `addr0`, `addr1`) as one Frag4, each read writing its half directly.
    #[track_caller]
    pub fn load_lds_tr_b16x2(&mut self, addr0: impl Into<Opnd>, addr1: impl Into<Opnd>) -> ValueId {
        let r0 = self.load_lds_tr_b16(addr0);
        let r1 = self.load_lds_tr_b16(addr1);
        self.cat_frag4(r0, r1)
    }
    /// Mark a fresh load result (`buffer_load_b128`, ...) for IN-PLACE use: `set_var(var, b.inplace(load))` makes the load write the loop
    /// variable's own registers (no copy, no wait at the definition). Plain `set_var(var, load)` keeps the copy (the shipped behaviour).
    #[track_caller]
    pub fn inplace(&mut self, v: impl Into<Opnd>) -> ValueId {
        let v = v.into();
        self.frag_cat(&[v])
    }
    /// Two dwords as an (even-aligned) Frag2 with fresh single-use parts written IN PLACE - the operand form of the packed f32 ops
    /// (`v_pk_add_f32` / `v_pk_mul_f32` / `v_pk_fma_f32`): the two defining instructions target the halves of one register pair, no moves
    /// (`frag2` copies).
    #[track_caller]
    pub fn pack2(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        let p = [a.into(), b.into()];
        self.frag_cat(&p)
    }
    /// Four dwords as a Frag4 with fresh single-use parts written in place (`frag4` copies; this does not).
    #[track_caller]
    pub fn pack4(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>, d: impl Into<Opnd>) -> ValueId {
        let p = [a.into(), b.into(), c.into(), d.into()];
        self.frag_cat(&p)
    }
    /// Concatenate one-dword values / fragments into a fragment (total 2, 4, 8 or 16 dwords).
    #[track_caller]
    pub fn frag_cat(&mut self, parts: &[Opnd]) -> ValueId {
        let total: u32 = parts.iter().map(|p| self.opnd_dwords(*p)).sum();
        assert!(matches!(total, 2 | 4 | 8 | 16), "frag_cat: {total} dwords (2, 4, 8 or 16)");
        self.let_(Op::FragCat { parts: parts.to_vec(), dwords: total as u8 })
    }
    #[track_caller]
    fn opnd_dwords(&self, o: Opnd) -> u32 {
        match o {
            Opnd::Val(v) => match self.f.value_tys[v as usize] {
                DType::Frag2 | DType::Ptr => 2,
                DType::Frag4 => 4,
                DType::Frag8 => 8,
                DType::Frag16 => 16,
                _ => 1,
            },
            Opnd::Var(v) => match self.f.var_tys[v.0 as usize] {
                DType::Frag2 => 2,
                DType::Frag4 => 4,
                DType::Frag8 => 8,
                DType::Frag16 => 16,
                _ => 1,
            },
            _ => 1,
        }
    }
    /// `v_readlane_b32`: lane `lane`'s `x` (uniform).
    #[track_caller]
    pub fn read_lane(&mut self, x: impl Into<Opnd>, lane: u8) -> ValueId {
        self.let_(Op::ReadLane { x: x.into(), lane })
    }
    /// `v_writelane_b32`: `old` with lane `lane` set to `x`.
    #[track_caller]
    pub fn write_lane(&mut self, x: impl Into<Opnd>, lane: u8, old: impl Into<Opnd>) -> ValueId {
        self.let_(Op::WriteLane { x: x.into(), lane, old: old.into() })
    }
    /// (low, high) halves of the wave mask of lanes where `cond` holds.
    #[track_caller]
    pub fn ballot(&mut self, cond: impl Into<Opnd>) -> (ValueId, ValueId) {
        let c = cond.into();
        (self.let_(Op::Ballot { cond: c, hi: false }), self.let_(Op::Ballot { cond: c, hi: true }))
    }
    /// The lane's index among the lanes set in (`lo`, `hi`) below it (`v_mbcnt_lo` + `v_mbcnt_hi`); with -1 / -1: the lane id.
    #[track_caller]
    pub fn mbcnt(&mut self, lo: impl Into<Opnd>, hi: impl Into<Opnd>) -> ValueId {
        let l = self.let_(Op::Bin(BinOp::MbcntLo, lo.into(), Opnd::I32(0)));
        self.let_(Op::Bin(BinOp::MbcntHi, hi.into(), Opnd::Val(l)))
    }
    /// `ds_bpermute_b32`: lane k gets `x` of lane addr[k] / 4.
    #[track_caller]
    pub fn bpermute(&mut self, addr: impl Into<Opnd>, x: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Bpermute { addr: addr.into(), x: x.into(), offset: 0 })
    }
    /// `ds_permute_b32`: lane k sends `x` to lane addr[k] / 4.
    #[track_caller]
    pub fn permute(&mut self, addr: impl Into<Opnd>, x: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Permute { addr: addr.into(), x: x.into(), offset: 0 })
    }
    /// `ds_swizzle_b32` with the raw 16-bit pattern.
    #[track_caller]
    pub fn swizzle(&mut self, x: impl Into<Opnd>, pattern: u16) -> ValueId {
        self.let_(Op::Swizzle { x: x.into(), pattern })
    }
    /// MUBUF atomic without return (`AddU32`, `AddF32`, `PkAddBf16`, `PkAddF16`).
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    /// Returning `buffer_atomic_add` (sc0): the old value per lane. Out-of-range lanes are dropped (return 0) - mask the lanes that must not
    /// take part with an out-of-range offset, e.g. `voff = select(tid == 0, off, 0x7fff_fff0)` for a single-lane ticket.
    pub fn buffer_atomic_add_ret(&mut self, rsrc: ValueId, voff: impl Into<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BufferAtomicRet { op: AtomicOp::AddU32, rsrc: Opnd::Val(rsrc), voff: Some(voff.into()), soff: soff.into(), imm, value: value.into() })
    }

    #[track_caller]
    pub fn buffer_atomic(&mut self, op: AtomicOp, rsrc: impl Into<Opnd>, voff: Option<Opnd>, soff: impl Into<Opnd>, imm: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::BufferAtomicNoRet { op, rsrc: rsrc.into(), voff, soff: soff.into(), imm, value: value.into() });
    }
    /// `v_mov_b32_dpp` with a full DPP control; lanes not written keep `old`.
    #[track_caller]
    pub fn dpp(&mut self, dpp: Dpp, x: impl Into<Opnd>, old: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Dpp { dpp, x: x.into(), old: old.into() })
    }
    /// `dpp(a) op b` in one instruction (`v_<op>_dpp`); lanes not written keep `old`.
    #[track_caller]
    pub fn dpp_alu(&mut self, op: DppAluOp, dpp: Dpp, a: impl Into<Opnd>, b: impl Into<Opnd>, old: impl Into<Opnd>) -> ValueId {
        self.let_(Op::DppAlu { op, dpp, a: a.into(), b: b.into(), old: old.into() })
    }
    /// `v_permlane16_swap_b32`: (new a, new b) as a Frag2.
    #[track_caller]
    pub fn permlane16_swap(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Permlane16Swap { a: a.into(), b: b.into() })
    }
    /// A conversion (`CvtOp`); pass the used operands in order.
    #[track_caller]
    pub fn cvt(&mut self, op: CvtOp, srcs: &[Opnd], sel: u8) -> ValueId {
        let mut s = [Opnd::I32(0); 4];
        assert_eq!(srcs.len(), op.sig().0, "cvt {op:?}: operand count");
        s[..srcs.len()].copy_from_slice(srcs);
        self.let_(Op::Cvt { op, srcs: s, sel })
    }

    // ---- P1-d MFMA ----
    /// D = A x B + C (`c: None`: zero accumulator).
    #[track_caller]
    pub fn mfma(&mut self, kind: MmaKind, a: impl Into<Opnd>, b: impl Into<Opnd>, c: Option<Opnd>) -> ValueId {
        self.let_(Op::Mfma { kind, a: a.into(), b: b.into(), c, scale: None })
    }
    /// Block-scaled D = A x B + C of a scaled f8f6f4 kind (`MmaKind::scaled_f8f6f4`), E8M0 scales `s`.
    #[track_caller]
    pub fn mfma_scaled(&mut self, kind: MmaKind, a: impl Into<Opnd>, b: impl Into<Opnd>, c: Option<Opnd>, s: MmaScale) -> ValueId {
        self.let_(Op::Mfma { kind, a: a.into(), b: b.into(), c, scale: Some(s) })
    }
    /// Block-scaled var = A x B + var (`zero`: var = A x B).
    #[track_caller]
    pub fn var_mfma_scaled(&mut self, var: VarId, kind: MmaKind, a: impl Into<Opnd>, b: impl Into<Opnd>, s: MmaScale, zero: bool) {
        self.push_stmt(Stmt::VarMfma { var, kind, a: a.into(), b: b.into(), scale: Some(s), zero });
    }
    /// var = A x B + var (`zero`: var = A x B).
    #[track_caller]
    pub fn var_mfma(&mut self, var: VarId, kind: MmaKind, a: impl Into<Opnd>, b: impl Into<Opnd>, zero: bool) {
        self.push_stmt(Stmt::VarMfma { var, kind, a: a.into(), b: b.into(), scale: None, zero });
    }

    // ---- capture mode (docs/weave-capture.md) ----

    /// Starts capturing: the statements emitted from now on (and the regions opened AND closed meanwhile) go into a detached buffer instead
    /// of the current block. Ids are allocated as usual, so values defined inside are usable by later code once the capture is spliced
    /// before it (the weaver orders by the dependency facts, `facts::must_order`). Captures nest (LIFO).
    #[track_caller]
    pub fn capture_begin(&mut self) -> CaptureId {
        let inner = TreeBuilder { count: self.tree.count, barriers: self.tree.barriers, ..Default::default() };
        let outer = std::mem::replace(&mut self.tree, inner);
        self.captures.push(outer);
        CaptureId(self.captures.len())
    }

    /// Ends the capture `id` (the innermost open one) and returns its statements; the regions opened inside must be closed.
    #[track_caller]
    pub fn capture_end(&mut self, id: CaptureId) -> Captured {
        assert_eq!(id.0, self.captures.len(), "kernel `{}`: capture_end out of order (captures nest LIFO)", self.f.name);
        let outer = self.captures.pop().expect("capture_end without capture_begin");
        let inner = std::mem::replace(&mut self.tree, outer);
        self.tree.count = inner.count;
        self.tree.barriers = inner.barriers;
        let nodes = inner.finish().unwrap_or_else(|e| panic!("kernel `{}`: capture: {e}", self.f.name));
        Captured { nodes }
    }

    /// Closure form: `(captured statements, closure result)`.
    #[track_caller]
    pub fn capture<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> (Captured, R) {
        let id = self.capture_begin();
        let r = f(self);
        (self.capture_end(id), r)
    }

    /// Appends captured statements to the current block (in their captured order). Splicing every capture once, in capture order, gives
    /// exactly the function a direct build gives.
    #[track_caller]
    pub fn splice(&mut self, c: Captured) {
        self.splice_nodes(c.nodes);
    }

    /// Appends nodes (of one or more captures, in a chosen order) to the current block. A tile region may not be spliced into an open tile region.
    #[track_caller]
    pub fn splice_nodes(&mut self, nodes: impl IntoIterator<Item = Node>) {
        let in_region = self.tree.open.iter().any(|f| matches!(f, Frame::Region { .. }));
        for n in nodes {
            if in_region && contains_region(&n) {
                panic!("kernel `{}`: splicing a tile region into an open tile region", self.f.name);
            }
            self.tree.cur().push(n);
        }
    }

    /// Dependency facts of captured statements in this function (value types known). A use of a register view of a loop variable (a lane /
    /// slice / bitcast of it, no instruction of its own) is also a READ of the variable: the user reads the registers at its own position.
    #[track_caller]
    pub fn facts(&self, c: &Captured) -> Vec<facts::Facts> {
        let mut fs = c.facts(&self.f);
        for f in fs.iter_mut() {
            for u in f.uses.clone() {
                if let Some(vs) = self.views.get(&u) {
                    for v in vs {
                        if !f.var_reads.contains(v) {
                            f.var_reads.push(*v);
                        }
                    }
                }
            }
        }
        fs
    }
    /// The loop variables a value is a register view of (empty when it is a value of its own).
    #[track_caller]
    pub fn view_vars(&self, v: ValueId) -> &[VarId] {
        self.views.get(&v).map_or(&[], |x| x.as_slice())
    }

    /// Statements of the flat view emitted so far (markers included).
    pub(crate) fn tree_count(&self) -> usize {
        self.tree.count
    }

    /// Appends a statement to the innermost open region (markers open / close regions); an unbalanced end panics here, at build time.
    #[track_caller]
    fn push_stmt(&mut self, st: Stmt) {
        if let Err(e) = self.tree.push(st) {
            panic!("kernel `{}`: {e}", self.f.name);
        }
    }

    /// Closure form of `begin_loop` / `end_loop`: `body` receives the loop variables.
    #[track_caller]
    pub fn loop_<R>(&mut self, inits: &[(DType, Opnd)], body: impl FnOnce(&mut Self, &[VarId]) -> R) -> R {
        let vars = self.begin_loop(inits);
        let r = body(self, &vars);
        self.end_loop();
        r
    }
    /// Closure form of `begin_if` / `end_if`.
    #[track_caller]
    pub fn if_<R>(&mut self, cond: impl Into<Opnd>, then: impl FnOnce(&mut Self) -> R) -> R {
        self.begin_if(cond);
        let r = then(self);
        self.end_if();
        r
    }
    /// Closure form of `begin_if` / `else_` / `end_if`.
    #[track_caller]
    pub fn if_else(&mut self, cond: impl Into<Opnd>, then: impl FnOnce(&mut Self), els: impl FnOnce(&mut Self)) {
        self.begin_if(cond);
        then(self);
        self.else_();
        els(self);
        self.end_if();
    }
    /// Closure form of `tile_produce` / `tile_end`.
    #[track_caller]
    pub fn produce_region<R>(&mut self, tile: &str, slot: impl Into<Opnd>, version: impl Into<Opnd>, body: impl FnOnce(&mut Self) -> R) -> R {
        self.tile_produce(tile, slot, version);
        let r = body(self);
        self.tile_end();
        r
    }

    // ---- convenience constructors -------------------------------------

    #[track_caller]
    pub fn c_i32(&mut self, v: i32) -> ValueId {
        self.let_(Op::ConstI32(v))
    }

    #[track_caller]
    pub fn c_f32(&mut self, v: f32) -> ValueId {
        self.let_(Op::ConstF32(v))
    }

    #[track_caller]
    pub fn add_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::AddI32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn min_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MinI32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn mul_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MulI32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn shl_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ShlI32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn add_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::AddF32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn sub_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::SubF32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn mul_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MulF32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn max_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MaxF32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn xor_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::XorI32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn max3_f32(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::Max3F32(a.into(), b.into(), c.into()))
    }

    #[track_caller]
    pub fn exp_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ExpF32(a.into()))
    }

    #[track_caller]
    pub fn rcp_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::RcpF32(a.into()))
    }

    #[track_caller]
    pub fn cvt_f32_to_f16(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtF32ToF16(a.into()))
    }

    #[track_caller]
    pub fn cmp_lt_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CmpLtF32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn cmp_lt_i32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CmpLtI32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn load_u16(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadU16 { base: base.into(), byte_off })
    }

    /// Coherent (`sc1`) f32 load: reads through to the device-coherent level.
    #[track_caller]
    pub fn load_f32_coherent(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadF32Coherent { base: base.into(), byte_off })
    }

    /// `global_atomic_add` returning the OLD value, device scope. Per-lane semantics: every active lane adds ITS `value`; pass `1` in lane 0 and `0`
    /// elsewhere for a one-ticket-per-wave counter, then `read_first_lane` the result.
    #[track_caller]
    pub fn atomic_add_u32(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) -> ValueId {
        self.let_(Op::AtomicAddU32 { base: base.into(), byte_off, value: value.into() })
    }

    #[track_caller]
    pub fn store_f32_wt(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreF32Wt { base: base.into(), byte_off, value: value.into() });
    }

    /// Device-scope release: wait for all earlier loads/stores, write back L2 (`buffer_wbl2 sc1`), wait again.
    #[track_caller]
    pub fn fence_release(&mut self) {
        self.push_stmt(Stmt::FenceRelease { wb: true });
    }

    /// Release without the L2 write-back (valid when every shared store used `store_*_wt`).
    #[track_caller]
    pub fn fence_release_lite(&mut self) {
        self.push_stmt(Stmt::FenceRelease { wb: false });
    }

    /// Device-scope acquire: wait for the ticket, invalidate (`buffer_inv sc1`).
    #[track_caller]
    pub fn fence_acquire(&mut self) {
        self.push_stmt(Stmt::FenceAcquire { inv: true });
    }

    #[track_caller]
    pub fn fence_acquire_lite(&mut self) {
        self.push_stmt(Stmt::FenceAcquire { inv: false });
    }

    #[track_caller]
    pub fn load_f32(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadF32 { base: base.into(), byte_off })
    }

    #[track_caller]
    pub fn ptr_add(&mut self, base: impl Into<Opnd>, byte_off: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PtrAdd { base: base.into(), byte_off: byte_off.into() })
    }

    #[track_caller]
    pub fn frag2(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Frag2(a.into(), b.into()))
    }

    /// `v_permlane32_swap`: `(a', b')` with a' = [a lanes 0..31, b lanes 0..31] and b' = [a lanes 32..63, b lanes 32..63] (lanes 32..63 of `a`
    /// exchanged with lanes 0..31 of `b`). 32-bit values.
    #[track_caller]
    pub fn permlane32_swap(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> (ValueId, ValueId) {
        let p = self.let_(Op::Permlane32Swap { a: a.into(), b: b.into() });
        (self.frag2_lane(p, 0), self.frag2_lane(p, 1))
    }

    #[track_caller]
    pub fn frag2_lane(&mut self, frag: impl Into<Opnd>, index: u8) -> ValueId {
        self.let_(Op::Frag2Lane { frag: frag.into(), index })
    }

    #[track_caller]
    pub fn frag4(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
        d: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::Frag4(a.into(), b.into(), c.into(), d.into()))
    }

    #[track_caller]
    pub fn frag4_lane(&mut self, frag: impl Into<Opnd>, index: u8) -> ValueId {
        self.let_(Op::Frag4Lane { frag: frag.into(), index })
    }

    #[track_caller]
    pub fn mfma_f16(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::MfmaF16 { a: a.into(), b: b.into(), c: c.into() })
    }

    #[track_caller]
    pub fn bitcast_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BitcastF32(a.into()))
    }

    #[track_caller]
    pub fn bitcast_i32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::BitcastI32(a.into()))
    }

    #[track_caller]
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
    #[track_caller]
    pub fn cvt_pknorm_u16_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPknormU16(a.into(), b.into()))
    }

    /// Pack two f32 into two E4M3 fp8 bytes (upper 16 bits are zero).
    #[track_caller]
    pub fn exp2_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Exp2F32(a.into()))
    }

    #[track_caller]
    pub fn log2_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Log2F32(a.into()))
    }

    #[track_caller]
    pub fn cvt_pk_bf16_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPkBf16F32(a.into(), b.into()))
    }

    #[track_caller]
    pub fn cat_frag8(&mut self, lo: impl Into<Opnd>, hi: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CatFrag8(lo.into(), hi.into()))
    }

    #[track_caller]
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

    #[track_caller]
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

    #[track_caller]
    pub fn frag16_lane(&mut self, frag: impl Into<Opnd>, index: u8) -> ValueId {
        self.let_(Op::Frag16Lane { frag: frag.into(), index })
    }

    /// Sparse fp8 MFMA (`v_smfmac_f32_16x16x128_fp8_fp8`): D = compressed A x dense B + C.
    #[track_caller]
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
    #[track_caller]
    pub fn frag_zero(&mut self, ty: DType) -> ValueId {
        self.let_(Op::FragZero { ty })
    }

    /// One dword read from workgroup LDS.
    #[track_caller]
    pub fn load_lds_b32(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB32 { addr: addr.into() })
    }

    /// One dword write into workgroup LDS.
    #[track_caller]
    pub fn store_lds_b32(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreLdsB32 { addr: addr.into(), value: value.into() });
    }

    /// Four dwords (16 B) read from workgroup LDS.
    #[track_caller]
    pub fn load_lds_b128(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB128 { addr: addr.into() })
    }

    /// `ds_read_b64`: two dwords from LDS (8-byte aligned) as a `Frag2`.
    /// Wave-uniform select of two buffer descriptors (`s_cmp` + 2 x `s_cselect_b64`, no branch): `cond` must be a (possibly negated) signed compare
    /// of wave-uniform integers. The GPU-semantics verifier proves every access through the result against BOTH descriptors.
    #[track_caller]
    pub fn select_uniform_rsrc(&mut self, cond: impl Into<Opnd>, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Select { ty: DType::Rsrc, cond: cond.into(), a: a.into(), b: b.into() })
    }
    /// Wave-uniform select of two SGPR pointers (kernarg pointers or uniform pointer arithmetic; `s_cmp` + `s_cselect_b64`). Accesses through the
    /// result are proven against BOTH pointers' extents.
    #[track_caller]
    pub fn select_uniform_ptr(&mut self, cond: impl Into<Opnd>, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Select { ty: DType::Ptr, cond: cond.into(), a: a.into(), b: b.into() })
    }

    #[track_caller]
    pub fn load_lds_b64(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB64 { addr: addr.into() })
    }

    /// Four dwords (16 B) write into workgroup LDS.
    /// Transposing 8-bit LDS read into two dwords per lane.
    /// 16-byte global load into a `Frag4`.
    #[track_caller]
    pub fn load_b128(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadGlobalB128 { base: base.into(), byte_off })
    }

    /// Four transposing reads into one `Frag8` (bytes 8r..8r+7 from the r-th address).
    #[track_caller]
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
    #[track_caller]
    pub fn load_lds_b128x2(&mut self, lo: impl Into<Opnd>, hi: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB128x2 { lo: lo.into(), hi: hi.into() })
    }

    #[track_caller]
    pub fn partner_max_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PartnerMaxF32(a.into()))
    }
    #[track_caller]
    pub fn frag2_dup(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Frag2Dup(a.into()))
    }
    #[track_caller]
    pub fn pk_add_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PkAddF32(a.into(), b.into()))
    }
    #[track_caller]
    pub fn pk_mul_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PkMulF32(a.into(), b.into()))
    }
    #[track_caller]
    pub fn cvt_pk_fp8x4_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>, d: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPkFp8x4F32(a.into(), b.into(), c.into(), d.into()))
    }
    #[track_caller]
    pub fn var_mfma_scale_c(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>, scale_a: impl Into<Opnd>, scale_b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaScaleC { var, a: a.into(), b: b.into(), c: c.into(), scale_a: scale_a.into(), scale_b: scale_b.into() });
    }

    /// Unscaled fp8 MFMA `a x b + c` (value form; c = frag_zero gives the zero-accumulator instruction).
    #[track_caller]
    pub fn mfma_fp8_32x32x64(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>) -> ValueId {
        self.let_(Op::MfmaFp8_32x32x64 { a: a.into(), b: b.into(), c: c.into() })
    }
    /// `var += a x b` (unscaled fp8, in place).
    #[track_caller]
    pub fn var_mfma_fp8_32x32x64(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaFp8 { var, a: a.into(), b: b.into() });
    }
    /// `var = a x b` (unscaled fp8, zero accumulator).
    #[track_caller]
    pub fn var_mfma_fp8_32x32x64_z(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaFp8Z { var, a: a.into(), b: b.into() });
    }
    /// `var = a x b + c` (unscaled fp8; c a separate Frag16 value).
    #[track_caller]
    pub fn var_mfma_fp8_32x32x64_c(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaFp8C { var, a: a.into(), b: b.into(), c: c.into() });
    }
    #[track_caller]
    pub fn var_scale_lane(&mut self, var: VarId, index: u8, scalar: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarScaleLane { var, index, scalar: scalar.into() });
    }

    /// `var.lane[index] = value` (32-bit value into one dword of a fragment variable).
    #[track_caller]
    pub fn var_set_lane(&mut self, var: VarId, index: u8, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarSetLane { var, index, value: value.into() });
    }

    /// Dwords 8 half .. 8 half + 7 of the Frag16 variable `var` as a Frag8 operand without a copy (see `Op::VarSlice8`).
    #[track_caller]
    pub fn var_slice8(&mut self, var: VarId, half: u8) -> ValueId {
        assert!(half < 2, "var_slice8: half is 0 or 1");
        self.let_(Op::VarSlice8 { var, half })
    }
    /// Dwords 4 idx .. 4 idx + 3 of a Frag8 / Frag16 variable as a Frag4 operand without a copy (see `Op::VarSlice4`).
    #[track_caller]
    pub fn var_slice4(&mut self, var: VarId, idx: u8) -> ValueId {
        self.let_(Op::VarSlice4 { var, idx })
    }
    #[track_caller]
    pub fn perm_b32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, sel: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PermB32(a.into(), b.into(), sel.into()))
    }
    #[track_caller]
    pub fn pk_sub_u16(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PkSubU16(a.into(), b.into()))
    }
    #[track_caller]
    pub fn cvt_pknorm_pair(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPknormPair(a.into(), b.into()))
    }
    #[track_caller]
    pub fn cvt_p8x4_from_u16(&mut self, u0: impl Into<Opnd>, u1: impl Into<Opnd>, scale: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtP8x4FromU16(u0.into(), u1.into(), scale.into()))
    }
    #[track_caller]
    pub fn cvt_pknorm_p8x4(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>, d: impl Into<Opnd>, scale: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPknormP8x4(a.into(), b.into(), c.into(), d.into(), scale.into()))
    }
    #[track_caller]
    pub fn var_add_pk(&mut self, var: VarId, pair: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarAddPk { var, pair: pair.into() });
    }
    #[track_caller]
    pub fn var_mfma_f16_z(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarMfmaF16Z { var, a: a.into(), b: b.into() });
    }
    #[track_caller]
    pub fn var_scale_pk(&mut self, var: VarId, pair: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarScalePk { var, pair: pair.into() });
    }

    /// `var = scaled_mfma(a, b, var)` in place.
    #[track_caller]
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
    #[track_caller]
    pub fn var_mfma_scale_sel(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, scale_a: impl Into<Opnd>, scale_b: impl Into<Opnd>, byte_a: u8, byte_b: u8) {
        assert!(byte_a < 4 && byte_b < 4, "scale byte select is 0..3");
        self.push_stmt(Stmt::VarMfmaScale { var, a: a.into(), b: b.into(), scale_a: scale_a.into(), scale_b: scale_b.into(), opsel: byte_a | (byte_b << 4) });
    }

    /// `var += smfmac_16x16x128_fp8_fp8(a, b, idx)`: accumulate into a `Frag4` loop variable in place.
    #[track_caller]
    pub fn var_smfmac_fp8(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, idx: impl Into<Opnd>) {
        self.push_stmt(Stmt::VarSmfmac { var, a: a.into(), b: b.into(), idx: idx.into() });
    }

    /// `var += v_mfma_scale_f32_16x16x128_f8f6f4(a, b; scale_a.byte[byte_a], scale_b.byte[byte_b])` on a `Frag4` loop variable.
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn var_mfma_scale16(&mut self, var: VarId, a: impl Into<Opnd>, b: impl Into<Opnd>, scale_a: impl Into<Opnd>, scale_b: impl Into<Opnd>, byte_a: u8, byte_b: u8) {
        assert!(byte_a < 4 && byte_b < 4, "scale byte select is 0..3");
        self.push_stmt(Stmt::VarMfmaScale16 { var, a: a.into(), b: b.into(), scale_a: scale_a.into(), scale_b: scale_b.into(), opsel: byte_a | (byte_b << 4) });
    }

    /// Update several loop variables in place: the i-th `Seq::inout` port is bound to `vars[i]`.
    #[track_caller]
    pub fn var_asm_n(&mut self, vars: &[VarId], block: amdgpu_safeasm::AsmBlock, inputs: &[Opnd]) {
        self.push_stmt(Stmt::VarAsmN { vars: vars.to_vec(), block: std::sync::Arc::new(block), inputs: inputs.to_vec() });
    }

    #[track_caller]
    pub fn load_lds_tr_b8(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsTrB8 { addr: addr.into() })
    }

    #[track_caller]
    pub fn store_lds_b128(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreLdsB128 { addr: addr.into(), value: value.into() });
    }

    /// `ds_write_b64`: a `Frag2` into LDS (8-byte aligned).
    #[track_caller]
    pub fn store_lds_b64(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>) {
        self.push_stmt(Stmt::StoreLdsB64 { addr: addr.into(), value: value.into() });
    }

    /// Broadcast lane 0 of a vector value into a scalar register.
    #[track_caller]
    pub fn read_first_lane(&mut self, v: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ReadFirstLane(v.into()))
    }

    /// `v_mov_b32_dpp quad_perm:[p0,p1,p2,p3]`: lane `l` gets lane `4 (l / 4) + perm[l % 4]` of `x` (a 32-bit value; e.g. `[1,0,3,2]` swaps
    /// neighbours, `[2,3,0,1]` swaps pairs). The result is a U32 bit pattern.
    #[track_caller]
    pub fn dpp_quad_perm(&mut self, x: impl Into<Opnd>, perm: [u8; 4]) -> ValueId {
        assert!(perm.iter().all(|p| *p < 4), "dpp_quad_perm: lane selectors are 0..3");
        self.let_(Op::DppQuadPerm { x: x.into(), perm })
    }

    /// `s_waitcnt lgkmcnt(count)`.
    #[track_caller]
    pub fn wait_lds(&mut self, count: u8) {
        self.push_stmt(Stmt::WaitLds { count });
    }

    /// `s_barrier`.
    #[track_caller]
    pub fn wait_vm(&mut self, count: u8) {
        self.push_stmt(Stmt::WaitVm { count });
    }
    /// Scheduler directive, the analogue of ``__builtin_amdgcn_sched_group_barrier(mask, size, sync)``. Consecutive directives of one basic block form the issue
    /// pipeline of that block: group 1 = the next ``size`` instructions of the ``mask`` class (in program order, among those not placed yet), then group 2, ...;
    /// everything a selected instruction depends on is pulled in just before it, every register / memory / barrier dependence is respected, and the instructions
    /// no group selects follow in program order. Honoured by the scheduler of ``CompileProfile::Gemm`` (or explicit ``sched_opts``); otherwise ignored.
    /// Mask bits as in LLVM: ``SCHED_ALU`` 0x1 (any ALU incl. MFMA), ``SCHED_VALU`` 0x2, ``SCHED_SALU`` 0x4, ``SCHED_MFMA`` 0x8, ``SCHED_VMEM`` 0x10,
    /// ``SCHED_VMEM_READ`` 0x20 (buffer/global loads, LDS-DMA included), ``SCHED_VMEM_WRITE`` 0x40, ``SCHED_DS`` 0x80, ``SCHED_DS_READ`` 0x100, ``SCHED_DS_WRITE`` 0x200.
    #[track_caller]
    pub fn sched_group_barrier(&mut self, mask: u32, size: u32, sync: u32) {
        self.push_stmt(Stmt::SchedGroup { mask, size, sync });
    }

    /// ``repeat`` times the group sequence ``groups`` (``(mask, size)`` pairs) as ``sched_group_barrier`` directives, e.g. one MFMA then two ``ds_read``s then one DMA:
    /// ``b.sched_pattern(&[(SCHED_MFMA, 1), (SCHED_DS_READ, 2), (SCHED_VMEM_READ, 1)], 64)``.
    #[track_caller]
    pub fn sched_pattern(&mut self, groups: &[(u32, u32)], repeat: u32) {
        for _ in 0..repeat {
            for &(mask, size) in groups {
                self.sched_group_barrier(mask, size, 0);
            }
        }
    }
    /// 16-byte (dwords = 4) or 4-byte (dwords = 1) global -> LDS DMA load; see `Stmt::LoadGlobalLds`.
    #[track_caller]
    pub fn load_global_lds(&mut self, base: impl Into<Opnd>, lds_addr: impl Into<Opnd>, byte_off: i32, dwords: u8) {
        self.push_stmt(Stmt::LoadGlobalLds { base: base.into(), lds_addr: lds_addr.into(), byte_off, dwords });
    }
    #[track_caller]
    pub fn set_prio(&mut self, prio: u8) {
        self.push_stmt(Stmt::SetPrio { prio });
    }
    #[track_caller]
    pub fn barrier(&mut self) {
        self.push_stmt(Stmt::Barrier);
    }
    /// Timing stamp: the low 32 bits of this wave's `s_memtime` (core-clock ticks), as a wave-uniform i32 value. Subtract two stamps (wrapping `sub`) for cycles;
    /// store them through a debug-buffer argument (`store_f32` of the value / bitcast) to read per-wave stamps on the host (`tools/stamp_host.py`, `docs/stamps.md`).
    /// Opt-in: a kernel without stamps compiles to the same instructions as before. Cost: one `s_memtime` + a full `s_waitcnt lgkmcnt(0)` + one `s_mov_b32`; put
    /// stamps at points where the kernel already drains (e.g. right after a tile's `vmcnt(0) lgkmcnt(0)` + barrier). Pinned in program order against other memory ops.
    /// Byte offset of this wave's stamp record in a debug buffer laid out `[workgroup][wave][slot]` of u32: `((wg * waves_per_wg + tid / 64) * slots) * 4`.
    /// `wg` is the linear workgroup index (for a 2-D grid pass `bx + by * grid_x`).
    #[track_caller]
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
    #[track_caller]
    pub fn stamp_store(&mut self, buf: impl Into<Opnd>, rec_off: impl Into<Opnd>, slot: u32, stamp: ValueId) {
        let p = self.ptr_add(buf, rec_off);
        let f = self.bitcast_f32(stamp);
        self.store_f32(p, slot * 4, f);
    }
    #[track_caller]
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

/// Type / range checks of the P1 memory statements (any-width stores, two-address LDS stores).
fn check_mem_stmt(f: &Function, st: &Stmt) -> Vec<String> {
    let mut e = Vec::new();
    let data = |e: &mut Vec<String>, value: Opnd, ty: MemTy, hi: bool| {
        let want = match ty {
            MemTy::B64 => Some(DType::Frag2),
            MemTy::B96 | MemTy::B128 => Some(DType::Frag4),
            _ => None,
        };
        match (want, opnd_ty(f, value)) {
            (Some(w), Ok(t)) if t != w => e.push(format!("{ty:?} store: expected {}, found {}", w.name(), t.name())),
            (None, Ok(_)) => {
                if let Err(x) = expect_dword(f, value, "store value") {
                    e.push(x);
                }
            }
            (_, Err(x)) => e.push(x),
            _ => {}
        }
        if hi && ty.bytes() > 2 {
            e.push(format!("{ty:?} store: only 8 / 16-bit stores have a d16_hi form"));
        }
    };
    match st {
        Stmt::StoreGlobalN { base, byte_off, value, ty, hi, .. } => {
            if !matches!(opnd_ty(f, *base), Ok(DType::Ptr)) {
                e.push("global store: base must be a pointer".into());
            }
            if *byte_off > 4095 {
                e.push(format!("global store: byte offset {byte_off} exceeds the 13-bit signed immediate (4095)"));
            }
            data(&mut e, *value, *ty, *hi);
        }
        Stmt::BufferStoreN { rsrc, voff, soff, imm, value, ty, hi, .. } => {
            e.extend(check_buffer_access(f, *rsrc, *voff, *soff, *imm, 1));
            data(&mut e, *value, *ty, *hi);
        }
        Stmt::StoreLdsN { addr, value, ty, hi } => {
            if let Err(x) = expect_int(f, *addr, "lds store address") {
                e.push(x);
            }
            data(&mut e, *value, *ty, *hi);
        }
        Stmt::BufferAtomicNoRet { op, rsrc, voff, soff, imm, value } => {
            if !matches!(op, AtomicOp::AddU32 | AtomicOp::AddF32 | AtomicOp::PkAddBf16 | AtomicOp::PkAddF16) {
                e.push(format!("buffer atomic {op:?}: only AddU32 / AddF32 / PkAddBf16 / PkAddF16"));
            }
            e.extend(check_buffer_access(f, *rsrc, *voff, *soff, *imm, 1));
            if let Err(x) = expect_dword(f, *value, "atomic value") {
                e.push(x);
            }
        }
        Stmt::AtomicLdsNoRet { addr, value, op } => {
            if matches!(op, AtomicOp::PkAddBf16 | AtomicOp::PkAddF16) {
                e.push(format!("lds atomic {op:?}: packed 16-bit atomics are global / buffer only"));
            }
            if let Err(x) = expect_int(f, *addr, "lds atomic address") {
                e.push(x);
            }
            if let Err(x) = expect_dword(f, *value, "atomic value") {
                e.push(x);
            }
        }
        Stmt::AtomicGlobalNoRet { base, value, byte_off, .. } => {
            if !matches!(opnd_ty(f, *base), Ok(DType::Ptr)) {
                e.push("global atomic: base must be a pointer".into());
            }
            if *byte_off > 4095 {
                e.push(format!("global atomic: byte offset {byte_off} exceeds 4095"));
            }
            if let Err(x) = expect_dword(f, *value, "atomic value") {
                e.push(x);
            }
        }
        Stmt::StoreLds2 { addr, v0, v1, b64, .. } => {
            if let Err(x) = expect_int(f, *addr, "lds store address") {
                e.push(x);
            }
            for v in [*v0, *v1] {
                data(&mut e, v, if *b64 { MemTy::B64 } else { MemTy::B32 }, false);
            }
        }
        _ => {}
    }
    e
}

/// Operand types of an MFMA of `kind` (and that lowering supports the combination).
fn check_mfma(f: &Function, kind: MmaKind, a: Opnd, b: Opnd, c: Option<Opnd>, scale: Option<&MmaScale>) -> Vec<String> {
    let mut e = Vec::new();
    if !kind.lowerable() {
        e.push(format!("mfma {kind:?}: combination not supported yet"));
        return e;
    }
    if scale.is_some() != kind.scaled {
        e.push(format!("mfma {kind:?}: scale operands {} for a {}scaled kind", if scale.is_some() { "given" } else { "missing" }, if kind.scaled { "" } else { "non-" }));
    }
    for (what, o, want) in [("A", a, kind.operand_ty(kind.a)), ("B", b, kind.operand_ty(kind.b))] {
        match opnd_ty(f, o) {
            Ok(t) if t == want => {}
            Ok(t) => e.push(format!("mfma {what}: expected {}, found {}", want.name(), t.name())),
            Err(x) => e.push(x),
        }
    }
    if let Some(c) = c {
        match opnd_ty(f, c) {
            Ok(t) if t == kind.acc_ty() => {}
            Ok(t) => e.push(format!("mfma C: expected {}, found {}", kind.acc_ty().name(), t.name())),
            Err(x) => e.push(x),
        }
    }
    e
}

/// Type / range checks of the P1 memory loads.
fn check_mem_op(f: &Function, op: &Op) -> Vec<String> {
    let mut e = Vec::new();
    match op {
        Op::LoadGlobalN { base, byte_off, .. } | Op::LoadGlobalD16 { base, byte_off, .. } => {
            if !matches!(opnd_ty(f, *base), Ok(DType::Ptr)) {
                e.push("global load: base must be a pointer".into());
            }
            if *byte_off > 4095 {
                e.push(format!("global load: byte offset {byte_off} exceeds the 13-bit signed immediate (4095)"));
            }
        }
        Op::BufferLoadN { rsrc, voff, soff, imm, .. } => e.extend(check_buffer_access(f, *rsrc, *voff, *soff, *imm, 1)),
        Op::Mfma { kind, a, b, c, scale } => e.extend(check_mfma(f, *kind, *a, *b, *c, scale.as_ref())),
        Op::AtomicLds { addr, value, op } => {
            if matches!(op, AtomicOp::PkAddBf16 | AtomicOp::PkAddF16) {
                e.push(format!("lds atomic {op:?}: packed 16-bit atomics are global / buffer only"));
            }
            if let Err(x) = expect_int(f, *addr, "lds atomic address") {
                e.push(x);
            }
            if let Err(x) = expect_dword(f, *value, "atomic value") {
                e.push(x);
            }
        }
        Op::BufferAtomicRet { op, rsrc, voff, soff, imm, value } => {
            if *op != AtomicOp::AddU32 {
                e.push(format!("returning buffer atomic {op:?}: only AddU32"));
            }
            e.extend(check_buffer_access(f, *rsrc, *voff, *soff, *imm, 1));
            if let Err(x) = expect_dword(f, *value, "atomic value") {
                e.push(x);
            }
        }
        Op::AtomicGlobal { base, value, byte_off, .. } => {
            if !matches!(opnd_ty(f, *base), Ok(DType::Ptr)) {
                e.push("global atomic: base must be a pointer".into());
            }
            if *byte_off > 4095 {
                e.push(format!("global atomic: byte offset {byte_off} exceeds 4095"));
            }
            if let Err(x) = expect_dword(f, *value, "atomic value") {
                e.push(x);
            }
        }
        Op::Dpp { x, old, dpp } | Op::DppAlu { a: x, old, dpp, .. } => {
            for s in [*x, *old] {
                if let Err(e2) = expect_dword(f, s, "dpp operand") {
                    e.push(e2);
                }
            }
            let ok = match dpp.ctrl {
                DppCtrl::RowShl(n) | DppCtrl::RowShr(n) | DppCtrl::RowRor(n) => (1..=15).contains(&n),
                DppCtrl::QuadPerm(p) => p.iter().all(|x| *x < 4),
                _ => true,
            };
            if !ok {
                e.push(format!("dpp {:?}: shift / permutation out of range", dpp.ctrl));
            }
        }
        Op::Cvt { op, srcs, sel } => {
            let (n, _, max_sel) = op.sig();
            if *sel > max_sel {
                e.push(format!("cvt {op:?}: select {sel} out of range (0..={max_sel})"));
            }
            // the fp6 / bf6 packs take whole fragments (Frag16 / Frag8) except their last (scale) operand
            let frag = |o: CvtOp, k: usize| -> Option<DType> {
                use CvtOp::*;
                match o {
                    Scale2xPk16Fp6F32 | Scale2xPk16Bf6F32 if k < 2 => Some(DType::Frag16),
                    ScalePk32Fp6F16 | ScalePk32Fp6Bf16 | ScalePk32Bf6F16 | ScalePk32Bf6Bf16 if k == 0 => Some(DType::Frag16),
                    ScalePk32F16Fp6 | ScalePk32Bf16Fp6 | ScalePk32F16Bf6 | ScalePk32Bf16Bf6 if k == 0 => Some(DType::Frag8),
                    _ => None,
                }
            };
            for (k, s) in srcs[..n].iter().enumerate() {
                match frag(*op, k) {
                    Some(want) => {
                        if !matches!(opnd_ty(f, *s), Ok(t) if t == want) {
                            e.push(format!("cvt {op:?}: operand {k} must be a {want:?}"));
                        }
                    }
                    None => {
                        if let Err(x) = expect_dword(f, *s, "cvt operand") {
                            e.push(x);
                        }
                    }
                }
            }
            // the scale operand: only its exponent is used (E8M0) - a constant that is not a power of two is certainly a mistake
            let scale_slot = match op {
                CvtOp::ScalePkBf16Fp8 | CvtOp::ScalePkF16Fp8 | CvtOp::ScalePkBf16Bf8 | CvtOp::ScalePkF16Bf8 | CvtOp::ScalePkF32Fp8 | CvtOp::ScalePkF32Bf8
                | CvtOp::ScalePkF32Fp4 | CvtOp::ScalePkBf16Fp4 | CvtOp::ScalePkF16Fp4 => Some(1),
                CvtOp::ScalePkFp8F32 | CvtOp::ScalePkBf8F32 | CvtOp::ScalePkFp4F32 | CvtOp::Scale2xPk16Fp6F32 | CvtOp::Scale2xPk16Bf6F32 => Some(2),
                CvtOp::ScalePk32Fp6F16 | CvtOp::ScalePk32Fp6Bf16 | CvtOp::ScalePk32Bf6F16 | CvtOp::ScalePk32Bf6Bf16 | CvtOp::ScalePk32F16Fp6 | CvtOp::ScalePk32Bf16Fp6
                | CvtOp::ScalePk32F16Bf6 | CvtOp::ScalePk32Bf16Bf6 => Some(1),
                _ => None,
            };
            if let Some(k) = scale_slot {
                let c = match srcs[k] {
                    Opnd::F32(x) => Some(x),
                    Opnd::Val(v) => match f.stmts().iter().find_map(|s| match s {
                        Stmt::Let { dst, op: Op::ConstF32(x) } if *dst == v => Some(*x),
                        _ => None,
                    }) {
                        Some(x) => Some(x),
                        None => None,
                    },
                    _ => None,
                };
                if let Some(x) = c {
                    let m = x.abs().to_bits() & 0x7f_ffff;
                    if m != 0 || x == 0.0 || !x.is_finite() {
                        e.push(format!("cvt {op:?}: scale {x} is not a power of two; the hardware uses only its exponent (E8M0): it would act as {}", if x.is_finite() && x != 0.0 { f32::from_bits(x.abs().to_bits() & 0x7f80_0000).to_string() } else { "garbage".to_string() }));
                    }
                }
            }
        }
        Op::LoadLdsN { addr, .. } | Op::LoadLdsD16 { addr, .. } | Op::LoadLds2 { addr, .. } | Op::LoadLdsTrB16 { addr } => {
            if let Err(x) = expect_int(f, *addr, "lds load address") {
                e.push(x);
            }
        }
        _ => {}
    }
    e
}

/// A one-dword scalar value of any type (packed containers).
fn expect_dword(f: &Function, o: Opnd, what: &str) -> Result<(), String> {
    let t = opnd_ty(f, o)?;
    if t.is_int() || t.is_float() || t == DType::U32 {
        Ok(())
    } else {
        Err(format!("{what}: expected a one-dword operand, found {}", t.name()))
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
/// Register-planning regions (`Stmt::RegionMark`): every begin has its end in the same structural block (loop / if arm / tile / stage).
fn check_region_marks(f: &Function, errs: &mut Vec<String>) {
    let mut depth = 0i64;
    let mut stack: Vec<(String, i64, usize)> = Vec::new();
    for (idx, st) in f.stmts().iter().enumerate() {
        match st {
            Stmt::BeginLoop { .. } | Stmt::BeginIf { .. } | Stmt::TileProduce { .. } | Stmt::TileConsume { .. } | Stmt::StageBegin { .. } => depth += 1,
            Stmt::EndLoop | Stmt::EndIf | Stmt::TileEnd | Stmt::StageEnd => depth -= 1,
            Stmt::RegionMark { name, begin: true } => stack.push((name.clone(), depth, idx)),
            Stmt::RegionMark { begin: false, .. } => match stack.pop() {
                Some((n, d, at)) if d != depth => errs.push(format!("stmt {idx}: region `{n}` (opened at stmt {at}) is closed in another structural block")),
                Some(_) => {}
                None => errs.push(format!("stmt {idx}: weave_region_end without an open region")),
            },
            _ => {}
        }
    }
    for (n, _, at) in stack {
        errs.push(format!("stmt {at}: region `{n}` is never closed"));
    }
}

pub fn verify(f: &Function) -> Vec<String> {
    let mut errs = Vec::new();
    check_var_slices(f, &mut errs);
    check_uniform_branches(f, &mut errs);
    check_region_marks(f, &mut errs);

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
                errs.extend(check_mem_op(f, op).into_iter().map(|e| here(e)));
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
                    Op::Un(o, a) => {
                        let r = if o.float_in() { expect_float(f, *a, "unary op") } else { expect_int(f, *a, "unary op") };
                        if let Err(e) = r {
                            errs.push(here(e));
                        }
                    }
                    Op::Bin(o, a, b) => {
                        for x in [*a, *b] {
                            let r = match o {
                                BinOp::MinF32 => expect_float(f, x, "binary op"),
                                o if o.packed() => expect_dword(f, x, "packed op"),
                                _ => expect_int(f, x, "binary op"),
                            };
                            if let Err(e) = r {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::Tri(o, a, b, c) => {
                        for x in [*a, *b, *c] {
                            let r = match o {
                                TriOp::Min3F32 | TriOp::Med3F32 => expect_float(f, x, "ternary op"),
                                TriOp::PkFmaF16 => expect_dword(f, x, "packed op"),
                                _ => expect_int(f, x, "ternary op"),
                            };
                            if let Err(e) = r {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::Cmp { pred, ty, a, b } => {
                        if matches!(pred, CmpPred::Lg | CmpPred::Uno | CmpPred::Ord) && *ty != CmpTy::F32 {
                            errs.push(here(format!("{pred:?} is a float-only predicate")));
                        }
                        for x in [*a, *b] {
                            let r = if *ty == CmpTy::F32 { expect_float(f, x, "compare") } else { expect_int(f, x, "compare") };
                            if let Err(e) = r {
                                errs.push(here(e));
                            }
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
            Stmt::VarMfma { var, kind, a, b, scale, .. } => {
                for o in stmt_operands(st) {
                    if let Opnd::Val(v) = o {
                        if !visible(&scopes, v) {
                            errs.push(here(format!("operand v{v} is not visible here")));
                        }
                    }
                }
                if !vars_defined.contains(var) {
                    errs.push(here(format!("var{var} used before its loop")));
                } else if f.var_tys.get(var.0 as usize) != Some(&kind.acc_ty()) {
                    errs.push(here(format!("var_mfma: var{var} must be {}", kind.acc_ty().name())));
                }
                errs.extend(check_mfma(f, *kind, *a, *b, None, scale.as_ref()).into_iter().map(here));
            }
            Stmt::StoreGlobalN { .. } | Stmt::BufferStoreN { .. } | Stmt::StoreLdsN { .. } | Stmt::StoreLds2 { .. } | Stmt::AtomicLdsNoRet { .. } | Stmt::AtomicGlobalNoRet { .. } | Stmt::BufferAtomicNoRet { .. } => {
                for o in stmt_operands(st) {
                    if let Opnd::Val(v) = o {
                        if !visible(&scopes, v) {
                            errs.push(here(format!("operand v{v} is not visible here")));
                        }
                    }
                }
                errs.extend(check_mem_stmt(f, st).into_iter().map(here));
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
            Stmt::WaitLds { .. } | Stmt::Barrier | Stmt::SetPrio { .. } | Stmt::SchedGroup { .. } | Stmt::WaitVm { .. } | Stmt::StageEnd | Stmt::RegionMark { .. } => {}
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
            Stmt::VarPkFma { var, start, t: tt, s } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var)) || done_vars.contains(var);
                if !ok {
                    errs.push(here(format!("var{var} updated outside its loop")));
                }
                let vw = match f.var_ty(*var) {
                    Some(DType::Frag2) => 2,
                    Some(DType::Frag4) => 4,
                    Some(DType::Frag8) => 8,
                    Some(DType::Frag16) => 16,
                    _ => 0,
                };
                let tw = match opnd_ty(f, *tt) {
                    Ok(DType::Frag2) => 2,
                    Ok(DType::Frag4) => 4,
                    Ok(DType::Frag8) => 8,
                    Ok(DType::Frag16) => 16,
                    _ => 0,
                };
                if vw == 0 || tw == 0 || *start % 2 != 0 || *start as u32 + tw > vw {
                    errs.push(here(format!("var_pk_fma: var{var} ({vw} dwords) start {start} t ({tw} dwords): t must be a fragment inside the variable at an even offset")));
                }
                if !matches!(opnd_ty(f, *s), Ok(DType::Frag2)) {
                    errs.push(here("var_pk_fma: the scale must be a Frag2 (e.g. frag2_dup(s))".to_string()));
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
pub mod facts;
pub use facts::Captured;
pub mod ir_text;
mod ir_codec;
pub mod specialize;
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
        Stmt::StoreGlobalN { base, value, .. } => vec![*base, *value],
        Stmt::BufferStoreN { rsrc, voff, soff, value, .. } => {
            let mut v = vec![*rsrc];
            v.extend(voff.iter().copied());
            v.push(*soff);
            v.push(*value);
            v
        }
        Stmt::StoreLdsN { addr, value, .. } => vec![*addr, *value],
        Stmt::StoreLds2 { addr, v0, v1, .. } => vec![*addr, *v0, *v1],
        Stmt::VarPkFma { t, s, .. } => vec![*t, *s],
        Stmt::AtomicLdsNoRet { addr, value, .. } => vec![*addr, *value],
        Stmt::AtomicGlobalNoRet { base, value, .. } => vec![*base, *value],
        Stmt::BufferAtomicNoRet { rsrc, voff, soff, value, .. } => {
            let mut v = vec![*rsrc];
            v.extend(voff.iter().copied());
            v.push(*soff);
            v.push(*value);
            v
        }
        Stmt::VarMfma { a, b, scale, .. } => {
            let mut v = vec![*a, *b];
            if let Some(s) = scale {
                v.push(s.sa);
                v.push(s.sb);
            }
            v
        }
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
        | Stmt::VarPkFma { var, .. }
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
        | Stmt::VarMfma { var, .. }
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
        if let Stmt::Let { dst, op: Op::VarSlice4 { var, idx: q } } = st {
            let ok = match f.var_ty(*var) {
                Some(DType::Frag8) => *q < 2,
                Some(DType::Frag16) => *q < 4,
                _ => false,
            };
            if !ok {
                errs.push(format!("stmt {idx}: var_slice4 needs a frag8 (idx 0..1) or frag16 (idx 0..3) variable"));
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

/// Wave-uniform values (a conservative forward analysis): constants, kernel arguments, workgroup ids, `workitem_id >> k` (k >= 6: one value
/// per 64-lane wave of a 1-D workgroup), `readfirstlane` / `readlane` / `ballot`, and pure ops / loads whose operands are all uniform (a load
/// from a uniform address returns the same value in every lane); loop variables whose initial value and every write are uniform.
pub fn uniform_values(f: &Function) -> (std::collections::HashSet<ValueId>, std::collections::HashSet<VarId>) {
    use std::collections::HashSet;
    let stmts = f.stmts();
    let mut uvars: HashSet<VarId> = (0..f.var_tys.len() as u32).map(VarId).collect();
    loop {
        let mut u: HashSet<ValueId> = HashSet::new();
        let mut wid: HashSet<ValueId> = HashSet::new();
        let mut consts: std::collections::HashMap<ValueId, i32> = std::collections::HashMap::new();
        let is_u = |o: &Opnd, u: &HashSet<ValueId>, uv: &HashSet<VarId>| match o {
            Opnd::I32(_) | Opnd::F32(_) => true,
            Opnd::Val(v) => u.contains(v),
            Opnd::Var(v) => uv.contains(v),
        };
        let mut changed = false;
        for st in stmts {
            match st {
                Stmt::Let { dst, op } => {
                    if let Op::ConstI32(k) = op {
                        consts.insert(*dst, *k);
                    }
                    let uni = match op {
                        Op::WorkitemId => {
                            wid.insert(*dst);
                            false
                        }
                        Op::LshrI32(Opnd::Val(a), b) if wid.contains(a) => match b {
                            Opnd::I32(k) => *k >= 6,
                            Opnd::Val(v) => consts.get(v).map_or(false, |k| *k >= 6),
                            _ => false,
                        },
                        Op::ReadFirstLane(_) | Op::ReadLane { .. } | Op::Ballot { .. } | Op::AnyLane(_) => true,
                        // per-lane by construction (lane permutes, matrix results, atomics' per-lane old values, asm)
                        Op::Dpp { .. } | Op::DppAlu { .. } | Op::DppQuadPerm { .. } | Op::Permlane32Swap { .. } | Op::Permlane16Swap { .. }
                        | Op::Bpermute { .. } | Op::Permute { .. } | Op::Swizzle { .. } | Op::WriteLane { .. } | Op::PartnerMaxF32(..)
                        | Op::MfmaF16 { .. } | Op::MfmaF16Wide { .. } | Op::MfmaScaleFp8_32x32x64 { .. } | Op::MfmaFp8_32x32x64 { .. }
                        | Op::SmfmacFp8_16x16x128 { .. } | Op::Mfma { .. } | Op::AtomicLds { .. } | Op::AtomicGlobal { .. } | Op::BufferAtomicRet { .. } | Op::AtomicAddU32 { .. }
                        | Op::Asm { .. } | Op::LoadLdsTrB8 { .. } | Op::LoadLdsTr4B8 { .. } | Op::LoadLdsTrB16 { .. } => false,
                        _ => op.operands().iter().all(|o| is_u(o, &u, &uvars)),
                    };
                    if uni {
                        u.insert(*dst);
                    }
                }
                Stmt::BeginLoop { vars, init } => {
                    for (v, i) in vars.iter().zip(init.iter()) {
                        if uvars.contains(v) && !is_u(i, &u, &uvars) {
                            uvars.remove(v);
                            changed = true;
                        }
                    }
                }
                Stmt::SetVar { var, value } => {
                    if uvars.contains(var) && !is_u(value, &u, &uvars) {
                        uvars.remove(var);
                        changed = true;
                    }
                }
                _ => {
                    for w in stmt_var_writes(st) {
                        if uvars.remove(&w) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            return (u, uvars);
        }
    }
}

/// `break_if` and `begin_if` branch on the WAVE (`s_cbranch_vccnz` / scalar compare): a per-lane condition would exit / skip for the whole wave
/// when ANY lane's condition holds, silently running the wrong trip count. Every branch condition must be wave-uniform.
fn check_uniform_branches(f: &Function, errs: &mut Vec<String>) {
    let (u, uv) = uniform_values(f);
    for (idx, st) in f.stmts().iter().enumerate() {
        let (cond, what) = match st {
            Stmt::BreakIf { cond } => (cond, "break_if"),
            Stmt::BeginIf { cond } => (cond, "if"),
            _ => continue,
        };
        let ok = match cond {
            Opnd::I32(_) | Opnd::F32(_) => true,
            Opnd::Val(v) => u.contains(v),
            Opnd::Var(v) => uv.contains(v),
        };
        if !ok {
            errs.push(format!(
                "stmt {idx}: {what} condition {cond:?} is not provably wave-uniform (wrap it in `any_lane(..)` if the wave-level 'any lane' reading is intended): DSL branches are wave-level (the whole wave branches when ANY \
                 lane's condition holds), so a per-lane condition silently runs the wrong trip count / skips lanes that should run. Use wave-uniform \
                 bounds (kernel arguments, workgroup ids, readfirstlane / readlane / ballot results) and predicate the per-lane work instead (select, \
                 masked buffer offsets)"
            ));
        }
    }
}
