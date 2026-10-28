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
        }
    }

    pub fn dwords(self) -> u32 {
        match self {
            DType::Ptr | DType::Frag2 => 2,
            DType::Frag4 => 4,
            DType::Frag8 => 8,
            DType::Frag16 => 16,
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

    WorkitemId,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,

    AddI32(Opnd, Opnd),
    MulI32(Opnd, Opnd),
    ShlI32(Opnd, Opnd),
    LshrI32(Opnd, Opnd),
    AndI32(Opnd, Opnd),
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
    /// Read one dword from workgroup LDS; `addr` is a byte offset.
    LoadLdsB32 { addr: Opnd },
    /// Read four dwords (16 B) from workgroup LDS.
    LoadLdsB128 { addr: Opnd },
    /// 8-bit transposing LDS read (ds_read_b64_tr_b8), two dwords per lane.
    LoadLdsTrB8 { addr: Opnd },
    /// Four `ds_read_b64_tr_b8` reads written straight into the four quarters of one `Frag8`.
    LoadLdsTr4B8 { addrs: [Opnd; 4] },
    /// Two `ds_read_b128` reads written straight into the two halves of one `Frag8`.
    LoadLdsB128x2 { lo: Opnd, hi: Opnd },
    /// Broadcast lane 0 of a vector value into a scalar register.
    ReadFirstLane(Opnd),

    /// Build a two-dword fragment from two 32-bit values.
    Frag2(Opnd, Opnd),
    /// Read dword `index` (0 or 1) of a two-dword fragment.
    Frag2Lane { frag: Opnd, index: u8 },
    /// Build a four-dword fragment from four 32-bit values.
    Frag4(Opnd, Opnd, Opnd, Opnd),
    /// Read dword `index` (0..3) of a four-dword fragment.
    Frag4Lane { frag: Opnd, index: u8 },
    /// `v_mfma_f32_16x16x16_f16` with an existing accumulator.
    MfmaF16 { a: Opnd, b: Opnd, c: Opnd },
    /// `v_mfma_f32_32x32x8_f16` with an existing accumulator.
    MfmaF16Wide { a: Opnd, b: Opnd, c: Opnd },
    /// `v_mfma_f32_16x16x32_fp8_fp8`: fp8 A/B, f32 accumulator.
    MfmaFp8_16x16x32 { a: Opnd, b: Opnd, c: Opnd },
    /// `v_mfma_f32_32x32x16_fp8_fp8`: fp8 A/B, 16-dword accumulator.
    MfmaFp8_32x32x16 { a: Opnd, b: Opnd, c: Opnd },
    /// Build an eight-dword fragment from eight 32-bit values.
    Frag8(Opnd, Opnd, Opnd, Opnd, Opnd, Opnd, Opnd, Opnd),
    /// Concatenate two four-dword fragments into an eight-dword fragment.
    CatFrag8(Opnd, Opnd),
    /// Build a sixteen-dword fragment from sixteen scalar dwords.
    Frag16Build(Vec<Opnd>),
    /// Pack two f32 values into two unsigned-normalized 16-bit values (low = a).
    CvtPknormU16(Opnd, Opnd),
    /// Scale two bf16 values and pack them into two fp8 values, replacing one
    /// 16-bit half of old (high selects the upper half).
    CvtScalef32PkFp8Bf16 { old: Opnd, src: Opnd, scale: Opnd, high: bool },
    /// `v_mfma_scale_f32_32x32x64_f8f6f4`: block-scaled fp8 MFMA. The two scale
    /// operands are E8M0 exponents in scalar registers.
    MfmaScaleFp8_32x32x64 { a: Opnd, b: Opnd, c: Opnd, scale_a: Opnd, scale_b: Opnd },
    /// Read dword `index` (0..15) of a sixteen-dword fragment.
    Frag16Lane { frag: Opnd, index: u8 },
    /// A fragment of `ty` zeroed out.
    FragZero { ty: DType },
    /// Reinterpret the bits of an integer as `f32` (register renaming only).
    BitcastF32(Opnd),
    /// Register reinterpretation to i32 (no instruction).
    BitcastI32(Opnd),
    /// Pack two f32 into two E4M3 fp8 bytes (low 16 bits of the result).
    CvtPkFp8F32(Opnd, Opnd),
    /// Two f32 -> two bf16 (round to nearest even) packed into one dword (_cvt_pk_bf16_f32).
    CvtPkBf16F32(Opnd, Opnd),
    /// Base-2 exponential: the hardware primitive (_exp_f32), no log2(e) scaling.
    Exp2F32(Opnd),
    /// Dwords `index` and `index + 1` (`index` even) of a fragment as a `Frag2` (register alias, no instruction).
    Frag16Pair { frag: Opnd, index: u8 },
    /// A `Frag2` with both dwords set to one 32-bit value.
    Frag2Dup(Opnd),
    /// `v_pk_add_f32` on two `Frag2` operands.
    PkAddF32(Opnd, Opnd),
    /// `v_pk_mul_f32` on two `Frag2` operands.
    PkMulF32(Opnd, Opnd),
    /// Four f32 -> four E4M3 bytes of one dword (a, b low half; c, d high half): two `v_cvt_pk_fp8_f32`.
    CvtPkFp8x4F32(Opnd, Opnd, Opnd, Opnd),
}

impl Op {
    pub fn result_ty(&self) -> DType {
        match self {
            Op::ConstI32(_) | Op::KernargI32 { .. } | Op::AddI32(..) | Op::MulI32(..) => DType::I32,
            Op::ShlI32(..) | Op::LshrI32(..) | Op::AndI32(..) | Op::XorI32(..) | Op::LoadU16 { .. } => DType::I32,
            Op::ConstF32(_) | Op::KernargF32 { .. } => DType::F32,
            Op::AddF32(..) | Op::SubF32(..) | Op::MulF32(..) | Op::MaxF32(..) | Op::FmaF32(..) => {
                DType::F32
            }
            Op::ExpF32(..) | Op::Exp2F32(..) | Op::RcpF32(..) | Op::CvtF16ToF32(..) | Op::LoadF32 { .. } => DType::F32,
            Op::CvtF32ToF16(..) => DType::I32,
            Op::WorkitemId | Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => DType::U32,
            Op::CmpLtI32(..) | Op::CmpGeI32(..) | Op::CmpLtF32(..) | Op::NotI1(..) => DType::I1,
            Op::Max3F32(..) => DType::F32,
            Op::Select { ty, .. } => *ty,
            Op::PtrAdd { .. } | Op::KernargPtr { .. } => DType::Ptr,
            Op::Frag2(..) => DType::Frag2,
            Op::Frag2Lane { .. } | Op::Frag4Lane { .. } => DType::U32,
            Op::Frag4(..) | Op::MfmaF16 { .. } => DType::Frag4,
            Op::MfmaF16Wide { .. } => DType::Frag16,
            Op::MfmaFp8_16x16x32 { .. } => DType::Frag4,
            Op::MfmaFp8_32x32x16 { .. } => DType::Frag16,
            Op::Frag8(..) => DType::Frag8,
            Op::CatFrag8(..) => DType::Frag8,
            Op::Frag16Build(..) => DType::Frag16,
            Op::CvtPknormU16(..) | Op::CvtScalef32PkFp8Bf16 { .. } | Op::CvtPkFp8F32(..) | Op::CvtPkBf16F32(..) => DType::U32,
            Op::MfmaScaleFp8_32x32x64 { .. } => DType::Frag16,
            Op::FragZero { ty } => *ty,
            Op::Frag16Lane { .. } => DType::U32,
            Op::Frag16Pair { .. } | Op::Frag2Dup(..) | Op::PkAddF32(..) | Op::PkMulF32(..) => DType::Frag2,
            Op::CvtPkFp8x4F32(..) => DType::U32,
            Op::BitcastF32(..) => DType::F32,
            Op::BitcastI32(..) => DType::I32,
            Op::LoadLdsB32 { .. } => DType::U32,
            Op::LoadLdsB128 { .. } => DType::Frag4,
            Op::LoadLdsTrB8 { .. } => DType::Frag2,
            Op::LoadLdsTr4B8 { .. } | Op::LoadLdsB128x2 { .. } => DType::Frag8,
            Op::LoadGlobalB128 { .. } => DType::Frag4,
            Op::ReadFirstLane(..) => DType::U32,
        }
    }

    pub fn operands(&self) -> Vec<Opnd> {
        match self {
            Op::ConstI32(_) | Op::ConstF32(_) | Op::KernargI32 { .. } | Op::KernargF32 { .. }
            | Op::KernargPtr { .. } | Op::WorkitemId | Op::WorkgroupIdX | Op::WorkgroupIdY
            | Op::WorkgroupIdZ => vec![],
            Op::AddI32(a, b)
            | Op::MulI32(a, b)
            | Op::ShlI32(a, b)
            | Op::LshrI32(a, b)
            | Op::AndI32(a, b)
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
            Op::ExpF32(a) | Op::Exp2F32(a) | Op::RcpF32(a) | Op::CvtF16ToF32(a) | Op::CvtF32ToF16(a) | Op::NotI1(a) => {
                vec![*a]
            }
            Op::Select { cond, a, b, .. } => vec![*cond, *a, *b],
            Op::LoadU16 { base, .. } | Op::LoadF32 { base, .. } => vec![*base],
            Op::Frag2(a, b) => vec![*a, *b],
            Op::Frag2Lane { frag, .. } => vec![*frag],
            Op::Frag4(a, b, c, d) => vec![*a, *b, *c, *d],
            Op::Frag4Lane { frag, .. } => vec![*frag],
            Op::MfmaF16 { a, b, c } => vec![*a, *b, *c],
            Op::MfmaF16Wide { a, b, c } => vec![*a, *b, *c],
            Op::MfmaFp8_16x16x32 { a, b, c } => vec![*a, *b, *c],
            Op::MfmaFp8_32x32x16 { a, b, c } => vec![*a, *b, *c],
            Op::Frag8(a, b, c, d, e, f, g, h) => vec![*a, *b, *c, *d, *e, *f, *g, *h],
            Op::CatFrag8(a, b) => vec![*a, *b],
            Op::Frag16Build(v) => v.clone(),
            Op::CvtPknormU16(a, b) | Op::CvtPkFp8F32(a, b) | Op::CvtPkBf16F32(a, b) => vec![*a, *b],
            Op::CvtScalef32PkFp8Bf16 { old, src, scale, .. } => vec![*old, *src, *scale],
            Op::MfmaScaleFp8_32x32x64 { a, b, c, scale_a, scale_b } => {
                vec![*a, *b, *c, *scale_a, *scale_b]
            }
            Op::Frag16Lane { frag, .. } | Op::Frag16Pair { frag, .. } => vec![*frag],
            Op::Frag2Dup(a) => vec![*a],
            Op::PkAddF32(a, b) | Op::PkMulF32(a, b) => vec![*a, *b],
            Op::CvtPkFp8x4F32(a, b, c, d) => vec![*a, *b, *c, *d],
            Op::FragZero { .. } => vec![],
            Op::BitcastF32(a) | Op::BitcastI32(a) => vec![*a],
            Op::LoadLdsB32 { addr } => vec![*addr],
            Op::LoadLdsB128 { addr } => vec![*addr],
            Op::LoadLdsTrB8 { addr } => vec![*addr],
            Op::LoadLdsTr4B8 { addrs } => addrs.to_vec(),
            Op::LoadGlobalB128 { base, .. } => vec![*base],
            Op::LoadLdsB128x2 { lo, hi } => vec![*lo, *hi],
            Op::ReadFirstLane(a) => vec![*a],
        }
    }
}

/// Statements: a flat sequence with explicit structured markers.
#[derive(Clone, Debug)]
pub enum Stmt {
    Let { dst: ValueId, op: Op },
    StoreU16 { base: Opnd, byte_off: u32, value: Opnd },
    StoreF32 { base: Opnd, byte_off: u32, value: Opnd },
    /// Write one dword into workgroup LDS; `addr` is a byte offset.
    StoreLdsB32 { addr: Opnd, value: Opnd },
    /// Write four dwords (16 B) into workgroup LDS.
    StoreLdsB128 { addr: Opnd, value: Opnd },
    /// `s_waitcnt lgkmcnt(count)`.
    WaitLds { count: u8 },
    /// `s_barrier`: synchronise every wave in the workgroup.
    Barrier,
    SetVar { var: VarId, value: Opnd },
    /// Multiply all sixteen dwords of a `Frag16` loop variable by an f32 scalar, in place.
    VarScaleF32 { var: VarId, scalar: Opnd },
    /// Multiply all sixteen dwords of a Frag16 loop variable by the (duplicated) scalar held in a Frag2, in place, with _pk_mul_f32.
    VarScalePk { var: VarId, pair: Opnd },
    /// 16-byte global load written straight into a `Frag4` loop variable.
    VarLoadB128 { var: VarId, base: Opnd, byte_off: u32 },
    /// `var = scaled_mfma_32x32x64_fp8(a, b, var)`: accumulate into a loop variable in place.
    VarMfmaScale { var: VarId, a: Opnd, b: Opnd, scale_a: Opnd, scale_b: Opnd },
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

#[derive(Clone, Debug)]
pub struct Function {
    pub name: String,
    pub args: Vec<Arg>,
    pub kernarg_size: u32,
    pub workgroup_size: u32,
    /// Bytes of workgroup LDS this kernel declares.
    pub lds_bytes: u32,
    pub stmts: Vec<Stmt>,
    pub var_tys: Vec<DType>,
    pub value_tys: Vec<DType>,
}

impl Function {
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
        for st in &self.stmts {
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
                Stmt::StoreLdsB32 { addr, value } => {
                    let _ = writeln!(s, "{pad}lds[{addr:?}] <- {value:?}");
                }
                Stmt::StoreLdsB128 { addr, value } => {
                    let _ = writeln!(s, "{pad}lds128[{addr:?}] <- {value:?}");
                }
                Stmt::WaitLds { count } => {
                    let _ = writeln!(s, "{pad}waitlds({count})");
                }
                Stmt::Barrier => {
                    let _ = writeln!(s, "{pad}barrier");
                }
                Stmt::SetVar { var, value } => {
                    let _ = writeln!(s, "{pad}var{} <- {value:?}", var.0);
                }
                Stmt::VarScaleF32 { var, scalar } => {
                    let _ = writeln!(s, "{pad}var{} *= {scalar:?}", var.0);
                }
                Stmt::VarScalePk { var, pair } => {
                    let _ = writeln!(s, "{pad}var{} *=pk {pair:?}", var.0);
                }
                Stmt::VarLoadB128 { var, base, byte_off } => {
                    let _ = writeln!(s, "{pad}var{} <- load128 [{base:?}+{byte_off}]", var.0);
                }
                Stmt::VarMfmaScale { var, a, b, scale_a, scale_b } => {
                    let _ = writeln!(s, "{pad}var{} += mfma_scale({a:?}, {b:?}; {scale_a:?}, {scale_b:?})", var.0);
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
                stmts: Vec::new(),
                var_tys: Vec::new(),
                value_tys: Vec::new(),
            },
        }
    }

    /// Declares how many bytes of workgroup LDS the kernel uses.
    pub fn lds(&mut self, bytes: u32) {
        self.f.lds_bytes = bytes;
    }

    pub fn arg(&mut self, name: &str, ty: DType, offset: u32, size: u32) {
        self.f.args.push(Arg { name: name.to_string(), ty, kernarg_offset: offset, size });
        self.f.kernarg_size = self.f.kernarg_size.max(offset + size);
    }

    pub fn let_(&mut self, op: Op) -> ValueId {
        let ty = op.result_ty();
        let dst = self.f.value_tys.len() as ValueId;
        self.f.value_tys.push(ty);
        self.f.stmts.push(Stmt::Let { dst, op });
        dst
    }

    pub fn store_u16(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::StoreU16 { base: base.into(), byte_off, value: value.into() });
    }

    pub fn store_f32(&mut self, base: impl Into<Opnd>, byte_off: u32, value: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::StoreF32 { base: base.into(), byte_off, value: value.into() });
    }

    pub fn set_var(&mut self, var: VarId, value: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::SetVar { var, value: value.into() });
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
        self.f.stmts.push(Stmt::BeginLoop { vars: vars.clone(), init });
        vars
    }

    pub fn break_if(&mut self, cond: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::BreakIf { cond: cond.into() });
    }

    pub fn end_loop(&mut self) {
        self.f.stmts.push(Stmt::EndLoop);
    }

    pub fn begin_if(&mut self, cond: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::BeginIf { cond: cond.into() });
    }

    pub fn else_(&mut self) {
        self.f.stmts.push(Stmt::Else);
    }

    pub fn end_if(&mut self) {
        self.f.stmts.push(Stmt::EndIf);
    }

    pub fn finish(self) -> Function {
        self.f
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

    pub fn fma_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>, c: impl Into<Opnd>) -> ValueId {
        self.let_(Op::FmaF32(a.into(), b.into(), c.into()))
    }

    pub fn exp_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ExpF32(a.into()))
    }

    pub fn rcp_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::RcpF32(a.into()))
    }

    pub fn cvt_f16_to_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtF16ToF32(a.into()))
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

    pub fn load_f32(&mut self, base: impl Into<Opnd>, byte_off: u32) -> ValueId {
        self.let_(Op::LoadF32 { base: base.into(), byte_off })
    }

    pub fn ptr_add(&mut self, base: impl Into<Opnd>, byte_off: impl Into<Opnd>) -> ValueId {
        self.let_(Op::PtrAdd { base: base.into(), byte_off: byte_off.into() })
    }

    pub fn frag2(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Frag2(a.into(), b.into()))
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

    pub fn mfma_fp8_16x16x32(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::MfmaFp8_16x16x32 { a: a.into(), b: b.into(), c: c.into() })
    }

    pub fn mfma_fp8_32x32x16(
        &mut self,
        a: impl Into<Opnd>,
        b: impl Into<Opnd>,
        c: impl Into<Opnd>,
    ) -> ValueId {
        self.let_(Op::MfmaFp8_32x32x16 { a: a.into(), b: b.into(), c: c.into() })
    }

    #[allow(clippy::too_many_arguments)]
    /// Concatenate two Frag4 values into a Frag8 (low half first).
    /// Pack two f32 into two unsigned-normalized 16-bit values.
    pub fn cvt_pknorm_u16_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPknormU16(a.into(), b.into()))
    }

    /// Scale two bf16 and pack into fp8, writing one 16-bit half of old.
    #[allow(clippy::too_many_arguments)]
    pub fn cvt_scalef32_pk_fp8_bf16(
        &mut self,
        old: impl Into<Opnd>,
        src: impl Into<Opnd>,
        scale: impl Into<Opnd>,
        high: bool,
    ) -> ValueId {
        self.let_(Op::CvtScalef32PkFp8Bf16 {
            old: old.into(),
            src: src.into(),
            scale: scale.into(),
            high,
        })
    }

    /// Pack two f32 into two E4M3 fp8 bytes (upper 16 bits are zero).
    pub fn exp2_f32(&mut self, a: impl Into<Opnd>) -> ValueId {
        self.let_(Op::Exp2F32(a.into()))
    }

    pub fn cvt_pk_bf16_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPkBf16F32(a.into(), b.into()))
    }

    pub fn cvt_pk_fp8_f32(&mut self, a: impl Into<Opnd>, b: impl Into<Opnd>) -> ValueId {
        self.let_(Op::CvtPkFp8F32(a.into(), b.into()))
    }

    /// Build a Frag16 from sixteen scalar values (raw dwords).
    pub fn frag16_build(&mut self, lanes: &[ValueId]) -> ValueId {
        assert_eq!(lanes.len(), 16);
        self.let_(Op::Frag16Build(lanes.iter().map(|v| Opnd::Val(*v)).collect()))
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
        self.f.stmts.push(Stmt::StoreLdsB32 { addr: addr.into(), value: value.into() });
    }

    /// Four dwords (16 B) read from workgroup LDS.
    pub fn load_lds_b128(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsB128 { addr: addr.into() })
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

    /// Load 16 bytes from global memory directly into a `Frag4` loop variable (the load stays in
    /// flight across the code that follows; the loop back edge waits for it).
    pub fn var_load_b128(&mut self, var: VarId, base: impl Into<Opnd>, byte_off: u32) {
        self.f.stmts.push(Stmt::VarLoadB128 { var, base: base.into(), byte_off });
    }

    /// `var *= scalar` on every dword of a `Frag16` loop variable.
    pub fn frag16_pair(&mut self, frag: impl Into<Opnd>, index: u8) -> ValueId {
        self.let_(Op::Frag16Pair { frag: frag.into(), index })
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
    pub fn var_scale_pk(&mut self, var: VarId, pair: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::VarScalePk { var, pair: pair.into() });
    }
    pub fn var_scale_f32(&mut self, var: VarId, scalar: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::VarScaleF32 { var, scalar: scalar.into() });
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
        self.f.stmts.push(Stmt::VarMfmaScale {
            var,
            a: a.into(),
            b: b.into(),
            scale_a: scale_a.into(),
            scale_b: scale_b.into(),
        });
    }

    pub fn load_lds_tr_b8(&mut self, addr: impl Into<Opnd>) -> ValueId {
        self.let_(Op::LoadLdsTrB8 { addr: addr.into() })
    }

    pub fn store_lds_b128(&mut self, addr: impl Into<Opnd>, value: impl Into<Opnd>) {
        self.f.stmts.push(Stmt::StoreLdsB128 { addr: addr.into(), value: value.into() });
    }

    /// Broadcast lane 0 of a vector value into a scalar register.
    pub fn read_first_lane(&mut self, v: impl Into<Opnd>) -> ValueId {
        self.let_(Op::ReadFirstLane(v.into()))
    }

    /// `s_waitcnt lgkmcnt(count)`.
    pub fn wait_lds(&mut self, count: u8) {
        self.f.stmts.push(Stmt::WaitLds { count });
    }

    /// `s_barrier`.
    pub fn barrier(&mut self) {
        self.f.stmts.push(Stmt::Barrier);
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

    // Region stack: values visible in the current scope.
    let mut scopes: Vec<HashSet<ValueId>> = vec![HashSet::new()];
    let mut loop_depth = 0usize;
    let mut if_depth = 0usize;
    let mut loop_vars: Vec<Vec<VarId>> = Vec::new();
    // Vars stay readable after their loop ends: the register holds the final
    // value carried by the last iteration.
    let mut vars_defined: HashSet<VarId> = HashSet::new();

    let visible = |scopes: &Vec<HashSet<ValueId>>, v: ValueId| {
        scopes.iter().any(|s| s.contains(&v))
    };

    for (idx, st) in f.stmts.iter().enumerate() {
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
                    Op::AddI32(a, b)
                    | Op::MulI32(a, b)
                    | Op::ShlI32(a, b)
                    | Op::LshrI32(a, b)
                    | Op::AndI32(a, b)
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
                    Op::ExpF32(a) | Op::Exp2F32(a) | Op::RcpF32(a) => {
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
                    Op::LoadU16 { base, .. } | Op::LoadF32 { base, .. } => {
                        if let Err(e) = expect_ptr(f, *base, "load") {
                            errs.push(here(e));
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
                    Op::CvtPkFp8F32(a, b) => {
                        for o in [a, b] {
                            if let Err(e) = expect_float(f, *o, "cvt_pk_fp8") {
                                errs.push(here(e));
                            }
                        }
                    }
                    Op::CvtScalef32PkFp8Bf16 { old, src, scale, .. } => {
                        if let Err(e) = expect_int(f, *old, "pack fp8 destination") {
                            errs.push(here(e));
                        }
                        if let Err(e) = expect_int(f, *src, "pack fp8 source") {
                            errs.push(here(e));
                        }
                        if let Err(e) = expect_float(f, *scale, "pack fp8 scale") {
                            errs.push(here(e));
                        }
                    }
                    Op::Frag16Build(v) => {
                        if v.len() != 16 {
                            errs.push(here("frag16 needs sixteen lanes".to_string()));
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
                    Op::MfmaFp8_16x16x32 { a, b, c } => {
                        for (o, want, what) in [
                            (*a, DType::Frag2, "a"),
                            (*b, DType::Frag2, "b"),
                            (*c, DType::Frag4, "c"),
                        ] {
                            if let Ok(t) = opnd_ty(f, o) {
                                if t != want {
                                    errs.push(here(format!(
                                        "fp8 mfma: operand {what} is {}, expected {}",
                                        t.name(),
                                        want.name()
                                    )));
                                }
                            }
                        }
                    }
                    Op::MfmaFp8_32x32x16 { a, b, c } => {
                        for (o, want, what) in [
                            (*a, DType::Frag2, "a"),
                            (*b, DType::Frag2, "b"),
                            (*c, DType::Frag16, "c"),
                        ] {
                            if let Ok(t) = opnd_ty(f, o) {
                                if t != want {
                                    errs.push(here(format!(
                                        "wide fp8 mfma: operand {what} is {}, expected {}",
                                        t.name(),
                                        want.name()
                                    )));
                                }
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
            Stmt::WaitLds { .. } | Stmt::Barrier => {}
            Stmt::SetVar { var, value } => {
                // Any enclosing loop's variable may be updated; an inner loop can
                // therefore update an outer loop's accumulator.
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var));
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
            Stmt::VarLoadB128 { var, base, .. } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var));
                if !ok {
                    errs.push(here(format!("var{var} loaded outside its loop")));
                }
                if f.var_ty(*var) != Some(DType::Frag4) {
                    errs.push(here(format!("var{var} must be a frag4 to take a 16-byte load")));
                }
                if let Opnd::Val(v) = base {
                    if !visible(&scopes, *v) {
                        errs.push(here(format!("operand v{v} is not visible here")));
                    }
                }
            }
            Stmt::VarScaleF32 { var, scalar } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var));
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
            Stmt::VarScalePk { var, pair } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var));
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
            Stmt::VarMfmaScale { var, a, b, scale_a, scale_b } => {
                let ok = loop_vars.iter().rev().any(|vs| vs.contains(var));
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
                    loop_vars.pop();
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
pub mod kernels;
