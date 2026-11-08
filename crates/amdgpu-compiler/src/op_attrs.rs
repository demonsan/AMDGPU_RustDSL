//! Per-opcode attribute table: the SINGLE source of truth for what a machine op is (docs/op-attributes.md).
//!
//! Every pass that needs a property of an opcode class reads it here instead of enumerating opcode names: the scheduler's cost
//! model and effects (memory space / direction / base operand / size, `smem`, fences, `vcc` / `scc` reads and writes, sub-register
//! and partial writes), the value- and physical-level hazard passes (MFMA latency / in-place chains / scaled / zero-C, transcendental,
//! VALU-writer delay, readers of VALU hazards, movable fill candidates, wide-store data operand), AGPR legality, the `M0` source of
//! LDS-DMA forms, dead-code elimination, the backend policy (LLVM-tail row vs own-only) and the emitted mnemonic.
//! (TIRx: operators carry attributes; passes look them up.)
//!
//! `attrs_of` is an exhaustive `match` (no wildcard): a new `MOp` does not compile until it has a row. Identity checks for one
//! specific instruction (`op == MOp::SNop`, the builders in `lower.rs`) are not classes and stay where they are.

use crate::mir::MOp;
use crate::sched::effects::Space;
use crate::sched::model::{c, co, Grade, OpCost, Pipe};

/// Fence class of an instruction (scheduler ordering).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceAttr {
    None,
    Nop,
    /// `s_waitcnt vmcnt(imm)`
    WaitVm,
    /// `s_waitcnt lgkmcnt(imm)`
    WaitLgkm,
    Barrier,
    SetPrio,
}

/// How the register operands map to read / written dword slots beyond "sources read, destination written".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegFx {
    Plain,
    /// writes only the dword `imm` of the destination (fragment lane write)
    SubWrite,
    /// reads only the dword `imm` of source 0 (fragment lane read)
    SubRead,
    /// 16-bit half writer: the other half survives, so the destination is also read
    PartialWrite,
    /// `v_permlane32_swap`: sources 0 and 1 are both written
    SwapSrc01,
}

/// Base address operand of a memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Base {
    /// the first dword slot of source `k`
    Src(usize),
    /// not an affine function of one register (MUBUF): always may-alias
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemAttr {
    pub space: Space,
    pub write: bool,
    pub base: Base,
    /// bytes
    pub size: i64,
}

/// Matrix-core properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MfmaAttr {
    /// independent wait states before the destination may be read (CDNA4 ISA 7.6)
    pub latency: usize,
    /// a dependent in-place accumulation of the same opcode is interlocked by the hardware
    pub chain: bool,
    /// the 32x32x64 f8f6f4 family and the scaled 16x16x128 (reader of VALU hazards; hardware-confirmed +1 wait state)
    pub scaled: bool,
    /// inline-zero accumulator (no C operand)
    pub zero_c: bool,
    /// sparse `v_smfmac` (A/B/SRC2/C operand rules differ)
    pub sparse: bool,
}

/// Candidate for the hazard pass's in-block fill scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Movable {
    No,
    Yes,
}

/// Dead-code elimination class (`lower::dce_full`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dce {
    Keep,
    /// a load: removable when its result is unused (also in the loads-only mode)
    Load,
    /// pure ALU: removable when its result (and the `vcc` it writes) is unused
    Pure,
}

/// Backend policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// has rows in `llvm20.tbl` and `llvm24.tbl`
    Llvm,
    /// the LLVM tail cannot express it: a kernel using it is compiled by the own tail (`uses_own_only_ops`)
    OwnOnly,
    /// no LLVM rows and not (yet) routed to the own tail: only the own backend can emit it (documented gap)
    NoRow,
}

/// Which 16-bit half a partial writer writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    Lo,
    Hi,
}

#[derive(Clone, Copy, Debug)]
pub struct OpAttrs {
    /// primary assembly mnemonic (pseudo ops: the expansion's first instruction)
    pub mnemonic: &'static str,
    pub cost: OpCost,
    pub fence: FenceAttr,
    /// fence-only instruction: no register effects at all (effects return before the operands)
    pub fence_only: bool,
    pub mem: Option<MemAttr>,
    /// scalar memory load (lgkmcnt, out of order)
    pub smem: bool,
    pub regfx: RegFx,
    /// effects: implicit `vcc` read / write (includes the carry of the 64-bit pointer add)
    pub vcc_read: bool,
    pub vcc_write: bool,
    /// `MOp::defines_vcc` / `MOp::consumes_vcc`: the explicit vcc producers / consumers that must stay adjacent (excludes the pointer-add pseudo)
    pub defines_vcc: bool,
    pub consumes_vcc: bool,
    pub scc_read: bool,
    pub scc_write: bool,
    /// source index holding the `M0` value (LDS-DMA forms)
    pub m0_src: Option<usize>,
    pub mfma: Option<MfmaAttr>,
    /// `v_exp` / `v_log` / `v_rcp`: result latency (TRANS_LATENCY)
    pub transcendental: bool,
    pub movable: Movable,
    /// wait states a VALU result needs before an MFMA operand / permlane / readfirstlane / DPP read (None: not such a writer)
    pub valu_writer_delay: Option<usize>,
    /// reads operands subject to the VALU-write hazard
    pub reads_valu_hazard: bool,
    /// wide (> 8 bytes) VMEM store: index of the data source (2 wait states before a VALU may overwrite it)
    pub wide_store_data: Option<usize>,
    /// the destination may be an AGPR
    pub agpr_dst: bool,
    /// bit k: source k may be an AGPR
    pub agpr_srcs: u8,
    /// 16-bit half writer
    pub half: Option<Half>,
    pub dce: Dce,
    pub backend: Backend,
}

const fn a(mnemonic: &'static str, cost: OpCost) -> OpAttrs {
    OpAttrs {
        mnemonic,
        cost,
        fence: FenceAttr::None,
        fence_only: false,
        mem: None,
        smem: false,
        regfx: RegFx::Plain,
        vcc_read: false,
        vcc_write: false,
        defines_vcc: false,
        consumes_vcc: false,
        scc_read: false,
        scc_write: false,
        m0_src: None,
        mfma: None,
        transcendental: false,
        movable: Movable::No,
        valu_writer_delay: None,
        reads_valu_hazard: false,
        wide_store_data: None,
        agpr_dst: false,
        agpr_srcs: 0,
        half: None,
        dce: Dce::Keep,
        backend: Backend::Llvm,
    }
}

impl OpAttrs {
    const fn fence(self, f: FenceAttr) -> Self {
        OpAttrs { fence: f, ..self }
    }
    const fn fence_only(self, f: FenceAttr) -> Self {
        OpAttrs { fence: f, fence_only: true, ..self }
    }
    const fn mem(self, space: Space, write: bool, base: Base, size: i64) -> Self {
        OpAttrs { mem: Some(MemAttr { space, write, base, size }), ..self }
    }
    const fn smem(self) -> Self {
        OpAttrs { smem: true, ..self }
    }
    const fn regfx(self, r: RegFx) -> Self {
        OpAttrs { regfx: r, ..self }
    }
    const fn vcc(self, read: bool, write: bool) -> Self {
        OpAttrs { vcc_read: read, vcc_write: write, ..self }
    }
    const fn def_vcc(self) -> Self {
        OpAttrs { defines_vcc: true, ..self }
    }
    const fn use_vcc(self) -> Self {
        OpAttrs { consumes_vcc: true, ..self }
    }
    const fn scc(self, read: bool, write: bool) -> Self {
        OpAttrs { scc_read: read, scc_write: write, ..self }
    }
    const fn m0(self, k: usize) -> Self {
        OpAttrs { m0_src: Some(k), ..self }
    }
    const fn mfma(self, latency: usize, chain: bool, scaled: bool, zero_c: bool, sparse: bool) -> Self {
        OpAttrs { mfma: Some(MfmaAttr { latency, chain, scaled, zero_c, sparse }), reads_valu_hazard: true, ..self }
    }
    const fn trans(self) -> Self {
        OpAttrs { transcendental: true, ..self }
    }
    const fn movable(self, m: Movable) -> Self {
        OpAttrs { movable: m, ..self }
    }
    const fn wdelay(self, n: usize) -> Self {
        OpAttrs { valu_writer_delay: Some(n), ..self }
    }
    const fn hz_reader(self) -> Self {
        OpAttrs { reads_valu_hazard: true, ..self }
    }
    const fn wide_store(self, k: usize) -> Self {
        OpAttrs { wide_store_data: Some(k), ..self }
    }
    const fn agpr(self, dst: bool, srcs: u8) -> Self {
        OpAttrs { agpr_dst: dst, agpr_srcs: srcs, ..self }
    }
    const fn half(self, h: Half) -> Self {
        OpAttrs { half: Some(h), ..self }
    }
    const fn dce(self, d: Dce) -> Self {
        OpAttrs { dce: d, ..self }
    }
    const fn backend(self, b: Backend) -> Self {
        OpAttrs { backend: b, ..self }
    }
    /// A matrix op of the AGPR-capable family (every MFMA except the sparse `v_smfmac`).
    pub fn is_mfma(&self) -> bool {
        matches!(self.mfma, Some(m) if !m.sparse)
    }
    /// Source `k` may be an AGPR.
    pub fn agpr_src(&self, k: usize) -> bool {
        k < 8 && self.agpr_srcs >> k & 1 == 1
    }
    /// An LDS read (hoisting candidate).
    pub fn is_lds_read(&self) -> bool {
        matches!(self.mem, Some(m) if m.space as u8 == Space::Lds as u8 && !m.write)
    }
}

/// Every `MOp` variant in declaration order (`tests::every_variant_is_listed` checks it against the enum).
pub const ALL: &[MOp] = &[
    MOp::VMovB32,
    MOp::VMovFromS,
    MOp::VMovFromSPair,
    MOp::VPtrAdd,
    MOp::VReadPhys,
    MOp::VReadFirstLane,
    MOp::VMovDpp,
    MOp::VCvtPknormU16,
    MOp::VCvtPkFp8F32,
    MOp::VCvtPkBf16F32,
    MOp::VCvtPkFp8Lo,
    MOp::VCvtPkFp8Hi,
    MOp::VCvtPkFp8F32Hi,
    MOp::VPkAddF32,
    MOp::VPermlane32Swap,
    MOp::VPkMulF32,
    MOp::VAddU32,
    MOp::VAddCoU32,
    MOp::VAddcCoU32,
    MOp::VAddF32,
    MOp::VSubF32,
    MOp::VMulF32,
    MOp::VMaxF32,
    MOp::VMax3F32,
    MOp::VPermB32,
    MOp::VPkSubU16,
    MOp::VFmaF32,
    MOp::VExpF32,
    MOp::VLogF32,
    MOp::VRcpF32,
    MOp::VCvtF32F16,
    MOp::VCvtF16F32,
    MOp::VAndB32,
    MOp::VMinI32,
    MOp::VPkFmaF32,
    MOp::SMulHiU32,
    MOp::VMadU64U32,
    MOp::VReadLane,
    MOp::VWriteLane,
    MOp::VMbcntLo,
    MOp::VMbcntHi,
    MOp::VBcntU32,
    MOp::VFfblB32,
    MOp::VFfbhU32,
    MOp::VCmpMaskNeU32,
    MOp::SMaskLo,
    MOp::SMaskHi,
    MOp::LdsBpermute,
    MOp::LdsPermute,
    MOp::LdsSwizzle,
    MOp::VMovDppX,
    MOp::VAddF32Dpp,
    MOp::VSubF32Dpp,
    MOp::VMulF32Dpp,
    MOp::VMaxF32Dpp,
    MOp::VMinF32Dpp,
    MOp::VAddU32Dpp,
    MOp::VMaxI32Dpp,
    MOp::VMinI32Dpp,
    MOp::VMaxU32Dpp,
    MOp::VMinU32Dpp,
    MOp::VAndB32Dpp,
    MOp::VOrB32Dpp,
    MOp::VXorB32Dpp,
    MOp::VPermlane16Swap,
    MOp::VCvtPkF32Fp8,
    MOp::VCvtF32Fp8,
    MOp::VCvtPkF32Bf8,
    MOp::VCvtF32Bf8,
    MOp::VCvtPkBf8F32,
    MOp::VCvtScalePkBf16Fp8,
    MOp::VCvtScalePkF16Fp8,
    MOp::VCvtScalePkBf16Bf8,
    MOp::VCvtScalePkF16Bf8,
    MOp::VCvtScalePkF32Fp8,
    MOp::VCvtScalePkF32Bf8,
    MOp::VCvtScalePkF32Fp4,
    MOp::VCvtScale2xPk16Fp6F32,
    MOp::VCvtScale2xPk16Bf6F32,
    MOp::VCvtScalePk32Fp6F16,
    MOp::VCvtScalePk32Fp6Bf16,
    MOp::VCvtScalePk32Bf6F16,
    MOp::VCvtScalePk32Bf6Bf16,
    MOp::VCvtScalePk32F16Fp6,
    MOp::VCvtScalePk32Bf16Fp6,
    MOp::VCvtScalePk32F16Bf6,
    MOp::VCvtScalePk32Bf16Bf6,
    MOp::VCvtScalePkBf16Fp4,
    MOp::VCvtScalePkF16Fp4,
    MOp::VCvtScalePkFp8F32,
    MOp::VCvtScalePkBf8F32,
    MOp::VCvtScalePkFp4F32,
    MOp::VCvtPkU8F32,
    MOp::VCvtPkI16I32,
    MOp::VCvtPkU16U32,
    MOp::VFloorF32,
    MOp::VCeilF32,
    MOp::VTruncF32,
    MOp::VRndneF32,
    MOp::VFractF32,
    MOp::VRsqF32,
    MOp::VSqrtF32,
    MOp::VSinF32,
    MOp::VCosF32,
    MOp::VCvtI32F32,
    MOp::VCvtU32F32,
    MOp::VCvtF32I32,
    MOp::VCvtF32U32,
    MOp::VCvtF32Ubyte0,
    MOp::VCvtF32Ubyte1,
    MOp::VCvtF32Ubyte2,
    MOp::VCvtF32Ubyte3,
    MOp::VMinF32,
    MOp::VMaxI32,
    MOp::VMaxU32,
    MOp::VMinU32,
    MOp::VSubU32,
    MOp::VMulU32U24,
    MOp::VAshrI32,
    MOp::VMulHiU32,
    MOp::VMulHiI32,
    MOp::VMin3F32,
    MOp::VMed3F32,
    MOp::VMin3I32,
    MOp::VMin3U32,
    MOp::VMax3I32,
    MOp::VMax3U32,
    MOp::VMed3I32,
    MOp::VMed3U32,
    MOp::VMadU32U24,
    MOp::VLshlAddU32,
    MOp::VAdd3U32,
    MOp::VBfeU32,
    MOp::VBfeI32,
    MOp::VBfiB32,
    MOp::VPkAddF16,
    MOp::VPkMulF16,
    MOp::VPkMaxF16,
    MOp::VPkMinF16,
    MOp::VPkFmaF16,
    MOp::VCmpLeF32,
    MOp::VCmpGtF32,
    MOp::VCmpGeF32,
    MOp::VCmpEqF32,
    MOp::VCmpNeF32,
    MOp::VCmpLgF32,
    MOp::VCmpUF32,
    MOp::VCmpOF32,
    MOp::VCmpLeI32,
    MOp::VCmpEqI32,
    MOp::VCmpLtU32,
    MOp::VCmpLeU32,
    MOp::VCmpGtU32,
    MOp::VCmpGeU32,
    MOp::VCmpEqU32,
    MOp::VOrB32,
    MOp::VXorB32,
    MOp::VLshrB32,
    MOp::VLshlB32,
    MOp::VMulLoU32,
    MOp::VCmpLtI32,
    MOp::VCmpGeI32,
    MOp::VCmpGtI32,
    MOp::VCmpMaskLtI32,
    MOp::VCndmaskM,
    MOp::VCmpNeI32,
    MOp::VCmpLtF32,
    MOp::VCndmask,
    MOp::VWriteSub,
    MOp::VReadSub,
    MOp::VMfma16x16x16F16,
    MOp::VMfma32x32x8F16,
    MOp::VMfma32x32x8F16Z,
    MOp::VMfma16x16x32Fp8,
    MOp::VMfma32x32x16Fp8,
    MOp::VMfmaScale32x32x64Fp8,
    MOp::VMfmaScale16x16x128Fp8,
    MOp::VMfmaScale32x32x64Fp8Z,
    MOp::VMfma32x32x64Fp8,
    MOp::VMfma32x32x64Fp8Z,
    MOp::VSmfmac16x16x128Fp8,
    MOp::VMfmaPlain16x16x128A4B4,
    MOp::VMfmaPlain16x16x128A4B4Z,
    MOp::VMfmaPlain16x16x128A4B6,
    MOp::VMfmaPlain16x16x128A4B6Z,
    MOp::VMfmaPlain16x16x128A4B8,
    MOp::VMfmaPlain16x16x128A4B8Z,
    MOp::VMfmaPlain16x16x128A6B4,
    MOp::VMfmaPlain16x16x128A6B4Z,
    MOp::VMfmaPlain16x16x128A6B6,
    MOp::VMfmaPlain16x16x128A6B6Z,
    MOp::VMfmaPlain16x16x128A6B8,
    MOp::VMfmaPlain16x16x128A6B8Z,
    MOp::VMfmaPlain16x16x128A8B4,
    MOp::VMfmaPlain16x16x128A8B4Z,
    MOp::VMfmaPlain16x16x128A8B6,
    MOp::VMfmaPlain16x16x128A8B6Z,
    MOp::VMfmaPlain16x16x128A8B8,
    MOp::VMfmaPlain16x16x128A8B8Z,
    MOp::VMfmaPlain32x32x64A4B4,
    MOp::VMfmaPlain32x32x64A4B4Z,
    MOp::VMfmaPlain32x32x64A4B6,
    MOp::VMfmaPlain32x32x64A4B6Z,
    MOp::VMfmaPlain32x32x64A4B8,
    MOp::VMfmaPlain32x32x64A4B8Z,
    MOp::VMfmaPlain32x32x64A6B4,
    MOp::VMfmaPlain32x32x64A6B4Z,
    MOp::VMfmaPlain32x32x64A6B6,
    MOp::VMfmaPlain32x32x64A6B6Z,
    MOp::VMfmaPlain32x32x64A6B8,
    MOp::VMfmaPlain32x32x64A6B8Z,
    MOp::VMfmaPlain32x32x64A8B4,
    MOp::VMfmaPlain32x32x64A8B4Z,
    MOp::VMfmaPlain32x32x64A8B6,
    MOp::VMfmaPlain32x32x64A8B6Z,
    MOp::VMfmaPlain32x32x64A8B8,
    MOp::VMfmaPlain32x32x64A8B8Z,
    MOp::VMfmaScale16x16x128A4B4,
    MOp::VMfmaScale16x16x128A4B4Z,
    MOp::VMfmaScale16x16x128A4B6,
    MOp::VMfmaScale16x16x128A4B6Z,
    MOp::VMfmaScale16x16x128A4B8,
    MOp::VMfmaScale16x16x128A4B8Z,
    MOp::VMfmaScale16x16x128A6B4,
    MOp::VMfmaScale16x16x128A6B4Z,
    MOp::VMfmaScale16x16x128A6B6,
    MOp::VMfmaScale16x16x128A6B6Z,
    MOp::VMfmaScale16x16x128A6B8,
    MOp::VMfmaScale16x16x128A6B8Z,
    MOp::VMfmaScale16x16x128A8B4,
    MOp::VMfmaScale16x16x128A8B4Z,
    MOp::VMfmaScale16x16x128A8B6,
    MOp::VMfmaScale16x16x128A8B6Z,
    MOp::VMfmaScale16x16x128A8B8,
    MOp::VMfmaScale16x16x128A8B8Z,
    MOp::VMfmaScale32x32x64A4B4,
    MOp::VMfmaScale32x32x64A4B4Z,
    MOp::VMfmaScale32x32x64A4B6,
    MOp::VMfmaScale32x32x64A4B6Z,
    MOp::VMfmaScale32x32x64A4B8,
    MOp::VMfmaScale32x32x64A4B8Z,
    MOp::VMfmaScale32x32x64A6B4,
    MOp::VMfmaScale32x32x64A6B4Z,
    MOp::VMfmaScale32x32x64A6B6,
    MOp::VMfmaScale32x32x64A6B6Z,
    MOp::VMfmaScale32x32x64A6B8,
    MOp::VMfmaScale32x32x64A6B8Z,
    MOp::VMfmaScale32x32x64A8B4,
    MOp::VMfmaScale32x32x64A8B4Z,
    MOp::VMfmaScale32x32x64A8B6,
    MOp::VMfmaScale32x32x64A8B6Z,
    MOp::VMfmaScale32x32x64A8B8,
    MOp::VMfmaScale32x32x64A8B8Z,
    MOp::VMfma16x16x32Bf16,
    MOp::VMfma16x16x32Bf16Z,
    MOp::VMfma32x32x16Bf16,
    MOp::VMfma32x32x16Bf16Z,
    MOp::VMfma16x16x32F16,
    MOp::VMfma16x16x32F16Z,
    MOp::VMfma32x32x16F16,
    MOp::VMfma32x32x16F16Z,
    MOp::VMfma16x16x64I8,
    MOp::VMfma16x16x64I8Z,
    MOp::VMfma32x32x32I8,
    MOp::VMfma32x32x32I8Z,
    MOp::VAccvgprRead,
    MOp::VAccvgprWrite,
    MOp::VAccvgprMov,
    MOp::LdsWriteB32,
    MOp::LdsReadB32,
    MOp::LdsWriteB128,
    MOp::LdsReadB128,
    MOp::LdsReadB64,
    MOp::LdsWriteB64,
    MOp::LdsReadB64TrB8,
    MOp::SBarrier,
    MOp::SSetPrio,
    MOp::SchedGroup,
    MOp::GlobalLoadU16,
    MOp::GlobalLoadF32,
    MOp::GlobalLoadB128,
    MOp::GlobalLoadLdsB128,
    MOp::GlobalLoadLdsB32,
    MOp::GlobalLoadLdsB128S,
    MOp::GlobalLoadLdsB32S,
    MOp::BufferLoadDwordX1,
    MOp::BufferLoadDwordX2,
    MOp::BufferLoadDwordX4,
    MOp::BufferStoreDwordX1,
    MOp::BufferStoreDwordX2,
    MOp::BufferStoreDwordX4,
    MOp::BufferLoadDwordX4Lds,
    MOp::BufferLoadDwordX1Lds,
    MOp::SBufferRsrc,
    MOp::SBufferRsrcDyn,
    MOp::GlobalStoreU16,
    MOp::GlobalStoreF32,
    MOp::GlobalStoreF32x4,
    MOp::GlobalStoreF32x2,
    MOp::GlobalLoadU8,
    MOp::GlobalLoadI8,
    MOp::GlobalLoadI16,
    MOp::GlobalLoadB64,
    MOp::GlobalLoadB96,
    MOp::GlobalLoadD16U8,
    MOp::GlobalLoadD16U8Hi,
    MOp::GlobalLoadD16I8,
    MOp::GlobalLoadD16I8Hi,
    MOp::GlobalLoadD16B16,
    MOp::GlobalLoadD16B16Hi,
    MOp::GlobalStoreB8,
    MOp::GlobalStoreB8Hi,
    MOp::GlobalStoreB16Hi,
    MOp::GlobalStoreB96,
    MOp::BufferLoadU8,
    MOp::BufferLoadI8,
    MOp::BufferLoadU16,
    MOp::BufferLoadI16,
    MOp::BufferLoadB96,
    MOp::BufferStoreB8,
    MOp::BufferStoreB16,
    MOp::BufferStoreB8Hi,
    MOp::BufferStoreB16Hi,
    MOp::BufferStoreB96,
    MOp::LdsReadU8,
    MOp::LdsReadI8,
    MOp::LdsReadU16,
    MOp::LdsReadI16,
    MOp::LdsReadB96,
    MOp::LdsReadB64TrB16,
    MOp::LdsReadD16U8,
    MOp::LdsReadD16U8Hi,
    MOp::LdsReadD16I8,
    MOp::LdsReadD16I8Hi,
    MOp::LdsReadD16B16,
    MOp::LdsReadD16B16Hi,
    MOp::LdsWriteB8,
    MOp::LdsWriteB16,
    MOp::LdsWriteB8Hi,
    MOp::LdsWriteB16Hi,
    MOp::LdsWriteB96,
    MOp::LdsRead2B32,
    MOp::LdsRead2B64,
    MOp::LdsRead2St64B32,
    MOp::LdsRead2St64B64,
    MOp::LdsWrite2B32,
    MOp::LdsWrite2B64,
    MOp::LdsWrite2St64B32,
    MOp::LdsWrite2St64B64,
    MOp::GlobalLoadF32Sc1,
    MOp::GlobalStoreF32Sc1,
    MOp::GlobalAtomicAddRet,
    MOp::BufferAtomicAddU32,
    MOp::BufferAtomicAddF32,
    MOp::BufferAtomicPkAddBf16,
    MOp::BufferAtomicPkAddF16,
    MOp::GlobalAtomicPkAddBf16,
    MOp::GlobalAtomicPkAddBf16Rtn,
    MOp::GlobalAtomicPkAddF16,
    MOp::GlobalAtomicPkAddF16Rtn,
    MOp::LdsAtomicAddU32,
    MOp::LdsAtomicAddU32Rtn,
    MOp::LdsAtomicMaxU32,
    MOp::LdsAtomicMaxU32Rtn,
    MOp::LdsAtomicMinU32,
    MOp::LdsAtomicMinU32Rtn,
    MOp::LdsAtomicMaxI32,
    MOp::LdsAtomicMaxI32Rtn,
    MOp::LdsAtomicMinI32,
    MOp::LdsAtomicMinI32Rtn,
    MOp::LdsAtomicIncU32,
    MOp::LdsAtomicIncU32Rtn,
    MOp::LdsAtomicAddF32,
    MOp::LdsAtomicAddF32Rtn,
    MOp::GlobalAtomicAddU32,
    MOp::GlobalAtomicAddU32Rtn,
    MOp::GlobalAtomicMaxU32,
    MOp::GlobalAtomicMaxU32Rtn,
    MOp::GlobalAtomicMinU32,
    MOp::GlobalAtomicMinU32Rtn,
    MOp::GlobalAtomicMaxI32,
    MOp::GlobalAtomicMaxI32Rtn,
    MOp::GlobalAtomicMinI32,
    MOp::GlobalAtomicMinI32Rtn,
    MOp::GlobalAtomicIncU32,
    MOp::GlobalAtomicIncU32Rtn,
    MOp::GlobalAtomicAddF32,
    MOp::GlobalAtomicAddF32Rtn,
    MOp::BufferWbl2,
    MOp::BufferInv,
    MOp::SLoadDword,
    MOp::SLoadDwordx2,
    MOp::SLoadDwordx4,
    MOp::SMemTime,
    MOp::SLoPart,
    MOp::SReadPhys,
    MOp::SMovB32,
    MOp::SAddU32,
    MOp::SAddcU32,
    MOp::SMulI32,
    MOp::SShlB32,
    MOp::SLshrB32,
    MOp::SAndB32,
    MOp::SXorB32,
    MOp::SCmpLtI32,
    MOp::SMinI32,
    MOp::SCselectB32,
    MOp::SCselectB64,
    MOp::SCselectB128,
    MOp::SNop,
    MOp::SWaitVm,
    MOp::SWaitLgkm,
    MOp::SEndpgm,
];

/// The attributes of `op`.
pub fn attrs(op: MOp) -> OpAttrs {
    attrs_of(op)
}

const fn attrs_of(op: MOp) -> OpAttrs {
    use Base::Src;
    use Dce::*;
    use FenceAttr as F;
    use Grade::*;
    use MOp::*;
    use Movable::*;
    use Pipe::*;
    use Space::{Global as G, Lds as L};
    // VALU issue: one wave64 VALU every 4 cycles (V)
    let valu = c(Valu, 4, 4, Verified);
    let valu8 = c(Valu, 8, 8, Verified);
    let salu = c(Salu, 4, 4, Verified);
    let ctrl = c(Ctrl, 4, 4, Verified);
    let mfma64 = c(Mfma, 64, 84, Verified);
    let mfma32 = c(Mfma, 32, 48, Verified);
    let mfma16 = c(Mfma, 16, 32, Verified);
    let ld4 = c(Vmem, 4, 220, Experimental);
    let ld16 = c(Vmem, 14, 240, Experimental);
    let st = c(Vmem, 8, 4, Unknown);
    let dma = c(Vmem, 14, 240, Unknown);
    let smem = c(Smem, 4, 200, Unknown);
    match op {
        // ---- vector ALU
        VMovB32 => a("v_mov_b32", co(Valu, 4, 2, 4, Verified)).movable(Yes).wdelay(2).dce(Pure),
        VMovFromS => a("v_mov_b32", valu).wdelay(2).dce(Pure),
        VMovFromSPair => a("v_mov_b32", c(Valu, 8, 8, Experimental)).wdelay(2).dce(Pure),
        VPtrAdd => a("v_add_co_u32", c(Valu, 8, 8, Experimental)).vcc(false, true).wdelay(4).dce(Pure),
        VReadPhys => a("v_mov_b32", valu).wdelay(2).dce(Pure),
        VReadFirstLane => a("v_readfirstlane_b32", c(Valu, 4, 12, Unknown)).hz_reader().dce(Pure),
        VMovDpp => a("v_mov_b32_dpp", c(Valu, 4, 8, Unknown)).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VCvtPknormU16 => a("v_cvt_pknorm_u16_f32", valu).wdelay(2),
        VCvtPkFp8F32 => a("v_cvt_pk_fp8_f32", valu8).regfx(RegFx::PartialWrite).half(Half::Lo).movable(Yes).wdelay(2),
        VCvtPkBf16F32 => a("v_cvt_pk_bf16_f32", valu).movable(Yes).wdelay(2),
        VCvtPkFp8Lo => a("v_cvt_scalef32_pk_fp8_f32", valu8).regfx(RegFx::PartialWrite).half(Half::Lo).wdelay(2),
        VCvtPkFp8Hi => a("v_cvt_scalef32_pk_fp8_f32", valu8).regfx(RegFx::PartialWrite).half(Half::Hi).wdelay(2),
        VCvtPkFp8F32Hi => a("v_cvt_pk_fp8_f32", valu8).regfx(RegFx::PartialWrite).half(Half::Hi).movable(Yes).wdelay(2),
        VPkAddF32 => a("v_pk_add_f32", valu).movable(Yes).wdelay(2),
        VPermlane32Swap => a("v_permlane32_swap_b32", valu8).regfx(RegFx::SwapSrc01).wdelay(2).hz_reader(),
        VPkMulF32 => a("v_pk_mul_f32", valu).movable(Yes).wdelay(2),
        VAddU32 => a("v_add_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VAddCoU32 => a("v_add_co_u32", valu).vcc(false, true).def_vcc().wdelay(4).dce(Pure),
        VAddcCoU32 => a("v_addc_co_u32", valu).vcc(true, true).def_vcc().use_vcc().wdelay(4).dce(Pure),
        VAddF32 => a("v_add_f32", co(Valu, 4, 2, 4, Verified)).movable(Yes).wdelay(2).dce(Pure),
        VSubF32 => a("v_sub_f32", co(Valu, 4, 2, 4, Verified)).movable(Yes).wdelay(2).dce(Pure),
        VMulF32 => a("v_mul_f32", co(Valu, 4, 2, 4, Verified)).movable(Yes).wdelay(2).dce(Pure),
        VMaxF32 => a("v_max_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMax3F32 => a("v_max3_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VPermB32 => a("v_perm_b32", valu).movable(Yes).wdelay(2),
        VPkSubU16 => a("v_pk_sub_u16", valu).movable(Yes).wdelay(2),
        VFmaF32 => a("v_fma_f32", co(Valu, 4, 2, 4, Verified)).movable(Yes).wdelay(2).dce(Pure),
        VExpF32 => a("v_exp_f32", valu8).trans().dce(Pure),
        VLogF32 => a("v_log_f32", valu8).trans().dce(Pure),
        VRcpF32 => a("v_rcp_f32", valu8).trans().dce(Pure),
        VCvtF32F16 => a("v_cvt_f32_f16", valu).wdelay(2).dce(Pure),
        VCvtF16F32 => a("v_cvt_f16_f32", valu).wdelay(2).dce(Pure),
        VAndB32 => a("v_and_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMinI32 => a("v_min_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VPkFmaF32 => a("v_pk_fma_f32", valu).movable(Yes).wdelay(2),
        SMulHiU32 => a("s_mul_hi_u32", salu).dce(Pure),
        VMadU64U32 => a("v_mad_u64_u32", c(Valu, 8, 8, Experimental)).vcc(false, true).wdelay(4).dce(Pure),
        VReadLane => a("v_readlane_b32", c(Valu, 4, 12, Unknown)).hz_reader().dce(Pure),
        VWriteLane => a("v_writelane_b32", valu8).regfx(RegFx::PartialWrite).wdelay(2),
        VMbcntLo => a("v_mbcnt_lo_u32_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMbcntHi => a("v_mbcnt_hi_u32_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VBcntU32 => a("v_bcnt_u32_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VFfblB32 => a("v_ffbl_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VFfbhU32 => a("v_ffbh_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCmpMaskNeU32 => a("v_cmp_ne_u32_e64", valu).wdelay(2).dce(Pure),
        SMaskLo => a("s_mov_b32", salu).dce(Pure),
        SMaskHi => a("s_mov_b32", salu).dce(Pure),
        LdsBpermute => a("ds_bpermute_b32", c(Lds, 4, 52, Experimental)).mem(L, false, Base::Unknown, 4).dce(Load), // returns through lgkmcnt: tracked like an LDS read by the wait pass (it touches no LDS memory; the ordering is conservative)
        LdsPermute => a("ds_permute_b32", c(Lds, 4, 52, Experimental)).mem(L, false, Base::Unknown, 4).dce(Load), // returns through lgkmcnt: tracked like an LDS read by the wait pass (it touches no LDS memory; the ordering is conservative)
        LdsSwizzle => a("ds_swizzle_b32", c(Lds, 4, 52, Experimental)).mem(L, false, Base::Unknown, 4).dce(Load), // returns through lgkmcnt: tracked like an LDS read by the wait pass (it touches no LDS memory; the ordering is conservative)
        VMovDppX => a("v_mov_b32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VAddF32Dpp => a("v_add_f32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VSubF32Dpp => a("v_sub_f32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VMulF32Dpp => a("v_mul_f32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VMaxF32Dpp => a("v_max_f32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VMinF32Dpp => a("v_min_f32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VAddU32Dpp => a("v_add_u32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VMaxI32Dpp => a("v_max_i32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VMinI32Dpp => a("v_min_i32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VMaxU32Dpp => a("v_max_u32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VMinU32Dpp => a("v_min_u32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VAndB32Dpp => a("v_and_b32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VOrB32Dpp => a("v_or_b32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VXorB32Dpp => a("v_xor_b32_dpp", c(Valu, 4, 8, Unknown)).regfx(RegFx::PartialWrite).wdelay(2).hz_reader().backend(Backend::OwnOnly),
        VPermlane16Swap => a("v_permlane16_swap_b32", valu8).regfx(RegFx::SwapSrc01).wdelay(2).hz_reader(),
        VCvtPkF32Fp8 => a("v_cvt_pk_f32_fp8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32Fp8 => a("v_cvt_f32_fp8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtPkF32Bf8 => a("v_cvt_pk_f32_bf8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32Bf8 => a("v_cvt_f32_bf8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtPkBf8F32 => a("v_cvt_pk_bf8_f32", valu8).regfx(RegFx::PartialWrite).movable(Yes).wdelay(2),
        VCvtScalePkBf16Fp8 => a("v_cvt_scalef32_pk_bf16_fp8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkF16Fp8 => a("v_cvt_scalef32_pk_f16_fp8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkBf16Bf8 => a("v_cvt_scalef32_pk_bf16_bf8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkF16Bf8 => a("v_cvt_scalef32_pk_f16_bf8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkF32Fp8 => a("v_cvt_scalef32_pk_f32_fp8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkF32Bf8 => a("v_cvt_scalef32_pk_f32_bf8", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkF32Fp4 => a("v_cvt_scalef32_pk_f32_fp4", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScale2xPk16Fp6F32 => a("v_cvt_scalef32_2xpk16_fp6_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScale2xPk16Bf6F32 => a("v_cvt_scalef32_2xpk16_bf6_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32Fp6F16 => a("v_cvt_scalef32_pk32_fp6_f16", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32Fp6Bf16 => a("v_cvt_scalef32_pk32_fp6_bf16", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32Bf6F16 => a("v_cvt_scalef32_pk32_bf6_f16", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32Bf6Bf16 => a("v_cvt_scalef32_pk32_bf6_bf16", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32F16Fp6 => a("v_cvt_scalef32_pk32_f16_fp6", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32Bf16Fp6 => a("v_cvt_scalef32_pk32_bf16_fp6", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32F16Bf6 => a("v_cvt_scalef32_pk32_f16_bf6", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePk32Bf16Bf6 => a("v_cvt_scalef32_pk32_bf16_bf6", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkBf16Fp4 => a("v_cvt_scalef32_pk_bf16_fp4", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkF16Fp4 => a("v_cvt_scalef32_pk_f16_fp4", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtScalePkFp8F32 => a("v_cvt_scalef32_pk_fp8_f32", valu8).regfx(RegFx::PartialWrite).movable(Yes).wdelay(2),
        VCvtScalePkBf8F32 => a("v_cvt_scalef32_pk_bf8_f32", valu8).regfx(RegFx::PartialWrite).movable(Yes).wdelay(2),
        VCvtScalePkFp4F32 => a("v_cvt_scalef32_pk_fp4_f32", valu8).regfx(RegFx::PartialWrite).movable(Yes).wdelay(2),
        VCvtPkU8F32 => a("v_cvt_pk_u8_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtPkI16I32 => a("v_cvt_pk_i16_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtPkU16U32 => a("v_cvt_pk_u16_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VFloorF32 => a("v_floor_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCeilF32 => a("v_ceil_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VTruncF32 => a("v_trunc_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VRndneF32 => a("v_rndne_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VFractF32 => a("v_fract_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VRsqF32 => a("v_rsq_f32", valu8).trans().dce(Pure),
        VSqrtF32 => a("v_sqrt_f32", valu8).trans().dce(Pure),
        VSinF32 => a("v_sin_f32", valu8).trans().dce(Pure),
        VCosF32 => a("v_cos_f32", valu8).trans().dce(Pure),
        VCvtI32F32 => a("v_cvt_i32_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtU32F32 => a("v_cvt_u32_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32I32 => a("v_cvt_f32_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32U32 => a("v_cvt_f32_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32Ubyte0 => a("v_cvt_f32_ubyte0", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32Ubyte1 => a("v_cvt_f32_ubyte1", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32Ubyte2 => a("v_cvt_f32_ubyte2", valu).movable(Yes).wdelay(2).dce(Pure),
        VCvtF32Ubyte3 => a("v_cvt_f32_ubyte3", valu).movable(Yes).wdelay(2).dce(Pure),
        VMinF32 => a("v_min_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMaxI32 => a("v_max_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMaxU32 => a("v_max_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMinU32 => a("v_min_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VSubU32 => a("v_sub_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMulU32U24 => a("v_mul_u32_u24", valu).movable(Yes).wdelay(2).dce(Pure),
        VAshrI32 => a("v_ashrrev_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMulHiU32 => a("v_mul_hi_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMulHiI32 => a("v_mul_hi_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMin3F32 => a("v_min3_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMed3F32 => a("v_med3_f32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMin3I32 => a("v_min3_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMin3U32 => a("v_min3_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMax3I32 => a("v_max3_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMax3U32 => a("v_max3_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMed3I32 => a("v_med3_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMed3U32 => a("v_med3_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMadU32U24 => a("v_mad_u32_u24", valu).movable(Yes).wdelay(2).dce(Pure),
        VLshlAddU32 => a("v_lshl_add_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VAdd3U32 => a("v_add3_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VBfeU32 => a("v_bfe_u32", valu).movable(Yes).wdelay(2).dce(Pure),
        VBfeI32 => a("v_bfe_i32", valu).movable(Yes).wdelay(2).dce(Pure),
        VBfiB32 => a("v_bfi_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VPkAddF16 => a("v_pk_add_f16", valu).movable(Yes).wdelay(2).dce(Pure),
        VPkMulF16 => a("v_pk_mul_f16", valu).movable(Yes).wdelay(2).dce(Pure),
        VPkMaxF16 => a("v_pk_max_f16", valu).movable(Yes).wdelay(2).dce(Pure),
        VPkMinF16 => a("v_pk_min_f16", valu).movable(Yes).wdelay(2).dce(Pure),
        VPkFmaF16 => a("v_pk_fma_f16", valu).movable(Yes).wdelay(2).dce(Pure),
        VCmpLeF32 => a("v_cmp_le_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpGtF32 => a("v_cmp_gt_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpGeF32 => a("v_cmp_ge_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpEqF32 => a("v_cmp_eq_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpNeF32 => a("v_cmp_neq_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpLgF32 => a("v_cmp_lg_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpUF32 => a("v_cmp_u_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpOF32 => a("v_cmp_o_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpLeI32 => a("v_cmp_le_i32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpEqI32 => a("v_cmp_eq_i32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpLtU32 => a("v_cmp_lt_u32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpLeU32 => a("v_cmp_le_u32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpGtU32 => a("v_cmp_gt_u32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpGeU32 => a("v_cmp_ge_u32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpEqU32 => a("v_cmp_eq_u32", valu).vcc(false, true).def_vcc().dce(Pure),
        VOrB32 => a("v_or_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VXorB32 => a("v_xor_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VLshrB32 => a("v_lshrrev_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VLshlB32 => a("v_lshlrev_b32", valu).movable(Yes).wdelay(2).dce(Pure),
        VMulLoU32 => a("v_mul_lo_u32", c(Valu, 16, 16, Unknown)).movable(Yes).wdelay(2).dce(Pure),
        VCmpLtI32 => a("v_cmp_lt_i32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpGeI32 => a("v_cmp_ge_i32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpGtI32 => a("v_cmp_gt_i32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpMaskLtI32 => a("v_cmp_lt_i32_e64", valu).wdelay(2).backend(Backend::NoRow),
        VCndmaskM => a("v_cndmask_b32_e64", valu).hz_reader().backend(Backend::NoRow),
        VCmpNeI32 => a("v_cmp_ne_u32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCmpLtF32 => a("v_cmp_lt_f32", valu).vcc(false, true).def_vcc().dce(Pure),
        VCndmask => a("v_cndmask_b32", valu).vcc(true, false).use_vcc().wdelay(2).dce(Pure),
        // ---- fragments / matrix cores
        VWriteSub => a("v_mov_b32", valu).regfx(RegFx::SubWrite).wdelay(2),
        VReadSub => a("v_mov_b32", valu).regfx(RegFx::SubRead).wdelay(2),
        VMfma16x16x16F16 => a("v_mfma_f32_16x16x16_f16", mfma16).mfma(12, false, false, false, false).agpr(true, 0b111),
        VMfma32x32x8F16 => a("v_mfma_f32_32x32x8_f16", mfma32).mfma(12, false, false, false, false).agpr(true, 0b111),
        VMfma32x32x8F16Z => a("v_mfma_f32_32x32x8_f16", mfma32).mfma(12, false, false, true, false).agpr(true, 0b11),
        VMfma16x16x32Fp8 => a("v_mfma_f32_16x16x32_fp8_fp8", mfma16).mfma(12, false, false, false, false).agpr(true, 0b111),
        VMfma32x32x16Fp8 => a("v_mfma_f32_32x32x16_fp8_fp8", mfma32).mfma(12, false, false, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64Fp8 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        // 8-pass XDL op: conservatively the 16-pass wait count until the gfx950 table entry is pinned; latency from the 16-pass table until measured
        VMfmaScale16x16x128Fp8 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64Fp8Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfma32x32x64Fp8 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfma32x32x64Fp8Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        // sparse fp8 16x16x128: arch table says 32 cycles (MI350X, F8 involved); latency not measured
        VSmfmac16x16x128Fp8 => a("v_smfmac_f32_16x16x128_fp8_fp8", c(Mfma, 32, 32, Experimental)).mfma(20, true, false, false, true),
        VMfmaPlain16x16x128A4B4 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A4B4Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A4B6 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A4B6Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A4B8 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A4B8Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A6B4 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A6B4Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A6B6 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A6B6Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A6B8 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A6B8Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A8B4 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A8B4Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A8B6 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A8B6Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain16x16x128A8B8 => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain16x16x128A8B8Z => a("v_mfma_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A4B4 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A4B4Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A4B6 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A4B6Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A4B8 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A4B8Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A6B4 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A6B4Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A6B6 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A6B6Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A6B8 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A6B8Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A8B4 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A8B4Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A8B6 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A8B6Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaPlain32x32x64A8B8 => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaPlain32x32x64A8B8Z => a("v_mfma_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A4B4 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A4B4Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A4B6 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A4B6Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A4B8 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A4B8Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A6B4 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A6B4Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A6B6 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A6B6Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(8, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A6B8 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A6B8Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A8B4 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A8B4Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A8B6 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A8B6Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale16x16x128A8B8 => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale16x16x128A8B8Z => a("v_mfma_scale_f32_16x16x128_f8f6f4", c(Mfma, 32, 48, Experimental)).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A4B4 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A4B4Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A4B6 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A4B6Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A4B8 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A4B8Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A6B4 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A6B4Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A6B6 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A6B6Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(12, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A6B8 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A6B8Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A8B4 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A8B4Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A8B6 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A8B6Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfmaScale32x32x64A8B8 => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, true, true, false, false).agpr(true, 0b111),
        VMfmaScale32x32x64A8B8Z => a("v_mfma_scale_f32_32x32x64_f8f6f4", mfma64).mfma(20, false, true, true, false).agpr(true, 0b11),
        VMfma16x16x32Bf16 => a("v_mfma_f32_16x16x32_bf16", mfma16).mfma(12, true, false, false, false).agpr(true, 0b111),
        VMfma16x16x32Bf16Z => a("v_mfma_f32_16x16x32_bf16", mfma16).mfma(12, false, false, true, false).agpr(true, 0b11),
        VMfma32x32x16Bf16 => a("v_mfma_f32_32x32x16_bf16", mfma32).mfma(20, true, false, false, false).agpr(true, 0b111),
        VMfma32x32x16Bf16Z => a("v_mfma_f32_32x32x16_bf16", mfma32).mfma(20, false, false, true, false).agpr(true, 0b11),
        VMfma16x16x32F16 => a("v_mfma_f32_16x16x32_f16", mfma16).mfma(12, true, false, false, false).agpr(true, 0b111),
        VMfma16x16x32F16Z => a("v_mfma_f32_16x16x32_f16", mfma16).mfma(12, false, false, true, false).agpr(true, 0b11),
        VMfma32x32x16F16 => a("v_mfma_f32_32x32x16_f16", mfma32).mfma(20, true, false, false, false).agpr(true, 0b111),
        VMfma32x32x16F16Z => a("v_mfma_f32_32x32x16_f16", mfma32).mfma(20, false, false, true, false).agpr(true, 0b11),
        VMfma16x16x64I8 => a("v_mfma_i32_16x16x64_i8", mfma16).mfma(12, true, false, false, false).agpr(true, 0b111),
        VMfma16x16x64I8Z => a("v_mfma_i32_16x16x64_i8", mfma16).mfma(12, false, false, true, false).agpr(true, 0b11),
        VMfma32x32x32I8 => a("v_mfma_i32_32x32x32_i8", mfma32).mfma(20, true, false, false, false).agpr(true, 0b111),
        VMfma32x32x32I8Z => a("v_mfma_i32_32x32x32_i8", mfma32).mfma(20, false, false, true, false).agpr(true, 0b11),
        VAccvgprRead => a("v_accvgpr_read_b32", c(Valu, 4, 8, Unknown)).wdelay(2).agpr(false, 0b1),
        VAccvgprWrite => a("v_accvgpr_write_b32", c(Valu, 4, 8, Unknown)).wdelay(2).agpr(true, 0),
        VAccvgprMov => a("v_accvgpr_mov_b32", c(Valu, 4, 8, Unknown)).wdelay(2).agpr(true, 0b1),
        // ---- LDS (E): b128 issue 8, b32 4, tr_b8 ~6, write b128 ~26; latency ~52 (reads) / ~75 (write)
        LdsWriteB32 => a("ds_write_b32", c(Lds, 8, 4, Unknown)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsReadB32 => a("ds_read_b32", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 4).agpr(true, 0).dce(Load),
        LdsWriteB128 => a("ds_write_b128", c(Lds, 26, 4, Experimental)).mem(L, true, Src(0), 16).agpr(false, 0b10),
        LdsReadB128 => a("ds_read_b128", c(Lds, 8, 52, Experimental)).mem(L, false, Src(0), 16).agpr(true, 0).dce(Load),
        LdsReadB64 => a("ds_read_b64", c(Lds, 4, 48, Experimental)).mem(L, false, Src(0), 8).dce(Load).backend(Backend::NoRow),
        LdsWriteB64 => a("ds_write_b64", c(Lds, 8, 4, Experimental)).mem(L, true, Src(0), 8).backend(Backend::NoRow),
        LdsReadB64TrB8 => a("ds_read_b64_tr_b8", c(Lds, 6, 52, Experimental)).mem(L, false, Src(0), 8).agpr(true, 0).dce(Load),
        SBarrier => a("s_barrier", ctrl).fence_only(F::Barrier),
        SSetPrio => a("s_setprio", ctrl).fence_only(F::SetPrio),
        SchedGroup => a("sched_group_barrier", ctrl),
        // ---- global memory: L1 hit ~100-108, L2 ~210-250 (E); streaming dwordx4 issue 14
        GlobalLoadU16 => a("global_load_ushort", ld4).mem(G, false, Src(0), 2).agpr(true, 0).dce(Load),
        GlobalLoadF32 => a("global_load_dword", ld4).mem(G, false, Src(0), 4).agpr(true, 0).dce(Load),
        GlobalLoadB128 => a("global_load_dwordx4", ld16).mem(G, false, Src(0), 16).agpr(true, 0).dce(Load),
        // global -> LDS DMA: a global read with a fence-like LDS write through M0
        GlobalLoadLdsB128 => a("global_load_lds_dwordx4", dma).mem(G, false, Src(0), 16).fence(F::Barrier).m0(1).backend(Backend::OwnOnly),
        GlobalLoadLdsB32 => a("global_load_lds_dword", dma).mem(G, false, Src(0), 16).fence(F::Barrier).m0(1).backend(Backend::OwnOnly),
        GlobalLoadLdsB128S => a("global_load_lds_dwordx4", dma).mem(G, false, Src(1), 16).fence(F::Barrier).m0(2).backend(Backend::OwnOnly),
        GlobalLoadLdsB32S => a("global_load_lds_dword", dma).mem(G, false, Src(1), 16).fence(F::Barrier).m0(2).backend(Backend::OwnOnly),
        // ---- MUBUF: the per-lane address is not an affine function of one base register (unknown base, always may-alias)
        BufferLoadDwordX1 => a("buffer_load_dword", ld4).mem(G, false, Base::Unknown, 4).agpr(true, 0),
        BufferLoadDwordX2 => a("buffer_load_dwordx2", ld16).mem(G, false, Base::Unknown, 8).agpr(true, 0),
        BufferLoadDwordX4 => a("buffer_load_dwordx4", ld16).mem(G, false, Base::Unknown, 16).agpr(true, 0),
        BufferStoreDwordX1 => a("buffer_store_dword", st).mem(G, true, Base::Unknown, 4).agpr(false, 0b1),
        BufferStoreDwordX2 => a("buffer_store_dwordx2", st).mem(G, true, Base::Unknown, 8).agpr(false, 0b1),
        BufferStoreDwordX4 => a("buffer_store_dwordx4", st).mem(G, true, Base::Unknown, 16).wide_store(0).agpr(false, 0b1),
        BufferLoadDwordX4Lds => a("buffer_load_dwordx4", dma).mem(G, false, Base::Unknown, 16).fence(F::Barrier).m0(3).backend(Backend::OwnOnly),
        BufferLoadDwordX1Lds => a("buffer_load_dword", dma).mem(G, false, Base::Unknown, 16).fence(F::Barrier).m0(3).backend(Backend::OwnOnly),
        SBufferRsrc => a("s_mov_b32", c(Salu, 4, 4, Unknown)),
        SBufferRsrcDyn => a("s_mov_b32", c(Salu, 4, 4, Unknown)),
        GlobalStoreU16 => a("global_store_short", st).mem(G, true, Src(0), 2).agpr(false, 0b10),
        GlobalStoreF32 => a("global_store_dword", st).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalStoreF32x4 => a("global_store_dwordx4", st).mem(G, true, Src(0), 16).wide_store(1).agpr(false, 0b10),
        GlobalStoreF32x2 => a("global_store_dwordx2", st).mem(G, true, Src(0), 8).agpr(false, 0b10),
        GlobalLoadU8 => a("global_load_ubyte", ld4).mem(G, false, Src(0), 1).agpr(true, 0).dce(Load),
        GlobalLoadI8 => a("global_load_sbyte", ld4).mem(G, false, Src(0), 1).agpr(true, 0).dce(Load),
        GlobalLoadI16 => a("global_load_sshort", ld4).mem(G, false, Src(0), 2).agpr(true, 0).dce(Load),
        GlobalLoadB64 => a("global_load_dwordx2", ld16).mem(G, false, Src(0), 8).agpr(true, 0).dce(Load),
        GlobalLoadB96 => a("global_load_dwordx3", ld16).mem(G, false, Src(0), 12).agpr(true, 0).dce(Load),
        GlobalLoadD16U8 => a("global_load_ubyte_d16", ld4).mem(G, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Lo).dce(Load),
        GlobalLoadD16U8Hi => a("global_load_ubyte_d16_hi", ld4).mem(G, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Hi).dce(Load),
        GlobalLoadD16I8 => a("global_load_sbyte_d16", ld4).mem(G, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Lo).dce(Load),
        GlobalLoadD16I8Hi => a("global_load_sbyte_d16_hi", ld4).mem(G, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Hi).dce(Load),
        GlobalLoadD16B16 => a("global_load_short_d16", ld4).mem(G, false, Src(0), 2).regfx(RegFx::PartialWrite).half(Half::Lo).dce(Load),
        GlobalLoadD16B16Hi => a("global_load_short_d16_hi", ld4).mem(G, false, Src(0), 2).regfx(RegFx::PartialWrite).half(Half::Hi).dce(Load),
        GlobalStoreB8 => a("global_store_byte", st).mem(G, true, Src(0), 1).agpr(false, 0b10),
        GlobalStoreB8Hi => a("global_store_byte_d16_hi", st).mem(G, true, Src(0), 1).agpr(false, 0b10),
        GlobalStoreB16Hi => a("global_store_short_d16_hi", st).mem(G, true, Src(0), 2).agpr(false, 0b10),
        GlobalStoreB96 => a("global_store_dwordx3", st).mem(G, true, Src(0), 12).wide_store(1).agpr(false, 0b10),
        BufferLoadU8 => a("buffer_load_ubyte", ld4).mem(G, false, Base::Unknown, 1).agpr(true, 0),
        BufferLoadI8 => a("buffer_load_sbyte", ld4).mem(G, false, Base::Unknown, 1).agpr(true, 0),
        BufferLoadU16 => a("buffer_load_ushort", ld4).mem(G, false, Base::Unknown, 2).agpr(true, 0),
        BufferLoadI16 => a("buffer_load_sshort", ld4).mem(G, false, Base::Unknown, 2).agpr(true, 0),
        BufferLoadB96 => a("buffer_load_dwordx3", ld16).mem(G, false, Base::Unknown, 12).agpr(true, 0),
        BufferStoreB8 => a("buffer_store_byte", st).mem(G, true, Base::Unknown, 1).agpr(false, 0b1),
        BufferStoreB16 => a("buffer_store_short", st).mem(G, true, Base::Unknown, 2).agpr(false, 0b1),
        BufferStoreB8Hi => a("buffer_store_byte_d16_hi", st).mem(G, true, Base::Unknown, 1).agpr(false, 0b1),
        BufferStoreB16Hi => a("buffer_store_short_d16_hi", st).mem(G, true, Base::Unknown, 2).agpr(false, 0b1),
        BufferStoreB96 => a("buffer_store_dwordx3", st).mem(G, true, Base::Unknown, 12).wide_store(0).agpr(false, 0b1),
        LdsReadU8 => a("ds_read_u8", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 1).agpr(true, 0).dce(Load),
        LdsReadI8 => a("ds_read_i8", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 1).agpr(true, 0).dce(Load),
        LdsReadU16 => a("ds_read_u16", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 2).agpr(true, 0).dce(Load),
        LdsReadI16 => a("ds_read_i16", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 2).agpr(true, 0).dce(Load),
        LdsReadB96 => a("ds_read_b96", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 12).agpr(true, 0).dce(Load),
        LdsReadB64TrB16 => a("ds_read_b64_tr_b16", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 8).agpr(true, 0).dce(Load),
        LdsReadD16U8 => a("ds_read_u8_d16", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Lo).dce(Load),
        LdsReadD16U8Hi => a("ds_read_u8_d16_hi", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Hi).dce(Load),
        LdsReadD16I8 => a("ds_read_i8_d16", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Lo).dce(Load),
        LdsReadD16I8Hi => a("ds_read_i8_d16_hi", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 1).regfx(RegFx::PartialWrite).half(Half::Hi).dce(Load),
        LdsReadD16B16 => a("ds_read_u16_d16", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 2).regfx(RegFx::PartialWrite).half(Half::Lo).dce(Load),
        LdsReadD16B16Hi => a("ds_read_u16_d16_hi", c(Lds, 4, 52, Experimental)).mem(L, false, Src(0), 2).regfx(RegFx::PartialWrite).half(Half::Hi).dce(Load),
        LdsWriteB8 => a("ds_write_b8", c(Lds, 8, 4, Unknown)).mem(L, true, Src(0), 1).agpr(false, 0b10),
        LdsWriteB16 => a("ds_write_b16", c(Lds, 8, 4, Unknown)).mem(L, true, Src(0), 2).agpr(false, 0b10),
        LdsWriteB8Hi => a("ds_write_b8_d16_hi", c(Lds, 8, 4, Unknown)).mem(L, true, Src(0), 1).agpr(false, 0b10),
        LdsWriteB16Hi => a("ds_write_b16_d16_hi", c(Lds, 8, 4, Unknown)).mem(L, true, Src(0), 2).agpr(false, 0b10),
        LdsWriteB96 => a("ds_write_b96", c(Lds, 8, 4, Unknown)).mem(L, true, Src(0), 12).agpr(false, 0b10),
        LdsRead2B32 => a("ds_read2_b32", c(Lds, 8, 52, Experimental)).mem(L, false, Base::Unknown, 8).agpr(true, 0).dce(Load),
        LdsRead2B64 => a("ds_read2_b64", c(Lds, 8, 52, Experimental)).mem(L, false, Base::Unknown, 16).agpr(true, 0).dce(Load),
        LdsRead2St64B32 => a("ds_read2st64_b32", c(Lds, 8, 52, Experimental)).mem(L, false, Base::Unknown, 8).agpr(true, 0).dce(Load),
        LdsRead2St64B64 => a("ds_read2st64_b64", c(Lds, 8, 52, Experimental)).mem(L, false, Base::Unknown, 16).agpr(true, 0).dce(Load),
        LdsWrite2B32 => a("ds_write2_b32", c(Lds, 8, 4, Unknown)).mem(L, true, Base::Unknown, 8).agpr(false, 0b110),
        LdsWrite2B64 => a("ds_write2_b64", c(Lds, 8, 4, Unknown)).mem(L, true, Base::Unknown, 16).agpr(false, 0b110),
        LdsWrite2St64B32 => a("ds_write2st64_b32", c(Lds, 8, 4, Unknown)).mem(L, true, Base::Unknown, 8).agpr(false, 0b110),
        LdsWrite2St64B64 => a("ds_write2st64_b64", c(Lds, 8, 4, Unknown)).mem(L, true, Base::Unknown, 16).agpr(false, 0b110),
        GlobalLoadF32Sc1 => a("global_load_dword", c(Vmem, 4, 900, Unknown)).mem(G, false, Src(0), 4).agpr(true, 0),
        GlobalStoreF32Sc1 => a("global_store_dword", st).mem(G, true, Src(0), 4).agpr(false, 0b10),
        // read-modify-write with a result: a global access (vmcnt queue entry whose destination is the old value)
        GlobalAtomicAddRet => a("global_atomic_add", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        BufferAtomicAddU32 => a("buffer_atomic_add", c(Vmem, 8, 4, Unknown)).mem(G, true, Base::Unknown, 4).agpr(false, 0b1),
        BufferAtomicAddF32 => a("buffer_atomic_add_f32", c(Vmem, 8, 4, Unknown)).mem(G, true, Base::Unknown, 4).agpr(false, 0b1),
        BufferAtomicPkAddBf16 => a("buffer_atomic_pk_add_bf16", c(Vmem, 8, 4, Unknown)).mem(G, true, Base::Unknown, 4).agpr(false, 0b1),
        BufferAtomicPkAddF16 => a("buffer_atomic_pk_add_f16", c(Vmem, 8, 4, Unknown)).mem(G, true, Base::Unknown, 4).agpr(false, 0b1),
        GlobalAtomicPkAddBf16 => a("global_atomic_pk_add_bf16", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicPkAddBf16Rtn => a("global_atomic_pk_add_bf16", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        GlobalAtomicPkAddF16 => a("global_atomic_pk_add_f16", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicPkAddF16Rtn => a("global_atomic_pk_add_f16", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        LdsAtomicAddU32 => a("ds_add_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicAddU32Rtn => a("ds_add_rtn_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMaxU32 => a("ds_max_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMaxU32Rtn => a("ds_max_rtn_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMinU32 => a("ds_min_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMinU32Rtn => a("ds_min_rtn_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMaxI32 => a("ds_max_i32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMaxI32Rtn => a("ds_max_rtn_i32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMinI32 => a("ds_min_i32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicMinI32Rtn => a("ds_min_rtn_i32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicIncU32 => a("ds_inc_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicIncU32Rtn => a("ds_inc_rtn_u32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicAddF32 => a("ds_add_f32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        LdsAtomicAddF32Rtn => a("ds_add_rtn_f32", c(Lds, 8, 52, Experimental)).mem(L, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicAddU32 => a("global_atomic_add", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicAddU32Rtn => a("global_atomic_add", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        GlobalAtomicMaxU32 => a("global_atomic_umax", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicMaxU32Rtn => a("global_atomic_umax", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        GlobalAtomicMinU32 => a("global_atomic_umin", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicMinU32Rtn => a("global_atomic_umin", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        GlobalAtomicMaxI32 => a("global_atomic_smax", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicMaxI32Rtn => a("global_atomic_smax", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        GlobalAtomicMinI32 => a("global_atomic_smin", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicMinI32Rtn => a("global_atomic_smin", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        GlobalAtomicIncU32 => a("global_atomic_inc", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicIncU32Rtn => a("global_atomic_inc", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        GlobalAtomicAddF32 => a("global_atomic_add_f32", c(Vmem, 8, 4, Unknown)).mem(G, true, Src(0), 4).agpr(false, 0b10),
        GlobalAtomicAddF32Rtn => a("global_atomic_add_f32", c(Vmem, 8, 900, Unknown)).mem(G, true, Src(0), 4),
        // cache maintenance: ordered against every memory access, never reordered
        BufferWbl2 => a("buffer_wbl2", c(Ctrl, 4, 4, Unknown)).fence(F::Barrier),
        BufferInv => a("buffer_inv", c(Ctrl, 4, 4, Unknown)).fence(F::Barrier),
        // ---- scalar ALU / memory
        SLoadDword => a("s_load_dword", smem).smem().dce(Load),
        SLoadDwordx2 => a("s_load_dwordx2", smem).smem().dce(Load),
        SLoadDwordx4 => a("s_load_dwordx4", smem).smem().dce(Load).backend(Backend::OwnOnly),
        // a timing stamp: SMEM result, pinned in program order against every other memory op
        SMemTime => a("s_memtime", smem).smem().fence(F::Barrier).backend(Backend::NoRow),
        SLoPart => a("s_mov_b32", salu).backend(Backend::NoRow).dce(Pure),
        SReadPhys => a("s_mov_b32", salu).dce(Pure),
        SMovB32 => a("s_mov_b32", salu).dce(Pure),
        SAddU32 => a("s_add_u32", salu).scc(false, true).dce(Pure),
        SAddcU32 => a("s_addc_u32", salu).scc(true, true).dce(Pure),
        SMulI32 => a("s_mul_i32", salu).dce(Pure),
        SShlB32 => a("s_lshl_b32", salu).scc(false, true).dce(Pure),
        SLshrB32 => a("s_lshr_b32", salu).scc(false, true).dce(Pure),
        SAndB32 => a("s_and_b32", salu).scc(false, true).dce(Pure),
        SXorB32 => a("s_xor_b32", salu).scc(false, true).dce(Pure),
        SCmpLtI32 => a("s_cmp_lt_i32", salu).scc(false, true),
        SMinI32 => a("s_min_i32", salu).scc(false, true).backend(Backend::OwnOnly).dce(Pure),
        SCselectB32 => a("s_cselect_b32", salu).scc(true, false).backend(Backend::OwnOnly).dce(Pure),
        SCselectB64 => a("s_cselect_b64", salu).scc(true, false).backend(Backend::OwnOnly).dce(Pure),
        SCselectB128 => a("s_cselect_b64", salu).scc(true, false).backend(Backend::OwnOnly).dce(Pure),
        // ---- misc
        SNop => a("s_nop", ctrl).fence_only(F::Nop),
        SWaitVm => a("s_waitcnt", ctrl).fence_only(F::WaitVm),
        SWaitLgkm => a("s_waitcnt", ctrl).fence_only(F::WaitLgkm),
        SEndpgm => a("s_endpgm", ctrl).fence_only(F::Nop),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_is_listed() {
        // ALL must list every variant of the enum (the attribute match is exhaustive at compile time)
        let src = include_str!("mir.rs");
        let body = &src[src.find("pub enum MOp {").unwrap()..];
        let body = &body[..body.find("\n}").unwrap()];
        let names: Vec<&str> = body.lines().map(|l| l.trim()).filter(|l| l.ends_with(',') && l.chars().next().map_or(false, |c| c.is_ascii_uppercase())).map(|l| l.trim_end_matches(',')).collect();
        let listed: Vec<String> = ALL.iter().map(|o| format!("{o:?}")).collect();
        for n in &names {
            assert!(listed.iter().any(|l| l == n), "MOp::{n} is missing from op_attrs::ALL");
        }
        assert_eq!(names.len(), listed.len(), "op_attrs::ALL has duplicates or unknown entries");
    }

    #[test]
    fn llvm_coverage_list_matches_the_backend_attribute() {
        // backend::ALL_MOPS (the LLVM-tail table coverage list) = the ops whose backend attribute says they have rows
        let mut rows: Vec<String> = ALL.iter().filter(|o| matches!(attrs(**o).backend, Backend::Llvm)).map(|o| format!("{o:?}")).collect();
        let mut tail: Vec<String> = crate::backend::ALL_MOPS.iter().map(|o| format!("{o:?}")).collect();
        tail.sort();
        rows.sort();
        assert_eq!(tail, rows, "backend::ALL_MOPS must be exactly the ops with LLVM rows (backend attribute)");
    }
}
