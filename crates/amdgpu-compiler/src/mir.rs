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
}

impl RClass {
    pub fn dwords(self) -> u32 {
        match self {
            RClass::V2 | RClass::S2 => 2,
            RClass::V4 => 4,
            RClass::V8 => 8,
            RClass::V16 => 16,
            _ => 1,
        }
    }

    /// Vector register classes share one physical file.
    pub fn is_vector(self) -> bool {
        matches!(self, RClass::V1 | RClass::V2 | RClass::V4 | RClass::V8 | RClass::V16)
    }
}

#[derive(Clone, Copy, Debug)]
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
    VRcpF32,
    VCvtF32F16,
    VCvtF16F32,
    VAndB32,
    VOrB32,
    VXorB32,
    VLshrB32,
    VLshlB32,
    VMulLoU32,
    VCmpLtI32,
    VCmpGeI32,
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
    /// Scaled MFMA with an inline-zero accumulator (no C operand): D = A x B.
    VMfmaScale32x32x64Fp8Z,
    // workgroup LDS
    LdsWriteB32,
    LdsReadB32,
    LdsWriteB128,
    LdsReadB128,
    LdsReadB64TrB8,
    SBarrier,
    /// `s_setprio` imm.
    SSetPrio,
    // global memory
    GlobalLoadU16,
    GlobalLoadF32,
    GlobalLoadB128,
    /// `global_load_dwordx4 v[d], v_off, s[base]`: scalar base + 32-bit vector offset (no 64-bit address arithmetic).
    GlobalLoadB128S,
    GlobalLoadF32S,
    /// Global -> LDS DMA (M0 = LDS base from srcs[1], an S1 register): srcs: ptr (V2), lds base (S1); imm = byte offset.
    GlobalLoadLdsB128,
    GlobalLoadLdsB32,
    GlobalStoreU16,
    GlobalStoreF32,
    // scalar ALU / memory
    SLoadDword,
    SLoadDwordx2,
    SReadPhys,
    SMovB32,
    SAddU32,
    SAddcU32,
    SMulI32,
    SShlB32,
    SCmpLtI32,
    // misc
    SNop,
    SWaitVm,
    SWaitLgkm,
    SEndpgm,
}

impl MOp {
    /// Does this instruction consume `vcc` from the immediately preceding
    /// instruction? Used by the verifier to guarantee adjacency.
    pub fn consumes_vcc(self) -> bool {
        matches!(self, MOp::VAddcCoU32 | MOp::VCndmask)
    }

    /// Does this instruction define `vcc`?
    pub fn defines_vcc(self) -> bool {
        matches!(
            self,
            MOp::VAddCoU32
                | MOp::VAddcCoU32
                | MOp::VCmpLtI32
                | MOp::VCmpGeI32
                | MOp::VCmpNeI32
                | MOp::VCmpLtF32
        )
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
}

impl MInst {
    pub fn new(op: MOp) -> Self {
        MInst { op, dst: None, srcs: Vec::new(), imm: 0, auto: false }
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
}

impl MFunction {
    /// The vreg that owns the register behind `r` (itself unless `r` is an alias).
    pub fn canon(&self, r: VReg) -> VReg {
        self.alias.get(&r).map(|a| a.0).unwrap_or(r)
    }

    pub fn vreg_class(&self, r: VReg) -> RClass {
        self.vregs[r as usize]
    }

    pub fn linear_index(&self, block: usize, inst: usize) -> usize {
        let mut n = 0usize;
        for b in 0..block {
            n += self.blocks[b].insts.len();
        }
        n + inst
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
            // Class sanity for the operations with fixed register shapes.
            match inst.op {
                MOp::GlobalLoadU16 | MOp::GlobalLoadF32 => {
                    if let Some(d) = inst.dst {
                        if m.vreg_class(d) != RClass::V1 {
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
                MOp::GlobalStoreU16 | MOp::GlobalStoreF32 => {
                    if let Some(Src::Reg(a)) = inst.srcs.first() {
                        if m.vreg_class(*a) != RClass::V2 {
                            errs.push(format!("block {bi} inst {ii}: store base must be V2"));
                        }
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
                        if m.vreg_class(d) != RClass::V4 {
                            errs.push(format!("block {bi} inst {ii}: mfma dst must be V4"));
                        }
                    }
                    let want = [RClass::V2, RClass::V2, RClass::V4];
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r) == *w => {}
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
                        if m.vreg_class(d) != RClass::V16 {
                            errs.push(format!("block {bi} inst {ii}: scaled mfma dst must be V16"));
                        }
                    }
                    for (k, w) in want.iter().enumerate().skip(1) {
                        match inst.srcs.get(k - 1) {
                            Some(Src::Reg(r)) if m.vreg_class(*r) == *w => {}
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
                MOp::VMfma16x16x32Fp8 | MOp::VMfma32x32x16Fp8 => {
                    let wide = inst.op == MOp::VMfma32x32x16Fp8;
                    let acc = if wide { RClass::V16 } else { RClass::V4 };
                    if let Some(d) = inst.dst {
                        if m.vreg_class(d) != acc {
                            errs.push(format!("block {bi} inst {ii}: fp8 mfma dst class mismatch"));
                        }
                    }
                    let want = [RClass::V2, RClass::V2, acc];
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r) == *w => {}
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
                        if m.vreg_class(d) != RClass::V16 {
                            errs.push(format!("block {bi} inst {ii}: f16 mfma dst must be V16"));
                        }
                    }
                }
                MOp::VMfma32x32x8F16 => {
                    if let Some(d) = inst.dst {
                        if m.vreg_class(d) != RClass::V16 {
                            errs.push(format!("block {bi} inst {ii}: wide mfma dst must be V16"));
                        }
                    }
                    let want = [RClass::V2, RClass::V2, RClass::V16];
                    for (k, w) in want.iter().enumerate() {
                        match inst.srcs.get(k) {
                            Some(Src::Reg(r)) if m.vreg_class(*r) == *w => {}
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
