//! Standalone use of a safe-asm block (tests, inspection): the block becomes a one-block MIR function so the existing register
//! allocation and emission produce text. Kernels embed blocks through `Op::Asm` instead (see `lower::lower_asm`).

use crate::mir::{Block, MFunction, MInst, MOp, RClass, Src};
use amdgpu_safeasm::{AClass, ASrc, AsmBlock};

fn rclass(c: AClass) -> RClass {
    match c {
        AClass::V1 => RClass::V1,
        AClass::V2 => RClass::V2,
        AClass::V4 => RClass::V4,
        AClass::V8 => RClass::V8,
        AClass::V16 => RClass::V16,
        AClass::S1 => RClass::S1,
        AClass::S2 => RClass::S2,
    }
}

/// `MOp` by variant name.
pub fn mop_by_name(n: &str) -> Option<MOp> {
    match n {
        "SNop" => Some(MOp::SNop),
        "SWaitLgkm" => Some(MOp::SWaitLgkm),
        "SWaitVm" => Some(MOp::SWaitVm),
        "VWriteSub" => Some(MOp::VWriteSub),
        "VCmpMaskLtI32" => Some(MOp::VCmpMaskLtI32),
        "VCndmaskM" => Some(MOp::VCndmaskM),
        _ => crate::backend::ALL_MOPS.iter().copied().find(|m| format!("{m:?}") == n),
    }
}

/// The block's instructions as MIR instructions with registers numbered as in the block.
pub fn block_insts(b: &AsmBlock) -> Result<Vec<MInst>, String> {
    b.insts()
        .iter()
        .map(|i| {
            let op = mop_by_name(i.op).ok_or_else(|| format!("unknown op `{}`", i.op))?;
            let mut mi = MInst::new(op);
            if let Some(d) = i.dst {
                mi = mi.def(d);
            }
            for s in &i.srcs {
                mi = match s {
                    ASrc::Reg(r) => mi.srcreg(*r),
                    ASrc::Imm(v) => mi.src(Src::Imm(*v)),
                };
            }
            Ok(mi.imm(i.imm))
        })
        .collect()
}

pub fn block_to_mfunction(b: &AsmBlock, name: &str) -> Result<MFunction, String> {
    b.verify().map_err(|e| format!("unverified sequence: {}", e.join("; ")))?;
    let mut blk = Block::new(0);
    blk.insts = block_insts(b)?;
    Ok(MFunction {
        name: name.to_string(),
        args: Vec::new(),
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: vec![blk],
        vregs: b.regs().iter().map(|c| rclass(*c)).collect(),
        loops: Vec::new(),
        alias: b.alias().iter().map(|(a, base, off)| (*a, (*base, *off))).collect(),
        elide_fallthrough: false,
    })
}

/// Assembly text of a standalone block (own backend: allocation + emission).
pub fn block_to_asm(b: &AsmBlock) -> Result<String, String> {
    let m = block_to_mfunction(b, "safeasm_seq")?;
    let a = crate::regalloc::allocate(&m)?;
    Ok(crate::emit::emit(&m, &a))
}