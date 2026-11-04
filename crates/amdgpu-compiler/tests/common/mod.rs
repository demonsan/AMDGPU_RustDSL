//! Sample-instruction builder shared by the LLVM-tail tests: one tiny instruction per table row.
#![allow(dead_code)]

use amdgpu_compiler::backend::llvm20::{Row, SigClass};
use amdgpu_compiler::mir::*;

pub struct Samples {
    pub vregs: Vec<RClass>,
    pub phys: Vec<u32>,
    vnext: u32,
    snext: u32,
    anext: u32,
}

impl Samples {
    pub fn new() -> Samples {
        Samples { vregs: vec![], phys: vec![], vnext: 8, snext: 8, anext: 0 }
    }

    /// Restart physical numbering (virtual indices stay unique).
    pub fn reset_phys(&mut self) {
        self.vnext = 8;
        self.snext = 8;
        self.anext = 0;
    }

    pub fn reg(&mut self, c: RClass) -> VReg {
        let n = c.dwords();
        let p = if c.is_agpr() {
            self.anext = (self.anext + n - 1) / n * n;
            let p = AGPR_TAG + self.anext;
            self.anext += n;
            p
        } else if c.is_vector() {
            self.vnext = (self.vnext + n - 1) / n * n;
            let p = self.vnext;
            self.vnext += n;
            p
        } else {
            self.snext = (self.snext + n - 1) / n * n;
            let p = self.snext;
            self.snext += n;
            p
        };
        self.vregs.push(c);
        self.phys.push(p);
        (self.vregs.len() - 1) as u32
    }

    /// Instruction for `op` following the row signature; `link = (src slot, vreg)` substitutes
    /// an existing vreg for that source (to create a data dependency).
    pub fn inst(&mut self, op: MOp, row: &Row, link: Option<(usize, VReg)>) -> MInst {
        let mut inst = MInst::new(op);
        if let Some(c) = row.dst {
            inst.dst = Some(self.reg(c));
        }
        for (k, s) in row.srcs.iter().enumerate() {
            match s {
                SigClass::Reg(c) => {
                    let r = match link {
                        Some((slot, v)) if slot == k => v,
                        _ => self.reg(*c),
                    };
                    inst.srcs.push(Src::Reg(r));
                }
                SigClass::Imm => inst.srcs.push(Src::Imm(3)),
            }
        }
        inst.imm = match op {
            MOp::VWriteSub | MOp::VReadSub => 1,
            MOp::SWaitVm | MOp::SWaitLgkm => 3,
            MOp::VReadPhys | MOp::SReadPhys | MOp::VMfmaScale16x16x128Fp8 => 0,
            _ => 16,
        };
        inst
    }

    pub fn alloc(&self) -> amdgpu_compiler::regalloc::Alloc {
        amdgpu_compiler::regalloc::Alloc { phys: self.phys.clone(), max_v: 256, max_s: 100, scalar: self.vregs.iter().map(|c| c.is_scalar()).collect() }
    }
}

pub fn func(name: &str, blocks: Vec<Block>, vregs: Vec<RClass>) -> MFunction {
    MFunction {
        name: name.into(),
        args: vec![],
        kernarg_size: 0,
        workgroup_size: 256,
        lds_bytes: 0,
        blocks,
        vregs,
        loops: vec![],
        alias: std::collections::HashMap::new(),
        elide_fallthrough: false, edge_pad: false,
        agpr_pin: Default::default(),
    }
}