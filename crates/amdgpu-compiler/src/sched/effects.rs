//! Per-instruction effects used by the scheduler: register dwords read/written (sub-register
//! precise), implicit `vcc`/`scc`, memory access descriptors and fence classes.
//!
//! Register slots: every vector/scalar virtual register is mapped to its canonical tuple and a
//! dword offset (`MFunction::alias`), so writes to disjoint dword ranges of one tuple are
//! independent. `vcc` and `scc` are two extra slots.

use crate::mir::*;

pub type Slot = u32;
pub const VCC: Slot = u32::MAX;
pub const SCC: Slot = u32::MAX - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Lds,
    Global,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemAccess {
    pub space: Space,
    pub write: bool,
    /// First dword slot of the address register, when known.
    pub base: Option<Slot>,
    pub off: i64,
    pub size: i64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Fence {
    #[default]
    None,
    Barrier,
    SetPrio,
    /// Explicit `s_waitcnt lgkmcnt(n)`.
    WaitLgkm(i64),
    /// Explicit `s_waitcnt vmcnt(n)`.
    WaitVm(i64),
    Nop,
}

#[derive(Clone, Debug, Default)]
pub struct Effects {
    pub reads: Vec<Slot>,
    pub writes: Vec<Slot>,
    pub mem: Option<MemAccess>,
    pub fence: Fence,
    /// Scalar memory load (kernarg): returns through lgkmcnt but touches no shared state.
    pub smem: bool,
}

/// Dword slots occupied by `r` (an alias maps into its tuple).
pub fn reg_slots(m: &MFunction, r: VReg) -> Vec<Slot> {
    let (t, off) = m.alias.get(&r).map(|a| (a.0, a.1 as u32)).unwrap_or((r, 0));
    let w = m.vreg_class(r).dwords();
    (0..w).map(|k| t * 16 + off + k).collect()
}

fn push_unique(v: &mut Vec<Slot>, s: Slot) {
    if !v.contains(&s) {
        v.push(s);
    }
}

pub fn effects(m: &MFunction, inst: &MInst) -> Effects {
    let mut e = Effects::default();
    match inst.op {
        MOp::SNop | MOp::SEndpgm => {
            e.fence = Fence::Nop;
            return e;
        }
        MOp::SWaitVm => {
            e.fence = Fence::WaitVm(inst.imm);
            return e;
        }
        MOp::SWaitLgkm => {
            e.fence = Fence::WaitLgkm(inst.imm);
            return e;
        }
        MOp::SBarrier => {
            e.fence = Fence::Barrier;
            return e;
        }
        MOp::SSetPrio => {
            e.fence = Fence::SetPrio;
            return e;
        }
        _ => {}
    }
    for s in &inst.srcs {
        if let Src::Reg(r) = s {
            for x in reg_slots(m, *r) {
                push_unique(&mut e.reads, x);
            }
        }
    }
    if let Some(d) = inst.dst {
        for x in reg_slots(m, d) {
            push_unique(&mut e.writes, x);
        }
    }
    let src_reg = |k: usize| -> Option<VReg> {
        match inst.srcs.get(k) {
            Some(Src::Reg(r)) => Some(*r),
            _ => None,
        }
    };
    let base_slot = |k: usize| -> Option<Slot> { src_reg(k).map(|r| reg_slots(m, r)[0]) };
    match inst.op {
        MOp::VWriteSub => {
            if let Some(d) = inst.dst {
                let sl = reg_slots(m, d);
                e.writes = vec![sl[(inst.imm as usize).min(sl.len() - 1)]];
            }
        }
        MOp::VReadSub => {
            if let Some(r) = src_reg(0) {
                let sl = reg_slots(m, r);
                e.reads = vec![sl[(inst.imm as usize).min(sl.len() - 1)]];
            }
        }
        // Partial (16-bit) writers: the other half of the destination survives, so the old value is read.
        MOp::VCvtPkFp8F32 | MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi => {
            if let Some(d) = inst.dst {
                for x in reg_slots(m, d) {
                    push_unique(&mut e.reads, x);
                }
            }
        }
        MOp::VPermlane32Swap => {
            for k in 0..2 {
                if let Some(r) = src_reg(k) {
                    for x in reg_slots(m, r) {
                        push_unique(&mut e.writes, x);
                    }
                }
            }
        }
        MOp::VPtrAdd | MOp::VAddCoU32 => push_unique(&mut e.writes, VCC),
        MOp::VAddcCoU32 => {
            push_unique(&mut e.reads, VCC);
            push_unique(&mut e.writes, VCC);
        }
        MOp::VCmpLtI32 | MOp::VCmpGeI32 | MOp::VCmpGtI32 | MOp::VCmpNeI32 | MOp::VCmpLtF32 => push_unique(&mut e.writes, VCC),
        MOp::VCndmask => push_unique(&mut e.reads, VCC),
        MOp::SAddU32 | MOp::SShlB32 | MOp::SLshrB32 | MOp::SAndB32 | MOp::SXorB32 | MOp::SCmpLtI32 => push_unique(&mut e.writes, SCC),
        MOp::SAddcU32 => {
            push_unique(&mut e.reads, SCC);
            push_unique(&mut e.writes, SCC);
        }
        MOp::LdsReadB32 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: false, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::LdsReadB128 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: false, base: base_slot(0), off: inst.imm, size: 16 })
        }
        MOp::LdsReadB64TrB8 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: false, base: base_slot(0), off: inst.imm, size: 8 })
        }
        MOp::LdsWriteB32 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: true, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::LdsWriteB128 => {
            e.mem = Some(MemAccess { space: Space::Lds, write: true, base: base_slot(0), off: inst.imm, size: 16 })
        }
        MOp::GlobalLoadU16 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 2 })
        }
        MOp::GlobalLoadF32 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::GlobalLoadB128S => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(1), off: inst.imm, size: 16 })
        }
        MOp::GlobalLoadF32S => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(1), off: inst.imm, size: 4 })
        }
        MOp::GlobalLoadB128 => {
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 16 })
        }
        MOp::GlobalLoadLdsB128 | MOp::GlobalLoadLdsB32 => {
            // reads global memory and writes LDS through M0: modelled as a global read with a fence-like LDS write
            e.mem = Some(MemAccess { space: Space::Global, write: false, base: base_slot(0), off: inst.imm, size: 16 });
            e.fence = Fence::Barrier;
        }
        MOp::GlobalStoreU16 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 2 })
        }
        MOp::GlobalStoreF32 => {
            e.mem = Some(MemAccess { space: Space::Global, write: true, base: base_slot(0), off: inst.imm, size: 4 })
        }
        MOp::SLoadDword | MOp::SLoadDwordx2 => e.smem = true,
        _ => {}
    }
    e
}

/// Slots read by a block terminator.
pub fn term_reads(t: &Term) -> Vec<Slot> {
    match t {
        Term::CondGoto { on: Pred::Vccnz, .. } | Term::CondGoto { on: Pred::Vccz, .. } => vec![VCC],
        Term::CondGoto { .. } => vec![SCC],
        _ => Vec::new(),
    }
}