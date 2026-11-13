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

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
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
    /// Explicit combined `s_waitcnt vmcnt(vm) lgkmcnt(lgkm)`.
    WaitBoth(i64, i64),
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
    /// Pinned (hand-authored) instruction: never reordered, see `MInst::pin`.
    pub pinned: bool,
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
    let mut e = effects_core(m, inst);
    e.pinned = inst.pin;
    e
}

fn effects_core(m: &MFunction, inst: &MInst) -> Effects {
    use crate::op_attrs::{Base, FenceAttr, RegFx};
    let at = crate::op_attrs::attrs(inst.op);
    let mut e = Effects::default();
    let fence = match at.fence {
        FenceAttr::None => Fence::None,
        FenceAttr::Nop => Fence::Nop,
        FenceAttr::WaitVm => Fence::WaitVm(inst.imm),
        FenceAttr::WaitLgkm => Fence::WaitLgkm(inst.imm),
        FenceAttr::WaitBoth => Fence::WaitBoth(inst.imm & 0xff, inst.imm >> 8),
        FenceAttr::Barrier => Fence::Barrier,
        FenceAttr::SetPrio => Fence::SetPrio,
    };
    // fence-only instructions (nops, waits, barriers, setprio, endpgm): no register effects
    if at.fence_only {
        e.fence = fence;
        return e;
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
    match at.regfx {
        RegFx::Plain => {}
        RegFx::SubWrite => {
            if let Some(d) = inst.dst {
                let sl = reg_slots(m, d);
                e.writes = vec![sl[(inst.imm as usize).min(sl.len() - 1)]];
            }
        }
        RegFx::SubWrite2 => {
            if let Some(d) = inst.dst {
                let sl = reg_slots(m, d);
                let k = (inst.imm as usize).min(sl.len().saturating_sub(2));
                e.writes = sl[k..(k + 2).min(sl.len())].to_vec();
            }
        }
        RegFx::SubRead => {
            if let Some(r) = src_reg(0) {
                let sl = reg_slots(m, r);
                e.reads = vec![sl[(inst.imm as usize).min(sl.len() - 1)]];
            }
        }
        // Partial (16-bit) writers: the other half of the destination survives, so the old value is read.
        RegFx::PartialWrite => {
            if let Some(d) = inst.dst {
                for x in reg_slots(m, d) {
                    push_unique(&mut e.reads, x);
                }
            }
        }
        RegFx::SwapSrc01 => {
            for k in 0..2 {
                if let Some(r) = src_reg(k) {
                    for x in reg_slots(m, r) {
                        push_unique(&mut e.writes, x);
                    }
                }
            }
        }
    }
    if at.vcc_read {
        push_unique(&mut e.reads, VCC);
    }
    if at.vcc_write {
        push_unique(&mut e.writes, VCC);
    }
    if at.scc_read {
        push_unique(&mut e.reads, SCC);
    }
    if at.scc_write {
        push_unique(&mut e.writes, SCC);
    }
    if let Some(ma) = at.mem {
        let base = match ma.base {
            Base::Src(k) => src_reg(k).map(|r| reg_slots(m, r)[0]),
            Base::Unknown => None,
        };
        e.mem = Some(MemAccess { space: ma.space, write: ma.write, base, off: inst.imm, size: ma.size });
    }
    e.fence = fence;
    e.smem = at.smem;
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