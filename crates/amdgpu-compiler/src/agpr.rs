//! AGPR assignment: moves the vregs the kernel asked for (`MFunction::agpr_pin`: DSL `pin_agpr` / `agpr`) into the accumulation-register
//! file and repairs every instruction that cannot touch it.
//!
//! gfx950 (unified 512-register file): an AGPR can be written by MFMA (`acc_cd`), by `ds_read` / `global_load` / `buffer_load`, by
//! `v_accvgpr_write/mov`, and read by MFMA (srcA/srcB/srcC), `ds_write`, `global_store` / `buffer_store` and `v_accvgpr_read/mov` - never by
//! VALU (assembler-verified, docs/agpr.md). MFMA needs vdst and srcC in the SAME file. Everything else is rewritten here:
//!  * `VMovB32` / `VWriteSub` / `VReadSub` that touch an AGPR become `v_accvgpr_write/mov/read` (an inline constant when it fits, else through a VGPR);
//!  * any other reader of an AGPR gets the dwords copied into a fresh VGPR tuple with `v_accvgpr_read_b32` right before it;
//!  * any other writer of an AGPR writes a fresh VGPR tuple that is copied with `v_accvgpr_write_b32` right after it (read-modify-write converts are rejected);
//!  * an MFMA whose srcC is in the other file than its destination gets srcC copied into a temporary of the destination's file.
//! The pass runs on the virtual-register MIR before scheduling, so the scheduler, hazard pass and allocator see the final instruction forms.

use std::collections::HashMap;

use crate::mir::*;

/// Marker in `MFunction::agpr_pin`: "put this vreg in the AGPR file, any placement".
pub const AGPR_ANY: u32 = u32::MAX;

#[derive(Debug, Clone, Default)]
pub struct AgprStats {
    pub vregs: usize,
    pub reads: usize,
    pub writes: usize,
    pub movs: usize,
}

pub fn legal_src(op: MOp, k: usize) -> bool {
    use MOp::*;
    match op {
        LdsWriteB32 | LdsWriteB128 => k == 1,
        GlobalStoreU16 | GlobalStoreF32 | GlobalStoreF32x4 | GlobalStoreF32x2 | GlobalStoreF32Sc1 | GlobalStoreF32x4Sc1 => k == 1,
        BufferStoreDwordX1 | BufferStoreDwordX2 | BufferStoreDwordX4 => k == 0,
        VMfma16x16x16F16 | VMfma32x32x8F16 | VMfma16x16x32Fp8 | VMfma32x32x16Fp8 | VMfmaScale32x32x64Fp8 | VMfma32x32x64Fp8 | VMfmaScale16x16x128Fp8 => k <= 2,
        VMfma32x32x8F16Z | VMfmaScale32x32x64Fp8Z | VMfma32x32x64Fp8Z => k <= 1,
        VAccvgprRead | VAccvgprMov => k == 0,
        _ => false,
    }
}

pub fn is_mfma(op: MOp) -> bool {
    use MOp::*;
    matches!(
        op,
        VMfma16x16x16F16 | VMfma32x32x8F16 | VMfma32x32x8F16Z | VMfma16x16x32Fp8 | VMfma32x32x16Fp8 | VMfmaScale32x32x64Fp8 | VMfmaScale32x32x64Fp8Z | VMfma32x32x64Fp8 | VMfma32x32x64Fp8Z | VMfmaScale16x16x128Fp8
    )
}

pub fn legal_dst(op: MOp) -> bool {
    use MOp::*;
    is_mfma(op)
        || matches!(
            op,
            LdsReadB32
                | LdsReadB128
                | LdsReadB64TrB8
                | GlobalLoadU16
                | GlobalLoadF32
                | GlobalLoadB128
                | GlobalLoadB128S
                | GlobalLoadF32S
                | GlobalLoadF32Sc1
                | GlobalLoadB128Sc1
                | BufferLoadDwordX1
                | BufferLoadDwordX2
                | BufferLoadDwordX4
                | VAccvgprWrite
                | VAccvgprMov
        )
}

/// Ops whose destination is also an input without listing it as a source (16-bit half writers).
fn rmw_dst(op: MOp) -> bool {
    matches!(op, MOp::VCvtPkFp8F32 | MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi | MOp::VPermlane32Swap)
}

struct Cx<'a> {
    m: &'a mut MFunction,
    stats: AgprStats,
    alias_cache: HashMap<(VReg, u8, bool), VReg>,
}

impl<'a> Cx<'a> {
    fn is_a(&self, r: VReg) -> bool {
        self.m.vreg_class(r).is_agpr()
    }

    fn new_vreg(&mut self, c: RClass) -> VReg {
        self.m.vregs.push(c);
        (self.m.vregs.len() - 1) as VReg
    }

    /// Dword `k` of register `r` (a tuple or an alias) as a one-dword vreg of the file of `r`.
    fn dword(&mut self, r: VReg, k: u32) -> VReg {
        let agpr = self.m.vreg_class(r).is_agpr();
        let (t, base) = self.m.alias.get(&r).map(|a| (a.0, a.1)).unwrap_or((r, 0));
        let off = base + k as u8;
        if self.m.vreg_class(r).dwords() == 1 && k == 0 {
            return r;
        }
        if let Some(a) = self.alias_cache.get(&(t, off, agpr)) {
            return *a;
        }
        let c = if agpr { RClass::A1 } else { RClass::V1 };
        let a = self.new_vreg(c);
        self.m.alias.insert(a, (t, off));
        self.alias_cache.insert((t, off, agpr), a);
        a
    }

    fn read_into(&mut self, out: &mut Vec<MInst>, a: VReg) -> VReg {
        let w = self.m.vreg_class(a).dwords();
        let class = self.m.vreg_class(a).to_vgpr().unwrap();
        let t = self.new_vreg(class);
        for k in 0..w {
            let s = self.dword(a, k);
            let d = self.dword(t, k);
            out.push(MInst::new(MOp::VAccvgprRead).def(d).srcreg(s));
            self.stats.reads += 1;
        }
        t
    }

    fn write_from(&mut self, out: &mut Vec<MInst>, a: VReg, v: VReg) {
        let w = self.m.vreg_class(a).dwords();
        for k in 0..w {
            let d = self.dword(a, k);
            let s = self.dword(v, k);
            out.push(MInst::new(MOp::VAccvgprWrite).def(d).srcreg(s));
            self.stats.writes += 1;
        }
    }

    /// `dst(A1) <- imm`: inline constant when it fits (VOP3 has no literal), else through a VGPR.
    fn write_imm(&mut self, out: &mut Vec<MInst>, dst: VReg, bits: i64) {
        let v = bits as i32;
        if (-16..=64).contains(&v) {
            out.push(MInst::new(MOp::VAccvgprWrite).def(dst).src(Src::Imm(v as i64)));
        } else {
            let t = self.new_vreg(RClass::V1);
            out.push(MInst::new(MOp::VMovB32).def(t).imm(bits));
            out.push(MInst::new(MOp::VAccvgprWrite).def(dst).srcreg(t));
        }
        self.stats.writes += 1;
    }
}

pub fn assign(m: &mut MFunction) -> Result<AgprStats, String> {
    let reqs: Vec<(VReg, u32)> = m.agpr_pin.iter().map(|(r, p)| (*r, *p)).collect();
    if reqs.is_empty() {
        return Ok(AgprStats::default());
    }
    // 1. classes: every requested canonical tuple and all its alias views move to the AGPR file
    let mut stats = AgprStats::default();
    let mut canon_set: std::collections::HashSet<VReg> = std::collections::HashSet::new();
    for (r, _) in &reqs {
        let c = m.canon(*r);
        let cls = m.vreg_class(c);
        if cls.is_scalar() {
            return Err(format!("AGPR request on scalar vreg r{r}"));
        }
        if cls.is_vector() {
            canon_set.insert(c);
        }
    }
    for c in &canon_set {
        m.vregs[*c as usize] = m.vregs[*c as usize].to_agpr().unwrap();
        stats.vregs += 1;
    }
    let aliases: Vec<VReg> = m.alias.keys().copied().collect();
    for a in aliases {
        let t = m.alias[&a].0;
        if canon_set.contains(&t) && m.vregs[a as usize].is_vector() {
            m.vregs[a as usize] = m.vregs[a as usize].to_agpr().unwrap();
        }
    }
    // explicit pins stay (RA honours them); "any" requests are done
    let pins: HashMap<VReg, u32> = reqs.iter().filter(|(_, p)| *p != AGPR_ANY).map(|(r, p)| (m.canon(*r), *p)).collect();
    m.agpr_pin = pins;

    // 2. repair instructions
    let blocks = std::mem::take(&mut m.blocks);
    let mut cx = Cx { m, stats, alias_cache: HashMap::new() };
    let mut new_blocks = Vec::with_capacity(blocks.len());
    for (bi, mut b) in blocks.into_iter().enumerate() {
        let mut out: Vec<MInst> = Vec::with_capacity(b.insts.len() + 16);
        for (ii, inst) in b.insts.drain(..).enumerate() {
            let touches_a = inst.dst.map_or(false, |d| cx.is_a(d)) || inst.uses().iter().any(|r| cx.is_a(*r));
            if !touches_a {
                out.push(inst);
                continue;
            }
            let opname = inst.op;
            let ctx = move |e: String| format!("block {bi} inst {ii} ({:?}): {e}", opname);
            match inst.op {
                MOp::VMovB32 => {
                    let d = inst.dst.unwrap();
                    let src = inst.srcs.first().copied();
                    if cx.is_a(d) {
                        match src {
                            None => cx.write_imm(&mut out, d, inst.imm),
                            Some(Src::Imm(i)) => cx.write_imm(&mut out, d, i),
                            Some(Src::Reg(r)) if cx.is_a(r) => {
                                out.push(MInst::new(MOp::VAccvgprMov).def(d).srcreg(r));
                                cx.stats.movs += 1;
                            }
                            Some(Src::Reg(r)) => {
                                out.push(MInst::new(MOp::VAccvgprWrite).def(d).srcreg(r));
                                cx.stats.writes += 1;
                            }
                        }
                    } else if let Some(Src::Reg(r)) = src {
                        out.push(MInst::new(MOp::VAccvgprRead).def(d).srcreg(r));
                        cx.stats.reads += 1;
                    } else {
                        return Err(ctx("unexpected mov form".into()));
                    }
                    continue;
                }
                MOp::VWriteSub if inst.dst.map_or(false, |d| cx.is_a(d)) => {
                    let d = inst.dst.unwrap();
                    let lane = cx.dword(d, inst.imm as u32);
                    match inst.srcs.first().copied() {
                        Some(Src::Reg(r)) if cx.is_a(r) => {
                            out.push(MInst::new(MOp::VAccvgprMov).def(lane).srcreg(r));
                            cx.stats.movs += 1;
                        }
                        Some(Src::Reg(r)) => {
                            out.push(MInst::new(MOp::VAccvgprWrite).def(lane).srcreg(r));
                            cx.stats.writes += 1;
                        }
                        Some(Src::Imm(i)) => cx.write_imm(&mut out, lane, i),
                        None => return Err(ctx("VWriteSub without a source".into())),
                    }
                    continue;
                }
                MOp::VReadSub => {
                    if let Some(Src::Reg(r)) = inst.srcs.first().copied() {
                        if cx.is_a(r) {
                            let lane = cx.dword(r, inst.imm as u32);
                            out.push(MInst::new(MOp::VAccvgprRead).def(inst.dst.unwrap()).srcreg(lane));
                            cx.stats.reads += 1;
                            continue;
                        }
                    }
                }
                _ => {}
            }
            let mut inst = inst;
            // MFMA: vdst and srcC in one file
            if is_mfma(inst.op) {
                if let (Some(d), Some(Src::Reg(c))) = (inst.dst, inst.srcs.get(2).copied()) {
                    if inst.op != MOp::VMfmaScale32x32x64Fp8Z && !matches!(inst.op, MOp::VMfma32x32x8F16Z | MOp::VMfma32x32x64Fp8Z) && cx.is_a(d) != cx.is_a(c) {
                        let nc = if cx.is_a(d) {
                            let t = cx.new_vreg(cx.m.vreg_class(c).to_agpr().unwrap());
                            for k in 0..cx.m.vreg_class(c).dwords() {
                                let (td, cs) = (cx.dword(t, k), cx.dword(c, k));
                                out.push(MInst::new(MOp::VAccvgprWrite).def(td).srcreg(cs));
                                cx.stats.writes += 1;
                            }
                            t
                        } else {
                            cx.read_into(&mut out, c)
                        };
                        inst.srcs[2] = Src::Reg(nc);
                    }
                }
            }
            // illegal readers of an AGPR
            let op = inst.op;
            for k in 0..inst.srcs.len() {
                if let Src::Reg(r) = inst.srcs[k] {
                    if cx.is_a(r) && !legal_src(op, k) {
                        let t = cx.read_into(&mut out, r);
                        inst.srcs[k] = Src::Reg(t);
                    }
                }
            }
            // illegal writer of an AGPR
            let mut post: Vec<MInst> = Vec::new();
            if let Some(d) = inst.dst {
                if cx.is_a(d) && !legal_dst(op) {
                    if rmw_dst(op) {
                        return Err(ctx("a read-modify-write converter cannot write an AGPR directly".into()));
                    }
                    let class = cx.m.vreg_class(d).to_vgpr().unwrap();
                    let t = cx.new_vreg(class);
                    inst.dst = Some(t);
                    cx.write_from(&mut post, d, t);
                }
            }
            out.push(inst);
            out.extend(post);
        }
        b.insts = out;
        new_blocks.push(b);
    }
    cx.m.blocks = new_blocks;
    Ok(cx.stats)
}