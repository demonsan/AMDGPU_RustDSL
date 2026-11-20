//! gfx1250 software hazards (docs/gfx1250-hazards.md): the TRANS co-execution rule, the WMMA co-execution windows and the TDM drain, on the
//! allocated MIR. gfx1250 interlocks every other VALU dependency (no `s_nop` wait states: the shared pass's `s_nop`s are dropped for it).
//! Distances count VALU instructions (`v_nop` included) along every path: a forward dataflow over the blocks with the pending producers
//! (each with the VALUs issued since) merged conservatively (the smaller distance wins).

use crate::mir::*;
use crate::regalloc::Alloc;
use crate::sched::effects::{effects, VCC};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
struct Producer {
    /// None: TRANS; Some((next-WMMA window, next-VALU window)): an XDL WMMA
    wmma: Option<(u32, u32)>,
    dst: Vec<u32>,
    srcs: Vec<u32>,
    /// for a WMMA consumer: its A / B registers are `srcs[..ab]`
    dist: u32,
}

/// The co-execution windows of a gfx1250 WMMA (next XDL WMMA reading D as A / B, next other VALU), by its latency category.
pub fn wmma_windows(op: MOp) -> Option<(u32, u32)> {
    Some(match op {
        // latency 4: 16x16x64 FP8 / BF8
        MOp::VWmmaF32_16x16x64Fp8 | MOp::VWmmaF32_16x16x64Fp8Z => (2, 1),
        // latency 8: 16x16x128 FP8, 32x16x128 F4 (scaled)
        MOp::VWmmaF32_16x16x128Fp8 | MOp::VWmmaF32_16x16x128Fp8Z | MOp::VWmmaScaleF32_32x16x128F4 | MOp::VWmmaScaleF32_32x16x128F4Z => (5, 4),
        _ => return None,
    })
}

fn keys(m: &MFunction, a: &Alloc, slots: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &s in slots {
        if s == VCC || s >= u32::MAX - 1 {
            continue;
        }
        let (t, off) = (s / 16, s % 16);
        let Some(&p) = a.phys.get(t as usize) else { continue };
        let c = m.vreg_class(t);
        if c.is_vector() {
            out.push(p + off);
        }
    }
    out
}

fn regs_of(m: &MFunction, a: &Alloc, r: VReg) -> Vec<u32> {
    keys(m, a, &crate::sched::effects::reg_slots(m, r))
}

fn is_valu(op: MOp) -> bool {
    let mn = crate::op_attrs::attrs(op).mnemonic;
    mn.starts_with("v_") && !matches!(op, MOp::VReadPhys)
}

#[derive(Default)]
pub struct Plan {
    /// `v_nop` count / extra lines before instruction (block, index)
    pub before: HashMap<(usize, usize), Vec<String>>,
}

pub fn plan(m: &MFunction, a: &Alloc) -> Plan {
    let n = m.blocks.len();
    let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let succ: Vec<Vec<usize>> = m
        .blocks
        .iter()
        .map(|b| match &b.term {
            Term::Goto(t) => vec![idx[t]],
            Term::CondGoto { target, fall, .. } => vec![idx[target], idx[fall]],
            Term::Endpgm => vec![],
        })
        .collect();
    let has_tdm = m.blocks.iter().flat_map(|b| b.insts.iter()).any(|i| matches!(i.op, MOp::TensorLoadToLds | MOp::TensorStoreFromLds));
    // entry states: pending producers, and "a tensor op may be in flight beyond the drain bound"
    let mut entry: Vec<Option<(Vec<Producer>, u32)>> = vec![None; n];
    entry[0] = Some((Vec::new(), 0));
    let mut plan = Plan::default();
    // VOPD: a pair issues as ONE VALU instruction (its second instruction sees the distances of the first and adds none); a pair whose second
    // instruction would need padding is broken (the plan is then recomputed with the two instructions issued separately)
    let mut broken = false;
    for _round in 0..64 {
        let mut changed = false;
        plan.before.clear();
        for bi in 0..n {
            let Some((mut pend, mut tdm)) = entry[bi].clone() else { continue };
            for (ii, inst) in m.blocks[bi].insts.iter().enumerate() {
                let mut lines = Vec::new();
                // TDM drain: at most 10 tensor operations outstanding before another one
                if matches!(inst.op, MOp::TensorLoadToLds | MOp::TensorStoreFromLds) {
                    if tdm > 10 {
                        lines.push("s_wait_tensorcnt 0xa".to_string());
                        tdm = 10;
                    }
                    tdm += 1;
                }
                if inst.op == MOp::SWaitTensor || (inst.op == MOp::SWaitCtr && inst.imm >> 8 == 6) {
                    tdm = tdm.min((inst.imm & 0xff) as u32);
                }
                let second = super::vopd::is_second(bi, ii);
                if second && is_valu(inst.op) {
                    for p in pend.iter_mut() {
                        p.dist = p.dist.saturating_sub(1);
                    }
                }
                if is_valu(inst.op) {
                    let e = effects(m, inst);
                    let reads = keys(m, a, &e.reads);
                    let writes = keys(m, a, &e.writes);
                    let me_wmma = wmma_windows(inst.op).is_some();
                    let me_trans = crate::op_attrs::attrs(inst.op).transcendental;
                    let ab: Vec<u32> = if me_wmma {
                        inst.srcs.iter().take(2).filter_map(|s| if let Src::Reg(r) = s { Some(regs_of(m, a, *r)) } else { None }).flatten().collect()
                    } else {
                        Vec::new()
                    };
                    let mut need = 0u32;
                    for p in &pend {
                        let hit = |xs: &[u32], ys: &[u32]| xs.iter().any(|x| ys.contains(x));
                        match p.wmma {
                            None if !me_trans && p.dist == 0 && ((hit(&reads, &p.dst) && !trans_coexec_proven(inst.op, false)) || (hit(&writes, &p.srcs) && !trans_coexec_proven(inst.op, true))) => need = need.max(1),
                            Some((ww, _)) if me_wmma => {
                                if hit(&ab, &p.dst) && p.dist < ww {
                                    need = need.max(ww - p.dist);
                                }
                            }
                            Some((_, vw)) => {
                                if p.dist < vw && (hit(&reads, &p.dst) || hit(&writes, &p.dst) || hit(&writes, &p.srcs)) {
                                    need = need.max(vw - p.dist);
                                }
                            }
                            None => {}
                        }
                    }
                    if need > 0 && second {
                        super::vopd::break_pair(bi, ii - 1);
                        broken = true;
                    }
                    if need > 0 {
                        lines.push(if need == 1 { "v_nop".to_string() } else { format!("v_nop ; x{need}") });
                        for p in pend.iter_mut() {
                            p.dist += need;
                        }
                    }
                    for p in pend.iter_mut() {
                        p.dist += 1;
                    }
                    pend.retain(|p| p.dist < p.wmma.map_or(1, |(w, v)| w.max(v)));
                    if me_trans || me_wmma {
                        pend.push(Producer { wmma: wmma_windows(inst.op), dst: writes.clone(), srcs: reads.clone(), dist: 0 });
                    }
                }
                if !lines.is_empty() {
                    // a v_nop run is spelled once per instruction
                    let mut out = Vec::new();
                    for l in lines {
                        if let Some(k) = l.strip_prefix("v_nop ; x") {
                            for _ in 0..k.parse::<u32>().unwrap_or(1) {
                                out.push("v_nop".to_string());
                            }
                        } else {
                            out.push(l);
                        }
                    }
                    plan.before.insert((bi, ii), out);
                }
            }
            for &s in &succ[bi] {
                let merged = match &entry[s] {
                    None => (pend.clone(), tdm),
                    Some((q, t)) => {
                        let mut r = q.clone();
                        for p in &pend {
                            match r.iter_mut().find(|x| x.wmma == p.wmma && x.dst == p.dst && x.srcs == p.srcs) {
                                Some(x) => x.dist = x.dist.min(p.dist),
                                None => r.push(p.clone()),
                            }
                        }
                        (r, (*t).max(tdm))
                    }
                };
                if entry[s].as_ref().map_or(true, |e| e.0 != merged.0 || e.1 != merged.1) {
                    entry[s] = Some(merged);
                    changed = true;
                }
            }
        }
        if !changed && !broken {
            break;
        }
        if broken {
            // restart from scratch with the broken pairs issued separately
            broken = false;
            entry = vec![None; n];
            entry[0] = Some((Vec::new(), 0));
        }
    }
    let _ = has_tdm;
    plan
}

/// The VALU readers for which the TRANS co-execution `v_nop` is not needed on gfx1250, by hardware evidence (MI450,
/// tools/mi450/trans_all.sh: every TRANS op - exp / log / rcp / rsq / sqrt / sin / cos - read at distance 0, RAW in every source position and
/// WAR, by these classes, bit-identical to the padded build over 5 runs x 1024 workgroups x 512 work-items; docs/gfx1250-hazards.md). LLVM
/// (fixVALUTransCoexecutionHazards) pads every VALU; every class not listed here (DOT, conversions, moves, ...) keeps the `v_nop`.
/// `war`: the reader overwrites the TRANS op's source (else it reads the TRANS result). The second evidence set (`probe_g1250_trans_rd2`:
/// `v_sub_f32` in both source positions, `v_cvt_pk_fp8_f32` as the low / the high source; 84 / 83 sites) covers RAW only.
pub fn trans_coexec_proven(op: MOp, war: bool) -> bool {
    // (`AMDC_TRANS_PROVEN_OLD`: the first evidence set only, for A/B timing)
    let second = !war && std::env::var("AMDC_TRANS_PROVEN_OLD").is_err() && matches!(op, MOp::VSubF32 | MOp::VCvtPkFp8F32 | MOp::VCvtPkFp8F32Hi | MOp::VCvtPkFp8F32Half);
    second || matches!(op, MOp::VAddF32 | MOp::VMulF32 | MOp::VFmaF32 | MOp::VMaxF32 | MOp::VPkAddF32)
}
