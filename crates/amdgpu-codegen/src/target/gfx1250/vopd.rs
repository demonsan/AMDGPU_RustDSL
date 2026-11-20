//! VOPD dual issue (gfx1250, opt-in `AMDC_VOPD`): two adjacent independent VALU instructions of the dual-issue opcode sets become one
//! `v_dual_<x> .. :: v_dual_<y> ..` instruction (measured on MI450: 16 VOPD pairs run 32 `v_fmac_f32` at 0.77 vs 1.30 cycles per fmac per
//! wave, tools/mi450/vopd_bench.sh).
//!
//! Rules (public LLVM GCNVOPDUtils.cpp / AMDGPUBaseInfo.cpp, VOPDInstructions.td for gfx1250):
//! * X in {mul, add, sub, subrev, max_num, min_num (f32), mov_b32}; Y additionally {add_nc_u32, sub_nc_u32} (the other X / Y forms are not
//!   produced here);
//! * the destinations differ in parity; each source slot (src0, vsrc1) of X and Y in different banks (register & 3) unless the same
//!   register; vsrc1 a VGPR; no literal; the same VGPR window (MSBs) per slot;
//! * Y does not read X's destination (both read their sources before either writes; X is the earlier instruction).
//!
//! Pairs are chosen on the allocated MIR before the hazard plan (which counts a pair as one VALU issue and breaks a pair whose second
//! instruction would need padding); the emitter re-checks the final text and emits the two instructions separately whenever a rule
//! fails there or a wait / pad lands between them (separate issue only lengthens the hazard distances, so that is always safe).

use crate::mir::*;
use crate::regalloc::Alloc;
use std::cell::RefCell;
use std::collections::HashSet;

thread_local! {
    /// (block, index of the FIRST instruction) of every chosen pair; the second is index + 1
    static PAIRS: RefCell<HashSet<(usize, usize)>> = RefCell::new(HashSet::new());
}

/// The kernel's `vopd` lowering token, or `AMDC_VOPD` (A/B).
pub fn enabled(m: &MFunction) -> bool {
    // (AMDC_NO_VOPD: off even for a kernel with the token, A/B only)
    std::env::var("AMDC_NO_VOPD").is_err() && (m.vopd || std::env::var("AMDC_VOPD").is_ok())
}

pub fn clear() {
    PAIRS.with(|p| p.borrow_mut().clear());
}

pub fn is_first(b: usize, i: usize) -> bool {
    PAIRS.with(|p| p.borrow().contains(&(b, i)))
}

pub fn is_second(b: usize, i: usize) -> bool {
    i > 0 && is_first(b, i - 1)
}

pub fn break_pair(b: usize, first: usize) {
    PAIRS.with(|p| p.borrow_mut().remove(&(b, first)));
}

/// (is an X component, is a Y component)
fn kind(op: MOp) -> Option<(bool, bool)> {
    Some(match op {
        MOp::VMulF32 | MOp::VAddF32 | MOp::VSubF32 | MOp::VMaxF32 | MOp::VMinF32 | MOp::VMovB32 | MOp::VMovFromS => (true, true),
        MOp::VAddU32 | MOp::VSubU32 => (false, true),
        _ => return None,
    })
}

fn phys(m: &MFunction, a: &Alloc, r: VReg) -> Option<(bool, u32, u32)> {
    // (the allocation's own register of the value: a tuple's sub-register alias has its dword's register, as the emitter prints it)
    let p = *a.phys.get(r as usize)?;
    if p == u32::MAX {
        return None;
    }
    Some((a.scalar.get(r as usize).copied().unwrap_or(false), p, m.vreg_class(r).dwords()))
}

/// Choose the pairs of `m` (adjacent instructions only: the pairing potential measured on the MLA kernels is all adjacent,
/// tools/mi450/vopd_estimate.py).
pub fn select(m: &MFunction, a: &Alloc) {
    clear();
    if !enabled(m) || m.exact {
        return;
    }
    let ok1 = |i: &MInst| -> bool {
        kind(i.op).is_some() && !i.pin && i.neg == 0 && !i.e64 && i.words.is_none() && i.dst.is_some() && i.srcs.len() <= 2
    };
    PAIRS.with(|p| {
        let mut p = p.borrow_mut();
        for (bi, b) in m.blocks.iter().enumerate() {
            let mut i = 0;
            while i + 1 < b.insts.len() {
                let (x, y) = (&b.insts[i], &b.insts[i + 1]);
                let good = ok1(x) && ok1(y) && kind(x.op).unwrap().0 && kind(y.op).unwrap().1 && {
                    let dx = phys(m, a, x.dst.unwrap());
                    let dy = phys(m, a, y.dst.unwrap());
                    match (dx, dy) {
                        (Some((false, px, 1)), Some((false, py, 1))) => {
                            // parity of the destinations, Y independent of X's destination
                            let reads_x = y.srcs.iter().any(|s| if let Src::Reg(r) = s { phys(m, a, *r).map_or(true, |(sc, q, w)| !sc && q <= px && px < q + w) } else { false });
                            (px & 1) != (py & 1) && px / 256 == py / 256 && !reads_x && px != py
                        }
                        _ => false,
                    }
                };
                if good {
                    p.insert((bi, i));
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
    });
}

/// A VGPR operand token: its register number (absolute, the text before the window pass).
fn vgpr(t: &str) -> Option<u32> {
    t.trim().strip_prefix('v').and_then(|x| x.parse::<u32>().ok())
}

fn is_inline(t: &str) -> bool {
    let t = t.trim();
    if t.starts_with('s') || t.starts_with("vcc") || t.starts_with("exec") || t.starts_with("m0") || t.starts_with("null") {
        return true;
    }
    if let Ok(v) = t.parse::<i64>() {
        return (-16..=64).contains(&v);
    }
    matches!(t, "0.5" | "-0.5" | "1.0" | "-1.0" | "2.0" | "-2.0" | "4.0" | "-4.0")
}

/// The dual form of two emitted (gfx1250-spelled, absolute-register) lines, if every VOPD rule holds on the final text.
pub fn merge(x: &str, y: &str) -> Option<String> {
    let parse = |l: &str| -> Option<(String, Vec<String>)> {
        let l = l.trim();
        let (mn, rest) = l.split_once(' ')?;
        let base = mn.strip_suffix("_e32")?;
        let ops: Vec<String> = rest.split(", ").map(|s| s.trim().to_string()).collect();
        if ops.iter().any(|o| o.contains(' ') || o.starts_with('-') || o.starts_with('|')) {
            return None;
        }
        Some((base.to_string(), ops))
    };
    let (mx, ox) = parse(x)?;
    let (my, oy) = parse(y)?;
    let xs = ["v_mul_f32", "v_add_f32", "v_sub_f32", "v_max_num_f32", "v_min_num_f32", "v_mov_b32"];
    let ys = ["v_mul_f32", "v_add_f32", "v_sub_f32", "v_max_num_f32", "v_min_num_f32", "v_mov_b32", "v_add_nc_u32", "v_sub_nc_u32"];
    if !xs.contains(&mx.as_str()) || !ys.contains(&my.as_str()) {
        return None;
    }
    let want = |mn: &str, ops: &Vec<String>| if mn == "v_mov_b32" { ops.len() == 2 } else { ops.len() == 3 };
    if !want(&mx, &ox) || !want(&my, &oy) {
        return None;
    }
    let (dx, dy) = (vgpr(&ox[0])?, vgpr(&oy[0])?);
    if (dx & 1) == (dy & 1) || dx / 256 != dy / 256 {
        return None;
    }
    // src0: a VGPR (bank / window rule) or an SGPR / inline constant (no literal)
    let s0 = |o: &str| -> Option<Option<u32>> {
        match vgpr(o) {
            Some(r) => Some(Some(r)),
            None if is_inline(o) => Some(None),
            None => None,
        }
    };
    let (s0x, s0y) = (s0(&ox[1])?, s0(&oy[1])?);
    if let (Some(a), Some(b)) = (s0x, s0y) {
        if a / 256 != b / 256 || (a != b && (a & 3) == (b & 3)) {
            return None;
        }
    }
    let s1 = |mn: &str, ops: &Vec<String>| -> Option<Option<u32>> { if mn == "v_mov_b32" { Some(None) } else { vgpr(&ops[2]).map(Some) } };
    let (s1x, s1y) = (s1(&mx, &ox)?, s1(&my, &oy)?);
    if let (Some(a), Some(b)) = (s1x, s1y) {
        if a / 256 != b / 256 || (a != b && (a & 3) == (b & 3)) {
            return None;
        }
    }
    // Y must not read X's destination
    if [s0y, s1y].iter().flatten().any(|r| *r == dx) {
        return None;
    }
    let dual = |mn: &str, ops: &Vec<String>| format!("v_dual_{} {}", mn.trim_start_matches("v_"), ops.join(", "));
    Some(format!("{} :: {}", dual(&mx, &ox), dual(&my, &oy)))
}
