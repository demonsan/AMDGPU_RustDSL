//! Layout conversion between two thread-value (TV) layouts of the SAME logical tensor (fusion hand-off, docs/fusion-layout-handoff.md).
//!
//! `src` and `dst` map `(thread, value)` (thread = mode 0, value = mode 1, static) to the element index of one tensor. The conversion map
//! `M = dst^-1 o src` says, for every destination slot `(t', v')`, which source slot `(t, v)` holds its element. It is classified by which
//! positions move, from free to expensive:
//! * `Identity`       - same slot everywhere: a view, 0 instructions;
//! * `Register`       - only the value index moves inside each lane, whole registers: renaming / RA coalescing, 0 instructions;
//! * `RegisterBytes`  - inside each lane, but sub-register elements change their byte position: `v_perm_b32` per destination register;
//! * `Quad`           - across lanes inside each lane quad (`t >> 2` kept): `v_mov_b32_dpp quad_perm` (+ `v_cndmask` merges);
//! * `Half`           - lane `t` <-> `t ^ 32` only: `v_permlane32_swap`;
//! * `CrossLane`      - anything else: `ds_bpermute` per register or a wave-private LDS round trip.
//!
//! When all sizes are powers of two the map is also tested for GF(2) linearity on the bits of `(t, v)` (`gf2`): a bit permutation tells exactly
//! which lane bits become value bits (the "which bits move" view of the BitForm normal form).

use crate::layout::Layout;

/// How the elements move between the two layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Movement {
    Identity,
    Register,
    RegisterBytes,
    Quad,
    Half,
    CrossLane,
}

/// The conversion of one fragment.
#[derive(Clone, Debug)]
pub struct ConvPlan {
    pub threads: usize,
    pub values: usize,
    /// elements per 32-bit register (1 for f32, 2 for bf16 pairs, 4 for fp8)
    pub per_reg: usize,
    /// `src_of[t' * values + v'] = (t, v)`
    pub src_of: Vec<(usize, usize)>,
    pub movement: Movement,
    /// the destination register `r'` of every lane reads whole register `reg_map[r']` (valid for `Identity` / `Register`)
    pub reg_map: Option<Vec<usize>>,
    /// GF(2) matrix (column per input bit of `t | v << log2(threads)`) when the map is linear on the index bits
    pub gf2: Option<Vec<u32>>,
    /// estimated instructions per wave (the cost model of the module doc)
    pub cost: usize,
}

fn table(tv: &Layout) -> (usize, usize, Vec<i64>) {
    let t = tv.get(0).size().st() as usize;
    let v = tv.get(1).size().st() as usize;
    // column-major over (thread, value): flat index t + T v
    let tab = (0..t * v).map(|i| tv.at(i as i64)).collect();
    (t, v, tab)
}

/// Plans the conversion `src -> dst` (both static TV layouts of the same tensor, same thread / value counts). Panics when the layouts are not
/// bijections onto the same element set.
pub fn plan_convert(src: &Layout, dst: &Layout, per_reg: usize) -> ConvPlan {
    let (ts, vs, stab) = table(src);
    let (td, vd, dtab) = table(dst);
    assert!(ts == td && vs == vd, "plan_convert: src {ts}x{vs} vs dst {td}x{vd} (threads x values)");
    plan_convert_tables(&stab, &dtab, ts, vs, per_reg)
}

/// `plan_convert` on element tables indexed `[t + threads v]` (column-major, as `Layout::at` of a TV layout).
pub fn plan_convert_tables(stab: &[i64], dtab: &[i64], ts: usize, vs: usize, per_reg: usize) -> ConvPlan {
    assert!(per_reg >= 1 && vs % per_reg == 0, "plan_convert: {vs} values per thread, {per_reg} per register");
    let (nt, nv) = (ts, vs);
    let mut pos = std::collections::HashMap::with_capacity(nt * nv);
    for t in 0..nt {
        for v in 0..nv {
            assert!(pos.insert(stab[t + nt * v], (t, v)).is_none(), "plan_convert: src maps element {} twice", stab[t + nt * v]);
        }
    }
    let mut src_of = vec![(0usize, 0usize); nt * nv];
    for t in 0..nt {
        for v in 0..nv {
            let e = dtab[t + nt * v];
            src_of[t * nv + v] = *pos.get(&e).unwrap_or_else(|| panic!("plan_convert: dst element {e} is not in src"));
        }
    }
    let all = |f: &dyn Fn(usize, usize, usize, usize) -> bool| (0..nt).all(|t| (0..nv).all(|v| { let (a, b) = src_of[t * nv + v]; f(t, v, a, b) }));
    let nreg = nv / per_reg;
    // whole-register renaming: every destination register of every lane is one source register of the same lane, bytes in place, same for all lanes
    let reg_map: Option<Vec<usize>> = if all(&|t, v, a, b| a == t && b % per_reg == v % per_reg) {
        let m: Vec<usize> = (0..nreg).map(|r| src_of[r * per_reg].1 / per_reg).collect();
        if all(&|_, v, _, b| b / per_reg == m[v / per_reg]) { Some(m) } else { None }
    } else {
        None
    };
    let movement = if all(&|t, v, a, b| a == t && b == v) {
        Movement::Identity
    } else if reg_map.is_some() {
        Movement::Register
    } else if all(&|t, _, a, _| a == t) {
        Movement::RegisterBytes
    } else if all(&|t, _, a, _| a >> 2 == t >> 2) {
        Movement::Quad
    } else if all(&|t, _, a, _| a == t || a == t ^ 32) {
        Movement::Half
    } else {
        Movement::CrossLane
    };
    // cost model (instructions per wave)
    let cost = match movement {
        Movement::Identity | Movement::Register => 0,
        Movement::RegisterBytes => (0..nreg).map(|r| sources_per_dst_reg(&src_of, nv, per_reg, 0, r).div_ceil(2)).sum(),
        Movement::Quad | Movement::Half => {
            // one cross-lane move per (destination register, source register) pair + merges + byte placement
            (0..nreg).map(|r| { let s = (0..nt).map(|t| sources_per_dst_reg(&src_of, nv, per_reg, t, r)).max().unwrap_or(1); 2 * s - 1 + usize::from(per_reg > 1) }).sum()
        }
        // wave-private LDS round trip: one write per source register, one read per destination register (+ the read address)
        Movement::CrossLane => 2 * nreg + 1,
    };
    ConvPlan { threads: nt, values: nv, per_reg, gf2: gf2_matrix(&src_of, nt, nv), src_of, movement, reg_map, cost }
}

fn sources_per_dst_reg(src_of: &[(usize, usize)], nv: usize, per_reg: usize, t: usize, r: usize) -> usize {
    let mut s: Vec<(usize, usize)> = (0..per_reg).map(|k| { let (a, b) = src_of[t * nv + r * per_reg + k]; (a, b / per_reg) }).collect();
    s.sort();
    s.dedup();
    s.len()
}

/// GF(2) matrix of `x = t | v << tb -> y = t' | v' << tb` (dst slot from src slot), when all sizes are powers of two and the map is linear.
fn gf2_matrix(src_of: &[(usize, usize)], nt: usize, nv: usize) -> Option<Vec<u32>> {
    if !nt.is_power_of_two() || !nv.is_power_of_two() {
        return None;
    }
    let tb = nt.trailing_zeros();
    let bits = tb + nv.trailing_zeros();
    // forward map src slot -> dst slot
    let mut fwd = vec![0u32; nt * nv];
    for t in 0..nt {
        for v in 0..nv {
            let (a, b) = src_of[t * nv + v];
            fwd[a | (b << tb)] = (t | (v << tb)) as u32;
        }
    }
    if fwd[0] != 0 {
        return None;
    }
    let cols: Vec<u32> = (0..bits).map(|i| fwd[1usize << i]).collect();
    let lin = (0..fwd.len()).all(|x| { let mut y = 0u32; for (i, c) in cols.iter().enumerate() { if x >> i & 1 == 1 { y ^= c; } } y == fwd[x] });
    lin.then_some(cols)
}

impl ConvPlan {
    /// "lane bit 3 -> value bit 1"-style description of a bit-permutation map (None if not a permutation of bits).
    pub fn bit_moves(&self) -> Option<Vec<String>> {
        let cols = self.gf2.as_ref()?;
        let tb = self.threads.trailing_zeros() as usize;
        let name = |i: usize| if i < tb { format!("lane bit {i}") } else { format!("value bit {}", i - tb) };
        let mut out = Vec::new();
        for (i, c) in cols.iter().enumerate() {
            if c.count_ones() != 1 {
                return None;
            }
            let j = c.trailing_zeros() as usize;
            if j != i {
                out.push(format!("{} -> {}", name(i), name(j)));
            }
        }
        Some(out)
    }
}
