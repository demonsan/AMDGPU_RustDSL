//! Fusion hand-off: `convert_layout` lowers a [`ConvPlan`] (amdgpu_layout::convert) of one register fragment to instructions, cheapest form first
//! (docs/fusion-layout-handoff.md):
//! * `Identity` / `Register`: no instruction - the destination registers are the source registers renamed (`reg_map`);
//! * `Quad`: per destination register one `v_mov_b32_dpp quad_perm` per source register it draws from, merged with `v_cndmask` on `lane % 4`;
//! * `CrossLane` / `Half` (32-bit elements, GF(2)-linear maps): a round trip through a wave-private LDS scratch - every lane writes its registers at
//!   `4 (64 v + t)`, then reads destination register `r'` from the address of its source slot, an XOR of lane-bit groups (the GF(2) matrix).
//! Every lowering is checked at build time by simulating the emitted moves on element ids (`simulate_*`): the result must equal the plan's map.

use amdgpu_dsl::tile::{ev, PrivRef};
use amdgpu_dsl::{FnBuilder, ValueId, I32};
use amdgpu_layout::{ConvPlan, Movement};

/// What `convert_layout` emitted (for reports / tests).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConvStats {
    pub dpp: usize,
    pub selects: usize,
    pub lds_writes: usize,
    pub lds_reads: usize,
}

/// A quad lowering: per destination register, the (source register, quad perm, lanes-in-quad mask) moves.
pub type QuadMoves = Vec<Vec<(usize, [u8; 4], u8)>>;

/// The quad moves of a plan (32-bit elements): `None` if some (destination register, source register) pair needs different quad patterns in
/// different quads (not one `quad_perm`).
pub fn quad_moves(p: &ConvPlan) -> Option<QuadMoves> {
    assert_eq!(p.per_reg, 1, "quad lowering: 32-bit elements");
    let (nt, nv) = (p.threads, p.values);
    let mut out = Vec::with_capacity(nv);
    for r in 0..nv {
        // source register -> (perm, mask): perm[j] = source lane position of quad position j
        let mut groups: Vec<(usize, [Option<u8>; 4], u8)> = Vec::new();
        for t in 0..nt {
            let (a, v) = p.src_of[t * nv + r];
            let j = t % 4;
            let g = match groups.iter_mut().position(|g| g.0 == v) {
                Some(i) => i,
                None => {
                    groups.push((v, [None; 4], 0));
                    groups.len() - 1
                }
            };
            let want = (a % 4) as u8;
            match groups[g].1[j] {
                None => groups[g].1[j] = Some(want),
                Some(x) if x == want => {}
                Some(_) => return None,
            }
            groups[g].2 |= 1 << j;
        }
        // a quad position served by two source registers in different quads cannot be one select on lane % 4
        let mut seen = 0u8;
        for g in groups.iter() {
            if seen & g.2 != 0 {
                return None;
            }
            seen |= g.2;
        }
        out.push(groups.into_iter().map(|(v, pm, m)| (v, pm.map(|x| x.unwrap_or(0)), m)).collect());
    }
    Some(out)
}

/// Simulates quad moves on element ids: `src[t][v]` -> the destination fragment.
pub fn simulate_quad(p: &ConvPlan, mv: &QuadMoves) -> Vec<(usize, usize)> {
    let (nt, nv) = (p.threads, p.values);
    let mut dst = vec![(usize::MAX, usize::MAX); nt * nv];
    for (r, moves) in mv.iter().enumerate() {
        for t in 0..nt {
            let j = t % 4;
            for &(v, pm, m) in moves {
                if m >> j & 1 == 1 {
                    dst[t * nv + r] = (4 * (t / 4) + pm[j] as usize, v);
                }
            }
        }
    }
    dst
}

/// LDS round-trip addresses: destination register `r'` of lane `t'` reads byte `4 (64 v + t)` where `(t, v)` is its source slot; for a GF(2) plan
/// the address is `base(r') XOR lin(t')` with `lin` linear in the lane bits. Returns per register (constant part, per-lane-bit XOR masks).
pub fn lds_addr_form(p: &ConvPlan) -> Option<Vec<(i64, Vec<i64>)>> {
    assert_eq!(p.per_reg, 1);
    let (nt, nv) = (p.threads, p.values);
    if nt != 64 {
        return None;
    }
    let addr = |t: usize, r: usize| { let (a, v) = p.src_of[t * nv + r]; (4 * (64 * v + a)) as i64 };
    let mut out = Vec::with_capacity(nv);
    for r in 0..nv {
        let c = addr(0, r);
        let masks: Vec<i64> = (0..6).map(|b| addr(1 << b, r) ^ c).collect();
        // check XOR-linearity over all lanes
        for t in 0..nt {
            let mut x = c;
            for (b, m) in masks.iter().enumerate() {
                if t >> b & 1 == 1 {
                    x ^= m;
                }
            }
            if x != addr(t, r) {
                return None;
            }
        }
        out.push((c, masks));
    }
    Some(out)
}

/// Lowers the conversion of `src` (one `ValueId` per 32-bit register of this lane, `p.values / p.per_reg` of them) and returns the destination
/// registers. `lane`: the lane id (0..63); `scratch`: a wave-private LDS region of at least `256 p.values` bytes (only for cross-lane plans) and the
/// generation of the round trip.
pub fn convert_layout(b: &mut FnBuilder, p: &ConvPlan, src: &[ValueId], lane: &I32, scratch: Option<(&PrivRef<'_>, i32)>, st: &mut ConvStats) -> Vec<ValueId> {
    let nreg = p.values / p.per_reg;
    assert_eq!(src.len(), nreg, "convert_layout: {} source registers for a {nreg}-register fragment", src.len());
    match p.movement {
        Movement::Identity => src.to_vec(),
        Movement::Register => p.reg_map.as_ref().expect("register plan").iter().map(|&r| src[r]).collect(),
        Movement::Quad if quad_moves(p).is_some() => {
            let mv = quad_moves(p).expect("quad");
            assert_eq!(simulate_quad(p, &mv), p.src_of, "quad lowering does not reproduce the plan");
            let j = b.bind_i32(lane.clone() & 3);
            mv.iter()
                .map(|moves| {
                    let mut acc: Option<ValueId> = None;
                    for &(v, pm, m) in moves {
                        let x = if pm == [0, 1, 2, 3] {
                            src[v]
                        } else {
                            st.dpp += 1;
                            let y = b.dpp_quad_perm(src[v], pm);
                            b.bitcast_i32(y)
                        };
                        acc = Some(match acc {
                            None => x,
                            Some(prev) => {
                                st.selects += 1;
                                // lanes whose quad position is in mask m take x
                                let in_m = (I32::c(m as i32).lshr(ev(j)) & 1).eq(1);
                                b.bind_i32(in_m.select_i32(ev(x), ev(prev)))
                            }
                        });
                    }
                    acc.expect("every destination register has a source")
                })
                .collect()
        }
        _ => {
            let (pr, gen) = scratch.expect("cross-lane conversion needs a wave-private LDS scratch");
            let form = lds_addr_form(p).expect("cross-lane conversion: the source-slot address must be GF(2)-linear in the lane bits");
            let base = b.bind_i32(pr.base() + (lane.clone() << 2));
            pr.write(b, gen, |b| {
                for (v, x) in src.iter().enumerate() {
                    st.lds_writes += 1;
                    let a = b.bind_i32(ev(base) + 256 * v as i32);
                    b.store_lds_b32(a, *x);
                }
            });
            b.wait_lds(0);
            let pb = pr.base();
            pr.read_back(b, gen, |b| {
                form.iter()
                    .map(|(c, masks)| {
                        st.lds_reads += 1;
                        let mut e = I32::c(*c as i32);
                        for (bit, m) in masks.iter().enumerate() {
                            if *m != 0 {
                                e = e ^ ((lane.clone().lshr(bit as i32) & 1) * (*m as i32));
                            }
                        }
                        let a = b.bind_i32(pb.clone() + e);
                        b.load_lds_b32(a)
                    })
                    .collect()
            })
        }
    }
}
