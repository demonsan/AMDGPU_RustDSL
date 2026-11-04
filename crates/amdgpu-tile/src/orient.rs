//! MMA orientation search at a hand-off between two MMAs (docs/fusion-layout-handoff.md §2).
//!
//! The attention pattern: a producer MMA computes the scores `S[q][key]` (or `S^T`), an elementwise step (softmax, fp8 conversion) turns them into
//! `P`, and a consumer MMA contracts `P` over the keys with `V`. Free choices:
//! * the producer atom and its orientation: `S` (A = Q, B = K: C rows = queries) or `S^T` (A = K, B = Q: C rows = keys);
//! * the consumer atom and the role of `P`: A operand (`O = P V`, O rows = queries) or B operand (`O^T = V^T P^T`, O columns = queries);
//! * a permutation `sigma` of the consumer's K (keys) inside one K block - absorbed by the other operand's (V's) read layout, so it is free.
//! The cost of a candidate = the `convert_layout` cost of S's register fragment to P's operand fragment (zero when every lane already holds the
//! elements it needs, up to sigma: a register renaming - the fp8 packing instruction takes any 4 registers) + the cost of bringing the per-query
//! softmax statistics (held by the lanes that hold S's queries) to the lanes of O that the rescale touches.
//! Ties prefer the larger atom (fewer MFMAs and operand registers per FLOP).

use amdgpu_isa::MmaAtom;
use amdgpu_layout::{plan_convert_tables, Layout};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum POperand {
    /// `O = P V`: P is the A operand (rows = queries)
    A,
    /// `O^T = V^T P^T`: P is the B operand (columns = queries)
    B,
}

#[derive(Clone, Debug)]
pub struct Orientation {
    pub producer: &'static str,
    /// the producer computes `S^T` (A = K, B = Q): C rows are keys, lanes hold queries
    pub s_transposed: bool,
    pub consumer: &'static str,
    pub p_operand: POperand,
    /// consumer k -> key (within one K block) making the S -> P conversion a register renaming; `None` if no such permutation
    pub sigma: Option<Vec<i64>>,
    /// S -> P conversion cost (instructions per wave per consumer K block)
    pub conv_cost: usize,
    /// softmax-statistics -> O alignment cost (0 = every O element's query is held by its own lane in S)
    pub stats_cost: usize,
    /// MACs per instruction of the consumer / producer atoms (tie breaks: larger first)
    pub macs: i64,
    pub pmacs: i64,
}

impl Orientation {
    pub fn cost(&self) -> usize {
        self.conv_cost + self.stats_cost
    }
}

fn tab(tv: &Layout) -> (usize, usize, Vec<i64>) {
    let t = tv.get(0).size().st() as usize;
    let v = tv.get(1).size().st() as usize;
    (t, v, (0..t * v).map(|i| tv.at(i as i64)).collect())
}

/// Per lane: the (query, key) pairs the producer's accumulators hold over the consumer block `[0, mq) x [0, kk)` (tiles in key-major order).
pub fn producer_pairs(p: &MmaAtom, s_t: bool, mq: i64, kk: i64) -> Option<Vec<Vec<(i64, i64)>>> {
    let (nt, nv, c) = tab(&p.c_tv);
    // C element index = row + M col
    let (pq, pk) = if s_t { (p.n, p.m) } else { (p.m, p.n) };
    if mq % pq != 0 || kk % pk != 0 || nt != 64 {
        return None;
    }
    let mut out = vec![Vec::new(); nt];
    for ik in 0..kk / pk {
        for iq in 0..mq / pq {
            for (t, lane) in out.iter_mut().enumerate() {
                for v in 0..nv {
                    let e = c[t + nt * v];
                    let (row, col) = (e % p.m, e / p.m);
                    let (q, key) = if s_t { (col, row) } else { (row, col) };
                    lane.push((q + pq * iq, key + pk * ik));
                }
            }
        }
    }
    Some(out)
}

/// Per lane: the (query, k) pairs of the consumer operand holding P (element `m + M k`, m = query).
pub fn consumer_pairs(c: &MmaAtom, role: POperand) -> Vec<Vec<(i64, i64)>> {
    let (tv, mrows) = match role {
        POperand::A => (&c.a_tv, c.m),
        POperand::B => (&c.b_tv, c.n),
    };
    let (nt, nv, t) = tab(tv);
    (0..nt).map(|l| (0..nv).map(|v| { let e = t[l + nt * v]; (e % mrows, e / mrows) }).collect()).collect()
}

/// A permutation sigma (k -> key) under which every lane's multiset of (query, sigma(k)) equals its producer multiset: keys and ks are matched by
/// their signature (the sorted (lane, query) incidences).
fn find_sigma(src: &[Vec<(i64, i64)>], dst: &[Vec<(i64, i64)>], kk: i64) -> Option<Vec<i64>> {
    let sig = |side: &[Vec<(i64, i64)>]| -> Vec<Vec<(usize, i64)>> {
        let mut s = vec![Vec::new(); kk as usize];
        for (l, lane) in side.iter().enumerate() {
            for &(q, k) in lane {
                s[k as usize].push((l, q));
            }
        }
        for x in s.iter_mut() {
            x.sort();
        }
        s
    };
    let (ss, ds) = (sig(src), sig(dst));
    let mut used = vec![false; kk as usize];
    let mut sigma = vec![-1i64; kk as usize];
    for k in 0..kk as usize {
        let key = (0..kk as usize).find(|&key| !used[key] && ss[key] == ds[k])?;
        used[key] = true;
        sigma[k] = key as i64;
    }
    Some(sigma)
}

/// The query of every O element (consumer C) per lane, for the statistics alignment.
fn o_queries(c: &MmaAtom, role: POperand) -> Vec<Vec<i64>> {
    let (nt, nv, t) = tab(&c.c_tv);
    (0..nt).map(|l| (0..nv).map(|v| { let e = t[l + nt * v]; let (row, col) = (e % c.m, e / c.m); if role == POperand::A { row } else { col } }).collect()).collect()
}

/// Every candidate (producer atom x orientation x consumer atom x P role), cheapest first.
pub fn search_attention(atoms: &[MmaAtom]) -> Vec<Orientation> {
    let mut out = Vec::new();
    for p in atoms {
        for s_t in [false, true] {
            for c in atoms {
                for role in [POperand::A, POperand::B] {
                    let (mq, kk) = (if role == POperand::A { c.m } else { c.n }, c.k);
                    let Some(src) = producer_pairs(p, s_t, mq, kk) else { continue };
                    let dst = consumer_pairs(c, role);
                    if src.iter().map(|x| x.len()).sum::<usize>() != dst.iter().map(|x| x.len()).sum::<usize>() || src[0].len() != dst[0].len() {
                        continue;
                    }
                    let sigma = find_sigma(&src, &dst, kk);
                    let conv_cost = if sigma.is_some() {
                        0
                    } else {
                        // sigma = identity: the element-level conversion plan (element = q + mq key)
                        let nv = src[0].len();
                        let st: Vec<i64> = (0..nv).flat_map(|v| src.iter().map(move |l| l[v].0 + mq * l[v].1)).collect();
                        let dt: Vec<i64> = (0..nv).flat_map(|v| dst.iter().map(move |l| l[v].0 + mq * l[v].1)).collect();
                        plan_convert_tables(&st, &dt, 64, nv, 1).cost
                    };
                    // statistics: queries of S per lane vs queries of O per lane
                    let oq = o_queries(c, role);
                    let stats_cost = oq.iter().enumerate().filter(|(l, qs)| qs.iter().any(|q| !src[*l].iter().any(|(sq, _)| sq == q))).count();
                    out.push(Orientation { producer: p.name, s_transposed: s_t, consumer: c.name, p_operand: role, sigma, conv_cost, stats_cost, macs: c.m * c.n * c.k, pmacs: p.m * p.n * p.k });
                }
            }
        }
    }
    out.sort_by(|a, b| a.cost().cmp(&b.cost()).then(b.macs.cmp(&a.macs)).then(b.pmacs.cmp(&a.pmacs)));
    out
}
