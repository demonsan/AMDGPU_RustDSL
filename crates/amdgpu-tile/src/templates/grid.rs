//! L2 template T5 (docs/l2-templates.md §3.5): grid maps. These are the logical work ids (tile, head, batch, part) derived from the hardware
//! workgroup ids. Each map emits its index maths together with the verifier facts that make the derived ids usable for store-disjointness proofs:
//! `wg_bijection` / `wg_union` and the trusted ranges of the ids the interval engine cannot bound.
//!
//! The emission order of every map is the one of the kernel code it was lifted from (md5-identical). Which map a kernel or shape uses stays a
//! kernel / L1 decision; the measurements behind those choices are in the call sites.

pub use crate::components::kv_split_decode;
use amdgpu_dsl::{ev, Ext, FnBuilder, Op, Opnd, ValueId, I32};

/// XCD-aware (query tile, head) order of the FMHA forward family (the vendor's block order). Workgroups are dealt round robin to the 8 XCDs
/// (linear id % 8), each with its own L2. The remap gives XCD x the contiguous logical range [x N / 8, (x + 1) N / 8) of the (query tile, head)
/// plane, N = tiles x heads, so the blocks that share K / V share an L2. For X > 16 tiles it instead deals chunks of 16 consecutive logical
/// blocks, cycling. Applied when N % 8 == 0, H <= 32 and (X <= 16 or N % 128 == 0); otherwise the ids are the workgroup ids.
///
/// `s_len` = the query length; `rows_lg` = log2 query rows per tile; `h_ka` / `sq_ka` = the kernarg offsets of H and SQ (for the facts).
/// Facts: the trusted ranges of both logical ids and `wg_bijection` over (x, y).
pub fn xcd_tile_head(b: &mut FnBuilder, s_len: I32, rows_lg: i32, h_ka: u32, sq_ka: u32) -> (ValueId, ValueId) {
    let x0u = b.let_(Op::WorkgroupIdX);
    let y0u = b.let_(Op::WorkgroupIdY);
    let x0 = b.let_(Op::AddI32(Opnd::Val(x0u), Opnd::I32(0)));
    let y0 = b.let_(Op::AddI32(Opnd::Val(y0u), Opnd::I32(0)));
    let hh = b.let_(Op::KernargI32 { offset: h_ka });
    let xq = b.bind_i32((s_len + ((1 << rows_lg) - 1)).lshr(rows_lg));
    let n = b.bind_i32(ev(xq) * ev(hh));
    // ok = (N % 8 == 0) && (H <= 32) && (X <= 16 || N % 128 == 0), as 0 / 1
    let ok = b.bind_i32((ev(n) & 7).eq(0).select_i32(ev(hh).le(32).select_i32(ev(xq).le(16).select_i32(1, (ev(n) & 127).eq(0).select_i32(1, 0)), 0), 0));
    let l = b.bind_i32(ev(y0) * ev(xq) + ev(x0));
    // X <= 16: the vendor's stride (XCD x owns the logical range [x N / 8, (x + 1) N / 8)). X > 16 (S > 4096): XCD x takes chunks of 16
    // consecutive logical blocks, cycling - with the vendor's stride all 32 CUs of an XCD streamed one head in lockstep and the L2 tag /
    // DRAM-credit stalls grew 8x (B8H8S8192 -6.9 %); chunks of 16 keep half a head per XCD (+0.2 ... +3.3 % at S 8192 / 16384)
    let k = b.bind_i32(ev(l).lshr(3));
    let r_v = b.bind_i32((ev(l) & 7) * ev(n).lshr(3) + ev(k));
    let r_c = b.bind_i32((ev(k).lshr(4) << 7) + ((ev(l) & 7) << 4) + (ev(k) & 15));
    let mut r = b.bind_i32(ev(xq).le(16).select_i32(ev(r_v), ev(r_c)));
    // head = M / X by binary long division (H <= 32: 5 steps)
    let mut y = b.c_i32(0);
    for bit in [16i32, 8, 4, 2, 1] {
        let xb = b.bind_i32(ev(xq) * bit);
        let take = b.bind_i32(ev(r).ge(ev(xb)).select_i32(1, 0));
        y = b.bind_i32(ev(y) + ev(take) * bit);
        r = b.bind_i32(ev(r) - ev(take) * ev(xb));
    }
    // clamps (no-ops for N % 8 == 0, H <= 32) bound the logical ids for the verifier
    let rc = b.bind_i32(ev(r).max(0).min(ev(xq) - 1));
    let yc = b.bind_i32(ev(y).max(0).min(ev(hh) - 1));
    // wave-uniform: back to SGPRs like the workgroup ids (otherwise the address math downstream turns vector)
    let qt = b.bind_i32(ev(ok).eq(1).select_i32(ev(rc), ev(x0)));
    let hi = b.bind_i32(ev(ok).eq(1).select_i32(ev(yc), ev(y0)));
    let qt = b.read_first_lane(qt);
    let hi = b.read_first_lane(hi);
    let qt = b.let_(Op::AddI32(Opnd::Val(qt), Opnd::I32(0)));
    let hi = b.let_(Op::AddI32(Opnd::Val(hi), Opnd::I32(0)));
    // the remap is a bijection of [0, X) x [0, H) (N % 8 == 0); the interval engine cannot follow the division: trusted contract facts
    b.assume_trusted(qt, Ext::c(0), Ext::ka(sq_ka).ceil_div(1 << rows_lg) - Ext::c(1), "XCD block order: the logical query tile M % X of the bijection M = (L % 8) N / 8 + L / 8 over [0, X H) lies in [0, X)");
    b.assume_trusted(hi, Ext::c(0), Ext::ka(h_ka) - Ext::c(1), "XCD block order: the logical head M / X of the bijection over [0, X H) lies in [0, H)");
    b.wg_bijection(&[qt, hi], &[0, 1], "XCD block order: (query tile, head) is a bijection of the workgroup (x, y) (the identity when the remap is off)");
    (qt, hi)
}

/// XCD-aware raster of the GEMM tile grid: workgroups are dealt round robin to the 8 XCDs, so every XCD gets whole 4 x 4 blocks of tiles
/// (4 A + 4 B panels instead of all A panels). Applied only to a 16 x 16 tile grid (`n == n16`, `m == m16`, the full N / M of 16 CTA tiles);
/// the identity otherwise. Facts: the trusted tile ranges (`NT` / `MT` symbols) and `wg_bijection` over (x, y).
pub fn xcd_raster_16x16(b: &mut FnBuilder, bx0: ValueId, by0: ValueId, n_v: ValueId, m_v: ValueId, n16: i32, m16: i32) -> (ValueId, ValueId) {
    let okv = b.bind_i32((ev(n_v) ^ n16) | (ev(m_v) ^ m16));
    let id = b.bind_i32(ev(bx0) + (ev(by0) << 4));
    let loc = b.bind_i32(ev(id).lshr(3));
    // block (of 16) = 2 (id % 8) + loc / 16 -> (row, col) = (block / 4, block % 4); inside the block: loc % 16 -> (n4, m4)
    let blk = b.bind_i32(((ev(id) & 7) << 1) + ev(loc).lshr(4));
    let (bc, br) = (ev(blk) & 3, ev(blk).lshr(2));
    let nx = b.bind_i32((bc << 2) + (ev(loc) & 3));
    let ny = b.bind_i32((br << 2) + (ev(loc).lshr(2) & 3));
    let s = b.bind_i32(ev(okv).eq(0).select_i32(1, 0));
    let px = b.bind_i32(ev(bx0) + ev(s) * (ev(nx) - ev(bx0)));
    let py = b.bind_i32(ev(by0) + ev(s) * (ev(ny) - ev(by0)));
    b.assume_trusted(px, Ext::c(0), Ext::sym("NT") - Ext::c(1), "XCD raster: (px, py) is a permutation of the 16 x 16 tile grid when the select applies (N = 16 cta_n, M = 16 cta_m: 4 x 4 blocks, 2 per XCD), the identity (workgroup ids) otherwise");
    b.assume_trusted(py, Ext::c(0), Ext::sym("MT") - Ext::c(1), "XCD raster: see px");
    b.wg_bijection(&[px, py], &[0, 1], "XCD raster: (px, py) is a permutation of the tile grid (the identity unless N = 16 cta_n, M = 16 cta_m)");
    (px, py)
}

/// How a causal pairing makes its second tile index wave-uniform (the forms of the kernels it was lifted from).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairForm {
    /// `read_first_lane(nt - 1 - x)` (bf16 v3)
    Plain,
    /// `read_first_lane(max(nt - 1 - x, 0))` re-typed as i32 (per-tensor FP8)
    ClampedI32,
}

/// Causal q-tile pairing: CTA x runs q tiles x and nt - 1 - x (the second only when nt - 1 - x > x). Every CTA gets equal work, and the grid
/// x is ceil(nt / 2). Returns (the second tile, its guard). Fact: `wg_union` over (x, second) on grid axis 0 covering [0, `tiles` - 1],
/// so the output-tile stores of the two passes are disjoint across CTAs.
pub fn causal_pair(b: &mut FnBuilder, x: ValueId, nt: ValueId, tiles: Ext, form: PairForm) -> (ValueId, ValueId) {
    let qb = match form {
        PairForm::Plain => {
            let v = b.bind_i32(ev(nt) - 1 - ev(x));
            b.read_first_lane(v)
        }
        PairForm::ClampedI32 => {
            let v = b.bind_i32((ev(nt) - 1 - ev(x)).max(0));
            let u = b.read_first_lane(v);
            b.let_(Op::AddI32(Opnd::Val(u), Opnd::I32(0)))
        }
    };
    let second = b.bind_cond(ev(qb).gt(ev(x)));
    b.wg_union(&[x, qb], &[0], Ext::c(0), tiles - Ext::c(1), "causal pairing: CTA x takes q tiles x and nt - 1 - x (the second only when nt - 1 - x > x), grid x = ceil(nt / 2): every q tile once");
    (qb, second)
}

/// Split-major grid z = batch 2^lg_s + part (the bf16 v3 `_ks` prefill forms; S = 2^lg_s parts, a power of two). Returns (batch, part).
/// Facts: trusted ranges against the `BB` (batch) and `SPL` (parts) contract symbols.
pub fn split_pow2_z(b: &mut FnBuilder, z: ValueId, lg_s: ValueId) -> (ValueId, ValueId) {
    let bt = { let v = b.bind_i32(ev(z).lshr(ev(lg_s))); b.read_first_lane(v) };
    let pt = { let v = b.bind_i32(ev(z) - (ev(bt) << ev(lg_s))); b.read_first_lane(v) };
    b.assume_trusted(bt, Ext::c(0), Ext::sym("BB") - Ext::c(1), "split: z = batch 2^lg_s + part (grid z = B S, S = 2^lg_s)");
    b.assume_trusted(pt, Ext::c(0), Ext::sym("SPL") - Ext::c(1), "split: the part index < S");
    (bt, pt)
}

/// Longest-first launch order (LPT) of the causal backward kernels. The hardware takes workgroups in linear-id order and deals them to the
/// XCDs, so the tile index is put on z (slowest) with the heaviest tiles first. Grid (`heads`, B, `tiles`); the plain order is (`tiles`,
/// `heads`, B).
pub fn tile_grid(b: &mut FnBuilder, lpt: bool, tiles: Ext, heads: Ext) {
    if lpt {
        b.grid_is(0, heads);
        b.grid_is(1, Ext::sym("B"));
        b.grid_is(2, tiles);
    } else {
        b.grid_is(0, tiles);
        b.grid_is(1, heads);
        b.grid_is(2, Ext::sym("B"));
    }
}

/// The (tile, head, batch) ids of `tile_grid`. With `lpt`, `reverse_from = Some(nt)` makes the heaviest tile nt - 1 - z (the causal dq: the
/// last query tile sees the most keys); `None` keeps z (the causal dkdv: key tile 0 sees the most queries).
pub fn tile_ids(b: &mut FnBuilder, lpt: bool, reverse_from: Option<ValueId>) -> (ValueId, ValueId, ValueId) {
    if !lpt {
        return (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    }
    match reverse_from {
        Some(nt) => {
            let z = b.let_(Op::WorkgroupIdZ);
            let t = { let v = b.bind_i32(ev(nt) - 1 - ev(z)); b.read_first_lane(v) };
            (t, b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY))
        }
        None => (b.let_(Op::WorkgroupIdZ), b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY)),
    }
}

/// Decode key splits, per wave: a decode CTA of grid z part `z` runs 4 waves, each with its own key part, so there are NS = 4 S parts in all
/// and wave `wv` of part z owns part ns = 4 z + wv. Returns (NS, ns), with ns wave-uniform. Used by the paged decode kernels and MLA decode.
pub fn wave_split_index(b: &mut FnBuilder, z: ValueId, wv: ValueId, s_n: ValueId) -> (ValueId, ValueId) {
    let nsplit = b.bind_i32(ev(s_n) << 2);
    let ns = { let v = b.bind_i32((ev(z) << 2) + ev(wv)); b.read_first_lane(v) };
    (nsplit, ns)
}

/// log2(NS) for NS = 4 S with S a power of two <= 2^`levels`, from `levels` compares. The result is wave-uniform.
pub fn split_lg(b: &mut FnBuilder, s_n: ValueId, levels: u32) -> ValueId {
    let sv = ev(s_n);
    let mut acc = I32::c(0);
    for l in 0..levels as i32 {
        acc = sv.clone().gt(1 << l).select_i32(l + 1, acc);
    }
    let v = b.bind_i32(acc + 2);
    b.read_first_lane(v)
}

/// The key range [k_lo, k_hi) of part `ns`. Each part gets a chunk of ceil(tiles / NS) whole tiles of 2^`kt_lg` keys; the range is
/// [ns chunk, min(ctx, (ns + 1) chunk)) and is empty past the context. Both bounds are wave-uniform.
pub fn key_chunk(b: &mut FnBuilder, ctx: ValueId, ns: ValueId, nsplit: ValueId, lg: ValueId, kt_lg: i32) -> (ValueId, ValueId) {
    let ntt = b.bind_i32((ev(ctx) + ((1 << kt_lg) - 1)).lshr(kt_lg));
    let per = b.bind_i32(((ev(ntt) + ev(nsplit) - 1).lshr(ev(lg))) << kt_lg);
    let k_lo = { let v = b.bind_i32(ev(ns) * ev(per)); b.read_first_lane(v) };
    let k_hi = { let v = b.bind_i32((ev(k_lo) + ev(per)).min(ev(ctx))); b.read_first_lane(v) };
    (k_lo, k_hi)
}