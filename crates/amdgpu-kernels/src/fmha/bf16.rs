//! bf16 attention forward (batch 3 of docs/op-coverage-design.md): `fmha_fwd_hd128_bf16[_causal]`, the aiter fmha_v3 656-byte ABI
//! (csrc/include/mha_fwd.h `fmha_fwd_v3_args`, every field in a 16-byte slot), BSHD Q / K / V / O bf16, optional fp32 LSE [B, H, SQ] (natural
//! log), GQA (H / HK), causal = bottom-right aligned. Grid (ceil(SQ / 128), H, B), 256 threads.
//!
//! Structure (first coverage version):
//! * 4 waves x 32 query rows; orientation S^T = K Q^T and O^T = V^T P^T with `v_mfma_f32_32x32x16_bf16`: a lane owns ONE query (column l % 32)
//!   of S^T / O^T, so the online softmax statistics are per-lane scalars combined across the two lane halves with `v_permlane32_swap`;
//! * Q^T (the B operand) in registers for the whole loop (8 x 16 bytes per lane);
//! * K / V tiles of 64 keys (16 KB each) in a 2-slot LDS ring, `buffer_load_dwordx4 ... lds` DMA of tile t + 1 issued after the top-of-tile
//!   barrier of tile t; two tiles per loop pass (static slots), the key count rounded up to an even number of tiles (the extra tile is masked);
//! * K rows swizzled (16-byte chunk c of key k at chunk c ^ (k % 16)) for the 16-byte A-operand reads; V plain, read as 16-bit elements for
//!   the transposed A operand of PV (`ds_read_u16_d16[_hi]`; to be replaced by `ds_read_b64_tr_b16`);
//! * P = exp2(S c - m) packed to bf16 straight from the S^T accumulator: the PV k order per lane half h is keys {4h + 0..3, 8 + 4h + 0..3} of
//!   each 16-key step, and the V^T reads use the same key order.
use amdgpu_dsl::{CompileProfile, D16Ty, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId, I32};
use amdgpu_tile::components::gqa_kv_head;
use amdgpu_tile::{ev, LdsPlan, Lane, Layout as TLayout, SlotIdx};

/// keys per tile
const KT: i32 = 64;

/// Build options.
#[derive(Clone, Copy, Debug)]
pub struct AttnCfg {
    /// head dim (Q / K / V): 64, 128, 256
    pub d: i32,
    /// bottom-right aligned causal mask
    pub causal: bool,
    /// sliding window: keys in [q' - left, q' + right] with q' = q + SK - SQ (our ABI extension: left / right at kernarg 656 / 672, int32);
    /// combined with `causal` the right edge is min(right, 0)
    pub swa: bool,
}

/// `fmha_fwd_hd128_bf16[_causal]` (the shipped coverage names).
pub fn fmha_fwd_hd128_bf16(name: &str, causal: bool) -> Function {
    fmha_fwd_bf16(name, AttnCfg { d: 128, causal, swa: false })
}

/// The kernel for `cfg`.
#[allow(non_snake_case)]
pub fn fmha_fwd_bf16(name: &str, cfg: AttnCfg) -> Function {
    let (D, causal) = (cfg.d, cfg.causal);
    assert!(matches!(D, 64 | 128 | 256), "head dim 64 / 128 / 256");
    // bytes of one K / V row (D bf16), 16-byte chunks per row, the K swizzle mask (chunk ^ key % 16, or % 8 for 8-chunk rows)
    let ROW: i32 = 2 * D;
    let cpr = D / 8;
    let swz = cpr.min(16) - 1;
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    // ---- ABI: field i at 16 i ----
    let ptr_names = [("ptr_o", 0u32), ("ptr_q", 1), ("ptr_k", 2), ("ptr_v", 3), ("ptr_lse", 4)];
    for (n, i) in ptr_names {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    let i32_names = [
        ("scalar", 5u32), ("s_seq_len", 6), ("s_Seqs", 7), ("s_Ts", 8), ("s_Hs", 9), ("s_Bs", 10), ("s_gqa", 11), ("s_k_Seqs", 12), ("s_k_Hs", 13),
        ("s_k_Bs", 14), ("s_opt", 15), ("s_lse", 16), ("s_kv_seq_len", 17), ("s_qk_head_dim", 18), ("s_v_head_dim", 19), ("s_q_head_num", 20),
        ("s_v_Seqs", 21), ("s_v_Hs", 22), ("s_v_Bs", 23), ("s_o_Seqs", 24), ("s_o_Hs", 25), ("s_o_Bs", 26), ("s_lse_Hs", 29),
    ];
    if cfg.swa {
        b.arg("window_left", DType::I32, 656, 4);
        b.arg("window_right", DType::I32, 672, 4);
        b.ka_range(656, 0, 1 << 20);
        b.ka_range(672, 0, 1 << 20);
    }
    for (n, i) in i32_names {
        b.arg(n, if n == "scalar" { DType::F32 } else { DType::I32 }, 16 * i, 4);
    }
    // ---- contract: dense BSHD, D = 128 ----
    let (bb, sq, sk, hh, gq) = (Ext::grid_z(), Ext::ka(96), Ext::ka(272), Ext::ka(320), Ext::ka(176));
    b.ka_range(96, 1, 1 << 20);
    b.ka_range(272, 1, 1 << 20);
    b.ka_range(176, 1, 16);
    b.sym_range("HK", 1, 64);
    b.ka_is(320, Ext::sym("HK") * gq.clone());
    let hk = Ext::sym("HK");
    b.ka_is(288, Ext::c(D as i64));
    b.ka_is(304, Ext::c(D as i64));
    b.ka_is(112, hh.clone() * Ext::c(ROW as i64));
    b.ka_is(144, Ext::c(ROW as i64));
    b.ka_is(160, sq.clone() * hh.clone() * Ext::c(ROW as i64));
    for (s, hs, bs) in [(192u32, 208u32, 224u32), (336, 352, 368)] {
        b.ka_is(s, hk.clone() * Ext::c(ROW as i64));
        b.ka_is(hs, Ext::c(ROW as i64));
        b.ka_is(bs, sk.clone() * hk.clone() * Ext::c(ROW as i64));
    }
    b.ka_is(384, hh.clone() * Ext::c(ROW as i64));
    b.ka_is(400, Ext::c(ROW as i64));
    b.ka_is(416, sq.clone() * hh.clone() * Ext::c(ROW as i64));
    b.ka_is(464, sq.clone() * Ext::c(4));
    b.grid_is(0, sq.clone().ceil_div(128));
    b.grid_is(1, hh.clone());
    b.arg_tensor("ptr_q", bb.clone() * sq.clone() * hh.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_o", bb.clone() * sq.clone() * hh.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_k", bb.clone() * sk.clone() * hk.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_v", bb.clone() * sk.clone() * hk * Ext::c(ROW as i64));
    b.arg_tensor("ptr_lse", bb * hh * sq * Ext::c(4));

    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 16 * i });
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    let (p_o, p_q, p_k, p_v, p_lse) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4));
    let scale = b.let_(Op::KernargF32 { offset: 80 });
    let (s_q, q_seqs, q_hs, q_bs, gqa) = (ki(&mut b, 6), ki(&mut b, 7), ki(&mut b, 9), ki(&mut b, 10), ki(&mut b, 11));
    let (k_seqs, k_hs, k_bs, lse_on, s_kv) = (ki(&mut b, 12), ki(&mut b, 13), ki(&mut b, 14), ki(&mut b, 16), ki(&mut b, 17));
    let (v_seqs, v_hs, v_bs, o_seqs, o_hs, o_bs, lse_hs) = (ki(&mut b, 21), ki(&mut b, 22), ki(&mut b, 23), ki(&mut b, 24), ki(&mut b, 25), ki(&mut b, 26), ki(&mut b, 29));

    let qt = b.let_(Op::WorkgroupIdX);
    let h = b.let_(Op::WorkgroupIdY);
    let bat = b.let_(Op::WorkgroupIdZ);
    let lane = Lane::new(&mut b);
    // GQA ratio a power of two <= 16 (the vendor's condition)
    let hkv = gqa_kv_head(&mut b, h, gqa, "HK");
    let hkv = b.read_first_lane(hkv);
    // the lane's query row (column of S^T / O^T) and lane half
    let row = b.bind_i32((ev(qt) << 7) + (lane.wave() << 5) + lane.j());
    let half = b.bind_i32(lane.half());
    let row_c = b.bind_i32(ev(row).min(ev(s_q) - 1));

    // ---- buffer descriptors (over the batches up to this one: rows past the end of a batch read the next one, masked) ----
    let desc = |b: &mut FnBuilder, p: ValueId, bs: ValueId| {
        let n = b.bind_i32((ev(bat) + 1) * ev(bs));
        let n = b.read_first_lane(n);
        b.buffer_desc_dyn(p, n)
    };
    let (r_q, r_k, r_v, r_o) = (desc(&mut b, p_q, q_bs), desc(&mut b, p_k, k_bs), desc(&mut b, p_v, v_bs), desc(&mut b, p_o, o_bs));

    // ---- Q^T fragments: B operand of 32x32x16 bf16, lane (column = query row, k = d 16 c + 8 half + 0..7): 16 bytes at d 16 c + 8 h ----
    let q_so = { let v = b.bind_i32(ev(bat) * ev(q_bs) + ev(h) * ev(q_hs)); b.read_first_lane(v) };
    let q_vo = b.bind_i32(ev(row_c) * ev(q_seqs) + (ev(half) << 4));
    let qf: Vec<ValueId> = (0..D / 16).map(|c| b.buffer_load_b128(r_q, q_vo, q_so, (32 * c) as u32)).collect();

    // ---- LDS: K / V rings (2 slots x 16 KB) ----
    let mut lds = LdsPlan::new();
    let ks = lds.ring("K", (KT * ROW) as i64, 2, TLayout::Linear);
    let vs = lds.ring("V", (KT * ROW) as i64, 2, TLayout::Linear);
    lds.commit(&mut b);
    // DMA: per tile 128 D bytes = D / 8 blocks of 1 KB; wave w issues blocks 4 i + w (i < D / 32): keys kpb (4 i + w) + lane / cpr (kpb = 1 KB / row
    // bytes), 16-byte chunk lane % cpr. K: the source chunk is the LDS position XOR (key & swz) (the reader's swizzle); V: plain.
    let (kpb, n_iss) = (512 / D, D / 32);
    let kv_so0 = |b: &mut FnBuilder, bs: ValueId, hs: ValueId| { let v = b.bind_i32(ev(bat) * ev(bs) + ev(hkv) * ev(hs)); b.read_first_lane(v) };
    let (k_so0, v_so0) = (kv_so0(&mut b, k_bs, k_hs), kv_so0(&mut b, v_bs, v_hs));
    let lc = b.bind_i32(lane.lane() & (cpr - 1));
    let key_of = |b: &mut FnBuilder, i: i32| b.bind_i32((lane.wave() + 4 * i) * kpb + lane.lane().lshr(cpr.trailing_zeros() as i32));
    let k_vo: Vec<ValueId> = (0..n_iss)
        .map(|i| {
            let key = key_of(&mut b, i);
            b.bind_i32(ev(key) * ev(k_seqs) + ((ev(lc) ^ (ev(key) & swz)) << 4))
        })
        .collect();
    let v_vo: Vec<ValueId> = (0..n_iss)
        .map(|i| {
            let key = key_of(&mut b, i);
            b.bind_i32(ev(key) * ev(v_seqs) + (ev(lc) << 4))
        })
        .collect();
    let stage = |b: &mut FnBuilder, tile: ValueId, slot: i32, gen: Opnd| {
        for (sh, vo, so0, seqs, r) in [(&ks, &k_vo, k_so0, k_seqs, r_k), (&vs, &v_vo, v_so0, v_seqs, r_v)] {
            let so = { let v = b.bind_i32(ev(so0) + ev(tile) * KT * ev(seqs)); b.read_first_lane(v) };
            let base = sh.base as i32 + slot * KT * ROW;
            sh.at(&SlotIdx::new(slot, gen)).write(b, |b| {
                for (i, v) in vo.iter().enumerate() {
                    let m0 = { let x = b.bind_i32(((lane.wave() + 4 * i as i32) << 10) + base); b.read_first_lane(x) };
                    b.buffer_load_lds_b128(r, *v, so, m0, 0);
                }
            });
        }
    };

    // ---- per-lane LDS read offsets ----
    // K (A of S^T, lane: key l % 32 of the 32-key block, k = d 16 c + 8 h): 16 bytes at key row, chunk (2 c + h) ^ (key % 16)
    let kr: Vec<ValueId> = (0..D / 16)
        .map(|c| b.bind_i32(lane.j() * ROW + (((ev(half) + 2 * c) ^ (lane.lane() & swz)) << 4)))
        .collect();
    // V^T (A of O^T, lane: d = 32 e + l % 32, k slot kk -> key 4 h + kk (kk < 4) or 8 + 4 h + kk - 4): bf16 at key row, column d
    let vr: Vec<ValueId> = (0..8)
        .map(|kk| {
            let krow = if kk < 4 { kk } else { 8 + kk - 4 };
            b.bind_i32((ev(half) * 4 + krow) * ROW + (lane.j() << 1))
        })
        .collect();

    // ---- the key range: causal keys k <= q + SK - SQ; tiles rounded up to an even count ----
    // the visible keys of query q: [q' - left, q' + right] (q' = q + SK - SQ), right = 0 for causal, unbounded without a window
    let wins = cfg.swa.then(|| (b.let_(Op::KernargI32 { offset: 656 }), b.let_(Op::KernargI32 { offset: 672 })));
    let right: Option<amdgpu_dsl::I32> = match (causal, wins) {
        (true, Some((_, r))) => Some(I32::c(0).min(ev(r))),
        (true, None) => Some(I32::c(0)),
        (false, Some((_, r))) => Some(ev(r)),
        (false, None) => None,
    };
    let qoff = b.bind_i32(ev(s_kv) - ev(s_q));
    let kv_end = match &right {
        Some(r) => {
            let last_q = b.bind_i32(((ev(qt) + 1) << 7) - 1);
            b.bind_i32((ev(last_q) + ev(qoff) + r.clone() + 1).min(ev(s_kv)).max(0))
        }
        None => s_kv,
    };
    // first tile: (first query of the CTA)' - left, rounded down to a tile
    let kt0 = match wins {
        Some((l, _)) => { let v = b.bind_i32(((ev(qt) << 7) + ev(qoff) - ev(l)).max(0).lshr(6)); b.read_first_lane(v) }
        None => b.c_i32(0),
    };
    let ntile = b.bind_i32((ev(kv_end) + (KT - 1)).lshr(6) - ev(kt0));
    let npair = b.bind_i32((ev(ntile).max(1) + 1).lshr(1));
    let npair = b.read_first_lane(npair);
    let lim = b.bind_i32(match &right { Some(r) => ev(row) + ev(qoff) + r.clone(), None => ev(s_kv) - 1 });
    let lim = b.bind_i32(ev(lim).min(ev(s_kv) - 1));
    let low = wins.map(|(l, _)| b.bind_i32(ev(row) + ev(qoff) - ev(l)));
    // softmax constants: c = scale log2(e)
    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let kind = MmaKind::new(MmaShape::M32N32K16, MmaTy::Bf16);
    let zero16 = b.frag_zero(DType::Frag16);
    let m_init = b.c_f32(-1.0e30);
    let l_init = b.c_f32(0.0);

    // prologue: tile 0 into slot 0
    stage(&mut b, kt0, 0, Opnd::I32(0));
    // loop vars: pair index, m, l, O^T x 4
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(m_init)), (DType::F32, Opnd::Val(l_init))];
    let nblk = (D / 32) as usize;
    inits.extend((0..nblk).map(|_| (DType::Frag16, Opnd::Val(zero16))));
    let vars = b.begin_loop(&inits);
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..3 + nblk].to_vec();
    let itx = amdgpu_dsl::varx(it);
    for slot in 0..2i32 {
        let tile = b.bind_i32((itx.clone() << 1) + slot + ev(kt0));
        // top of tile: this tile's DMA done and visible; the other slot free (every wave is past tile - 1)
        b.wait_vm(0);
        b.wait_lds(0);
        b.barrier();
        // the next tile into the other slot (the last pass stages a tile past the end: rows past the sequence, masked; clamped to the pair count)
        let nxt = b.bind_i32(ev(tile) + 1);
        let last = b.bind_i32((ev(npair) << 1) - 1 + ev(kt0));
        let nxt = b.bind_i32(ev(nxt).min(ev(last)));
        let gen_n = if slot == 0 { Opnd::Var(it) } else { Opnd::Val(b.bind_i32(itx.clone() + 1)) };
        stage(&mut b, nxt, 1 - slot, gen_n);
        // ---- S^T = K Q^T: 2 blocks of 32 keys, 8 MFMAs each ----
        let kbase = ks.base as i32 + slot * KT * ROW;
        let s: Vec<ValueId> = ks.at(&SlotIdx::new(slot, Opnd::Var(it))).read(&mut b, |b| {
            (0..2)
                .map(|sb| {
                    let mut acc: Option<ValueId> = None;
                    for c in 0..(D / 16) as usize {
                        let a = b.bind_i32(ev(kr[c]) + kbase + sb * 32 * ROW);
                        let kf = b.load_lds_b128(a);
                        acc = Some(b.mfma(kind, kf, qf[c], acc.map(Opnd::Val)));
                    }
                    acc.expect("8 MFMAs")
                })
                .collect()
        });
        // ---- online softmax over the 64 keys of the tile (lane: one query, values = keys 32 sb + 8 (r / 4) + 4 h + r % 4) ----
        let key0 = b.bind_i32(ev(tile) * KT + (ev(half) << 2));
        let mut sv: Vec<ValueId> = Vec::with_capacity(32);
        for sb in 0..2usize {
            for r in 0..16u8 {
                let x = b.frag16_lane(s[sb], r);
                let x = b.bitcast_f32(x);
                let off = 32 * sb as i32 + 8 * (r as i32 / 4) + (r as i32 % 4);
                let keep = match low {
                    Some(lo) => b.bind_cond((ev(key0) + off).le(ev(lim)).and((ev(key0) + off).ge(ev(lo)))),
                    None => b.bind_cond((ev(key0) + off).le(ev(lim))),
                };
                sv.push(b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(keep), a: Opnd::Val(x), b: Opnd::Val(neg_inf) }));
            }
        }
        let mut mx = sv[0];
        for x in &sv[1..] {
            mx = b.max_f32(mx, *x);
        }
        let (pa, pb) = b.permlane32_swap(mx, mx);
        let (pa, pb) = (b.bitcast_f32(pa), b.bitcast_f32(pb));
        let mt = b.max_f32(pa, pb);
        let mt = b.mul_f32(mt, csc);
        let m_new = b.max_f32(Opnd::Var(m_v), mt);
        let dm = b.sub_f32(Opnd::Var(m_v), m_new);
        let alpha = b.exp2_f32(dm);
        let zf = b.c_f32(0.0);
        let nm = b.sub_f32(zf, m_new);
        let p: Vec<ValueId> = sv
            .iter()
            .map(|x| {
                let t = b.let_(Op::FmaF32(Opnd::Val(*x), Opnd::Val(csc), Opnd::Val(nm)));
                b.exp2_f32(t)
            })
            .collect();
        let mut ps = p[0];
        for x in &p[1..] {
            ps = b.add_f32(ps, *x);
        }
        let l_new = b.let_(Op::FmaF32(Opnd::Var(l_v), Opnd::Val(alpha), Opnd::Val(ps)));
        b.set_var(m_v, m_new);
        b.set_var(l_v, l_new);
        // O^T *= alpha (one query per lane: a scalar)
        for e in 0..nblk {
            for r in 0..16u8 {
                let x = b.frag16_lane(Opnd::Var(ov[e]), r);
                let x = b.bitcast_f32(x);
                let y = b.mul_f32(x, alpha);
                b.var_set_lane(ov[e], r, y);
            }
        }
        // ---- O^T += V^T P^T: 4 steps of 16 keys x 4 blocks of 32 d ----
        let vbase = vs.base as i32 + slot * KT * ROW;
        vs.at(&SlotIdx::new(slot, Opnd::Var(it))).read(&mut b, |b| {
            for t in 0..4usize {
                let (sb, tt) = (t / 2, t % 2);
                // P^T (B operand): k slots = values 8 tt + 0..7 of block sb
                let i0 = 16 * sb + 8 * tt;
                let w: Vec<ValueId> = (0..4).map(|j| b.cvt_pk_bf16_f32(p[i0 + 2 * j], p[i0 + 2 * j + 1])).collect();
                let pf = b.frag4(w[0], w[1], w[2], w[3]);
                for e in 0..nblk as i32 {
                    let words: Vec<ValueId> = (0..4)
                        .map(|j| {
                            let ld = |b: &mut FnBuilder, kk: usize| {
                                let a = b.bind_i32(ev(vr[kk]) + vbase + (32 * sb as i32 + 16 * tt as i32) * ROW + 64 * e);
                                let v = b.load_lds_d16(a, D16Ty::B16, false);
                                b.bind_i32(ev(v) & 0xffff)
                            };
                            let lo = ld(b, 2 * j);
                            let hi = ld(b, 2 * j + 1);
                            b.bind_i32(ev(lo) | (ev(hi) << 16))
                        })
                        .collect();
                    let vf = b.frag4(words[0], words[1], words[2], words[3]);
                    b.var_mfma(ov[e as usize], kind, vf, pf, false);
                }
            }
        });
    }
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(npair)));
    b.break_if(done);
    b.end_loop();

    // ---- epilogue: O = O^T / l (both lane halves' row sums), bf16, 4 consecutive d per 8-byte store; LSE = (m + log2 l) ln 2 ----
    let (la, lb) = b.permlane32_swap(Opnd::Var(l_v), Opnd::Var(l_v));
    let (la, lb) = (b.bitcast_f32(la), b.bitcast_f32(lb));
    let l_tot = b.add_f32(la, lb);
    let inv = b.rcp_f32(l_tot);
    let o_so = { let v = b.bind_i32(ev(bat) * ev(o_bs) + ev(h) * ev(o_hs)); b.read_first_lane(v) };
    // rows past the sequence: an offset past the descriptor (the store is dropped)
    let live = b.bind_cond(ev(row).lt(ev(s_q)));
    let o_vo_r = b.bind_i32(ev(row) * ev(o_seqs) + (ev(half) << 3));
    let big = b.c_i32(0x7fff_fff0);
    let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo_r), b: Opnd::Val(big) });
    for e in 0..nblk {
        for u in 0..4u8 {
            let v: Vec<ValueId> = (0..4u8)
                .map(|k| {
                    let x = b.frag16_lane(Opnd::Var(ov[e]), 4 * u + k);
                    let x = b.bitcast_f32(x);
                    b.mul_f32(x, inv)
                })
                .collect();
            let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
            let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
            let f2 = b.frag2(w0, w1);
            b.buffer_store_b64(r_o, o_vo, o_so, ((32 * e as u32) + 8 * u as u32) * 2, f2);
        }
    }
    let lse_c = b.bind_cond(ev(lse_on).ne(0));
    b.begin_if(Opnd::Val(lse_c));
    {
        let lg = b.log2_f32(l_tot);
        let x = b.add_f32(Opnd::Var(m_v), lg);
        let ln2 = b.c_f32(std::f32::consts::LN_2);
        let lse = b.mul_f32(x, ln2);
        let hq = b.let_(Op::KernargI32 { offset: 320 });
        let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(lse_hs));
        let n = b.read_first_lane(n);
        let r_l = b.buffer_desc_dyn(p_lse, n);
        let so = { let v = b.bind_i32((ev(bat) * ev(hq) + ev(h)) * ev(lse_hs)); b.read_first_lane(v) };
        let vo_r = b.bind_i32(ev(row) << 2);
        let vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(vo_r), b: Opnd::Val(big) });
        b.buffer_store_f32(r_l, vo, so, 0, lse);
    }
    b.end_if();
    b.finish()
}