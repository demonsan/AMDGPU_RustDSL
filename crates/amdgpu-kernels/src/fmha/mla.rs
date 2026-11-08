//! MLA decode (DeepSeek latent attention), bf16: `mla_decode_bf16_p<P>` (batch 3 of docs/op-coverage-design.md). One latent KV head: the
//! cache rows are 576 bf16 (512 latent + 64 rope), K = the whole row, V = its first 512 dims; Q [B, H, 576] (one query per sequence);
//! scale = 1 / sqrt(576) from the host.
//!
//! * Grid (ceil(H / 32), B, S): 32 heads of a CTA are the 32 "rows" (S^T / O^T orientation: one head per lane); S key splits per sequence
//!   (fp32 partial O / l [B, H, S, 512] + LSE [B, H, S], merged by `decode::attn_combine_bf16(.., 512)`).
//! * 32-key tiles in a 2-slot LDS ring (36 KB per slot, rows of 1152 bytes, 16-byte chunk j of key k stored at chunk j ^ (k % 8)); the DMA
//!   writes the tile linearly (LDS chunk c = 72 key + position) and translates every key row through the block table.
//! * QK: wave w computes the partial S^T over its 9 of the 36 16-dim chunks; the 4 partials are summed through LDS (every wave gets the full
//!   S^T); the online softmax runs in every wave (identical); PV: wave w owns v-dims [128 w, 128 w + 128) (4 O^T blocks).
//!
//! ABI (16-byte slots): 0 ptr_po, 1 ptr_q, 2 ptr_kv ([num_pages, P, 576]), 3 ptr_bt, 4 ptr_cl, 5 scale (f32), 6 H, 7 max_pages, 8 num_pages,
//! 9 S, 10 ptr_plse.
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId};
use amdgpu_tile::{ev, LdsPlan, Lane, Layout as TLayout, SlotIdx};

const DK: i32 = 576;
const DV: i32 = 512;
const ROW: i32 = 2 * DK;
/// keys per tile
const KT: i32 = 32;
const CHUNKS: i32 = ROW / 16; // 72

pub fn mla_decode_bf16(name: &str, page: i32) -> Function {
    assert!(page.count_ones() == 1, "page size: a power of two");
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, i) in [("ptr_po", 0u32), ("ptr_q", 1), ("ptr_kv", 2), ("ptr_bt", 3), ("ptr_cl", 4), ("ptr_plse", 10)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    b.arg("scale", DType::F32, 80, 4);
    for (n, i) in [("H", 6u32), ("max_pages", 7), ("num_pages", 8), ("S", 9)] {
        b.arg(n, DType::I32, 16 * i, 4);
    }
    let (hh, mp, np, ss) = (Ext::ka(96), Ext::ka(112), Ext::ka(128), Ext::ka(144));
    b.sym_range("B", 1, 1 << 16);
    b.ka_range(96, 1, 256);
    b.ka_range(112, 1, 1 << 20);
    b.ka_range(128, 1, 1 << 20);
    b.ka_range(144, 1, 64);
    b.grid_is(0, hh.clone().ceil_div(32));
    b.grid_is(1, Ext::sym("B"));
    b.grid_is(2, ss.clone());
    let bb = Ext::sym("B");
    b.arg_tensor("ptr_q", bb.clone() * hh.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_kv", np * Ext::c((page * ROW) as i64));
    b.arg_tensor("ptr_bt", bb.clone() * mp * Ext::c(4));
    b.arg_tensor("ptr_cl", bb.clone() * Ext::c(4));
    b.arg_tensor("ptr_po", bb.clone() * hh.clone() * ss.clone() * Ext::c(4 * DV as i64));
    b.arg_tensor("ptr_plse", bb * hh * ss * Ext::c(4));

    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 16 * i });
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    let (p_po, p_q, p_kv, p_bt, p_cl, p_lse) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4), kp(&mut b, 10));
    let scale = b.let_(Op::KernargF32 { offset: 80 });
    let (hq, maxp, npages, s_n) = (ki(&mut b, 6), ki(&mut b, 7), ki(&mut b, 8), ki(&mut b, 9));
    let hg = b.let_(Op::WorkgroupIdX);
    let bat = b.let_(Op::WorkgroupIdY);
    let z = b.let_(Op::WorkgroupIdZ);
    let lane = Lane::new(&mut b);
    let half = b.bind_i32(lane.half());
    let wv = { let v = b.bind_i32(lane.wave()); b.read_first_lane(v) };
    let c0 = b.c_i32(0);
    // the lane's head (clamped) and whether it is real
    let head_r = b.bind_i32((ev(hg) << 5) + lane.j());
    let row_live = b.bind_cond(ev(head_r).lt(ev(hq)));
    let head = b.bind_i32(ev(head_r).min(ev(hq) - 1));
    // context length and the CTA's key split [k_lo, k_hi): chunks of ceil(tiles / S) tiles (S a power of two <= 64)
    let ncl = { let v = b.bind_i32((ev(bat) + 1) << 2); b.read_first_lane(v) };
    let r_cl = b.buffer_desc_dyn(p_cl, ncl);
    let cl_so = { let v = b.bind_i32(ev(bat) << 2); b.read_first_lane(v) };
    let ctx = { let v = b.buffer_load_f32(r_cl, c0, cl_so, 0); let v = b.bitcast_i32(v); b.read_first_lane(v) };
    b.assume_trusted(ctx, Ext::c(0), Ext::ka(112) * Ext::c(page as i64), "MLA decode: a context length lies in [0, max_pages P] (host contract)");
    let lg = {
        let sv = ev(s_n);
        let mut acc = I32c(0);
        for (thr, val) in [(1, 1), (2, 2), (4, 3), (8, 4), (16, 5), (32, 6)] {
            acc = sv.clone().gt(thr).select_i32(val, acc);
        }
        let v = b.bind_i32(acc);
        b.read_first_lane(v)
    };
    let ntt = b.bind_i32((ev(ctx) + (KT - 1)).lshr(5));
    let per = b.bind_i32(((ev(ntt) + ev(s_n) - 1).lshr(ev(lg))) << 5);
    let k_lo = { let v = b.bind_i32(ev(z) * ev(per)); b.read_first_lane(v) };
    let k_hi = { let v = b.bind_i32((ev(k_lo) + ev(per)).min(ev(ctx))); b.read_first_lane(v) };
    let ntile = { let v = b.bind_i32((ev(k_hi) - ev(k_lo) + (KT - 1)).max(0).lshr(5)); b.read_first_lane(v) };
    let npair = { let v = b.bind_i32((ev(ntile) + 1).lshr(1)); b.read_first_lane(v) };

    // ---- descriptors ----
    let r_q = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ROW); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_q, n) };
    let r_kv = { let n = b.bind_i32(ev(npages) * (page * ROW)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_kv, n) };
    let r_bt = { let n = b.bind_i32((ev(bat) + 1) * ev(maxp) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_bt, n) };
    let bt_so = { let v = b.bind_i32(ev(bat) * ev(maxp) << 2); b.read_first_lane(v) };

    // ---- Q^T: this wave's 9 chunks (dims 16 (9 w + c) + 8 h, 16 bytes) ----
    let q_vo = b.bind_i32((ev(bat) * ev(hq) + ev(head)) * ROW + (ev(half) << 4) + ev(wv) * (9 * 32));
    let qf: Vec<ValueId> = (0..9).map(|c| b.buffer_load_b128(r_q, q_vo, c0, (32 * c) as u32)).collect();

    // ---- LDS: the KV ring (2 x 36 KB) + the S^T reduction scratch (4 waves x 64 lanes x 16 floats) ----
    let mut lds = LdsPlan::new();
    let kvs = lds.ring("KV", (KT * ROW) as i64, 2, TLayout::Linear);
    let red = lds.ring("SRED", 4096, 4, TLayout::Linear);
    lds.commit(&mut b);
    // DMA: 36 KB = 36 blocks of 1 KB, wave w issues blocks 9 w + i; LDS chunk c = 64 block + lane = 72 key + position; the source chunk is
    // position ^ (key % 8). key = c / 72 by a multiply-shift (exact for c < 2304, checked here)
    for c in 0..2304i64 {
        assert_eq!((c * 1821) >> 17, c / 72, "c / 72 multiply-shift");
    }
    let dma_lane: Vec<(ValueId, ValueId)> = (0..9)
        .map(|i| {
            let c = b.bind_i32(((ev(wv) * 9 + i) << 6) + lane.lane());
            let key = b.bind_i32((ev(c) * 1821).lshr(17));
            let pos = b.bind_i32(ev(c) - ev(key) * CHUNKS);
            let src = b.bind_i32(ev(pos) ^ (ev(key) & 7));
            (key, src)
        })
        .collect();
    let maxk = b.bind_i32(ev(maxp) * page - 1);
    let stage = |b: &mut FnBuilder, tile: ValueId, slot: i32, gen: Opnd| {
        let key_t = { let v = b.bind_i32(ev(k_lo) + (ev(tile) << 5)); b.read_first_lane(v) };
        let base = kvs.base as i32 + slot * KT * ROW;
        kvs.at(&SlotIdx::new(slot, gen)).write(b, |b| {
            for (i, (key, src)) in dma_lane.iter().enumerate() {
                let kc = b.bind_i32((ev(key_t) + ev(*key)).min(ev(k_hi) - 1).max(0).min(ev(maxk)));
                let bvo = b.bind_i32(ev(kc).lshr(page.trailing_zeros() as i32) << 2);
                let pv = b.buffer_load_f32(r_bt, bvo, bt_so, 0);
                let pg = b.bitcast_i32(pv);
                let pg = b.bind_i32(ev(pg).max(0).min(ev(npages) - 1));
                let vo = b.bind_i32((ev(pg) * page + (ev(kc) & (page - 1))) * ROW + (ev(*src) << 4));
                let m0 = { let x = b.bind_i32(((ev(wv) * 9 + i as i32) << 10) + base); b.read_first_lane(x) };
                b.buffer_load_lds_b128(r_kv, vo, c0, m0, 0);
            }
        });
    };
    // per-lane read offsets: K (key l % 32, chunk (2 (9 w + c) + h) ^ (key % 8)); V^T tr_b16 (key row 4 h + s / 4, s = l % 16; dims
    // 128 w + 32 e + 16 ((l % 32) / 16) + 4 (s % 4); chunk ^ (key % 8))
    let kr: Vec<ValueId> = (0..9)
        .map(|c| b.bind_i32(lane.j() * ROW + (((ev(wv) * 9 + c) * 2 + ev(half)) ^ (lane.lane() & 7)) * 16))
        .collect();
    let vkey = b.bind_i32(ev(half) * 4 + (lane.lane() & 15).lshr(2));
    let vr: Vec<ValueId> = (0..4)
        .map(|e| {
            let dim = b.bind_i32(ev(wv) * 128 + 32 * e + ((lane.lane() & 31).lshr(4) << 4) + ((lane.lane() & 3) << 2));
            b.bind_i32(ev(vkey) * ROW + ((ev(dim).lshr(3) ^ (ev(vkey) & 7)) << 4) + ((ev(dim) & 7) << 1))
        })
        .collect();
    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let kind = MmaKind::new(MmaShape::M32N32K16, MmaTy::Bf16);
    let zero16 = b.frag_zero(DType::Frag16);
    let m_init = b.c_f32(-1.0e30);
    let l_init = b.c_f32(0.0);
    let rbase = red.base as i32;

    let any = b.bind_cond(ev(ntile).gt(0));
    b.begin_if(Opnd::Val(any));
    let t0 = b.c_i32(0);
    stage(&mut b, t0, 0, Opnd::I32(0));
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0)), (DType::F32, Opnd::Val(m_init)), (DType::F32, Opnd::Val(l_init))];
    inits.extend((0..4).map(|_| (DType::Frag16, Opnd::Val(zero16))));
    let vars = b.begin_loop(&inits);
    let (it, m_v, l_v) = (vars[0], vars[1], vars[2]);
    let ov: Vec<VarId> = vars[3..7].to_vec();
    let itx = amdgpu_dsl::varx(it);
    for slot in 0..2i32 {
        let tile = b.bind_i32((itx.clone() << 1) + slot);
        b.wait_vm(0);
        b.wait_lds(0);
        b.barrier();
        let last = b.bind_i32((ev(npair) << 1) - 1);
        let nxt = b.bind_i32((ev(tile) + 1).min(ev(last)));
        let gen_n = if slot == 0 { Opnd::Var(it) } else { Opnd::Val(b.bind_i32(itx.clone() + 1)) };
        stage(&mut b, nxt, 1 - slot, gen_n);
        let kbase = kvs.base as i32 + slot * KT * ROW;
        let sl = kvs.at(&SlotIdx::new(slot, Opnd::Var(it)));
        // ---- partial S^T over this wave's 9 chunks ----
        let sp = sl.read(&mut b, |b| {
            let mut acc: Option<ValueId> = None;
            for c in 0..9usize {
                let a = b.bind_i32(ev(kr[c]) + kbase);
                let kf = b.load_lds_b128(a);
                acc = Some(b.mfma(kind, kf, qf[c], acc.map(Opnd::Val)));
            }
            acc.expect("9 MFMAs")
        });
        // ---- the 4 partials summed through LDS: wave w writes its 16 floats (lane-major), barrier, every wave reads all 4 ----
        let wa = b.bind_i32(lane.lane() * 64 + rbase + ev(wv) * 4096);
        red.at(&SlotIdx::new(Opnd::Val(wv), Opnd::Var(it))).write(&mut b, |b| {
            for q in 0..4u8 {
                let w: Vec<ValueId> = (0..4u8).map(|k| b.frag16_lane(sp, 4 * q + k)).collect();
                let f = b.frag4(w[0], w[1], w[2], w[3]);
                let a = b.bind_i32(ev(wa) + 16 * q as i32);
                b.store_lds_b128(a, f);
            }
        });
        b.wait_lds(0);
        b.barrier();
        let mut sum: Vec<ValueId> = Vec::new();
        for w in 0..4i32 {
            let part = red.at(&SlotIdx::new(w, Opnd::Var(it)));
            let vals: Vec<ValueId> = part.read(&mut b, |b| {
                (0..4)
                    .flat_map(|q| {
                        let a = b.bind_i32(lane.lane() * 64 + rbase + w * 4096 + 16 * q);
                        let f = b.load_lds_b128(a);
                        (0..4u8).map(|k| b.frag4_lane(f, k)).collect::<Vec<_>>()
                    })
                    .collect()
            });
            sum = if sum.is_empty() {
                vals.into_iter().map(|v| b.bitcast_f32(v)).collect()
            } else {
                sum.iter().zip(vals).map(|(s, v)| { let v = b.bitcast_f32(v); b.add_f32(*s, v) }).collect()
            };
        }
        // ---- online softmax over the 32 keys (values: keys 8 (r / 4) + 4 h + r % 4) ----
        let key0 = b.bind_i32(ev(k_lo) + (ev(tile) << 5) + (ev(half) << 2));
        let sv: Vec<ValueId> = (0..16usize)
            .map(|r| {
                let off = 8 * (r as i32 / 4) + (r as i32 % 4);
                let keep = b.bind_cond((ev(key0) + off).lt(ev(k_hi)));
                b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(keep), a: Opnd::Val(sum[r]), b: Opnd::Val(neg_inf) })
            })
            .collect();
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
        for e in 0..4usize {
            for r in 0..16u8 {
                let x = b.frag16_lane(Opnd::Var(ov[e]), r);
                let x = b.bitcast_f32(x);
                let y = b.mul_f32(x, alpha);
                b.var_set_lane(ov[e], r, y);
            }
        }
        // ---- O^T (this wave's 128 v-dims) += V^T P^T: 2 steps of 16 keys ----
        sl.read(&mut b, |b| {
            for tt in 0..2usize {
                let i0 = 8 * tt;
                let w: Vec<ValueId> = (0..4).map(|j| b.cvt_pk_bf16_f32(p[i0 + 2 * j], p[i0 + 2 * j + 1])).collect();
                let pf = b.frag4(w[0], w[1], w[2], w[3]);
                for e in 0..4usize {
                    let mut words: Vec<ValueId> = Vec::with_capacity(4);
                    for r in 0..2i32 {
                        let a = b.bind_i32(ev(vr[e]) + kbase + (16 * tt as i32 + 8 * r) * ROW);
                        let x = b.load_lds_tr_b16(a);
                        let w0 = b.frag2_lane(x, 0);
                        let w1 = b.frag2_lane(x, 1);
                        words.push(b.bitcast_i32(w0));
                        words.push(b.bitcast_i32(w1));
                    }
                    let vf = b.frag4(words[0], words[1], words[2], words[3]);
                    b.var_mfma(ov[e], kind, vf, pf, false);
                }
            }
        });
    }
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(npair)));
    b.break_if(done);
    b.end_loop();
    // ---- the split's partial: O / l and LSE ----
    let (la, lb) = b.permlane32_swap(Opnd::Var(l_v), Opnd::Var(l_v));
    let (la, lb) = (b.bitcast_f32(la), b.bitcast_f32(lb));
    let l_tot = b.add_f32(la, lb);
    let zf = b.c_f32(0.0);
    let lbits = b.bitcast_i32(l_tot);
    let pos = b.bind_cond(ev(lbits).gt(0));
    let inv0 = b.rcp_f32(l_tot);
    let inv = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(inv0), b: Opnd::Val(zf) });
    let big = b.c_i32(0x7fff_fff0);
    let row_po = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(s_n) + ev(z)) * (4 * DV) + ev(wv) * (4 * 128));
    let po_vo_r = b.bind_i32(ev(row_po) + (ev(half) << 4));
    let po_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(po_vo_r), b: Opnd::Val(big) });
    let r_po = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(s_n) * (4 * DV)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    for e in 0..4usize {
        for u in 0..4u8 {
            let v: Vec<ValueId> = (0..4u8)
                .map(|k| {
                    let x = b.frag16_lane(Opnd::Var(ov[e]), 4 * u + k);
                    let x = b.bitcast_f32(x);
                    let y = b.mul_f32(x, inv);
                    b.bitcast_i32(y)
                })
                .collect();
            let f4 = b.frag4(v[0], v[1], v[2], v[3]);
            b.buffer_store_b128(r_po, po_vo, c0, ((32 * e as u32) + 8 * u as u32) * 4, f4);
        }
    }
    let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(s_n) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
    let l_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(s_n) + ev(z)) << 2);
    let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
    let lgl = b.log2_f32(l_tot);
    let x = b.add_f32(Opnd::Var(m_v), lgl);
    let ln2 = b.c_f32(std::f32::consts::LN_2);
    let lse0 = b.mul_f32(x, ln2);
    let lse = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(pos), a: Opnd::Val(lse0), b: Opnd::Val(neg_inf) });
    let w0 = b.bind_cond(ev(wv).eq(0));
    b.begin_if(Opnd::Val(w0));
    b.buffer_store_f32(r_l, l_vo, c0, 0, lse);
    b.end_if();
    b.else_();
    {
        let r_l = { let n = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(s_n) * 4); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_lse, n) };
        let l_vo_r = b.bind_i32(((ev(bat) * ev(hq) + ev(head)) * ev(s_n) + ev(z)) << 2);
        let big = b.c_i32(0x7fff_fff0);
        let l_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_live), a: Opnd::Val(l_vo_r), b: Opnd::Val(big) });
        b.buffer_store_f32(r_l, l_vo, c0, 0, neg_inf);
    }
    b.end_if();
    b.finish()
}

#[allow(non_snake_case)]
fn I32c(v: i32) -> amdgpu_dsl::I32 {
    amdgpu_dsl::I32::c(v)
}
