//! Attention backward, bf16, D = 128 (batch 5 of docs/op-coverage-design.md), FlashAttention-2 split into three kernels without atomics:
//! * `fmha_bwd_dot_hd128_bf16`: Dot[b, h, q] = sum_d dO O (fp32);
//! * `fmha_bwd_dq_hd128_bf16[_causal]`: per 128-query tile (the forward orientation, one query per lane): S^T = K Q^T, P^T = exp(scale S^T -
//!   LSE), dP^T = V dO^T, dS^T = P^T (dP^T - Dot), dQ^T += K^T dS^T; dQ = scale dQ (bf16);
//! * `fmha_bwd_dkdv_hd128_bf16[_causal]`: per 128-key tile (one key per lane), over the queries of every head of the GQA group: S = Q K^T,
//!   P = exp(scale S - LSE), dV += P^T dO, dP = dO V^T, dS = P (dP - Dot), dK += dS^T Q; dK = scale dK.
//!
//! ABI (16-byte slots): 0 q, 1 k, 2 v, 3 do, 4 lse (fp32 [B, H, SQ], natural log, the forward's), 5 dot (fp32 [B, H, SQ]), 6 dq, 7 dk, 8 dv,
//! 9 scale (f32), 10 SQ, 11 SK, 12 H, 13 HK, 14 o (the dot kernel). Q / dO / O / dQ [B, SQ, H, 128], K / V [B, SK, HK, 128], dK / dV per Q head
//! [B, SK, H, 128] (dense),
//! causal = bottom-right (key k visible to query q iff k <= q + SK - SQ).
use amdgpu_tile::components::gqa_kv_head;
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId};
use amdgpu_tile::{ev, LdsPlan, Lane, Layout as TLayout, SlotIdx};

const D: i32 = 128;
const ROW: i32 = 2 * D;
const KT: i32 = 64;

fn abi(b: &mut FnBuilder) {
    for (n, i) in [("q", 0u32), ("k", 1), ("v", 2), ("do", 3), ("lse", 4), ("dot", 5), ("dq", 6), ("dk", 7), ("dv", 8), ("o", 14)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    b.arg("scale", DType::F32, 144, 4);
    for (n, i) in [("SQ", 10u32), ("SK", 11), ("H", 12), ("HK", 13)] {
        b.arg(n, DType::I32, 16 * i, 4);
    }
    b.ka_range(160, 1, 1 << 20);
    b.ka_range(176, 1, 1 << 20);
    b.sym_range("G", 1, 16);
    b.sym_range("B", 1, 1 << 12);
    b.ka_range(208, 1, 64);
    b.ka_is(192, Ext::ka(208) * Ext::sym("G"));
    let (bb, sq, sk, hh, hk) = (Ext::sym("B"), Ext::ka(160), Ext::ka(176), Ext::ka(192), Ext::ka(208));
    for n in ["q", "do", "dq", "o"] {
        b.arg_tensor(n, bb.clone() * sq.clone() * hh.clone() * Ext::c(ROW as i64));
    }
    for n in ["k", "v"] {
        b.arg_tensor(n, bb.clone() * sk.clone() * hk.clone() * Ext::c(ROW as i64));
    }
    // dK / dV per Q head (the vendor convention; a GQA group is summed afterwards)
    for n in ["dk", "dv"] {
        b.arg_tensor(n, bb.clone() * sk.clone() * hh.clone() * Ext::c(ROW as i64));
    }
    for n in ["lse", "dot"] {
        b.arg_tensor(n, bb.clone() * hh.clone() * sq.clone() * Ext::c(4));
    }
}

struct Args {
    p: Vec<ValueId>,
    scale: ValueId,
    sq: ValueId,
    sk: ValueId,
    hq: ValueId,
    hkn: ValueId,
}

fn args(b: &mut FnBuilder) -> Args {
    let p: Vec<ValueId> = [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 14].iter().map(|i| b.let_(Op::KernargPtr { offset: 16 * i })).collect();
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    Args { p, scale: b.let_(Op::KernargF32 { offset: 144 }), sq: ki(b, 10), sk: ki(b, 11), hq: ki(b, 12), hkn: ki(b, 13) }
}

/// Dot[b, h, q] = sum_d dO[b, q, h, d] O[b, q, h, d]. Grid (ceil(SQ / 256), H, B), one row per thread.
pub fn fmha_bwd_dot(name: &str) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    abi(&mut b);
    b.grid_is(0, Ext::ka(160).ceil_div(256));
    b.grid_is(1, Ext::ka(192));
    b.grid_is(2, Ext::sym("B"));
    let a = args(&mut b);
    let (qt, h, bat) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    let tid = b.let_(Op::WorkitemId);
    let row = b.bind_i32((ev(qt) << 8) + ev(tid));
    let live = b.bind_cond(ev(row).lt(ev(a.sq)));
    let rc = b.bind_i32(ev(row).min(ev(a.sq) - 1));
    let nrow = { let v = b.bind_i32((ev(bat) + 1) * ev(a.sq) * ev(a.hq) * ROW); b.read_first_lane(v) };
    let (r_do, r_o) = (b.buffer_desc_dyn(a.p[3], nrow), b.buffer_desc_dyn(a.p[9], nrow));
    let vo = b.bind_i32(((ev(bat) * ev(a.sq) + ev(rc)) * ev(a.hq) + ev(h)) * ROW);
    let c0 = b.c_i32(0);
    let mut acc = b.c_f32(0.0);
    for c in 0..(ROW / 16) {
        let x = b.buffer_load_b128(r_do, vo, c0, (16 * c) as u32);
        let y = b.buffer_load_b128(r_o, vo, c0, (16 * c) as u32);
        for w in 0..4u8 {
            let (xw, yw) = (b.frag4_lane(x, w), b.frag4_lane(y, w));
            for hi in [false, true] {
                let f = |b: &mut FnBuilder, v: ValueId| {
                    let u = b.bitcast_i32(v);
                    let s = if hi { b.bind_i32(ev(u) & (0xffff_0000u32 as i32)) } else { b.bind_i32(ev(u) << 16) };
                    b.bitcast_f32(s)
                };
                let (xf, yf) = (f(&mut b, xw), f(&mut b, yw));
                acc = b.let_(Op::FmaF32(Opnd::Val(xf), Opnd::Val(yf), Opnd::Val(acc)));
            }
        }
    }
    let nl = { let v = b.bind_i32((ev(bat) + 1) * ev(a.hq) * ev(a.sq) << 2); b.read_first_lane(v) };
    let r_dot = b.buffer_desc_dyn(a.p[5], nl);
    let big = b.c_i32(0x7fff_fff0);
    let lo = b.bind_i32(((ev(bat) * ev(a.hq) + ev(h)) * ev(a.sq) + ev(row)) << 2);
    let lo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(lo), b: Opnd::Val(big) });
    b.buffer_store_f32(r_dot, lo, c0, 0, acc);
    b.finish()
}

/// dQ per 128-query tile (see the module doc). Grid (ceil(SQ / 128), H, B), 4 waves x 32 queries.
pub fn fmha_bwd_dq(name: &str, causal: bool) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Vs128);
    abi(&mut b);
    b.grid_is(0, Ext::ka(160).ceil_div(128));
    b.grid_is(1, Ext::ka(192));
    b.grid_is(2, Ext::sym("B"));
    let a = args(&mut b);
    let (qt, h, bat) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    let lane = Lane::new(&mut b);
    // GQA group G = H / HK (power of two): the KV head
    let gqa = {
        // H / HK by shifts (power of two up to 16)
        let mut q = b.c_i32(1);
        for s in 1..=4 {
            let t = b.bind_i32((ev(a.hkn) << s).le(ev(a.hq)).select_i32(1 << s, ev(q)));
            q = t;
        }
        b.read_first_lane(q)
    };
    let hkv = gqa_kv_head(&mut b, h, gqa, "HK");
    let hkv = b.read_first_lane(hkv);
    let half = b.bind_i32(lane.half());
    let row = b.bind_i32((ev(qt) << 7) + (lane.wave() << 5) + lane.j());
    let row_c = b.bind_i32(ev(row).min(ev(a.sq) - 1));
    let c0 = b.c_i32(0);
    let qbytes = { let v = b.bind_i32((ev(bat) + 1) * ev(a.sq) * ev(a.hq) * ROW); b.read_first_lane(v) };
    let kbytes = { let v = b.bind_i32((ev(bat) + 1) * ev(a.sk) * ev(a.hkn) * ROW); b.read_first_lane(v) };
    let (r_q, r_do, r_dq) = (b.buffer_desc_dyn(a.p[0], qbytes), b.buffer_desc_dyn(a.p[3], qbytes), b.buffer_desc_dyn(a.p[6], qbytes));
    let (r_k, r_v) = (b.buffer_desc_dyn(a.p[1], kbytes), b.buffer_desc_dyn(a.p[2], kbytes));
    // Q^T / dO^T fragments (B operand: lane = query, k = d 16 c + 8 h)
    let q_vo = b.bind_i32(((ev(bat) * ev(a.sq) + ev(row_c)) * ev(a.hq) + ev(h)) * ROW + (ev(half) << 4));
    let qf: Vec<ValueId> = (0..8).map(|c| b.buffer_load_b128(r_q, q_vo, c0, (32 * c) as u32)).collect();
    let dof: Vec<ValueId> = (0..8).map(|c| b.buffer_load_b128(r_do, q_vo, c0, (32 * c) as u32)).collect();
    // LSE (natural) -> log2 units, Dot
    let nl = { let v = b.bind_i32((ev(bat) + 1) * ev(a.hq) * ev(a.sq) << 2); b.read_first_lane(v) };
    let (r_l, r_d) = (b.buffer_desc_dyn(a.p[4], nl), b.buffer_desc_dyn(a.p[5], nl));
    let lo = b.bind_i32(((ev(bat) * ev(a.hq) + ev(h)) * ev(a.sq) + ev(row_c)) << 2);
    let lse = b.buffer_load_f32(r_l, lo, c0, 0);
    let dot = b.buffer_load_f32(r_d, lo, c0, 0);
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let lse2 = b.mul_f32(lse, l2e);
    let csc = b.mul_f32(a.scale, l2e);
    let nlse = { let z = b.c_f32(0.0); b.sub_f32(z, lse2) };

    // LDS: K / V rings (2 x 16 KB), both XOR-swizzled (16-byte chunk c of key k at c ^ (k % 16))
    let mut lds = LdsPlan::new();
    let ks = lds.ring("K", (KT * ROW) as i64, 2, TLayout::Linear);
    let vs = lds.ring("V", (KT * ROW) as i64, 2, TLayout::Linear);
    lds.commit(&mut b);
    let k_so0 = { let v = b.bind_i32((ev(bat) * ev(a.sk) * ev(a.hkn) + ev(hkv)) * ROW); b.read_first_lane(v) };
    let kseq = { let v = b.bind_i32(ev(a.hkn) * ROW); b.read_first_lane(v) };
    let l16 = b.bind_i32(lane.lane() & 15);
    let dvo: Vec<ValueId> = (0..4)
        .map(|i| {
            let key = b.bind_i32(((lane.wave() + 4 * i) << 2) + lane.lane().lshr(4));
            b.bind_i32(ev(key) * ev(kseq) + ((ev(l16) ^ (ev(key) & 15)) << 4))
        })
        .collect();
    let stage = |b: &mut FnBuilder, tile: ValueId, slot: i32, gen: Opnd| {
        let so = { let v = b.bind_i32(ev(k_so0) + ev(tile) * KT * ev(kseq)); b.read_first_lane(v) };
        for (sh, r) in [(&ks, r_k), (&vs, r_v)] {
            let base = sh.base as i32 + slot * KT * ROW;
            sh.at(&SlotIdx::new(slot, gen.clone())).write(b, |b| {
                for (i, v) in dvo.iter().enumerate() {
                    let m0 = { let x = b.bind_i32(((lane.wave() + 4 * i as i32) << 10) + base); b.read_first_lane(x) };
                    b.buffer_load_lds_b128(r, *v, so, m0, 0);
                }
            });
        }
    };
    // per-lane read offsets: K / V rows as A operands (key l % 32, chunk (2 c + h) ^ (l % 16)); K^T by tr_b16 (key row 8 r + 4 h + s / 4,
    // s = l % 16; dims 32 e + 16 ((l % 32) / 16) + 4 (s % 4), swizzled)
    let kr: Vec<ValueId> = (0..8).map(|c| b.bind_i32(lane.j() * ROW + (((ev(half) + 2 * c) ^ (lane.lane() & 15)) << 4))).collect();
    let ktr: Vec<Vec<ValueId>> = (0..4)
        .map(|e| {
            (0..2)
                .map(|r| {
                    let krow = b.bind_i32(8 * r + ev(half) * 4 + (lane.lane() & 15).lshr(2));
                    let dim = b.bind_i32(32 * e + ((lane.lane() & 31).lshr(4) << 4) + ((lane.lane() & 3) << 2));
                    b.bind_i32(ev(krow) * ROW + (((ev(dim).lshr(3) ^ ev(krow)) & 15) << 4) + ((lane.lane() & 1) << 3))
                })
                .collect()
        })
        .collect();
    // key range: causal keys <= last query of the tile + SK - SQ
    let kv_end = if causal {
        let v = b.bind_i32((((ev(qt) + 1) << 7) - 1 + ev(a.sk) - ev(a.sq) + 1).min(ev(a.sk)).max(0));
        b.read_first_lane(v)
    } else {
        a.sk
    };
    let npair = { let v = b.bind_i32((ev(kv_end) + 127).lshr(7)); b.read_first_lane(v) };
    let lim = b.bind_i32(if causal { (ev(row) + ev(a.sk) - ev(a.sq)).min(ev(a.sk) - 1) } else { ev(a.sk) - 1 });
    let kind = MmaKind::new(MmaShape::M32N32K16, MmaTy::Bf16);
    let zero16 = b.frag_zero(DType::Frag16);
    let zf = b.c_f32(0.0);

    let any = b.bind_cond(ev(npair).gt(0));
    b.begin_if(Opnd::Val(any));
    let t0 = b.c_i32(0);
    stage(&mut b, t0, 0, Opnd::I32(0));
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0))];
    inits.extend((0..8).map(|_| (DType::Frag16, Opnd::Val(zero16))));
    let vars = b.begin_loop(&inits);
    let it = vars[0];
    let dqv: Vec<VarId> = vars[1..5].to_vec();
    let sv_vars: Vec<VarId> = vars[5..7].to_vec();
    let dp_vars: Vec<VarId> = vars[7..9].to_vec();
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
        let kbase = ks.base as i32 + slot * KT * ROW;
        let vbase = vs.base as i32 + slot * KT * ROW;
        let (kslot, vslot) = (ks.at(&SlotIdx::new(slot, Opnd::Var(it))), vs.at(&SlotIdx::new(slot, Opnd::Var(it))));
        // S^T = K Q^T, dP^T = V dO^T (2 blocks of 32 keys)
        for c in 0..8usize {
            kslot.read(&mut b, |b| {
                for sb in 0..2usize {
                    let ad = b.bind_i32(ev(kr[c]) + kbase + sb as i32 * 32 * ROW);
                    let kf = b.load_lds_b128(ad);
                    b.var_mfma(sv_vars[sb], kind, kf, qf[c], c == 0);
                }
            });
            vslot.read(&mut b, |b| {
                for sb in 0..2usize {
                    let ad = b.bind_i32(ev(kr[c]) + vbase + sb as i32 * 32 * ROW);
                    let vf = b.load_lds_b128(ad);
                    b.var_mfma(dp_vars[sb], kind, vf, dof[c], c == 0);
                }
            });
        }
        // P^T = exp2(csc S - lse2) (masked keys 0), dS^T = P^T (dP^T - Dot)
        let key0 = b.bind_i32(ev(tile) * KT + (ev(half) << 2));
        let mut ds: Vec<ValueId> = Vec::with_capacity(32);
        for sb in 0..2usize {
            for r in 0..16u8 {
                let off = 32 * sb as i32 + 8 * (r as i32 / 4) + (r as i32 % 4);
                let s = b.frag16_lane(Opnd::Var(sv_vars[sb]), r);
                let s = b.bitcast_f32(s);
                let t = b.let_(Op::FmaF32(Opnd::Val(s), Opnd::Val(csc), Opnd::Val(nlse)));
                let p = b.exp2_f32(t);
                let keep = b.bind_cond((ev(key0) + off).le(ev(lim)));
                let p = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(keep), a: Opnd::Val(p), b: Opnd::Val(zf) });
                let dp = b.frag16_lane(Opnd::Var(dp_vars[sb]), r);
                let dp = b.bitcast_f32(dp);
                let x = b.sub_f32(dp, dot);
                ds.push(b.mul_f32(p, x));
            }
        }
        // dQ^T += K^T dS^T: 4 steps of 16 keys x 4 blocks of 32 d
        kslot.read(&mut b, |b| {
            for t in 0..4usize {
                let (sb, tt) = (t / 2, t % 2);
                let i0 = 16 * sb + 8 * tt;
                let w: Vec<ValueId> = (0..4).map(|j| b.cvt_pk_bf16_f32(ds[i0 + 2 * j], ds[i0 + 2 * j + 1])).collect();
                let pf = b.pack4(w[0], w[1], w[2], w[3]);
                for e in 0..4usize {
                    let a0 = b.bind_i32(ev(ktr[e][0]) + kbase + (32 * sb as i32 + 16 * tt as i32) * ROW);
                    let a1 = b.bind_i32(ev(ktr[e][1]) + kbase + (32 * sb as i32 + 16 * tt as i32) * ROW);
                    let kf = b.load_lds_tr_b16x2(a0, a1);
                    b.var_mfma(dqv[e], kind, kf, pf, false);
                }
            }
        });
    }
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(npair)));
    b.break_if(done);
    b.end_loop();
    // dQ = scale dQ^T^T (bf16, 4 consecutive d per 8-byte store)
    let live = b.bind_cond(ev(row).lt(ev(a.sq)));
    let big = b.c_i32(0x7fff_fff0);
    let o_vo = b.bind_i32(((ev(bat) * ev(a.sq) + ev(row)) * ev(a.hq) + ev(h)) * ROW + (ev(half) << 3));
    let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo), b: Opnd::Val(big) });
    for e in 0..4usize {
        for u in 0..4u8 {
            let v: Vec<ValueId> = (0..4u8)
                .map(|k| {
                    let x = b.frag16_lane(Opnd::Var(dqv[e]), 4 * u + k);
                    let x = b.bitcast_f32(x);
                    b.mul_f32(x, a.scale)
                })
                .collect();
            let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
            let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
            let f2 = b.frag2(w0, w1);
            b.buffer_store_b64(r_dq, o_vo, c0, ((32 * e as u32) + 8 * u as u32) * 2, f2);
        }
    }
    b.end_if();
    let mut f = b.finish();
    // (as dkdv: Vs128 + the scheduler tokens; dq 224 -> 222 us)
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}

/// dK / dV per 128-key tile and query head (dK, dV written per Q head [B, SK, H, 128] like the vendor: the host / a reduce sums a GQA group).
/// Grid (ceil(SK / 128), H, B), 4 waves x 32 keys; loop over 64-query tiles (Q and dO through a 2-slot LDS ring).
pub fn fmha_bwd_dkdv(name: &str, causal: bool) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Vs128);
    abi(&mut b);
    b.grid_is(0, Ext::ka(176).ceil_div(128));
    b.grid_is(1, Ext::ka(192));
    b.grid_is(2, Ext::sym("B"));
    // per-region scheduler strategies (docs/weaver-sched.md): the list scheduler on the S / dP and dV / dK MFMA regions (ldshoist: the LDS operand reads of a chain are hoisted ahead of its MFMAs instead of the
    // strict `ds_read -> lgkmcnt(0) -> mfma` pattern on one register) and on the softmax / pack region (with hazard-nop filling): dkdv 139.5 -> 114.4 us (B1 H8 2048, non-causal), registers 328 -> 360 (one wave per SIMD).
    b.declare_region("dkdv_sdp_mma", false, "reorder,ldshoist");
    b.declare_region("dkdv_dv_dk_mma", false, "reorder,ldshoist");
    b.declare_region("dkdv_softmax_pack", false, "reorder,hzfill");
    let a = args(&mut b);
    let (kt, h, bat) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    let lane = Lane::new(&mut b);
    let gqa = {
        let mut q = b.c_i32(1);
        for s in 1..=4 {
            let t = b.bind_i32((ev(a.hkn) << s).le(ev(a.hq)).select_i32(1 << s, ev(q)));
            q = t;
        }
        b.read_first_lane(q)
    };
    let hkv = gqa_kv_head(&mut b, h, gqa, "HK");
    let hkv = b.read_first_lane(hkv);
    let half = b.bind_i32(lane.half());
    let key = b.bind_i32((ev(kt) << 7) + (lane.wave() << 5) + lane.j());
    let key_c = b.bind_i32(ev(key).min(ev(a.sk) - 1));
    let c0 = b.c_i32(0);
    let qbytes = { let v = b.bind_i32((ev(bat) + 1) * ev(a.sq) * ev(a.hq) * ROW); b.read_first_lane(v) };
    let kbytes = { let v = b.bind_i32((ev(bat) + 1) * ev(a.sk) * ev(a.hkn) * ROW); b.read_first_lane(v) };
    let dkbytes = { let v = b.bind_i32((ev(bat) + 1) * ev(a.sk) * ev(a.hq) * ROW); b.read_first_lane(v) };
    let (r_q, r_do) = (b.buffer_desc_dyn(a.p[0], qbytes), b.buffer_desc_dyn(a.p[3], qbytes));
    let (r_k, r_v) = (b.buffer_desc_dyn(a.p[1], kbytes), b.buffer_desc_dyn(a.p[2], kbytes));
    let (r_dk, r_dv) = (b.buffer_desc_dyn(a.p[7], dkbytes), b.buffer_desc_dyn(a.p[8], dkbytes));
    // K / V fragments (B operands: lane = key, k = d 16 c + 8 h)
    let k_vo = b.bind_i32(((ev(bat) * ev(a.sk) + ev(key_c)) * ev(a.hkn) + ev(hkv)) * ROW + (ev(half) << 4));
    let kf: Vec<ValueId> = (0..8).map(|c| b.buffer_load_b128(r_k, k_vo, c0, (32 * c) as u32)).collect();
    let vf: Vec<ValueId> = (0..8).map(|c| b.buffer_load_b128(r_v, k_vo, c0, (32 * c) as u32)).collect();
    let nl = { let v = b.bind_i32((ev(bat) + 1) * ev(a.hq) * ev(a.sq) << 2); b.read_first_lane(v) };
    let (r_l, r_d) = (b.buffer_desc_dyn(a.p[4], nl), b.buffer_desc_dyn(a.p[5], nl));
    let l_so = { let v = b.bind_i32((ev(bat) * ev(a.hq) + ev(h)) * ev(a.sq) << 2); b.read_first_lane(v) };
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(a.scale, l2e);

    // LDS: Q / dO rings (2 x 16 KB), XOR-swizzled rows
    let mut lds = LdsPlan::new();
    let qs = lds.ring("Q", (KT * ROW) as i64, 2, TLayout::Linear);
    let ds_ = lds.ring("dO", (KT * ROW) as i64, 2, TLayout::Linear);
    lds.commit(&mut b);
    let q_so0 = { let v = b.bind_i32((ev(bat) * ev(a.sq) * ev(a.hq) + ev(h)) * ROW); b.read_first_lane(v) };
    let qseq = { let v = b.bind_i32(ev(a.hq) * ROW); b.read_first_lane(v) };
    let l16 = b.bind_i32(lane.lane() & 15);
    let dvo: Vec<ValueId> = (0..4)
        .map(|i| {
            let r = b.bind_i32(((lane.wave() + 4 * i) << 2) + lane.lane().lshr(4));
            // rows past SQ read inside the batch (finite data; their P is masked to 0)
            b.bind_i32(ev(r) * ev(qseq) + ((ev(l16) ^ (ev(r) & 15)) << 4))
        })
        .collect();
    let stage = |b: &mut FnBuilder, tile: ValueId, slot: i32, gen: Opnd| {
        let so = { let v = b.bind_i32(ev(q_so0) + ev(tile) * KT * ev(qseq)); b.read_first_lane(v) };
        for (sh, r) in [(&qs, r_q), (&ds_, r_do)] {
            let base = sh.base as i32 + slot * KT * ROW;
            sh.at(&SlotIdx::new(slot, gen.clone())).write(b, |b| {
                for (i, v) in dvo.iter().enumerate() {
                    let m0 = { let x = b.bind_i32(((lane.wave() + 4 * i as i32) << 10) + base); b.read_first_lane(x) };
                    b.buffer_load_lds_b128(r, *v, so, m0, 0);
                }
            });
        }
    };
    // read offsets: Q / dO rows as A (row l % 32, chunk (2 c + h) ^ (l % 16)); Q^T / dO^T by tr_b16 (row 8 r + 4 h + s / 4, dims 32 e + ..)
    let rr: Vec<ValueId> = (0..8).map(|c| b.bind_i32(lane.j() * ROW + (((ev(half) + 2 * c) ^ (lane.lane() & 15)) << 4))).collect();
    let tr: Vec<Vec<ValueId>> = (0..4)
        .map(|e| {
            (0..2)
                .map(|r| {
                    let krow = b.bind_i32(8 * r + ev(half) * 4 + (lane.lane() & 15).lshr(2));
                    let dim = b.bind_i32(32 * e + ((lane.lane() & 31).lshr(4) << 4) + ((lane.lane() & 3) << 2));
                    b.bind_i32(ev(krow) * ROW + (((ev(dim).lshr(3) ^ ev(krow)) & 15) << 4) + ((lane.lane() & 1) << 3))
                })
                .collect()
        })
        .collect();
    // query range: causal queries q >= key - (SK - SQ) for the first key of the CTA
    let qt0 = if causal {
        let v = b.bind_i32(((ev(kt) << 7) - ev(a.sk) + ev(a.sq)).max(0).lshr(6));
        b.read_first_lane(v)
    } else {
        b.c_i32(0)
    };
    let npair = { let v = b.bind_i32((((ev(a.sq) + (KT - 1)).lshr(6) - ev(qt0)).max(0) + 1).lshr(1)); b.read_first_lane(v) };
    let qoff = b.bind_i32(ev(a.sk) - ev(a.sq));
    let kind = MmaKind::new(MmaShape::M32N32K16, MmaTy::Bf16);
    let zero16 = b.frag_zero(DType::Frag16);
    let zf = b.c_f32(0.0);
    let big = b.c_i32(0x7fff_fff0);
    let live = b.bind_cond(ev(key).lt(ev(a.sk)));
    let o_vo = b.bind_i32(((ev(bat) * ev(a.sk) + ev(key)) * ev(a.hq) + ev(h)) * ROW + (ev(half) << 3));
    let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo), b: Opnd::Val(big) });

    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::I32(0))];
    inits.extend((0..12).map(|_| (DType::Frag16, Opnd::Val(zero16))));
    let any = b.bind_cond(ev(npair).gt(0));
    b.begin_if(Opnd::Val(any));
    stage(&mut b, qt0, 0, Opnd::I32(0));
    let vars = b.begin_loop(&inits);
    let it = vars[0];
    let dkv: Vec<VarId> = vars[1..5].to_vec();
    let dvv: Vec<VarId> = vars[5..9].to_vec();
    let s_vars: Vec<VarId> = vars[9..11].to_vec();
    let dp_vars: Vec<VarId> = vars[11..13].to_vec();
    for v in dkv.iter().chain(dvv.iter()) {
        b.pin_agpr(*v);
    }
    let itx = amdgpu_dsl::varx(it);
    for slot in 0..2i32 {
        let tile = b.bind_i32((itx.clone() << 1) + slot + ev(qt0));
        b.wait_vm(0);
        b.wait_lds(0);
        b.barrier();
        b.weave_region_begin("dkdv_stage");
        let last = b.bind_i32((ev(npair) << 1) - 1 + ev(qt0));
        let nxt = b.bind_i32((ev(tile) + 1).min(ev(last)));
        let gen_n = if slot == 0 { Opnd::Var(it) } else { Opnd::Val(b.bind_i32(itx.clone() + 1)) };
        stage(&mut b, nxt, 1 - slot, gen_n);
        b.weave_region_end();
        let qbase = qs.base as i32 + slot * KT * ROW;
        let dbase = ds_.base as i32 + slot * KT * ROW;
        let (qsl, dsl) = (qs.at(&SlotIdx::new(slot, Opnd::Var(it))), ds_.at(&SlotIdx::new(slot, Opnd::Var(it))));
        // S = Q K^T, dP = dO V^T (2 blocks of 32 queries; C lane = key, values = queries 32 sb + 8 (r / 4) + 4 h + r % 4)
        // S and dP interleaved: 4 independent accumulator chains (c outer)
        b.weave_region_begin("dkdv_sdp_mma");
        for c in 0..8usize {
            qsl.read(&mut b, |b| {
                for sb in 0..2usize {
                    let ad = b.bind_i32(ev(rr[c]) + qbase + sb as i32 * 32 * ROW);
                    let x = b.load_lds_b128(ad);
                    b.var_mfma(s_vars[sb], kind, x, kf[c], c == 0);
                }
            });
            dsl.read(&mut b, |b| {
                for sb in 0..2usize {
                    let ad = b.bind_i32(ev(rr[c]) + dbase + sb as i32 * 32 * ROW);
                    let x = b.load_lds_b128(ad);
                    b.var_mfma(dp_vars[sb], kind, x, vf[c], c == 0);
                }
            });
        }
        b.weave_region_end();
        // P = exp2(csc S - lse2[q]) (masked 0), dS = P (dP - Dot[q]); the per-query LSE / Dot: 4 consecutive queries per load
        b.weave_region_begin("dkdv_softmax_pack");
        let q0 = b.bind_i32(ev(tile) * KT + (ev(half) << 2));
        // P / dS packed to bf16 pairs as soon as a group of 4 values is done (word 8 sb + 2 g + u / 2): fewer live registers
        let mut p: Vec<ValueId> = Vec::with_capacity(16);
        let mut dsv: Vec<ValueId> = Vec::with_capacity(16);
        for sb in 0..2usize {
            for g in 0..4usize {
                let mut pg: Vec<ValueId> = Vec::with_capacity(4);
                let mut dg: Vec<ValueId> = Vec::with_capacity(4);
                let qq = b.bind_i32(ev(q0) + (32 * sb + 8 * g) as i32);
                // the 4 consecutive queries of this value group (4-aligned): one 16-byte load each; queries past SQ read 0 (masked anyway)
                let qa = b.bind_i32(ev(qq) << 2);
                let lv = b.buffer_load_b128(r_l, qa, l_so, 0);
                let dv = b.buffer_load_b128(r_d, qa, l_so, 0);
                for u in 0..4usize {
                    let r = (4 * g + u) as u8;
                    let qv = b.bind_i32(ev(qq) + u as i32);
                    let lse = { let x = b.frag4_lane(lv, u as u8); b.bitcast_f32(x) };
                    let dot = { let x = b.frag4_lane(dv, u as u8); b.bitcast_f32(x) };
                    let s = b.frag16_lane(Opnd::Var(s_vars[sb]), r);
                    let s = b.bitcast_f32(s);
                    let x = b.mul_f32(s, csc);
                    let lse2 = b.mul_f32(lse, l2e);
                    let x = b.sub_f32(x, lse2);
                    let pv = b.exp2_f32(x);
                    let vis = if causal {
                        b.bind_cond((ev(qv).lt(ev(a.sq)).select_i32(1, 0) & ev(key).le(ev(qv) + ev(qoff)).select_i32(1, 0)).eq(1))
                    } else {
                        b.bind_cond(ev(qv).lt(ev(a.sq)))
                    };
                    let pv = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(vis), a: Opnd::Val(pv), b: Opnd::Val(zf) });
                    let dp = b.frag16_lane(Opnd::Var(dp_vars[sb]), r);
                    let dp = b.bitcast_f32(dp);
                    let y = b.sub_f32(dp, dot);
                    pg.push(pv);
                    dg.push(b.mul_f32(pv, y));
                }
                for k in 0..2usize {
                    p.push(b.cvt_pk_bf16_f32(pg[2 * k], pg[2 * k + 1]));
                    dsv.push(b.cvt_pk_bf16_f32(dg[2 * k], dg[2 * k + 1]));
                }
            }
        }
        b.weave_region_end();
        // dV^T += dO^T P, dK^T += Q^T dS: 4 steps of 16 queries x 4 blocks of 32 d (A by tr_b16 from the dO / Q tiles)
        b.weave_region_begin("dkdv_dv_dk_mma");
        for (sl, base, src, acc) in [(&dsl, dbase, &p, &dvv), (&qsl, qbase, &dsv, &dkv)] {
            sl.read(&mut b, |b| {
                for t in 0..4usize {
                    let (sb, tt) = (t / 2, t % 2);
                    let w0 = 8 * sb + 4 * tt;
                    let bf = b.frag4(src[w0], src[w0 + 1], src[w0 + 2], src[w0 + 3]);
                    for e in 0..4usize {
                        let a0 = b.bind_i32(ev(tr[e][0]) + base + (32 * sb as i32 + 16 * tt as i32) * ROW);
                        let a1 = b.bind_i32(ev(tr[e][1]) + base + (32 * sb as i32 + 16 * tt as i32) * ROW);
                        let af = b.load_lds_tr_b16x2(a0, a1);
                        b.var_mfma(acc[e], kind, af, bf, false);
                    }
                }
            });
        }
        b.weave_region_end();
    }
    let it1 = b.bind_i32(itx.clone() + 1);
    b.set_var(it, it1);
    let done = b.bind_cond(ev(it1).ge(ev(npair)));
    b.break_if(done);
    b.end_loop();
    // dK = scale dK^T^T, dV (bf16; lane = key, 4 consecutive d per 8-byte store)
    for (acc, r_out, sc) in [(&dkv, r_dk, Some(a.scale)), (&dvv, r_dv, None)] {
        for e in 0..4usize {
            for u in 0..4u8 {
                let v: Vec<ValueId> = (0..4u8)
                    .map(|k| {
                        let x = b.frag16_lane(Opnd::Var(acc[e]), 4 * u + k);
                        let x = b.bitcast_f32(x);
                        match sc { Some(s) => b.mul_f32(x, s), None => x }
                    })
                    .collect();
                let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
                let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
                let f2 = b.frag2(w0, w1);
                b.buffer_store_b64(r_out, o_vo, c0, ((32 * e as u32) + 8 * u as u32) * 2, f2);
            }
        }
    }
    b.else_();
    // no visible query: dK = dV = 0
    {
        let zw = b.c_i32(0);
        let f2 = b.frag2(zw, zw);
        for r_out in [r_dk, r_dv] {
            for e in 0..4u32 {
                for u in 0..4u32 {
                    b.buffer_store_b64(r_out, o_vo, c0, (32 * e + 8 * u) * 2, f2);
                }
            }
        }
    }
    b.end_if();
    let mut f = b.finish();
    // the Vs128 profile + ldsink / wsink / ldahead32 tokens: dkdv 585 -> 407 us at B1 H16 4096² (same run; Gemm profile + tokens alone 1065 vs 1072,
    // Vs128 alone slower)
    f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    f
}
