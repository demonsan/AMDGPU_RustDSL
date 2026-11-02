//! FMHA forward kernels written on the tile library (`crate::tile`, docs/dsl-tile-abstractions.md). Port of the shipped hand-written kernels
//! (`kernels.rs`: `fmha_t_build` + `emit_fmha_vs128_role`), default recipe only (no `FMHA_OPT` experiments).
//!
//! Status (port milestone M1): L1-L3 of the library carry the kernarg ABI + host contract, the LDS plan, the K/V/KD DMA copies, the two-role ping-pong
//! pipeline (2 barriers per 128 keys) and the peeled tail. The per-phase math (QK / online softmax / PV / epilogue) is still written inline below and moves
//! onto the L4-L6 components (fragments, `mma_tile`, `OnlineSoftmax`, `Epilogue`) as they land.

use crate::tile::*;
use crate::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId, VarId, I32};

/// Values computed once per workgroup that the phase bodies use (the role-independent part of the old `FmhaTCtx`).
struct Blk {
    lane: Lane,
    /// lane >> 5 (the K-half of an MFMA operand) and its byte shifts
    g4: ValueId,
    g8: ValueId,
    kv_len: ValueId,
    /// Q fragments (B operand of S^T = K Q^T, resident) and their raw words (LDS staging)
    b0: ValueId,
    b1: ValueId,
    q0w: Vec<ValueId>,
    q1w: Vec<ValueId>,
    /// Q scale bytes - 9 (S in units of 1/512)
    sq0m: ValueId,
    sq1m: ValueId,
    /// lane-invariant LDS addresses of the K fragment reads (ds_read_b128 x2) and the V^T transposing reads (ds_read_b64_tr_b8 x4), slot 0
    kfx: Vec<ValueId>,
    trx: Vec<ValueId>,
    /// K-scale read address of this lane (slot 0)
    kd_cur: ValueId,
}

/// The per-role pipeline body of `_vs128` (default recipe).
struct Vs128Body<'a> {
    bk: &'a Blk,
    ks: &'a Shared,
    vs: &'a Shared,
    kd: &'a Shared,
    ones: &'a Shared,
    q0s: &'a Shared,
    q1s: &'a Shared,
    kcp: &'a RowsCopy,
    vcp: &'a RowsCopy,
    kdcp: &'a DwordRowsCopy,
    wid: ValueId,
    /// short sequence: the LUT-corrected P conversion (second code copy)
    corr: bool,
    // constants
    zero16: ValueId,
    ones_f16: ValueId,
    bias_init: ValueId,
    a_rs: ValueId,
    idx01: ValueId,
    idx23: ValueId,
    c127: ValueId,
    c123: ValueId,
    one_f: ValueId,
    // state
    s_init: Vec<ValueId>,
    bias2: Option<VarId>,
    o: Vec<VarId>,
    s: Vec<VarId>,
    la: Option<VarId>,
    /// P words of the tile (16) + the new bias word, from the V phase to the M phase
    p: Vec<ValueId>,
}

impl Vs128Body<'_> {
    /// QK of a whole tile in place: S[s] = ones x bias (f16 MFMA) + K(slot)[32 s ..] . Q^T (2 scaled MFMAs per 32 keys).
    fn qk(&mut self, b: &mut FnBuilder, slot: &SlotIdx, bias: ValueId) {
        let bk = self.bk;
        let kref = self.ks.at(slot);
        let kdref = self.kd.at(slot);
        let koff = ev(b.bind_i32(kref.off()));
        let kfb: Vec<ValueId> = (0..4).map(|k| b.bind_i32(ev(bk.kfx[k]) + koff.clone())).collect();
        let kdoff = ev(b.bind_i32(kdref.off()));
        let kd_rd = b.bind_i32(ev(bk.kd_cur) + kdoff);
        for s in 0..4usize {
            b.var_mfma_f16_z(self.s[s], self.ones_f16, Opnd::Val(bias));
            let ko = 4096 * s as i32;
            let a: Vec<ValueId> = (0..4).map(|k| b.bind_i32(ev(kfb[k]) + ko)).collect();
            let (k8_0, k8_1) = kref.read(b, |b| (b.load_lds_b128x2(Opnd::Val(a[0]), Opnd::Val(a[1])), b.load_lds_b128x2(Opnd::Val(a[2]), Opnd::Val(a[3]))));
            let ka = b.bind_i32(ev(kd_rd) + 128 * s as i32);
            let kd = kdref.read(b, |b| b.load_lds_b32(Opnd::Val(ka)));
            let sk0 = b.bind_i32(ev(kd).lshr(ev(bk.g8)));
            let sk1 = b.bind_i32(ev(kd).lshr(ev(bk.g8) + 16));
            b.var_mfma_scale(self.s[s], k8_0, bk.b0, sk0, bk.sq0m);
            b.var_mfma_scale(self.s[s], k8_1, bk.b1, sk1, bk.sq1m);
        }
    }

    /// Softmax of the tile: one max / reference decision over the 64 scores of a lane, then P (16 fp8x4 words) + the new bias word.
    fn softmax(&mut self, b: &mut FnBuilder, t: &TileIt) -> Vec<ValueId> {
        let bk = self.bk;
        let (bias2, la) = (self.bias2.unwrap(), self.la.unwrap());
        let masked = t.masked;
        let kb: Vec<I32> = (0..4).map(|s| b.let_i32(((ox(t.it) << 7) + ev(bk.g4)) + 32 * s as i32)).collect();
        let valid = |b: &mut FnBuilder, s: usize, r: u8| -> ValueId { b.bind_cond((&kb[s] + (8 * (r / 4) + (r % 4)) as i32).lt(ev(bk.kv_len))) };
        let neg_big = b.c_f32(-1.0e30);
        let zero_f = b.c_f32(0.0);
        let mut flat: Vec<ValueId> = Vec::with_capacity(64);
        for s in 0..4usize {
            for r in 0..16u8 {
                let lane = b.frag16_lane(Opnd::Var(self.s[s]), r);
                let v = b.bitcast_f32(lane);
                flat.push(if masked {
                    let ok = valid(b, s, r);
                    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(neg_big) })
                } else {
                    v
                });
            }
        }
        let mut mloc = b.max3_f32(flat[0], flat[1], flat[2]);
        let mut k = 3usize;
        while k + 1 < flat.len() {
            mloc = b.max3_f32(mloc, flat[k], flat[k + 1]);
            k += 2;
        }
        if k < flat.len() {
            mloc = b.max_f32(mloc, flat[k]);
        }
        let mt = b.partner_max_f32(mloc);
        let c_thr = b.c_f32(120.0 / 512.0);
        let grow = b.cmp_lt_f32(c_thr, mt);
        // exact reference shift with a two-word f16 bias (hi + residual): b = 4 (hi + lo); delta = b_q - b_old for the rolling-back lanes, 0 for the others
        let w_old_u = b.frag2_lane(Opnd::Var(bias2), 0);
        let w_old = b.bitcast_i32(w_old_u);
        let c16 = b.c_i32(16);
        let hi_o = b.let_(Op::CvtF16ToF32(Opnd::Val(w_old)));
        let lo_s = b.let_(Op::LshrI32(Opnd::Val(w_old), Opnd::Val(c16)));
        let lo_o = b.let_(Op::CvtF16ToF32(Opnd::Val(lo_s)));
        let c4f = b.c_f32(4.0);
        let c025 = b.c_f32(0.25);
        let c_b131 = b.c_f32(131.0 / 512.0);
        let sum_o = b.add_f32(hi_o, lo_o);
        let b_old = b.mul_f32(sum_o, c4f);
        let tt = b.sub_f32(c_b131, mt);
        let b_new = b.add_f32(b_old, tt);
        let q = b.mul_f32(b_new, c025);
        let h_raw = b.let_(Op::CvtF32ToF16(Opnd::Val(q)));
        let hq = b.let_(Op::CvtF16ToF32(Opnd::Val(h_raw)));
        let r = b.sub_f32(q, hq);
        let l_raw = b.let_(Op::CvtF32ToF16(Opnd::Val(r)));
        let lq = b.let_(Op::CvtF16ToF32(Opnd::Val(l_raw)));
        let sum_q = b.add_f32(hq, lq);
        let b_q = b.mul_f32(sum_q, c4f);
        let d_full = b.sub_f32(b_q, b_old);
        let delta = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(grow), a: Opnd::Val(d_full), b: Opnd::Val(zero_f) });
        let l_sh = b.shl_i32(l_raw, c16);
        let w_q = b.let_(Op::AddI32(Opnd::Val(h_raw), Opnd::Val(l_sh)));
        let w_new = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(grow), a: Opnd::Val(w_q), b: Opnd::Val(w_old) });
        let nb = b.frag2(w_new, w_new);
        b.set_var(bias2, nb);
        b.begin_if(Opnd::Val(grow));
        {
            let dupd = b.frag2_dup(delta);
            for s in 0..4usize {
                b.var_add_pk(self.s[s], dupd);
            }
            let c512f = b.c_f32(512.0);
            let ex = b.mul_f32(delta, c512f);
            let corr = b.exp2_f32(ex);
            let cd = b.frag2_dup(corr);
            for e in 0..4usize {
                b.var_scale_pk(self.o[e], cd);
            }
            b.var_scale_lane(la, 2, corr);
        }
        b.end_if();
        // P = fp8(2^S'): pknorm + scalef32 (+ the LUT correction for short sequences)
        let c4s = b.c_i32(4);
        let cmk = b.c_i32(0x0007_0007);
        let lut_lo = b.c_i32(i32::from_le_bytes([0, 6, 9, 11]));
        let lut_hi = b.c_i32(i32::from_le_bytes([11, 10, 7, 3]));
        let mut words: Vec<ValueId> = Vec::with_capacity(17);
        for s in 0..4usize {
            let mut xs: Vec<ValueId> = Vec::with_capacity(16);
            for r in 0..16u8 {
                let lane = b.frag16_lane(Opnd::Var(self.s[s]), r);
                let v = b.bitcast_f32(lane);
                xs.push(if masked {
                    let ok = valid(b, s, r);
                    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(v), b: Opnd::Val(zero_f) })
                } else {
                    v
                });
            }
            for m in 0..4usize {
                words.push(if self.corr {
                    let mut us: Vec<ValueId> = Vec::with_capacity(2);
                    for h in 0..2usize {
                        let u = b.cvt_pknorm_pair(xs[4 * m + 2 * h], xs[4 * m + 2 * h + 1]);
                        let s1 = b.let_(Op::LshrI32(Opnd::Val(u), Opnd::Val(c4s)));
                        let s2 = b.let_(Op::AndI32(Opnd::Val(s1), Opnd::Val(cmk)));
                        let pm = b.perm_b32(lut_hi, lut_lo, s2);
                        us.push(b.pk_sub_u16(u, pm));
                    }
                    b.cvt_p8x4_from_u16(us[0], us[1], self.one_f)
                } else {
                    b.cvt_pknorm_p8x4(xs[4 * m], xs[4 * m + 1], xs[4 * m + 2], xs[4 * m + 3], self.one_f)
                });
            }
        }
        words.push(nb);
        words
    }

    /// PV of the tile (V(slot), P words) and the row sums (4 in-place smfmacs).
    fn pv(&mut self, b: &mut FnBuilder, slot: &SlotIdx, w: &[ValueId]) {
        let bk = self.bk;
        let la = self.la.unwrap();
        let vref = self.vs.at(slot);
        let voff = ev(b.bind_i32(vref.off()));
        let trb: Vec<ValueId> = (0..4).map(|e| b.bind_i32(ev(bk.trx[e]) + voff.clone())).collect();
        let pa = b.frag8(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        let pb = b.frag8(w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]);
        b.var_smfmac_fp8(la, self.a_rs, pa, self.idx01);
        b.var_smfmac_fp8(la, self.a_rs, pb, self.idx01);
        b.var_smfmac_fp8(la, self.a_rs, pa, self.idx23);
        b.var_smfmac_fp8(la, self.a_rs, pb, self.idx23);
        for (hh, p8) in [pa, pb].into_iter().enumerate() {
            for e in 0..4usize {
                let a: Vec<ValueId> = (0..4).map(|r| b.bind_i32(ev(trb[e]) + ((hh as i32) * 8192 + 2048 * r))).collect();
                let v8 = vref.read(b, |b| b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3]));
                b.var_mfma_scale(self.o[e], v8, p8, self.c127, self.c123);
            }
        }
    }
}

impl PipeBody for Vs128Body<'_> {
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
        match role {
            Role::A => {
                self.kdcp.issue(b, key0, &self.kd.at(slot));
                self.kcp.issue(b, key0, &self.ks.at(slot));
            }
            _ => self.vcp.issue(b, key0, &self.vs.at(slot)),
        }
    }
    fn prologue_stores(&mut self, b: &mut FnBuilder, _role: Role) {
        // a 2 KB block of E4M3 ones and this wave's Q fragments (32 B per lane each), as the shipped kernel stages them
        let bk = self.bk;
        let ones_w = b.c_i32(0x38383838);
        let s0 = SlotIdx::new(0, 0);
        let oa = b.bind_i32((ev(self.wid) * 64 + bk.lane.lane()) * 4 + self.ones.base as i32);
        self.ones.at(&s0).write(b, |b| b.store_lds_b32(Opnd::Val(oa), Opnd::Val(ones_w)));
        for (sh, w) in [(self.q0s, &bk.q0w), (self.q1s, &bk.q1w)] {
            let a0 = b.bind_i32((ev(self.wid) * 2048 + sh.base as i32) + (bk.lane.lane() << 4));
            let a1 = b.bind_i32(ev(a0) + 1024);
            let g0 = b.frag4(w[0], w[1], w[2], w[3]);
            let g1 = b.frag4(w[4], w[5], w[6], w[7]);
            sh.at(&s0).write(b, |b| {
                b.store_lds_b128(Opnd::Val(a0), Opnd::Val(g0));
                b.store_lds_b128(Opnd::Val(a1), Opnd::Val(g1));
            });
        }
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        let bk = self.bk;
        let kref = self.ks.at(&t0.cur);
        let kdref = self.kd.at(&t0.cur);
        self.s_init.clear();
        for s in 0..4usize {
            let biasf = b.mfma_f16_wide(self.ones_f16, self.bias_init, self.zero16);
            let ko = 4096 * s as i32;
            let a: Vec<ValueId> = (0..4).map(|k| b.bind_i32(ev(bk.kfx[k]) + ko)).collect();
            let (k8_0, k8_1) = kref.read(b, |b| (b.load_lds_b128x2(Opnd::Val(a[0]), Opnd::Val(a[1])), b.load_lds_b128x2(Opnd::Val(a[2]), Opnd::Val(a[3]))));
            let ka = b.bind_i32(ev(bk.kd_cur) + 128 * s as i32);
            let kd = kdref.read(b, |b| b.load_lds_b32(Opnd::Val(ka)));
            let sk0 = b.bind_i32(ev(kd).lshr(ev(bk.g8)));
            let sk1 = b.bind_i32(ev(kd).lshr(ev(bk.g8) + 16));
            let s0 = b.mfma_scale_fp8(k8_0, bk.b0, biasf, sk0, bk.sq0m);
            self.s_init.push(b.mfma_scale_fp8(k8_1, bk.b1, s0, sk1, bk.sq1m));
        }
    }
    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let z4 = b.frag_zero(DType::Frag4);
        let mut v = vec![(DType::Frag2, Opnd::Val(self.bias_init))];
        v.extend((0..4).map(|_| (DType::Frag16, Opnd::Val(self.zero16))));
        v.extend(self.s_init.iter().map(|&s| (DType::Frag16, Opnd::Val(s))));
        v.push((DType::Frag4, Opnd::Val(z4)));
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        self.bias2 = Some(vars[0]);
        self.o = vars[1..5].to_vec();
        self.s = vars[5..9].to_vec();
        self.la = Some(vars[9]);
    }
    fn v_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        self.p = self.softmax(b, t);
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        let mut w = std::mem::take(&mut self.p);
        let bias_now = w.pop().expect("bias word");
        self.pv(b, &t.cur, &w);
        if t.staged {
            self.qk(b, &t.next, bias_now);
        }
    }
}

/// `_vs128` on the tile library: 256 query rows per block (8 waves, two groups of 4 in ping-pong), 128-key tiles in a 2-slot LDS ring.
pub fn fmha_fwd_hd128_mxfp8_dsl_vs128() -> Function {
    let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_dsl_vs128", 512);
    fb.compile_profile(CompileProfile::Vs128);
    build_vs128(&mut fb);
    fb.finish()
}

fn build_vs128(b: &mut FnBuilder) {
    // ---- ABI: tensors (+ the host contract they imply) ----
    let mut ka = KArgs::new();
    // GQA (H = HK * gqa, gqa a power of two <= 16) first: the tensor contracts below are stated in terms of H
    let gqa = ka.i32(b, "s_gqa", 0x0B0);
    b.ka_range(0x0B0, 1, 16);
    b.ka_is(0x140, Ext::sym("HK") * Ext::ka(0x0B0));
    let s = ka.dim(b, "s_seq_len", 0x060, 1, i64::MAX);
    let skv = ka.dim(b, "s_kv_seq_len", 0x110, 1, i64::MAX);
    let h = Dim::Ka { off: 0x140, val: ka.i32(b, "s_q_head_num", 0x140) };
    let (hk, bat, d) = (Dim::sym("HK"), Dim::grid(2), Dim::c(128));
    let q = ka.tensor("ptr_q", 0x010, 1, vec![bat.clone(), s.clone(), h.clone(), d.clone()], vec![Stride::Ka("s_Bs", 0x0A0), Stride::Ka("s_Seqs", 0x070), Stride::Ka("s_Hs", 0x090), Stride::Unit]).packed().declare(&mut ka, b);
    let k = ka.tensor("ptr_k", 0x020, 1, vec![bat.clone(), skv.clone(), hk.clone(), d.clone()], vec![Stride::Ka("s_k_Bs", 0x0E0), Stride::Ka("s_k_Seqs", 0x0C0), Stride::Ka("s_k_Hs", 0x0D0), Stride::Unit]).packed().declare(&mut ka, b);
    let v = ka.tensor("ptr_v", 0x030, 1, vec![bat.clone(), skv.clone(), hk.clone(), d.clone()], vec![Stride::Ka("s_v_Bs", 0x170), Stride::Ka("s_v_Seqs", 0x150), Stride::Ka("s_v_Hs", 0x160), Stride::Unit]).same_strides_as(&k).declare(&mut ka, b);
    let o = ka.tensor("ptr_o", 0x000, 2, vec![bat.clone(), s.clone(), h.clone(), d], vec![Stride::Ka("s_o_Bs", 0x1A0), Stride::Ka("s_o_Seqs", 0x180), Stride::Ka("s_o_Hs", 0x190), Stride::Unit]).packed().declare(&mut ka, b);
    // E8M0 descales [B, S, H, 4] / [B, SK, HK, 4]; the row stride is the tensor's seq stride / 32; zeroed slack rows behind them
    let qd_stride = ev(b.bind_i32(q.strides[1].clone().lshr(5)));
    let kd_stride = ev(b.bind_i32(k.strides[1].clone().lshr(5)));
    let qsc = ka.tensor("ptr_q_descale", 0x200, 1, vec![bat.clone(), s.clone(), h.clone(), Dim::c(4)], vec![Stride::Ka("s_descale_q_Bs", 0x230), Stride::Derived(qd_stride), Stride::Ka("s_descale_q_Hs", 0x240), Stride::Unit]).packed().slack(Ext::c(512 * 4) * Ext::ka(0x140)).declare(&mut ka, b);
    let ksc = ka.tensor("ptr_k_descale", 0x210, 1, vec![bat, skv.clone(), hk, Dim::c(4)], vec![Stride::Ka("s_descale_k_Bs", 0x250), Stride::Derived(kd_stride), Stride::Ka("s_descale_k_Hs", 0x260), Stride::Unit]).packed().slack(Ext::c(640 * 4) * Ext::ka(0x140)).declare(&mut ka, b);
    let ptr_vd = ka.ptr(b, "ptr_v_descale", 0x220);
    b.arg_tensor("ptr_v_descale", Ext::c(4096));
    ka.arg(b, "abi_tail", DType::I32, 0x280, 16);
    b.grid_is(1, Ext::ka(0x140));
    b.grid_is(0, Ext::ka(0x060).ceil_div(256));
    let (s_len, kv_len) = (s.val().unwrap(), skv.val().unwrap());
    let kv_len_v = b.bind_i32(kv_len.clone());

    // ---- block / lane ----
    let qtile = b.let_(Op::WorkgroupIdX);
    let hidx = b.let_(Op::WorkgroupIdY);
    let bidx = b.let_(Op::WorkgroupIdZ);
    let lane = Lane::new(b);
    let g4 = b.bind_i32(lane.half() << 2);
    let g8 = b.bind_i32(lane.half() << 3);
    let row0 = b.let_i32((ev(qtile) << 8) + (lane.wave() << 5));
    let s_last = b.let_i32(s_len.clone() - 1);
    let qt = q.tile(vec![ev(bidx), I32::c(0), ev(hidx), I32::c(0)]).clamp(1, s_len.clone(), s_last.clone());
    let row_raw = b.let_i32(row0.clone() + lane.j());
    // Q fragments: two 16-byte loads per 64-byte group (words 0-3 and 8-11), K order permuted like the K fragments
    let qp = qt.ptr(b, &[I32::c(0), row_raw.clone(), I32::c(0), lane.half() << 4]);
    let mut qw: Vec<Vec<ValueId>> = Vec::new();
    for c in 0..2u32 {
        let mut w = Vec::with_capacity(8);
        for half in 0..2u32 {
            let f4 = b.load_b128(qp, 64 * c + 32 * half);
            for i in 0..4u8 {
                let l = b.frag4_lane(f4, i);
                w.push(b.bitcast_i32(l));
            }
        }
        qw.push(w);
    }
    let b0 = b.frag8(qw[0][0], qw[0][1], qw[0][2], qw[0][3], qw[0][4], qw[0][5], qw[0][6], qw[0][7]);
    let b1 = b.frag8(qw[1][0], qw[1][1], qw[1][2], qw[1][3], qw[1][4], qw[1][5], qw[1][6], qw[1][7]);
    let qdp = qsc.tile(vec![ev(bidx), I32::c(0), ev(hidx), I32::c(0)]).clamp(1, s_len.clone(), s_last).ptr(b, &[I32::c(0), row_raw, I32::c(0), I32::c(0)]);
    let qd_raw = b.load_f32(qdp, 0);
    let qd = b.bitcast_i32(qd_raw);
    let sq0m = b.bind_i32(ev(qd).lshr(ev(g8)) - 9);
    let sq1m = b.bind_i32(ev(qd).lshr(ev(g8) + 16) - 9);
    let vd_raw = b.load_f32(ptr_vd, 0);
    // KV head = head >> log2(gqa)
    let gshift = b.bind_i32(ev(gqa).gt(8).select_i32(4, ev(gqa).gt(4).select_i32(3, ev(gqa).gt(2).select_i32(2, ev(gqa).gt(1).select_i32(1, 0)))));
    let kvh = b.bind_i32(ev(hidx).lshr(ev(gshift)));
    b.assume_trusted(kvh, Ext::c(0), Ext::sym("HK") - Ext::c(1), "kv head = head >> log2(gqa) with head < H = HK * gqa (gqa a power of two <= 16): variable shift, no interval rule");
    let nkv = b.bind_i32((ev(kv_len_v) + 127).lshr(7));
    b.assume_trusted(nkv, Ext::c(1), Ext::ka(0x110).ceil_div(128), "nkv = (kv_len + 127) >> 7 with 1 <= kv_len <= SK; the engine floors the lower bound of a shift to 0");
    let kv_last = b.let_i32(ev(kv_len_v) - 1);

    // ---- LDS plan (= fmha_vs_tiles) ----
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(2, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
    let kds = lds.ring("KD", 512, 2, Layout::Linear);
    lds.at(67584);
    let ones = lds.tile("ones", 2048, Layout::Linear);
    let q1s = lds.tile("Q1", 16384, Layout::Linear);
    let q0s = lds.tile("Q0", 16384, Layout::Linear);
    let _epi = lds.private("epi", 4096, 8, &[&q1s, &q0s]);
    lds.commit(b);

    // ---- fragment read addresses (swizzled K rows / transposed V reads), lane invariants ----
    let sj = fmha_swz(b, &lane.j());
    let xk0 = b.let_i32(lane.half() ^ sj);
    let jrow = b.let_i32(lane.j() * 128);
    let kfx: Vec<ValueId> = (0..4).map(|kk| b.bind_i32((jrow.clone() + ((xk0.clone() ^ (2 * kk)) << 4)) + ks.base as i32)).collect();
    let key_part = b.let_i32((((lane.lane() & 15).lshr(3) << 3) + (lane.half() << 2)) + (lane.lane().lshr(1) & 3));
    let m_gl = b.let_i32(lane.lane().lshr(4) & 1);
    let m_h = b.let_i32(lane.lane() & 1);
    let sv = fmha_swz(b, &key_part);
    let trow = b.let_i32(((key_part.clone() * 128) + (m_h << 3)) + vs.base as i32);
    let z = b.let_i32(m_gl ^ sv);
    let trx: Vec<ValueId> = (0..4).map(|e| b.bind_i32(trow.clone() + ((z.clone() ^ (2 * e)) << 4))).collect();
    let kd_cur = b.bind_i32((lane.j() << 2) + kds.base as i32);

    // ---- copies: K/V tiles (4 DMA per wave) and the K scales (1 DMA per wave), rows clamped to the last key ----
    let roles = Roles::split(b, &lane, 4);
    let kt = k.tile(vec![ev(bidx), I32::c(0), ev(kvh), I32::c(0)]).clamp(1, ev(kv_len_v), kv_last.clone());
    let vt = v.tile(vec![ev(bidx), I32::c(0), ev(kvh), I32::c(0)]).clamp(1, ev(kv_len_v), kv_last.clone());
    let kdt = ksc.tile(vec![ev(bidx), I32::c(0), ev(kvh), I32::c(0)]).clamp(1, ev(kv_len_v), kv_last);
    let (ksrc, vsrc, kdsrc) = (kt.row_source(b, 1), vt.row_source(b, 1), kdt.row_source(b, 1));
    let kcp = RowsCopy::new(b, ksrc, &ks, roles.wave_in_group, &lane, 4);
    let vcp = RowsCopy::new(b, vsrc, &vs, roles.wave_in_group, &lane, 4);
    let kdcp = DwordRowsCopy::new(b, kdsrc, &kds, roles.wave_in_group, &lane);

    // ---- constants of the phase bodies ----
    let zero16 = b.frag_zero(DType::Frag16);
    let ones_h = b.c_i32(0x3C003C00);
    let ones_f16 = b.frag2(ones_h, ones_h);
    let w_init = b.c_i32(0x48004800u32 as i32);
    let bias_init = b.frag2(w_init, w_init);
    let ones_w = b.c_i32(0x38383838);
    let a_rs = {
        // per-lane A operand of the row-sum smfmacs (ones in the lanes of this query, see vendor-spec 4.5.3)
        let b0k = b.bind_i32(lane.lane() & 3);
        let b1k = b.bind_i32(lane.lane() & 4);
        let c0 = b.c_i32(0);
        let lt2 = b.bind_cond(ev(b0k).lt(2));
        let b1z = b.bind_cond(ev(b1k).lt(1));
        let v0 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(lt2), a: Opnd::Val(c0), b: Opnd::Val(ones_w) });
        let a01 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(b1z), a: Opnd::Val(c0), b: Opnd::Val(v0) });
        let a23 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(b1z), a: Opnd::Val(v0), b: Opnd::Val(c0) });
        b.frag4(a23, a23, a01, a01)
    };
    let idx01 = b.c_i32(0x4444_4444_u32 as i32);
    let idx23 = b.c_i32(0xeeee_eeee_u32 as i32);
    let c127 = b.c_i32(127);
    let c123 = b.c_i32(123);
    let one_f = b.c_f32(1.0);
    let bk = Blk { lane, g4, g8, kv_len: kv_len_v, b0, b1, q0w: qw[0].clone(), q1w: qw[1].clone(), sq0m, sq1m, kfx, trx, kd_cur };

    // ---- the two code copies (LUT-corrected P for sequences shorter than 9 tiles), the two roles, the pipeline, the epilogue ----
    let short = b.bind_cond(ev(nkv).lt(9));
    let pipe = Pipeline::new(Cadence::TwoPhaseOffset, nkv, kv_len_v, 128, Ext::ka(0x110).ceil_div(128));
    let o_row = o.tile(vec![ev(bidx), I32::c(0), ev(hidx), I32::c(0)]).clamp(1, s_len.clone(), s_len.clone() - 1);
    versioned(b, short, |b, corr| {
        roles.each(b, |b, role| {
            let wid = roles.wave_id(b, role);
            let mut body = Vs128Body {
                bk: &bk, ks: &ks, vs: &vs, kd: &kds, ones: &ones, q0s: &q0s, q1s: &q1s, kcp: &kcp, vcp: &vcp, kdcp: &kdcp, wid, corr,
                zero16, ones_f16, bias_init, a_rs, idx01, idx23, c127, c123, one_f,
                s_init: Vec::new(), bias2: None, o: Vec::new(), s: Vec::new(), la: None, p: Vec::new(),
            };
            pipe.run(b, role, &mut body);
            // epilogue: O * (V descale / l) -> bf16, one 8-byte store per 4 values (rows past S recompute and store the last row)
            let d2 = b.frag4_lane(Opnd::Var(body.la.unwrap()), 2);
            let lf = b.bitcast_f32(d2);
            let c0625 = b.c_f32(0.0625);
            let l_tot = b.mul_f32(lf, c0625);
            let inv_l = b.rcp_f32(l_tot);
            let fin = b.mul_f32(inv_l, vd_raw);
            let o_ptr0 = o_row.ptr(b, &[I32::c(0), row0.clone() + bk.lane.j(), I32::c(0), bk.lane.half() << 2]);
            let live = b.bind_cond(row0.lt(s_len.clone()));
            b.begin_if(Opnd::Val(live));
            for e in 0..4usize {
                for qq in 0..4usize {
                    let mut v4: Vec<ValueId> = Vec::with_capacity(4);
                    for kk in 0..4usize {
                        let l = b.frag16_lane(Opnd::Var(body.o[e]), (4 * qq + kk) as u8);
                        let lv = b.bitcast_f32(l);
                        v4.push(b.mul_f32(lv, fin));
                    }
                    let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
                    let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
                    let f2 = b.frag2(w0, w1);
                    b.store_f32x2(o_ptr0, (64 * e + 16 * qq) as u32, f2);
                }
            }
            b.end_if();
        });
    });
}
