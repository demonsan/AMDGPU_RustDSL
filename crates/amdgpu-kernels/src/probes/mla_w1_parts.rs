//! Shared pieces of the gfx1250 MLA window kernels (`mla_g1250_qh16_w1p`, the role kernels): everything except the LDS plan, the roles,
//! the loop schedule, the barriers and the protocol markers, which stay with the caller. Window = `wp` pages; ring slot of window t =
//! t & (depth - 1), generation t; the exchange rings (RED, PBUF, ALPHA) use slot t & 1, generation t.
//!
//! Lane roles: `w` = the token quarter (0..3: tokens 16 w .. of each page; V dims 128 w ..), lane (m, g) = head m, lane half g.

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, MemTy, Op, Opnd, TdmCpol, TensorTile, ValueId, VarId, WmmaKind, I32};
use amdgpu_stage::attention::mla::{PageLayout, DV, PAGE_BYTES, PAGE_TOKENS, QUARTER, ROPE_OFF};
use amdgpu_stage::gemm::tdm::tdm_group0;
use amdgpu_tile::shared::{Shared, SlotIdx};

const MAX_PAGES: i32 = 1024;

fn f32_of(b: &mut FnBuilder, v: ValueId) -> ValueId {
    b.let_(Op::BitcastF32(Opnd::Val(v)))
}

pub struct W1Parts {
    pub wp: i32,
    pub depth: i32,
    pub layout: PageLayout,
    pub pr: ValueId,
    pub plse: ValueId,
    pub pkv: ValueId,
    pub pltd: ValueId,
    pub bidx: ValueId,
    pub w: ValueId,
    pub lane: ValueId,
    pub m: ValueId,
    pub g: ValueId,
    pub kv0: ValueId,
    pub kv_end: ValueId,
    pub npages: ValueId,
    pub ctx: ValueId,
    pub ksf: ValueId,
    pub cscale: ValueId,
    pub neg: ValueId,
    pub q: Vec<ValueId>,
    pub g1: ValueId,
    pub kofs: ValueId,
    pub rofs: ValueId,
    pub vofs: ValueId,
    pub kdelta: Vec<i32>,
    pub vdelta: Vec<i32>,
}

impl W1Parts {
    /// The MLA kernargs and their extents (call before the LDS plan's commit).
    pub fn declare_args(b: &mut FnBuilder) {
        let names = ["R", "LSE", "Q", "KV", "LTP", "LTD", "LTL", "QTP", "QS", "KS", "VSC"];
        for (i, n) in names.iter().enumerate() {
            b.arg(n, DType::Ptr, 8 * i as u32, 8);
        }
        for (i, n) in ["out16", "scalar", "qlens", "passes", "stride_page", "log2_page", "stride_q", "flags"].iter().enumerate() {
            b.arg(n, DType::I32, 88 + 4 * i as u32, 4);
        }
        b.arg_tensor("R", Ext::grid_y() * Ext::c(16 * 512 * 2));
        b.arg_tensor("LSE", Ext::grid_y() * Ext::c(16 * 4));
        b.arg_tensor("Q", Ext::grid_y() * Ext::c(16 * 768));
        b.arg_tensor("LTP", (Ext::grid_y() + Ext::c(1)) * Ext::c(4));
        b.arg_tensor("LTL", Ext::grid_y() * Ext::c(4));
        b.arg_tensor("QS", Ext::c(4));
        b.arg_tensor("KS", Ext::c(4));
        b.sym_range("P", 1, 1 << 24);
        b.sym_range("NPG", 1, 1 << 26);
        b.arg_tensor("KV", Ext::sym("P") * Ext::c(PAGE_BYTES as i64));
        b.arg_tensor("LTD", Ext::sym("NPG") * Ext::c(4));
    }

    /// The per-sequence scalars, the Q fragments, the TDM descriptor and the lane offsets. `w`: the token quarter of the calling wave
    /// (wave-uniform, 0..3); `lane`: tid & 31.
    pub fn new(b: &mut FnBuilder, layout: PageLayout, wp: i32, depth: i32, w: ValueId, lane: ValueId) -> W1Parts {
        assert!(depth >= 4 && (depth & (depth - 1)) == 0 && (wp == 1 || wp == 2));
        let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 8 * i });
        let (pr, plse, pq, pkv, pltp, pltd, pltl) = (kp(b, 0), kp(b, 1), kp(b, 2), kp(b, 3), kp(b, 4), kp(b, 5), kp(b, 6));
        let (pqs, pks) = (kp(b, 8), kp(b, 9));
        let bidx = b.let_(Op::WorkgroupIdY);
        let m = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(15)));
        let g = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));
        let ld = |b: &mut FnBuilder, p: ValueId, off: ValueId| {
            let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
            b.let_(Op::LoadScalar { base: Opnd::Val(q), byte_off: 0 })
        };
        let zero = b.c_i32(0);
        let qs = ld(b, pqs, zero);
        let ks = ld(b, pks, zero);
        let bo = b.let_(Op::ShlI32(Opnd::Val(bidx), Opnd::I32(2)));
        let kv0 = ld(b, pltp, bo);
        let bo4 = b.add_i32(bo, Opnd::I32(4));
        let kv1 = ld(b, pltp, bo4);
        let last = ld(b, pltl, bo);
        let npages = b.bind_i32(ev(kv1) - ev(kv0));
        b.assume_trusted(npages, Ext::c(1), Ext::c(MAX_PAGES as i64), "host contract: every sequence has 1 .. 1024 pages (ctx <= 65536)");
        b.assume_trusted(kv0, Ext::c(0), Ext::sym("NPG") - Ext::c(1), "host contract: LTP is monotone and LTP[B] <= NPG, the page table's entry count");
        let kv_end = b.bind_i32(ev(kv0) + ev(npages));
        b.assume_trusted(kv_end, Ext::c(1), Ext::sym("NPG"), "host contract: LTP[b + 1] <= NPG (a sequence's entries lie inside the page table)");
        b.assume_trusted(last, Ext::c(1), Ext::c(PAGE_TOKENS as i64), "host contract: the last page holds 1 .. 64 tokens");
        let ctx = b.bind_i32(((ev(npages) - 1) << 6) + ev(last));
        let qsf = f32_of(b, qs);
        let ksf = f32_of(b, ks);
        let sc = b.let_(Op::KernargI32 { offset: 92 });
        let scf = f32_of(b, sc);
        let c1 = b.mul_f32(qsf, ksf);
        let c2 = b.mul_f32(c1, scf);
        let log2e = b.c_f32(std::f32::consts::LOG2_E);
        let cscale = b.mul_f32(c2, log2e);
        let neg = b.c_f32(-1.0e30);
        let qrow = b.bind_i32((ev(bidx) * 16 + ev(m)) * 768 + ev(g) * 32);
        let pqr = b.let_(Op::PtrAdd { base: Opnd::Val(pq), byte_off: Opnd::Val(qrow) });
        let q: Vec<ValueId> = (0..9u32)
            .map(|c| {
                let parts: Vec<Opnd> = (0..2u32).map(|j| Opnd::Val(b.load_global(pqr, 64 * c + 16 * j, MemTy::B128))).collect();
                b.frag_cat(&parts)
            })
            .collect();
        let tile = TensorTile { data_size: 8, tensor_dim: [Opnd::I32(1152), Opnd::I32(1)], stride0: Opnd::I32(1152), tile: [1152, 1], pad: layout.pad, wg_mask: 0 };
        let (_, g1) = b.tensor_desc_2d(&tile, Opnd::Val(pkv), Opnd::I32(0), Opnd::I32(0));
        let kofs = layout.at_expr(b, (ev(w) * 16 + ev(m)) * DV + ev(g) * 32);
        let rofs = layout.at_expr(b, (ev(w) * 16 + ev(m)) * 64 + ev(g) * 32 + ROPE_OFF);
        let vofs = layout.at_expr(b, (((ev(lane) >> 4) << 3) + (ev(lane) & 3) + (((ev(lane) >> 3) & 1) << 2)) * DV + (((ev(lane) >> 2) & 1) << 3) + ev(w) * 128);
        let kbases: Vec<i32> = (0..128).map(|t| (t / 32 * 16 + (t & 15)) * DV + ((t & 31) >> 4) * 32).collect();
        let kdelta: Vec<i32> = (0..8).map(|c| layout.delta(&kbases, 64 * c)).collect();
        let vbases: Vec<i32> = (0..128).map(|t| { let f = t & 31; (8 * (f >> 4) + (f & 3) + 4 * ((f >> 3) & 1)) * DV + 8 * ((f >> 2) & 1) + (t / 32) * 128 }).collect();
        let vdelta: Vec<i32> = (0..32).map(|i| layout.delta(&vbases, 16 * (i / 8) * DV + 16 * (i % 8))).collect();
        W1Parts { wp, depth, layout, pr, plse, pkv, pltd, bidx, w, lane, m, g, kv0, kv_end, npages, ctx, ksf, cscale, neg, q, g1, kofs, rofs, vofs, kdelta, vdelta }
    }

    /// Windows of the sequence.
    pub fn nwin(&self, b: &mut FnBuilder) -> ValueId {
        b.bind_i32((ev(self.npages) + (self.wp - 1)) >> (self.wp.trailing_zeros() as i32))
    }

    fn kv_slot(&self, b: &mut FnBuilder, t: ValueId) -> SlotIdx {
        let si = b.bind_i32(ev(t) & (self.depth - 1));
        SlotIdx::new(Opnd::Val(si), Opnd::Val(t))
    }

    fn x_slot(&self, b: &mut FnBuilder, t: ValueId) -> SlotIdx {
        let si = b.bind_i32(ev(t) & 1);
        SlotIdx::new(Opnd::Val(si), Opnd::Val(t))
    }

    /// The page ids of window t (scalar loads from the page table, clamped to the sequence's last entry). Issue them a window before
    /// their `stage`.
    pub fn load_pids(&self, b: &mut FnBuilder, t: ValueId) -> Vec<ValueId> {
        (0..self.wp)
            .map(|pg| {
                let er = b.bind_i32(ev(self.kv0) + ev(t) * self.wp + pg);
                let el = b.bind_i32(ev(self.kv_end) - 1);
                let ec = b.let_(Op::MinI32(Opnd::Val(er), Opnd::Val(el)));
                let eo = b.bind_i32(ev(ec) * 4);
                let pe = b.let_(Op::PtrAdd { base: Opnd::Val(self.pltd), byte_off: Opnd::Val(eo) });
                b.let_(Op::LoadScalar { base: Opnd::Val(pe), byte_off: 0 })
            })
            .collect()
    }

    /// This wave's quarter of every page of window t (one tensor op per page) into the KV ring.
    pub fn stage(&self, b: &mut FnBuilder, kv: &Shared, t: ValueId, pids: &[I32]) {
        let sl = self.kv_slot(b, t);
        let r = kv.at(&sl);
        for pg in 0..self.wp {
            let pid = b.bind_i32(pids[pg as usize].clone() + 0);
            b.assume_trusted(pid, Ext::c(0), Ext::sym("P") - Ext::c(1), "host contract: every page-table entry < P, the KV pool's page count (KV holds P x 36864 bytes)");
            let off = b.bind_i32(ev(pid) * PAGE_BYTES + ev(self.w) * QUARTER);
            let lds_a = b.bind_i32(r.base() + pg * self.layout.slot_bytes + ev(self.w) * self.layout.quarter_lds);
            let g0 = tdm_group0(b, self.pkv, off, Opnd::Val(lds_a));
            let g1 = self.g1;
            r.write(b, |b| b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default()));
        }
    }

    /// The K fragments of window t (per page: 9 chunks, each a b128 pair: lane (m, g) = token 16 w + m, 32 bytes of the chunk).
    pub fn k_frags(&self, b: &mut FnBuilder, kv: &Shared, t: ValueId) -> Vec<Vec<ValueId>> {
        let sl = self.kv_slot(b, t);
        let r = kv.at(&sl);
        let sb = b.bind_i32(r.base());
        (0..self.wp)
            .map(|pg| {
                let pb = b.bind_i32(ev(sb) + pg * self.layout.slot_bytes);
                let nope_row = b.bind_i32(ev(pb) + ev(self.kofs));
                let rope_row = b.bind_i32(ev(pb) + ev(self.rofs));
                r.read(b, |b| {
                    (0..9)
                        .map(|c| {
                            let lo = if c < 8 { b.bind_i32(ev(nope_row) + self.kdelta[c as usize]) } else { b.bind_i32(ev(rope_row)) };
                            let hi = b.bind_i32(ev(lo) + 16);
                            b.load_lds_b128x2(lo, hi)
                        })
                        .collect()
                })
            })
            .collect()
    }

    /// The V fragments of window t for this wave's 128 dims (per page: 8 tiles of ds_load_tr8_b64 x 4).
    pub fn v_frags(&self, b: &mut FnBuilder, kv: &Shared, t: ValueId) -> Vec<Vec<ValueId>> {
        let sl = self.kv_slot(b, t);
        let r = kv.at(&sl);
        let kb = b.bind_i32(r.base());
        (0..self.wp)
            .map(|pg| {
                let vrow = b.bind_i32(ev(kb) + pg * self.layout.slot_bytes + ev(self.vofs));
                r.read(b, |b| {
                    (0..8)
                        .map(|nt| {
                            let a: Vec<ValueId> = (0..4).map(|jj| b.bind_i32(ev(vrow) + self.vdelta[8 * jj as usize + nt as usize])).collect();
                            b.load_lds_tr_b8x4(a[0], a[1], a[2], a[3])
                        })
                        .collect()
                })
            })
            .collect()
    }

    /// S^T of the window from its K fragments (one Frag8 per page: lane (m, g) = head m, tokens 16 w + 8 g + r).
    pub fn s_wmma(&self, b: &mut FnBuilder, kfr: &[Vec<ValueId>]) -> Vec<ValueId> {
        let mut acc: Vec<Option<Opnd>> = vec![None; self.wp as usize];
        for c in 0..9usize {
            for pg in 0..self.wp as usize {
                let d = b.wmma(WmmaKind::Fp8_16x16x64, kfr[pg][c], self.q[c], acc[pg]);
                acc[pg] = Some(Opnd::Val(d));
            }
        }
        acc.into_iter().map(|a| match a { Some(Opnd::Val(v)) => v, _ => unreachable!() }).collect()
    }

    /// The 8 wp raw S floats of a lane (element 8 pg + r = token 64 pg + 16 w + 8 g + r of the window).
    pub fn s_floats(&self, b: &mut FnBuilder, s: &[ValueId]) -> Vec<ValueId> {
        let mut x = Vec::new();
        for pg in 0..self.wp as usize {
            for r in 0..8u8 {
                let si = b.frag_slice(Opnd::Val(s[pg]), r, 1);
                x.push(f32_of(b, si));
            }
        }
        x
    }

    /// The raw S of window t with the tokens at or beyond ctx at -1e30; the mask only when the window reaches past ctx (a uniform
    /// branch over VALU; the values are returned as variables).
    pub fn raw_masked(&self, b: &mut FnBuilder, t: ValueId, x: &[ValueId]) -> Vec<Opnd> {
        let init: Vec<(DType, Opnd)> = x.iter().map(|&v| (DType::F32, Opnd::Val(v))).collect();
        let xm = b.declare_vars(&init);
        let wend = b.bind_i32((ev(t) + 1) * (64 * self.wp));
        let part = b.cmp(CmpPred::Gt, CmpTy::I32, wend, self.ctx);
        b.begin_if(part);
        for pg in 0..self.wp {
            let lim = b.bind_i32(ev(self.ctx) - (ev(t) * (64 * self.wp) + (64 * pg) + ev(self.w) * 16 + ev(self.g) * 8));
            for r in 0..8u8 {
                let k = (8 * pg) as usize + r as usize;
                let ok = b.cmp(CmpPred::Gt, CmpTy::I32, lim, Opnd::I32(r as i32));
                let v = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(ok), a: Opnd::Val(x[k]), b: Opnd::Val(self.neg) });
                b.set_var(xm[k], Opnd::Val(v));
            }
        }
        b.end_if();
        xm.iter().map(|&v| Opnd::Var(v)).collect()
    }

    /// This lane's head-m maximum of `x` into RED slot t (layout [head m][8 partials]: word 8 m + part, part = 2 w + g).
    pub fn red_write(&self, b: &mut FnBuilder, red: &Shared, t: ValueId, x: &[Opnd]) {
        let mut mx = b.max_f32(x[0], x[1]);
        for k in 2..x.len() {
            mx = b.max_f32(mx, x[k]);
        }
        let sl = self.x_slot(b, t);
        let r = red.at(&sl);
        let a = b.bind_i32(r.base() + ev(self.m) * 32 + (ev(self.w) * 2 + ev(self.g)) * 4);
        r.write(b, |b| b.store_lds(a, mx, MemTy::B32, false));
    }

    /// The raw maximum of head m over the window (8 partials: one b128 pair).
    pub fn red_read_max(&self, b: &mut FnBuilder, red: &Shared, t: ValueId) -> ValueId {
        let sl = self.x_slot(b, t);
        let r = red.at(&sl);
        let lo = b.bind_i32(r.base() + ev(self.m) * 32);
        let hi = b.bind_i32(ev(lo) + 16);
        r.read(b, |b| {
            let v8 = b.load_lds_b128x2(lo, hi);
            let f: Vec<ValueId> = (0..8u8).map(|e| { let x = b.frag_slice(Opnd::Val(v8), e, 1); f32_of(b, x) }).collect();
            let a1 = b.max3_f32(f[0], f[1], f[2]);
            let a2 = b.max3_f32(a1, f[3], f[4]);
            let a3 = b.max3_f32(a2, f[5], f[6]);
            b.max_f32(a3, f[7])
        })
    }

    /// P of the window (no LDS): m_new = max(m_old, pm_raw * scale); exponents by v_pk_fma_f32 (x * scale - m_new); packed sums.
    /// Returns (P, alpha = exp2(m_old - m_new), m_new, l_new).
    pub fn p_lean(&self, b: &mut FnBuilder, x: &[Opnd], m_old: Opnd, l_old: Opnd, pm_raw: ValueId) -> (Vec<ValueId>, ValueId, ValueId, ValueId) {
        let pms = b.mul_f32(pm_raw, self.cscale);
        let m_new = b.max_f32(m_old, pms);
        let dm = b.sub_f32(m_old, m_new);
        let alpha = b.exp2_f32(dm);
        let sc2 = b.frag2_dup(self.cscale);
        let nm = b.neg_f32(m_new);
        let nm2 = b.frag2_dup(nm);
        let dp: Vec<ValueId> = (0..x.len() / 2)
            .map(|k| {
                let pair = b.frag_cat(&[x[2 * k], x[2 * k + 1]]);
                b.pk_fma_f32(pair, sc2, nm2)
            })
            .collect();
        let mut p = Vec::with_capacity(x.len());
        for k in 0..dp.len() {
            for h in 0..2u8 {
                let dv = b.frag_slice(Opnd::Val(dp[k]), h, 1);
                let df = f32_of(b, dv);
                p.push(b.exp2_f32(df));
            }
        }
        let mut pairs: Vec<ValueId> = (0..p.len() / 2).map(|k| b.frag_cat(&[Opnd::Val(p[2 * k]), Opnd::Val(p[2 * k + 1])])).collect();
        while pairs.len() > 1 {
            pairs = (0..pairs.len() / 2).map(|k| b.pk_add_f32(pairs[2 * k], pairs[2 * k + 1])).collect();
        }
        let s0 = b.frag_slice(Opnd::Val(pairs[0]), 0, 1);
        let s1 = b.frag_slice(Opnd::Val(pairs[0]), 1, 1);
        let s0f = f32_of(b, s0);
        let s1f = f32_of(b, s1);
        let ps = b.add_f32(s0f, s1f);
        let la = b.mul_f32(l_old, alpha);
        let lsum = b.add_f32(la, ps);
        (p, alpha, m_new, lsum)
    }

    /// P (fp8) of this lane into PBUF slot t: per page 1 KB, lane l's 32 bytes at 32 l, wave quarter w's 8 bytes at + 8 w.
    pub fn p_store(&self, b: &mut FnBuilder, pbuf: &Shared, t: ValueId, p: &[ValueId]) {
        let sl = self.x_slot(b, t);
        let r = pbuf.at(&sl);
        let pw = b.bind_i32(r.base() + ev(self.lane) * 32 + ev(self.w) * 8);
        r.write(b, |b| {
            for pg in 0..self.wp as usize {
                let p0 = b.cvt_pk_fp8x4_f32(p[8 * pg], p[8 * pg + 1], p[8 * pg + 2], p[8 * pg + 3]);
                let p1 = b.cvt_pk_fp8x4_f32(p[8 * pg + 4], p[8 * pg + 5], p[8 * pg + 6], p[8 * pg + 7]);
                let a0 = b.bind_i32(ev(pw) + 1024 * pg as i32);
                b.store_lds(a0, p0, MemTy::B32, false);
                let a1 = b.bind_i32(ev(pw) + (1024 * pg as i32 + 4));
                b.store_lds(a1, p1, MemTy::B32, false);
            }
        });
    }

    /// The A fragments of P for the PV WMMAs (per page: lane (m, g)'s 32 bytes).
    pub fn p_load(&self, b: &mut FnBuilder, pbuf: &Shared, t: ValueId) -> Vec<ValueId> {
        let sl = self.x_slot(b, t);
        let r = pbuf.at(&sl);
        let pbase = b.bind_i32(r.base() + ev(self.lane) * 32);
        r.read(b, |b| {
            (0..self.wp)
                .map(|pg| {
                    let lo = b.bind_i32(ev(pbase) + 1024 * pg);
                    let hi = b.bind_i32(ev(lo) + 16);
                    b.load_lds_b128x2(lo, hi)
                })
                .collect()
        })
    }

    /// alpha of head m into ALPHA slot t (word m; both lane halves write the same value).
    pub fn alpha_store(&self, b: &mut FnBuilder, alpha_ring: &Shared, t: ValueId, alpha: ValueId) {
        let sl = self.x_slot(b, t);
        let r = alpha_ring.at(&sl);
        let a = b.bind_i32(r.base() + ev(self.m) * 4);
        r.write(b, |b| b.store_lds(a, alpha, MemTy::B32, false));
    }

    /// The alphas of rows 8 g + r (r = 0..7) from ALPHA slot t, and the "some head's max grew" lane condition (any alpha < 1).
    pub fn alpha_rows_load(&self, b: &mut FnBuilder, alpha_ring: &Shared, t: ValueId) -> (Vec<ValueId>, ValueId) {
        let sl = self.x_slot(b, t);
        let r = alpha_ring.at(&sl);
        let base = b.bind_i32(r.base() + 0);
        let a16: Vec<ValueId> = r.read(b, |b| {
            let lo = b.bind_i32(ev(base) + 0);
            let v0 = b.load_lds_b128(lo);
            let a1 = b.bind_i32(ev(base) + 16);
            let v1 = b.load_lds_b128(a1);
            let a2 = b.bind_i32(ev(base) + 32);
            let v2 = b.load_lds_b128(a2);
            let a3 = b.bind_i32(ev(base) + 48);
            let v3 = b.load_lds_b128(a3);
            let vs = [v0, v1, v2, v3];
            (0..16u8).map(|h| { let x = b.frag_slice(Opnd::Val(vs[(h / 4) as usize]), h % 4, 1); f32_of(b, x) }).collect()
        });
        let rows = self.rows_of(b, &a16);
        // some row's alpha < 1 (the two lane halves cover the 16 heads; the branch takes any lane): max(-alpha) > -1
        let mut nm = b.neg_f32(rows[0]);
        for r in 1..8 {
            let x = b.neg_f32(rows[r]);
            nm = b.max_f32(nm, x);
        }
        let grew = b.cmp(CmpPred::Gt, CmpTy::F32, nm, Opnd::F32(-1.0));
        (rows, grew)
    }

    /// The alphas of rows 8 g + r from the lanes' own alpha (lane m holds head m): v_readlane, select-free choice of the lane half.
    pub fn alpha_rows_from_lanes(&self, b: &mut FnBuilder, alpha: Opnd) -> (Vec<ValueId>, ValueId) {
        let a16: Vec<ValueId> = (0..16u8)
            .map(|h| {
                let s = b.read_lane(alpha, h);
                f32_of(b, s)
            })
            .collect();
        let rows = self.rows_of(b, &a16);
        let grew = b.cmp(CmpPred::Lt, CmpTy::F32, alpha, Opnd::F32(1.0));
        (rows, grew)
    }

    fn rows_of(&self, b: &mut FnBuilder, a16: &[ValueId]) -> Vec<ValueId> {
        let gbits = b.bind_i32(ev(self.g) * 0x3F80_0000);
        let gf = f32_of(b, gbits);
        let one = b.c_f32(1.0);
        let hf = b.sub_f32(one, gf);
        (0..8usize)
            .map(|r| {
                let u0 = b.mul_f32(a16[r], hf);
                let u1 = b.mul_f32(a16[8 + r], gf);
                b.add_f32(u0, u1)
            })
            .collect()
    }

    /// O *= alpha (rows 8 g + r), in place by v_pk_mul_f32, only when some lane's `grew` holds (a uniform branch).
    pub fn rescale_lazy(&self, b: &mut FnBuilder, ov: &[VarId], rows: &[ValueId], grew: ValueId) {
        let anyg = b.let_(Op::AnyLane(Opnd::Val(grew)));
        b.begin_if(anyg);
        let ap: Vec<ValueId> = (0..4).map(|k| b.frag_cat(&[Opnd::Val(rows[2 * k]), Opnd::Val(rows[2 * k + 1])])).collect();
        for &o in ov {
            for k in 0..4u8 {
                b.var_pk_mul(o, 2 * k, ap[k as usize]);
            }
        }
        b.end_if();
    }

    /// O += P V for the window (this wave's 128 dims: ov[nt], nt = 0..7).
    pub fn pv_wmma(&self, b: &mut FnBuilder, ov: &[VarId], pa: &[ValueId], vfr: &[Vec<ValueId>]) {
        for nt in 0..8usize {
            for pg in 0..self.wp as usize {
                let d = b.wmma(WmmaKind::Fp8_16x16x64, pa[pg], vfr[pg][nt], Some(Opnd::Var(ov[nt])));
                b.set_var(ov[nt], Opnd::Val(d));
            }
        }
    }

    /// The row-sum partial of this lane (head m, its tokens) into LSUM: word 8 m + (2 w + g); and (wave quarter 0, lane half 0) the
    /// running max of head m at word 128 + m.
    pub fn lsum_write(&self, b: &mut FnBuilder, lsum: &Shared, m_run: Opnd, l_partial: Opnd) {
        let t00 = SlotIdx::new(Opnd::I32(0), Opnd::I32(0));
        let r = lsum.at(&t00);
        let a = b.bind_i32(r.base() + ev(self.m) * 32 + (ev(self.w) * 2 + ev(self.g)) * 4);
        let am = b.bind_i32(r.base() + 512 + ev(self.m) * 4);
        r.write(b, |b| {
            b.store_lds(a, l_partial, MemTy::B32, false);
            b.store_lds(am, m_run, MemTy::B32, false);
        });
    }

    /// The epilogue of the waves holding O (this wave's 128 dims of the 16 heads): O / L * ks to bf16 through the OUT tile, 16-byte
    /// stores (thread `tid` < 128 stores 128 contiguous bytes), the LSE (wave quarter 0). LSUM must be published (barrier) before.
    pub fn epilogue(&self, b: &mut FnBuilder, ov: &[VarId], lsum: &Shared, out: &Shared, tid: ValueId) {
        let t00 = SlotIdx::new(Opnd::I32(0), Opnd::I32(0));
        let lr = lsum.at(&t00);
        let lb = b.bind_i32(lr.base() + 0);
        let sum_head = |b: &mut FnBuilder, head: I32| -> ValueId {
            let lo = b.bind_i32(ev(lb) + head * 32);
            let hi = b.bind_i32(ev(lo) + 16);
            lr.read(b, |b| {
                let v8 = b.load_lds_b128x2(lo, hi);
                let f: Vec<ValueId> = (0..8u8).map(|e| { let x = b.frag_slice(Opnd::Val(v8), e, 1); f32_of(b, x) }).collect();
                let mut s = b.add_f32(f[0], f[1]);
                for k in 2..8 {
                    s = b.add_f32(s, f[k]);
                }
                s
            })
        };
        let inv: Vec<ValueId> = (0..8)
            .map(|r| {
                let l = sum_head(b, ev(self.g) * 8 + r);
                let rl = b.rcp_f32(l);
                b.mul_f32(rl, self.ksf)
            })
            .collect();
        let ob = out.base as i32;
        out.at(&t00).write(b, |b| {
            for nt in 0..8usize {
                for r in 0..8u8 {
                    let e = b.frag_slice(Opnd::Var(ov[nt]), r, 1);
                    let ef = f32_of(b, e);
                    let y = b.mul_f32(ef, inv[r as usize]);
                    let h = b.cvt_pk_bf16_f32(y, y);
                    let a = b.bind_i32(((ev(self.g) * 8 + r as i32) * 512 + ev(self.w) * 128 + (16 * nt as i32) + ev(self.m)) * 2 + ob);
                    b.store_lds(a, h, MemTy::U16, false);
                }
            }
        });
        b.wait_lds(0);
        b.barrier();
        let rb = b.bind_i32(ev(self.bidx) * (16 * 512 * 2) + ev(tid) * 128);
        let prb = b.let_(Op::PtrAdd { base: Opnd::Val(self.pr), byte_off: Opnd::Val(rb) });
        out.at(&t00).read(b, |b| {
            let rows: Vec<ValueId> = (0..8).map(|k| { let a = b.bind_i32(ev(tid) * 128 + (16 * k + ob)); b.load_lds_b128(a) }).collect();
            for k in 0..8u32 {
                b.store_global(prb, 16 * k, rows[k as usize], MemTy::B128, false);
            }
        });
        // LSE of head m: ln(2) (m + log2 L)
        let l = sum_head(b, ev(self.m) + 0);
        let am = b.bind_i32(ev(lb) + 512 + ev(self.m) * 4);
        let mr = lr.read(b, |b| b.load_lds(am, MemTy::B32));
        let mf = f32_of(b, mr);
        let lg = b.log2_f32(l);
        let s2 = b.add_f32(mf, lg);
        let ln2 = b.c_f32(std::f32::consts::LN_2);
        let lse = b.mul_f32(s2, ln2);
        let off = b.bind_i32((ev(self.bidx) * 16 + ev(self.m)) * 4);
        let pl = b.let_(Op::PtrAdd { base: Opnd::Val(self.plse), byte_off: Opnd::Val(off) });
        b.store_global(pl, 0, lse, MemTy::B32, false);
    }
}
