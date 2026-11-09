//! bf16 attention forward on the v3 structure (perf pass of docs/op-coverage-design.md §7.5): `fmha_fwd_hd128_bf16_v3[_causal]`, the aiter
//! fmha_v3 656-byte ABI (as `bf16.rs`). 512 threads = two groups of 4 waves (`Roles::split`), 256 query rows per CTA (32 per wave), 128-key
//! K / V tiles in a 2-slot LDS ring, `tile::Pipeline` with `Cadence::TwoPhaseOffset`: group B runs one phase behind group A, so one group's
//! softmax (VALU) overlaps the other group's matrix phase (PV of tile t + QK of tile t + 1); group A stages K, group B stages V.
//! Math as `bf16.rs` (S^T = K Q^T, O^T = V^T P^T with bf16 32x32x16; one query per lane; V^T by ds_read_b64_tr_b16 in the P key order), lazy
//! reference (moves when a tile max passes it by 8, wave-level branch), the key mask only on the tiles that need it (uniform branch in the
//! VALU phase).
use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, MmaKind, MmaShape, MmaTy, Op, Opnd, ValueId, VarId, I32};
use amdgpu_tile::components::gqa_kv_head;
use amdgpu_tile::pipeline::{Cadence, PipeBody, Pipeline, TileIt};
use amdgpu_tile::{ev, LdsPlan, Lane, Layout as TLayout, Role, Roles, Shared, SlotIdx};

const D: i32 = 128;
const ROW: i32 = 2 * D;
/// keys per tile
const KT: i32 = 128;
/// query rows per CTA
const QR: i32 = 256;

struct Ring {
    ks: Shared,
    vs: Shared,
    /// per issue (8 per wave and tensor): the loop-invariant voffset of K (swizzled source chunk) and V
    k_vo: Vec<ValueId>,
    v_vo: Vec<ValueId>,
    r_k: ValueId,
    r_v: ValueId,
    k_so0: ValueId,
    v_so0: ValueId,
    seqs: ValueId,
    wig: ValueId,
    /// hd192 (MLA prefill): the K rope part (dims 128..191) in its own ring, rows of 128 bytes (chunk ^ key % 8), 4 issues per wave
    kbs: Option<Shared>,
    kb_vo: Vec<ValueId>,
    /// hd192: the V row stride (K and V rows differ); None = `seqs` for both
    v_seqs: Option<ValueId>,
    /// waves per group (the DMA block stride)
    wpg: i32,
}

struct Body<'a> {
    ring: &'a Ring,
    kind: MmaKind,
    qf: &'a [ValueId],
    kr: &'a [ValueId],
    /// hd192: the rope-part K read offsets (4 chunks of 16 dims)
    kr2: &'a [ValueId],
    /// V^T read offsets per (e, r): index 2 e + r
    vr: Vec<ValueId>,
    csc: ValueId,
    neg_inf: ValueId,
    zero16: ValueId,
    half: ValueId,
    /// keys visible to the lane's query: k <= lim; the CTA-uniform smallest limit (tiles ending at or below it need no mask)
    lim: ValueId,
    lim_min: ValueId,
    s_init: Vec<ValueId>,
    m: Option<VarId>,
    l: Option<VarId>,
    o: Vec<VarId>,
    s: Vec<VarId>,
    p: Vec<ValueId>,
    /// the vendor's priority ping-pong: s_setprio 1 in the matrix phase, 0 in the softmax phase
    prio: bool,
    /// keys per tile and its 32-key blocks
    kt: i32,
    nb: usize,
}

impl Body<'_> {
    fn kbase(&self, b: &mut FnBuilder, slot: &SlotIdx) -> I32 {
        let o = self.ring.ks.at(slot).off_v(b);
        o + self.ring.ks.base as i32
    }
    fn vbase(&self, b: &mut FnBuilder, slot: &SlotIdx) -> I32 {
        let o = self.ring.vs.at(slot).off_v(b);
        o + self.ring.vs.base as i32
    }
    /// S^T of the tile in `slot`: 8 k-chunks x 4 blocks of 32 keys; the 4 K fragments of chunk c + 1 are read before the 4 MFMAs of chunk c
    /// (one group of reads in flight behind the MFMAs; 4 independent accumulator chains); the lane bases are bound once per phase
    fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, into: Option<&[VarId]>) -> Vec<ValueId> {
        let kb = self.kbase(b, slot);
        let (kind, qf, kr) = (self.kind, self.qf, self.kr);
        let bases: Vec<ValueId> = kr.iter().map(|r| b.bind_i32(ev(*r) + kb.clone())).collect();
        let acc = self.ring.ks.at(slot).read(b, |b| {
            let rd = |b: &mut FnBuilder, c: usize| -> Vec<ValueId> {
                (0..self.nb as i32).map(|blk| { let a = b.bind_i32(ev(bases[c]) + blk * 32 * ROW); b.load_lds_b128(a) }).collect()
            };
            let mut acc: Vec<Option<ValueId>> = vec![None; self.nb];
            let mut cur = rd(b, 0);
            for c in 0..8usize {
                let nxt = (c + 1 < 8).then(|| rd(b, c + 1));
                for blk in 0..self.nb {
                    match into {
                        Some(v) => b.var_mfma(v[blk], kind, cur[blk], qf[c], c == 0),
                        None => acc[blk] = Some(b.mfma(kind, cur[blk], qf[c], acc[blk].map(Opnd::Val))),
                    }
                }
                if let Some(n) = nxt {
                    cur = n;
                }
            }
            acc
        });
        // hd192: the rope part (dims 128..191: 4 more k-chunks from the second ring)
        let acc = match &self.ring.kbs {
            Some(kbs) => {
                let o = kbs.at(slot).off_v(b);
                let kb2 = o + kbs.base as i32;
                let bases2: Vec<ValueId> = self.kr2.iter().map(|r| b.bind_i32(ev(*r) + kb2.clone())).collect();
                let mut acc = acc;
                kbs.at(slot).read(b, |b| {
                    for c in 0..4usize {
                        for blk in 0..self.nb {
                            let a = b.bind_i32(ev(bases2[c]) + (blk as i32) * 32 * 128);
                            let kf = b.load_lds_b128(a);
                            match into {
                                Some(v) => b.var_mfma(v[blk], kind, kf, qf[8 + c], false),
                                None => acc[blk] = Some(b.mfma(kind, kf, qf[8 + c], acc[blk].map(Opnd::Val))),
                            }
                        }
                    }
                });
                acc
            }
            None => acc,
        };
        acc.into_iter().flatten().collect()
    }
    /// O^T += V^T P^T over the 128 keys of the tile in `slot`: 8 steps of 16 keys x 4 blocks of 32 d; the 8 transposed reads of step t + 1 are
    /// issued before the 4 MFMAs of step t
    fn pv(&self, b: &mut FnBuilder, slot: &SlotIdx) {
        let vb = self.vbase(b, slot);
        let kind = self.kind;
        let bases: Vec<ValueId> = self.vr.iter().map(|v| b.bind_i32(ev(*v) + vb.clone())).collect();
        let p = self.p.clone();
        let o = self.o.clone();
        self.ring.vs.at(slot).read(b, |b| {
            let rd = |b: &mut FnBuilder, t: usize| -> Vec<ValueId> {
                let (sb, tt) = (t / 2, t % 2);
                (0..4i32)
                    .map(|e| {
                        // the two transposed reads written straight into the operand's halves (no copies)
                        let a0 = b.bind_i32(ev(bases[(2 * e) as usize]) + (32 * sb as i32 + 16 * tt as i32) * ROW);
                        let a1 = b.bind_i32(ev(bases[(2 * e + 1) as usize]) + (32 * sb as i32 + 16 * tt as i32) * ROW);
                        b.load_lds_tr_b16x2(a0, a1)
                    })
                    .collect()
            };
            let mut cur = rd(b, 0);
            let steps = (self.kt / 16) as usize;
            for t in 0..steps {
                let nxt = (t + 1 < steps).then(|| rd(b, t + 1));
                let pf = b.pack4(p[4 * t], p[4 * t + 1], p[4 * t + 2], p[4 * t + 3]);
                for e in 0..4usize {
                    b.var_mfma(o[e], kind, cur[e], pf, false);
                }
                if let Some(n) = nxt {
                    cur = n;
                }
            }
        });
    }
}
impl PipeBody for Body<'_> {
    fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
        // group A stages K, group B stages V (each the whole 32 KB tile: 8 issues of 1 KB per wave)
        let r = self.ring;
        let (sh, vo, rs, so0) = match role {
            Role::B => (&r.vs, &r.v_vo, r.r_v, r.v_so0),
            _ => (&r.ks, &r.k_vo, r.r_k, r.k_so0),
        };
        let st = match (role, r.v_seqs) { (Role::B, Some(vs)) => vs, _ => r.seqs };
        let so = { let v = b.bind_i32(ev(so0) + key0.clone() * ev(st)); b.read_first_lane(v) };
        let base = { let o = sh.at(slot).off_v(b); b.bind_i32(o + sh.base as i32) };
        sh.at(slot).write(b, |b| {
            for (i, v) in vo.iter().enumerate() {
                let m0 = { let x = b.bind_i32(ev(base) + ((ev(r.wig) + r.wpg * i as i32) << 10)); b.read_first_lane(x) };
                b.buffer_load_lds_b128(rs, *v, so, m0, 0);
            }
        });
        // hd192: group A also stages the K rope part (16 KB: 4 issues per wave)
        if let (Some(kbs), Role::A) = (&r.kbs, role) {
            let base2 = { let o = kbs.at(slot).off_v(b); b.bind_i32(o + kbs.base as i32) };
            kbs.at(slot).write(b, |b| {
                for (i, v) in r.kb_vo.iter().enumerate() {
                    let m0 = { let x = b.bind_i32(ev(base2) + ((ev(r.wig) + r.wpg * i as i32) << 10)); b.read_first_lane(x) };
                    b.buffer_load_lds_b128(rs, *v, so, m0, 0);
                }
            });
        }
    }
    fn prologue_compute(&mut self, b: &mut FnBuilder, _role: Role, t0: &TileIt) {
        self.s_init = self.qk(b, &t0.cur, None);
    }
    fn loop_state(&mut self, b: &mut FnBuilder) -> Vec<(DType, Opnd)> {
        let m0 = b.c_f32(-1.0e30);
        let l0 = b.c_f32(0.0);
        let mut v = vec![(DType::F32, Opnd::Val(m0)), (DType::F32, Opnd::Val(l0))];
        v.extend((0..4).map(|_| (DType::Frag16, Opnd::Val(self.zero16))));
        v.extend(self.s_init.iter().map(|s| (DType::Frag16, Opnd::Val(*s))));
        v
    }
    fn bind_state(&mut self, vars: &[VarId]) {
        self.m = Some(vars[0]);
        self.l = Some(vars[1]);
        self.o = vars[2..6].to_vec();
        self.s = vars[6..6 + self.nb].to_vec();
    }
    fn v_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if self.prio {
            b.set_prio(0);
        }
        let (m_v, l_v) = (self.m.expect("m"), self.l.expect("l"));
        let s = self.s.clone();
        // the key mask on tiles past the smallest limit of the CTA (uniform branch, VALU only): masked lanes' S = -inf in place
        let key_t = b.bind_i32(t.key0());
        let partial = b.bind_cond((ev(key_t) + (self.kt - 1)).gt(ev(self.lim_min)));
        b.begin_if(Opnd::Val(partial));
        {
            let key0 = b.bind_i32(ev(key_t) + (ev(self.half) << 2));
            for blk in 0..self.nb {
                for r in 0..16u8 {
                    let off = 32 * blk as i32 + 8 * (r as i32 / 4) + (r as i32 % 4);
                    let x = b.frag16_lane(Opnd::Var(s[blk]), r);
                    let x = b.bitcast_f32(x);
                    let keep = b.bind_cond((ev(key0) + off).le(ev(self.lim)));
                    let y = b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(keep), a: Opnd::Val(x), b: Opnd::Val(self.neg_inf) });
                    b.var_set_lane(s[blk], r, y);
                }
            }
        }
        b.end_if();
        let mut sv: Vec<ValueId> = Vec::with_capacity(64);
        for blk in 0..self.nb {
            for r in 0..16u8 {
                let x = b.frag16_lane(Opnd::Var(s[blk]), r);
                sv.push(b.bitcast_f32(x));
            }
        }
        // row max by v_max3_f32 (two values per instruction)
        let mut mx = sv[0];
        for pair in sv[1..].chunks(2) {
            mx = if pair.len() == 2 { b.max3_f32(mx, pair[0], pair[1]) } else { b.max_f32(mx, pair[0]) };
        }
        let (pa, pb) = b.permlane32_swap(mx, mx);
        let (pa, pb) = (b.bitcast_f32(pa), b.bitcast_f32(pb));
        let mt = b.max_f32(pa, pb);
        let mt = b.mul_f32(mt, self.csc);
        // lazy reference: moves (to max(m, tile max)) when some lane's tile max passes m by 8 (wave-level branch)
        let thr = b.c_f32(8.0);
        let mlim = b.add_f32(Opnd::Var(m_v), thr);
        let grow = b.cmp_lt_f32(mlim, mt);
        let grow = b.any_lane(grow);
        b.begin_if(Opnd::Val(grow));
        {
            let mn = b.max_f32(Opnd::Var(m_v), mt);
            let dm = b.sub_f32(Opnd::Var(m_v), mn);
            let alpha = b.exp2_f32(dm);
            let ad = b.frag2_dup(alpha);
            for v in &self.o {
                b.var_scale_pk(*v, ad);
            }
            let l2 = b.mul_f32(Opnd::Var(l_v), alpha);
            b.set_var(l_v, l2);
            b.set_var(m_v, mn);
        }
        b.end_if();
        let zf = b.c_f32(0.0);
        let nm = b.sub_f32(zf, Opnd::Var(m_v));
        let p: Vec<ValueId> = sv
            .iter()
            .map(|x| {
                let t = b.let_(Op::FmaF32(Opnd::Val(*x), Opnd::Val(self.csc), Opnd::Val(nm)));
                b.exp2_f32(t)
            })
            .collect();
        let mut ps = p[0];
        for x in &p[1..] {
            ps = b.add_f32(ps, *x);
        }
        let l_new = b.add_f32(Opnd::Var(l_v), ps);
        b.set_var(l_v, l_new);
        // P words in the PV order: step t (16 keys) = block t / 2, values 8 (t % 2) + 0..7
        self.p = (0..(self.kt / 16) as usize)
            .flat_map(|t| {
                let i0 = 16 * (t / 2) + 8 * (t % 2);
                (0..4).map(move |j| (i0 + 2 * j, i0 + 2 * j + 1)).collect::<Vec<_>>()
            })
            .map(|(a, c)| b.cvt_pk_bf16_f32(p[a], p[c]))
            .collect();
    }
    fn m_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if self.prio {
            b.set_prio(1);
        }
        self.pv(b, &t.cur);
        if t.staged {
            let s = self.s.clone();
            self.qk(b, &t.next, Some(&s));
        }
    }
}

pub fn fmha_fwd_hd128_bf16_v3(name: &str, causal: bool) -> Function {
    fmha_fwd_hd128_bf16_v3_x(name, causal, false)
}

/// `split` (`..._ks`, for small grids): the key range of every q tile is cut into S = 2^lg_s parts (grid z = B S; our ABI extension: lg_s
/// at kernarg 656, S = 2^lg_s at 660, fp32 partial O / l [B, H, SQ, S, 128] at 672, partial LSE [B, H, SQ, S] at 688); `attn_combine_bf16_hd128_q16` merges them.
pub fn fmha_fwd_hd128_bf16_v3_x(name: &str, causal: bool, split: bool) -> Function {
    fmha_fwd_bf16_v3_d(name, causal, split, 128)
}

/// MLA prefill (`fmha_fwd_hd192_hd128_bf16_v3[_causal]`, dqk = 192): Q / K head dim 192 (128 nope + 64 rope), V / O 128 (the vendor's
/// fwd_hd192_hd128 ABI: qk_head_dim 192). K's rope part rides a second ring (rows of 128 bytes); QK runs 12 k-chunks.
pub fn fmha_fwd_bf16_v3_d(name: &str, causal: bool, split: bool, dqk: i32) -> Function {
    // hd192 causal (not the key-split form): the q-tile pairing with fresh lane partitions for the second pass (without them the pair does
    // not fit the VGPRs); `_np` names the unpaired form (small grids, `dispatch::hd192_prefill_kernel`)
    let pair = dqk == 192 && !split && !name.contains("_np");
    // `_q128`: the 128-row / 64-key-tile form (mid-size causal grids, `dispatch`)
    let small = name.contains("_q128");
    fmha_fwd_bf16_v3_o(name, causal, split, dqk, V3Opts { fresh2: pair, pair192: pair, small })
}

/// Build options of `fmha_fwd_bf16_v3_d` (A/B variants).
#[derive(Clone, Copy, Debug, Default)]
pub struct V3Opts {
    /// the paired causal form rebinds the lane partitions (DMA voffsets, read offsets) for its second pass
    pub fresh2: bool,
    /// hd192 causal pairs q tiles too
    pub pair192: bool,
    /// 128 query rows (two groups of 2 waves) x 64-key tiles: 64 KB (hd192 80 KB) of LDS, two workgroups per CU
    pub small: bool,
}

/// `fmha_fwd_bf16_v3_d` with build options.
pub fn fmha_fwd_bf16_v3_o(name: &str, causal: bool, split: bool, dqk: i32, opts: V3Opts) -> Function {
    assert!(dqk == 128 || dqk == 192, "bf16 v3: head dim 128, or 192 / 128");
    let hd192 = dqk == 192;
    #[allow(non_snake_case)]
    let QROW = 2 * dqk;
    // geometry: 256 query rows (two groups of 4 waves) x 128-key tiles, or (`opts.small`) 128 rows (two groups of 2 waves) x 64-key tiles:
    // half the LDS (two workgroups per CU) and twice the q tiles, for grids that leave CUs idle
    let (qr, kt, wpg) = if opts.small { (128i32, 64i32, 2i32) } else { (QR, KT, 4) };
    let (qr_lg, kt_lg) = (qr.trailing_zeros() as i32, kt.trailing_zeros() as i32);
    let mut b = FnBuilder::new(name, (2 * wpg * 64) as u32);
    // the `_vs128` policy (full reordering list scheduler) + ldsink / wsink / ldahead32: measured +6.6 % / +7.7 % / +6.7 % (8192², 2048², 1024²)
    // over the Gemm profile
    // (causal, also the causal key-split form: the Gemm profile measured faster (955 vs 915 TF/s at 8192², 350 vs 328 _ks S=2 at 4096²);
    // the non-causal _ks form gains on Vs128: 660 vs 618 TF/s at 4096² S=1 - its earlier HW fault was the in-flight WAW backend bug, fixed)
    // (hd192 causal: the Gemm profile runs out of VGPRs - Vs128 for every hd192 form)
    let vs_policy = !causal || dqk == 192;
    b.compile_profile(if vs_policy { CompileProfile::Vs128 } else { CompileProfile::Gemm });
    // ---- ABI (the vendor's v3, as bf16.rs) ----
    for (n, i) in [("ptr_o", 0u32), ("ptr_q", 1), ("ptr_k", 2), ("ptr_v", 3), ("ptr_lse", 4)] {
        b.arg(n, DType::Ptr, 16 * i, 8);
    }
    let i32_names = [
        ("scalar", 5u32), ("s_seq_len", 6), ("s_Seqs", 7), ("s_Ts", 8), ("s_Hs", 9), ("s_Bs", 10), ("s_gqa", 11), ("s_k_Seqs", 12), ("s_k_Hs", 13),
        ("s_k_Bs", 14), ("s_opt", 15), ("s_lse", 16), ("s_kv_seq_len", 17), ("s_qk_head_dim", 18), ("s_v_head_dim", 19), ("s_q_head_num", 20),
        ("s_v_Seqs", 21), ("s_v_Hs", 22), ("s_v_Bs", 23), ("s_o_Seqs", 24), ("s_o_Hs", 25), ("s_o_Bs", 26), ("s_lse_Hs", 29),
    ];
    for (n, i) in i32_names {
        b.arg(n, if n == "scalar" { DType::F32 } else { DType::I32 }, 16 * i, 4);
    }
    let (bb, sq, sk, hh, gq) = (Ext::grid_z(), Ext::ka(96), Ext::ka(272), Ext::ka(320), Ext::ka(176));
    b.ka_range(96, 1, 1 << 20);
    b.ka_range(272, 1, 1 << 20);
    b.ka_range(176, 1, 16);
    b.sym_range("HK", 1, 64);
    b.ka_is(320, Ext::sym("HK") * gq.clone());
    let hk = Ext::sym("HK");
    let rowb = Ext::c(ROW as i64);
    // Q / K rows (hd192: 384 bytes)
    let qrowb = Ext::c(QROW as i64);
    b.ka_is(288, Ext::c(dqk as i64));
    b.ka_is(304, Ext::c(D as i64));
    b.ka_is(112, hh.clone() * qrowb.clone());
    b.ka_is(144, qrowb.clone());
    b.ka_is(160, sq.clone() * hh.clone() * qrowb.clone());
    for (s, hs, bs, rb) in [(192u32, 208u32, 224u32, qrowb.clone()), (336, 352, 368, rowb.clone())] {
        b.ka_is(s, hk.clone() * rb.clone());
        b.ka_is(hs, rb.clone());
        b.ka_is(bs, sk.clone() * hk.clone() * rb);
    }
    b.ka_is(384, hh.clone() * rowb.clone());
    b.ka_is(400, rowb.clone());
    b.ka_is(416, sq.clone() * hh.clone() * rowb.clone());
    b.ka_is(464, sq.clone() * Ext::c(4));
    // causal: the vendor's pairing (two q tiles per CTA); hd192 causal runs one q tile per CTA (the second pass's registers do not fit)
    let pair = causal && (!hd192 || opts.pair192);
    b.grid_is(0, if pair { sq.clone().ceil_div(qr as i64).ceil_div(2) } else { sq.clone().ceil_div(qr as i64) });
    b.grid_is(1, hh.clone());
    if split {
        b.arg("lg_s", DType::I32, 656, 4);
        b.arg("S", DType::I32, 660, 4);
        b.ka_is(660, Ext::sym("SPL"));
        b.arg("ptr_po", DType::Ptr, 672, 8);
        b.arg("ptr_plse", DType::Ptr, 688, 8);
        b.sym_range("LS", 0, 4);
        b.ka_is(656, Ext::sym("LS"));
        b.sym_range("SPL", 1, 16);
        b.sym_range("BB", 1, 1 << 12);
        b.grid_is(2, Ext::sym("BB") * Ext::sym("SPL"));
        b.arg_tensor("ptr_po", Ext::sym("BB") * hh.clone() * sq.clone() * Ext::sym("SPL") * Ext::c(4 * D as i64));
        b.arg_tensor("ptr_plse", Ext::sym("BB") * hh.clone() * sq.clone() * Ext::sym("SPL") * Ext::c(4));
    }
    let bb = if split { Ext::sym("BB") } else { bb };
    b.arg_tensor("ptr_q", bb.clone() * sq.clone() * hh.clone() * qrowb.clone());
    b.arg_tensor("ptr_o", bb.clone() * sq.clone() * hh.clone() * rowb.clone());
    b.arg_tensor("ptr_k", bb.clone() * sk.clone() * hk.clone() * qrowb);
    b.arg_tensor("ptr_v", bb.clone() * sk.clone() * hk * rowb);
    b.arg_tensor("ptr_lse", bb * hh * sq * Ext::c(4));

    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 16 * i });
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    let (p_o, p_q, p_k, p_v, p_lse) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4));
    let scale = b.let_(Op::KernargF32 { offset: 80 });
    let (s_q, q_seqs, q_hs, q_bs, gqa) = (ki(&mut b, 6), ki(&mut b, 7), ki(&mut b, 9), ki(&mut b, 10), ki(&mut b, 11));
    let (k_seqs, k_hs, k_bs, lse_on, s_kv) = (ki(&mut b, 12), ki(&mut b, 13), ki(&mut b, 14), ki(&mut b, 16), ki(&mut b, 17));
    let (v_seqs, v_hs, v_bs, o_seqs, o_hs, o_bs, lse_hs) = (ki(&mut b, 21), ki(&mut b, 22), ki(&mut b, 23), ki(&mut b, 24), ki(&mut b, 25), ki(&mut b, 26), ki(&mut b, 29));
    let (qt0, h0, z0) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    // split: z = batch S + part
    let lg_s = split.then(|| b.let_(Op::KernargI32 { offset: 656 }));
    let (bat0, part) = match lg_s {
        Some(l) => {
            let bt = { let v = b.bind_i32(ev(z0).lshr(ev(l))); b.read_first_lane(v) };
            let pt = { let v = b.bind_i32(ev(z0) - (ev(bt) << ev(l))); b.read_first_lane(v) };
            b.assume_trusted(bt, Ext::c(0), Ext::sym("BB") - Ext::c(1), "split: z = batch 2^lg_s + part (grid z = B S, S = 2^lg_s)");
            b.assume_trusted(pt, Ext::c(0), Ext::sym("SPL") - Ext::c(1), "split: the part index < S");
            (bt, Some(pt))
        }
        None => (z0, None),
    };
    // (XCD order of the (q tile, head) grid measured no gain: not used)
    let (qt, h, bat) = (qt0, h0, bat0);
    let lane = Lane::new(&mut b);
    let roles = Roles::split(&mut b, &lane, wpg as u32);
    let hkv = gqa_kv_head(&mut b, h, gqa, "HK");
    let hkv = b.read_first_lane(hkv);
    let half = b.bind_i32(lane.half());
    let desc = |b: &mut FnBuilder, p: ValueId, bs: ValueId| {
        let n = b.bind_i32((ev(bat) + 1) * ev(bs));
        let n = b.read_first_lane(n);
        b.buffer_desc_dyn(p, n)
    };
    let (r_q, r_k, r_v, r_o) = (desc(&mut b, p_q, q_bs), desc(&mut b, p_k, k_bs), desc(&mut b, p_v, v_bs), desc(&mut b, p_o, o_bs));
    let q_so = { let v = b.bind_i32(ev(bat) * ev(q_bs) + ev(h) * ev(q_hs)); b.read_first_lane(v) };

    // ---- LDS: the K / V rings (2 slots x 32 KB each) ----
    let mut lds = LdsPlan::new();
    let ks = lds.ring("K", (kt * ROW) as i64, 2, TLayout::Linear);
    let vs = lds.ring("V", (kt * ROW) as i64, 2, TLayout::Linear);
    // hd192: the K rope ring (128 keys x 128 bytes per slot)
    let kbs = hd192.then(|| lds.ring("KR", (kt * 128) as i64, 2, TLayout::Linear));
    lds.commit(&mut b);
    // DMA (per group, 4 waves): block bi = w + 4 i (i < 8) of 1 KB = keys 4 bi + lane / 16, chunk lane % 16; K source chunk ^ (key % 16)
    let wig = roles.wave_in_group;
    let l16 = b.bind_i32(lane.lane() & 15);
    let k_so0 = { let v = b.bind_i32(ev(bat) * ev(k_bs) + ev(hkv) * ev(k_hs)); b.read_first_lane(v) };
    let v_so0 = { let v = b.bind_i32(ev(bat) * ev(v_bs) + ev(hkv) * ev(v_hs)); b.read_first_lane(v) };
    let parts1 = lane_parts(&mut b, &lane, wig, wpg, l16, half, k_seqs, v_seqs, hd192);
    // the second pass of the paired causal form rebinds them (`fresh2`): reusing the first pass's values keeps them live through its last tile
    let fresh2 = pair && opts.fresh2;
    let (ks, vs) = (&ks, &vs);

    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let zero16 = b.frag_zero(DType::Frag16);
    let kind = MmaKind::new(MmaShape::M32N32K16, MmaTy::Bf16);
    let big = b.c_i32(0x7fff_fff0);
    let c0z = { let z = b.c_i32(0); b.read_first_lane(z) };
    let qoff = b.bind_i32(ev(s_kv) - ev(s_q));
    // causal: the vendor's pairing - CTA x takes q tiles x and nt - 1 - x (equal work per CTA; the grid is halved)
    let nt = { let v = b.bind_i32((ev(s_q) + (qr - 1)).lshr(qr_lg)); b.read_first_lane(v) };
    let passes: Vec<(ValueId, Option<ValueId>)> = if pair {
        let qb = { let v = b.bind_i32(ev(nt) - 1 - ev(qt)); b.read_first_lane(v) };
        let second = b.bind_cond(ev(qb).gt(ev(qt)));
        b.wg_union(&[qt, qb], &[0], Ext::c(0), Ext::ka(96).ceil_div(qr as i64) - Ext::c(1), "causal pairing: CTA x takes q tiles x and nt - 1 - x (the second only when nt - 1 - x > x), grid x = ceil(nt / 2): every q tile once");
        vec![(qt, None), (qb, Some(second))]
    } else {
        vec![(qt, None)]
    };
    for (qt, guard) in passes {
    if let Some(g) = guard {
        // the LDS rings are reused by the second pass: every wave is done with the first
        b.wait_lds(0);
        b.barrier();
        b.begin_if(Opnd::Val(g));
    }
    let row = b.bind_i32((ev(qt) << qr_lg) + (lane.wave() << 5) + lane.j());
    let row_c = b.bind_i32(ev(row).min(ev(s_q) - 1));
    let q_vo = b.bind_i32(ev(row_c) * ev(q_seqs) + (ev(half) << 4));
    let qf: Vec<ValueId> = (0..if hd192 { 12 } else { 8 }).map(|c| b.buffer_load_b128(r_q, q_vo, q_so, (32 * c) as u32)).collect();
    // ---- key range: causal keys k <= q + SK - SQ ----
    let kv_end = if causal {
        let v = b.bind_i32((((ev(qt) + 1) << qr_lg) - 1 + ev(qoff) + 1).min(ev(s_kv)).max(0));
        b.read_first_lane(v)
    } else {
        s_kv
    };
    let ntk = { let v = b.bind_i32((ev(kv_end) + (kt - 1)).lshr(kt_lg)); b.read_first_lane(v) };
    // split: this part's tiles [t_lo, t_hi) of ceil(ntk / S) each; keys relative to k_lo = 128 t_lo (the K / V origins move, the limits shift)
    let (nkv, kv_len, k_lo) = match (lg_s, part) {
        (Some(l), Some(pt)) => {
            let per = { let v = b.bind_i32((ev(ntk) + (1 << 0) * ((I32::c(1) << ev(l)) - 1)).lshr(ev(l))); b.read_first_lane(v) };
            let t_lo = { let v = b.bind_i32((ev(pt) * ev(per)).min(ev(ntk))); b.read_first_lane(v) };
            let t_hi = { let v = b.bind_i32((ev(t_lo) + ev(per)).min(ev(ntk))); b.read_first_lane(v) };
            let n = { let v = b.bind_i32(ev(t_hi) - ev(t_lo)); b.read_first_lane(v) };
            let klo = { let v = b.bind_i32(ev(t_lo) << kt_lg); b.read_first_lane(v) };
            let len = { let v = b.bind_i32((ev(kv_end) - ev(klo)).max(0)); b.read_first_lane(v) };
            (n, len, Some(klo))
        }
        _ => (ntk, kv_end, None),
    };
    let lim = b.bind_i32(if causal { (ev(row) + ev(qoff)).min(ev(s_kv) - 1) } else { ev(s_kv) - 1 });
    let lim_min = {
        let v = b.bind_i32(if causal { ((ev(qt) << qr_lg) + ev(qoff)).min(ev(s_kv) - 1) } else { ev(s_kv) - 1 });
        b.read_first_lane(v)
    };
    let (lim, lim_min, k_so, v_so) = match k_lo {
        Some(kl) => {
            let lr = b.bind_i32(ev(lim) - ev(kl));
            let lmr = { let v = b.bind_i32(ev(lim_min) - ev(kl)); b.read_first_lane(v) };
            let ko = { let v = b.bind_i32(ev(k_so0) + ev(kl) * ev(k_seqs)); b.read_first_lane(v) };
            let vo = { let v = b.bind_i32(ev(v_so0) + ev(kl) * ev(v_seqs)); b.read_first_lane(v) };
            (lr, lmr, ko, vo)
        }
        None => (lim, lim_min, k_so0, v_so0),
    };
    // pass 2 of the paired form with `fresh2`: its own lane partitions
    let fresh = (fresh2 && guard.is_some()).then(|| {
        let half2 = b.bind_i32(lane.half());
        let l16b = b.bind_i32(lane.lane() & 15);
        (half2, lane_parts(&mut b, &lane, wig, wpg, l16b, half2, k_seqs, v_seqs, hd192))
    });
    let (half, lp) = match &fresh {
        Some((h, p)) => (*h, p),
        None => (half, &parts1),
    };
    let ring = Ring { ks: ks.clone(), vs: vs.clone(), k_vo: lp.k_vo.clone(), v_vo: lp.v_vo.clone(), r_k, r_v, k_so0: k_so, v_so0: v_so, seqs: k_seqs, wig, kbs: kbs.clone(), kb_vo: lp.kb_vo.clone(), v_seqs: hd192.then_some(v_seqs), wpg };
    let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, nkv, kv_len, kt, Ext::ka(272).ceil_div(kt as i64));
    pipe.key_unclamped = true;

    let any = b.bind_cond(ev(nkv).gt(0));
    b.begin_if(Opnd::Val(any));
    roles.each(&mut b, |b, role| {
        let mut body = Body {
            ring: &ring, kind, qf: &qf, kr: &lp.kr, kr2: &lp.kr2, vr: lp.vr.clone(), csc, neg_inf, zero16, half, lim, lim_min,
            s_init: Vec::new(), m: None, l: None, o: Vec::new(), s: Vec::new(), p: Vec::new(), prio: true, kt, nb: (kt / 32) as usize,
        };
        pipe.run(b, role, &mut body);
        // ---- epilogue: O = O^T / l (bf16, 8-byte stores), LSE ----
        let (m_v, l_v) = (body.m.expect("m"), body.l.expect("l"));
        let (la, lb) = b.permlane32_swap(Opnd::Var(l_v), Opnd::Var(l_v));
        let (la, lb) = (b.bitcast_f32(la), b.bitcast_f32(lb));
        let l_tot = b.add_f32(la, lb);
        let inv = b.rcp_f32(l_tot);
        let live = b.bind_cond(ev(row).lt(ev(s_q)));
        if let (Some(_), Some(pt)) = (lg_s, part) {
            // partial O / l (fp32) and LSE (ln) of this part: row (b, h, q) x S + part
            let hq = b.let_(Op::KernargI32 { offset: 320 });
            let (p_po, p_pl) = (b.let_(Op::KernargPtr { offset: 672 }), b.let_(Op::KernargPtr { offset: 688 }));
            let s_cnt = b.let_(Op::KernargI32 { offset: 660 });
            let nrow = { let v = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(s_q) * ev(s_cnt)); b.read_first_lane(v) };
            let npo = { let v = b.bind_i32(ev(nrow) * (4 * D)); b.read_first_lane(v) };
            let npl = { let v = b.bind_i32(ev(nrow) << 2); b.read_first_lane(v) };
            let (r_po, r_pl) = (b.buffer_desc_dyn(p_po, npo), b.buffer_desc_dyn(p_pl, npl));
            let prow = b.bind_i32((((ev(bat) * ev(hq) + ev(h)) * ev(s_q) + ev(row)) * ev(s_cnt)) + ev(pt));
            let pvo = b.bind_i32(ev(prow) * (4 * D) + (ev(half) << 4));
            let pvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(pvo), b: Opnd::Val(big) });
            for e in 0..4usize {
                for u in 0..4u8 {
                    let v: Vec<ValueId> = (0..4u8)
                        .map(|k| {
                            let x = b.frag16_lane(Opnd::Var(body.o[e]), 4 * u + k);
                            let x = b.bitcast_f32(x);
                            let y = b.mul_f32(x, inv);
                            b.bitcast_i32(y)
                        })
                        .collect();
                    let f4 = b.frag4(v[0], v[1], v[2], v[3]);
                    b.buffer_store_b128(r_po, pvo, c0z, ((32 * e as u32) + 8 * u as u32) * 4, f4);
                }
            }
            let lg = b.log2_f32(l_tot);
            let x = b.add_f32(Opnd::Var(m_v), lg);
            let ln2 = b.c_f32(std::f32::consts::LN_2);
            let lse = b.mul_f32(x, ln2);
            let lvo = b.bind_i32(ev(prow) << 2);
            let lvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(lvo), b: Opnd::Val(big) });
            b.buffer_store_f32(r_pl, lvo, c0z, 0, lse);
            return;
        }
        let o_so = { let v = b.bind_i32(ev(bat) * ev(o_bs) + ev(h) * ev(o_hs)); b.read_first_lane(v) };
        let o_vo = b.bind_i32(ev(row) * ev(o_seqs) + (ev(half) << 3));
        let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo), b: Opnd::Val(big) });
        for e in 0..4usize {
            for u in 0..4u8 {
                let v: Vec<ValueId> = (0..4u8)
                    .map(|k| {
                        let x = b.frag16_lane(Opnd::Var(body.o[e]), 4 * u + k);
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
    });
    if let (Some(_), Some(pt)) = (lg_s, part) {
        // an empty part (no tile): O = 0, LSE = -inf (the combine weights it 0)
        b.else_();
        let hq = b.let_(Op::KernargI32 { offset: 320 });
        let (p_po, p_pl) = (b.let_(Op::KernargPtr { offset: 672 }), b.let_(Op::KernargPtr { offset: 688 }));
        let s_cnt = b.let_(Op::KernargI32 { offset: 660 });
        let nrow = { let v = b.bind_i32((ev(bat) + 1) * ev(hq) * ev(s_q) * ev(s_cnt)); b.read_first_lane(v) };
        let npo = { let v = b.bind_i32(ev(nrow) * (4 * D)); b.read_first_lane(v) };
        let npl = { let v = b.bind_i32(ev(nrow) << 2); b.read_first_lane(v) };
        let (r_po, r_pl) = (b.buffer_desc_dyn(p_po, npo), b.buffer_desc_dyn(p_pl, npl));
        let live = b.bind_cond(ev(row).lt(ev(s_q)));
        let prow = b.bind_i32((((ev(bat) * ev(hq) + ev(h)) * ev(s_q) + ev(row)) * ev(s_cnt)) + ev(pt));
        let pvo = b.bind_i32(ev(prow) * (4 * D) + (ev(half) << 4));
        let pvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(pvo), b: Opnd::Val(big) });
        let z = b.c_i32(0);
        let f4 = b.frag4(z, z, z, z);
        for e in 0..4u32 {
            for u in 0..4u32 {
                b.buffer_store_b128(r_po, pvo, c0z, (32 * e + 8 * u) * 4, f4);
            }
        }
        let lvo = b.bind_i32(ev(prow) << 2);
        let lvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(lvo), b: Opnd::Val(big) });
        b.buffer_store_f32(r_pl, lvo, c0z, 0, neg_inf);
    }
    b.end_if();
    if guard.is_some() {
        b.end_if();
    }
    }
    let mut f = b.finish();
    // TEMP A/B: lowering options by name suffix
    if name.contains("_xaf") {
        f.lower_opts.push("addrfold".into());
    }
    // the scheduler tokens also on the causal Gemm-profile build (+2 % at 8192², +1 % at 2048²)
    if !split || vs_policy {
        f.sched_opts = Some("ldsink=1,wsink,ldahead32=1".to_string());
    }
    f
}

/// Merges the S = 2^lg_s key parts of `fmha_fwd_*_bf16_v3[_causal]_ks`: O[b, q, h, :] (bf16, BSHD) = sum_s w_s po_s / sum_s w_s with
/// w_s = exp(lse_s - M); optional LSE [B, H, SQ] = M + ln(sum w). ABI: 0 ptr_o, 8 ptr_po, 16 ptr_plse, 24 ptr_lse, 32 H, 36 SQ, 40 S, 44 lse_on.
/// 256 threads = 16 query rows of one (batch, head) x 16 threads, 8 dims per thread (two 16-byte loads per part), one online pass over the
/// parts (running max / rescale); grid (ceil(SQ / 16), H, B). Replaces the one-row-per-workgroup `attn_combine_bf16_hd128_q` (one dword per
/// thread per part, two passes, latency bound): hd192 B1 H8 2048^2 S4 54.4 -> 41.3 us, 4096^2 S2 122.5 -> 111.8, B4 H16 1024^2 S2 112 -> 86.
pub fn attn_combine_bf16_hd128_q16(name: &str) -> Function {
    let mut b = FnBuilder::new(name, 256);
    b.compile_profile(CompileProfile::Gemm);
    for (n, o) in [("ptr_o", 0u32), ("ptr_po", 8), ("ptr_plse", 16), ("ptr_lse", 24)] {
        b.arg(n, DType::Ptr, o, 8);
    }
    for (n, o) in [("H", 32u32), ("SQ", 36), ("S", 40), ("lse_on", 44)] {
        b.arg(n, DType::I32, o, 4);
    }
    let (hh, sq) = (Ext::ka(32), Ext::ka(36));
    b.ka_range(32, 1, 256);
    b.ka_range(36, 1, 1 << 20);
    b.sym_range("SPL", 1, 16);
    b.ka_is(40, Ext::sym("SPL"));
    b.sym_range("B", 1, 1 << 12);
    b.grid_is(0, sq.clone().ceil_div(16));
    b.grid_is(1, hh.clone());
    b.grid_is(2, Ext::sym("B"));
    let rows = Ext::sym("B") * hh * sq;
    b.arg_tensor("ptr_o", rows.clone() * Ext::c(ROW as i64));
    b.arg_tensor("ptr_po", rows.clone() * Ext::sym("SPL") * Ext::c(4 * D as i64));
    b.arg_tensor("ptr_plse", rows.clone() * Ext::sym("SPL") * Ext::c(4));
    b.arg_tensor("ptr_lse", rows * Ext::c(4));
    let kp = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargPtr { offset: o });
    let ki = |b: &mut FnBuilder, o: u32| b.let_(Op::KernargI32 { offset: o });
    let (p_o, p_po, p_pl, p_l) = (kp(&mut b, 0), kp(&mut b, 8), kp(&mut b, 16), kp(&mut b, 24));
    let (hq, s_q, ns, lse_on) = (ki(&mut b, 32), ki(&mut b, 36), ki(&mut b, 40), ki(&mut b, 44));
    let (qb, h, bat) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    let tid = b.let_(Op::WorkitemId);
    // the thread's query row (clamped for the loads; the stores are guarded) and dims 8 (tid % 16) ..
    let q = b.bind_i32((ev(qb) << 4) + ev(tid).lshr(4));
    let live = b.bind_cond(ev(q).lt(ev(s_q)));
    let qc = b.bind_i32(ev(q).min(ev(s_q) - 1));
    let d0 = b.bind_i32((ev(tid) & 15) << 3);
    let row = b.bind_i32((ev(bat) * ev(hq) + ev(h)) * ev(s_q) + ev(qc));
    let nrow = { let v = b.bind_i32((((ev(bat) * ev(hq) + ev(h)) + 1) * ev(s_q)) * ev(ns)); b.read_first_lane(v) };
    let r_pl = { let n = b.bind_i32(ev(nrow) << 2); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_pl, n) };
    let r_po = { let n = b.bind_i32(ev(nrow) * (4 * D)); let n = b.read_first_lane(n); b.buffer_desc_dyn(p_po, n) };
    let base = b.bind_i32(ev(row) * ev(ns));
    let c0 = b.c_i32(0);
    let neg_big = b.c_f32(-1.0e30);
    let zf = b.c_f32(0.0);
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let mut inits = vec![(DType::I32, Opnd::Val(c0)), (DType::F32, Opnd::Val(neg_big)), (DType::F32, Opnd::Val(zf))];
    inits.extend((0..8).map(|_| (DType::F32, Opnd::Val(zf))));
    let v = b.begin_loop(&inits);
    let s = amdgpu_dsl::varx(v[0]);
    let lo = b.bind_i32((ev(base) + s.clone()) << 2);
    let l = b.buffer_load_f32(r_pl, lo, c0, 0);
    let po = b.bind_i32((((ev(base) + s.clone()) * D) + ev(d0)) << 2);
    let o0 = b.buffer_load_b128(r_po, po, c0, 0);
    let o1 = b.buffer_load_b128(r_po, po, c0, 16);
    let m_new = b.max_f32(Opnd::Var(v[1]), l);
    let d_old = b.sub_f32(Opnd::Var(v[1]), m_new);
    let d_new = b.sub_f32(l, m_new);
    let t_old = b.mul_f32(d_old, l2e);
    let t_new = b.mul_f32(d_new, l2e);
    let a = b.exp2_f32(t_old);
    let w = b.exp2_f32(t_new);
    let den_a = b.mul_f32(Opnd::Var(v[2]), a);
    let den = b.add_f32(den_a, w);
    for k in 0..8usize {
        let f = if k < 4 { o0 } else { o1 };
        let x = b.frag4_lane(f, (k % 4) as u8);
        let x = b.bitcast_f32(x);
        let acc_a = b.mul_f32(Opnd::Var(v[3 + k]), a);
        let acc = b.let_(Op::FmaF32(Opnd::Val(w), Opnd::Val(x), Opnd::Val(acc_a)));
        b.set_var(v[3 + k], acc);
    }
    b.set_var(v[1], m_new);
    b.set_var(v[2], den);
    let n1 = b.bind_i32(s + 1);
    b.set_var(v[0], n1);
    let done = b.bind_cond(ev(n1).ge(ev(ns)));
    b.break_if(done);
    b.end_loop();
    let tiny = b.c_f32(1e-30);
    let den = b.max_f32(Opnd::Var(v[2]), tiny);
    let inv = b.rcp_f32(den);
    let ys: Vec<ValueId> = (0..8).map(|k| b.mul_f32(Opnd::Var(v[3 + k]), inv)).collect();
    let w: Vec<ValueId> = (0..4).map(|k| b.cvt_pk_bf16_f32(ys[2 * k], ys[2 * k + 1])).collect();
    let f4 = b.frag4(w[0], w[1], w[2], w[3]);
    let nob = { let v = b.bind_i32(((ev(bat) + 1) * ev(s_q) * ev(hq)) * ROW); b.read_first_lane(v) };
    let r_o = b.buffer_desc_dyn(p_o, nob);
    let big = b.c_i32(0x7fff_fff0);
    let o_vo = b.bind_i32(((ev(bat) * ev(s_q) + ev(qc)) * ev(hq) + ev(h)) * ROW + (ev(d0) << 1));
    let o_vo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(o_vo), b: Opnd::Val(big) });
    b.buffer_store_b128(r_o, o_vo, c0, 0, f4);
    let on = b.bind_cond(ev(lse_on).ne(0));
    b.begin_if(Opnd::Val(on));
    {
        // natural-log LSE of the row from its first thread
        let lg2 = b.log2_f32(den);
        let ln2 = b.c_f32(std::f32::consts::LN_2);
        let lnl = b.mul_f32(lg2, ln2);
        let lse = b.add_f32(Opnd::Var(v[1]), lnl);
        let nl = { let v = b.bind_i32(((ev(bat) * ev(hq) + ev(h)) + 1) * ev(s_q) << 2); b.read_first_lane(v) };
        let r_l = b.buffer_desc_dyn(p_l, nl);
        let first = b.bind_cond(ev(d0).eq(0));
        let lvo = b.bind_i32(ev(row) << 2);
        let lvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(first), a: Opnd::Val(lvo), b: Opnd::Val(big) });
        let lvo = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(lvo), b: Opnd::Val(big) });
        b.buffer_store_f32(r_l, lvo, c0, 0, lse);
    }
    b.end_if();
    b.finish()
}

/// The lane partitions of the K / V staging and reads (bound values).
struct LaneParts {
    k_vo: Vec<ValueId>,
    v_vo: Vec<ValueId>,
    kb_vo: Vec<ValueId>,
    kr2: Vec<ValueId>,
    kr: Vec<ValueId>,
    vr: Vec<ValueId>,
}

/// DMA voffsets (per group, 4 waves: block bi = w + 4 i (i < 8) of 1 KB = keys 4 bi + lane / 16, chunk lane % 16; source chunk ^ (key % 16)),
/// the hd192 rope-part voffsets and the K / K-rope / V^T read offsets of the lane.
#[allow(clippy::too_many_arguments)]
fn lane_parts(b: &mut FnBuilder, lane: &Lane, wig: ValueId, wpg: i32, l16: ValueId, half: ValueId, k_seqs: ValueId, v_seqs: ValueId, hd192: bool) -> LaneParts {
    let keys: Vec<ValueId> = (0..8).map(|i| b.bind_i32(((ev(wig) + wpg * i) << 2) + lane.lane().lshr(4))).collect();
    let k_vo: Vec<ValueId> = keys.iter().map(|k| b.bind_i32(ev(*k) * ev(k_seqs) + ((ev(l16) ^ (ev(*k) & 15)) << 4))).collect();
    // V swizzled like K (the transposed reads of 4 rows x 4 columns per 16-lane group then hit distinct banks)
    let v_vo: Vec<ValueId> = keys.iter().map(|k| b.bind_i32(ev(*k) * ev(v_seqs) + ((ev(l16) ^ (ev(*k) & 15)) << 4))).collect();
    // hd192 rope part: 1 KB block bi = w + 4 i (i < 4) = keys 8 bi + lane / 8, chunk lane % 8 (source dims 128 + 8 chunk ..: byte 256 +), ^ key % 8
    let kb_vo: Vec<ValueId> = if hd192 {
        (0..4)
            .map(|i| {
                let k = b.bind_i32(((ev(wig) + wpg * i) << 3) + lane.lane().lshr(3));
                b.bind_i32(ev(k) * ev(k_seqs) + 256 + (((lane.lane() & 7) ^ (ev(k) & 7)) << 4))
            })
            .collect()
    } else {
        Vec::new()
    };
    let kr2: Vec<ValueId> = if hd192 { (0..4).map(|c| b.bind_i32(lane.j() * 128 + (((ev(half) + 2 * c) ^ (lane.lane() & 7)) << 4))).collect() } else { Vec::new() };
    // read offsets (as bf16.rs)
    let kr: Vec<ValueId> = (0..8).map(|c| b.bind_i32(lane.j() * ROW + (((ev(half) + 2 * c) ^ (lane.lane() & 15)) << 4))).collect();
    // V^T tr_b16 reads, swizzled: per (dim block e, read r) the lane's row 8 r + 4 h + s / 4 (s = lane % 16; the step rows are multiples of 16:
    // the same key % 16), 16-byte chunk (4 e + 2 ((l % 32) / 16) + (s % 4) / 2) ^ (row % 16), byte 8 (s % 2)
    let vr: Vec<ValueId> = (0..4)
        .flat_map(|e| (0..2).map(move |r| (e, r)))
        .map(|(e, r)| {
            let krow = b.bind_i32(8 * r + ev(half) * 4 + (lane.lane() & 15).lshr(2));
            let ch = b.bind_i32(4 * e + ((lane.lane() & 31).lshr(4) << 1) + (lane.lane() & 3).lshr(1));
            b.bind_i32(ev(krow) * ROW + (((ev(ch) ^ ev(krow)) & 15) << 4) + ((lane.lane() & 1) << 3))
        })
        .collect();
    LaneParts { k_vo, v_vo, kb_vo, kr2, kr, vr }
}