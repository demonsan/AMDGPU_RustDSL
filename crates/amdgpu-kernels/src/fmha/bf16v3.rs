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
}

struct Body<'a> {
    ring: &'a Ring,
    kind: MmaKind,
    qf: &'a [ValueId],
    kr: &'a [ValueId],
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
        self.ring.ks.at(slot).read(b, |b| {
            let rd = |b: &mut FnBuilder, c: usize| -> Vec<ValueId> {
                (0..4).map(|blk| { let a = b.bind_i32(ev(bases[c]) + blk * 32 * ROW); b.load_lds_b128(a) }).collect()
            };
            let mut acc: Vec<Option<ValueId>> = vec![None; 4];
            let mut cur = rd(b, 0);
            for c in 0..8usize {
                let nxt = (c + 1 < 8).then(|| rd(b, c + 1));
                for blk in 0..4usize {
                    match into {
                        Some(v) => b.var_mfma(v[blk], kind, cur[blk], qf[c], c == 0),
                        None => acc[blk] = Some(b.mfma(kind, cur[blk], qf[c], acc[blk].map(Opnd::Val))),
                    }
                }
                if let Some(n) = nxt {
                    cur = n;
                }
            }
            acc.into_iter().flatten().collect()
        })
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
            for t in 0..8usize {
                let nxt = (t + 1 < 8).then(|| rd(b, t + 1));
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
        let so = { let v = b.bind_i32(ev(so0) + key0.clone() * ev(r.seqs)); b.read_first_lane(v) };
        let base = { let o = sh.at(slot).off_v(b); b.bind_i32(o + sh.base as i32) };
        sh.at(slot).write(b, |b| {
            for (i, v) in vo.iter().enumerate() {
                let m0 = { let x = b.bind_i32(ev(base) + ((ev(r.wig) + 4 * i as i32) << 10)); b.read_first_lane(x) };
                b.buffer_load_lds_b128(rs, *v, so, m0, 0);
            }
        });
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
        self.s = vars[6..10].to_vec();
    }
    fn v_phase(&mut self, b: &mut FnBuilder, _role: Role, t: &TileIt) {
        if self.prio {
            b.set_prio(0);
        }
        let (m_v, l_v) = (self.m.expect("m"), self.l.expect("l"));
        let s = self.s.clone();
        // the key mask on tiles past the smallest limit of the CTA (uniform branch, VALU only): masked lanes' S = -inf in place
        let key_t = b.bind_i32(t.key0());
        let partial = b.bind_cond((ev(key_t) + (KT - 1)).gt(ev(self.lim_min)));
        b.begin_if(Opnd::Val(partial));
        {
            let key0 = b.bind_i32(ev(key_t) + (ev(self.half) << 2));
            for blk in 0..4usize {
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
        for blk in 0..4usize {
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
        self.p = (0..8usize)
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
    let mut b = FnBuilder::new(name, 512);
    b.compile_profile(CompileProfile::Gemm);
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
    b.ka_is(288, Ext::c(D as i64));
    b.ka_is(304, Ext::c(D as i64));
    b.ka_is(112, hh.clone() * rowb.clone());
    b.ka_is(144, rowb.clone());
    b.ka_is(160, sq.clone() * hh.clone() * rowb.clone());
    for (s, hs, bs) in [(192u32, 208u32, 224u32), (336, 352, 368)] {
        b.ka_is(s, hk.clone() * rowb.clone());
        b.ka_is(hs, rowb.clone());
        b.ka_is(bs, sk.clone() * hk.clone() * rowb.clone());
    }
    b.ka_is(384, hh.clone() * rowb.clone());
    b.ka_is(400, rowb.clone());
    b.ka_is(416, sq.clone() * hh.clone() * rowb.clone());
    b.ka_is(464, sq.clone() * Ext::c(4));
    b.grid_is(0, if causal { sq.clone().ceil_div(QR as i64).ceil_div(2) } else { sq.clone().ceil_div(QR as i64) });
    b.grid_is(1, hh.clone());
    b.arg_tensor("ptr_q", bb.clone() * sq.clone() * hh.clone() * rowb.clone());
    b.arg_tensor("ptr_o", bb.clone() * sq.clone() * hh.clone() * rowb.clone());
    b.arg_tensor("ptr_k", bb.clone() * sk.clone() * hk.clone() * rowb.clone());
    b.arg_tensor("ptr_v", bb.clone() * sk.clone() * hk * rowb);
    b.arg_tensor("ptr_lse", bb * hh * sq * Ext::c(4));

    let kp = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargPtr { offset: 16 * i });
    let ki = |b: &mut FnBuilder, i: u32| b.let_(Op::KernargI32 { offset: 16 * i });
    let (p_o, p_q, p_k, p_v, p_lse) = (kp(&mut b, 0), kp(&mut b, 1), kp(&mut b, 2), kp(&mut b, 3), kp(&mut b, 4));
    let scale = b.let_(Op::KernargF32 { offset: 80 });
    let (s_q, q_seqs, q_hs, q_bs, gqa) = (ki(&mut b, 6), ki(&mut b, 7), ki(&mut b, 9), ki(&mut b, 10), ki(&mut b, 11));
    let (k_seqs, k_hs, k_bs, lse_on, s_kv) = (ki(&mut b, 12), ki(&mut b, 13), ki(&mut b, 14), ki(&mut b, 16), ki(&mut b, 17));
    let (v_seqs, v_hs, v_bs, o_seqs, o_hs, o_bs, lse_hs) = (ki(&mut b, 21), ki(&mut b, 22), ki(&mut b, 23), ki(&mut b, 24), ki(&mut b, 25), ki(&mut b, 26), ki(&mut b, 29));
    let (qt0, h0, bat0) = (b.let_(Op::WorkgroupIdX), b.let_(Op::WorkgroupIdY), b.let_(Op::WorkgroupIdZ));
    // (XCD order of the (q tile, head) grid measured no gain: not used)
    let (qt, h, bat) = (qt0, h0, bat0);
    let lane = Lane::new(&mut b);
    let roles = Roles::split(&mut b, &lane, 4);
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
    let ks = lds.ring("K", (KT * ROW) as i64, 2, TLayout::Linear);
    let vs = lds.ring("V", (KT * ROW) as i64, 2, TLayout::Linear);
    lds.commit(&mut b);
    // DMA (per group, 4 waves): block bi = w + 4 i (i < 8) of 1 KB = keys 4 bi + lane / 16, chunk lane % 16; K source chunk ^ (key % 16)
    let wig = roles.wave_in_group;
    let l16 = b.bind_i32(lane.lane() & 15);
    let k_so0 = { let v = b.bind_i32(ev(bat) * ev(k_bs) + ev(hkv) * ev(k_hs)); b.read_first_lane(v) };
    let v_so0 = { let v = b.bind_i32(ev(bat) * ev(v_bs) + ev(hkv) * ev(v_hs)); b.read_first_lane(v) };
    let keys: Vec<ValueId> = (0..8).map(|i| b.bind_i32(((ev(wig) + 4 * i) << 2) + lane.lane().lshr(4))).collect();
    let k_vo: Vec<ValueId> = keys.iter().map(|k| b.bind_i32(ev(*k) * ev(k_seqs) + ((ev(l16) ^ (ev(*k) & 15)) << 4))).collect();
    // V swizzled like K (the transposed reads of 4 rows x 4 columns per 16-lane group then hit distinct banks)
    let v_vo: Vec<ValueId> = keys.iter().map(|k| b.bind_i32(ev(*k) * ev(v_seqs) + ((ev(l16) ^ (ev(*k) & 15)) << 4))).collect();
    let ring = Ring { ks, vs, k_vo, v_vo, r_k, r_v, k_so0, v_so0, seqs: k_seqs, wig };
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

    let c_l2e = b.c_f32(std::f32::consts::LOG2_E);
    let csc = b.mul_f32(scale, c_l2e);
    let neg_inf = b.c_f32(f32::NEG_INFINITY);
    let zero16 = b.frag_zero(DType::Frag16);
    let kind = MmaKind::new(MmaShape::M32N32K16, MmaTy::Bf16);
    let big = b.c_i32(0x7fff_fff0);
    let qoff = b.bind_i32(ev(s_kv) - ev(s_q));
    // causal: the vendor's pairing - CTA x takes q tiles x and nt - 1 - x (equal work per CTA; the grid is halved)
    let nt = { let v = b.bind_i32((ev(s_q) + (QR - 1)).lshr(8)); b.read_first_lane(v) };
    let passes: Vec<(ValueId, Option<ValueId>)> = if causal {
        let qb = { let v = b.bind_i32(ev(nt) - 1 - ev(qt)); b.read_first_lane(v) };
        let second = b.bind_cond(ev(qb).gt(ev(qt)));
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
    let row = b.bind_i32((ev(qt) << 8) + (lane.wave() << 5) + lane.j());
    let row_c = b.bind_i32(ev(row).min(ev(s_q) - 1));
    let q_vo = b.bind_i32(ev(row_c) * ev(q_seqs) + (ev(half) << 4));
    let qf: Vec<ValueId> = (0..8).map(|c| b.buffer_load_b128(r_q, q_vo, q_so, (32 * c) as u32)).collect();
    // ---- key range: causal keys k <= q + SK - SQ ----
    let kv_end = if causal {
        let v = b.bind_i32((((ev(qt) + 1) << 8) - 1 + ev(qoff) + 1).min(ev(s_kv)).max(0));
        b.read_first_lane(v)
    } else {
        s_kv
    };
    let nkv = { let v = b.bind_i32((ev(kv_end) + (KT - 1)).lshr(7)); b.read_first_lane(v) };
    let lim = b.bind_i32(if causal { (ev(row) + ev(qoff)).min(ev(s_kv) - 1) } else { ev(s_kv) - 1 });
    let lim_min = {
        let v = b.bind_i32(if causal { ((ev(qt) << 8) + ev(qoff)).min(ev(s_kv) - 1) } else { ev(s_kv) - 1 });
        b.read_first_lane(v)
    };
    let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, nkv, kv_end, KT, Ext::ka(272).ceil_div(KT as i64));
    pipe.key_unclamped = true;

    let any = b.bind_cond(ev(nkv).gt(0));
    b.begin_if(Opnd::Val(any));
    roles.each(&mut b, |b, role| {
        let mut body = Body {
            ring: &ring, kind, qf: &qf, kr: &kr, vr: vr.clone(), csc, neg_inf, zero16, half, lim, lim_min,
            s_init: Vec::new(), m: None, l: None, o: Vec::new(), s: Vec::new(), p: Vec::new(), prio: true,
        };
        pipe.run(b, role, &mut body);
        // ---- epilogue: O = O^T / l (bf16, 8-byte stores), LSE ----
        let (m_v, l_v) = (body.m.expect("m"), body.l.expect("l"));
        let (la, lb) = b.permlane32_swap(Opnd::Var(l_v), Opnd::Var(l_v));
        let (la, lb) = (b.bitcast_f32(la), b.bitcast_f32(lb));
        let l_tot = b.add_f32(la, lb);
        let inv = b.rcp_f32(l_tot);
        let o_so = { let v = b.bind_i32(ev(bat) * ev(o_bs) + ev(h) * ev(o_hs)); b.read_first_lane(v) };
        let live = b.bind_cond(ev(row).lt(ev(s_q)));
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
    b.end_if();
    if guard.is_some() {
        b.end_if();
    }
    }
    b.finish()
}
