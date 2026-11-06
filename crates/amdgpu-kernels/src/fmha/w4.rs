//! EXPERIMENT (docs/fmha-4wave-design.md, geometry B): the 4-wave FMHA steady loop — one wave per SIMD owns 64 query rows as two 32-row groups
//! g0 / g1 offset by half a tile inside the wave (the two-group ping-pong of `_vs128` moved into one wave), O accumulators pinned to AGPRs,
//! a 3-slot K / V ring (K staged two tiles ahead, V one), one barrier per tile. Timing prototype: no key masking (SK a multiple of 128) and no
//! query tail (S a multiple of 256); long-sequence recipe only (vendor reference + lazy update).
//!
//! Per tile t (both groups, one wave):
//!   stage K(t+2), V(t+1)
//!   step 1: g1.PV(t-1) + g1.QK(t)   ||  g0 softmax(t)
//!   step 2: g0.PV(t)   + g0.QK(t+1) ||  g1 softmax(t)
//!   wait copies, barrier
//!
//! Variants (name suffixes, combinable, for the scheduler work): `_a` S of both groups in AGPRs (written by the QK MFMAs) with one VGPR working copy
//! per softmax; `_i` the QK blocks of the MFMA-phase group authored between the conversion chunks of the other group; `_l` late rollback: the row
//! max, the conversion with the current reference and the other group's MFMAs form ONE straight-line region per step, the rare rollback branch
//! (rebase S / O / l + re-convert) sits at the end of the step instead of between the row max and the conversion.

use super::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct W4Opts {
    pub s_agpr: bool,
    pub inter: bool,
    pub late: bool,
}

impl W4Opts {
    pub fn from_name(name: &str) -> W4Opts {
        let suf = name.strip_prefix("fmha_fwd_hd128_mxfp8_v4w_proto").unwrap_or("");
        let has = |s: &str| suf.split('_').any(|x| x == s);
        W4Opts { s_agpr: has("a"), inter: has("i"), late: has("l") }
    }
}

pub fn fmha_fwd_hd128_mxfp8_v4w_proto() -> Function {
    v4w_named("fmha_fwd_hd128_mxfp8_v4w_proto")
}

/// QK of `q` (into its S vars), each 32-key block followed by one conversion chunk of `c`'s softmax when `inter`, else QK then all chunks.
#[allow(clippy::too_many_arguments)]
fn qk_and_convert(b: &mut FnBuilder, q: &AttnBody3, slot: &SlotIdx, bias: Opnd, c: &AttnBody3, sv: &AccTile<C32x32F32, 4>, inter: bool) -> Vec<ValueId> {
    let s = q.s.expect("S");
    let mut w = Vec::new();
    if !inter {
        q.qk(b, slot, QkOut::Vars(&s), bias);
        for i in 0..4usize {
            w.extend(c.sm.convert(b, sv, None, i, i + 1));
        }
        return w;
    }
    let r = q.ring;
    let k = r.kpart.at(b, &r.ks.at(slot));
    let kd = r.kdpart.at(b, &r.kd.at(slot));
    if q.biasonce {
        b.var_mfma_f16_z(s.accs[3].var, q.sm.ones_f16, bias);
        q.bias_tile.set(Some(Opnd::Var(s.accs[3].var)));
    }
    for i in 0..4usize {
        q.qk_block(b, &QkOut::Vars(&s), bias, &k, &kd, i);
        w.extend(c.sm.convert(b, sv, None, i, i + 1));
    }
    q.bias_tile.set(None);
    w
}

/// One step: the other group's PV + QK (MFMA phase of `m`) with the softmax of `c`; the P words of `c` end in `pv` (loop variables).
#[allow(clippy::too_many_arguments)]
fn step(
    b: &mut FnBuilder,
    o: W4Opts,
    m: &AttnBody3,
    pv_slot: &SlotIdx,
    pv_words: &[ValueId],
    qk_slot: &SlotIdx,
    qk_bias: Opnd,
    c: &AttnBody3,
    sv: &AccTile<C32x32F32, 4>,
    pvars: &[VarId],
) {
    m.pv(b, pv_slot, pv_words);
    let cs = c.s.expect("S");
    let co = c.o.expect("O");
    // the softmax operand: S itself (VGPR) or its working copy (S in AGPR)
    let s = if o.s_agpr {
        for (d, x) in sv.accs.iter().zip(&cs.accs) {
            b.set_var(d.var, Opnd::Var(x.var));
        }
        *sv
    } else {
        cs
    };
    if o.late {
        let (mt, grow) = c.sm.lazy_check(b, &s, None);
        let w = qk_and_convert(b, m, qk_slot, qk_bias, c, &s, o.inter);
        for (v, x) in pvars.iter().zip(&w) {
            b.set_var(*v, *x);
        }
        b.begin_if(Opnd::Val(grow));
        c.sm.lazy_rebase(b, mt, grow, &s, &co);
        let w2: Vec<ValueId> = (0..4usize).flat_map(|i| c.sm.convert(b, &s, None, i, i + 1)).collect();
        for (v, x) in pvars.iter().zip(&w2) {
            b.set_var(*v, *x);
        }
        b.end_if();
    } else {
        c.sm.decide_lazy(b, &s, &co, None);
        let w = qk_and_convert(b, m, qk_slot, qk_bias, c, &s, o.inter);
        for (v, x) in pvars.iter().zip(&w) {
            b.set_var(*v, *x);
        }
    }
}

fn read_words(b: &mut FnBuilder, vs: &[VarId]) -> Vec<ValueId> {
    vs.iter().map(|&v| b.let_(amdgpu_dsl::Op::AddI32(Opnd::Var(v), Opnd::I32(0)))).collect()
}

pub fn v4w_named(name: &str) -> Function {
    let opts = W4Opts::from_name(name);
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Vs128);
    let b = &mut fb;
    let io = declare_io(b, 256, false);
    let blk0 = block3_g(&io, b, 8, Some(0));
    let blk1 = block3_g(&io, b, 8, Some(1));
    let roles = Roles::single(&blk0.lane, 4);
    let mut ring = KvRing3::new_with(b, &io, &blk0, &roles, 3, true);
    ring.enable_buffer_dma(b);
    ring.enable_buffer_kd(b);
    let ring = ring;
    let wave = blk0.lane.wave;
    let wid0 = b.bind_i32(ev(wave) << 1);
    let wid1 = b.bind_i32((ev(wave) << 1) + 1);
    let mut g0 = AttnBody3::new(b, &ring, &blk0, wid0, false);
    let mut g1 = AttnBody3::new(b, &ring, &blk1, wid1, false);
    for g in [&mut g0, &mut g1] {
        g.kscale_sel = true;
        let p = amdgpu_tile::rowsum::plan();
        g.sm.use_rows2(b, &g.blk.lane, p.dwords, p.idx);
        g.sm.set_vendor_ref(b);
        g.lazy_bias = true;
        g.biasonce = true;
    }
    let nkv = blk0.nkv;
    let key = |b: &mut FnBuilder, t: I32| -> I32 {
        let n = ev(nkv);
        b.let_i32(t.lt(&n).select_i32(t.clone(), n - 1) << 7)
    };
    // ---- prologue: K0, K1, V0 (and V0 again into slot 2: the g1 PV of "tile -1" multiplies it by P = 0) ----
    let k0 = key(b, I32::c(0));
    ring.stage(b, Role::A, &k0, &SlotIdx::new(0, 0));
    ring.stage(b, Role::B, &k0, &SlotIdx::new(0, 0));
    let k1 = key(b, I32::c(1));
    ring.stage(b, Role::A, &k1, &SlotIdx::new(1, 1));
    ring.stage(b, Role::B, &k0, &SlotIdx::new(2, -1));
    b.wait_vm(0);
    ring.store_constants(b, wid0, &blk0.lane, &blk0.q);
    ring.store_constants(b, wid1, &blk1.lane, &blk1.q);
    b.wait_lds(0);
    b.barrier();
    // g0: S(0) now; g1 computes its S(0) in step 1 of tile 0
    let t0 = TileIt { it: Opnd::I32(0), cur: SlotIdx::new(0, 0), next: SlotIdx::new(1, 1), n2: None, masked: false, staged: true, tile_rows: 128 };
    g0.prologue_compute(b, Role::All, &t0);
    g1.s_init = vec![g1.zero16; 4];
    b.wait_lds(0);
    // ---- loop state: it, slot(t), g0 state, g1 state, P words of g0 (16) and g1 (16), [the S working copy] ----
    let zc = b.c_i32(0);
    let mut inits: Vec<(DType, Opnd)> = vec![(DType::I32, Opnd::Val(zc)), (DType::I32, Opnd::Val(zc))];
    let s0 = g0.loop_state(b);
    let s1 = g1.loop_state(b);
    let (n0, n1) = (s0.len(), s1.len());
    inits.extend(s0);
    inits.extend(s1);
    for _ in 0..32 {
        inits.push((DType::I32, Opnd::Val(zc)));
    }
    if opts.s_agpr {
        for _ in 0..4 {
            inits.push((DType::Frag16, Opnd::Val(g0.zero16)));
        }
    }
    let vars = b.begin_loop(&inits);
    let (it, idx) = (vars[0], vars[1]);
    g0.bind_state(&vars[2..2 + n0]);
    g1.bind_state(&vars[2 + n0..2 + n0 + n1]);
    let base = 2 + n0 + n1;
    let p0v: Vec<VarId> = vars[base..base + 16].to_vec();
    let p1v: Vec<VarId> = vars[base + 16..base + 32].to_vec();
    let sv: AccTile<C32x32F32, 4> = if opts.s_agpr { AccTile::from_vars(&vars[base + 32..base + 36]) } else { g0.s.expect("S") };
    for g in [&g0, &g1] {
        for a in &g.o.expect("O").accs {
            b.pin_agpr(a.var);
        }
        if opts.s_agpr {
            for a in &g.s.expect("S").accs {
                b.pin_agpr(a.var);
            }
        }
    }
    b.assume_trusted(it, Ext::c(0), Ext::ka(0x110).ceil_div(128), "tile counter: 0 <= it <= nkv - 1");
    // slots: t -> idx, t+1 -> i1, t+2 (= t-1) -> i2
    let i1 = b.bind_i32(varx(idx).lt(2).select_i32(varx(idx) + 1, 0));
    let i2 = b.bind_i32(I32::c(0).lt(varx(idx)).select_i32(varx(idx) - 1, 2));
    let it_p1 = b.bind_i32(varx(it) + 1);
    let it_p2 = b.bind_i32(varx(it) + 2);
    let it_m1 = b.bind_i32(varx(it) - 1);
    let sl_t = SlotIdx::new(idx, it);
    let sl_t1 = SlotIdx::new(i1, it_p1);
    let sl_t2 = SlotIdx::new(i2, it_p2);
    let sl_tm1 = SlotIdx::new(i2, it_m1);
    // copies: K(t+2) into the slot of K(t-1), V(t+1) into the slot of V(t-2) (= slot of t+1)
    let kk = key(b, ev(it_p2));
    ring.stage(b, Role::A, &kk, &sl_t2);
    let kv = key(b, ev(it_p1));
    ring.stage(b, Role::B, &kv, &sl_t1);
    b.set_prio(1);
    // the references: loop variables (updated lazily by the group's softmax; bias_init at tile 0)
    let (bias0, bias1) = (Opnd::Var(vars[2]), Opnd::Var(vars[2 + n0]));
    // ---- step 1: g1.PV(t-1) + g1.QK(t) || g0 softmax(t) ----
    let w1 = read_words(b, &p1v);
    let sv0 = if opts.s_agpr { sv } else { g0.s.expect("S") };
    step(b, opts, &g1, &sl_tm1, &w1, &sl_t, bias1, &g0, &sv0, &p0v);
    // ---- step 2: g0.PV(t) + g0.QK(t+1) || g1 softmax(t) ----
    let w0 = read_words(b, &p0v);
    let sv1 = if opts.s_agpr { sv } else { g1.s.expect("S") };
    step(b, opts, &g0, &sl_t, &w0, &sl_t1, bias0, &g1, &sv1, &p1v);
    b.wait_vm(0);
    b.wait_lds(0);
    b.barrier();
    b.set_var(idx, i1);
    b.set_var(it, it_p1);
    let stop = b.bind_cond(!ev(it_p1).lt(ev(nkv)));
    b.break_if(stop);
    b.end_loop();
    // ---- tail: g1.PV(last) ----
    let it_l = b.bind_i32(ev(nkv) - 1);
    let w1 = read_words(b, &p1v);
    let idx_l = b.bind_i32(varx(idx).lt(1).select_i32(2, varx(idx) - 1));
    g1.pv(b, &SlotIdx::new(idx_l, it_l), &w1);
    for (g, blk) in [(&g0, &blk0), (&g1, &blk1)] {
        let fin = g.out_scale(b);
        store_out3(&io, b, blk, &g.o.expect("O"), fin);
    }
    fb.finish()
}
