//! GEMM epilogues: the MoE epilogue from the accumulators and the LDS-staged bf16 / row-col scaled epilogue.

use super::*;

/// The MoE epilogue straight from the accumulators (lane: row = the c_tv row, 4 consecutive columns): per m block the sorted row's
/// (token, slot) and routing weight; stage 1: act(gate) up of the interleaved block pairs -> bf16 [M topk, n_out] at row token topk + slot;
/// stage 2: fp32 [M topk, n_out] x the weight. Padding rows (token >= M) store past the descriptor (dropped).
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_epilogue(
    b: &mut FnBuilder, ctx: &Ctx, mma: &TiledMma, acc: &[VarId], s_lane: &Ix, mc: &MoeCx, cfg: Cfg, rsa: ValueId, rsb: ValueId, ptr_c: ValueId, m_v: ValueId,
    n_v: ValueId, bx: ValueId, wm: ValueId, wn: ValueId, wave_m: i64, wave_n: i64,
) {
    let d = &ctx.dyns;
    let (nm, nn) = ((wave_m / 16) as usize, (wave_n / 16) as usize);
    let idx = IssuePlan::new(mma.c_tv(), 4).offsets(&make_layout(tup![16, 16], tup![1024, 1]), s_lane, &Ix::C(0), d);
    let o = b.bind_i32(ctx.lower(&idx[0]));
    let (rl, cl) = (b.bind_i32(ev(o).lshr(10)), b.bind_i32(ev(o) & 1023));
    let up = matches!(mc.st, MoeStage::Up { .. });
    let half_out = mc.st == MoeStage::DownH;
    let esz = if up || half_out { 2 } else { 4 };
    let rsrc_c = { let n = b.bind_i32(ev(mc.mtok) * ev(mc.topk) * ev(mc.nout) * esz); let n = b.read_first_lane(n); b.buffer_desc_dyn(ptr_c, n) };
    let so = { let v = b.bind_i32(ev(mc.row0) << 2); b.read_first_lane(v) };
    let big = b.c_i32(0x7fff_fff0);
    let c0 = b.c_i32(0);
    // the first output column of this lane in the wave
    let col0 = if up {
        b.bind_i32((ev(bx) << (cfg.cta_n.trailing_zeros() as i32 - 1)) + ev(wn) * (wave_n as i32 / 2) + ev(cl))
    } else {
        b.bind_i32((ev(bx) << cfg.cta_n.trailing_zeros() as i32) + ev(wn) * (wave_n as i32) + ev(cl))
    };
    let l2e = b.c_f32(std::f32::consts::LOG2_E);
    let one = b.c_f32(1.0);
    // gpt-oss (`moe_bias`): the expert's fp32 bias row(s) (stage 1: [2 n_out] gate then up; stage 2: [n_out]), base moved per expert
    let rsrc_bias = cfg.moe_bias.then(|| {
        let p = b.let_(Op::KernargPtr { offset: 144 });
        let n = { let v = b.bind_i32(ev(mc.nout) * if up { 8 } else { 4 }); b.read_first_lane(v) };
        let pe = b.ptr_mad(p, mc.e, n);
        b.buffer_desc_dyn(pe, n)
    });
    let bias4 = |b: &mut FnBuilder, col: ValueId, plus: Option<ValueId>| -> Vec<ValueId> {
        let rs = rsrc_bias.expect("bias");
        let a = match plus { Some(p) => b.bind_i32((ev(col) + ev(p)) << 2), None => b.bind_i32(ev(col) << 2) };
        let f = b.buffer_load_b128(rs, a, 0i32, 0);
        (0..4u8).map(|k| { let x = b.frag4_lane(f, k); b.bitcast_f32(x) }).collect()
    };
    let mut orows: Vec<ValueId> = Vec::new();
    for i in 0..nm {
        let sr = b.bind_i32(ev(wm) * (wave_m as i32) + 16 * i as i32 + ev(rl));
        let vo = b.bind_i32(ev(sr) << 2);
        let sid = b.buffer_load_f32(mc.rsrc_sid, vo, so, 0);
        let sid = b.bitcast_i32(sid);
        // the routing weight only where it is used (a dead load's destination must not be reused while it is in flight: backend issue)
        let use_w = !matches!(mc.st, MoeStage::Up { doweight: false, .. });
        let sw = if use_w { b.buffer_load_f32(mc.rsrc_sw, vo, so, 0) } else { b.c_f32(1.0) };
        let tok = b.bind_i32(ev(sid) & 0xff_ffff);
        let orow = b.bind_i32(ev(tok) * ev(mc.topk) + ev(sid).lshr(24));
        orows.push(orow);
        let live = b.bind_cond(ev(tok).lt(ev(mc.mtok)));
        // per-token (A row) scale
        let xs = (cfg.scales == Scales::RowCol).then(|| {
            let arow = if up { tok } else { orow };
            let ar = b.bind_i32(ev(arow).min(ev(m_v) - 1) << 2);
            b.buffer_load_f32(rsa, ar, 0i32, 0)
        });
        let wsc = |b: &mut FnBuilder, brow: ValueId| -> Vec<ValueId> {
            let a = b.bind_i32(ev(brow) << 2); // the expert is in the descriptor base (rsrc_sfb)
            let f = b.buffer_load_b128(rsb, a, 0i32, 0);
            (0..4u8).map(|k| { let x = b.frag4_lane(f, k); b.bitcast_f32(x) }).collect()
        };
        let vals = |b: &mut FnBuilder, n: usize| -> Vec<ValueId> {
            (0..4u8).map(|r| { let l = b.frag4_lane(Opnd::Var(acc[n]), r); b.bitcast_f32(l) }).collect()
        };
        if up {
            let MoeStage::Up { gelu, doweight } = mc.st else { unreachable!() };
            // the row / first column part of the store offset (padding rows: past the descriptor); the 16-column block p in the instruction offset
            let vr0 = b.bind_i32((ev(orow) * ev(mc.nout) + ev(col0)) << 1);
            let vrow = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(vr0), b: Opnd::Val(big) });
            for p in 0..nn / 2 {
                let col = b.bind_i32(ev(col0) + 16 * p as i32);
                let mut g = vals(b, nn * i + 2 * p);
                let mut u = vals(b, nn * i + 2 * p + 1);
                if let Some(xs) = xs {
                    // the weight scales follow the interleaved row order: output column c = 16 q + r -> gate row 32 q + r, up row 32 q + 16 + r
                    let grow = b.bind_i32(((ev(col).lshr(4)) << 5) + (ev(col) & 15));
                    let urow = b.bind_i32(ev(grow) + 16);
                    let (wg, wu) = (wsc(b, grow), wsc(b, urow));
                    for k in 0..4 {
                        let sg = b.mul_f32(xs, wg[k]);
                        g[k] = b.mul_f32(g[k], sg);
                        let su = b.mul_f32(xs, wu[k]);
                        u[k] = b.mul_f32(u[k], su);
                    }
                }
                if cfg.moe_bias {
                    // gpt-oss: + bias (gate cols, up cols at + n_out)
                    let bg = bias4(b, col, None);
                    let bu = bias4(b, col, Some(mc.nout));
                    for k in 0..4 {
                        g[k] = b.add_f32(g[k], bg[k]);
                        u[k] = b.add_f32(u[k], bu[k]);
                    }
                }
                let y: Vec<ValueId> = (0..4)
                    .map(|k| {
                        let a = if cfg.moe_bias {
                            // gpt-oss clamped swiglu: gate = min(g, 7), up = clamp(u, -7, 7), (up + 1) gate sigmoid(1.702 gate)
                            let lim = b.c_f32(7.0);
                            let nlim = b.c_f32(-7.0);
                            // min(x, 7) = -max(-x, -7)
                            let zf = b.c_f32(0.0);
                            let ngk = b.sub_f32(zf, g[k]);
                            let gm = b.max_f32(ngk, nlim);
                            let gc = b.sub_f32(zf, gm);
                            let nuk = b.sub_f32(zf, u[k]);
                            let um = b.max_f32(nuk, nlim);
                            let uc = b.sub_f32(zf, um);
                            let uc = b.max_f32(uc, nlim);
                            let _ = lim;
                            let ng = b.sub_f32(zf, gc);
                            let alpha_l2e = b.c_f32(1.702 * std::f32::consts::LOG2_E);
                            let x = b.mul_f32(ng, alpha_l2e);
                            let ex = b.exp2_f32(x);
                            let den = b.add_f32(ex, one);
                            let r = b.rcp_f32(den);
                            let glu = b.mul_f32(gc, r);
                            let up1 = b.add_f32(uc, one);
                            u[k] = up1;
                            glu
                        } else if gelu {
                            // gelu (tanh form): g (1 - 1 / (exp(2 t) + 1)), t = 0.7978845608 (g + 0.044715 g^3)
                            let g2 = b.mul_f32(g[k], g[k]);
                            let c1 = b.c_f32(0.044715);
                            let inner = b.let_(Op::FmaF32(Opnd::Val(g2), Opnd::Val(c1), Opnd::Val(one)));
                            let t0 = b.mul_f32(g[k], inner);
                            let c2 = b.c_f32(2.0 * 0.797_884_56 * std::f32::consts::LOG2_E);
                            let x2 = b.mul_f32(t0, c2);
                            let ex = b.exp2_f32(x2);
                            let den = b.add_f32(ex, one);
                            let r = b.rcp_f32(den);
                            let om = b.sub_f32(one, r);
                            b.mul_f32(g[k], om)
                        } else {
                            // silu: g / (1 + exp(-g))
                            let zf = b.c_f32(0.0);
                            let ng = b.sub_f32(zf, g[k]);
                            let x = b.mul_f32(ng, l2e);
                            let ex = b.exp2_f32(x);
                            let den = b.add_f32(ex, one);
                            let r = b.rcp_f32(den);
                            b.mul_f32(g[k], r)
                        };
                        let yy = b.mul_f32(a, u[k]);
                        if doweight { b.mul_f32(yy, sw) } else { yy }
                    })
                    .collect();
                let w0 = b.cvt_pk_bf16_f32(y[0], y[1]);
                let w1 = b.cvt_pk_bf16_f32(y[2], y[3]);
                let f2 = b.frag2(w0, w1);
                b.buffer_store_b64(rsrc_c, vrow, c0, (32 * p) as u32, f2);
            }
        } else {
            let vr0 = if half_out {
                b.bind_i32((ev(orow) * ev(mc.nout) + ev(col0)) << 1)
            } else {
                b.bind_i32((ev(orow) * ev(mc.nout) + ev(col0)) << 2)
            };
            let vrow = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(live), a: Opnd::Val(vr0), b: Opnd::Val(big) });
            for j in 0..nn {
                let col = b.bind_i32(ev(col0) + 16 * j as i32);
                let mut v = vals(b, nn * i + j);
                let ws = xs.map(|_| wsc(b, col));
                let bs2 = cfg.moe_bias.then(|| bias4(b, col, None));
                for k in 0..4 {
                    if let (Some(xs), Some(ws)) = (xs, &ws) {
                        let s = b.mul_f32(xs, ws[k]);
                        v[k] = b.mul_f32(v[k], s);
                    }
                    if let Some(bs2) = &bs2 {
                        // gpt-oss stage 2: + the expert's output bias (before the routing weight)
                        v[k] = b.add_f32(v[k], bs2[k]);
                    }
                    let y = b.mul_f32(v[k], sw);
                    v[k] = if half_out { y } else { b.bitcast_i32(y) };
                }
                if half_out {
                    let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
                    let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
                    let f2 = b.frag2(w0, w1);
                    b.buffer_store_b64(rsrc_c, vrow, c0, (32 * j) as u32, f2);
                    continue;
                }
                let f4 = b.frag4(v[0], v[1], v[2], v[3]);
                b.buffer_store_b128(rsrc_c, vrow, c0, (64 * j) as u32, f4);
            }
        }
    }
    // data fact (moe_sorting's output, counted): every live sorted row is one (token, slot) pair exactly once, so the output row token topk + slot
    // is an injective function of (M tile = workgroup y, row in the tile); padding rows write nothing (masked stores)
    b.wg_union(&orows, &[1], Ext::c(0), Ext::ka(140) * Ext::ka(128) - Ext::c(1), "data: moe_sorting lists every (token, slot) once (host contract)");
}

/// The LDS-staged epilogue: every accumulator (16 x 16, lane = row m, 4 consecutive n) is converted to bf16 and written to the wave's staging
/// block E[m][n] (row pitch EPI_PITCH); then each lane reads 16 bytes = 8 consecutive n of one row and stores them with one `buffer_store_dwordx4`
/// (16 lanes cover a 256-byte row segment). All addresses from layouts: the C TV layout of the atom over E, a row-contiguous read TV layout over E
/// and the same TV layout over the global C (voffset per lane, the wave-uniform per-issue row step in the SGPR soffset).
#[allow(clippy::too_many_arguments)]
pub(crate) fn lds_epilogue(
    b: &mut FnBuilder, ctx: &Ctx, mma: &TiledMma, epi: &Shared, acc: &[VarId], lane: &Lane, s_lane: &Ix, s_n: &Ix, ptr_c: ValueId, m_v: ValueId,
    n_v: ValueId, bx: ValueId, by: ValueId, wm: ValueId, wn: ValueId, lg_m: i32, lg_n: i32, wave_m: i64, wave_n: i64, phases: usize,
    rowcol: Option<(ValueId, ValueId)>,
    per_tensor: bool,
) {
    let d = &ctx.dyns;
    let (nm, nn) = ((wave_m / 16) as usize, (wave_n / 16) as usize);
    assert!(wave_n % 8 == 0 && 64 % (wave_n / 8) == 0 && wave_n * 2 <= EPI_PITCH, "lds_epilogue: wave rows of 8-column chunks that tile a wave");
    // per-token x per-channel scales: the lane's row of every m block (one f32) and its 4 consecutive columns of every n block (one 16-byte
    // load), from the C TV layout of the atom over a (row 1024 + col) index; loaded before the staging barrier
    let scl: Option<(Vec<ValueId>, Vec<ValueId>)> = rowcol.map(|(rs_a, rs_b)| {
        if per_tensor {
            // one A and one B scale (the whole tensor): the same value for every row / column
            let z = b.c_i32(0);
            let sa = b.buffer_load_f32(rs_a, z, 0i32, 0);
            let sbv = b.buffer_load_f32(rs_b, z, 0i32, 0);
            let sbi = b.bitcast_i32(sbv);
            let sb4 = b.frag4(sbi, sbi, sbi, sbi);
            return ((0..nm).map(|_| sa).collect(), (0..nn).map(|_| sb4).collect());
        }
        let idx = IssuePlan::new(mma.c_tv(), 4).offsets(&make_layout(tup![16, 16], tup![1024, 1]), s_lane, &Ix::C(0), d);
        let o = b.bind_i32(ctx.lower(&idx[0]));
        let row = b.bind_i32((ev(by) << lg_m) + ev(wm) * (wave_m as i32) + ev(o).lshr(10));
        let col = b.bind_i32((ev(bx) << lg_n) + ev(wn) * (wave_n as i32) + (ev(o) & 1023));
        let va = b.bind_i32(ev(row) << 2);
        let vb = b.bind_i32(ev(col) << 2);
        let sa: Vec<ValueId> = (0..nm).map(|i| b.buffer_load_f32(rs_a, va, 0i32, (64 * i) as u32)).collect();
        let sb: Vec<ValueId> = (0..nn).map(|j| b.buffer_load_b128(rs_b, vb, 0i32, (64 * j) as u32)).collect();
        (sa, sb)
    });
    // every wave is done with the ring / panel storage the staging reuses: its reads completed (wait_lds) AND the last loop iteration's prefetch DMAs
    // into the ring have landed (wait_vm: an LDS-direct load still in flight would overwrite the staged C - out-of-range lanes write zeros), then the barrier
    // (GV-TILE-ALIAS-RACE found the missing wait_vm)
    b.wait_vm(0);
    b.wait_lds(0);
    b.barrier();
    let wid = b.bind_i32(lane.wave());
    let me = epi.mine(wid);
    let base = b.bind_i32(me.base());
    // staging layout over the C block element index m + 16 n of one 16 x 16 block: (m, n) -> m EPI_PITCH + 2 n
    let e_blk = make_layout(tup![16, 16], tup![EPI_PITCH, 2]);
    let w_plan = IssuePlan::new(mma.c_tv(), 4); // 4 bf16 = one ds_write_b64
    let mut exprs = Vec::new();
    for i in 0..nm as i64 {
        for j in 0..nn as i64 {
            exprs.extend(w_plan.offsets(&e_blk, s_lane, &Ix::C(16 * i * EPI_PITCH + 32 * j), d));
        }
    }
    let wp = layout_to_offsets(&exprs, &Swizzle::none(), ImmRange::DS, d);
    let wbases: Vec<ValueId> = wp.bases.iter().map(|e| b.bind_i32(ev(base) + ctx.lower(e))).collect();
    // read / store TV over the wave block element index n + wave_n m: lane -> (chunk = lane % 16: 8 n, row = lane / 16), value -> (8 n, row step 4)
    let rows_per = 64 / (wave_n / 8); // 4
    let issues = wave_m / rows_per; // 32
    let tv = make_layout(tup![tup![wave_n / 8, rows_per], tup![8, issues]], tup![tup![8, wave_n], tup![1, wave_n * rows_per]]);
    let e_blk2 = make_layout(tup![wave_n, wave_m], tup![2, EPI_PITCH]);
    let rp = IssuePlan::new(tv.clone(), 8);
    let rexprs = rp.offsets(&e_blk2, s_lane, &Ix::C(0), d);
    let rpl = layout_to_offsets(&rexprs, &Swizzle::none(), ImmRange::DS, d);
    let rbases: Vec<ValueId> = rpl.bases.iter().map(|e| b.bind_i32(ev(base) + ctx.lower(e))).collect();
    // global: C row-major bf16, the wave block at rows m_cta + wave_m wm, cols n_cta + wave_n wn
    let g_blk = make_layout(tup![wave_n, wave_m], tup![2, s_n.mul_c(2)]);
    let gofs = rp.offsets(&g_blk, s_lane, &Ix::C(0), d);
    let voff = b.bind_i32(ctx.lower(&gofs[0]) + (ev(wm) * (wave_m as i32) * ev(n_v) + ev(wn) * (wave_n as i32)) * 2);
    let rsrc_c = {
        let nr = b.bind_i32(ev(m_v) * ev(n_v) * 2);
        let nr = b.read_first_lane(nr);
        b.buffer_desc_dyn(ptr_c, nr)
    };
    let c_org = { let v = b.bind_i32((((ev(by) << lg_m) * ev(n_v)) + (ev(bx) << lg_n)) << 1); b.read_first_lane(v) };
    // phases of wave_m / phases rows: the conversion + LDS writes of phase p+1 are issued before the stores of phase p (which only wait for
    // phase p's reads), so conversion, LDS traffic and global stores overlap (the vendor stages quadrant by quadrant)
    // a phase writes whole 16-row blocks (all nn column blocks) and reads back exactly those rows: at most nm phases, dividing nm
    let mut phases = phases.clamp(1, nm);
    while nm % phases != 0 {
        phases -= 1;
    }
    let accs_per = acc.len() / phases;
    let iss_per = issues as usize / phases;
    assert_eq!(accs_per % nn, 0, "lds_epilogue: a phase covers whole row blocks");
    let write_phase = |b: &mut FnBuilder, p: usize| {
        me.write(b, p as i32, |b| {
            for n in p * accs_per..(p + 1) * accs_per {
                let a = acc[n];
                let mut v: Vec<ValueId> = (0..4u8).map(|r| {
                    let l = b.frag4_lane(Opnd::Var(a), r);
                    b.bitcast_f32(l)
                }).collect();
                if let Some((sa, sb)) = &scl {
                    // acc n = block (n / nn, n % nn): times the row scale and the column scales
                    let (i, j) = (n / nn, n % nn);
                    for (r, x) in v.iter_mut().enumerate() {
                        let cl = b.frag4_lane(Opnd::Val(sb[j]), r as u8);
                        let c = b.bitcast_f32(cl);
                        let s = b.mul_f32(sa[i], c);
                        *x = b.mul_f32(*x, s);
                    }
                }
                // 4 bf16 = one ds_write_b64
                let w0 = b.cvt_pk_bf16_f32(v[0], v[1]);
                let w1 = b.cvt_pk_bf16_f32(v[2], v[3]);
                let f2 = b.frag2(w0, w1);
                let (bi, imm) = wp.issues[n];
                let ad = b.bind_i32(ev(wbases[bi]) + imm as i32);
                b.store_lds_b64(ad, f2);
            }
        });
    };
    write_phase(b, 0);
    for p in 0..phases {
        b.wait_lds(0);
        let vals: Vec<ValueId> = me.read_back(b, p as i32, |b| {
            rpl.issues[p * iss_per..(p + 1) * iss_per].iter().map(|&(bi, imm)| {
                let ad = b.bind_i32(ev(rbases[bi]) + imm as i32);
                b.load_lds_b128(ad)
            }).collect()
        });
        if p + 1 < phases {
            write_phase(b, p + 1);
        }
        for (k, v) in vals.iter().enumerate() {
            let q = p * iss_per + k;
            // the per-issue step is wave-uniform (rows_per rows): checked on the layout expressions
            let dq = gofs[q].sub(&gofs[0]);
            let dv = b.bind_i32(ev(c_org) + ctx.lower(&dq));
            let so = b.read_first_lane(dv);
            b.buffer_store_b128(rsrc_c, voff, so, 0, *v);
        }
    }
}
