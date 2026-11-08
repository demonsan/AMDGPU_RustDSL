//! Row kernels, one row per wave (docs/op-coverage-memtemplates.md T2 / T4): lane `l` holds the elements `8 (l + 64 i) + k` of the row (one b128 per chunk `i`), the row stays in registers
//! (HBM is read once), statistics come from `wave_reduce` (DPP quads + the wave's private LDS scratch). 4 rows per 256-thread block, grid = ceil(M / 4).
//!
//! Names (N a multiple of 8, 8..=8192; chunks past the row end are masked in the statistics and their stores dropped by the buffer descriptor):
//! * `rmsnorm_bf16_n<N>`      `out = x * rsqrt(mean(x^2) + eps) * gamma`.                   ABI x @0, gamma @8, out @16, M @24 (i32), eps @28 (f32)
//! * `add_rmsnorm_bf16_n<N>`  `res_out = bf16(x + res)`, `out = rmsnorm(res_out) * gamma`. ABI x @0, res @8, gamma @16, out @24, res_out @32, M @40, eps @44
//! * `layernorm_bf16_n<N>`    `out = (x - mean) * rsqrt(var + eps) * gamma + beta`.        ABI x @0, gamma @8, beta @16, out @24, M @32, eps @36
//! * `softmax_bf16_n<N>`      `out = softmax(x)` along the row.                            ABI x @0, out @8, M @16

pub mod quant;

use amdgpu_dsl::{CachePolicy, MemTy, CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, ValueId};
use amdgpu_tile::prims::{bf16x8_to_f32, dpp_reduce, group_reduce_dpp, lane_reduce, rsqrt_f32, wave_reduce, wave_reduce_dpp, RedOp};
use amdgpu_tile::{ev, Lane, LdsPlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowOp {
    RmsNorm,
    AddRmsNorm,
    LayerNorm,
    Softmax,
}

pub fn by_name(name: &str) -> Option<Function> {
    let (rest, op) = if let Some(r) = name.strip_prefix("add_rmsnorm_bf16_n") {
        (r, RowOp::AddRmsNorm)
    } else if let Some(r) = name.strip_prefix("rmsnorm_bf16_n") {
        (r, RowOp::RmsNorm)
    } else if let Some(r) = name.strip_prefix("layernorm_bf16_n") {
        (r, RowOp::LayerNorm)
    } else if let Some(r) = name.strip_prefix("softmax_bf16_n") {
        (r, RowOp::Softmax)
    } else {
        return None;
    };
    let n: i64 = rest.parse().ok()?;
    (n % 8 == 0 && (8..=8192).contains(&n)).then(|| row_kernel(name, n, op))
}

/// Wave-wide reduction: the LDS exchange (`wave_reduce`) or, with `AMDC_DPP_REDUCE=1`, the DPP + lane-swap all-reduce.
fn wr(b: &mut FnBuilder, op: RedOp, x: ValueId, scratch: &amdgpu_tile::shared::PrivRef, off: i32, gen: i32, lane: &Lane) -> ValueId {
    if dpp_reduce() {
        wave_reduce_dpp(b, op, x)
    } else {
        wave_reduce(b, op, x, scratch, off, gen, lane)
    }
}

fn sel(b: &mut FnBuilder, cond: ValueId, x: ValueId, y: ValueId) -> ValueId {
    b.let_(Op::Select { ty: DType::F32, cond: Opnd::Val(cond), a: Opnd::Val(x), b: Opnd::Val(y) })
}

fn pack(b: &mut FnBuilder, r: &[ValueId; 8]) -> ValueId {
    let w: Vec<ValueId> = (0..4).map(|k| b.cvt_pk_bf16_f32(r[2 * k], r[2 * k + 1])).collect();
    b.frag4(w[0], w[1], w[2], w[3])
}

/// Rows one wave processes at once: short rows (< 2 KB) need several in flight per wave for the 64 KB / CU of loads that saturate HBM (docs/op-coverage-memtemplates.md 5.1); the rows of a wave are consecutive.
pub fn rows_per_wave(n: i64) -> i64 {
    if n < 512 {
        8.max(64 / group_lanes(n))
    } else if n < 1024 {
        2
    } else {
        1
    }
}

/// Lanes per row: 64 (a whole wave) for rows of 512 or more elements; a shorter row takes the smallest power of two >= its 16-byte chunks (>= 4), so a wave holds `64 / g` rows side by side
/// (reductions over each group of `g` lanes, DPP) instead of leaving most lanes idle.
pub fn group_lanes(n: i64) -> i64 {
    if n >= 512 {
        64
    } else {
        ((n / 8) as u64).next_power_of_two().max(4) as i64
    }
}

pub fn row_kernel(name: &str, n: i64, op: RowOp) -> Function {
    let g = group_lanes(n);
    let par = 64 / g;                                   // rows side by side in a wave
    let nld = ((n / 8 + g - 1) / g) as usize;           // chunks per lane
    let cstride = (16 * g) as i32;                      // bytes between a lane's consecutive chunks
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Default);
    let b = &mut fb;
    // argument layout: pointers in order, then M (i32) and eps (f32)
    let ptrs: &[&str] = match op {
        RowOp::RmsNorm => &["x", "gamma", "out"],
        RowOp::AddRmsNorm => &["x", "res", "gamma", "out", "res_out"],
        RowOp::LayerNorm => &["x", "gamma", "beta", "out"],
        RowOp::Softmax => &["x", "out"],
    };
    let has_eps = op != RowOp::Softmax;
    for (i, p) in ptrs.iter().enumerate() {
        b.arg(p, DType::Ptr, 8 * i as u32, 8);
    }
    let m_off = 8 * ptrs.len() as u32;
    b.arg("m", DType::I32, m_off, 4);
    if has_eps {
        b.arg("eps", DType::F32, m_off + 4, 4);
    }
    let mt = Ext::sym("MT");
    b.sym_range("MT", 1, 1 << 20);
    b.ka_is(m_off, mt.clone() * Ext::c(4 * rows_per_wave(n)));
    b.grid_is(0, mt.clone());
    for p in ptrs {
        let vec = matches!(*p, "gamma" | "beta");
        b.arg_tensor(p, if vec { Ext::c(2 * n) } else { mt.clone() * Ext::c(4 * rows_per_wave(n) * 2 * n) });
    }
    let pv: Vec<ValueId> = (0..ptrs.len()).map(|i| b.let_(Op::KernargPtr { offset: 8 * i as u32 })).collect();
    let m_v = b.let_(Op::KernargI32 { offset: m_off });
    let eps = if has_eps { Some(b.let_(Op::KernargF32 { offset: m_off + 4 })) } else { None };
    let tid = b.let_(Op::WorkitemId);
    let lane = Lane::from_tid(b, tid);
    let bx = b.let_(Op::WorkgroupIdX);
    let mut lds = LdsPlan::new();
    let red = lds.private("RED", 128, 4, &[]);
    lds.commit(b);
    let idx = |nm: &str| ptrs.iter().position(|p| *p == nm).unwrap();
    // descriptors: out-of-range rows read zeros and their stores are dropped (num_records = the tensor bytes)
    let mut dyn_desc = |b: &mut FnBuilder, nm: &str| -> ValueId {
        let v = b.bind_i32(ev(m_v) * (2 * n) as i32);
        let v = b.read_first_lane(v);
        b.buffer_desc_dyn(pv[idx(nm)], v)
    };
    let r_x = dyn_desc(b, "x");
    let r_res = if op == RowOp::AddRmsNorm { Some(dyn_desc(b, "res")) } else { None };
    let r_out = dyn_desc(b, "out");
    let r_rout = if op == RowOp::AddRmsNorm { Some(dyn_desc(b, "res_out")) } else { None };
    let r_g = if matches!(op, RowOp::RmsNorm | RowOp::AddRmsNorm | RowOp::LayerNorm) { Some(b.buffer_desc(pv[idx("gamma")], (2 * n) as u32)) } else { None };
    let r_be = if op == RowOp::LayerNorm { Some(b.buffer_desc(pv[idx("beta")], (2 * n) as u32)) } else { None };
    let rpw = rows_per_wave(n) as usize;                // rows per wave in total
    let steps = rpw / par as usize;                      // sequential steps of `par` rows
    let lr = b.bind_i32(lane.lane() & (g as i32 - 1));
    let grp = b.bind_i32(lane.lane().lshr(g.trailing_zeros() as i32));
    let wid = b.bind_i32(lane.wave());
    let me = red.mine(wid);
    // reduction over a row's lanes: the whole wave (LDS exchange / DPP) or a group of g < 64 lanes (DPP all-reduce)
    let red = |b: &mut FnBuilder, op: RedOp, x: ValueId, off: i32, gen: i32, lane: &Lane| -> ValueId {
        if g == 64 {
            wr(b, op, x, &me, off, gen, lane)
        } else {
            group_reduce_dpp(b, op, x, g)
        }
    };
    let ngrp = nld.div_ceil(4);
    let row0 = b.bind_i32(((ev(bx) << 2) + lane.wave()) * rpw as i32);
    let vx0 = b.bind_i32((ev(row0) + ev(grp)) * (2 * n) as i32 + (ev(lr) << 4));
    let vg = b.bind_i32(ev(lr) << 4);
    // per row r: the group base offsets (one voffset per 4 loads, imm 1024 (i % 4))
    let gx_of = |b: &mut FnBuilder, r: usize| -> Vec<ValueId> {
        let vr = if r == 0 { vx0 } else { b.bind_i32(ev(vx0) + (r as i64 * par * 2 * n) as i32) };
        (0..ngrp).map(|q| if q == 0 { vr } else { b.bind_i32(ev(vr) + 4 * cstride * q as i32) }).collect()
    };
    let gg: Vec<ValueId> = (0..ngrp).map(|q| if q == 0 { vg } else { b.bind_i32(ev(vg) + 4 * cstride * q as i32) }).collect();
    let load = |b: &mut FnBuilder, rs: ValueId, g: &[ValueId], c: CachePolicy| -> Vec<ValueId> { (0..nld).map(|i| b.buffer_load_c(rs, Some(Opnd::Val(g[i / 4])), 0i32, (cstride as usize * (i % 4)) as u32, MemTy::B128, c)).collect() };
    let gxs: Vec<Vec<ValueId>> = (0..steps).map(|r| gx_of(b, r)).collect();
    // every row's loads are issued before any compute: rpw * (row bytes) in flight per wave
    let xs_all: Vec<Vec<ValueId>> = gxs.iter().map(|g| load(b, r_x, g, CachePolicy::NT)).collect();
    let rs_all: Option<Vec<Vec<ValueId>>> = r_res.map(|r| gxs.iter().map(|g| load(b, r, g, CachePolicy::NT)).collect());
    let gs = r_g.map(|r| load(b, r, &gg, CachePolicy::NONE));
    let bs = r_be.map(|r| load(b, r, &gg, CachePolicy::NONE));
    b.wait_vm(0);
    // chunk validity: only the last chunk can reach past the row (n % 512 != 0)
    let valid: Vec<Option<ValueId>> = (0..nld)
        .map(|i| {
            if (i as i64 + 1) * 8 * g <= n {
                None
            } else {
                let e = b.bind_i32((ev(lr) << 3) + (8 * g) as i32 * i as i32);
                Some(b.let_(Op::CmpLtI32(Opnd::Val(e), Opnd::I32(n as i32))))
            }
        })
        .collect();
    let zero = b.c_f32(0.0);
    let inv_n = b.c_f32(1.0 / n as f32);
    let masked = |b: &mut FnBuilder, v: &[ValueId; 8], i: usize, fill: ValueId| -> [ValueId; 8] {
        match valid[i] {
            None => *v,
            Some(c) => {
                let mut o = [0; 8];
                for k in 0..8 {
                    o[k] = sel(b, c, v[k], fill);
                }
                o
            }
        }
    };
    for r in 0..steps {
        let (xs, rs) = (&xs_all[r], rs_all.as_ref().map(|a| &a[r]));
        let (gen0, gen1) = (2 * r as i32, 2 * r as i32 + 1);
        // the row stays PACKED (bf16, 4 dwords per chunk) between the passes and is unpacked where it is used: fewer live VGPRs (occupancy) for a few extra shifts
        let mut packed: Vec<ValueId> = Vec::with_capacity(nld);
        for i in 0..nld {
            let v = if let Some(rv) = rs {
                let a = bf16x8_to_f32(b, xs[i]);
                let c = bf16x8_to_f32(b, rv[i]);
                let mut sm = [0; 8];
                for k in 0..8 {
                    sm[k] = b.add_f32(a[k], c[k]);
                }
                // res_out = bf16(x + res); the normalised value is that rounded sum
                pack(b, &sm)
            } else {
                xs[i]
            };
            packed.push(v);
        }
        let rout_words: Vec<ValueId> = if rs.is_some() { packed.clone() } else { Vec::new() };
        let mut out: Vec<[ValueId; 8]> = Vec::with_capacity(nld);
        match op {
            RowOp::RmsNorm | RowOp::AddRmsNorm => {
                let mut part: Option<ValueId> = None;
                for i in 0..nld {
                    let xi = bf16x8_to_f32(b, packed[i]);
                    let v = masked(b, &xi, i, zero);
                    let sq: Vec<ValueId> = v.iter().map(|&x| b.mul_f32(x, x)).collect();
                    let s = lane_reduce(b, RedOp::Sum, &sq);
                    part = Some(match part { None => s, Some(p) => b.add_f32(p, s) });
                }
                let tot = red(b, RedOp::Sum, part.unwrap(), 0, gen0, &lane);
                let mean = b.mul_f32(tot, inv_n);
                let me_eps = b.add_f32(mean, eps.unwrap());
                let inv = rsqrt_f32(b, me_eps);
                for i in 0..nld {
                    let xi = bf16x8_to_f32(b, packed[i]);
                    let g = bf16x8_to_f32(b, gs.as_ref().unwrap()[i]);
                    let mut y = [0; 8];
                    for k in 0..8 {
                        let t = b.mul_f32(xi[k], inv);
                        y[k] = b.mul_f32(t, g[k]);
                    }
                    out.push(y);
                }
            }
            RowOp::LayerNorm => {
                let mut part: Option<ValueId> = None;
                for i in 0..nld {
                    let xi = bf16x8_to_f32(b, packed[i]);
                    let v = masked(b, &xi, i, zero);
                    let s = lane_reduce(b, RedOp::Sum, &v);
                    part = Some(match part { None => s, Some(p) => b.add_f32(p, s) });
                }
                let tot = red(b, RedOp::Sum, part.unwrap(), 0, gen0, &lane);
                let mean = b.mul_f32(tot, inv_n);
                let mut part2: Option<ValueId> = None;
                for i in 0..nld {
                    let xi = bf16x8_to_f32(b, packed[i]);
                    let mut dv = [0; 8];
                    for k in 0..8 {
                        dv[k] = b.sub_f32(xi[k], mean);
                    }
                    let dv = masked(b, &dv, i, zero);
                    let sq: Vec<ValueId> = dv.iter().map(|&x| b.mul_f32(x, x)).collect();
                    let s = lane_reduce(b, RedOp::Sum, &sq);
                    part2 = Some(match part2 { None => s, Some(p) => b.add_f32(p, s) });
                }
                let tot2 = red(b, RedOp::Sum, part2.unwrap(), 64, gen1, &lane);
                let var = b.mul_f32(tot2, inv_n);
                let ve = b.add_f32(var, eps.unwrap());
                let inv = rsqrt_f32(b, ve);
                for i in 0..nld {
                    let xi = bf16x8_to_f32(b, packed[i]);
                    let g = bf16x8_to_f32(b, gs.as_ref().unwrap()[i]);
                    let be = bf16x8_to_f32(b, bs.as_ref().unwrap()[i]);
                    let mut y = [0; 8];
                    for k in 0..8 {
                        // (x - mean) is recomputed here: the deviations are not kept across the second reduction (register pressure)
                        let d = b.sub_f32(xi[k], mean);
                        let t = b.mul_f32(d, inv);
                        let u = b.mul_f32(t, g[k]);
                        y[k] = b.add_f32(u, be[k]);
                    }
                    out.push(y);
                }
            }
            RowOp::Softmax => {
                let neg = b.c_f32(-3.0e38);
                let mut part: Option<ValueId> = None;
                for i in 0..nld {
                    let xi = bf16x8_to_f32(b, packed[i]);
                    let v = masked(b, &xi, i, neg);
                    let s = lane_reduce(b, RedOp::Max, &v);
                    part = Some(match part { None => s, Some(p) => b.max_f32(p, s) });
                }
                let mx = red(b, RedOp::Max, part.unwrap(), 0, gen0, &lane);
                let l2e = b.c_f32(std::f32::consts::LOG2_E);
                let mut ex: Vec<[ValueId; 8]> = Vec::with_capacity(nld);
                let mut part2: Option<ValueId> = None;
                for i in 0..nld {
                    let xi = bf16x8_to_f32(b, packed[i]);
                    let mut e = [0; 8];
                    for k in 0..8 {
                        let t = b.sub_f32(xi[k], mx);
                        let u = b.mul_f32(t, l2e);
                        e[k] = b.exp2_f32(u);
                    }
                    let e = masked(b, &e, i, zero);
                    let s = lane_reduce(b, RedOp::Sum, &e);
                    part2 = Some(match part2 { None => s, Some(p) => b.add_f32(p, s) });
                    ex.push(e);
                }
                let tot = red(b, RedOp::Sum, part2.unwrap(), 64, gen1, &lane);
                let inv = b.rcp_f32(tot);
                for i in 0..nld {
                    let mut y = [0; 8];
                    for k in 0..8 {
                        y[k] = b.mul_f32(ex[i][k], inv);
                    }
                    out.push(y);
                }
            }
        }
        // stores: lanes past the row end store to an offset outside every descriptor (dropped by the hardware)
        for i in 0..nld {
            let f4 = pack(b, &out[i]);
            let base = b.bind_i32(ev(gxs[r][i / 4]) + (cstride as usize * (i % 4)) as i32);
            let voff = match valid[i] {
                None => base,
                Some(c) => b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::Val(base), b: Opnd::I32(-16) }),
            };
            b.buffer_store_c(r_out, Some(Opnd::Val(voff)), 0i32, 0, f4, MemTy::B128, false, CachePolicy::NT);
            if let Some(ro) = r_rout {
                b.buffer_store_c(ro, Some(Opnd::Val(voff)), 0i32, 0, rout_words[i], MemTy::B128, false, CachePolicy::NT);
            }
        }
    }
    fb.finish()
}