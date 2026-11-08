//! RMSNorm + per-token fp8 quantization, built from the generic stage primitives (docs/fusion-stage-library-design.md §4.4, P2):
//! `y = x * rsqrt(mean(x^2) + eps) * gamma`, `scale[m] = max|y[m]| / 448`, `q = fp8_e4m3(y / scale)`.
//!
//! Shapes: x [M][N] bf16, gamma [N] bf16, q [M][N] fp8 (e4m3), scale [M] f32; N a compile-time multiple of 512 (the stage instantiation), M at
//! run time. One wave per row (4 rows per 256-thread block): lane l holds elements 8 (l + 64 i) + k (one 16-byte load per i).
//! ABI: x @0, gamma @8, q @16, scale @24, M @32 (i32), eps @36 (f32). Grid x = ceil(M / 4).
//!
//! Stages (each a `Node::Stage` region with its declared hand-overs, checked by GV-STAGE):
//! `load` (M: x and gamma rows -> f32) -> `rms` (R sum of squares over the wave + E scale by rsqrt and gamma) -> `quant` (R absolute maximum +
//! Q per-row fp8) -> `store` (M: q and the row scale).

use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Op, StageContract, StageScalar, StageVal, ValueId};
use amdgpu_tile::prims::{abs_f32, bf16x8_to_f32, fp8_pack, fp8_scale, lane_reduce, rsqrt_f32, wave_reduce, RedOp};
use amdgpu_tile::stage::{Prim, StageInst};
use amdgpu_tile::weave::Units;
use amdgpu_tile::{ev, Lane, LdsPlan};

pub mod graph;
mod skeleton_rmsnorm;

pub fn by_name(name: &str) -> Option<Function> {
    let n: i64 = name.strip_prefix("rmsnorm_quant_fp8_n")?.parse().ok()?;
    (n % 512 == 0 && (512..=8192).contains(&n)).then(|| rmsnorm_quant_fp8(name, n))
}

fn sval(name: &str, values: Vec<ValueId>, bits: u32, table: Vec<i64>) -> StageVal {
    StageVal { name: name.into(), values, table, elem_bits: bits, storage: "vgpr".into(), layout: String::new() }
}

pub fn rmsnorm_quant_fp8(name: &str, n: i64) -> Function {
    skeleton_rmsnorm::build_rmsnorm(name, &mut NormSlots { n })
}

/// Values shared between the slots of the RMSNorm + quant kernel (what the generated skeleton's `Cx` carries).
pub(crate) struct NormCx {
    n: i64,
    nld: usize,
    nv: usize,
    lane: Lane,
    row: ValueId,
    wid: ValueId,
    red: amdgpu_tile::Shared,
    roles: Vec<String>,
    eps: ValueId,
    r_x: ValueId,
    r_g: ValueId,
    r_q: ValueId,
    r_s: ValueId,
    xs: Vec<ValueId>,
    gs: Vec<ValueId>,
    inv: Option<ValueId>,
    qw: Vec<ValueId>,
    sc: Option<ValueId>,
    /// packed element table: fragment i of lane t holds elements 8 (t + 64 i) .. + 7
    ptbl: Vec<i64>,
    /// q words: word w of lane t holds elements 8 (t + 64 (w / 2)) + 4 (w % 2) + 0..3 (fp8: 4 per dword), recorded by the first element
    qt: Vec<i64>,
}

/// The slots of the RMSNorm + quant kernel (`skeleton_rmsnorm.rs`, generated from `graph::graph()`); `n` is the compile-time row length.
pub(crate) struct NormSlots {
    n: i64,
}

impl skeleton_rmsnorm::RmsnormSlots for NormSlots {
    type Cx = NormCx;
    fn prelude(&mut self, b: &mut FnBuilder) -> NormCx {
        let n = self.n;
        let nld = (n / 512) as usize; // 16-byte loads per lane
        let nv = 8 * nld; // values per lane
        b.compile_profile(CompileProfile::Gemm);
        b.arg("x", DType::Ptr, 0, 8);
        b.arg("gamma", DType::Ptr, 8, 8);
        b.arg("q", DType::Ptr, 16, 8);
        b.arg("scale", DType::Ptr, 24, 8);
        b.arg("m", DType::I32, 32, 4);
        b.arg("eps", DType::F32, 36, 4);
        let mt = Ext::sym("MT");
        b.sym_range("MT", 1, 1 << 20);
        b.ka_is(32, mt.clone() * Ext::c(4));
        b.grid_is(0, mt.clone());
        b.arg_tensor("x", mt.clone() * Ext::c(4 * 2 * n));
        b.arg_tensor("gamma", Ext::c(2 * n));
        b.arg_tensor("q", mt.clone() * Ext::c(4 * n));
        b.arg_tensor("scale", mt * Ext::c(16));
        let (p_x, p_g, p_q, p_s) = (b.let_(Op::KernargPtr { offset: 0 }), b.let_(Op::KernargPtr { offset: 8 }), b.let_(Op::KernargPtr { offset: 16 }), b.let_(Op::KernargPtr { offset: 24 }));
        let m_v = b.let_(Op::KernargI32 { offset: 32 });
        let eps = b.let_(Op::KernargF32 { offset: 36 });
        let tid = b.let_(Op::WorkitemId);
        let lane = Lane::from_tid(b, tid);
        let bx = b.let_(Op::WorkgroupIdX);
        let mut lds = LdsPlan::new();
        let red = lds.private("RED", 128, 4, &[]);
        lds.commit(b);
        // descriptors: out-of-range rows read zeros and their stores are dropped (num_records = the tensor bytes)
        let r_x = { let v = b.bind_i32(ev(m_v) * (2 * n) as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_x, v) };
        let r_g = b.buffer_desc(p_g, (2 * n) as u32);
        let r_q = { let v = b.bind_i32(ev(m_v) * n as i32); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_q, v) };
        let r_s = { let v = b.bind_i32(ev(m_v) * 4); let v = b.read_first_lane(v); b.buffer_desc_dyn(p_s, v) };
        let row = b.bind_i32((ev(bx) << 2) + lane.wave());
        let wid = b.bind_i32(lane.wave());
        let roles = vec!["all 4 waves: one row each".to_string()];
        let ptbl: Vec<i64> = (0..64 * nld).map(|j| 8 * ((j % 64) as i64 + 64 * (j / 64) as i64)).collect();
        let qt: Vec<i64> = (0..64 * 2 * nld).map(|i| { let (t, w) = ((i % 64) as i64, (i / 64) as i64); 8 * (t + 64 * (w / 2)) + 4 * (w % 2) }).collect();
        NormCx { n, nld, nv, lane, row, wid, red, roles, eps, r_x, r_g, r_q, r_s, xs: Vec::new(), gs: Vec::new(), inv: None, qw: Vec::new(), sc: None, ptbl, qt }
    }

    // ---- stage `load` (M): the x row and gamma, kept packed (bf16, 8 per 16-byte fragment) ----
    fn load(&mut self, b: &mut FnBuilder, cx: &mut NormCx) {
        let (n, nld, row, r_x, r_g, lane) = (cx.n, cx.nld, cx.row, cx.r_x, cx.r_g, cx.lane);
        let mut u = Units::new();
        let (_, (xs, gs)) = u.capture(b, "load.rows", |b| {
            let vx = b.bind_i32(ev(row) * (2 * n) as i32 + (lane.lane() << 4));
            let vg = b.bind_i32(lane.lane() << 4);
            // the instruction offset reaches 4095 bytes: one voffset per group of 4 loads, imm 1024 (i % 4)
            let gx: Vec<ValueId> = (0..nld.div_ceil(4)).map(|q| if q == 0 { vx } else { b.bind_i32(ev(vx) + 4096 * q as i32) }).collect();
            let gg: Vec<ValueId> = (0..nld.div_ceil(4)).map(|q| if q == 0 { vg } else { b.bind_i32(ev(vg) + 4096 * q as i32) }).collect();
            let xs: Vec<ValueId> = (0..nld).map(|i| b.buffer_load_b128(r_x, gx[i / 4], 0i32, (1024 * (i % 4)) as u32)).collect();
            let gs: Vec<ValueId> = (0..nld).map(|i| b.buffer_load_b128(r_g, gg[i / 4], 0i32, (1024 * (i % 4)) as u32)).collect();
            b.wait_vm(0);
            (xs, gs)
        });
        let (x2, g2, pt1) = (xs.clone(), gs.clone(), cx.ptbl.clone());
        StageInst::new("move.row_load", Prim::M, "load", StageContract { roles: cx.roles.clone(), ..Default::default() }, u)
            .param("n", n)
            .param("dtype", "bf16")
            .handover(move |_b| (vec![sval("x", x2, 128, pt1.clone()), sval("gamma", g2, 128, pt1)], Vec::new()))
            .emit(b);
        (cx.xs, cx.gs) = (xs, gs);
    }

    // ---- stage `rms` (R): inv_rms = rsqrt(mean(x^2) + eps) ----
    fn rms(&mut self, b: &mut FnBuilder, cx: &mut NormCx) {
        let (n, eps, lane) = (cx.n, cx.eps, cx.lane);
        let me = cx.red.mine(cx.wid);
        let mut u = Units::new();
        let (_, inv) = u.capture(b, "rms.reduce", |b| {
            // streamed per fragment (8 squares, then a running sum): the unpacked row never lives in registers at once
            let mut part: Option<ValueId> = None;
            for &f in &cx.xs {
                let sq: Vec<ValueId> = bf16x8_to_f32(b, f).iter().map(|&x| b.mul_f32(x, x)).collect();
                let s = lane_reduce(b, RedOp::Sum, &sq);
                part = Some(match part { None => s, Some(p) => b.add_f32(p, s) });
            }
            let tot = wave_reduce(b, RedOp::Sum, part.expect("values"), &me, 0, 0, &lane);
            let inv_n = b.c_f32(1.0 / n as f32);
            let mean = b.mul_f32(tot, inv_n);
            let me_eps = b.add_f32(mean, eps);
            rsqrt_f32(b, me_eps)
        });
        let c_rms = StageContract { live_in: vec![sval("x", cx.xs.clone(), 128, cx.ptbl.clone())], roles: cx.roles.clone(), ..Default::default() };
        StageInst::new("reduce.row_sumsq", Prim::R, "rms", c_rms, u)
            .param("op", "sum x^2 / n")
            .handover(move |_b| (Vec::new(), vec![StageScalar { name: "inv_rms".into(), values: vec![inv], owner: "wave".into() }]))
            .emit(b);
        cx.inv = Some(inv);
    }

    // ---- stage `quant` (E + R + Q): y = x inv_rms gamma (recomputed per pass), amax over the row, per-row fp8 ----
    fn quant(&mut self, b: &mut FnBuilder, cx: &mut NormCx) {
        let (nld, nv, lane) = (cx.nld, cx.nv, cx.lane);
        let me = cx.red.mine(cx.wid);
        let r = cx.inv.expect("rms");
        let mut u = Units::new();
        let (_, (s, words)) = u.capture(b, "quant.amax_fp8", |b| {
            let (xf, gf) = (&cx.xs, &cx.gs);
            let y_of = |b: &mut FnBuilder, i: usize| -> Vec<ValueId> {
                let (xv, gv) = (bf16x8_to_f32(b, xf[i]), bf16x8_to_f32(b, gf[i]));
                scale_rows(b, &xv, &gv, r)
            };
            let mut amax = None;
            for i in 0..nld {
                let y = y_of(b, i);
                let ab: Vec<ValueId> = y.iter().map(|&v| abs_f32(b, v)).collect();
                let m = lane_reduce(b, RedOp::Max, &ab);
                amax = Some(match amax { None => m, Some(a) => b.max_f32(a, m) });
            }
            let amax = wave_reduce(b, RedOp::Max, amax.expect("values"), &me, 64, 1, &lane);
            let (s, inv_s) = fp8_scale(b, amax);
            let mut words = Vec::with_capacity(nv / 4);
            for i in 0..nld {
                let y = y_of(b, i);
                words.extend(fp8_pack(b, &y, inv_s));
            }
            (s, words)
        });
        let (q2, qt2) = (words.clone(), cx.qt.clone());
        let c_q = StageContract {
            live_in: vec![sval("x", cx.xs.clone(), 128, cx.ptbl.clone()), sval("gamma", cx.gs.clone(), 128, cx.ptbl.clone())],
            scalars_in: vec![StageScalar { name: "inv_rms".into(), values: vec![r], owner: "wave".into() }],
            roles: cx.roles.clone(),
            ..Default::default()
        };
        StageInst::new("quant.fp8_row", Prim::Q, "quant", c_q, u)
            .param("format", "e4m3")
            .param("granularity", "per-token")
            .param("scale", "f32")
            .param("map", "x * inv_rms * gamma")
            .handover(move |_b| (vec![sval("q", q2, 32, qt2)], vec![StageScalar { name: "scale".into(), values: vec![s], owner: "wave".into() }]))
            .emit(b);
        (cx.qw, cx.sc) = (words, Some(s));
    }

    // ---- stage `store` (M): q and the row scale ----
    fn store(&mut self, b: &mut FnBuilder, cx: &mut NormCx) {
        let (n, nld, row, r_q, r_s, lane) = (cx.n, cx.nld, cx.row, cx.r_q, cx.r_s, cx.lane);
        let s = cx.sc.expect("quant");
        let mut u = Units::new();
        u.capture(b, "store.rows", |b| {
            let words = &cx.qw;
            let vq = b.bind_i32(ev(row) * n as i32 + (lane.lane() << 3));
            let gq: Vec<ValueId> = (0..nld.div_ceil(8)).map(|q| if q == 0 { vq } else { b.bind_i32(ev(vq) + 4096 * q as i32) }).collect();
            for i in 0..nld {
                let f2 = b.frag2(words[2 * i], words[2 * i + 1]);
                b.buffer_store_b64(r_q, gq[i / 8], 0i32, (512 * (i % 8)) as u32, f2);
            }
            let vs = b.bind_i32(ev(row) << 2);
            b.buffer_store_f32(r_s, vs, 0i32, 0, s);
        });
        let c_store = StageContract {
            live_in: vec![sval("q", cx.qw.clone(), 32, cx.qt.clone())],
            scalars_in: vec![StageScalar { name: "scale".into(), values: vec![s], owner: "wave".into() }],
            roles: cx.roles.clone(),
            ..Default::default()
        };
        StageInst::new("move.row_store", Prim::M, "store", c_store, u).param("dtype", "fp8_e4m3 + f32 scale").emit(b);
    }
}
fn scale_rows(b: &mut FnBuilder, xs: &[ValueId], gs: &[ValueId], r: ValueId) -> Vec<ValueId> {
    xs.iter().zip(gs).map(|(&x, &g)| { let t = b.mul_f32(x, r); b.mul_f32(t, g) }).collect()
}