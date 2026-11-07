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

use amdgpu_dsl::{CompileProfile, DType, Ext, FnBuilder, Function, Op, Opnd, StageContract, StageScalar, StageVal, ValueId};
use amdgpu_tile::prims::{abs_f32, bf16x8_to_f32, lane_reduce, quant_fp8_row, rsqrt_f32, wave_reduce, RedOp};
use amdgpu_tile::stage::{Prim, StageInst};
use amdgpu_tile::weave::{Hint, Out, UnitClass, Units};
use amdgpu_tile::{ev, Lane, LdsPlan};

pub fn by_name(name: &str) -> Option<Function> {
    let n: i64 = name.strip_prefix("rmsnorm_quant_fp8_n")?.parse().ok()?;
    (n % 512 == 0 && (512..=16384).contains(&n)).then(|| rmsnorm_quant_fp8(name, n))
}

/// The element table of a per-lane value set: value v of lane t holds element 8 (t + 64 (v / 8)) + v % 8 (`[t + 64 v]`).
fn row_table(nv: usize) -> Vec<i64> {
    (0..64 * nv).map(|i| { let (t, v) = ((i % 64) as i64, (i / 64) as i64); 8 * (t + 64 * (v / 8)) + v % 8 }).collect()
}

fn sval(name: &str, values: Vec<ValueId>, bits: u32, table: Vec<i64>) -> StageVal {
    StageVal { name: name.into(), values, table, elem_bits: bits, storage: "vgpr".into(), layout: String::new() }
}

pub fn rmsnorm_quant_fp8(name: &str, n: i64) -> Function {
    let nld = (n / 512) as usize; // 16-byte loads per lane
    let nv = 8 * nld; // values per lane
    let mut fb = FnBuilder::new(name, 256);
    fb.compile_profile(CompileProfile::Gemm);
    let b = &mut fb;
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
    let me = red.mine(wid);
    let roles = vec!["all 4 waves: one row each".to_string()];
    let tbl = row_table(nv);

    // ---- stage `load` (M): x and gamma rows -> f32 ----
    let xs: Out<Vec<ValueId>> = Out::new();
    let gs: Out<Vec<ValueId>> = Out::new();
    let mut u = Units::new();
    let (x1, g1) = (xs.clone(), gs.clone());
    let lane2 = &lane;
    u.add("load.rows", UnitClass::Dma, Hint::LatencySensitive, &[], move |b| {
        let vx = b.bind_i32(ev(row) * (2 * n) as i32 + (lane2.lane() << 4));
        let vg = b.bind_i32(lane2.lane() << 4);
        let fx: Vec<ValueId> = (0..nld).map(|i| b.buffer_load_b128(r_x, vx, Opnd::I32(1024 * i as i32), 0)).collect();
        let fg: Vec<ValueId> = (0..nld).map(|i| b.buffer_load_b128(r_g, vg, Opnd::I32(1024 * i as i32), 0)).collect();
        b.wait_vm(0);
        x1.set(fx.iter().flat_map(|f| bf16x8_to_f32(b, *f)).collect());
        g1.set(fg.iter().flat_map(|f| bf16x8_to_f32(b, *f)).collect());
    });
    let (x2, g2) = (xs.clone(), gs.clone());
    let t1 = tbl.clone();
    StageInst::new("move.row_load", Prim::M, "load", StageContract { roles: roles.clone(), ..Default::default() }, u)
        .param("n", n)
        .param("dtype", "bf16")
        .handover(move |_b| (vec![sval("x", x2.get(), 32, t1.clone()), sval("gamma", g2.get(), 32, t1)], Vec::new()))
        .emit(b);

    // ---- stage `rms` (R + E): y = x rsqrt(mean(x^2) + eps) gamma ----
    let ys: Out<Vec<ValueId>> = Out::new();
    let inv: Out<ValueId> = Out::new();
    let mut u = Units::new();
    let (x3, y1, i1) = (xs.clone(), ys.clone(), inv.clone());
    let me1 = me.clone();
    let lane3 = &lane;
    u.add("rms.reduce", UnitClass::Valu, Hint::Skeleton, &[], move |b| {
        let sq: Vec<ValueId> = x3.get().iter().map(|&x| b.mul_f32(x, x)).collect();
        let part = lane_reduce(b, RedOp::Sum, &sq);
        let tot = wave_reduce(b, RedOp::Sum, part, &me1, 0, 0, lane3);
        let inv_n = b.c_f32(1.0 / n as f32);
        let mean = b.mul_f32(tot, inv_n);
        let me_eps = b.add_f32(mean, eps);
        i1.set(rsqrt_f32(b, me_eps));
    });
    let (i2, x4, g4) = (inv.clone(), xs.clone(), gs.clone());
    u.add("rms.scale", UnitClass::Valu, Hint::Filler, &[0], move |b| {
        let r = i2.get();
        y1.set(scale_rows(b, &x4.get(), &g4.get(), r));
    });
    let (y2, i3) = (ys.clone(), inv.clone());
    let t2 = tbl.clone();
    // the consumed sets: the producer has been emitted, its values exist
    let c_rms = StageContract { live_in: vec![sval("x", xs.get(), 32, tbl.clone()), sval("gamma", gs.get(), 32, tbl.clone())], roles: roles.clone(), ..Default::default() };    StageInst::new("map.rmsnorm", Prim::E, "rms", c_rms, u)
        .param("reduce", "sum x^2 over the row (wave)")
        .handover(move |_b| (vec![sval("y", y2.get(), 32, t2)], vec![StageScalar { name: "inv_rms".into(), values: vec![i3.get()], owner: "wave".into() }]))
        .emit(b);

    // ---- stage `quant` (R + Q): amax over the row, per-row fp8 ----
    let qw: Out<Vec<ValueId>> = Out::new();
    let sc: Out<ValueId> = Out::new();
    let mut u = Units::new();
    let (y3, q1, s1) = (ys.clone(), qw.clone(), sc.clone());
    let me2 = me.clone();
    let lane4 = &lane;
    u.add("quant.amax_fp8", UnitClass::Valu, Hint::Skeleton, &[], move |b| {
        let y = y3.get();
        let ab: Vec<ValueId> = y.iter().map(|&v| abs_f32(b, v)).collect();
        let part = lane_reduce(b, RedOp::Max, &ab);
        let amax = wave_reduce(b, RedOp::Max, part, &me2, 64, 1, lane4);
        let (s, words) = quant_fp8_row(b, &y, amax);
        s1.set(s);
        q1.set(words);
    });
    let (q2, s2) = (qw.clone(), sc.clone());
    // q words: word w of lane t holds elements 8 (t + 64 (w / 2)) + 4 (w % 2) + 0..3 (fp8: 4 per dword) -> table over the first element of each word
    let qt: Vec<i64> = (0..64 * 2 * nld).map(|i| { let (t, w) = ((i % 64) as i64, (i / 64) as i64); 8 * (t + 64 * (w / 2)) + 4 * (w % 2) }).collect();
    let qt2 = qt.clone();
    StageInst::new("quant.fp8_row", Prim::Q, "quant", StageContract { live_in: vec![sval("y", ys.get(), 32, tbl.clone())], roles: roles.clone(), ..Default::default() }, u)
        .param("format", "e4m3")
        .param("granularity", "per-token")
        .param("scale", "f32")
        .handover(move |_b| (vec![sval("q", q2.get(), 32, qt2)], vec![StageScalar { name: "scale".into(), values: vec![s2.get()], owner: "wave".into() }]))
        .emit(b);

    // ---- stage `store` (M): q and the row scale ----
    let mut u = Units::new();
    let (q3, s3) = (qw.clone(), sc.clone());
    let lane5 = &lane;
    u.add("store.rows", UnitClass::Dma, Hint::Anchor, &[], move |b| {
        let words = q3.get();
        let vq = b.bind_i32(ev(row) * n as i32 + (lane5.lane() << 3));
        for i in 0..nld {
            let f2 = b.frag2(words[2 * i], words[2 * i + 1]);
            b.buffer_store_b64(r_q, vq, Opnd::I32(512 * i as i32), 0, f2);
        }
        let vs = b.bind_i32(ev(row) << 2);
        b.buffer_store_f32(r_s, vs, 0i32, 0, s3.get());
    });
    let c_store = StageContract {
        live_in: vec![sval("q", qw.get(), 32, qt)],
        scalars_in: vec![StageScalar { name: "scale".into(), values: vec![sc.get()], owner: "wave".into() }],
        roles,
        ..Default::default()
    };
    StageInst::new("move.row_store", Prim::M, "store", c_store, u).param("dtype", "fp8_e4m3 + f32 scale").emit(b);
    fb.finish()
}

fn scale_rows(b: &mut FnBuilder, xs: &[ValueId], gs: &[ValueId], r: ValueId) -> Vec<ValueId> {
    xs.iter().zip(gs).map(|(&x, &g)| { let t = b.mul_f32(x, r); b.mul_f32(t, g) }).collect()
}