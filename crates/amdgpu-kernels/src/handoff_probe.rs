//! Self-checking hardware probes of `amdgpu_tile::handoff::convert_layout` (one wave): every lane fills its source fragment with the element ids
//! of the source TV layout, converts, and stores `got XOR expected` for every destination register (expected = the destination TV layout's element
//! ids); the host (tests/gpu/handoff_host.cpp) requires all zeros.

use amdgpu_tile::{ev, Lane, LdsPlan};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, ValueId};
use amdgpu_layout::{make_layout, plan_convert, tup, IntTuple, Ix, Layout};
use amdgpu_tile::handoff::{convert_layout, ConvStats};
use amdgpu_tile::Ctx;

/// (source, destination) TV layouts of the probe cases (64 threads x 16 values, f32 elements).
pub fn case(name: &str) -> Option<(Layout, Layout)> {
    let c32 = make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![32, 4], tup![1, 8]]);
    Some(match name {
        // whole registers renamed (value groups reversed)
        "handoff_probe_reg" => (c32.clone(), make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![32, 4], tup![8, 1]])),
        // lane bits 0 / 1 exchanged inside every quad
        "handoff_probe_quad" => (
            make_layout(tup![tup![2, 2, 16], 16], tup![tup![1, 2, 4], 64]),
            make_layout(tup![tup![2, 2, 16], 16], tup![tup![2, 1, 4], 64]),
        ),
        // the 32x32 MFMA C fragment to its transpose (lane = row): cross-lane, LDS round trip
        "handoff_probe_cross" => (c32, make_layout(tup![tup![32, 2], tup![4, 4]], tup![tup![1, 128], tup![32, 256]])),
        _ => return None,
    })
}


pub fn by_name(name: &str) -> Option<Function> {
    let (src, dst) = case(name)?;
    let p = plan_convert(&src, &dst, 1);
    let nv = p.values;
    let mut b = FnBuilder::new(name, 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(64 * nv as i64 * 4));
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let mut lds = LdsPlan::new();
    let scr = lds.private("scr", 256 * nv as i64, 1, &[]);
    lds.commit(&mut b);
    let ln = Lane::new(&mut b);
    let lane = ln.lane();
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.clone());
    let elems = |b: &mut FnBuilder, ctx: &Ctx, tv: &Layout| -> Vec<ValueId> {
        (0..nv as i64).map(|v| { let e = tv.crd2idx(&IntTuple::T(vec![IntTuple::L(l.clone()), IntTuple::L(Ix::C(v))]), &ctx.dyns); let x = ctx.lower(&e); b.bind_i32(x) }).collect()
    };
    let s = elems(&mut b, &ctx, &src);
    let want = elems(&mut b, &ctx, &dst);
    let w0 = b.c_i32(0);
    let me = scr.mine(w0);
    let mut st = ConvStats::default();
    let got = convert_layout(&mut b, &p, &s, &lane, Some((&me, 0)), &mut st);
    let rsrc = b.buffer_desc(out, (64 * nv * 4) as u32);
    for (r, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = b.bind_i32(ev(*g) ^ ev(*w));
        let vo = b.bind_i32((lane.clone() * nv as i32 + r as i32) << 2);
        let df = b.bitcast_f32(d);
        b.buffer_store_f32(rsrc, vo, 0i32, 0, df);
    }
    Some(b.finish())
}

