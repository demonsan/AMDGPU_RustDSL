//! MUBUF LDS-direct DMA partition (the aiter GEMM form): for every thread of 4 waves and every instruction, the emitted (voffset + soffset + imm,
//! M0 + imm + 16 lane) equals the layout's (source byte, LDS byte) of the element - on an A-like tile [256 rows x 128 B] with a 8192-byte source row
//! stride and an LDS image of 8-row slots of 1056 bytes (32-byte pad).
use std::collections::HashMap;

use amdgpu_dsl::tile::{ev, Lane, LdsPlan, Layout as DslLayout, SlotIdx};
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd, Stmt, I32};
use amdgpu_isa::BUFFER_LDS_X4;
use amdgpu_layout::*;
use amdgpu_tile::*;

fn eval(defs: &HashMap<u32, Op>, o: &Opnd, tid: i64, memo: &mut HashMap<u32, i64>) -> i64 {
    match o {
        Opnd::I32(c) => *c as i64,
        Opnd::Val(v) => {
            if let Some(x) = memo.get(v) {
                return *x;
            }
            let w = |x: i64| x as i32 as i64;
            let op = defs.get(v).unwrap_or_else(|| panic!("v{v}"));
            let mut e = |o: &Opnd| eval(defs, o, tid, memo);
            let r = match op {
                Op::ConstI32(c) => *c as i64,
                Op::WorkitemId => tid,
                Op::AddI32(a, b) => w(e(a) + e(b)),
                Op::MulI32(a, b) => w(e(a) * e(b)),
                Op::ShlI32(a, b) => w(e(a) << e(b)),
                Op::LshrI32(a, b) => ((e(a) as u32) >> e(b)) as i64,
                Op::AndI32(a, b) => e(a) & e(b),
                Op::XorI32(a, b) => e(a) ^ e(b),
                other => panic!("eval: unsupported op {other:?}"),
            };
            memo.insert(*v, r);
            r
        }
        other => panic!("eval: operand {other:?}"),
    }
}

#[test]
fn buffer_dma_partition_matches_the_layouts() {
    let mut b = FnBuilder::new("bufdma", 256);
    b.arg("a", DType::Ptr, 0, 8);
    let mut lds = LdsPlan::new();
    let a_s = lds.ring("A", 32 * 1056, 2, DslLayout::Linear);
    lds.commit(&mut b);
    let ptr = b.let_(Op::KernargPtr { offset: 0 });
    let rsrc = b.buffer_rsrc(ptr);
    let lane = Lane::new(&mut b);
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    let w = ctx.sym("wave", 0, 3, lane.wave());
    // tile element = row + 256 k; thread = (chunk = lane % 8, row = lane / 8, wave); value = (byte, issue)
    let tv = make_layout(tup![tup![8, 8, 4], tup![16, 8]], tup![tup![16 * 256, 1, 64], tup![256, 8]]);
    proofs::check_tv(&tv, 256 * 128, false).unwrap();
    let src = make_layout(tup![256, 128], tup![8192, 1]);
    let dst = ComposedLayout::plain(make_layout(tup![tup![8, 32], 128], tup![tup![128, 1056], 1]));
    let part = partition_buffer_dma(&mut b, &ctx, BUFFER_LDS_X4, &tv, &src, &dst, 8, &l, &w, &a_s);
    let soff = 0x1_0000;
    assert_eq!(part.voffsets(), 1, "the source advances by a thread-independent amount per issue: one voffset, the rest in the soffset");
    part.issue(&mut b, rsrc, Opnd::I32(soff), &a_s.at(&SlotIdx::new(1, 1)));
    let f = b.finish();
    let _ = ev;
    let _ = I32::c(0);

    let mut defs = HashMap::new();
    let mut dmas = Vec::new();
    for s in f.stmts() {
        match s {
            Stmt::Let { dst, op } => {
                defs.insert(*dst, op.clone());
            }
            Stmt::BufferLoadLds { voff, soff, lds_addr, imm, dwords, .. } => {
                assert_eq!(*dwords, 4);
                dmas.push((voff.expect("voff"), *soff, *lds_addr, *imm as i64));
            }
            _ => {}
        }
    }
    assert_eq!(dmas.len(), 8, "8 issues per wave");
    let slot_base = a_s.base + a_s.slot_stride;
    for tid in 0..256i64 {
        let mut memo = HashMap::new();
        for (q, (vo, so, la, imm)) in dmas.iter().enumerate() {
            let g = eval(&defs, vo, tid, &mut memo) + eval(&defs, so, tid, &mut memo) + imm;
            let m0 = eval(&defs, la, tid, &mut memo);
            let e = tv.get(0).at(tid) + tv.get(1).at(16 * q as i64);
            assert_eq!(g, soff as i64 + src.at(e), "thread {tid} issue {q}: source byte");
            assert_eq!(m0 + imm + 16 * (tid % 64), slot_base + dst.at(e), "thread {tid} issue {q}: LDS byte");
        }
    }
    // the vendor-style split: the destination steps of issues 0..3 (1056 q) fit the immediate
    assert_eq!(dmas.iter().filter(|d| d.3 > 0).count(), 3);
}
