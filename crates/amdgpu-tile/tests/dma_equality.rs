//! The layout-derived global -> LDS DMA (`partition_dma` over the swizzled FMHA K tile) emits, for every workitem of a 4-wave group and every
//! instruction, the same global address, instruction offset and LDS destination as the hand `RowsCopy` (tile::copy) - including the row clamp of a
//! partial tile, M0 at the top of the wave slice and the compensated negative immediates.
use std::collections::HashMap;

use amdgpu_dsl::tile::copy::RowsCopy;
use amdgpu_dsl::tile::{ev, Dim, KArgs, Lane, LdsPlan, Layout as DslLayout, Roles, SlotIdx, Stride};
use amdgpu_dsl::{FnBuilder, Function, Op, Opnd, Stmt, I32};
use amdgpu_isa::GLOBAL_LDS_X4;
use amdgpu_layout::*;
use amdgpu_tile::*;

const KV_LEN: i64 = 200; // a partial second tile: rows 200.. are clamped to row 199

fn eval(defs: &HashMap<u32, Op>, o: &Opnd, tid: i64, memo: &mut HashMap<u32, i64>) -> i64 {
    match o {
        Opnd::I32(c) => *c as i64,
        Opnd::Val(v) => {
            if let Some(x) = memo.get(v) {
                return *x;
            }
            let w = |x: i64| x as i32 as i64;
            let op = defs.get(v).unwrap_or_else(|| panic!("v{v} has no definition"));
            let mut e = |o: &Opnd| eval(defs, o, tid, memo);
            let r = match op {
                Op::ConstI32(c) => *c as i64,
                Op::WorkitemId => tid,
                Op::KernargI32 { .. } => KV_LEN,
                Op::KernargPtr { .. } => 0,
                Op::PtrAdd { base, byte_off } => e(base) + e(byte_off),
                Op::AddI32(a, b) => w(e(a) + e(b)),
                Op::MulI32(a, b) => w(e(a) * e(b)),
                Op::ShlI32(a, b) => w(e(a) << e(b)),
                Op::LshrI32(a, b) => ((e(a) as u32) >> e(b)) as i64,
                Op::AndI32(a, b) => e(a) & e(b),
                Op::XorI32(a, b) => e(a) ^ e(b),
                Op::MinI32(a, b) => e(a).min(e(b)),
                Op::CmpLtI32(a, b) => (e(a) < e(b)) as i64,
                Op::Select { cond, a, b, .. } => {
                    if e(cond) != 0 {
                        e(a)
                    } else {
                        e(b)
                    }
                }
                other => panic!("eval: unsupported op {other:?}"),
            };
            memo.insert(*v, r);
            r
        }
        other => panic!("eval: operand {other:?}"),
    }
}

/// (global byte address, instruction offset, LDS destination) of every DMA, per workitem 0..256.
fn dmas(f: &Function) -> Vec<Vec<(i64, i32, i64)>> {
    let mut defs: HashMap<u32, Op> = HashMap::new();
    let mut d: Vec<(Opnd, i32, Opnd)> = Vec::new();
    for s in &f.stmts {
        match s {
            Stmt::Let { dst, op } => {
                defs.insert(*dst, op.clone());
            }
            Stmt::LoadGlobalLds { base, lds_addr, byte_off, .. } => d.push((*base, *byte_off, *lds_addr)),
            _ => {}
        }
    }
    (0..256)
        .map(|tid| {
            let mut memo = HashMap::new();
            d.iter().map(|(b, off, l)| (eval(&defs, b, tid, &mut memo) + *off as i64, *off, eval(&defs, l, tid, &mut memo))).collect()
        })
        .collect()
}

/// Common kernel prologue: the K tensor [SK rows x 128 B], the ring, lanes and the clamped row source.
fn setup(b: &mut FnBuilder) -> (amdgpu_dsl::tile::Shared, Lane, Roles, amdgpu_dsl::tile::copy::RowSource) {
    let mut ka = KArgs::new();
    let skv = ka.dim(b, "kv_len", 0x10, 1, i64::MAX);
    let kv_len = ka.i32(b, "kv_len", 0x10);
    let k = ka.tensor("ptr_k", 0x00, 1, vec![skv, Dim::c(128)], vec![Stride::Derived(I32::c(128)), Stride::Unit]).packed().declare(&mut ka, b);
    let mut lds = LdsPlan::new();
    let [ks, _vs] = lds.ring_group(2, [("K", 16384, DslLayout::FMHA_SWZ), ("V", 16384, DslLayout::FMHA_SWZ)]);
    lds.commit(b);
    let lane = Lane::new(b);
    let roles = Roles::split(b, &lane, 4);
    let last = b.let_i32(ev(kv_len) - 1);
    let src = k.tile(vec![I32::c(0), I32::c(0)]).clamp(0, ev(kv_len), last).row_source(b, 0);
    (ks, lane, roles, src)
}

#[test]
fn layout_derived_dma_equals_rows_copy() {
    let key0 = I32::c(128); // the partial second tile
    let mut b = FnBuilder::new("old", 256);
    let (ks, lane, roles, src) = setup(&mut b);
    let cp = RowsCopy::new(&mut b, src, &ks, roles.wave_in_group, &lane, 4);
    cp.issue(&mut b, &key0, &ks.at(&SlotIdx::new(1, 1)));
    let old = dmas(&b.finish());

    let mut b = FnBuilder::new("new", 256);
    let (ks, lane, roles, src) = setup(&mut b);
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    let w = ctx.sym("wave", 0, 3, ev(roles.wave_in_group));
    let view = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![128, 128], tup![128, 1]));
    let part = partition_dma(&mut b, &ctx, GLOBAL_LDS_X4, &l, &w, &ev(roles.wave_in_group), 4, src, &ks, &view);
    part.issue(&mut b, &key0, &ks.at(&SlotIdx::new(1, 1)));
    let new = dmas(&b.finish());

    assert_eq!(old[0].len(), 4);
    assert_eq!(old, new, "per-workitem (global address, instruction offset, LDS destination) of the 4 DMAs");
    // sanity: the swizzled landing - the byte of tile row r chunk c is fetched for LDS chunk c ^ s(r); rows past KV_LEN re-read row KV_LEN-1
    let any_clamped = new.iter().flatten().any(|(g, _, _)| g / 128 == KV_LEN - 1);
    assert!(any_clamped);
}
