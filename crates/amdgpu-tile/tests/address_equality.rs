//! TiledCopy over the FMHA K/V ring reproduces, lane by lane and instruction by instruction, the LDS addresses of the hand-derived readers of
//! `amdgpu_dsl::tile::frag` (`RowsReader` = the K fragments, `TrReader` = the transposed V fragments), evaluated on the emitted IR.
use std::collections::HashMap;

use amdgpu_dsl::tile::frag::{RowsReader, TrReader};
use amdgpu_dsl::tile::{Lane, LdsPlan, Layout as DslLayout, SlotIdx};
use amdgpu_dsl::{FnBuilder, Function, Op, Opnd, Stmt};
use amdgpu_isa::{MmaAtom, DS_READ_B128X2, DS_READ_TR_B8X4};
use amdgpu_layout::*;
use amdgpu_tile::*;

/// Evaluates an i32 operand of straight-line IR for one workitem id.
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

/// The LDS addresses of every LDS read of `f`, per workitem 0..64, in program order.
fn lds_addresses(f: &Function) -> Vec<Vec<i64>> {
    let mut defs: HashMap<u32, Op> = HashMap::new();
    let mut reads: Vec<Vec<Opnd>> = Vec::new();
    for s in &f.stmts {
        if let Stmt::Let { dst, op } = s {
            match op {
                Op::LoadLdsB128 { addr } | Op::LoadLdsB32 { addr } | Op::LoadLdsTrB8 { addr } => reads.push(vec![*addr]),
                Op::LoadLdsB128x2 { lo, hi } => reads.push(vec![*lo, *hi]),
                Op::LoadLdsTr4B8 { addrs } => reads.push(addrs.to_vec()),
                _ => {}
            }
            defs.insert(*dst, op.clone());
        }
    }
    (0..64)
        .map(|tid| {
            let mut memo = HashMap::new();
            reads.iter().flatten().map(|o| eval(&defs, o, tid, &mut memo)).collect()
        })
        .collect()
}

fn ring(b: &mut FnBuilder) -> (amdgpu_dsl::tile::Shared, amdgpu_dsl::tile::Shared) {
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(2, [("K", 16384, DslLayout::FMHA_SWZ), ("V", 16384, DslLayout::FMHA_SWZ)]);
    lds.commit(b);
    (ks, vs)
}

fn kv_view() -> ComposedLayout {
    ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![128, 128], tup![128, 1]))
}

#[test]
fn tiled_copy_k_fragments_equal_rows_reader() {
    // hand: RowsReader (kfx) at slot 1, the 4 row groups x 2 k-groups
    let mut b = FnBuilder::new("old", 64);
    let (ks, _) = ring(&mut b);
    let lane = Lane::new(&mut b);
    let kr = RowsReader::new(&mut b, &ks, &lane);
    let k = kr.at(&mut b, &ks.at(&SlotIdx::new(1, 1)));
    for s in 0..4 {
        let _ = k.read_pair(&mut b, s);
    }
    let old = lds_addresses(&b.finish());

    // layout-derived: the A operand of the scaled 32x32x64 MFMA over the [key, d] tile, placements 4096 s + 64 c
    let mut b = FnBuilder::new("new", 64);
    let (ks, _) = ring(&mut b);
    let lane = Lane::new(&mut b);
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    let mma = make_tiled_mma(MmaAtom::mfma_scale_32x32x64_f8(), false);
    let rd = make_tiled_copy_a(DS_READ_B128X2, &mma);
    let view = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![32, 64], tup![128, 1]));
    let extras: Vec<i64> = (0..4).flat_map(|s| (0..2).map(move |c| 4096 * s + 64 * c)).collect();
    let part = rd.partition_lds(&mut b, &ctx, &l, &ks, &view, &extras);
    assert_eq!(part.bases.len(), 4, "the four kfx bases");
    let at = part.at(&mut b, &ks.at(&SlotIdx::new(1, 1)));
    for o in 0..at.len() {
        let _ = at.read(&mut b, o);
    }
    let new = lds_addresses(&b.finish());
    assert_eq!(old.len(), 64);
    assert_eq!(old[0].len(), 16);
    assert_eq!(old, new, "per-lane LDS addresses of the K fragment reads");
}

#[test]
fn tiled_copy_transposed_v_equals_tr_reader() {
    let mut b = FnBuilder::new("old", 64);
    let (_, vs) = ring(&mut b);
    let lane = Lane::new(&mut b);
    let vr = TrReader::new(&mut b, &vs, &lane);
    let v = vr.at(&mut b, &vs.at(&SlotIdx::new(1, 1)));
    for hh in 0..2 {
        for e in 0..4 {
            let _ = v.read(&mut b, hh, e);
        }
    }
    let old = lds_addresses(&b.finish());

    // V^T source addresses (tile coordinates key + 128 dbyte), see amdgpu-layout tests/fmha_vs128.rs; values (read r, half hh, column group e)
    let tr = make_layout(tup![tup![2, 2, 2, 2, 2, 2], tup![4, 2, 4]], tup![tup![8 * 128, 1, 2, 8, 16 * 128, 4], tup![16, 64, 32 * 128]]);
    let mut b = FnBuilder::new("new", 64);
    let (_, vs) = ring(&mut b);
    let lane = Lane::new(&mut b);
    let mut ctx = Ctx::new();
    let l = ctx.sym("lane", 0, 63, lane.lane());
    let rd = make_tiled_copy(DS_READ_TR_B8X4, tr, 128 * 128, 8);
    let part = rd.partition_lds(&mut b, &ctx, &l, &vs, &kv_view(), &[0]);
    assert_eq!(part.bases.len(), 4, "the four trx bases");
    let at = part.at(&mut b, &vs.at(&SlotIdx::new(1, 1)));
    for hh in 0..2 {
        for e in 0..4 {
            let _ = at.read(&mut b, hh + 2 * e);
        }
    }
    let new = lds_addresses(&b.finish());
    assert_eq!(old[0].len(), 32);
    assert_eq!(old, new, "per-lane LDS addresses of the transposed V reads");
}

#[test]
fn mma_atoms_cover_their_tiles_and_swap_transposes_c() {
    for a in [MmaAtom::mfma_scale_32x32x64_f8(), MmaAtom::mfma_scale_16x16x128_f8(), MmaAtom::mfma_32x32x8_f16()] {
        proofs::check_tv(&a.a_tv, a.m * a.k, false).unwrap_or_else(|e| panic!("{} A: {e}", a.name));
        proofs::check_tv(&a.b_tv, a.n * a.k, false).unwrap_or_else(|e| panic!("{} B: {e}", a.name));
        proofs::check_tv(&a.c_tv, a.m * a.n, false).unwrap_or_else(|e| panic!("{} C: {e}", a.name));
        let s = make_tiled_mma(a.clone(), true);
        let (t0, t1) = (proofs::tv_table(&a.c_tv), proofs::tv_table(&s.c_tv()));
        for (x, y) in t0.iter().zip(&t1) {
            assert_eq!(*y, (x / a.m) + a.m * (x % a.m), "{}: swapped C is the transpose", a.name);
        }
    }
}
