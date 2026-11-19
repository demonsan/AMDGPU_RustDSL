//! LDS float atomics in the tile protocol (gfx1250): a shared tile zeroed by plain stores in one phase, accumulated with `ds_add_f32` by
//! every wave in the next (concurrent adds to the same words are the point), read after a barrier. Mutations: the read without the barrier
//! after the adds (another wave may still be adding), and plain stores mixed with the atomics in one phase.

use amdgpu_dsl::{AtomicOp, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd};
use amdgpu_tile::shared::{Layout, LdsPlan, SlotIdx};

/// `mode`: 0 = correct, 1 = no barrier between the adds and the reads, 2 = a plain store inside the add phase.
fn kernel(mode: u8) -> Function {
    let mut b = FnBuilder::new("probe_lds_fadd_proto", 128);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::grid_x() * Ext::c(1024));
    let mut lds = LdsPlan::new();
    let acc = lds.tile("acc", 1024, Layout::Linear);
    lds.commit(&mut b);
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let wg = b.let_(Op::WorkgroupIdX);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let base = b.bind_i32(acc.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).base());
    let la = b.let_(Op::AddI32(Opnd::Val(a), Opnd::Val(base)));
    let zero = b.let_(Op::ConstI32(0));
    acc.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(0))).write(&mut b, |b| {
        b.store_lds_b32(la, zero);
        let la2 = b.let_(Op::AddI32(Opnd::Val(la), Opnd::I32(512)));
        b.store_lds_b32(la2, zero);
    });
    b.wait_lds(0);
    b.barrier();
    let xf = b.un(amdgpu_dsl::UnOp::CvtF32I32, tid);
    acc.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(1))).write(&mut b, |b| {
        for r in 0..4i32 {
            let i = b.let_(Op::AddI32(Opnd::Val(tid), Opnd::I32(7 * r)));
            let i = b.let_(Op::AndI32(Opnd::Val(i), Opnd::I32(255)));
            let o = b.let_(Op::ShlI32(Opnd::Val(i), Opnd::I32(2)));
            let o = b.let_(Op::AddI32(Opnd::Val(o), Opnd::Val(base)));
            b.atomic_lds(AtomicOp::AddF32, o, xf, false);
        }
        if mode == 2 {
            b.store_lds_b32(la, zero);
        }
    });
    b.wait_lds(0);
    if mode != 1 {
        b.barrier();
    }
    let v = acc.at(&SlotIdx::new(Opnd::I32(0), Opnd::I32(1))).read(&mut b, |b| b.load_lds(la, MemTy::B32));
    let g = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(10)));
    let o = b.let_(Op::AddI32(Opnd::Val(g), Opnd::Val(a)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
    b.store_global(q, 0, v, MemTy::B32, false);
    b.finish()
}

#[test]
fn atomic_accumulation_between_barriers_is_accepted() {
    amdgpu_codegen::compile_target(&kernel(0), "gfx1250").unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn reading_the_accumulator_without_the_barrier_is_refused() {
    let e = amdgpu_codegen::compile_target(&kernel(1), "gfx1250").err().expect("a read racing with other waves' atomic adds");
    assert!(e.contains("GV-TILE") || e.contains("GV-PROTO"), "{e}");
}

#[test]
fn plain_stores_mixed_with_atomics_in_one_phase_are_refused() {
    let e = amdgpu_codegen::compile_target(&kernel(2), "gfx1250").err().expect("plain stores racing with atomic adds");
    assert!(e.contains("GV-TILE-RACE"), "{e}");
}
