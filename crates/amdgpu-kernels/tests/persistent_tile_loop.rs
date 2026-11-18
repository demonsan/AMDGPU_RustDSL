//! The tile loop of a persistent kernel in the tile-protocol verifier (`FnBuilder::persistent_tile_loop`): a workgroup loops over output tiles, each iteration running a self-contained pipeline.
//! The verifier simulates the tile body once and CHECKS that it ends quiescent: every tensor operation drained and a barrier after the last LDS access, so the next tile's first stage cannot race
//! with the previous tile's reads. Without the launch fact a nested loop with markers stays unsupported.

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile};

/// A 2-slot ring; per tile: stage slot 0 and slot 1 (generation 0), wait, barrier, read both, (barrier), then the next tile. `drain` / `fence`: the body's last wait and last barrier.
fn kernel(declared: bool, drain: bool, fence: bool) -> Function {
    let mut b = FnBuilder::new("pt_loop", 128);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg_tensor("src", Ext::c(1 << 16));
    b.arg_tensor("out", Ext::grid_x() * Ext::c(128 * 4 * 64));
    b.lds(2 * 128 * 4);
    b.lds_ring("ring", 0, 512, 2);
    if declared {
        b.persistent_tile_loop();
    }
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let wg = b.let_(Op::WorkgroupIdX);
    let tile = TensorTile { data_size: 4, tensor_dim: [Opnd::I32(16), Opnd::I32(8)], stride0: Opnd::I32(16), tile: [16, 8], pad: None, wg_mask: 0 };
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    for i in 0..2usize {
        let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::I32((i * 512) as i32), Opnd::I32((i * 512) as i32));
        b.tile_produce("ring", Opnd::I32(i as i32), Opnd::I32(0));
        b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default());
        b.tile_end();
    }
    b.wait_tensor(0);
    b.barrier();
    let mut acc = b.c_i32(0);
    for i in 0..2usize {
        b.tile_consume("ring", Opnd::I32(i as i32), Opnd::I32(0));
        let x = b.load_lds(Opnd::Val(a), MemTy::B32);
        let xo = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32((i * 512) as i32)));
        let y = b.load_lds(Opnd::Val(xo), MemTy::B32);
        acc = b.add_i32(acc, x);
        acc = b.add_i32(acc, y);
        b.tile_end();
    }
    b.wait_lds(0);
    if fence {
        b.barrier();
    }
    // the tile's result: out[wg][tile][tid]
    let tl = b.let_(Op::ShlI32(Opnd::Var(vars[0]), Opnd::I32(9)));
    let wo = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(15)));
    let o = b.let_(Op::AddI32(Opnd::Val(tl), Opnd::Val(wo)));
    let o = b.let_(Op::AddI32(Opnd::Val(o), Opnd::Val(a)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
    b.store_global(q, 0, acc, MemTy::B32, false);
    let nx = b.add_i32(Opnd::Var(vars[0]), Opnd::I32(1));
    b.set_var(vars[0], nx);
    // an inner loop (the K pipeline stands in here: a trivial counted loop, so the tile loop is a loop nest)
    let inner = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let ni = b.add_i32(Opnd::Var(inner[0]), Opnd::I32(1));
    b.set_var(inner[0], ni);
    let c = b.cmp(amdgpu_dsl::CmpPred::Ge, amdgpu_dsl::CmpTy::I32, ni, Opnd::I32(2));
    b.break_if(c);
    b.end_loop();
    if drain {
        b.wait_tensor(0);
    }
    let c2 = b.cmp(amdgpu_dsl::CmpPred::Ge, amdgpu_dsl::CmpTy::I32, nx, Opnd::I32(4));
    b.break_if(c2);
    b.end_loop();
    b.finish()
}

fn errs(f: &Function) -> String {
    match amdgpu_codegen::compile_target(f, "gfx1250") {
        Ok(_) => String::new(),
        Err(e) => e,
    }
}

#[test]
fn a_declared_tile_loop_with_a_quiescent_body_is_accepted() {
    let e = errs(&kernel(true, true, true));
    assert!(e.is_empty(), "{e}");
}

#[test]
fn a_tile_body_without_the_final_barrier_is_rejected() {
    let e = errs(&kernel(true, true, false));
    assert!(e.contains("no workgroup barrier after its last LDS access"), "{e}");
}
