//! Runtime ifs inside loops in the tile-protocol verifier (`FnBuilder::loop_ifs_explored`): a wave-uniform "stage this way or that way" if inside the loop, whose arms hold tile markers, tensor
//! operations and waits. Without the launch fact the verifier rejects it (`GV-TILE-UNSUPPORTED`); with it, every occurrence of the if in every simulated iteration is a decision, so an arm that does
//! not complete its tensor load before the consume is found.

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile};

fn kernel(declare: bool, else_waits: bool) -> Function {
    let mut b = FnBuilder::new("li_loop", 64);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("flag", DType::I32, 16, 4);
    b.arg_tensor("src", Ext::c(1 << 16));
    b.arg_tensor("out", Ext::c(64 * 4));
    b.lds(512);
    b.lds_ring("ring", 0, 512, 1);
    if declare {
        b.loop_ifs_explored();
    }
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let flag = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let tile = TensorTile { data_size: 4, tensor_dim: [Opnd::I32(16), Opnd::I32(8)], stride0: Opnd::I32(16), tile: [16, 8], pad: None, wg_mask: 0 };
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let c = b.cmp(amdgpu_dsl::CmpPred::Gt, amdgpu_dsl::CmpTy::I32, flag, Opnd::I32(0));
    // the stage is uniform across the workgroup (a kernarg decides) but differs between the arms
    b.begin_if(Opnd::Val(c));
    let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::I32(0), Opnd::I32(0));
    b.tile_produce("ring", Opnd::I32(0), Opnd::I32(0));
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default());
    b.tile_end();
    b.wait_tensor(0);
    b.else_();
    let (h0, h1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::I32(512), Opnd::I32(0));
    b.tile_produce("ring", Opnd::I32(0), Opnd::I32(0));
    b.tensor_load_to_lds(Opnd::Val(h0), Opnd::Val(h1), None, TdmCpol::default());
    b.tile_end();
    if else_waits {
        b.wait_tensor(0);
    }
    b.end_if();
    b.barrier();
    b.tile_consume("ring", Opnd::I32(0), Opnd::I32(0));
    let x = b.load_lds(Opnd::Val(a), MemTy::B32);
    b.tile_end();
    b.wait_lds(0);
    b.barrier();
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(a) });
    b.store_global(q, 0, x, MemTy::B32, false);
    let nx = b.add_i32(Opnd::Var(vars[0]), Opnd::I32(1));
    b.set_var(vars[0], nx);
    let stop = b.cmp(amdgpu_dsl::CmpPred::Ge, amdgpu_dsl::CmpTy::I32, nx, Opnd::I32(3));
    b.break_if(stop);
    b.end_loop();
    b.wait_tensor(0);
    b.finish()
}

fn errs(f: &Function) -> String {
    match amdgpu_codegen::compile_target(f, "gfx1250") {
        Ok(_) => String::new(),
        Err(e) => e,
    }
}

#[test]
fn a_runtime_if_in_a_loop_is_unsupported_unless_declared() {
    let e = errs(&kernel(false, true));
    assert!(e.contains("a runtime if INSIDE a loop"), "{e}");
}

#[test]
fn a_declared_runtime_if_with_both_arms_completing_is_accepted() {
    let e = errs(&kernel(true, true));
    assert!(e.is_empty(), "{e}");
}

#[test]
fn a_declared_runtime_if_with_an_arm_that_does_not_complete_is_rejected() {
    let e = errs(&kernel(true, false));
    assert!(e.contains("GV-TILE"), "{e}");
}
