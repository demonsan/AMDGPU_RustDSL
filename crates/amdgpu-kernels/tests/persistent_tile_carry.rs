//! The CARRY form of a persistent kernel's tile loop (`FnBuilder::persistent_tile_carry`): the body issues the next tile's first stage late
//! (under a runtime `if`: the next tile exists) and the first tile's under its prologue `if` (the first iteration); an epilogue staging reuses
//! the consumed slot (`tile_over` of one slot) while the carried stage is in flight in the other. gpu_protocol (GV-TILE-CARRY) checks the
//! carried stage equals the prologue's, nothing precedes the prologue, the carried bytes stay untouched to the end of the body, the rest ends
//! quiescent; the reuse of the slot is checked against the ring per slot.

use amdgpu_dsl::{CmpPred, CmpTy, DType, Ext, FnBuilder, Function, MemTy, Op, Opnd, TdmCpol, TensorTile};

#[derive(Clone, Copy, PartialEq)]
enum Mut {
    None,
    /// the epilogue staging placed over slot 0 (the carried stage's slot)
    EpiOverCarried,
    /// the carried stage goes to slot 1 (not the prologue's slot)
    CarryOtherSlot,
    /// a read of slot 0 after the carried stage was issued
    ReadAfterCarry,
    /// an LDS access of the body before its prologue `if`
    AccessBeforePrologue,
    /// the carried stage with the prologue's version (no generation advance)
    SameVersion,
}

fn kernel(m: Mut) -> Function {
    let mut b = FnBuilder::new("pt_carry", 128);
    b.arg("src", DType::Ptr, 0, 8);
    b.arg("out", DType::Ptr, 8, 8);
    b.arg("ntiles", DType::I32, 16, 4);
    b.arg_tensor("src", Ext::c(1 << 16));
    b.arg_tensor("out", Ext::grid_x() * Ext::c(128 * 4 * 64));
    b.lds(2 * 512);
    b.lds_ring("ring", 0, 512, 2);
    // the epilogue staging over the consumed slot (slot 1), or over slot 0 (mutation)
    b.lds_tile("epi", if m == Mut::EpiOverCarried { 0 } else { 512 }, 512);
    b.lds_reuse("epi", "ring");
    b.persistent_tile_loop();
    b.persistent_tile_carry();
    let ps = b.let_(Op::KernargPtr { offset: 0 });
    let po = b.let_(Op::KernargPtr { offset: 8 });
    let nt = b.let_(Op::KernargI32 { offset: 16 });
    let tid = b.let_(Op::WorkitemId);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let a1 = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(512)));
    let wg = b.let_(Op::WorkgroupIdX);
    let tile = TensorTile { data_size: 4, tensor_dim: [Opnd::I32(16), Opnd::I32(8)], stride0: Opnd::I32(16), tile: [16, 8], pad: None, wg_mask: 0 };
    let stage = |b: &mut FnBuilder, slot: i32, ver: i32| {
        let (g0, g1) = b.tensor_desc_2d(&tile, Opnd::Val(ps), Opnd::I32(slot * 512), Opnd::I32(slot * 512));
        b.tile_produce("ring", Opnd::I32(slot), Opnd::I32(ver));
        b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default());
        b.tile_end();
    };
    // tile loop: (tile index, first-tile flag)
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0)), (DType::I32, Opnd::I32(1))]);
    if m == Mut::AccessBeforePrologue {
        b.tile_consume("ring", Opnd::I32(0), Opnd::I32(0));
        let _ = b.load_lds(Opnd::Val(a), MemTy::B32);
        b.tile_end();
        b.wait_lds(0);
    }
    // the prologue: the first tile's stage of slot 0
    let first = b.cmp(CmpPred::Eq, CmpTy::I32, Opnd::Var(vars[1]), Opnd::I32(1));
    b.begin_if(Opnd::Val(first));
    stage(&mut b, 0, 0);
    b.end_if();
    // the K pipeline (one window per slot): stage slot 1, consume slot 0 then slot 1
    stage(&mut b, 1, 0);
    b.wait_tensor(0);
    b.barrier();
    let mut acc = b.c_i32(0);
    for (s, addr) in [(0, a), (1, a1)] {
        b.tile_consume("ring", Opnd::I32(s), Opnd::I32(0));
        let x = b.load_lds(Opnd::Val(addr), MemTy::B32);
        acc = b.add_i32(acc, x);
        b.tile_end();
    }
    b.wait_lds(0);
    b.barrier();
    // an inner loop (the K loop stands in: the tile loop is a loop nest)
    let inner = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let ni = b.add_i32(Opnd::Var(inner[0]), Opnd::I32(1));
    b.set_var(inner[0], ni);
    let c = b.cmp(CmpPred::Ge, CmpTy::I32, ni, Opnd::I32(2));
    b.break_if(c);
    b.end_loop();
    // the carried stage: the next tile's slot 0 (generation + 1), when there is a next tile
    let nx = b.add_i32(Opnd::Var(vars[0]), Opnd::I32(1));
    let more = b.cmp(CmpPred::Lt, CmpTy::I32, nx, Opnd::Val(nt));
    b.begin_if(Opnd::Val(more));
    stage(&mut b, if m == Mut::CarryOtherSlot { 1 } else { 0 }, if m == Mut::SameVersion { 0 } else { 1 });
    b.end_if();
    if m == Mut::ReadAfterCarry {
        b.tile_consume("ring", Opnd::I32(0), Opnd::I32(0));
        let _ = b.load_lds(Opnd::Val(a), MemTy::B32);
        b.tile_end();
        b.wait_lds(0);
    }
    // the epilogue: stage the result through the consumed slot (each lane its own word), read it back, store it
    let ea = if m == Mut::EpiOverCarried { a } else { a1 };
    b.tile_produce("epi", Opnd::I32(0), Opnd::I32(0));
    b.store_lds(ea, acc, MemTy::B32, false);
    b.tile_end();
    b.wait_lds(0);
    b.barrier();
    b.tile_consume("epi", Opnd::I32(0), Opnd::I32(0));
    let y = b.load_lds(Opnd::Val(ea), MemTy::B32);
    b.tile_end();
    b.wait_lds(0);
    b.barrier();
    let tl = b.let_(Op::ShlI32(Opnd::Var(vars[0]), Opnd::I32(9)));
    let wo = b.let_(Op::ShlI32(Opnd::Val(wg), Opnd::I32(15)));
    let o = b.let_(Op::AddI32(Opnd::Val(tl), Opnd::Val(wo)));
    let o = b.let_(Op::AddI32(Opnd::Val(o), Opnd::Val(a)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(o) });
    b.store_global(q, 0, y, MemTy::B32, false);
    b.set_var(vars[0], nx);
    b.set_var(vars[1], Opnd::I32(0));
    let done = b.cmp(CmpPred::Ge, CmpTy::I32, nx, Opnd::Val(nt));
    b.break_if(done);
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
fn the_carry_form_is_accepted() {
    let e = errs(&kernel(Mut::None));
    assert!(e.is_empty(), "{e}");
}

#[test]
fn an_epilogue_over_the_carried_slot_is_rejected() {
    let e = errs(&kernel(Mut::EpiOverCarried));
    assert!(e.contains("GV-TILE-CARRY") || e.contains("GV-TILE-ALIAS-RACE"), "{e}");
}

#[test]
fn a_carried_stage_into_another_slot_is_rejected() {
    let e = errs(&kernel(Mut::CarryOtherSlot));
    assert!(e.contains("GV-TILE-CARRY"), "{e}");
}

#[test]
fn a_read_of_the_carried_slot_after_the_carry_is_rejected() {
    let e = errs(&kernel(Mut::ReadAfterCarry));
    assert!(e.contains("GV-TILE-CARRY") || e.contains("GV-TILE-RACE"), "{e}");
}

#[test]
fn an_access_before_the_prologue_is_rejected() {
    let e = errs(&kernel(Mut::AccessBeforePrologue));
    assert!(e.contains("GV-TILE-CARRY"), "{e}");
}

#[test]
fn a_carried_stage_without_a_generation_advance_is_rejected() {
    let e = errs(&kernel(Mut::SameVersion));
    assert!(e.contains("GV-TILE-CARRY"), "{e}");
}
