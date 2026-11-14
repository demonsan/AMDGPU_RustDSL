//! Peephole preconditions (docs/peephole-audit.md): adversarial kernels that have the SHAPE of a fold but not its semantics. Each fold must
//! check the semantic precondition (provenance of the operands), not just the instruction pattern.

use amdgpu_dsl::{CmpPred, CmpTy, DType, Ext, FnBuilder, Op, Opnd};

fn asm(f: &amdgpu_dsl::Function) -> String {
    amdgpu_codegen::compile(f).unwrap_or_else(|e| panic!("{e}")).asm
}

/// a = r + 4 is defined BEFORE r is advanced; the LDS store at `a` must use the old r: the add cannot fold into the DS offset of the
/// variable's (new) register.
#[test]
fn an_lds_address_snapshot_of_a_variable_is_not_folded_past_its_update() {
    let mut b = FnBuilder::new("pp_lds_snapshot", 64);
    b.lds(64 * 1024);
    let tid = b.let_(Op::WorkitemId);
    let a0 = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(4)));
    let v = b.begin_loop(&[(DType::I32, Opnd::Val(a0)), (DType::I32, Opnd::I32(0))]);
    let (r, i) = (v[0], v[1]);
    let a = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(4)));
    let r2 = b.let_(Op::AddI32(Opnd::Var(r), Opnd::I32(1024)));
    b.set_var(r, r2);
    b.store_lds_b32(a, Opnd::Val(tid));
    let i2 = b.let_(Op::AddI32(Opnd::Var(i), Opnd::I32(1)));
    b.set_var(i, i2);
    let done = b.let_(Op::Cmp { pred: CmpPred::Ge, ty: CmpTy::I32, a: Opnd::Val(i2), b: Opnd::I32(8) });
    b.break_if(done);
    b.end_loop();
    let s = asm(&b.finish());
    let store = s.lines().find(|l| l.trim_start().starts_with("ds_write_b32")).expect("the store");
    assert!(!store.contains("offset:4"), "the snapshot r + 4 was folded into the store at the updated r: {store}\n{s}");
}

/// Two LDS-DMAs whose LDS base is a variable advanced between them: each needs its own readfirstlane (no cached copy of the old value).
#[test]
fn a_variable_dma_base_is_read_again_after_its_update() {
    let mut b = FnBuilder::new("pp_dma_var", 64);
    b.arg("src", DType::Ptr, 0x00, 8);
    b.arg_tensor("src", Ext::c(1 << 20));
    b.lds(64 * 1024);
    let p = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    let v = b.begin_loop(&[(DType::I32, Opnd::Val(tid)), (DType::I32, Opnd::I32(0))]);
    let (l, i) = (v[0], v[1]);
    b.load_global_lds(p, Opnd::Var(l), 0, 1);
    let l2 = b.let_(Op::AddI32(Opnd::Var(l), Opnd::I32(4096)));
    b.set_var(l, l2);
    b.load_global_lds(p, Opnd::Var(l), 0, 1);
    let i2 = b.let_(Op::AddI32(Opnd::Var(i), Opnd::I32(1)));
    b.set_var(i, i2);
    let done = b.let_(Op::Cmp { pred: CmpPred::Ge, ty: CmpTy::I32, a: Opnd::Val(i2), b: Opnd::I32(4) });
    b.break_if(done);
    b.end_loop();
    let s = asm(&b.finish());
    let n = s.lines().filter(|l| l.contains("v_readfirstlane_b32")).count();
    assert!(n >= 2, "one M0 readfirstlane for two different bases:\n{s}");
}

/// frag_cat([x, x]): x is read twice although `read_counts` counts it once; its producer may not move into slot 0 only (slot 1 would copy
/// from a register nothing writes).
#[test]
fn a_part_given_twice_keeps_its_producer() {
    let mut b = FnBuilder::new("pp_cat_twice", 64);
    b.lds(64 * 1024);
    let tid = b.let_(Op::WorkitemId);
    let f = b.let_(Op::BitcastF32(Opnd::Val(tid)));
    let x = b.let_(Op::MulF32(Opnd::Val(f), Opnd::Val(f)));
    let pair = b.frag_cat(&[Opnd::Val(x), Opnd::Val(x)]);
    let a = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(3)));
    b.store_lds_b64(a, pair);
    let fun = b.finish();
    let (m, al) = amdgpu_codegen::compile_mir_alloc(&fun).unwrap_or_else(|e| panic!("{e}"));
    let v = amdgpu_codegen::regalloc::check_reaching(&m, &al);
    assert!(v.is_empty(), "a part reads a register its value never reaches: {v:?}");
}

/// remat4 must not clone an expression into a block that changes EXEC (the clone would run under the narrow EXEC and be read under the
/// full one): the narrow-EXEC block keeps no rematerialised copy.
#[test]
fn remat_does_not_clone_under_a_narrow_exec() {
    let mut b = FnBuilder::new("pp_remat_exec", 64);
    b.lds(64 * 1024);
    b.lower_opts(&["remat4"]);
    let tid = b.let_(Op::WorkitemId);
    let e = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(5)));
    let v = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    b.exec_save(20);
    b.exec_set(0xffff);
    b.store_lds_b32(e, Opnd::Val(tid));
    b.exec_restore(20);
    let a2 = b.let_(Op::AddI32(Opnd::Val(e), Opnd::I32(4)));
    b.store_lds_b32(a2, Opnd::Val(tid));
    let i2 = b.let_(Op::AddI32(Opnd::Var(v[0]), Opnd::I32(1)));
    b.set_var(v[0], i2);
    let done = b.let_(Op::Cmp { pred: CmpPred::Ge, ty: CmpTy::I32, a: Opnd::Val(i2), b: Opnd::I32(4) });
    b.break_if(done);
    b.end_loop();
    let s = asm(&b.finish());
    // the shift (tid << 5) is computed once, before the loop
    let shifts = s.lines().filter(|l| l.contains("v_lshlrev_b32") && l.contains(", 5,")).count();
    assert!(shifts <= 1, "the expression was rematerialised in the EXEC-changing loop block:\n{s}");
}
