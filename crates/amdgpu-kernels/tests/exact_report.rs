//! Exact mode E1 (docs/exact-mode.md): `exact::report` attributes every final instruction to the IR statement that produced it and marks
//! the ones the back end added on its own.

use amdgpu_dsl::{ev, DType, Ext, FnBuilder, Op};

#[test]
fn every_instruction_has_a_statement_or_is_marked() {
    let mut b = FnBuilder::new("probe_exact_report", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(64 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    let off = b.bind_i32(ev(tid) << 2);
    let q = b.ptr_add(po, off);
    let x = b.c_f32(2.0);
    b.store_f32(q, 0, x);
    let f = b.finish();
    let (m, a) = amdgpu_codegen::compile_mir_alloc(&f).unwrap();
    let stmts = amdgpu_codegen::lower::last_stmts();
    let r = amdgpu_codegen::exact::report(&f, &stmts, &m, &a, None);
    assert!(r.contains("stmt "), "{r}");
    assert!(r.contains("GlobalStoreF32") || r.contains("StoreF32"), "the store statement is listed:\n{r}");
    // every instruction either names a statement or is marked as added after lowering
    for b in &m.blocks {
        for i in &b.insts {
            assert!(i.stmt as usize <= stmts.len(), "stmt index in range");
            if i.stmt == 0 {
                assert!(amdgpu_codegen::exact::synthetic_kind(&m, &stmts, i).is_some());
            }
        }
    }
    assert!(r.lines().last().unwrap().contains("instructions"), "{r}");
}
