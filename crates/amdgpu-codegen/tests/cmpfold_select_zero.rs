//! cmpfold must not take a select with the inline-constant-0 false arm (`imm`: `v_cndmask d, 0, x`) for a 0/1 compare
//! materialisation. With `v = c ? x : 0` (x a per-lane value that can be 0) and a later `v != 0`, the fold re-issued the select's
//! compare `c` in place of `v != 0`: wrong for the lanes where c holds and x is 0. topk_per_row_decode was miscompiled this way
//! with `imm` (all rows wrong).
use amdgpu_codegen::mir::{MOp, Src};
use amdgpu_dsl::{CmpPred, CmpTy, DType, FnBuilder, Function, Op, Opnd};

fn kernel() -> Function {
    let mut b = FnBuilder::new("probe_cmpfold_select_zero", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg_tensor("p", amdgpu_dsl::Ext::c(64 * 4));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let five = b.c_i32(5);
    let one = b.c_i32(1);
    // c = tid < 5 (per lane), x = tid & 1 (0 on even lanes), v = c ? x : 0
    let c = b.let_(Op::CmpLtI32(Opnd::Val(tid), Opnd::Val(five)));
    let x = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(one)));
    let v = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c), a: Opnd::Val(x), b: Opnd::I32(0) });
    // z = v != 0 ; w = z ? 7 : 9   (correct: 7 on lanes 1 and 3 only)
    let z = b.let_(Op::Cmp { pred: CmpPred::Ne, ty: CmpTy::I32, a: Opnd::Val(v), b: Opnd::I32(0) });
    let w = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(z), a: Opnd::I32(7), b: Opnd::I32(9) });
    let four = b.c_i32(4);
    let off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(four)));
    let q = b.ptr_add(p, off);
    let wf = b.let_(Op::BitcastF32(Opnd::Val(w)));
    b.store_f32(q, 0, wf);
    b.finish()
}

#[test]
fn a_select_with_an_inline_zero_arm_is_not_refolded_into_its_compare() {
    let f = kernel();
    let m = amdgpu_codegen::lower::with_opts(&["dce", "cmpfold", "imm"], || amdgpu_codegen::lower::lower(&f)).expect("lowers");
    // the select v = c ? x : 0 with the inline 0 arm
    let sel: Vec<_> = m.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.op == MOp::VCndmask && matches!(i.srcs.first(), Some(Src::Imm(0)))).filter_map(|i| i.dst).collect();
    assert!(!sel.is_empty(), "the imm select form `v_cndmask d, 0, x` is expected:\n{m:?}");
    // its value must still be tested against zero (`v != 0`), not replaced by the compare `c`
    let tested = m.blocks.iter().flat_map(|b| &b.insts).any(|i| i.op == MOp::VCmpNeI32 && matches!(i.srcs.first(), Some(Src::Reg(r)) if sel.contains(r)));
    assert!(tested, "cmpfold re-folded `v != 0` into the select's compare (v = c ? x : 0 is not a 0/1 value):\n{m:?}");
}
