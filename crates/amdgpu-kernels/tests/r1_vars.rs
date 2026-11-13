//! (r1, docs/exact-mode.md): pointer / descriptor loop and scope variables live in SGPR pairs / quads (wave-uniform), set with `set_var`.

use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd};

/// A pointer scope variable initialised from the kernarg pointer and advanced in a loop (`set_var(p, p + 256)`); a descriptor scope
/// variable over it.
fn kernel(nonuniform: bool) -> Function {
    let mut b = FnBuilder::new("r1_vars", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg("n", DType::I32, 0x08, 4);
    b.arg("o2", DType::Ptr, 0x10, 8);
    b.arg_tensor("o", Ext::c(4096));
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let n = b.let_(Op::KernargI32 { offset: 0x08 });
    let vars = b.declare_vars(&[(DType::Ptr, Opnd::Val(po))]);
    let p = vars[0];
    let lv = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let it = lv[0];
    let step = if nonuniform { b.let_(Op::WorkitemId) } else { b.c_i32(256) };
    // (reading the variable as a pointer operand is (r2), backend-llvm-tail, and a uniform PtrAdd is an SGPR pair in exact mode only;
    // here the variable is re-set from another kernarg pointer, and from a lane-varying pointer in the rejected variant)
    let q = if nonuniform { b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(step) }) } else { b.let_(Op::KernargPtr { offset: 0x10 }) };
    b.set_var(p, q);
    let it1 = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    b.set_var(it, it1);
    let c = b.cmp(amdgpu_dsl::CmpPred::Ge, amdgpu_dsl::CmpTy::I32, it1, n);
    b.break_if(c);
    b.end_loop();
    b.finish()
}

#[test]
fn pointer_variable_is_an_sgpr_pair() {
    let m = amdgpu_codegen::compile_mir(&kernel(false)).unwrap_or_else(|e| panic!("{e}"));
    let smov = m.blocks.iter().flat_map(|b| b.insts.iter()).filter(|i| i.op == amdgpu_codegen::mir::MOp::SMovB32).count();
    assert!(smov >= 2, "the pointer variable is copied dword by dword in SGPRs");
}

#[test]
fn non_uniform_pointer_variable_is_rejected() {
    let e = amdgpu_codegen::compile_mir(&kernel(true)).err().expect("a lane-varying pointer variable must be rejected");
    assert!(e.contains("must be wave-uniform") || e.contains("pointer"), "{e}");
}
