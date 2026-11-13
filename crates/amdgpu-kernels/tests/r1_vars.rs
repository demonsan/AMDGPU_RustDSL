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

/// (s) A descriptor assembled from SGPR words of a kernarg pointer (the vendor's authored build: base lo, base hi & 0xffff, num_records,
/// format), read by a buffer load: the words are in-place views / s_movs into one SGPR quad, the load reads that quad.
fn rsrc_words_kernel() -> Function {
    let mut b = FnBuilder::new("s_words", 64);
    // uniform word arithmetic in SALU (exact mode: the PinS'd words)
    b.lower_opts(&["scalarize"]);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg("n", DType::I32, 0x08, 4);
    b.arg_tensor("o", Ext::c(4096));
    let p = b.let_(Op::KernargPtr { offset: 0x00 });
    let n = b.let_(Op::KernargI32 { offset: 0x08 });
    let lo = b.let_(Op::PtrLo(Opnd::Val(p)));
    let hi = b.let_(Op::PtrHi(Opnd::Val(p)));
    let m = b.c_i32(0xffff);
    let hi2 = b.let_(Op::AndI32(Opnd::Val(hi), Opnd::Val(m)));
    let fmt = b.c_i32(0x20000);
    let r = b.let_(Op::RsrcCat { words: [Opnd::Val(lo), Opnd::Val(hi2), Opnd::Val(n), Opnd::Val(fmt)] });
    let tid = b.let_(Op::WorkitemId);
    let off = b.bind_i32(amdgpu_dsl::ev(tid) << 2);
    let x = b.let_(Op::BufferLoad { rsrc: Opnd::Val(r), voff: Some(Opnd::Val(off)), soff: Opnd::I32(0), imm: 0, dwords: 1 });
    let q = b.let_(Op::PtrCat { lo: Opnd::Val(lo), hi: Opnd::Val(hi) });
    let a = b.let_(Op::PtrAdd { base: Opnd::Val(q), byte_off: Opnd::Val(off) });
    b.store_f32(a, 0, x);
    b.finish()
}

#[test]
fn descriptor_from_words_is_one_sgpr_quad() {
    let f = rsrc_words_kernel();
    let m = amdgpu_codegen::lower::with_opts(&["scalarize"], || amdgpu_codegen::lower::lower(&f)).unwrap_or_else(|e| panic!("{e}"));
    let insts: Vec<_> = m.blocks.iter().flat_map(|b| b.insts.iter()).collect();
    let quad = insts.iter().find(|i| i.op == amdgpu_codegen::mir::MOp::BufferLoadDwordX1).and_then(|i| i.srcs.iter().find_map(|s| if let amdgpu_codegen::mir::Src::Reg(r) = s { Some(*r) } else { None })).expect("a buffer load");
    assert_eq!(m.vreg_class(quad), amdgpu_codegen::mir::RClass::S4, "the descriptor is one SGPR quad");
    let words = insts.iter().filter(|i| i.op == amdgpu_codegen::mir::MOp::SMovB32).count();
    assert!(words >= 4, "the four words are written into the quad ({words} s_mov)");
    // the round trip through the irv1 text keeps the forms
    let txt = amdgpu_dsl::ir_text::print(&f);
    assert!(txt.contains("RsrcCat") && txt.contains("PtrLo") && txt.contains("PtrCat"), "{txt}");
    let g = amdgpu_dsl::ir_text::parse(&txt).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(amdgpu_dsl::ir_text::print(&g), txt);
}