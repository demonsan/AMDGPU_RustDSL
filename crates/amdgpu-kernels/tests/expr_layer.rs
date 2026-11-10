//! The operator expression layer (`amdgpu_dsl::expr`) lowers to EXACTLY the ops of the manual API: the demo kernels written with operators compile to the same assembly
//! (instruction by instruction, the kernel name aside) and the same post-RA MIR as their hand-written twins, and the documented decompositions of the operators the IR has no
//! single op for are semantically right.
use amdgpu_codegen::{compile, compile_mir_alloc};
use amdgpu_kernels::{kernel_by_name};
use amdgpu_dsl::{DType, FnBuilder, Op, I32};

fn norm(asm: &str, name: &str) -> String {
    asm.replace(name, "KERNEL")
}

fn twin(manual: &str, expr: &str) {
    let (fm, fe) = (kernel_by_name(manual).unwrap(), kernel_by_name(expr).unwrap());
    // the IR statement kinds are the same ops (constants may sit elsewhere; they are materialised at their use)
    let (am, ae) = (compile(&fm).unwrap().asm, compile(&fe).unwrap().asm);
    let (sm, se) = (norm(&am, manual), norm(&ae, expr));
    assert_eq!(sm, se, "{expr} must emit the same assembly as {manual}");
    let ((mm, alm), (me, ale)) = (compile_mir_alloc(&fm).unwrap(), compile_mir_alloc(&fe).unwrap());
    assert_eq!(format!("{:?}", mm.blocks), format!("{:?}", me.blocks), "post-RA MIR of {expr} differs from {manual}");
    assert_eq!(alm.phys, ale.phys, "register allocation differs");
    assert!(sm.lines().count() > 20, "suspiciously short assembly");
}

#[test]
fn the_ids_kernel_with_operators_emits_identical_assembly_and_mir() {
    twin("probe_llvm_ids", "probe_expr_ids");
}

#[test]
fn the_kernarg_kernel_with_operators_emits_identical_assembly_and_mir() {
    twin("probe_llvm_kernarg", "probe_expr_kernarg");
}

#[test]
fn the_demo_kernels_are_registered_and_verify_strict() {
    for k in ["probe_expr_ids", "probe_expr_kernarg"] {
        assert!(amdgpu_kernels::kernel_names().contains(&k), "{k} registered");
        let f = kernel_by_name(k).unwrap();
        let r = amdgpu_codegen::gpu_verify::analyse(&f).unwrap();
        assert!(r.errors().is_empty(), "{k}: {}", r.render());
    }
}

// ---- the documented decompositions are semantically right (host evaluation of the expression tree) ----------------------------------------------------------------

fn leaf_env(a: i32, b: i32) -> impl Fn(u32) -> i32 {
    move |v| if v == 0 { a } else { b }
}

#[test]
fn derived_integer_operators_match_rust_semantics() {
    let b = FnBuilder::new("x", 64);
    let (x, y) = (b.i32v(0), b.i32v(1));
    let samples = [0, 1, -1, 2, 7, -7, 255, 256, 0x7fff_ffff, i32::MIN, i32::MIN + 1, 123456789, -987654321];
    for &p in &samples {
        for &q in &samples {
            let env = leaf_env(p, q);
            assert_eq!((&x - &y).eval(&env), p.wrapping_sub(q), "{p} - {q}");
            assert_eq!((&x - 5).eval(&env), p.wrapping_sub(5));
            assert_eq!((-&x).eval(&env), p.wrapping_neg());
            assert_eq!((&x | &y).eval(&env), p | q, "{p} | {q}");
            assert_eq!((&x ^ &y).eval(&env), p ^ q);
            assert_eq!((&x & &y).eval(&env), p & q);
            assert_eq!(x.max(&y).eval(&env), p.max(q), "max({p},{q})");
            assert_eq!(x.min(&y).eval(&env), p.min(q));
            assert_eq!((!&x).eval(&env), !p);
            assert_eq!(x.lt(&y).eval(&env), p < q);
            assert_eq!(x.ge(&y).eval(&env), p >= q);
            assert_eq!(x.gt(&y).eval(&env), p > q);
            assert_eq!(x.le(&y).eval(&env), p <= q);
            assert_eq!(x.ne(&y).eval(&env), p != q, "{p} != {q}");
            assert_eq!(x.eq(&y).eval(&env), p == q, "{p} == {q}");
            assert_eq!(x.lt(&y).select_i32(&x * 3, &y).eval(&env), if p < q { p.wrapping_mul(3) } else { q });
        }
        let env = leaf_env(p, 0);
        assert_eq!((&x << 3).eval(&env), ((p as u32) << 3) as i32);
        assert_eq!((&x >> 3).eval(&env), ((p as u32) >> 3) as i32, "`>>` is the LOGICAL shift of this IR");
    }
}

// ---- misuse is caught by the unchanged DSL verifier ------------------------------------------------------------------------------------------------------------------

#[test]
fn an_expression_over_a_value_of_the_wrong_type_is_rejected_by_the_verifier() {
    let mut b = FnBuilder::new("misuse", 64);
    let f = b.let_(Op::KernargF32 { offset: 0 });
    let bad = b.i32v(f) * 4; // an f32 value used as an i32 leaf
    let v = b.bind_i32(bad);
    let _ = v;
    let errs = amdgpu_dsl::verify(&b.finish());
    assert!(!errs.is_empty(), "the verifier must reject an i32 operation on an f32 value");
}

#[test]
fn shared_subexpressions_are_lowered_once_per_bind() {
    let mut b = FnBuilder::new("share", 64);
    b.arg("p", DType::Ptr, 0, 8);
    let t = b.let_(Op::WorkitemId);
    let t = b.i32v(t);
    let shared: I32 = &t * 4 + 8;
    let before = b.stmt_count();
    let _v = b.bind_i32(&shared + &shared); // (t*4+8) + (t*4+8): the cloned node is the same Rc, lowered once
    let after = b.stmt_count();
    // const 4, mul, const 8, add, then the outer add = 5 statements (a duplicate lowering would need 9)
    assert_eq!(after - before, 5);
}