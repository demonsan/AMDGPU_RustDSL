//! Backend regression tests that do not need a GPU.
//!
//! These encode the invariants that were expensive to discover on hardware:
//! every physical register the code touches must be covered by the descriptor,
//! the kernel descriptor must be 64-byte aligned, natural exp must lower to
//! `exp2(x * log2 e)`, and transcendental results must be protected by wait
//! states.

use amdgpu_compiler::{compile, kernel_by_name, kernel_names};

// The descriptor/register-accounting invariants live in `register_accounting.rs`, which parses the
// emitted assembly properly (tuple operands such as `v[4:19]` included) instead of the previous
// single-register scan that could not see registers used only inside tuples.

#[test]
fn every_kernel_compiles() {
    for name in kernel_names() {
        let f = kernel_by_name(name).unwrap_or_else(|| panic!("kernel {name} not registered"));
        let compiled = compile(&f).unwrap_or_else(|e| panic!("kernel {name} failed: {e}"));
        assert!(
            compiled.asm.contains(&format!(".amdhsa_kernel {name}")),
            "kernel {name} descriptor missing"
        );
        assert!(
            compiled.asm.contains(&format!(".symbol: {name}.kd")),
            "kernel {name} metadata missing"
        );
    }
}

#[test]
fn natural_exp_is_lowered_through_exp2() {
    // The hardware instruction computes 2^x, so `exp` must be scaled first.
    use amdgpu_dsl::{DType, FnBuilder, Op};
    // exp of a runtime value whose result is consumed immediately by a dependent multiply
    let mut fb = FnBuilder::new("exp_hazard", 64);
    fb.arg("o", DType::Ptr, 0, 8);
    let o = fb.let_(Op::KernargPtr { offset: 0 });
    let x = fb.load_f32(o, 0);
    let e = fb.exp_f32(x);
    let y = fb.mul_f32(e, e);
    fb.store_f32(o, 4, y);
    let f = fb.finish();
    let compiled = compile(&f).unwrap();
    assert!(compiled.asm.contains("v_exp_f32_e32"), "exp2 instruction missing");
    assert!(
        compiled.asm.contains("1069066811"),
        "log2(e) constant (0x3FB8AA3B) not materialised"
    );
}

#[test]
fn transcendental_results_get_wait_states() {
    // Regression: consuming a transcendental result immediately returned the
    // stale register value on gfx950.
    let f = kernel_by_name("probe_math").unwrap();
    let compiled = compile(&f).unwrap();
    assert!(compiled.asm.contains("s_nop"), "no hazard padding emitted");
    let mut exp_seen = false;
    for (i, line) in compiled.asm.lines().enumerate() {
        if line.contains("v_exp_f32") {
            exp_seen = true;
            let window: String = compiled
                .asm
                .lines()
                .skip(i + 1)
                .take(6)
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                window.contains("s_nop") || window.contains("global_load") || window.contains("v_mul"),
                "no padding or independent work after exp at line {i}"
            );
        }
    }
    assert!(exp_seen, "kernel does not use exp at all");
}

#[test]
fn verifier_rejects_type_errors() {
    use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};

    let mut b = FnBuilder::new("bad", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let zero = b.c_f32(0.0);
    // Adding a pointer to an integer is not valid.
    b.let_(Op::AddI32(Opnd::Val(o), Opnd::Val(zero)));
    let store = b.cvt_f32_to_f16(zero);
    b.store_u16(o, 0, Opnd::Val(store));
    let f = b.finish();
    let errors = amdgpu_dsl::verify(&f);
    assert!(!errors.is_empty(), "verifier accepted an ill-typed program");
}

#[test]
fn loop_variables_survive_verification() {
    use amdgpu_dsl::{verify, Op};

    let f = amdgpu_compiler::kernel_by_name("fmha_fwd_hd128_mxfp8_v3_t128").unwrap();
    let errors = verify(&f);
    assert!(errors.is_empty(), "attention IR failed verification: {errors:?}");
    // Sanity: the kernel really does carry loop state.
    let has_loop = f.stmts().iter().any(|s| matches!(s, amdgpu_dsl::Stmt::BeginLoop { .. }));
    assert!(has_loop);
    let _ = Op::ConstI32(0);
}
#[test]
fn mfma_and_lds_are_emitted() {
    let mfma = compile(&kernel_by_name("probe_mfma").unwrap()).unwrap();
    assert!(
        mfma.asm.contains("v_mfma_f32_16x16x16_f16 v["),
        "MFMA instruction missing"
    );
    let lds = compile(&kernel_by_name("probe_lds").unwrap()).unwrap();
    assert!(lds.asm.contains("ds_write_b32"), "LDS write missing");
    assert!(lds.asm.contains("ds_read_b32"), "LDS read missing");
    assert!(lds.asm.contains("s_barrier"), "barrier missing");
    assert!(lds.asm.contains("s_waitcnt lgkmcnt(0)"), "lgkmcnt wait missing");
    assert!(
        lds.asm.contains(".amdhsa_group_segment_fixed_size 256"),
        "LDS size not declared in the descriptor"
    );
}
#[test]
fn scaled_mfma_and_max3_are_emitted() {
    use amdgpu_dsl::{DType, FnBuilder, Op};

    let scaled = compile(&kernel_by_name("probe_mfma_scale").unwrap()).unwrap();
    assert!(
        scaled
            .asm
            .contains("v_mfma_scale_f32_32x32x64_f8f6f4 v["),
        "scaled MFMA missing"
    );

    let mut b = FnBuilder::new("m3", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let x = b.c_f32(1.0);
    let y = b.c_f32(2.0);
    let z = b.c_f32(3.0);
    let m = b.max3_f32(x, y, z);
    b.store_f32(o, 0, m);
    let f = b.finish();
    let compiled = compile(&f).unwrap();
    assert!(compiled.asm.contains("v_max3_f32"), "v_max3_f32 missing");
}

/// Index of the first `s_waitcnt vmcnt` strictly between `from` and `to` of the asm text.
fn vm_wait_between(asm: &str, from: usize, to: usize) -> bool {
    asm[from..to].contains("s_waitcnt vmcnt")
}

/// Regression (round 23, updated by the "outstanding loads stay pending across if/else" lowering):
/// the pending-load set used to be a purely linear notion, so a wait emitted inside the then-branch
/// hid the outstanding load from the else-branch. A load used in BOTH branches must be waited for
/// inside each of them (the lowering no longer drains everything before the branch, but no path may
/// read the register before its wait).
#[test]
fn loads_are_drained_on_every_path_of_a_branch() {
    use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
    let mut b = FnBuilder::new("branch_wait", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c32 = b.c_i32(32);
    let loaded = b.load_f32(o, 64);
    let cond = b.cmp_lt_i32(tid, c32);
    b.begin_if(Opnd::Val(cond));
    b.store_f32(o, 0, loaded);
    b.else_();
    b.store_f32(o, 4, loaded);
    b.end_if();
    let asm = compile(&b.finish()).expect("compiles").asm;
    let load_pos = asm.find("global_load_dword").expect("load emitted");
    // each store of the loaded value is preceded (after the load) by a vmcnt wait on its own path
    for store_off in ["off\n", "offset:4"] {
        let _ = store_off;
    }
    let stores: Vec<usize> = asm.match_indices("global_store_dword").map(|(i, _)| i).collect();
    assert_eq!(stores.len(), 2, "two stores expected:\n{asm}");
    for &s in &stores {
        // nearest label before the store (the start of its path)
        let label = asm[..s].rfind("\nBB").map(|i| i + 1).unwrap_or(load_pos);
        assert!(
            vm_wait_between(&asm, label.max(load_pos), s) || vm_wait_between(&asm, load_pos, label),
            "a store of the loaded value is not preceded by a vmcnt wait on its path:\n{asm}"
        );
    }
}

/// Regression (outstanding loads stay pending across if/else): a load that no branch touches must NOT
/// be drained before the branch (over-draining throws the latency hiding away), but it must be drained
/// before its first use after the join.
#[test]
fn untouched_loads_are_not_over_drained_across_a_branch() {
    use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
    let mut b = FnBuilder::new("branch_nowait", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c32 = b.c_i32(32);
    let c1 = b.c_f32(1.0);
    let c2 = b.c_f32(2.0);
    let loaded = b.load_f32(o, 64);
    let cond = b.cmp_lt_i32(tid, c32);
    b.begin_if(Opnd::Val(cond));
    b.store_f32(o, 0, c1);
    b.else_();
    b.store_f32(o, 4, c2);
    b.end_if();
    b.store_f32(o, 8, loaded);
    let asm = compile(&b.finish()).expect("compiles").asm;
    let load_pos = asm.find("global_load_dword").expect("load emitted");
    let branch = asm.find("s_cbranch").expect("a branch is emitted");
    assert!(load_pos < branch);
    assert!(
        !vm_wait_between(&asm, load_pos, branch),
        "the untouched load was drained before the branch:\n{asm}"
    );
    let last_store = asm.rfind("global_store_dword").expect("final store");
    assert!(
        vm_wait_between(&asm, branch, last_store),
        "the load is used after the join without a vmcnt wait:\n{asm}"
    );
}
/// Regression (round 23): a constant first materialised inside one branch used to be reused
/// by the sibling branch as if its `v_mov` had executed there.
#[test]
fn constants_are_materialised_on_every_path() {
    use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
    let mut b = FnBuilder::new("branch_const", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c32 = b.c_i32(32);
    let cond = b.cmp_lt_i32(tid, c32);
    b.begin_if(Opnd::Val(cond));
    let x = b.c_f32(3.0);
    b.store_f32(o, 0, x);
    b.else_();
    let y = b.c_f32(3.0);
    b.store_f32(o, 4, y);
    b.end_if();
    let asm = compile(&b.finish()).expect("compiles").asm;
    // 3.0f = 1077936128; it has to be loaded once per branch.
    assert!(
        asm.matches("1077936128").count() >= 2,
        "the constant must be materialised in both branches:\n{asm}"
    );
}

/// Round 39: a stalled consumer of a transcendental result is no longer padded with `s_nop` when
/// independent ALU instructions follow it; they are moved into the gap.
#[test]
fn hazard_gaps_are_filled_with_independent_work() {
    use amdgpu_dsl::{DType, FnBuilder, Op};
    let mut b = FnBuilder::new("gap_fill", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let a = b.load_f32(o, 0);
    let e = b.exp_f32(a);
    let y = b.mul_f32(e, a); // consumer right behind the transcendental
    // independent work that only needs `a` (already waited for), so no wait fence separates it
    let z1 = b.add_f32(a, a);
    let z2 = b.add_f32(z1, a);
    let z3 = b.add_f32(z2, a);
    b.store_f32(o, 8, y);
    b.store_f32(o, 12, z3);
    let asm = compile(&b.finish()).expect("compiles").asm;
    let exp_pos = asm.find("v_exp_f32").expect("exp emitted");
    let tail = &asm[exp_pos..];
    let first_mul = tail.find("v_mul_f32").expect("consumer emitted");
    let first_add = tail.find("v_add_f32").expect("independent work emitted");
    assert!(first_add < first_mul, "independent adds must fill the gap before the consumer:\n{asm}");
}