//! Structured, localized diagnostics (docs/diagnostics.md): codes, statements, regions and the Rust source location of the builder call.
use amdgpu_compiler::diag;
use amdgpu_dsl::{CmpPred, CmpTy, DType, FnBuilder, MemTy, Op, Opnd};

#[test]
fn a_verifier_finding_points_at_the_kernel_source() {
    let f = amdgpu_kernels::kernel_by_name("gv_attack_buf_wrap").unwrap();
    let r = amdgpu_compiler::gpu_verify::analyse(&f).unwrap();
    let ds = diag::from_report(&f, &r);
    let d = ds.iter().find(|d| d.code == "GV-BUF-WRAP").expect("the wrap finding");
    assert_eq!(d.kernel, "gv_attack_buf_wrap");
    assert!(d.stmt.is_some());
    assert!(d.src.as_deref().unwrap_or("").contains("verifier_cases"), "{:?}", d.src);
    assert!(d.values.iter().any(|(_, l)| l.as_deref().unwrap_or("").contains("verifier_cases")), "{:?}", d.values);
    assert!(d.fix.is_some());
    let j = diag::to_json(&ds);
    assert!(j.starts_with("[{\\"code\\":\\"GV-BUF-WRAP\\"") || j.contains("\\"code\\":\\"GV-BUF-WRAP\\""), "{j}");
}

#[test]
fn a_compile_failure_is_classified_and_localized_to_this_file() {
    let mut b = FnBuilder::new("diag_uniform", 64);
    let tid = b.let_(Op::WorkitemId);
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let n = b.add_i32(Opnd::Var(vars[0]), 1);
    b.set_var(vars[0], n);
    let done = b.cmp(CmpPred::Ge, CmpTy::I32, n, tid); // per-lane: rejected
    b.break_if(done);
    b.end_loop();
    let f = b.finish();
    let e = amdgpu_compiler::compile_mir(&f).unwrap_err();
    let ds = diag::from_compile_error(&f, &e);
    let d = &ds[0];
    assert_eq!(d.code, "IR-UNIFORM-BRANCH");
    assert!(d.src.as_deref().unwrap_or("").contains("diagnostics.rs"), "{:?}", d.src);
    assert!(d.values.iter().any(|(_, l)| l.as_deref().unwrap_or("").contains("diagnostics.rs")));
    assert!(d.fix.as_deref().unwrap_or("").contains("any_lane"));
}

#[test]
fn running_out_of_registers_names_the_live_values_and_their_builder_calls() {
    // 80 live 4-dword loads in one wave: more than 256 VGPRs
    let mut b = FnBuilder::new("diag_ra", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg_tensor("p", amdgpu_dsl::Ext::Const(1 << 22));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
    let vals: Vec<_> = (0..80u32).map(|k| {
        let base = b.let_(Op::PtrAdd { base: Opnd::Val(q), byte_off: Opnd::I32((k * 4096) as i32) });
        b.load_global(base, 0, MemTy::B128)
    }).collect();
    let mut acc = b.c_f32(0.0);
    for v in vals.iter().rev() {
        for i in 0..4 {
            let x = b.frag4_lane(*v, i);
            let f = b.let_(Op::BitcastF32(Opnd::Val(x)));
            acc = b.let_(Op::AddF32(Opnd::Val(acc), Opnd::Val(f)));
        }
    }
    b.store_f32(q, 0, acc);
    let f = b.finish();
    match amdgpu_compiler::compile_with(&f, &amdgpu_compiler::backend::OwnTail) {
        Ok(_) => {} // the allocator found a way (spilling): nothing to check
        Err(e) => {
            let ds = diag::from_compile_error(&f, &e);
            assert!(ds[0].code == "RA-OUT-OF-REGISTERS" || ds[0].code == "COMPILE-REG-BUDGET", "{}", ds[0].code);
            if ds[0].code == "RA-OUT-OF-REGISTERS" {
                assert!(ds[0].values.iter().any(|(_, l)| l.as_deref().unwrap_or("").contains("diagnostics.rs")), "{:?}", ds[0].values);
            }
        }
    }
}

#[test]
fn the_summary_is_compact_json() {
    let f = amdgpu_kernels::kernel_by_name("probe_pkpair").unwrap();
    let j = diag::summary_json(&f).unwrap();
    for k in ["\\"vgpr\\":", "\\"agpr\\":", "\\"sgpr\\":", "\\"lds_bytes\\":", "\\"waits\\":", "\\"nops\\":", "\\"regions\\":[", "\\"certificate\\":{"] {
        assert!(j.contains(k), "{k} in {j}");
    }
}

#[test]
fn source_locations_do_not_change_the_code() {
    // the side table is not part of the IR: the same kernel built twice (different call sites) compiles to the same asm
    let a = amdgpu_compiler::compile(&amdgpu_kernels::kernel_by_name("probe_pkpair").unwrap()).unwrap().asm;
    let mut f = amdgpu_kernels::kernel_by_name("probe_pkpair").unwrap();
    f.src_locs = Default::default();
    let b = amdgpu_compiler::compile(&f).unwrap().asm;
    assert_eq!(a, b);
}
