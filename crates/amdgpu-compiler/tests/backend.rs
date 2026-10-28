//! Backend regression tests that do not need a GPU.
//!
//! These encode the invariants that were expensive to discover on hardware:
//! every physical register the code touches must be covered by the descriptor,
//! the kernel descriptor must be 64-byte aligned, natural exp must lower to
//! `exp2(x * log2 e)`, and transcendental results must be protected by wait
//! states.

use amdgpu_compiler::{compile, kernel_by_name, kernel_names};

/// Highest physical register index used by the instruction stream.
fn max_reg(asm: &str, prefix: char) -> u32 {
    let bytes = asm.as_bytes();
    let mut max = 0u32;
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] as char == prefix {
            let prev_ok = i == 0
                || (!(bytes[i - 1] as char).is_ascii_alphanumeric() && bytes[i - 1] != b'_');
            if prev_ok {
                let mut j = i + 1;
                let mut value = 0u32;
                let mut has_digit = false;
                while j < bytes.len() && (bytes[j] as char).is_ascii_digit() {
                    value = value * 10 + (bytes[j] - b'0') as u32;
                    j += 1;
                    has_digit = true;
                }
                if has_digit {
                    max = max.max(value);
                }
            }
        }
        i += 1;
    }
    max
}

fn declared(asm: &str, directive: &str) -> u32 {
    for line in asm.lines() {
        if let Some(rest) = line.trim().strip_prefix(directive) {
            if let Ok(v) = rest.trim().parse::<u32>() {
                return v;
            }
        }
    }
    panic!("directive {directive} missing from assembly");
}

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
fn descriptor_covers_all_registers() {
    // Regression: the allocator once reported only the slots still live at the
    // end, so the descriptor under-declared the VGPR count and the kernel
    // faulted on hardware.
    for name in ["attention_fwd", "attn_debug", "probe_store", "probe_math"] {
        let f = kernel_by_name(name).unwrap();
        let compiled = compile(&f).unwrap();
        let vgpr_declared = declared(&compiled.asm, ".amdhsa_next_free_vgpr");
        let sgpr_declared = declared(&compiled.asm, ".amdhsa_next_free_sgpr");
        let vgpr_used = max_reg(&compiled.asm, 'v');
        let sgpr_used = max_reg(&compiled.asm, 's');
        assert!(
            vgpr_used < vgpr_declared,
            "{name}: code uses v{vgpr_used} but only {vgpr_declared} VGPRs are declared"
        );
        assert!(
            sgpr_used < sgpr_declared,
            "{name}: code uses s{sgpr_used} but only {sgpr_declared} SGPRs are declared"
        );
        // The kernel descriptor must start 64 bytes after the code it describes.
        assert!(compiled.asm.contains(".p2align 6\n.amdhsa_kernel"), "{name}: descriptor alignment missing");
    }
}

#[test]
fn natural_exp_is_lowered_through_exp2() {
    // The hardware instruction computes 2^x, so `exp` must be scaled first.
    let f = kernel_by_name("probe_math").unwrap();
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
    let f = kernel_by_name("attention_fwd").unwrap();
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

    let f = amdgpu_dsl::kernels::attention_fwd();
    let errors = verify(&f);
    assert!(errors.is_empty(), "attention IR failed verification: {errors:?}");
    // Sanity: the kernel really does carry loop state.
    let has_loop = f.stmts.iter().any(|s| matches!(s, amdgpu_dsl::Stmt::BeginLoop { .. }));
    assert!(has_loop);
    let _ = Op::ConstI32(0);
}
