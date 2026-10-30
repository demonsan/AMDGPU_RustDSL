//! The lite fused-combine contract, checked statically on emitted assembly (own backend and LLVM tail).
use amdgpu_compiler::asm_check::check_lite_contract;
use amdgpu_compiler::backend::{LlvmTail, OwnTail};
use amdgpu_compiler::{compile_with, kernel_by_name};

fn asm_of(name: &str, backend: &str) -> String {
    let f = kernel_by_name(name).unwrap();
    if backend == "own" {
        compile_with(&f, &OwnTail).unwrap().asm
    } else {
        compile_with(&f, &LlvmTail::new()).unwrap().asm
    }
}

#[test]
fn lite_probes_satisfy_the_contract_on_both_backends() {
    for be in ["own", "llvm-tail"] {
        for k in ["probe_la8_lite", "probe_last_arriver_lite"] {
            let a = asm_of(k, be);
            check_lite_contract(&a).unwrap_or_else(|e| panic!("{k} [{be}] violates the lite contract: {e:?}"));
        }
    }
}

#[test]
fn the_checker_rejects_the_other_variants() {
    let want = [
        ("probe_la8", "buffer_wbl2"),                 // full fences: cache maintenance present
        ("probe_la8_plain", "without sc1"),            // plain shared accesses
        ("probe_la8_nodrain", "not drained"),         // no per-wave drain before the barrier / ticket
        ("probe_last_arriver_plain", "without sc1"),
    ];
    for (k, msg) in want {
        let e = check_lite_contract(&asm_of(k, "own")).expect_err(k).join("; ");
        assert!(e.contains(msg), "{k}: expected `{msg}` in {e}");
    }
}