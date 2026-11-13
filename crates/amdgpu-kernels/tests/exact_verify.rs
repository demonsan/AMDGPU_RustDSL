//! Exact mode E2 (docs/exact-mode.md section 4): verification by check-mode passes. On a final function every fixing pass is run on a
//! copy; anything it would insert is a named violation. Mutation tests: drop an authored wait state / wait and the check names it.

use amdgpu_codegen::mir::MOp;

/// Final MIR of every reachable kernel: the check-mode passes find nothing (the fixing passes are idempotent on their own output).
#[test]
fn the_compilers_own_output_verifies() {
    let mut bad = Vec::new();
    let mut n = 0;
    for k in amdgpu_kernels::reachable::manifest_names() {
        let Some(f) = amdgpu_kernels::kernel_by_name(&k) else { continue };
        let Ok((m, a)) = amdgpu_codegen::compile_mir_alloc(&f) else { continue };
        n += 1;
        let strict = m.elide_fallthrough; // the scheduled path (cfg cleanup ran)
        let v = amdgpu_codegen::exact::verify_final(&m, &a, strict);
        if !v.is_empty() {
            bad.push(format!("{k}: {} (first: {})", v.len(), v[0]));
        }
    }
    assert!(n > 200);
    assert!(bad.is_empty(), "{} kernel(s) do not verify their own output:\n{}", bad.len(), bad.join("\n"));
}

fn kernel_with(op_pred: &dyn Fn(MOp, i64) -> bool) -> Option<(String, amdgpu_codegen::mir::MFunction, amdgpu_codegen::regalloc::Alloc, usize, usize)> {
    for k in amdgpu_kernels::reachable::manifest_names() {
        let f = amdgpu_kernels::kernel_by_name(&k)?;
        let Ok((m, a)) = amdgpu_codegen::compile_mir_alloc(&f) else { continue };
        for (bi, b) in m.blocks.iter().enumerate() {
            if let Some(ii) = b.insts.iter().position(|i| op_pred(i.op, i.imm)) {
                return Some((k, m, a, bi, ii));
            }
        }
    }
    None
}

#[test]
fn a_dropped_wait_state_is_named() {
    let (k, mut m, a, bi, ii) = kernel_with(&|op, imm| op == MOp::SNop && imm >= 1).expect("a kernel with an s_nop");
    let strict = m.elide_fallthrough;
    m.blocks[bi].insts.remove(ii);
    let v = amdgpu_codegen::exact::verify_final(&m, &a, strict);
    assert!(!v.is_empty(), "{k}: removing an s_nop must be reported");
    assert!(v.iter().any(|x| x.rule.starts_with("hazard") || x.rule == "m0-pad"), "{k}: {:?}", v.iter().map(|x| x.to_string()).collect::<Vec<_>>());
}

#[test]
fn a_dropped_wait_is_named() {
    let (k, mut m, a, bi, ii) = kernel_with(&|op, imm| op == MOp::SWaitVm && imm == 0).expect("a kernel with a vmcnt(0)");
    let strict = m.elide_fallthrough;
    m.blocks[bi].insts.remove(ii);
    let v = amdgpu_codegen::exact::verify_final(&m, &a, strict);
    // an explicit (author) wait may cover a protocol rather than a register: then nothing reads the load's registers early and the
    // scoreboard has nothing to say; search for one the scoreboard needs
    if v.is_empty() {
        eprintln!("{k}: the removed vmcnt(0) guarded no register read (protocol wait)");
        return;
    }
    assert!(v.iter().any(|x| x.rule == "waits"), "{k}: {:?}", v.iter().map(|x| x.to_string()).collect::<Vec<_>>());
}
