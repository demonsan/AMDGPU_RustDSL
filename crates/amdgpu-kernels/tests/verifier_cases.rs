//! The verifier attack / audit kernels (`crates/amdgpu-kernels/src/verifier_cases/`): every ATTACK must be REJECTED with its specific
//! finding, and every trust escape must be EXPOSED in the certificate's ledger (never silently certified).
use amdgpu_compiler::gpu_verify::analyse;
use amdgpu_kernels::kernel_by_name;

fn codes(k: &str) -> Vec<String> {
    let f = kernel_by_name(k).unwrap_or_else(|| panic!("{k} not registered"));
    let r = analyse(&f).unwrap_or_else(|e| panic!("{k}: {e}"));
    r.errors().iter().map(|e| e.code.to_string()).collect()
}

#[test]
fn attacks_are_rejected_with_their_finding() {
    for (k, code, min) in [
        ("gv_attack_assume_loop", "GV-ASSUME-VIOLATED", 1),
        ("gv_attack_lds_const", "GV-LDS-OOB", 1),
        ("gv_attack_desc_over", "GV-BUF-OVER", 1),
        ("gv_attack_buf_wrap", "GV-BUF-WRAP", 1),
        ("gv_attack_clamp_path", "GV-GLB-OOB", 1),
        ("probe_lds_oob", "GV-LDS-OOB", 4),
    ] {
        let c = codes(k);
        let n = c.iter().filter(|x| x.as_str() == code).count();
        assert!(n >= min, "{k}: expected at least {min} {code} error(s), got {c:?}");
    }
}

#[test]
fn trust_escapes_are_counted_in_the_ledger() {
    // assume_trusted certifies an access the loop violates: no error, but the ledger must show the trusted assumption
    let f = kernel_by_name("gv_attack_assume_trusted").unwrap();
    let r = analyse(&f).unwrap();
    assert!(r.errors().is_empty());
    assert_eq!(r.ledger().assume_trusted, 1, "the trusted assumption must be counted");
    // an unsynchronised global region made "certified" by buf_trust: the ledger exposes the region trust
    let f = kernel_by_name("gv_audit_region_trust").unwrap();
    let r = analyse(&f).unwrap();
    assert!(r.errors().is_empty());
    assert_eq!(r.ledger().region_trust, 1, "the region trust must be counted");
    // a (possibly false) host contract is a declared, counted contract - not a proof: the kernel verifies only under it
    let f = kernel_by_name("gv_attack_false_contract").unwrap();
    let r = analyse(&f).unwrap();
    assert!(r.errors().is_empty());
    assert!(!f.contract.extents.is_empty(), "the claimed extent must appear as a trusted contract in the certificate");
}

/// A masked buffer store whose sentinel stays out of range after the instruction offset is accepted (no GV-BUF-WRAP).
#[test]
fn a_large_buffer_sentinel_is_accepted() {
    assert!(codes("gv_ok_buf_sentinel").is_empty(), "{:?}", codes("gv_ok_buf_sentinel"));
}
