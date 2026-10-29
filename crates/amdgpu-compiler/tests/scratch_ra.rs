use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::{kernel_by_name, lower, regalloc};

#[test]
fn ra_try() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    for opts in [&["dce", "cmpfold", "scalarize", "imm"][..], &["dce", "cmpfold", "scalarize", "imm", "ldsspill16"][..], &["dce", "cmpfold", "scalarize", "imm", "ldsspill32"][..], &["dce", "cmpfold", "scalarize", "imm", "ldsspill64"][..], &["dce", "cmpfold", "scalarize", "sgpr", "addrfold", "imm", "ldsspill64"][..]] {
        let m = with_opts(opts, || lower::lower(&f)).unwrap();
        let (peak, pos, blk) = regalloc::pressure_profile(&m);
        match regalloc::allocate(&m) {
            Ok(a) => println!("RAP {opts:?}: ALLOCATED max_v {} lds {} (peak {peak})", a.max_v, m.lds_bytes),
            Err(e) => println!("RAP {opts:?}: FAIL peak {peak} at {pos} (block {blk}) lds {}", m.lds_bytes),
        }
    }
}