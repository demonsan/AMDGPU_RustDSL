use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::{kernel_by_name, lower, regalloc};

#[test]
fn ra_try() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    for opts in [&["dce", "cmpfold", "scalarize"][..], &["dce", "cmpfold", "scalarize", "imm"][..], &["dce", "cmpfold", "scalarize", "sgpr", "addrfold", "imm"][..], &["dce", "cmpfold", "scalarize", "sgpr", "addrfold", "imm", "remat4"][..]] {
        let m = with_opts(opts, || lower::lower(&f)).unwrap();
        match regalloc::allocate(&m) {
            Ok(a) => println!("RAP {opts:?}: ALLOCATED max_v {}", a.max_v),
            Err(e) => println!("RAP {opts:?}: FAIL {}", e.split("||").last().unwrap_or("").trim()),
        }
    }
}