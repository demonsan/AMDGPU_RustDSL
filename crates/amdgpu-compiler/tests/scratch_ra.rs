use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::{kernel_by_name, lower, regalloc};

#[test]
fn model_vs_alloc() {
    for k in ["fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_sv", "fmha_fwd_hd128_mxfp8_pp", "fmha_fwd_hd128_mxfp8", "fmha_fwd_hd128_mxfp8_vs", "fmha_fwd_hd128_mxfp8_vs128"] {
        let Some(f) = kernel_by_name(k) else { continue };
        for opts in [&["dce", "cmpfold", "scalarize"][..], &["dce", "cmpfold", "scalarize", "imm"][..]] {
            let m = with_opts(opts, || lower::lower(&f)).unwrap();
            let (peak, pos, _) = regalloc::pressure_profile(&m);
            let l0 = regalloc::live_at(&m, 0).len();
            match regalloc::allocate(&m) {
                Ok(a) => println!("MODEL {k} {opts:?}: pruned peak {peak} at {pos} (live at 0: {l0}); allocated max_v {}", a.max_v),
                Err(e) => println!("MODEL {k} {opts:?}: pruned peak {peak} at {pos} (live at 0: {l0}); FAILED {}", e.split("||").last().unwrap_or("").chars().take(110).collect::<String>()),
            }
        }
    }
}