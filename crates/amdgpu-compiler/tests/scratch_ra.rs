use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::{kernel_by_name, lower, regalloc};

#[test]
fn model_vs_alloc() {
    for k in ["fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_sv", "fmha_fwd_hd128_mxfp8_pp", "fmha_fwd_hd128_mxfp8", "fmha_fwd_hd128_mxfp8_vs", "fmha_fwd_hd128_mxfp8_vs128"] {
        let Some(f) = kernel_by_name(k) else { continue };
        let m = with_opts(&["dce", "cmpfold", "scalarize"], || lower::lower(&f)).unwrap();
        let (peak, pos, _) = regalloc::pressure_profile(&m);
        match regalloc::allocate(&m) {
            Ok(a) => println!("MODEL {k}: true peak {peak} dwords at {pos}; allocated max_v {}", a.max_v),
            Err(_) => println!("MODEL {k}: true peak {peak} dwords at {pos}; allocation FAILED"),
        }
    }
}