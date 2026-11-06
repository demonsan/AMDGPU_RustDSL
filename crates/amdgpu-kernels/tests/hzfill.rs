//! `hzfill` (post-RA hazard-window filling): on the shipped kernels it only removes wait states, keeps the MIR valid and every value's
//! physical register reaching its readers (the allocator's reaching-writer check), and leaves the hazard model with nothing to add.
use amdgpu_compiler::{compile_mir_alloc, hzfill, mir, regalloc};

#[test]
fn hzfill_keeps_the_kernels_valid() {
    for k in ["fmha_fwd_hd128_mxfp8_v3_t128", "gemm_mxfp8_bs_v4"] {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let (mut m, a) = compile_mir_alloc(&f).unwrap();
        let s = hzfill::fill(&mut m, &a, true);
        println!("{k}: {s:?}");
        assert!(s.states_after <= s.states_before, "{k}");
        assert!(s.moved > 0, "{k}: nothing filled");
        assert!(mir::verify(&m).is_empty(), "{k}: {:?}", mir::verify(&m));
        assert!(regalloc::check_reaching(&m, &a).is_empty(), "{k}: {:?}", regalloc::check_reaching(&m, &a));
        let mut c = m.clone();
        amdgpu_compiler::hazards::fix_value_hazards_opts(&mut c, true);
        amdgpu_compiler::hazards::pad_physical_war(&mut c, &a);
        let n = |x: &mir::MFunction| x.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
        assert_eq!(n(&c), n(&m), "{k}: the hazard passes would still pad");
    }
}
