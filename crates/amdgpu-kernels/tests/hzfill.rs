//! `hzfill` (post-RA hazard-window filling): on the shipped kernels it only removes wait states, keeps the MIR valid and every value's
//! physical register reaching its readers (the allocator's reaching-writer check), and leaves the hazard model with nothing to add.
use amdgpu_codegen::{compile_mir_alloc, hzfill, mir, regalloc};

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
        amdgpu_codegen::hazards::fix_value_hazards_opts(&mut c, true);
        amdgpu_codegen::hazards::pad_physical_war(&mut c, &a);
        let n = |x: &mir::MFunction| x.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
        assert_eq!(n(&c), n(&m), "{k}: the hazard passes would still pad");
    }
}

/// `hzfill` as a region option: a region without it keeps its nops, a region with it is filled; without any region the kernel-level token fills everything (as before).
#[test]
fn hzfill_is_a_region_option() {
    let k = "fmha_fwd_hd128_mxfp8_v3_t128";
    let f = amdgpu_kernels::kernel_by_name(k).unwrap();
    let (m0, a0) = compile_mir_alloc(&f).unwrap();
    let all = {
        let mut m = m0.clone();
        hzfill::fill(&mut m, &a0, true).moved
    };
    assert!(all > 0);
    // no region has hzfill enabled: nothing moves
    let mut o = amdgpu_codegen::sched::Options::default();
    o.regions = vec![amdgpu_codegen::sched::region::RegionSpec { name: "r".into(), authored: true, tokens: String::new() }];
    let mut m = m0.clone();
    let none = hzfill::fill_with(&mut m, &a0, true, Some(&o)).moved;
    assert_eq!(none, 0, "region 0 and region 1 both lack hzfill");
    // every instruction in region 1, and region 1 enables hzfill: the same fills as without regions
    o.regions[0].tokens = "hzfill".into();
    let mut m = m0.clone();
    for b in m.blocks.iter_mut() {
        for i in b.insts.iter_mut() {
            i.region = 1;
        }
    }
    let some = hzfill::fill_with(&mut m, &a0, true, Some(&o)).moved;
    assert_eq!(some, all, "a region that enables hzfill fills as the kernel-level token does");
}