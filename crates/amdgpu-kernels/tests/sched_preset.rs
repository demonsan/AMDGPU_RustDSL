//! `scheduler_options`: a kernel's explicit `Function::sched_opts` tokens are applied ON TOP of its profile preset, never instead of it
//! (regression: `_t128` shipped with `sched_opts = "ldsink=1"` and silently lost its preset - ALU across the barrier + the 248 pressure limit - for -1.5..-2.9% on realistic data).

use amdgpu_compiler::{policy_name, scheduler_options};
use amdgpu_kernels::kernel_by_name;

fn opts(name: &str, tokens: Option<&str>) -> amdgpu_compiler::sched::Options {
    let mut f = kernel_by_name(name).unwrap_or_else(|| panic!("{name}"));
    if let Some(t) = tokens {
        f.sched_opts = Some(t.to_string());
    }
    let pn = policy_name(&f).to_string();
    scheduler_options(&f, &pn).unwrap_or_else(|| panic!("{name}: no scheduler options"))
}

#[test]
fn t128_keeps_its_preset_when_it_sets_sched_opts() {
    let o = opts("fmha_fwd_hd128_mxfp8_v3_t128", None); // ships with ldsink=1
    assert!(o.alu_across_barrier, "ALU across the barrier is the _t128 preset");
    assert_eq!(o.max_vgprs, 248, "the 248 pressure limit is the _t128 preset");
    assert_eq!(o.ldsink, 1, "the kernel's own token is applied on top");
    // any other token composes with the preset too
    let o = opts("fmha_fwd_hd128_mxfp8_v3_t128", Some("hoistchain"));
    assert!(o.alu_across_barrier && o.max_vgprs == 248 && o.hoist_chain && o.ldshoist);
}

#[test]
fn vs128_and_kvsplit_presets_are_the_default_options_plus_their_tokens() {
    for k in ["fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_kvsplit"] {
        let o = opts(k, None);
        assert!(o.reorder && !o.alu_across_barrier && o.max_vgprs == 240 && o.ldsink == 1, "{k}");
    }
}

#[test]
fn the_gemm_profile_preset_survives_its_own_tokens() {
    let o = opts("gemm_mxfp8_bs_v4", None);
    assert!(!o.reorder && o.ldshoist, "GEMM preset: scoreboard waits + ds_read hoisting, no reordering");
    let o = opts("gemm_mxfp8_bs_v4", Some("hoistchain"));
    assert!(!o.reorder && o.ldshoist && o.hoist_chain);
}

#[test]
fn a_kernel_without_a_preset_gets_the_default_options_plus_its_tokens() {
    let o = opts("probe_mfma", Some("ldsink=2"));
    assert!(o.reorder && o.ldsink == 2);
}