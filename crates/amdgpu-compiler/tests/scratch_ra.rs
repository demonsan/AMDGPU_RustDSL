use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::{kernel_by_name, lower, regalloc};

#[test]
fn ghosts() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_t").unwrap();
    let m = with_opts(&["dce", "cmpfold", "scalarize"], || lower::lower(&f)).unwrap();
    let (peak, pos, _) = regalloc::pressure_profile(&m);
    let l0 = regalloc::live_at(&m, 0);
    let dw: u32 = l0.iter().map(|r| m.vreg_class(*r).dwords()).sum();
    println!("GHOST peak {peak} at {pos}; live at pos 0: {} vregs, {dw} dwords: {:?}", l0.len(), l0.iter().take(12).map(|r| format!("r{r}:{:?}", m.vreg_class(*r))).collect::<Vec<_>>());
    // dwords live at the peak split into groups by class
    let lp = regalloc::live_at(&m, pos);
    let mut by = std::collections::BTreeMap::new();
    for r in &lp { *by.entry(format!("{:?}", m.vreg_class(*r))).or_insert(0u32) += 1; }
    println!("GHOST at peak: {by:?}");
}