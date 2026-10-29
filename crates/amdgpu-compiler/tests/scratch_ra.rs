use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::liveness;
use amdgpu_compiler::{kernel_by_name, lower};
use std::collections::HashSet;

#[test]
fn precise_peak() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    for opts in [&["dce", "cmpfold", "scalarize"][..], &["dce", "cmpfold", "scalarize", "sgpr", "addrfold", "imm"][..]] {
        let m = with_opts(opts, || lower::lower(&f)).unwrap();
        let lv = liveness::compute(&m);
        let (mut peak, mut at) = (0u32, (0usize, 0usize));
        for (bi, b) in m.blocks.iter().enumerate() {
            let mut live: HashSet<VReg> = lv.live_out[bi].iter().map(|r| m.canon(*r)).collect();
            for (ii, inst) in b.insts.iter().enumerate().rev() {
                let dw: u32 = live.iter().filter(|r| m.vreg_class(**r).is_vector()).map(|r| m.vreg_class(*r).dwords()).sum();
                if dw > peak { peak = dw; at = (bi, ii); }
                if let Some(d) = inst.dst { live.remove(&m.canon(d)); }
                for r in inst.uses() { live.insert(m.canon(r)); }
            }
        }
        println!("PEAK {opts:?}: precise max live vector dwords {peak} at block {} inst {}", at.0, at.1);
    }
}