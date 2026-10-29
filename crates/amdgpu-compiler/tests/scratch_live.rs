use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::liveness;
use amdgpu_compiler::{kernel_by_name, lower};

#[test]
fn live_at_loop_headers() {
    for k in ["fmha_fwd_hd128_mxfp8_t", "fmha_fwd_hd128_mxfp8_sv", "fmha_fwd_hd128_mxfp8_pp"] {
        let f = kernel_by_name(k).unwrap();
        for opts in [&["dce", "cmpfold"][..], &["dce", "cmpfold", "scalarize", "sgpr"][..]] {
            let m = with_opts(opts, || lower::lower(&f)).unwrap();
            let lv = liveness::compute(&m);
            for (h, _l) in &m.loops {
                let bi = m.blocks.iter().position(|b| b.label == *h).unwrap();
                let mut by: std::collections::BTreeMap<String, (u32, u32)> = Default::default();
                for r in lv.live_in[bi].iter() {
                    let c = m.vreg_class(*r);
                    let e = by.entry(format!("{c:?}")).or_default();
                    e.0 += 1;
                    e.1 += c.dwords();
                }
                let vdw: u32 = by.iter().filter(|(k, _)| k.starts_with('V')).map(|(_, v)| v.1).sum();
                println!("{k} {opts:?} loop header b{bi}: V dwords live-in {vdw}  {by:?}");
            }
        }
    }
}