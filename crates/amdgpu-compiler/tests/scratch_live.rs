use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::liveness;
use amdgpu_compiler::{kernel_by_name, lower};

#[test]
fn live_hdr() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs").unwrap();
    for opts in [&["dce", "cmpfold", "scalarize", "sgpr", "addrfold"][..], &["dce", "cmpfold", "scalarize", "sgpr", "addrfold", "imm"][..], &["dce", "cmpfold", "scalarize", "sgpr", "addrfold", "imm", "shl"][..]] {
        let m = with_opts(opts, || lower::lower(&f)).unwrap();
        let lv = liveness::compute(&m);
        let mut s = String::new();
        for (h, _l) in &m.loops {
            let bi = m.blocks.iter().position(|b| b.label == *h).unwrap();
            let (mut v1, mut vdw) = (0, 0);
            for r in lv.live_in[bi].iter() {
                let c = m.vreg_class(*r);
                if c.is_vector() { vdw += c.dwords(); }
                if c == RClass::V1 { v1 += 1; }
            }
            s += &format!(" [b{bi}: V1={v1} Vdw={vdw}]");
        }
        println!("HDR {opts:?}:{s}");
    }
}