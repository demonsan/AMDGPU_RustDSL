use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::liveness;
use amdgpu_compiler::{kernel_by_name, lower};
use std::collections::BTreeMap;

#[test]
fn live_cats() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    for opts in [&["dce", "cmpfold", "scalarize"][..], &["dce", "cmpfold", "scalarize", "sgpr", "addrfold", "imm"][..]] {
        let m = with_opts(opts, || lower::lower(&f)).unwrap();
        let lv = liveness::compute(&m);
        for (h, _l) in &m.loops {
            let bi = m.blocks.iter().position(|b| b.label == *h).unwrap();
            let mut cat: BTreeMap<String, (u32, u32)> = BTreeMap::new();
            let (mut vdw, mut v1) = (0u32, 0u32);
            for r in lv.live_in[bi].iter() {
                let c = m.vreg_class(*r);
                if !c.is_vector() { continue; }
                vdw += c.dwords();
                if c == RClass::V1 { v1 += 1; }
                // defining op(s) outside the loop: first def found
                let mut op = String::from("loopvar/other");
                'o: for blk in m.blocks.iter() {
                    for inst in &blk.insts {
                        if inst.dst == Some(*r) {
                            op = format!("{:?}{}", inst.op, if inst.srcs.is_empty() { "()" } else if inst.srcs.iter().all(|s| matches!(s, Src::Imm(_)) || matches!(s, Src::Reg(x) if !m.vreg_class(*x).is_vector())) { "(const/scalar srcs)" } else { "(vector srcs)" });
                            break 'o;
                        }
                    }
                }
                let e = cat.entry(format!("{:?} {}", c, op)).or_default();
                e.0 += 1;
                e.1 += c.dwords();
            }
            println!("CATS {opts:?} hdr b{bi}: V1={v1} Vdw={vdw}");
            for (k, v) in cat { println!("CATS    {k}: n={} dw={}", v.0, v.1); }
        }
    }
}