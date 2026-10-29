use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::{kernel_by_name, lower, regalloc};
use std::collections::BTreeMap;

#[test]
fn peak_composition() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    for opts in [&["dce", "cmpfold", "scalarize", "imm"][..], &["dce", "cmpfold", "scalarize", "imm", "ldsspill64"][..]] {
        let m = with_opts(opts, || lower::lower(&f)).unwrap();
        let (peak, pos, blk) = regalloc::pressure_profile(&m);
        let live = regalloc::live_at(&m, pos);
        let mut by: BTreeMap<String, (u32, u32)> = BTreeMap::new();
        for r in &live {
            let c = m.vreg_class(*r);
            // defined in a loop block?
            let mut loopdef = false;
            let idx: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
            for (h, l) in &m.loops {
                let (hi, li) = (idx[h], idx[l]);
                for b in hi.min(li)..=hi.max(li) {
                    if m.blocks[b].insts.iter().any(|i| i.dst == Some(*r)) { loopdef = true; }
                }
            }
            let e = by.entry(format!("{c:?}{}", if loopdef { " (defined in loop)" } else { " (outside loop)" })).or_default();
            e.0 += 1;
            e.1 += c.dwords();
        }
        println!("COMP {opts:?}: peak {peak} at pos {pos} block {blk}: {by:?}");
    }
}