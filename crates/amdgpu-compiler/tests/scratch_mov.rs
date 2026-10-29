use amdgpu_compiler::mir::*;
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name};
use std::collections::BTreeMap;

#[test]
fn mov_prov() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    std::env::set_var("AMDC_SCHED", "");
    let (m, _a) = compile_mir_alloc(&f).unwrap();
    let idx: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut in_loop = vec![false; m.blocks.len()];
    for (h, l) in &m.loops { for b in idx[h].min(idx[l])..=idx[h].max(idx[l]) { in_loop[b] = true; } }
    let mut by: BTreeMap<String, u32> = BTreeMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        if !in_loop[bi] { continue; }
        for (ii, i) in b.insts.iter().enumerate() {
            let is_mov = matches!(i.op, MOp::VMovB32 | MOp::VWriteSub | MOp::VMovFromS);
            if !is_mov { continue; }
            // producer of the source within the block, and distance
            let src = i.srcs.first().and_then(|s| if let Src::Reg(r) = s { Some(*r) } else { None });
            let mut prod = String::from("none/outside-block");
            if let Some(sr) = src {
                for k in (0..ii).rev() {
                    if b.insts[k].dst == Some(sr) { prod = format!("{:?} dist {}", b.insts[k].op, ii - k); break; }
                }
            }
            // consumers of the destination
            let d = i.dst;
            let mut cons = String::new();
            if let Some(d) = d {
                let cd = m.canon(d);
                for k in ii + 1..b.insts.len() {
                    if b.insts[k].uses().iter().any(|u| m.canon(*u) == cd) { cons = format!("{:?}", b.insts[k].op); break; }
                }
            }
            let dstkind = d.map(|d| if m.alias.contains_key(&d) { format!("lane of {:?}", m.vreg_class(m.canon(d))) } else { "plain".into() }).unwrap_or_default();
            *by.entry(format!("{:?} <- [{prod}] -> next use by {cons} ({dstkind})", i.op)).or_default() += 1;
        }
    }
    for (k, v) in by { println!("PROV {v:3} {k}"); }
}