use amdgpu_compiler::mir::*;
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name};
use std::collections::BTreeMap;

#[test]
fn mov_prov() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    let (m, _a) = compile_mir_alloc(&f).unwrap();
    let idx: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut in_loop = vec![false; m.blocks.len()];
    for (h, l) in &m.loops { for b in idx[h].min(idx[l])..=idx[h].max(idx[l]) { in_loop[b] = true; } }
    let mut by: BTreeMap<String, u32> = BTreeMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        if !in_loop[bi] { continue; }
        for (ii, i) in b.insts.iter().enumerate() {
            if !matches!(i.op, MOp::VMovFromS) && !(i.op == MOp::VMovB32 && i.srcs.is_empty()) { continue; }
            let d = i.dst.unwrap();
            let cd = m.canon(d);
            let mut cons = Vec::new();
            for k in ii + 1..b.insts.len() {
                if b.insts[k].uses().iter().any(|u| m.canon(*u) == cd) { cons.push(format!("{:?}", b.insts[k].op)); if cons.len() >= 3 { break; } }
            }
            let src = match i.srcs.first() { Some(Src::Reg(r)) => { let mut p = String::from("var/outside"); for k in (0..ii).rev() { if b.insts[k].dst == Some(*r) { p = format!("{:?}", b.insts[k].op); break; } } p } _ => format!("imm {}", i.imm) };
            *by.entry(format!("{:?} src={} imm={} -> uses {:?}", i.op, if matches!(i.op, MOp::VMovFromS) { src } else { String::new() }, if i.op == MOp::VMovB32 { i.imm } else { 0 }, cons)).or_default() += 1;
        }
    }
    for (k, v) in by { println!("PROV {v:3} {k}"); }
}