use amdgpu_compiler::mir::*;
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name};
use std::collections::BTreeMap;

#[test]
fn census() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    let (m, _a) = compile_mir_alloc(&f).unwrap();
    let idx: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut in_loop = vec![false; m.blocks.len()];
    for (h, l) in &m.loops { for b in idx[h].min(idx[l])..=idx[h].max(idx[l]) { in_loop[b] = true; } }
    let mut by: BTreeMap<String, (u32, u32)> = BTreeMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        for i in &b.insts {
            let n = format!("{:?}", i.op);
            if !n.starts_with('V') { continue; }
            let e = by.entry(n).or_default();
            e.0 += 1;
            if in_loop[bi] { e.1 += 1; }
        }
    }
    let mut v: Vec<_> = by.into_iter().collect();
    v.sort_by_key(|(_, c)| std::cmp::Reverse(c.1));
    for (k, c) in v.iter().take(24) { println!("OPS {k}: total {} in-loop {}", c.0, c.1); }
    let total: u32 = v.iter().map(|(_, c)| c.1).sum();
    println!("OPS in-loop vector instrs total {total}");
}