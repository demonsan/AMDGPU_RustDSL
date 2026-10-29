use amdgpu_compiler::mir::*;
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name};
use std::collections::BTreeMap;

#[test]
fn mov_census() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    std::env::set_var("AMDC_SCHED", "");
    let (m, a) = compile_mir_alloc(&f).unwrap();
    let idx: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut in_loop = vec![false; m.blocks.len()];
    for (h, l) in &m.loops {
        for b in idx[h].min(idx[l])..=idx[h].max(idx[l]) { in_loop[b] = true; }
    }
    let mut cat: BTreeMap<String, (u32, u32)> = BTreeMap::new(); // (static total, static in loops)
    let mut total_valu = (0u32, 0u32);
    for (bi, b) in m.blocks.iter().enumerate() {
        for i in &b.insts {
            let valu = !matches!(i.op, MOp::SNop | MOp::SWaitVm | MOp::SWaitLgkm | MOp::SBarrier | MOp::SLoadDword | MOp::SLoadDwordx2 | MOp::SReadPhys | MOp::SMovB32 | MOp::SAddU32 | MOp::SMulI32 | MOp::SShlB32 | MOp::SSetPrio | MOp::SEndpgm | MOp::SAddcU32 | MOp::SLshrB32 | MOp::SAndB32 | MOp::SXorB32 | MOp::SCmpLtI32) && format!("{:?}", i.op).starts_with('V');
            if valu { total_valu.0 += 1; if in_loop[bi] { total_valu.1 += 1; } }
            let key = match i.op {
                MOp::VWriteSub => "VWriteSub (fragment lane write)".to_string(),
                MOp::VMovFromS => "VMovFromS (sgpr->vgpr)".to_string(),
                MOp::VReadSub => "VReadSub".to_string(),
                MOp::VMovB32 => {
                    if i.srcs.is_empty() { "VMovB32 imm (constant)".to_string() }
                    else if let (Some(d), Some(Src::Reg(s))) = (i.dst, i.srcs.first()) {
                        let same = a.phys[m.canon(d) as usize] + m.alias.get(&d).map(|x| x.1 as u32).unwrap_or(0) == a.phys[m.canon(*s) as usize] + m.alias.get(s).map(|x| x.1 as u32).unwrap_or(0);
                        let dalias = m.alias.contains_key(&d);
                        let salias = m.alias.contains_key(s);
                        format!("VMovB32 reg->reg [{}{}{}]", if dalias { "dst=lane " } else { "dst=plain " }, if salias { "src=lane " } else { "src=plain " }, if same { "SELF-MOVE" } else { "" })
                    } else { "VMovB32 other".to_string() }
                }
                _ => continue,
            };
            let e = cat.entry(key).or_default();
            e.0 += 1;
            if in_loop[bi] { e.1 += 1; }
        }
    }
    println!("MOV total VALU-class instrs: {} (in loops {}); loops blocks: {}", total_valu.0, total_valu.1, in_loop.iter().filter(|x| **x).count());
    for (k, v) in cat { println!("MOV   {k}: total {} in-loop {}", v.0, v.1); }
}