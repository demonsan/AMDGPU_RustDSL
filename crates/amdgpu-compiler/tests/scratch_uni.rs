use amdgpu_compiler::mir::*;
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name};
use std::collections::{BTreeMap, HashMap, HashSet};

#[test]
fn uniform_census() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    let (m, _a) = compile_mir_alloc(&f).unwrap();
    let idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut in_loop = vec![false; m.blocks.len()];
    for (h, l) in &m.loops { for b in idx[h].min(idx[l])..=idx[h].max(idx[l]) { in_loop[b] = true; } }
    // lane-uniform vregs: fixpoint. Scalar regs, imm movs, VMovFromS, and pure integer VALU ops whose sources are all uniform.
    let pure_int = |op: MOp| matches!(op, MOp::VMovB32 | MOp::VAddU32 | MOp::VMulLoU32 | MOp::VAndB32 | MOp::VOrB32 | MOp::VXorB32 | MOp::VLshrB32 | MOp::VLshlB32 | MOp::VMovFromS);
    let mut defs: HashMap<VReg, Vec<MInst>> = HashMap::new();
    for b in &m.blocks { for i in &b.insts { if let Some(d) = i.dst { defs.entry(m.canon(d)).or_default().push(i.clone()); } } }
    let mut uni: HashSet<VReg> = HashSet::new();
    loop {
        let mut ch = false;
        for (r, ds) in &defs {
            if uni.contains(r) { continue; }
            let c = m.vreg_class(*r);
            let ok = if !c.is_vector() { true } else { m.alias.get(r).is_none() && ds.iter().all(|i| pure_int(i.op) && i.srcs.iter().all(|s| match s { Src::Imm(_) => true, Src::Reg(x) => !m.vreg_class(*x).is_vector() || uni.contains(&m.canon(*x)) })) };
            if ok { uni.insert(*r); ch = true; }
        }
        if !ch { break; }
    }
    let mut by: BTreeMap<String, (u32, u32, u32)> = BTreeMap::new(); // (in-loop total, uniform-dst, lane-dep)
    for (bi, b) in m.blocks.iter().enumerate() {
        if !in_loop[bi] { continue; }
        for i in &b.insts {
            if !pure_int(i.op) { if !matches!(i.op, MOp::VCmpLtI32 | MOp::VCmpGeI32 | MOp::VCndmask | MOp::VCmpNeI32 | MOp::VPtrAdd | MOp::VAddCoU32 | MOp::VAddcCoU32) { continue; } }
            let Some(d) = i.dst.map(|d| m.canon(d)).or(None) else { 
                // compares: uniform if both sources uniform
                let u = i.srcs.iter().all(|s| match s { Src::Imm(_) => true, Src::Reg(x) => !m.vreg_class(*x).is_vector() || uni.contains(&m.canon(*x)) });
                let e = by.entry(format!("{:?}", i.op)).or_default(); e.0 += 1; if u { e.1 += 1 } else { e.2 += 1 }; continue };
            let u = uni.contains(&d) && pure_int(i.op);
            let e = by.entry(format!("{:?}", i.op)).or_default();
            e.0 += 1; if u { e.1 += 1 } else { e.2 += 1 }
        }
    }
    let (mut t, mut u) = (0, 0);
    for (k, v) in &by { println!("UNI {k}: in-loop {} uniform {} lane-dep {}", v.0, v.1, v.2); t += v.0; u += v.1; }
    println!("UNI total int-class in-loop {t}, of which wave-uniform-computable {u}");
}