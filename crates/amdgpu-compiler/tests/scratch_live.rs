use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::liveness;
use amdgpu_compiler::{kernel_by_name, lower};
use std::collections::HashMap;

fn pure(op: MOp) -> bool {
    matches!(op, MOp::VMovB32 | MOp::VMovFromS | MOp::VAddU32 | MOp::VMulLoU32 | MOp::VAndB32 | MOp::VOrB32 | MOp::VXorB32 | MOp::VLshrB32 | MOp::VLshlB32 | MOp::VReadPhys)
}

#[test]
fn remat_potential() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    let m = with_opts(&["dce", "cmpfold", "scalarize", "imm"], || lower::lower(&f)).unwrap();
    let lv = liveness::compute(&m);
    let mut def: HashMap<VReg, (usize, usize)> = HashMap::new();
    let mut nd: HashMap<VReg, u32> = HashMap::new();
    for (b, blk) in m.blocks.iter().enumerate() {
        for (i, inst) in blk.insts.iter().enumerate() {
            if let Some(d) = inst.dst { *nd.entry(d).or_insert(0) += 1; def.insert(d, (b, i)); }
        }
    }
    // cost of rematerialising r from tid/SGPR/const leaves, None if impossible
    fn cost(m: &MFunction, def: &HashMap<VReg, (usize, usize)>, nd: &HashMap<VReg, u32>, r: VReg, depth: u32) -> Option<(u32, u32)> {
        // (instruction count, number of V1 leaves other than tid)
        if m.vreg_class(r) != RClass::V1 && m.vreg_class(r) != RClass::S1 { return None; }
        if m.vreg_class(r) == RClass::S1 { return Some((0, 0)); }
        if nd.get(&r) != Some(&1) || depth > 6 { return None; }
        let (b, i) = *def.get(&r)?;
        let inst = &m.blocks[b].insts[i];
        if !pure(inst.op) { return None; }
        if inst.op == MOp::VReadPhys { return Some((0, 0)); } // tid: stays live anyway
        let mut c = (1u32, 0u32);
        for s in &inst.srcs {
            if let Src::Reg(x) = s {
                let (a, l) = cost(m, def, nd, *x, depth + 1)?;
                c.0 += a; c.1 += l;
            }
        }
        Some(c)
    }
    for (h, _l) in &m.loops {
        let bi = m.blocks.iter().position(|b| b.label == *h).unwrap();
        let (mut total, mut ok, mut okcost) = (0, 0, 0u32);
        for r in lv.live_in[bi].iter() {
            if m.vreg_class(*r) != RClass::V1 { continue; }
            total += 1;
            if let Some((c, _)) = cost(&m, &def, &nd, *r, 0) { ok += 1; okcost += c; }
        }
        println!("REMAT hdr b{bi}: live V1 {total}, rematerialisable from tid/SGPR/const leaves: {ok}, total instrs per block-use set: {okcost}");
    }
}