use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::liveness;
use amdgpu_compiler::{kernel_by_name, lower};

#[test]
fn live_consts() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    let m = with_opts(&["dce", "cmpfold", "scalarize"], || lower::lower(&f)).unwrap();
    let lv = liveness::compute(&m);
    let (h, _) = m.loops[0];
    let bi = m.blocks.iter().position(|b| b.label == h).unwrap();
    for r in lv.live_in[bi].iter() {
        if m.vreg_class(*r) != RClass::V1 { continue; }
        let mut def = None;
        let mut ndefs = 0;
        for (b, blk) in m.blocks.iter().enumerate() {
            for (i, inst) in blk.insts.iter().enumerate() {
                if inst.dst == Some(*r) { ndefs += 1; if def.is_none() { def = Some((b, i, inst.op, inst.imm, inst.srcs.len())); } }
            }
        }
        let Some((b, i, op, imm, ns)) = def else { continue };
        if op != MOp::VMovB32 || ns != 0 { continue; }
        let mut users = std::collections::BTreeMap::new();
        for (ub, blk) in m.blocks.iter().enumerate() {
            for inst in &blk.insts {
                if inst.srcs.iter().any(|s| matches!(s, Src::Reg(x) if x == r)) { *users.entry(ub).or_insert(0) += 1; }
            }
        }
        println!("CONST r{r} = {imm} (def b{b}.{i}, ndefs {ndefs}) used in blocks {:?}", users);
    }
}