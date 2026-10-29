use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::liveness;
use amdgpu_compiler::{kernel_by_name, lower};

#[test]
fn live_defs() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_t").unwrap();
    let m = with_opts(&["dce", "cmpfold", "scalarize", "sgpr"], || lower::lower(&f)).unwrap();
    let lv = liveness::compute(&m);
    let (h, _) = m.loops[0];
    let bi = m.blocks.iter().position(|b| b.label == h).unwrap();
    for r in lv.live_in[bi].iter() {
        if m.vreg_class(*r) != RClass::V1 {
            continue;
        }
        let mut defs = Vec::new();
        for (b, blk) in m.blocks.iter().enumerate() {
            for (i, inst) in blk.insts.iter().enumerate() {
                if inst.dst == Some(*r) {
                    let srcs: Vec<String> = inst.srcs.iter().map(|s| match s {
                        Src::Reg(x) => format!("{:?}{}", m.vreg_class(*x), x),
                        Src::Imm(k) => format!("#{k}"),
                    }).collect();
                    defs.push(format!("b{b}.{i} {:?}({})", inst.op, srcs.join(",")));
                }
            }
        }
        let mut users = Vec::new();
        for (b, blk) in m.blocks.iter().enumerate() {
            for (i, inst) in blk.insts.iter().enumerate() {
                if inst.srcs.iter().any(|s| matches!(s, Src::Reg(x) if x == r)) {
                    users.push(format!("b{b}.{i}:{:?}(imm {})", inst.op, inst.imm));
                }
            }
        }
        users.truncate(6);
        println!("LIVE r{r}: {} || USERS {}", defs.join(" | "), users.join(" "));
    }
}