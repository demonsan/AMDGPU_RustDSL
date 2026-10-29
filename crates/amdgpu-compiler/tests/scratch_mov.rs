use amdgpu_compiler::mir::*;
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name};
use std::collections::{BTreeMap, HashMap};

#[test]
fn tuple_runs() {
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_vs128").unwrap();
    let (m, _a) = compile_mir_alloc(&f).unwrap();
    // uses per canonical vreg and defs
    let mut uses: HashMap<VReg, u32> = HashMap::new();
    let mut defs: HashMap<VReg, Vec<(usize, usize, MOp)>> = HashMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, i) in b.insts.iter().enumerate() {
            for u in i.uses() { *uses.entry(m.canon(u)).or_default() += 1; }
            if let Some(d) = i.dst { defs.entry(m.canon(d)).or_default().push((bi, ii, i.op)); }
        }
    }
    let mut by: BTreeMap<String, u32> = BTreeMap::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        let mut ii = 0;
        while ii < b.insts.len() {
            let i = &b.insts[ii];
            let lane = |x: VReg| m.alias.get(&x).copied();
            if i.op == MOp::VMovB32 {
                if let (Some(d), Some(Src::Reg(s))) = (i.dst, i.srcs.first()) {
                    if let (Some((dt, d0)), Some((st, s0))) = (lane(d), lane(*s)) {
                        // run
                        let mut n = 1;
                        while ii + n < b.insts.len() {
                            let j = &b.insts[ii + n];
                            if j.op != MOp::VMovB32 { break; }
                            let (Some(d2), Some(Src::Reg(s2))) = (j.dst, j.srcs.first()) else { break };
                            let (Some((dt2, d1)), Some((st2, s1))) = (lane(d2), lane(*s2)) else { break };
                            if dt2 != dt || st2 != st || d1 != d0 + n as u8 || s1 != s0 + n as u8 { break; }
                            n += 1;
                        }
                        let sd = defs.get(&st).map(|v| v.iter().map(|x| format!("{:?}@{}{}", x.2, if x.0 == bi { "same" } else { "other" }, "")).collect::<Vec<_>>().join(",")).unwrap_or_default();
                        let key = format!("run{n} {:?}<-{:?} src defs [{}] src uses {} dst defs {}", m.vreg_class(dt), m.vreg_class(st), sd.chars().take(60).collect::<String>(), uses.get(&st).copied().unwrap_or(0), defs.get(&dt).map(|v| v.len()).unwrap_or(0));
                        *by.entry(key).or_default() += 1;
                        ii += n;
                        continue;
                    }
                }
            }
            ii += 1;
        }
    }
    for (k, v) in by { println!("RUN {v:3} x {k}"); }
}