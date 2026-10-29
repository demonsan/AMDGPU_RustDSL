//! Regression: an asynchronous load whose result is never read must not share its destination register with a live
//! value. `s_load` completes at an arbitrary time up to the covering `s_waitcnt lgkmcnt(0)`: if the allocator
//! treats a dead load's destination as free right after the load (zero-length live range) a later value assigned to
//! the same SGPR is overwritten when the load lands. Found with `fmha_fwd_hd128_mxfp8_sv` built with
//! `FMHA_OPT=cvt4,dma,nodmak,nodmav,kh0` (the constant K/V base makes several kernarg loads dead; with the `dce`
//! lowering option the consumers disappear, two `s_load`s then targeted the same SGPR and the kernel faulted).
//!
//! This file contains a single test (it sets FMHA_OPT process-wide, which would race with other tests).

use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::effects::{effects, VCC};
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name};
use std::collections::HashSet;

fn phys(m: &MFunction, a: &amdgpu_compiler::regalloc::Alloc, slot: u32) -> (bool, u32) {
    let t = slot / 16;
    (m.vreg_class(t).is_vector(), a.phys[t as usize] + (slot - t * 16))
}

/// In-flight scalar loads: no other instruction may write their destination before the covering wait, and no second
/// load may target it.
fn sload_conflicts(m: &MFunction, a: &amdgpu_compiler::regalloc::Alloc) -> Vec<String> {
    let mut bad = Vec::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        let mut pending: HashSet<u32> = HashSet::new();
        for (ii, inst) in b.insts.iter().enumerate() {
            if inst.op == MOp::SWaitLgkm && inst.imm == 0 {
                pending.clear();
                continue;
            }
            let e = effects(m, inst);
            let is_sload = matches!(inst.op, MOp::SLoadDword | MOp::SLoadDwordx2);
            for w in &e.writes {
                if *w == VCC || *w == u32::MAX - 1 {
                    continue;
                }
                let (vec, p) = phys(m, a, *w);
                if vec {
                    continue;
                }
                if pending.contains(&p) {
                    bad.push(format!("block {bi} inst {ii} {:?} writes s{p} while an s_load into it is in flight", inst.op));
                }
                if is_sload {
                    pending.insert(p);
                }
            }
        }
    }
    bad
}

#[test]
fn dead_scalar_loads_do_not_alias_live_registers() {
    std::env::set_var("FMHA_OPT", "cvt4,dma,nodmak,nodmav,kh0");
    let f = kernel_by_name("fmha_fwd_hd128_mxfp8_sv").unwrap();
    let mut bad = Vec::new();
    for opts in [&["none"][..], &["dce", "cmpfold"][..], &["all"][..]] {
        let (m, a) = with_opts(opts, || compile_mir_alloc(&f)).unwrap();
        let c = sload_conflicts(&m, &a);
        if !c.is_empty() {
            bad.push(format!("{opts:?}: {} conflicts, first: {}", c.len(), c[0]));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}