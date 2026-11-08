//! Compiler guards found by kernel bring-up (docs/op-coverage-tileops.md): dead in-flight loads, wave-uniform branches, the launch register budget.
use amdgpu_dsl::{CmpPred, CmpTy, DType, FnBuilder, MemTy, Op, Opnd};

fn kernel_with_dead_buffer_load() -> amdgpu_dsl::Function {
    let mut b = FnBuilder::new("dead_load", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg_tensor("p", amdgpu_dsl::Ext::Const(1024));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    let r = b.buffer_desc(p, 1024);
    let used = b.buffer_load_n(r, Some(Opnd::Val(off)), 0, 0, MemTy::B32);
    let _dead = b.buffer_load_n(r, Some(Opnd::Val(off)), 0, 256, MemTy::B32); // never used
    let x = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(7)));
    let y = b.add_i32(used, x);
    b.buffer_store_n(r, Some(Opnd::Val(off)), 0, 512, y, MemTy::B32, false);
    b.finish()
}

/// A dead load's destination would be handed to another value while the load is in flight; its late write then clobbers that value
/// (kernel-opt's MoE epilogue). Dead loads are removed.
#[test]
fn a_dead_buffer_load_is_not_emitted() {
    let m = amdgpu_compiler::compile_mir(&kernel_with_dead_buffer_load()).unwrap();
    let n = m.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.op == amdgpu_compiler::mir::MOp::BufferLoadDwordX1).count();
    assert_eq!(n, 1, "only the used load remains");
}

/// `break_if` / `if` are wave-level: a per-lane condition is rejected unless marked `any_lane`.
#[test]
fn per_lane_branch_conditions_are_rejected_unless_marked() {
    for (mark, ok) in [(false, false), (true, true)] {
        let mut b = FnBuilder::new("lane_loop", 64);
        let tid = b.let_(Op::WorkitemId);
        let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
        let n = b.add_i32(Opnd::Var(vars[0]), 1);
        b.set_var(vars[0], n);
        let done = b.cmp(CmpPred::Ge, CmpTy::I32, n, tid); // per-lane trip count
        let c = if mark { b.any_lane(done) } else { done };
        b.break_if(c);
        b.end_loop();
        let f = b.finish();
        let errs = amdgpu_dsl::verify(&f);
        assert_eq!(errs.iter().all(|e| !e.contains("wave-uniform")), ok, "{errs:?}");
    }
    // uniform bounds (kernel argument, workgroup id) are fine
    let mut b = FnBuilder::new("uni_loop", 64);
    b.arg("n", DType::I32, 0, 4);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let i = b.add_i32(Opnd::Var(vars[0]), 1);
    b.set_var(vars[0], i);
    let done = b.cmp(CmpPred::Ge, CmpTy::I32, i, n);
    b.break_if(done);
    b.end_loop();
    assert!(amdgpu_dsl::verify(&b.finish()).is_empty());
}

/// 1024 threads = 16 waves = 4 per SIMD: at most 128 registers per wave; more fails at launch, so the compile step rejects it.
#[test]
fn a_kernel_over_the_launch_register_budget_is_rejected() {
    for (threads, ok) in [(1024u32, false), (256, true)] {
        let mut b = FnBuilder::new("fat", threads);
        b.arg("p", DType::Ptr, 0, 8);
        b.arg_tensor("p", amdgpu_dsl::Ext::Const(1 << 20));
        let p = b.let_(Op::KernargPtr { offset: 0 });
        let tid = b.let_(Op::WorkitemId);
        let off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(16)));
        let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
        // 40 live 4-dword values: 160 VGPRs
        let vals: Vec<_> = (0..40u32).map(|k| {
            let base = b.let_(Op::PtrAdd { base: Opnd::Val(q), byte_off: Opnd::I32((k * 16384) as i32) });
            b.load_global(base, 0, MemTy::B128)
        }).collect();
        let mut acc = b.c_f32(0.0);
        for v in &vals {
            for i in 0..4 {
                let x = b.frag4_lane(*v, i);
                let f = b.let_(Op::BitcastF32(Opnd::Val(x)));
                acc = b.let_(Op::AddF32(Opnd::Val(acc), Opnd::Val(f)));
            }
        }
        b.store_f32(q, 0, acc);
        let f = b.finish();
        let r = amdgpu_compiler::compile_with(&f, &amdgpu_compiler::backend::OwnTail);
        match r {
            Ok(c) => assert!(ok, "{threads} threads with {} VGPRs must be rejected", c.vgpr),
            Err(e) => assert!(!ok && e.contains("register budget"), "{threads}: {e}"),
        }
    }
}

/// The scheduler plans against the launch budget: a 1024-thread block (4 waves per SIMD) leaves 128 registers per wave (minus 8 headroom).
#[test]
fn the_scheduler_budget_follows_the_workgroup_size() {
    use amdgpu_compiler::sched::launch_vgpr_budget;
    assert_eq!(launch_vgpr_budget(1024), 120);
    assert_eq!(launch_vgpr_budget(768), 162);
    assert_eq!(launch_vgpr_budget(512), 240);
    assert_eq!(launch_vgpr_budget(256), 240);
    assert_eq!(launch_vgpr_budget(64), 240);
}

/// A > 4 GiB tensor (up to 8192 blocks of 1 MiB): the block's descriptor base is moved with `ptr_mad` (64-bit) and the verifier proves the
/// moved descriptor stays inside the allocation; an overstated descriptor (2 MiB per block) is rejected.
fn kv_block_store(per_block: u32) -> amdgpu_dsl::Function {
    let mut b = FnBuilder::new("kv_block_store", 64);
    b.arg("kv", DType::Ptr, 0, 8);
    b.arg("nblocks", DType::I32, 8, 4);
    b.ka_range(8, 1, 8192);
    b.arg_tensor("kv", amdgpu_dsl::Ext::ka(8) * amdgpu_dsl::Ext::Const(1 << 20));
    let kv = b.let_(Op::KernargPtr { offset: 0 });
    let nb = b.let_(Op::KernargI32 { offset: 8 });
    let last = b.add_i32(nb, -1);
    let wg = b.let_(Op::WorkgroupIdX);
    let blk = b.let_(Op::MinI32(Opnd::Val(wg), Opnd::Val(last)));
    let moved = b.ptr_mad(kv, blk, 1 << 20);
    let r = b.buffer_desc(moved, per_block);
    let tid = b.let_(Op::WorkitemId);
    let off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::I32(4)));
    b.buffer_store_n(r, Some(Opnd::Val(off)), 0, 0, tid, MemTy::B32, false);
    b.finish()
}

#[test]
fn a_moved_descriptor_over_a_large_tensor_is_proven_in_bounds() {
    let errs = |f: &amdgpu_dsl::Function| -> Vec<String> {
        amdgpu_compiler::gpu_verify::analyse(f).unwrap().errors().iter().map(|e| e.code.to_string()).collect()
    };
    assert!(errs(&kv_block_store(1 << 20)).is_empty(), "{:?}", errs(&kv_block_store(1 << 20)));
    assert!(errs(&kv_block_store(2 << 20)).iter().any(|c| c == "GV-BUF-OVER"), "{:?}", errs(&kv_block_store(2 << 20)));
}
