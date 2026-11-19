//! gfx1250 exact mode (docs/gfx1250-plan.md): the author's split-counter waits, XCNT and - under the relaxed scheduling mode - the
//! `s_wait_alu` dependency waits are verified (target/gfx1250/check.rs), never added to; nothing is inserted into an exact kernel.

use amdgpu_dsl::{ev, DType, Ext, FnBuilder, Function, Op, Opnd, RegRef, WaitCtr};

/// out[tid] = tid through a buffer: kernarg pointer s[8:9], descriptor s[12:15], tid v0, offset v1. `km`: the author waits for the kernarg load; `mode`:
/// the scheduling mode set first (None: not set); `alu`: an `s_wait_alu` before the store (the store reads v1, which a VALU just wrote).
fn tiny(km: bool, mode: Option<u8>, alu: Option<u16>) -> Function {
    let mut b = FnBuilder::new("probe_exact_1250", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(64 * 4));
    b.exact();
    b.lower_opts(&["imm", "sgpr"]);
    if let Some(md) = mode {
        b.set_sched_mode(md);
    }
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    if km {
        b.wait_counter(WaitCtr::Km, 0);
    }
    let off = b.bind_i32(ev(tid) << 2);
    if let Some(i) = alu {
        b.wait_alu(i);
    }
    let rs = b.buffer_rsrc_sized(po, 64 * 4);
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, tid);
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    b.pin_vgpr_at(RegRef::Val(tid), 0);
    b.pin_vgpr_at(RegRef::Val(off), 1);
    b.finish()
}

fn compile(f: &Function) -> Result<String, String> {
    amdgpu_codegen::compile_target(f, "gfx1250").map(|c| c.asm)
}

#[test]
fn an_authored_gfx1250_kernel_compiles_without_additions() {
    let asm = compile(&tiny(true, None, None)).unwrap_or_else(|e| panic!("{e}"));
    let body: Vec<&str> = asm.lines().map(|l| l.trim()).filter(|l| l.starts_with("s_") || l.starts_with("v_") || l.starts_with("buffer_")).collect();
    assert!(body.iter().any(|l| l.starts_with("s_wait_kmcnt 0x0")), "{body:?}");
    assert!(!body.iter().any(|l| l.starts_with("s_wait_xcnt") || l.starts_with("v_nop") || l.starts_with("s_mov_b64 s[6:7]")), "nothing added: {body:?}");
}

#[test]
fn a_missing_split_counter_wait_is_named() {
    let r = compile(&tiny(false, None, None));
    let e = r.clone().err().unwrap_or_else(|| panic!("the kernarg pointer is read before its load returned:\n{}", r.unwrap()));
    assert!(e.contains("s_wait_kmcnt"), "{e}");
}

#[test]
fn the_relaxed_scheduling_mode_requires_the_alu_waits() {
    // mode 2: the store (a non-VALU) reads v1, written by the VALU shift just before: va_vdst must be waited
    let e = compile(&tiny(true, Some(2), None)).err().expect("relaxed mode without s_wait_alu");
    assert!(e.contains("va_vdst") && e.contains("relaxed scheduling mode"), "{e}");
    // with the wait (va_vdst = 0, everything else default) it verifies
    compile(&tiny(true, Some(2), Some(0x0f9f))).unwrap_or_else(|e| panic!("{e}"));
    // mode 0 (the default): the hardware checks it, nothing needed
    compile(&tiny(true, Some(0), None)).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn the_scheduling_mode_is_exact_only() {
    let mut b = FnBuilder::new("probe_sched_mode", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.set_sched_mode(2);
    let f = b.finish();
    let e = compile(&f).err().expect("not exact");
    assert!(e.contains("exact-mode statement"), "{e}");
}

/// tid -> f32 (v1), rsq (v2, a TRANS), cvt_i32(r) (v3, a VALU reading the TRANS result right after it: the co-execution hazard wants one v_nop),
/// stored through a buffer. `trust`: a TrustHazard before the add; `trust_elsewhere`: one before the store instead (waives nothing).
fn trans(trust: bool, trust_elsewhere: bool) -> Function {
    let mut b = FnBuilder::new("probe_exact_1250_trans", 32);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(32 * 4));
    b.exact();
    b.lower_opts(&["imm", "sgpr"]);
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    b.wait_counter(WaitCtr::Km, 0);
    let x = b.un(amdgpu_dsl::UnOp::CvtF32I32, tid);
    let r = b.un(amdgpu_dsl::UnOp::RsqF32, x);
    if trust {
        b.trust_hazard("vendor omits the TRANS co-execution v_nop");
    }
    // a conversion: a reader class not proven exempt on hardware (docs/gfx1250-hazards.md), so the v_nop rule applies
    let s = b.un(amdgpu_dsl::UnOp::CvtI32F32, r);
    let s = b.bitcast_f32(s);
    let off = b.bind_i32(ev(tid) << 2);
    let rs = b.buffer_rsrc_sized(po, 32 * 4);
    if trust_elsewhere {
        b.trust_hazard("misplaced");
    }
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, s);
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    b.pin_vgpr_at(RegRef::Val(tid), 0);
    b.pin_vgpr_at(RegRef::Val(x), 1);
    b.pin_vgpr_at(RegRef::Val(r), 2);
    b.pin_vgpr_at(RegRef::Val(s), 3);
    b.pin_vgpr_at(RegRef::Val(off), 4);
    b.finish()
}

#[test]
fn a_trans_hazard_is_named_and_trust_hazard_waives_exactly_it() {
    let e = compile(&trans(false, false)).err().expect("the add reads the TRANS result right after it");
    assert!(e.contains("hazard") && e.contains("v_nop"), "{e}");
    compile(&trans(true, false)).unwrap_or_else(|e| panic!("{e}"));
    let e = compile(&trans(true, true)).err().expect("a TrustHazard before the store waives nothing");
    assert!(e.contains("TrustHazard") && e.contains("waives nothing"), "{e}");
    let rep = amdgpu_codegen::gpu_verify::verify_ir(&trans(true, false));
    assert_eq!(rep.ledger().hazard_trust, 1, "counted in the certificate");
}

/// A wait immediate beyond the counter's width (6-bit asynccnt: 72) is a compile error, not a silently non-blocking wait.
#[test]
fn a_wait_beyond_the_counter_width_is_a_compile_error() {
    for (c, n) in [(WaitCtr::Async, 72u8), (WaitCtr::Km, 32), (WaitCtr::Ds, 64)] {
        let mut b = FnBuilder::new("probe_wait_range", 32);
        b.wait_counter(c, n);
        let f = b.finish();
        let e = amdgpu_codegen::compile_target(&f, "gfx1250").err().unwrap_or_else(|| panic!("{c:?} {n} must be refused"));
        assert!(e.contains("exceeds"), "{c:?} {n}: {e}");
    }
    let mut b = FnBuilder::new("probe_wait_range_ok", 32);
    b.wait_counter(WaitCtr::Async, 63);
    let f = b.finish();
    amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{e}"));
}

/// A cluster launch's TTMP9 / TTMP7 are cluster ids: the plain workgroup-id ops are a compile error in a kernel that declares a cluster,
/// and keep their lowering without one.
#[test]
fn workgroup_ids_in_a_cluster_launch_are_refused() {
    for op in [Op::WorkgroupIdX, Op::WorkgroupIdY] {
        let mut b = FnBuilder::new("probe_cluster_wgid", 32);
        b.arg("out", DType::Ptr, 0, 8);
        b.arg_tensor("out", Ext::c(64));
        let p = b.let_(Op::KernargPtr { offset: 0 });
        let g = b.let_(op.clone());
        b.store_global(p, 0, g, amdgpu_dsl::MemTy::B32, false);
        let mut f = b.finish();
        amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("no cluster: {e}"));
        f.kd.push(("cluster_dims".into(), 4));
        let e = amdgpu_codegen::compile_target(&f, "gfx1250").err().expect("refused in a cluster launch");
        assert!(e.contains("cluster launch"), "{e}");
    }
}

/// A copy of a loop-carried accumulator lane taken before the lane is scaled in place keeps the old value: the gfx1250 value numbering
/// must not redirect the copy's reads to the lane register itself (they would read the scaled value).
#[test]
fn a_copy_of_a_lane_written_in_place_keeps_its_value() {
    let mut b = FnBuilder::new("probe_copy_inplace", 32);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(1 << 12));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let z = b.frag_cat(&[Opnd::I32(0); 8]);
    let vars = b.begin_loop(&[(DType::Frag8, Opnd::Val(z))]);
    let o = vars[0];
    let lane = b.frag_slice(Opnd::Var(o), 0, 1);
    // the copy (a VGPR move)
    let copy = b.let_(Op::MovV { ty: DType::I32, a: Opnd::Val(lane) });
    b.var_scale_lane(o, 0, Opnd::F32(2.0));
    b.store_global(p, 0, copy, amdgpu_dsl::MemTy::B32, false);
    let one = b.let_(Op::ConstI32(1));
    let one = b.cmp(amdgpu_dsl::CmpPred::Gt, amdgpu_dsl::CmpTy::I32, one, Opnd::I32(0));
    b.break_if(Opnd::Val(one));
    b.end_loop();
    let f = b.finish();
    let asm = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{e}")).asm;
    if std::env::var("SHOW_ASM").is_ok() { eprintln!("{asm}"); }
    // the store's data register is not the register the in-place multiply writes
    let mul_dst: Vec<String> = asm.lines().filter(|l| l.trim_start().starts_with("v_mul_f32")).filter_map(|l| l.split_whitespace().nth(1).map(|s| s.trim_end_matches(',').to_string())).collect();
    let store_data: Vec<String> = asm.lines().filter(|l| l.trim_start().starts_with("global_store_b32")).filter_map(|l| l.split_whitespace().nth(2).map(|s| s.trim_end_matches(',').to_string())).collect();
    assert!(!mul_dst.is_empty() && !store_data.is_empty(), "{asm}");
    for s in &store_data {
        assert!(!mul_dst.contains(s), "the stored copy reads the in-place scaled register {s}:\n{asm}");
    }
}

/// Two booleans materialised in one block (`v_cndmask d, 0, 1, vcc` after VCC was taken by another compare) stay two values: a select
/// on the first must not test the second's register (the 2026-11-18 MLA `_wsbr` miscompile: value numbering merged the two
/// materialisations, whose only difference is the implicit VCC input).
#[test]
fn two_materialised_booleans_stay_distinct() {
    let mut b = FnBuilder::new("probe_two_bools", 32);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg_tensor("out", Ext::c(1 << 12));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(31)));
    let half = b.let_(Op::LshrI32(Opnd::Val(lane), Opnd::I32(4)));
    let odd = b.let_(Op::AndI32(Opnd::Val(lane), Opnd::I32(1)));
    let c1 = b.cmp(amdgpu_dsl::CmpPred::Eq, amdgpu_dsl::CmpTy::I32, half, Opnd::I32(0));
    let c2 = b.cmp(amdgpu_dsl::CmpPred::Eq, amdgpu_dsl::CmpTy::I32, odd, Opnd::I32(0));
    // c1 used after c2 (VCC holds c2): both are materialised; then each is used again
    let s2 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c2), a: Opnd::I32(3), b: Opnd::I32(4) });
    let s1 = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c1), a: Opnd::I32(1), b: Opnd::I32(2) });
    let s2b = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c2), a: Opnd::I32(5), b: Opnd::I32(6) });
    let s1b = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(c1), a: Opnd::I32(7), b: Opnd::I32(8) });
    for (k, v) in [s1, s2, s1b, s2b].into_iter().enumerate() {
        b.store_global(p, 4 * k as u32, v, amdgpu_dsl::MemTy::B32, false);
    }
    let f = b.finish();
    let asm = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{e}")).asm;
    if std::env::var("SHOW_ASM").is_ok() {
        eprintln!("{asm}");
    }
    // both compares are materialised as 0 / 1 (`v_cndmask d, 0, 1, vcc`), each in its own register
    let mats: Vec<String> = asm.lines().filter(|l| l.contains("v_cndmask_b32") && l.contains(", 0, ")).filter_map(|l| l.split_whitespace().nth(1).map(|s| s.trim_end_matches(',').to_string())).collect();
    let distinct: std::collections::HashSet<&String> = mats.iter().collect();
    assert!(distinct.len() >= 2, "two booleans, one materialisation: {mats:?}\n{asm}");
}

/// Loads whose results are never read: the allocator may give a later load the same destination while the first is in flight. Two
/// loads on the in-order LOADcnt need no wait between them (the younger lands last); the wait self-check accepts it, and a VALU
/// write over an in-flight load's destination still gets its wait (`hazards::pad_inflight_waw`).
#[test]
fn dead_loads_with_reused_destinations_compile() {
    let mut b = FnBuilder::new("probe_dead_loads", 32);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg_tensor("p", Ext::c(1 << 12));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    for k in 0..4u32 {
        let _dead = b.let_(Op::LoadGlobalB128 { base: Opnd::Val(p), byte_off: 64 * k });
    }
    let x = b.let_(Op::LoadGlobalB128 { base: Opnd::Val(p), byte_off: 512 });
    b.store_f32x4(p, 1024, x);
    let f = b.finish();
    let asm = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{e}")).asm;
    if std::env::var("SHOW_ASM").is_ok() {
        eprintln!("{asm}");
    }
}

/// The wait self-check accepts the compiler's own waits for kernel-opt's MLA debug variants (not registered): a prologue whose Q-fragment
/// loads are stored after more than 63 younger STORES (the shared vmcnt saturation must not retire a load on gfx1250's split counters),
/// an early exit after the first tensor load, and Q loads before the page loop with global stores inside it (operations beyond the known
/// queue at a loop head are not counted as loads).
#[test]
fn mla_debug_variants_get_their_waits() {
    use amdgpu_kernels::probes::g1250_mla_ws as w;
    for f in [w::mla_g1250_qh16_wsx0(), w::mla_g1250_qh16_wsx1(), w::mla_g1250_qh16_wsstd_nowait()] {
        amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{}: {e}", f.name));
    }
}
