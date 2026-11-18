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