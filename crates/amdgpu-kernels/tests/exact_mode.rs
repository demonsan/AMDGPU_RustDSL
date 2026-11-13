//! Exact authored mode E3 (docs/exact-mode.md): a kernel whose every value is pinned and whose waits are the author's compiles without any
//! scheduling / allocation / inserted instruction; a missing pin, a missing wait or two live values on one register are named errors.

use amdgpu_dsl::{ev, DType, Ext, FnBuilder, Function, Op, Opnd, RegRef};

/// out[tid] = tid * 4 as bytes, exact: kernarg pointer in s[8:9], descriptor in s[12:15], tid in v0, offset in v1.
fn tiny(wait: bool, pin_off: Option<u32>) -> Function {
    let mut b = FnBuilder::new("probe_exact", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(64 * 4));
    b.exact();
    b.lower_opts(&["imm", "sgpr"]);
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    // the descriptor reads the kernarg pointer: the author waits for the scalar load first
    if wait {
        b.wait_lds(0);
    }
    let rs = b.buffer_rsrc_sized(po, 64 * 4);
    let off = b.bind_i32(ev(tid) << 2);
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, tid);
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    b.pin_vgpr_at(RegRef::Val(tid), 0);
    if let Some(v) = pin_off {
        b.pin_vgpr_at(RegRef::Val(off), v);
    }
    b.finish()
}

#[test]
fn a_fully_authored_kernel_compiles_verbatim() {
    let c = amdgpu_codegen::compile(&tiny(true, Some(1))).unwrap_or_else(|e| panic!("{e}"));
    let body: Vec<&str> = c.asm.lines().map(|l| l.trim()).filter(|l| l.starts_with("s_") || l.starts_with("v_") || l.starts_with("buffer_")).collect();
    // exactly the author's instructions: the kernarg load, the descriptor (fixed 4-word expansion), the wait, the shift, the store, the end
    assert!(body.iter().any(|l| l.starts_with("s_load_dwordx2 s[8:9]")), "{body:?}");
    assert!(body.iter().any(|l| l.starts_with("v_lshlrev_b32_e32 v1, 2, v0")), "{body:?}");
    assert!(body.iter().any(|l| l.starts_with("buffer_store_dword v0, v1, s[12:15]")), "{body:?}");
    assert!(!body.iter().any(|l| l.starts_with("s_nop") || l.starts_with("v_mov")), "nothing added: {body:?}");
}

#[test]
fn a_missing_pin_is_named() {
    let e = amdgpu_codegen::compile(&tiny(true, None)).err().expect("the shift result has no pin");
    assert!(e.contains("without a pin"), "{e}");
}

#[test]
fn a_missing_wait_is_named() {
    let e = amdgpu_codegen::compile(&tiny(false, Some(1))).err().expect("the descriptor is read before the kernarg load returned");
    assert!(e.contains("[waits]"), "{e}");
}

#[test]
fn a_clobbering_pin_is_named() {
    // the offset on v0 overwrites tid, which the store still reads
    let e = amdgpu_codegen::compile(&tiny(true, Some(0))).err().expect("two live values on v0");
    assert!(e.contains("clobber"), "{e}");
}

#[test]
fn exact_round_trips_through_irv1() {
    let f = tiny(true, Some(1));
    let t = amdgpu_dsl::ir_text::print(&f);
    assert!(t.lines().any(|l| l == "exact"), "{t}");
    let g = amdgpu_dsl::ir_text::parse(&t).unwrap();
    assert!(g.exact);
}

/// (j) the source-negate modifier, (k) authored moves, (l) the bf16 32x32x8 MFMA kind.
#[test]
fn neg_modifier_moves_and_bf16_mfma_lower_to_one_instruction_each() {
    let mut b = FnBuilder::new("probe_exact_forms", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(64 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    let x = b.let_(Op::KernargF32 { offset: 0x04 });
    let k = b.let_(Op::KernargF32 { offset: 0x00 });
    let n = b.neg_f32(k);
    let y = b.let_(Op::MulF32(Opnd::Val(n), Opnd::Val(x)));
    let z = b.mov_v(DType::F32, y);
    let rs = b.buffer_rsrc_sized(po, 64 * 4);
    let off = b.bind_i32(ev(tid) << 2);
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, z);
    let f = b.finish();
    let c = amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.lines().any(|l| l.trim().starts_with("v_mul_f32_e64") && l.contains(", -")), "{}", c.asm);
    assert!(!c.asm.contains("v_xor_b32"), "the negation folded into the modifier");
}

#[test]
fn bf16_wide_mfma_is_its_own_kind() {
    let mut b = FnBuilder::new("probe_bf16_wide", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(64 * 64));
    let z = b.let_(Op::FragZero { ty: DType::Frag16 });
    let one = b.c_i32(0x3f80_3f80);
    let a = b.let_(Op::Frag2(Opnd::Val(one), Opnd::Val(one)));
    let two = b.c_i32(0x4000_4000);
    let bb = b.let_(Op::Frag2(Opnd::Val(two), Opnd::Val(two)));
    let r = b.mfma_bf16_wide(a, bb, z);
    let r2 = b.mfma_bf16_wide(a, bb, r);
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let rs = b.buffer_rsrc_sized(po, 64 * 64);
    let tid = b.let_(Op::WorkitemId);
    let off = b.bind_i32(ev(tid) << 6);
    let l = b.let_(Op::Frag16Lane { frag: Opnd::Val(r2), index: 0 });
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, l);
    let f = b.finish();
    let c = amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.contains("v_mfma_f32_32x32x8_bf16"), "{}", c.asm);
}

/// The hardware probe's kernel (tools/exact_probe.sh): out[i] = 2 i + 7, exact, `t = i << 1` and `t + 7` share v2 (in-place reuse).
pub fn exact_probe_kernel() -> Function {
    let mut b = FnBuilder::new("probe_exact_hw", 64);
    b.arg("in", DType::Ptr, 0x00, 8);
    b.arg("out", DType::Ptr, 0x08, 8);
    b.arg_tensor("out", Ext::c(64 * 4));
    b.exact();
    let po = b.let_(Op::KernargPtr { offset: 0x08 });
    let tid = b.let_(Op::WorkitemId);
    b.wait_lds(0);
    let rs = b.buffer_rsrc_sized(po, 64 * 4);
    let off = b.bind_i32(ev(tid) << 2);
    let t = b.bind_i32(ev(tid) << 1);
    let v = b.let_(Op::AddI32(Opnd::Val(t), Opnd::I32(7)));
    let vf = b.let_(Op::BitcastF32(Opnd::Val(v)));
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, vf);
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    b.pin_vgpr_at(RegRef::Val(tid), 0);
    b.pin_vgpr_at(RegRef::Val(off), 1);
    b.pin_vgpr_at(RegRef::Val(t), 2);
    b.pin_vgpr_at(RegRef::Val(v), 2);
    b.pin_vgpr_at(RegRef::Val(vf), 2);
    b.finish()
}

#[test]
fn the_hardware_probe_kernel_compiles_exactly() {
    let f = exact_probe_kernel();
    let c = amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.contains("v_add_u32_e32 v2, 7, v2"), "{}", c.asm);
    // tools/exact_probe.sh creates build/exact and reads the kernel from there
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../build/exact");
    if std::path::Path::new(dir).is_dir() {
        std::fs::write(format!("{dir}/probe_exact_hw.ir"), amdgpu_dsl::ir_text::print(&f)).unwrap();
    }
}

/// (n) a buffer descriptor over a kernarg pointer plus a uniform offset: scalar `s_add_u32` + `s_addc_u32` in exact mode.
#[test]
fn a_descriptor_base_from_a_scalar_pointer_add() {
    let mut b = FnBuilder::new("probe_exact_ptradd", 64);
    b.arg("out", DType::Ptr, 0x00, 8);
    b.arg_tensor("out", Ext::c(128 * 4));
    b.exact();
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    b.wait_lds(0);
    let pb = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::I32(256) });
    let rs = b.buffer_rsrc_sized(pb, 64 * 4);
    let off = b.bind_i32(ev(tid) << 2);
    let vf = b.let_(Op::BitcastF32(Opnd::Val(off)));
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, vf);
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(pb), 10);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    b.pin_vgpr_at(RegRef::Val(tid), 0);
    b.pin_vgpr_at(RegRef::Val(off), 1);
    b.pin_vgpr_at(RegRef::Val(vf), 1);
    let f = b.finish();
    let c = amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.contains("s_add_u32 s10, s8, 0x100") || c.asm.contains("s_add_u32 s10, s8, 256"), "{}", c.asm);
    assert!(c.asm.contains("s_addc_u32 s11, s9, 0"), "{}", c.asm);
}

/// EXEC save / all / restore as authored statements (EXEC stays full); a restore without its save is named.
#[test]
fn exec_save_and_restore() {
    let build = |save: bool| {
        let mut f = tiny_with(|b| {
            if save {
                b.exec_save(20);
            }
            b.exec_all();
            b.exec_restore(20);
        });
        f.exact = true;
        f
    };
    let c = amdgpu_codegen::compile(&build(true)).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.contains("s_mov_b64 s[20:21], exec") && c.asm.contains("s_mov_b64 exec, -1") && c.asm.contains("s_mov_b64 exec, s[20:21]"), "{}", c.asm);
    let e = amdgpu_codegen::compile(&build(false)).err().expect("restore without a save");
    assert!(e.contains("without an EXEC save"), "{e}");
}

/// `tiny` with extra statements before the store.
fn tiny_with(extra: impl Fn(&mut FnBuilder)) -> Function {
    let mut b = FnBuilder::new("probe_exact_exec", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(64 * 4));
    b.exact();
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    b.wait_lds(0);
    let rs = b.buffer_rsrc_sized(po, 64 * 4);
    let off = b.bind_i32(ev(tid) << 2);
    extra(&mut b);
    let vf = b.let_(Op::BitcastF32(Opnd::Val(off)));
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, vf);
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    b.pin_vgpr_at(RegRef::Val(tid), 0);
    b.pin_vgpr_at(RegRef::Val(off), 1);
    b.pin_vgpr_at(RegRef::Val(vf), 1);
    b.finish()
}

#[test]
fn the_combined_waitcnt_is_one_instruction() {
    let f = tiny_with(|b| b.waitcnt(0, 0));
    let c = amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.contains("s_waitcnt vmcnt(0) lgkmcnt(0)"), "{}", c.asm);
}

/// `TrustWait`: waives the counter check of the next statement only (counted as GV-TRUST-TIMING); a waiver that waives nothing is an error.
#[test]
fn a_trusted_wait_is_counted_and_must_waive_something() {
    let build = |trust: bool, wait: bool| {
        let mut b = FnBuilder::new("probe_exact_trust", 64);
        b.arg("o", DType::Ptr, 0x00, 8);
        b.arg_tensor("o", Ext::c(64 * 4));
        b.exact();
        let po = b.let_(Op::KernargPtr { offset: 0x00 });
        let tid = b.let_(Op::WorkitemId);
        if wait {
            b.wait_lds(0);
        }
        if trust {
            b.trust_wait("test: the kernarg load is assumed complete");
        }
        let rs = b.buffer_rsrc_sized(po, 64 * 4);
        b.wait_lds(0);
        let off = b.bind_i32(ev(tid) << 2);
        let vf = b.let_(Op::BitcastF32(Opnd::Val(off)));
        b.buffer_store_f32(rs, off, Opnd::I32(0), 0, vf);
        b.pin_sgpr_at(RegRef::Val(po), 8);
        b.pin_sgpr_at(RegRef::Val(rs), 12);
        b.pin_vgpr_at(RegRef::Val(tid), 0);
        b.pin_vgpr_at(RegRef::Val(off), 1);
        b.pin_vgpr_at(RegRef::Val(vf), 1);
        b.finish()
    };
    // without the waiver: the missing wait is an error; with it: compiles, and the certificate counts it
    assert!(amdgpu_codegen::compile(&build(false, false)).is_err());
    let f = build(true, false);
    amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(amdgpu_codegen::gpu_verify::verify_ir(&f).ledger().timing_trust, 1);
    // a waiver where the wait is present waives nothing: error
    let e = amdgpu_codegen::compile(&build(true, true)).err().expect("an unused waiver");
    assert!(e.contains("waives nothing"), "{e}");
}

#[test]
fn an_undef_value_is_only_an_initial_value() {
    let mut b = FnBuilder::new("probe_undef", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    let u = b.undef(DType::F32);
    let k = b.let_(Op::KernargF32 { offset: 0x00 });
    let _ = b.let_(Op::AddF32(Opnd::Val(u), Opnd::Val(k)));
    let errs = amdgpu_dsl::verify(&b.finish());
    assert!(errs.iter().any(|e| e.contains("undefined value")), "{errs:?}");
}

/// `v_permlane32_swap_b32 vA, vB` in place on two separately pinned registers (the two-result form), no copies.
#[test]
fn permlane32_swap_in_place_on_two_registers() {
    let mut b = FnBuilder::new("probe_exact_swap", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg_tensor("o", Ext::c(64 * 4));
    b.exact();
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    b.wait_lds(0);
    let rs = b.buffer_rsrc_sized(po, 64 * 4);
    let off = b.bind_i32(ev(tid) << 2);
    let x = b.mov_v(DType::I32, Opnd::I32(5));
    let y = b.mov_v(DType::I32, Opnd::I32(7));
    let sa = b.let_(Op::Permlane32SwapA { a: Opnd::Val(x), b: Opnd::Val(y) });
    let sb = b.let_(Op::Permlane32SwapB { swap: Opnd::Val(sa) });
    let s = b.let_(Op::AddI32(Opnd::Val(sa), Opnd::Val(sb)));
    let vf = b.let_(Op::BitcastF32(Opnd::Val(s)));
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, vf);
    for (v, r) in [(tid, 0), (off, 1), (x, 2), (sa, 2), (y, 5), (sb, 5), (s, 6), (vf, 6)] {
        b.pin_vgpr_at(RegRef::Val(v), r);
    }
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    let c = amdgpu_codegen::compile(&b.finish()).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.contains("v_permlane32_swap_b32_e32 v2, v5"), "{}", c.asm);
    assert_eq!(c.asm.matches("v_mov_b32").count(), 2, "only the two authored moves: {}", c.asm);
}

/// (s) descriptor words: gpu_verify recognises [lo(p), hi(p) & 0xffff (| 0), num_records, 0x00020000] as BufferRsrc(p, num_records), and
/// names the word that breaks the pattern; `LoadScalar` reads from a scalar pointer, bounds-checked.
#[test]
fn descriptor_words_are_recognised_by_the_verifier() {
    let build = |fmt: i32, stride: i32, other_hi: bool| {
        let mut b = FnBuilder::new("probe_rsrc_words", 64);
        b.arg("a", DType::Ptr, 0x00, 8);
        b.arg("b", DType::Ptr, 0x08, 8);
        b.arg_tensor("a", Ext::c(64 * 4));
        b.arg_tensor("b", Ext::c(64 * 4));
        b.exact();
        let pa = b.let_(Op::KernargPtr { offset: 0x00 });
        let pb = b.let_(Op::KernargPtr { offset: 0x08 });
        b.wait_lds(0);
        let lo = b.let_(Op::PtrLo(Opnd::Val(pa)));
        let hi = b.let_(Op::PtrHi(Opnd::Val(if other_hi { pb } else { pa })));
        let m = b.let_(Op::AndI32(Opnd::Val(hi), Opnd::I32(0xffff)));
        let w1 = b.let_(Op::Bin(amdgpu_dsl::BinOp::OrB32, Opnd::Val(m), Opnd::I32(stride)));
        let nr = b.mov_s(DType::I32, Opnd::I32(64 * 4));
        let w3 = b.mov_s(DType::I32, Opnd::I32(fmt));
        let rs = b.let_(Op::RsrcCat { words: [Opnd::Val(lo), Opnd::Val(w1), Opnd::Val(nr), Opnd::Val(w3)] });
        let n = b.let_(Op::LoadScalar { base: Opnd::Val(pb), byte_off: 4 });
        let tid = b.let_(Op::WorkitemId);
        let off = b.bind_i32(ev(tid) << 2);
        b.wait_lds(0);
        let vf = b.let_(Op::BitcastF32(Opnd::Val(n)));
        b.buffer_store_f32(rs, off, Opnd::I32(0), 0, vf);
        let _ = (pb, tid);
        b.finish()
    };
    let errs = |f: &Function| amdgpu_codegen::gpu_verify::verify_ir(f).errors().iter().map(|e| e.msg.clone()).collect::<Vec<_>>();
    assert!(errs(&build(0x0002_0000, 0, false)).is_empty(), "{:?}", errs(&build(0x0002_0000, 0, false)));
    assert!(errs(&build(0x0003_0000, 0, false)).iter().any(|e| e.contains("word 3")));
    assert!(errs(&build(0x0002_0000, 0x10, false)).iter().any(|e| e.contains("stride")));
    assert!(errs(&build(0x0002_0000, 0, true)).iter().any(|e| e.contains("word 1")));
}

/// Waits on the PHYSICAL registers: two loads into the parts of a tuple assembled in place (pinned so the copies vanish), read as the
/// tuple without a wait, is a waits violation (the value-level scoreboard lost the load -> tuple link once the copies vanished); with the
/// author's wait it compiles. And an unused load stays (exact mode: no dead-load elimination).
#[test]
fn a_load_read_through_a_vanished_copy_needs_its_wait() {
    let build = |wait: bool, dead_load: bool| {
        let mut b = FnBuilder::new("probe_exact_phys_wait", 64);
        b.arg("o", DType::Ptr, 0x00, 8);
        b.arg_tensor("o", Ext::c(64 * 16));
        b.exact();
        let po = b.let_(Op::KernargPtr { offset: 0x00 });
        let tid = b.let_(Op::WorkitemId);
        b.wait_lds(0);
        let rs = b.buffer_rsrc_sized(po, 64 * 16);
        let off = b.bind_i32(ev(tid) << 3);
        let ld = |b: &mut FnBuilder, imm: u32| b.let_(Op::BufferLoadN { rsrc: Opnd::Val(rs), voff: Some(Opnd::Val(off)), soff: Opnd::I32(0), imm, ty: amdgpu_dsl::MemTy::B32, cache: amdgpu_dsl::CachePolicy::NONE });
        let p0 = ld(&mut b, 0);
        let p1 = ld(&mut b, 4);
        let cat = b.let_(Op::FragCat { parts: vec![Opnd::Val(p0), Opnd::Val(p1)], dwords: 2 });
        let dead = if dead_load { Some(ld(&mut b, 8)) } else { None };
        if wait {
            b.wait_vm(if dead_load { 1 } else { 0 });
        }
        b.buffer_store_b64(rs, off, Opnd::I32(0), 512, cat);
        b.wait_vm(0);
        b.pin_sgpr_at(RegRef::Val(po), 8);
        b.pin_sgpr_at(RegRef::Val(rs), 12);
        b.pin_vgpr_at(RegRef::Val(tid), 0);
        b.pin_vgpr_at(RegRef::Val(off), 1);
        b.pin_vgpr_at(RegRef::Val(p0), 4);
        b.pin_vgpr_at(RegRef::Val(p1), 5);
        b.pin_vgpr_at(RegRef::Val(cat), 4);
        if let Some(d) = dead {
            b.pin_vgpr_at(RegRef::Val(d), 6);
        }
        b.finish()
    };
    let e = amdgpu_codegen::compile(&build(false, false)).err().expect("the missing wait must be a violation");
    assert!(e.contains("[waits]"), "{e}");
    let c = amdgpu_codegen::compile(&build(true, false)).unwrap_or_else(|e| panic!("{e}"));
    assert!(!c.asm.contains("v_mov_b32"), "the tuple copies vanish: {}", c.asm);
    let c = amdgpu_codegen::compile(&build(true, true)).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(c.asm.matches("buffer_load_dword ").count(), 3, "the unused load stays: {}", c.asm);
}
