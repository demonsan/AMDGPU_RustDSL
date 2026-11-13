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
    b.buffer_store_f32(rs, off, Opnd::I32(0), 0, v);
    b.pin_sgpr_at(RegRef::Val(po), 8);
    b.pin_sgpr_at(RegRef::Val(rs), 12);
    b.pin_vgpr_at(RegRef::Val(tid), 0);
    b.pin_vgpr_at(RegRef::Val(off), 1);
    b.pin_vgpr_at(RegRef::Val(t), 2);
    b.pin_vgpr_at(RegRef::Val(v), 2);
    b.finish()
}

#[test]
fn the_hardware_probe_kernel_compiles_exactly() {
    let f = exact_probe_kernel();
    let c = amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(c.asm.contains("v_add_u32_e32 v2, 7, v2"), "{}", c.asm);
    if let Ok(dir) = std::env::var("EXACT_PROBE_OUT") {
        std::fs::write(format!("{dir}/probe_exact_hw.ir"), amdgpu_dsl::ir_text::print(&f)).unwrap();
    }
}
