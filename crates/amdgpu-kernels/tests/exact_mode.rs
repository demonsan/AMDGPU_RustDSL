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
    let rs = b.buffer_rsrc_sized(po, 64 * 4);
    let off = b.bind_i32(ev(tid) << 2);
    if wait {
        b.wait_lds(0);
    }
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
