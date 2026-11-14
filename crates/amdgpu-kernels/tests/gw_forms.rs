//! The cross-workgroup store forms of gpu_verify (docs/gpu-semantics-verifier.md): a `ka_pow2` split of a workgroup id into (id >> s,
//! id mod 2^s), the ceil-defined grid size, and a reversed (mirrored) digit - each proven on a disjoint layout and NOT proven on an
//! overlapping one (or without the contract fact).

use amdgpu_codegen::gpu_verify::verify_ir;
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd};

fn proven(f: &Function) -> bool {
    let r = verify_ir(f);
    r.proven.get("global-write-disjoint").is_some() && r.unchecked.get("global-write-untracked").is_none()
}

/// grid z = BB * SPL, lg = ka 0x10 (1 << lg == SPL when `pow2`); z = batch * SPL + split decoded by shifts; out[(split + batch * gap) * 256 + lane * 4].
fn split_kernel(pow2: bool, gap: i32) -> Function {
    let mut b = FnBuilder::new("gw_pow2", 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("lg", DType::I32, 0x10, 4);
    b.sym_range("LS", 0, 4);
    b.sym_range("SPL", 1, 16);
    b.sym_range("BB", 1, 4096);
    b.ka_is(0x10, Ext::sym("LS"));
    if pow2 {
        b.ka_pow2(0x10, Ext::sym("SPL"));
    }
    b.grid_is(2, Ext::sym("BB") * Ext::sym("SPL"));
    b.arg_tensor("out", Ext::sym("BB") * Ext::sym("SPL") * Ext::c(256 * 4));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let lg = b.let_(Op::KernargI32 { offset: 0x10 });
    let lane = b.let_(Op::WorkitemId);
    let z = b.let_(Op::WorkgroupIdZ);
    let batch = b.let_(Op::LshrI32(Opnd::Val(z), Opnd::Val(lg)));
    let bs = b.let_(Op::ShlI32(Opnd::Val(batch), Opnd::Val(lg)));
    let negbs = b.let_(Op::MulI32(Opnd::Val(bs), Opnd::I32(-1)));
    let split = b.let_(Op::AddI32(Opnd::Val(z), Opnd::Val(negbs)));
    // gap * SPL rows per batch: batch << lg is batch * SPL
    let brow = b.let_(Op::MulI32(Opnd::Val(bs), Opnd::I32(gap)));
    let row = b.let_(Op::AddI32(Opnd::Val(split), Opnd::Val(brow)));
    let r256 = b.let_(Op::MulI32(Opnd::Val(row), Opnd::I32(256)));
    let l4 = b.let_(Op::ShlI32(Opnd::Val(lane), Opnd::I32(2)));
    let off = b.let_(Op::AddI32(Opnd::Val(r256), Opnd::Val(l4)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
    let v = b.c_i32(1);
    b.store_global(q, 0, v, amdgpu_dsl::MemTy::B32, false);
    b.finish()
}

#[test]
fn a_pow2_split_of_the_workgroup_id_is_a_mixed_radix() {
    assert!(proven(&split_kernel(true, 1)), "{}", verify_ir(&split_kernel(true, 1)).render());
    assert!(proven(&split_kernel(true, 2)));
}

#[test]
fn without_the_pow2_fact_or_with_overlapping_batches_it_is_not_proven() {
    assert!(!proven(&split_kernel(false, 1)), "1 << lg == SPL is the host contract: without it the split is unbounded");
    // gap 0: every batch writes rows [0, SPL)
    assert!(!proven(&split_kernel(true, 0)));
}

/// grid x = ceil(S / 128) (S = ka 0x10); tiles in reverse: out[(ceil(S / 128) - 1 - x) * stride + lane * 4].
fn reversed_kernel(stride: i32) -> Function {
    let mut b = FnBuilder::new("gw_reversed", 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("s", DType::I32, 0x10, 4);
    b.ka_range(0x10, 1, 1 << 20);
    b.grid_is(0, Ext::ka(0x10).ceil_div(128));
    b.arg_tensor("out", Ext::ka(0x10).ceil_div(128) * Ext::c(256));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let s = b.let_(Op::KernargI32 { offset: 0x10 });
    let lane = b.let_(Op::WorkitemId);
    let x = b.let_(Op::WorkgroupIdX);
    let s127 = b.let_(Op::AddI32(Opnd::Val(s), Opnd::I32(127)));
    let nt = b.let_(Op::LshrI32(Opnd::Val(s127), Opnd::I32(7)));
    let negx = b.let_(Op::MulI32(Opnd::Val(x), Opnd::I32(-1)));
    let t0 = b.let_(Op::AddI32(Opnd::Val(nt), Opnd::Val(negx)));
    let t = b.let_(Op::AddI32(Opnd::Val(t0), Opnd::I32(-1)));
    let ts = b.let_(Op::MulI32(Opnd::Val(t), Opnd::I32(stride)));
    let l4 = b.let_(Op::ShlI32(Opnd::Val(lane), Opnd::I32(2)));
    let off = b.let_(Op::AddI32(Opnd::Val(ts), Opnd::Val(l4)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(p), byte_off: Opnd::Val(off) });
    let v = b.c_i32(1);
    b.store_global(q, 0, v, amdgpu_dsl::MemTy::B32, false);
    b.finish()
}

#[test]
fn a_reversed_tile_order_is_a_mirrored_digit() {
    assert!(proven(&reversed_kernel(256)), "{}", verify_ir(&reversed_kernel(256)).render());
}

#[test]
fn an_overlapping_reversed_order_is_not_proven() {
    // stride 128 < 64 lanes x 4 bytes
    assert!(!proven(&reversed_kernel(128)));
}
