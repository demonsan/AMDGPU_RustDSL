//! The LDS bounds prover on `AndI32` of two constants (a slot computed as `gen & 1` of a constant generation): the exact value, not the
//! larger operand (sched's MLA roles prologue: `(2 & 1) * 8192 + base` was bounded with slot 2 and reported out of the segment).

use amdgpu_codegen::gpu_verify::{verify_ir, Severity};
use amdgpu_dsl::{FnBuilder, MemTy, Op, Opnd};

#[test]
fn and_of_two_constants_is_exact_in_the_lds_bounds() {
    let mut b = FnBuilder::new("and_consts", 64);
    b.lds(1024);
    let tid = b.let_(Op::WorkitemId);
    let c2 = b.c_i32(2);
    let c1 = b.c_i32(1);
    let slot = b.let_(Op::AndI32(Opnd::Val(c2), Opnd::Val(c1)));
    let base = b.let_(Op::MulI32(Opnd::Val(slot), Opnd::I32(512)));
    let t4 = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let lane = b.let_(Op::AndI32(Opnd::Val(t4), Opnd::I32(508)));
    let a = b.let_(Op::AddI32(Opnd::Val(base), Opnd::Val(lane)));
    let one = b.c_i32(1);
    b.store_lds(a, one, MemTy::B32, false);
    let r = verify_ir(&b.finish());
    let lds: Vec<String> = r.findings.iter().filter(|f| f.severity == Severity::Error && f.code.starts_with("GV-LDS")).map(|f| format!("[{}] {}", f.code, f.msg)).collect();
    assert!(lds.is_empty(), "{}", lds.join("\n"));
}
