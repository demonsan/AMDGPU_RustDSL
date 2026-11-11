//! Instruction forms of the vendor-FMHA replica (W5, docs/fmha-replica.md section 3, docs/two-weavers.md section 5.1), each opt-in per
//! kernel by a lowering token so that no shipped kernel changes: `cmpneg` (`!(a >= b)` on f32 as one `v_cmp_nge_f32`, generic compares
//! branch on vcc), `movb64` (zero pairs of a tuple as one `v_mov_b64`).

use amdgpu_dsl::ev;
use amdgpu_dsl::{CmpPred, CmpTy, DType, Ext, FnBuilder, Function, Op};

fn prologue(b: &mut FnBuilder) -> (amdgpu_dsl::ValueId, amdgpu_dsl::ValueId, amdgpu_dsl::ValueId) {
    b.arg("x", DType::Ptr, 0x00, 8);
    b.arg("o", DType::Ptr, 0x08, 8);
    b.arg_tensor("x", Ext::c(256 * 4));
    b.arg_tensor("o", Ext::c(256 * 16));
    let px = b.let_(Op::KernargPtr { offset: 0x00 });
    let po = b.let_(Op::KernargPtr { offset: 0x08 });
    let tid = b.let_(Op::WorkitemId);
    (px, po, tid)
}

/// out[tid] = 1 when some lane of the wave has !(x[tid] >= 0.5) (NaN counts), else untouched.
fn nge_kernel(tokens: &[&str]) -> Function {
    let mut b = FnBuilder::new("probe_cmp_nge", 256);
    let (px, po, tid) = prologue(&mut b);
    b.lower_opts(tokens);
    let off = b.bind_i32(ev(tid) << 2);
    let p = b.ptr_add(px, off);
    let x = b.load_f32(p, 0);
    let half = b.c_f32(0.5);
    let ge = b.cmp(CmpPred::Ge, CmpTy::F32, x, half);
    let nge = b.let_(Op::NotI1(ge.into()));
    let any = b.any_lane(nge);
    b.begin_if(any);
    let q = b.ptr_add(po, off);
    let one = b.c_f32(1.0);
    b.store_f32(q, 0, one);
    b.end_if();
    b.finish()
}

/// out[tid] = a zero Frag4.
fn zero_kernel(tokens: &[&str]) -> Function {
    let mut b = FnBuilder::new("probe_zero_pairs", 256);
    let (_px, po, tid) = prologue(&mut b);
    b.lower_opts(tokens);
    let z = b.frag_zero(DType::Frag4);
    let off = b.bind_i32(ev(tid) << 4);
    let q = b.ptr_add(po, off);
    b.store_f32x4(q, 0, z);
    b.finish()
}

fn asm(f: &Function) -> String {
    amdgpu_codegen::compile(f).unwrap_or_else(|e| panic!("{}: {e}", f.name)).asm
}

fn strict_ok(f: &Function) {
    let r = amdgpu_codegen::gpu_verify::analyse(f).expect("analyse");
    let e: Vec<String> = r.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
    assert!(e.is_empty(), "{}: {e:?}", f.name);
}

#[test]
fn cmpneg_lowers_a_negated_f32_ge_to_one_v_cmp_nge() {
    let base = asm(&nge_kernel(&[]));
    assert!(!base.contains("v_cmp_nge_f32"), "without the token nothing changes:\n{base}");
    let f = nge_kernel(&["cmpneg"]);
    let a = asm(&f);
    assert!(a.contains("v_cmp_nge_f32"), "{a}");
    assert!(!a.contains("v_xor_b32"), "no 0/1 materialisation + xor:\n{a}");
    strict_ok(&f);
}

#[test]
fn movb64_zeroes_tuple_pairs_with_v_mov_b64() {
    let base = asm(&zero_kernel(&[]));
    assert!(!base.contains("v_mov_b64"), "without the token nothing changes:\n{base}");
    let f = zero_kernel(&["movb64"]);
    let a = asm(&f);
    assert_eq!(a.matches("v_mov_b64").count(), 2, "a Frag4 is two pairs:\n{a}");
    strict_ok(&f);
}
