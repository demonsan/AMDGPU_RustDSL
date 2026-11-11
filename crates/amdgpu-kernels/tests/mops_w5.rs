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

/// A loop over a kernarg count with uniform arithmetic: `it + 3` (addk), `n - it` and `it | 1` (s_sub / s_or), an unsigned loop exit
/// `it >= n` (s_cmp_ge_u32) and `v_mul_i32_i24` on small per-lane values.
fn salu_kernel(tokens: &[&str]) -> Function {
    use amdgpu_dsl::{varx, BinOp, Opnd};
    let mut b = FnBuilder::new("probe_salu32", 256);
    b.arg("x", DType::Ptr, 0x00, 8);
    b.arg("o", DType::Ptr, 0x08, 8);
    b.arg("n", DType::I32, 0x10, 4);
    b.arg("pad", DType::I32, 0x14, 4);
    b.arg_tensor("x", Ext::c(256 * 4));
    b.arg_tensor("o", Ext::c(256 * 4));
    b.ka_range(0x10, 1, 64);
    b.lower_opts(&[&["scalarize"][..], tokens].concat());
    let po = b.let_(Op::KernargPtr { offset: 0x08 });
    let n = b.let_(Op::KernargI32 { offset: 0x10 });
    let tid = b.let_(Op::WorkitemId);
    let v = b.begin_loop(&[(DType::I32, Opnd::I32(0)), (DType::I32, Opnd::I32(0))]);
    let (it, acc) = (v[0], v[1]);
    let left = b.let_(Op::Bin(BinOp::SubU32, n.into(), Opnd::Var(it)));
    let odd = b.let_(Op::Bin(BinOp::OrB32, Opnd::Var(it), Opnd::I32(1)));
    let s = b.bind_i32(varx(acc) + left + odd);
    b.set_var(acc, s);
    let nx = b.bind_i32(varx(it) + 3);
    b.set_var(it, nx);
    let done = b.cmp(CmpPred::Ge, CmpTy::U32, Opnd::Var(it), n);
    b.break_if(done);
    b.end_loop();
    let small = b.bind_i32(ev(tid) & 255);
    let prod = b.let_(Op::Bin(BinOp::MulI32I24, small.into(), Opnd::Var(acc)));
    let off = b.bind_i32(ev(tid) << 2);
    let q = b.ptr_add(po, off);
    let f = b.let_(Op::BitcastF32(prod.into()));
    b.store_f32(q, 0, f);
    b.finish()
}

#[test]
fn salu32_selects_the_scalar_forms() {
    let base = asm(&salu_kernel(&[]));
    for m in ["s_sub_u32", "s_or_b32", "s_addk_i32", "s_cmp_ge_u32"] {
        assert!(!base.contains(m), "{m} without the token:\n{base}");
    }
    assert!(base.contains("v_mul_i32_i24"), "the explicit op always lowers to v_mul_i32_i24:\n{base}");
    let f = salu_kernel(&["salu32"]);
    let a = asm(&f);
    for m in ["s_sub_u32", "s_or_b32", "s_addk_i32", "s_cmp_ge_u32"] {
        assert!(a.contains(m), "{m} missing:\n{a}");
    }
    strict_ok(&f);
}
