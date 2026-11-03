//! `v_mfma_scale_f32_16x16x128_f8f6f4` (P4 MXFP8 GEMM): IR statement `VarMfmaScale16` with the E8M0 scale-byte selects, lowered, hazard-padded, verified
//! and emitted (the asm is assembled by tools/asm_check on the node).
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

pub fn mfma16_demo(byte_a: u8, byte_b: u8) -> Function {
    let mut b = FnBuilder::new("probe_mfma16x16x128", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg("n", DType::I32, 8, 4);
    b.arg_tensor("o", amdgpu_dsl::Ext::c(64 * 16));
    b.ka_range(8, 1, 64);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let n = b.let_(Op::KernargI32 { offset: 8 });
    let tid = b.let_(Op::WorkitemId);
    let one = b.c_i32(0x3838_3838);
    let a = b.frag8(one, one, one, one, one, one, one, one);
    let bb = b.frag8(one, one, one, one, one, one, one, one);
    let sc = b.c_i32(0x7f7f_7f7f);
    let z = b.frag_zero(DType::Frag4);
    let c0 = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::Frag4, Opnd::Val(z))]);
    let (it, acc) = (vars[0], vars[1]);
    b.var_mfma_scale16(acc, a, bb, sc, sc, byte_a, byte_b);
    let c1 = b.c_i32(1);
    let nx = b.add_i32(it, c1);
    b.set_var(it, nx);
    let more = b.cmp_lt_i32(nx, n);
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(stop);
    b.end_loop();
    let c4 = b.c_i32(4);
    let c63 = b.c_i32(63);
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::Val(c63)));
    let off0 = b.shl_i32(lane, c4);
    let p = b.ptr_add(o, off0);
    let r = b.frag4_lane(Opnd::Var(acc), 0);
    let rf = b.bitcast_f32(r);
    b.store_f32(p, 0, rf);
    b.finish()
}

#[test]
fn scaled_mfma_16x16x128_lowers_verifies_and_emits_op_sel() {
    let f = mfma16_demo(1, 2);
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
    let r = amdgpu_compiler::gpu_verify::analyse(&f).expect("analyse");
    assert!(r.errors().is_empty(), "{}", r.render());
    let c = amdgpu_compiler::compile(&f).expect("compile");
    let line = c.asm.lines().find(|l| l.contains("v_mfma_scale_f32_16x16x128_f8f6f4")).unwrap_or_else(|| panic!("no 16x16x128 MFMA:\n{}", c.asm));
    assert!(line.contains("op_sel:[1,0,0] op_sel_hi:[0,1,0]"), "scale byte selects A=1, B=2: {line}");
    let plain = amdgpu_compiler::compile(&mfma16_demo(0, 0)).expect("compile").asm;
    assert!(!plain.contains("op_sel"), "byte 0 / 0 is the default encoding");
    std::fs::write(std::env::temp_dir().join("mfma16.s"), &c.asm).ok();
}
