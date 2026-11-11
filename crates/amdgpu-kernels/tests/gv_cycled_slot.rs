//! gpu_verify: a select-cycled ring slot (the `Cadence::OneWindow` counter `idx < 2 ? idx + 1 : 0` and its predecessor
//! `0 < idx ? idx - 1 : 2`) bounds the LDS address `slot * stride`. The select refinement applies to a loop variable operand, not only to
//! values (the loop variable used to keep its unrefined interval inside the select's arms: GV-LDS-OOB on the bf16 single-group pipeline).

use amdgpu_dsl::{ev, varx, DType, FnBuilder, Function, I32, Op, Opnd};

fn cycled(prev: bool) -> Function {
    let mut b = FnBuilder::new("probe_cycled_slot", 64);
    b.arg("n", DType::I32, 0x00, 4);
    b.ka_range(0x00, 1, 64);
    b.lds(3 * 1024);
    let n = b.let_(Op::KernargI32 { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    let v = b.begin_loop(&[(DType::I32, Opnd::I32(0)), (DType::I32, Opnd::I32(0))]);
    let (it, idx) = (v[0], v[1]);
    let stop = b.cmp(amdgpu_dsl::CmpPred::Ge, amdgpu_dsl::CmpTy::I32, Opnd::Var(it), n);
    b.break_if(stop);
    let slot = if prev { b.bind_i32(I32::c(0).lt(varx(idx)).select_i32(varx(idx) - 1, 2)) } else { b.bind_i32(varx(idx) + 0) };
    let addr = b.bind_i32(ev(slot) * 1024 + (ev(tid) << 4));
    let z = b.c_i32(0);
    let f4 = b.frag4(z, z, z, z);
    b.store_lds_b128(addr, f4);
    let next = b.bind_i32(varx(idx).lt(2).select_i32(varx(idx) + 1, 0));
    b.set_var(idx, next);
    let it1 = b.bind_i32(varx(it) + 1);
    b.set_var(it, it1);
    b.end_loop();
    b.finish()
}

#[test]
fn a_select_cycled_slot_keeps_the_lds_address_in_bounds() {
    for prev in [false, true] {
        let r = amdgpu_codegen::gpu_verify::verify_ir(&cycled(prev));
        let oob: Vec<String> = r.errors().iter().filter(|e| e.code == "GV-LDS-OOB").map(|e| e.msg.clone()).collect();
        assert!(oob.is_empty(), "prev={prev}: {oob:?}");
    }
}
