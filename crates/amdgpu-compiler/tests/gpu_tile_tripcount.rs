//! Audit: the v1.1 tile simulator unrolls loops only 1..=4 times. A protocol that is correct for
//! the modelled iterations but breaks at iteration 5 must not be certifiable silently.

use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}
fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}

/// Double-buffered two-role ring, but at loop iteration >= 5 the producer writes the consumer's
/// current slot with the next version (a real race on a >4-iteration run).
fn late_race() -> Function {
    let mut b = FnBuilder::new("late_race", 128);
    b.lds(8192);
    b.arg("n", DType::I32, 0, 4);
    b.lds_ring("buf", 0, 1024, 2);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let c63 = b.c_i32(63);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::Val(c63)));
    let c16 = b.c_i32(16);
    let lane16 = b.mul_i32(lane, c16);
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    b.begin_if(Opnd::Val(is_p));
    b.tile_produce("buf", Opnd::I32(0), Opnd::I32(0));
    b.store_lds_b128(lane16, v);
    b.tile_end();
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let it = vars[0];
    let nxt = b.add_i32(it, Opnd::I32(1));
    let mask = Opnd::I32(1);
    let slot_n = b.let_(Op::AndI32(Opnd::Val(nxt), mask));
    let slot_c = b.let_(Op::AndI32(Opnd::Var(it), mask));
    let c1024 = b.c_i32(1024);
    b.begin_if(Opnd::Val(is_p));
    // BUG (late): at it >= 5 overwrite the consumer's current slot with the next version.
    let c5 = b.c_i32(5);
    let cond = b.cmp_lt_i32(Opnd::Var(it), c5);
    let bad = b.let_(Op::NotI1(Opnd::Val(cond)));
    let slot_bad = b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(bad), a: Opnd::Val(slot_c), b: Opnd::Val(slot_n) });
    b.tile_produce("buf", slot_bad, nxt);
    let base_n = b.mul_i32(slot_bad, c1024);
    let wa = b.add_i32(base_n, lane16);
    b.store_lds_b128(wa, v);
    b.tile_end();
    b.else_();
    let ver = Opnd::Var(it);
    b.tile_consume("buf", slot_c, ver);
    let base_c = b.mul_i32(slot_c, c1024);
    let ra = b.add_i32(base_c, lane16);
    let _ = b.load_lds_b128(ra);
    b.tile_end();
    b.end_if();
    b.wait_lds(0);
    b.barrier();
    b.set_var(it, nxt);
    let stop = b.let_(Op::CmpGeI32(Opnd::Val(nxt), Opnd::Val(n)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.finish()
}

#[test]
fn a_race_only_on_iteration_five_is_missed() {
    let r = verify_ir(&late_race());
    println!("late_race: {}", r.render());
    assert!(has(&r, "GV-TILE-UNROLL-LIMIT") || has(&r, "GV-TILE-RACE") || has(&r, "GV-TILE-STALE"), "MISSED: verifier certified a >4-iteration race: {}", errors(&r));
}
