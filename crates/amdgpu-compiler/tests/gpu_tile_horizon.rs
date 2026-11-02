//! Audit of the v1.2 periodic-counter gate: can a marker be affine/periodic yet race
//! beyond the simulated horizon, and is the `tile_trust_periodic` escape listed/counted?

use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}
fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}

/// Two-role double buffer; `slot_mode` 0 = normal, 1 = affine large-period perturbation
/// (normal slot + ((nxt >> 8) & 1)), 2 = non-affine `Select` late perturbation.
fn ring(slot_mode: u8, version_mode: u8, trust_periodic: bool) -> Function {
    let mut b = FnBuilder::new(if trust_periodic { "ring_trust" } else { "ring_gate" }, 128);
    b.lds(8192);
    b.arg("n", DType::I32, 0, 4);
    b.lds_ring("buf", 0, 1024, 2);
    if trust_periodic {
        b.tile_trust_periodic("the protocol repeats every 2 iterations by construction");
    }
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
    let slot_n = b.let_(Op::AndI32(Opnd::Val(nxt), Opnd::I32(1)));
    let slot_c = b.let_(Op::AndI32(Opnd::Var(it), Opnd::I32(1)));
    let c1024 = b.c_i32(1024);
    b.begin_if(Opnd::Val(is_p));
    let slot_p = match slot_mode {
        0 => slot_n,
        1 => {
            // (nxt >> 8) & 1 is 0 for the first 256 iterations; adding it flips the producer slot only
            // then. The expression is affine/bitwise of the counter, so the v1.2 gate accepts it.
            let c8 = b.c_i32(8);
            let sh = b.let_(Op::LshrI32(Opnd::Val(nxt), Opnd::Val(c8)));
            let bit = b.let_(Op::AndI32(Opnd::Val(sh), Opnd::I32(1)));
            b.add_i32(slot_n, bit)
        }
        2 => {
            let c5 = b.c_i32(5);
            let cond = b.cmp_lt_i32(Opnd::Var(it), c5);
            let bad = b.let_(Op::NotI1(Opnd::Val(cond)));
            b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(bad), a: Opnd::Val(slot_c), b: Opnd::Val(slot_n) })
        }
        3 => {
            // non-affine Select, but the perturbation only starts at iteration 256 (beyond the 10-iteration horizon)
            let c256 = b.c_i32(256);
            let cond = b.cmp_lt_i32(Opnd::Var(it), c256);
            let bad = b.let_(Op::NotI1(Opnd::Val(cond)));
            b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(bad), a: Opnd::Val(slot_c), b: Opnd::Val(slot_n) })
        }
        _ => unreachable!(),
    };
    let ver_p = match version_mode {
        0 => nxt,
        1 => {
            // SHL is ignored by period_bits: the version sequence 0,256,512,... has no finite period
            let c8 = b.c_i32(8);
            b.shl_i32(nxt, Opnd::Val(c8))
        }
        2 => {
            // MUL by a constant is ignored by period_bits as well
            let c256 = b.c_i32(256);
            b.mul_i32(nxt, c256)
        }
        3 => {
            // XOR with a constant has period 512 here, but period_bits ignores marker XOR constants
            let c256 = b.c_i32(256);
            b.xor_i32(nxt, c256)
        }
        _ => unreachable!(),
    };
    b.tile_produce("buf", slot_p, ver_p);
    let base_n = b.mul_i32(slot_p, c1024);
    let wa = b.add_i32(base_n, lane16);
    b.store_lds_b128(wa, v);
    b.tile_end();
    b.else_();
    let ver_c: Opnd = match version_mode {
        0 => Opnd::Var(it),
        1 => {
            let c8 = b.c_i32(8);
            Opnd::Val(b.shl_i32(it, Opnd::Val(c8)))
        }
        2 => {
            let c256 = b.c_i32(256);
            Opnd::Val(b.mul_i32(it, c256))
        }
        3 => {
            let c256 = b.c_i32(256);
            Opnd::Val(b.xor_i32(it, c256))
        }
        _ => unreachable!(),
    };
    b.tile_consume("buf", slot_c, ver_c);
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
fn affine_period_256_is_now_rejected_by_the_period_bound() {
    // slot = (nxt & 1) + ((nxt >> 8) & 1): real period 256. v1.3 must reject because the period
    // bound is 2^9 = 512, needing 1026 simulated iterations > HORIZON_CAP 70.
    let r = verify_ir(&ring(1, 0, false));
    println!("ring affine period 256: {}", r.render());
    assert!(has(&r, "GV-TILE-UNROLL-LIMIT"), "period-256 attack not gated: {}", errors(&r));
    assert!(r.errors().iter().any(|e| e.msg.contains("period") || e.msg.contains("shift") || e.msg.contains("bit")), "message must name the period/shift: {}", r.render());
}

#[test]
fn shl_scaled_version_has_no_finite_period_but_is_accepted() {
    // version = nxt << 8: Shift-left is accepted by periodic_cone but ignored by period_bits.
    // The version sequence 0,256,512,... has no finite period, so the "period dividing 2" claim
    // printed by GV-TILE-PERIOD is mathematically false (no concrete race was found in this
    // protocol because only version differences matter).
    let r = verify_ir(&ring(0, 1, false));
    println!("ring version << 8: {}", r.render());
    assert!(!has(&r, "GV-TILE-UNROLL-LIMIT"), "SHL form unexpectedly gated: {}", errors(&r));
    assert!(r.findings.iter().any(|f| f.code == "GV-TILE-PERIOD"), "expected a GV-TILE-PERIOD info line: {}", r.render());
}

#[test]
fn mul_scaled_version_has_no_finite_period_but_is_accepted() {
    // version = nxt * 256: MUL by a constant is ignored by period_bits.
    let r = verify_ir(&ring(0, 2, false));
    println!("ring version * 256: {}", r.render());
    assert!(!has(&r, "GV-TILE-UNROLL-LIMIT"), "MUL form unexpectedly gated: {}", errors(&r));
    assert!(r.findings.iter().any(|f| f.code == "GV-TILE-PERIOD"), "expected a GV-TILE-PERIOD info line: {}", r.render());
}

#[test]
fn tile_trust_periodic_escape_is_listed_but_can_hide_a_late_race() {
    let r = verify_ir(&ring(3, 0, true));
    println!("ring Select(256) + tile_trust_periodic: {}", r.render());
    assert!(!has(&r, "GV-TILE-UNROLL-LIMIT"), "escape should skip the gate");
    let trusted: Vec<_> = r.findings.iter().filter(|f| f.code == "GV-ASSUME" && f.msg.contains("tile_trust_periodic")).collect();
    assert_eq!(trusted.len(), 1, "tile_trust_periodic must be listed once: {}", r.render());
    // The escape is a trust boundary: it can hide a real race beyond the simulated horizon.
    assert_eq!(r.errors().len(), 0, "with the trusted escape the late race is not simulated: {}", errors(&r));
}

#[test]
fn xor_marker_period_512_is_underestimated_as_2() {
    // version = nxt ^ 256: 0,257,258,...,511,0,1,... has period 512. period_bits only sees the
    // low-bit mask of the slot and prints "period dividing 2"; the gate accepts the marker.
    let r = verify_ir(&ring(0, 3, false));
    println!("ring version ^ 256: {}", r.render());
    assert!(!has(&r, "GV-TILE-UNROLL-LIMIT"), "XOR form unexpectedly gated: {}", errors(&r));
    let period = r.findings.iter().find(|f| f.code == "GV-TILE-PERIOD" && f.msg.contains("period dividing")).map(|f| f.msg.clone());
    assert!(period.is_some(), "expected a period claim: {}", r.render());
    assert!(period.unwrap().contains("dividing 2"), "the claim should be wrong (2, real 512): {}", r.render());
}
