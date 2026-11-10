//! Independent adversarial audit of the v4 four-axis protocol model (gpu_protocol).
//! Complements the shipped tests with mutations/axis-separation and multi-producer/consumer cases.

use amdgpu_codegen::gpu_verify::{verify_ir, Ledger, Report, Severity};
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd, Stmt};

fn has_alias(r: &Report, code: &str) -> bool {
    r.findings.iter().any(|f| f.severity == Severity::Error && (f.code == code || f.alias == code))
}
fn errors(r: &Report) -> String {
    r.errors().iter().map(|e| format!("[{}|{}] {}", e.code, e.alias, e.msg)).collect::<Vec<_>>().join("\n")
}

/// LDS: one producer role writes, optional completion wait, optional barrier publication, one reader.
fn lds_axes(wait: bool, barrier: bool, read_gen: i32) -> Function {
    let mut b = FnBuilder::new("lds_axes", 128);
    b.lds(4096);
    b.lds_tile("t", 0, 1024);
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    b.begin_if(Opnd::Val(is_p));
    b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
    b.store_lds_b128(Opnd::I32(0), v);
    b.tile_end();
    b.end_if();
    if wait {
        b.wait_lds(0);
    }
    if barrier {
        b.barrier();
    }
    b.begin_if(Opnd::Val(is_p));
    b.else_();
    b.tile_consume("t", Opnd::I32(0), Opnd::I32(read_gen));
    let _ = b.load_lds_b128(Opnd::I32(0));
    b.tile_end();
    b.end_if();
    b.finish()
}

/// GLOBAL region: one producer, completion/release/acquire switches, one reader.
fn global_axes(wait: bool, release: bool, acquire: bool, read_gen: i32) -> Function {
    let mut b = FnBuilder::new("global_axes", 128);
    b.arg("buf", DType::Ptr, 0, 8);
    b.arg_tensor("buf", Ext::c(8192));
    b.buf_region("r", "buf", Ext::c(0), Ext::c(8192));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(63)));
    let off = b.mul_i32(lane, Opnd::I32(4));
    let q = b.ptr_add(p, off);
    b.begin_if(Opnd::Val(is_p));
    b.buf_produce("r", Opnd::I32(0));
    b.store_f32(q, 0, Opnd::F32(1.0));
    b.tile_end();
    b.end_if();
    if wait { b.wait_vm(0); }
    if release { b.fence_release(); }
    b.barrier();
    if acquire { b.fence_acquire(); }
    b.begin_if(Opnd::Val(is_p));
    b.else_();
    b.buf_consume("r", Opnd::I32(read_gen));
    let _ = b.load_f32(q, 0);
    b.tile_end();
    b.end_if();
    b.finish()
}

/// Two producer roles (P writes low half, C writes high half) and two consumer roles. Same generation
/// is a cooperative fill; different generations are a write-write/read-generation violation.
fn coop_global(same_gen: bool) -> Function {
    let mut b = FnBuilder::new(if same_gen { "coop_global_same" } else { "coop_global_diff" }, 256);
    b.arg("buf", DType::Ptr, 0, 8);
    b.arg_tensor("buf", Ext::c(8192));
    b.buf_region("r", "buf", Ext::c(0), Ext::c(8192));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c128 = b.c_i32(128);
    let is_a = b.cmp_lt_i32(t, c128);
    let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(127)));
    let off = b.mul_i32(lane, Opnd::I32(4));
    let low = b.ptr_add(p, off);
    let hi_off = b.add_i32(off, Opnd::I32(4096));
    let high = b.ptr_add(p, hi_off);
    b.begin_if(Opnd::Val(is_a));
    b.buf_produce("r", Opnd::I32(0));
    b.store_f32(low, 0, Opnd::F32(1.0));
    b.tile_end();
    b.else_();
    b.buf_produce("r", Opnd::I32(if same_gen { 0 } else { 1 }));
    b.store_f32(high, 0, Opnd::F32(2.0));
    b.tile_end();
    b.end_if();
    b.wait_vm(0);
    b.fence_release();
    b.barrier();
    b.fence_acquire();
    b.begin_if(Opnd::Val(is_a));
    b.buf_consume("r", Opnd::I32(0));
    let _ = b.load_f32(low, 0);
    b.tile_end();
    b.else_();
    b.buf_consume("r", Opnd::I32(0));
    let _ = b.load_f32(high, 0);
    b.tile_end();
    b.end_if();
    b.finish()
}

/// `buf_trust` is an escape: the region's protocol is NOT checked. If it is used on a region that
/// has no completion/release/acquire at all, the certificate must still list and count it.
fn region_trust_escape() -> Function {
    let mut b = FnBuilder::new("region_trust_escape", 128);
    b.arg("buf", DType::Ptr, 0, 8);
    b.arg_tensor("buf", Ext::c(8192));
    b.buf_region("r", "buf", Ext::c(0), Ext::c(8192));
    b.buf_trust("r", "host launch order / cross-kernel visibility is argued outside the verifier");
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = b.let_(Op::WorkitemId);
    let c64 = b.c_i32(64);
    let is_p = b.cmp_lt_i32(t, c64);
    let _z = b.c_i32(0);
    b.begin_if(Opnd::Val(is_p));
    b.buf_produce("r", Opnd::I32(0));
    b.store_f32(p, 0, Opnd::F32(1.0));
    b.tile_end();
    b.buf_consume("r", Opnd::I32(0));
    let _ = b.load_f32(p, 0);
    b.tile_end();
    b.end_if();
    b.finish()
}

#[test]
fn a_lds_barrier_without_completion_is_no_completion_not_accepted() {
    let r = verify_ir(&lds_axes(false, true, 0));
    println!("lds barrier no wait: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-NO-COMPLETION"), "{}", errors(&r));
}

#[test]
fn b_lds_completion_without_publication_is_not_visible() {
    let r = verify_ir(&lds_axes(true, false, 0));
    println!("lds wait no barrier: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-UNPUBLISHED"), "{}", errors(&r));
}

#[test]
fn b_global_wait_without_release_is_not_visible() {
    let r = verify_ir(&global_axes(true, false, true, 0));
    println!("global no release: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-NOT-VISIBLE"), "{}", errors(&r));
}

#[test]
fn b_global_release_without_acquire_is_not_visible() {
    let r = verify_ir(&global_axes(true, true, false, 0));
    println!("global no acquire: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-NOT-VISIBLE"), "{}", errors(&r));
}

#[test]
fn c_barrier_with_stale_generation_is_stale_gen() {
    let r = verify_ir(&lds_axes(true, true, 1));
    println!("lds stale gen: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-STALE-GEN"), "{}", errors(&r));
}

#[test]
fn d_next_generation_overlap_is_a_race() {
    // copy the shipped global-loop race shape: producer writes it+1 while reader reads it
    let f = {
        // reuse the official test helper indirectly: build a simple two-phase loop
        let mut b = FnBuilder::new("g_race", 128);
        b.arg("buf", DType::Ptr, 0, 8);
        b.arg_tensor("buf", Ext::c(8192));
        b.buf_region("r", "buf", Ext::c(0), Ext::c(8192));
        let p = b.let_(Op::KernargPtr { offset: 0 });
        let t = b.let_(Op::WorkitemId);
        let c64 = b.c_i32(64);
        let is_p = b.cmp_lt_i32(t, c64);
        let lane = b.let_(Op::AndI32(Opnd::Val(t), Opnd::I32(63)));
        let off = b.mul_i32(lane, Opnd::I32(4));
        let q = b.ptr_add(p, off);
        b.begin_if(Opnd::Val(is_p));
        b.buf_produce("r", Opnd::I32(0));
        b.store_f32(q, 0, Opnd::F32(1.0));
        b.tile_end();
        b.end_if();
        b.wait_vm(0); b.fence_release(); b.barrier(); b.fence_acquire();
        let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
        let it = vars[0];
        let nxt = b.add_i32(it, Opnd::I32(1));
        b.begin_if(Opnd::Val(is_p));
        b.buf_produce("r", nxt);
        b.store_f32(q, 0, Opnd::F32(2.0));
        b.tile_end();
        b.else_();
        b.buf_consume("r", it);
        let _ = b.load_f32(q, 0);
        b.tile_end();
        b.end_if();
        b.wait_vm(0); b.fence_release(); b.barrier(); b.fence_acquire();
        b.set_var(it, nxt);
        let stop = b.let_(Op::CmpGeI32(Opnd::Val(nxt), Opnd::I32(3)));
        b.break_if(Opnd::Val(stop));
        b.end_loop();
        b.finish()
    };
    let r = verify_ir(&f);
    println!("global O3 race: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-RACE"), "{}", errors(&r));
}

#[test]
fn e_roles_with_inconsistent_generation_are_stale_gen() {
    let r = verify_ir(&lds_axes(true, true, 1));
    assert!(has_alias(&r, "GV-PROTO-STALE-GEN"), "{}", errors(&r));
    let g = verify_ir(&global_axes(true, true, true, 1));
    assert!(has_alias(&g, "GV-PROTO-STALE-GEN"), "{}", errors(&g));
}

#[test]
fn cooperative_same_generation_two_producers_two_consumers_is_allowed() {
    let r = verify_ir(&coop_global(true));
    println!("coop global same: {}", r.render());
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.proven.get("tile-reads").copied().unwrap_or(0) >= 2, "{}", r.render());
}

#[test]
fn different_generation_overlapping_writes_are_a_race() {
    let r = verify_ir(&coop_global(false));
    println!("coop global diff: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-RACE") || has_alias(&r, "GV-PROTO-STALE-GEN"), "{}", errors(&r));
}

#[test]
fn region_trust_escape_is_listed_and_counted_but_can_hide_a_violation() {
    let r = verify_ir(&region_trust_escape());
    println!("region trust: {}", r.render());
    assert_eq!(r.ledger(), Ledger { assume_trusted: 0, tile_trust: 0, region_trust: 1, asm_opaque: 0 }, "{}", r.render());
    assert_eq!(r.errors().len(), 0, "the escape must be a trust boundary, not silently rejected: {}", errors(&r));
    assert!(r.findings.iter().any(|f| f.alias == "GV-PROTO-TRUST" && f.msg.contains("buf_trust")), "{}", r.render());
}

fn probe(name: &str) -> Function {
    amdgpu_kernels::kernel_by_name(name).unwrap_or_else(|| panic!("unknown kernel {name}"))
}
fn without(f: &Function, pred: &dyn Fn(&Stmt) -> bool) -> Function {
    let mut g = f.clone();
    let i = g.stmts().iter().position(pred).expect("statement to remove");
    g.edit_stmts(|v| {
        v.remove(i);
    });
    g
}

#[test]
fn fused_last_arriver_missing_release_is_no_completion() {
    let f = probe("probe_last_arriver");
    let r = verify_ir(&without(&f, &|s| matches!(s, Stmt::FenceRelease { .. })));
    println!("fused no release: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-NO-COMPLETION"), "{}", errors(&r));
}

#[test]
fn fused_last_arriver_missing_acquire_is_not_visible() {
    let f = probe("probe_last_arriver");
    let r = verify_ir(&without(&f, &|s| matches!(s, Stmt::FenceAcquire { .. })));
    println!("fused no acquire: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-NOT-VISIBLE"), "{}", errors(&r));
}

#[test]
fn fused_last_arriver_missing_ticket_is_arrival_mismatch() {
    let mut f = probe("probe_last_arriver");
    f.contract.last_arrivers[0].counter_arg = "result".to_string();
    let r = verify_ir(&f);
    println!("fused wrong ticket arg: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-ARRIVAL-MISMATCH"), "{}", errors(&r));
}

#[test]
fn fused_last_arriver_lite_with_plain_loads_is_not_visible() {
    let f = probe("probe_last_arriver_lite");
    let mut g = f.clone();
    let mut flat = g.stmts().to_vec();
    for s in flat.iter_mut() {
        if let Stmt::Let { op, .. } = s {
            if let Op::LoadF32Coherent { base, byte_off } = op.clone() {
                *op = Op::LoadF32 { base, byte_off };
            }
        }
    }
    g.set_stmts(flat);
    let r = verify_ir(&g);
    println!("fused lite plain loads: {}", r.render());
    assert!(has_alias(&r, "GV-PROTO-NOT-VISIBLE"), "{}", errors(&r));
}
