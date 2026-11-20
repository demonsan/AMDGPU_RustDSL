//! Three wave roles in the tile protocol: a nested `if role == 0 { .. } else if role == 1 { .. } else { .. }` on a wave-uniform function of the
//! wave index. Each role is simulated with its own program (before: the nested else merged roles 1 and 2 into one simulated role).

use amdgpu_codegen::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::{CmpPred, CmpTy, FnBuilder, Function, MemTy, Op, Opnd};

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}
fn has_error(rep: &Report) -> bool {
    rep.findings.iter().any(|f| f.severity == Severity::Error && f.code.starts_with("GV-TILE"))
}

/// role 0 produces `t`; role 1 reads `t` after the barrier; role 2 writes `u` (clean) or `t` again (`race`) after the barrier.
fn kernel(race: bool) -> Function {
    let mut b = FnBuilder::new("roles3", 192);
    b.lds(2048);
    b.lds_tile("t", 0, 1024);
    b.lds_tile("u", 1024, 1024);
    let tid = b.let_(Op::WorkitemId);
    // the wave index (wave64 on the default target): role = tid >> 6
    let role = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::I32(6)));
    let r0 = b.cmp(CmpPred::Eq, CmpTy::I32, role, Opnd::I32(0));
    let r1 = b.cmp(CmpPred::Eq, CmpTy::I32, role, Opnd::I32(1));
    let lane = b.let_(Op::AndI32(Opnd::Val(tid), Opnd::I32(63)));
    let a = b.let_(Op::ShlI32(Opnd::Val(lane), Opnd::I32(2)));
    let au = b.let_(Op::AddI32(Opnd::Val(a), Opnd::I32(1024)));
    let one = b.c_i32(1);
    b.begin_if(Opnd::Val(r0));
    {
        b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
        b.store_lds(a, one, MemTy::B32, false);
        b.tile_end();
        b.wait_lds(0);
        b.barrier();
    }
    b.else_();
    {
        b.begin_if(Opnd::Val(r1));
        {
            b.barrier();
            b.tile_consume("t", Opnd::I32(0), Opnd::I32(0));
            let _ = b.load_lds(a, MemTy::B32);
            b.tile_end();
            b.wait_lds(0);
        }
        b.else_();
        {
            b.barrier();
            if race {
                b.tile_produce("t", Opnd::I32(0), Opnd::I32(1));
                b.store_lds(a, one, MemTy::B32, false);
            } else {
                b.tile_produce("u", Opnd::I32(0), Opnd::I32(0));
                b.store_lds(au, one, MemTy::B32, false);
            }
            b.tile_end();
            b.wait_lds(0);
        }
        b.end_if();
    }
    b.end_if();
    b.finish()
}

#[test]
fn three_roles_are_simulated_separately() {
    let r = verify_ir(&kernel(false));
    assert!(!has_error(&r), "{}", errors(&r));
}

#[test]
fn a_third_role_writing_the_tile_the_second_reads_is_a_race() {
    let r = verify_ir(&kernel(true));
    assert!(has_error(&r), "role 2 overwrites `t` while role 1 reads it in the same phase");
}
