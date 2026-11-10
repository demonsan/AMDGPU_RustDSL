//! Scoped branch facts of the region-tree walk (gpu_verify): inside `if n < k` a WAVE-UNIFORM `n` is known to be below `k`; a per-lane
//! condition proves nothing about the other lanes of the wave (`begin_if` is wave-level) and is not used.
use amdgpu_dsl::ev;
use amdgpu_dsl::{DType, FnBuilder, Function, Op};

fn lds_oob(f: &Function) -> usize {
    let r = amdgpu_codegen::gpu_verify::analyse(f).expect("analyse");
    r.errors().iter().filter(|e| e.code == "GV-LDS-OOB").count()
}

/// `if n < 64 { ds_read_b128 [n * 16] }` (else: `[n * 16 - 1024]` when `neg`) in a 1 KB segment; `n` is a kernarg in [0, 1000].
fn uniform(negate: bool, limit: i32) -> Function {
    let mut b = FnBuilder::new("sym_branch_uniform", 64);
    b.arg("n", DType::I32, 8, 4);
    b.ka_range(8, 0, 1000);
    b.lds(1024);
    let n = b.let_(Op::KernargI32 { offset: 8 });
    let k = b.c_i32(limit);
    let c = b.cmp_lt_i32(n, k);
    if negate {
        // if !(n < 64) {} else { read } : the else branch of the negation is the same fact
        let nc = b.let_(Op::NotI1(amdgpu_dsl::Opnd::Val(c)));
        b.if_else(nc, |_| {}, |b| {
            let a = b.bind_i32(ev(n) * 16);
            let _ = b.load_lds_b128(a);
        });
    } else {
        b.if_(c, |b| {
            let a = b.bind_i32(ev(n) * 16);
            let _ = b.load_lds_b128(a);
        });
    }
    b.finish()
}

#[test]
fn a_uniform_branch_condition_bounds_its_operand() {
    assert_eq!(lds_oob(&uniform(false, 64)), 0, "n < 64 => n * 16 + 16 <= 1024");
    assert_eq!(lds_oob(&uniform(true, 64)), 0, "the else branch of !(n < 64)");
    assert!(lds_oob(&uniform(false, 65)) > 0, "n < 65 allows n = 64: 64 * 16 + 16 > 1024");
}

#[test]
fn a_per_lane_condition_is_not_a_fact() {
    // if tid < 16 { ds_read_b128 [tid * 64] } with 256 threads: other lanes of the wave run the branch too (wave-level if): tid * 64 + 16 > 1024
    let mut b = FnBuilder::new("sym_branch_lane", 256);
    b.lds(1024);
    let tid = b.let_(Op::WorkitemId);
    let k = b.c_i32(16);
    let c = b.cmp_lt_i32(tid, k);
    let c = b.any_lane(c); // a wave-level branch on a per-lane condition
    b.if_(c, |b| {
        let a = b.bind_i32(ev(tid) * 64);
        let _ = b.load_lds_b128(a);
    });
    assert!(lds_oob(&b.finish()) > 0, "a per-lane condition must not bound the address");
}
