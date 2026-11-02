//! Independent safety audit of the DSL tile-library port: contracts/markers strength vs the hand
//! kernels, plus adversarial mutations of library-generated generations and a lying tensor alloc.

use amdgpu_compiler::gpu_verify::{verify_ir, Report, Severity};
use amdgpu_dsl::tile::{Dim, KArgs, Stride};
use amdgpu_dsl::{DType, Ext, FnBuilder, I32, Op, Opnd, Stmt};

fn has(r: &Report, code: &str) -> bool {
    r.findings.iter().any(|f| f.severity == Severity::Error && (f.code == code || f.alias == code))
}
fn contracts(f: &amdgpu_dsl::Function) -> usize {
    let c = &f.contract;
    c.extents.len() + c.ka_ranges.len() + c.grid_defs.len() + c.ka_defs.len() + c.sym_ranges.len()
}

#[test]
fn dsl_and_hand_kernels_have_the_same_or_stronger_certificate() {
    let pairs = [
        ("fmha_fwd_hd128_mxfp8_dsl_vs128", "fmha_fwd_hd128_mxfp8_vs128"),
        ("fmha_fwd_hd128_mxfp8_dsl_t128", "fmha_fwd_hd128_mxfp8_t128"),
        ("fmha_fwd_hd128_mxfp8_dsl_kvsplit", "fmha_fwd_hd128_mxfp8_kvsplit"),
        ("fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h", "fmha_fwd_hd128_mxfp8_kvsplit_combine8h"),
    ];
    for (dsl, hand) in pairs {
        let d = amdgpu_compiler::kernel_by_name(dsl).unwrap();
        let h = amdgpu_compiler::kernel_by_name(hand).unwrap();
        let rd = verify_ir(&d);
        let rh = verify_ir(&h);
        println!("{dsl}: errors={} ledger={:?} contracts={}", rd.errors().len(), rd.ledger(), contracts(&d));
        println!("{hand}: errors={} ledger={:?} contracts={}", rh.errors().len(), rh.ledger(), contracts(&h));
        assert!(rd.errors().is_empty(), "{dsl}: {}", rd.render());
        assert!(rh.errors().is_empty(), "{hand}: {}", rh.render());
        assert_eq!(rd.ledger().total(), rh.ledger().total(), "{dsl}: new trust escapes vs hand");
        assert_eq!(contracts(&d), contracts(&h), "{dsl}: contract set differs from hand");
        let dl = rd.proven.get("lds").copied().unwrap_or(0);
        let hl = rh.proven.get("lds").copied().unwrap_or(0);
        assert!(dl >= hl, "{dsl}: weaker LDS proof ({dl} < {hl})");
        let dc = rd.proven.get("tile-cover").copied().unwrap_or(0) + rd.proven.get("tile-reads").copied().unwrap_or(0);
        let hc = rh.proven.get("tile-cover").copied().unwrap_or(0) + rh.proven.get("tile-reads").copied().unwrap_or(0);
        assert!(dc >= hc, "{dsl}: weaker tile proof ({dc} < {hc})");
        if hc > 0 {
            assert!(dc > 0, "{dsl}: no tile markers proven");
        }
    }
}

#[test]
fn a_lying_tensor_allocation_is_rejected() {
    // The Tensor API generates `arg_tensor` from the declared dims/strides. Override the allocation to
    // 64 B while indexing row 7: the address is inside the declared tensor but outside the allocation.
    let mut b = FnBuilder::new("lying_tensor", 64);
    let mut ka = KArgs::new();
    let s = ka.dim(&mut b, "s_seq_len", 0x060, 1, i64::MAX);
    let t = ka
        .tensor("ptr_lying", 0x010, 1, vec![Dim::c(8), s], vec![Stride::Ka("s_lying_stride", 0x070), Stride::Unit])
        .alloc(Ext::c(64))
        .declare(&mut ka, &mut b);
    let off = b.bind_i32(t.offset(&[I32::c(7), I32::c(0)]));
    let p = b.ptr_add(t.ptr, off);
    let _ = b.load_f32(p, 0);
    let f = b.finish();
    let r = verify_ir(&f);
    println!("lying tensor: {}", r.render());
    assert!(has(&r, "GV-GLB-OOB"), "lying tensor allocation not rejected: {}", r.render());
}

#[test]
fn a_wrong_auto_generated_generation_is_rejected() {
    let mut f = amdgpu_compiler::kernel_by_name("fmha_fwd_hd128_mxfp8_dsl_vs128").unwrap();
    let mut changed = 0;
    for s in f.stmts.iter_mut() {
        if let Stmt::TileConsume { version, .. } = s {
            *version = Opnd::I32(1);
            changed += 1;
        }
    }
    assert!(changed > 0, "no auto-generated consume markers found");
    let r = verify_ir(&f);
    println!("wrong auto generation: changed={changed}, errors={}, findings={}", r.errors().len(), r.findings.len());
    assert!(has(&r, "GV-TILE-STALE") || has(&r, "GV-PROTO-STALE-GEN"), "wrong generation not rejected: {}", r.render());
}
