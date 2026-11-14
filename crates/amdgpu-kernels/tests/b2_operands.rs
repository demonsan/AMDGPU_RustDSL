//! W3 B2 (fmha-replica.md R2 findings), opt-in lowering tokens:
//! * `sgpr2`: a wave-uniform value in an SGPR is src0 of the commutative VOP2 ops (and of v_sub as the minuend) instead of being copied
//!   to a VGPR first (`v_mov_b32 v, s`);
//! * `m0reuse`: the M0 reuse of LDS-DMA issues (one M0 write per group) on any kernel, not only the GEMM profile.

use amdgpu_dsl::{ev, BinOp, DType, Ext, FnBuilder, Function, Op, Opnd};

/// out[tid] = max_u32(n, tid) - min_u32(n, tid) | (n - tid)   (n a kernarg, uniform)
pub fn uniform_operand_kernel(tokens: &[&str]) -> Function {
    let mut b = FnBuilder::new("probe_sgpr2", 64);
    b.arg("i", DType::Ptr, 0x00, 8);
    b.arg("o", DType::Ptr, 0x08, 8);
    b.arg("n", DType::I32, 0x10, 4);
    b.arg("pad", DType::I32, 0x14, 4);
    b.arg_tensor("i", Ext::c(4));
    b.arg_tensor("o", Ext::c(64 * 4));
    b.ka_range(0x10, 0, 1 << 20);
    b.lower_opts(&[&["scalarize"][..], tokens].concat());
    let po = b.let_(Op::KernargPtr { offset: 0x08 });
    let n = b.let_(Op::KernargI32 { offset: 0x10 });
    let tid = b.let_(Op::WorkitemId);
    let mx = b.let_(Op::Bin(BinOp::MaxU32, n.into(), tid.into()));
    let mn = b.let_(Op::Bin(BinOp::MinU32, tid.into(), n.into()));
    let df = b.let_(Op::Bin(BinOp::SubU32, n.into(), tid.into()));
    let a = b.let_(Op::Bin(BinOp::SubU32, mx.into(), mn.into()));
    let r = b.let_(Op::Bin(BinOp::OrB32, a.into(), df.into()));
    let off = b.bind_i32(ev(tid) << 2);
    let q = b.ptr_add(po, off);
    let f = b.let_(Op::BitcastF32(Opnd::Val(r)));
    b.store_f32(q, 0, f);
    b.finish()
}

fn asm(f: &Function) -> String {
    amdgpu_codegen::compile(f).unwrap_or_else(|e| panic!("{e}")).asm
}

#[test]
fn sgpr2_reads_the_uniform_operand_from_its_sgpr() {
    let base = amdgpu_codegen::without_scalar_defaults(|| asm(&uniform_operand_kernel(&[])));
    let s2 = asm(&uniform_operand_kernel(&["sgpr2"]));
    let has = |a: &str, re: &str| a.lines().any(|l| l.trim().starts_with(re) && l.contains(", s"));
    for mn in ["v_max_u32_e32", "v_min_u32_e32", "v_sub_u32_e32"] {
        assert!(!has(&base, mn), "{mn} with an SGPR operand without the token:\n{base}");
        assert!(has(&s2, mn), "{mn} with an SGPR operand:\n{s2}");
    }
    let movs = |a: &str| a.lines().filter(|l| l.trim().starts_with("v_mov_b32_e32") && l.contains(", s")).count();
    assert!(movs(&s2) < movs(&base), "fewer SGPR->VGPR copies ({} vs {})", movs(&s2), movs(&base));
    let f = uniform_operand_kernel(&["sgpr2"]);
    let r = amdgpu_codegen::gpu_verify::analyse(&f).expect("analyse");
    assert!(r.errors().is_empty());
}

/// Writes base / sgpr2 asm for tools/sgpr2_probe.sh.
#[test]
#[ignore]
fn write_sgpr2_asm() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/sgpr2");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, tok) in [("base", &[][..]), ("sgpr2", &["sgpr2"][..])] {
        std::fs::write(dir.join(format!("{name}.s")), asm(&uniform_operand_kernel(tok))).unwrap();
    }
}

#[test]
fn m0reuse_token_turns_on_the_m0_reuse_pass() {
    let Some(mut f) = amdgpu_kernels::kernel_by_name("fmha_fwd_hd128_mxfp8_v3_replica") else { return };
    let m0 = |a: &str| a.lines().filter(|l| l.trim().starts_with("s_mov_b32 m0") || l.trim().starts_with("s_add_u32 m0")).count();
    let before = m0(&asm(&f));
    f.lower_opts.push("m0reuse".into());
    let after = m0(&asm(&f));
    assert!(after <= before, "m0 writes {before} -> {after}");
    eprintln!("replica m0 writes: {before} -> {after}");
}
