//! G1 of docs/gfx1250-plan.md: the gfx1250 probes compile through the gfx1250 pipeline and the result assembles with the pinned gfx1250
//! assembler (backend.gfx1250.lock: the MI450 nodes' /opt/rocm/llvm/bin/clang; `GFX1250_CLANG` overrides). Skips (loudly) without it.

use std::process::Command;


fn clang() -> Option<String> {
    let c = std::env::var("GFX1250_CLANG").unwrap_or_else(|_| "/opt/rocm/llvm/bin/clang".into());
    let ok = Command::new(&c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED: no gfx1250 assembler at {c} (set GFX1250_CLANG)");
    }
    ok.then_some(c)
}

#[test]
fn the_probes_compile_for_gfx1250() {
    for k in amdgpu_kernels::gfx1250_kernel_names() {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{k}: {e}"));
        assert!(c.asm.contains(".amdhsa_wavefront_size32 1"), "{k}");
        assert!(!c.asm.contains("s_waitcnt"), "{k}: a gfx950 wait");
    }
}

#[test]
fn the_probes_assemble_for_gfx1250() {
    let Some(cl) = clang() else { return };
    let dir = std::env::temp_dir().join(format!("g1250_emit_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for k in amdgpu_kernels::gfx1250_kernel_names() {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap();
        let s = dir.join(format!("{k}.s"));
        let o = dir.join(format!("{k}.co"));
        std::fs::write(&s, &c.asm).unwrap();
        let out = Command::new(&cl)
            .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-mcode-object-version=6"])
            .arg(&s)
            .arg("-o")
            .arg(&o)
            .output()
            .unwrap();
        assert!(out.status.success(), "{k}:\n{}\n{}", String::from_utf8_lossy(&out.stderr), c.asm);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tensor_ops_are_gfx1250_only() {
    let f = amdgpu_kernels::kernel_by_name("probe_g1250_tdm").unwrap();
    let e = amdgpu_codegen::compile(&f).err().expect("gfx950 has no TDM");
    assert!(e.contains("gfx1250"), "{e}");
}

/// The exact-replay form: the vendor assembler's words (DMASK / DIM / VADDR4 zero, TH 2) are emitted verbatim; the named form is the node
/// assembler's spelling of the same operands.
#[test]
fn a_raw_tensor_op_is_emitted_verbatim() {
    let f = amdgpu_kernels::probes::g1250::probe_g1250_tdm_with(Some([0xd0310000, 0x0020_0000, 0x7c7c_2420]));
    let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap();
    assert!(c.asm.contains(".long 0xd0310000, 0x00200000, 0x7c7c2420"), "{}", c.asm);
    let n = amdgpu_codegen::compile_target(&amdgpu_kernels::kernel_by_name("probe_g1250_tdm").unwrap(), "gfx1250").unwrap();
    assert!(n.asm.lines().any(|l| l.trim_start().starts_with("tensor_load_to_lds s[")), "{}", n.asm);
    assert!(n.asm.contains("s_wait_tensorcnt 0x0") && n.asm.contains("s_barrier_signal -1"), "{}", n.asm);
}

/// The gfx1250 LLVM tail: the probes without own-only ops compile through the pinned gfx1250 llc (backend.gfx1250.lock) and assemble.
#[test]
fn the_probes_compile_through_the_gfx1250_llvm_tail() {
    let llc = amdgpu_codegen::target::gfx1250::llvm_bin().join("llc");
    if !llc.exists() {
        eprintln!("SKIPPED: no gfx1250 llc at {}", llc.display());
        return;
    }
    let Some(cl) = clang() else { return };
    let dir = std::env::temp_dir().join(format!("g1250_tail_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for k in ["probe_g1250_add", "probe_g1250_math"] {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let c = amdgpu_codegen::compile_target_with(&f, "gfx1250", "llvm-tail").unwrap_or_else(|e| panic!("{k}: {e}"));
        assert!(c.asm.contains("global_prefetch_b8"), "{k}: LLVM's prologue\n{}", c.asm);
        let s = dir.join(format!("{k}.s"));
        std::fs::write(&s, &c.asm).unwrap();
        let out = Command::new(&cl).args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-c"]).arg(&s).arg("-o").arg(dir.join(format!("{k}.o"))).output().unwrap();
        assert!(out.status.success(), "{k}:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    let _ = std::fs::remove_dir_all(&dir);
}
/// Exact-replay descriptor support: author directives replace the emitter's, and the preloaded kernarg dwords are SGPR values.
#[test]
fn kernel_descriptor_directives_and_kernarg_preload() {
    use amdgpu_dsl::{DType, Ext, FnBuilder, MemTy, Op, Opnd};
    let mut b = FnBuilder::new("probe_g1250_preload", 64);
    b.arg("out", DType::Ptr, 0, 8);
    b.arg("x", DType::I32, 8, 4);
    b.arg_tensor("out", Ext::c(64 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let x = b.let_(Op::KernargPreload { dword: 2 });
    let tid = b.let_(Op::WorkitemId);
    let v = b.let_(Op::AddI32(Opnd::Val(x), Opnd::Val(tid)));
    let off = b.let_(Op::ShlI32(Opnd::Val(tid), Opnd::I32(2)));
    let q = b.let_(Op::PtrAdd { base: Opnd::Val(po), byte_off: Opnd::Val(off) });
    b.store_global(q, 0, v, MemTy::B32, false);
    let mut f = b.finish();
    f.kd = vec![("user_sgpr_kernarg_preload_length".into(), 4), ("inst_pref_size".into(), 16)];
    let c = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap();
    assert!(c.asm.contains(".amdhsa_user_sgpr_count 6") && c.asm.contains(".amdhsa_user_sgpr_kernarg_preload_length 4"), "{}", c.asm);
    assert!(c.asm.contains(".amdhsa_inst_pref_size 16") && c.asm.contains(", s4"), "{}", c.asm);
    let mut g = f.clone();
    g.kd.push(("bogus".into(), 1));
    assert!(amdgpu_codegen::compile_target(&g, "gfx1250").is_err());
    // the IR text keeps the directives
    let t = amdgpu_dsl::ir_text::print(&f);
    assert_eq!(amdgpu_dsl::ir_text::parse(&t).unwrap().kd, f.kd);
    let Some(cl) = clang() else { return };
    let dir = std::env::temp_dir().join(format!("g1250_kd_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let s = dir.join("k.s");
    std::fs::write(&s, &c.asm).unwrap();
    let out = Command::new(&cl).args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-c"]).arg(&s).arg("-o").arg(dir.join("k.o")).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The kernarg preload's ABI inputs survive the kernel start sequence: the emitter's choice passes, the bug that hung a GPU (the start
/// sequence's scratch pair s[6:7] over preloaded dwords 4-5) is rejected.
#[test]
fn the_preloaded_kernarg_dwords_are_never_clobbered_before_their_copies() {
    let f = amdgpu_kernels::registry::kernel_by_name("probe_g1250_tdm").expect("probe");
    // compiles with the preload on (the emitter's own scratch pair passes the check)
    let asm = amdgpu_codegen::compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{e}"));
    assert!(asm.asm.contains(".amdhsa_user_sgpr_kernarg_preload_length"), "preload on");
    amdgpu_codegen::target::gfx1250::start_sequence_check(&f, None).expect("the emitter's pair");
    // mutation: the old fixed pair s[6:7], over preloaded dwords 4-5
    let e = amdgpu_codegen::target::gfx1250::start_sequence_check(&f, Some(6)).err().expect("s[6:7] holds preloaded dwords");
    assert!(e.contains("ABI input"), "{e}");
}
