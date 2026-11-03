//! Wave-uniform select of buffer descriptors / pointers (`select_uniform_rsrc`, `select_uniform_ptr`): lowered to `s_cmp` + `s_cselect_b64` (x2 for the
//! 128-bit descriptor) without a branch or a VGPR round trip, and proven by the GPU-semantics verifier against BOTH arms (a lying arm is rejected).
use amdgpu_dsl::tile::ev;
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd};

/// Kernarg layout: 0x00 ptr A, 0x08 ptr B, 0x10 ptr O, 0x18 i32 n (1..4), 0x1C i32 unused.
fn prologue(b: &mut FnBuilder, a_bytes: i64, b_bytes: i64) -> (amdgpu_dsl::ValueId, amdgpu_dsl::ValueId, amdgpu_dsl::ValueId, amdgpu_dsl::ValueId, amdgpu_dsl::ValueId) {
    b.arg("a", DType::Ptr, 0x00, 8);
    b.arg("bb", DType::Ptr, 0x08, 8);
    b.arg("o", DType::Ptr, 0x10, 8);
    b.arg("n", DType::I32, 0x18, 4);
    b.arg("pad", DType::I32, 0x1C, 4);
    b.arg_tensor("a", Ext::c(a_bytes));
    b.arg_tensor("bb", Ext::c(b_bytes));
    b.arg_tensor("o", Ext::c(256 * 4));
    b.ka_range(0x18, 1, 4);
    let pa = b.let_(Op::KernargPtr { offset: 0x00 });
    let pb = b.let_(Op::KernargPtr { offset: 0x08 });
    let po = b.let_(Op::KernargPtr { offset: 0x10 });
    let n = b.let_(Op::KernargI32 { offset: 0x18 });
    let tid = b.let_(Op::WorkitemId);
    (pa, pb, po, n, tid)
}

/// Each thread loads one dword through `select_uniform_rsrc(n < 2, rsrc(A, a_num), rsrc(B, b_num))` at byte `4 * tid + extra` and stores it.
fn rsrc_kernel(b_bytes: i64, b_num: u32, extra: u32) -> Function {
    let mut b = FnBuilder::new("probe_select_rsrc", 256);
    let (pa, pb, po, n, tid) = prologue(&mut b, 4096, b_bytes);
    let ra = b.buffer_rsrc_sized(pa, 4096);
    let rb = b.buffer_rsrc_sized(pb, b_num);
    let two = b.c_i32(2);
    let cond = b.cmp_lt_i32(n, two);
    let r = b.select_uniform_rsrc(cond, ra, rb);
    let voff = b.bind_i32(ev(tid) << 2);
    let x = b.buffer_load_f32(r, voff, Opnd::I32(0), extra);
    let p = b.ptr_add(po, voff);
    b.store_f32(p, 0, x);
    b.finish()
}

/// Each thread loads one dword through `select_uniform_ptr(n < 2, A, B) + 4 * (tid & 63) + extra` (a global load).
fn ptr_kernel(b_bytes: i64, extra: i32) -> Function {
    let mut b = FnBuilder::new("probe_select_ptr", 256);
    let (pa, pb, po, n, tid) = prologue(&mut b, 4096, b_bytes);
    let two = b.c_i32(2);
    let cond = b.cmp_lt_i32(n, two);
    let p = b.select_uniform_ptr(cond, pa, pb);
    let off = b.bind_i32(((ev(tid) & 63) << 2) + extra);
    let q = b.ptr_add(p, off);
    let x = b.load_f32(q, 0);
    let voff = b.bind_i32(ev(tid) << 2);
    let po2 = b.ptr_add(po, voff);
    b.store_f32(po2, 0, x);
    b.finish()
}

fn errors(f: &Function) -> Vec<String> {
    let r = amdgpu_compiler::gpu_verify::analyse(f).expect("analyse");
    r.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect()
}

fn asm(f: &Function) -> String {
    amdgpu_compiler::compile(f).unwrap_or_else(|e| panic!("compile: {e}")).asm
}

#[test]
fn descriptor_select_is_two_scalar_cselects_and_verifies_both_arms() {
    let f = rsrc_kernel(8192, 8192, 0);
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
    let e = errors(&f);
    assert!(e.is_empty(), "{e:?}");
    let s = asm(&f);
    let lines: Vec<&str> = s.lines().map(|l| l.trim()).collect();
    let i = lines.iter().position(|l| l.starts_with("s_cselect_b64")).unwrap_or_else(|| panic!("no s_cselect_b64:\n{s}"));
    assert!(lines[i + 1].starts_with("s_cselect_b64"), "the descriptor select is two consecutive s_cselect_b64: {} / {}", lines[i], lines[i + 1]);
    assert!(lines[..i].iter().rev().find(|l| l.starts_with("s_")).map_or(false, |l| l.starts_with("s_cmp_")), "s_cmp right before the selects");
    assert!(!s.contains("v_cndmask") && !s.contains("s_cbranch"), "no VGPR select, no branch:\n{s}");
    std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("select_rsrc.s"), &s).ok();
}

#[test]
fn a_lying_descriptor_arm_is_rejected() {
    // arm B claims 16 KB over an 8 KB allocation: the descriptor itself is rejected, whichever arm the select would take
    let e = errors(&rsrc_kernel(8192, 16384, 0));
    assert!(e.iter().any(|x| x.contains("GV-BUF-OVER")), "{e:?}");
}

#[test]
fn pointer_select_is_proven_against_both_extents() {
    // 4 * (tid & 63) + 0 + 4 <= 256: inside both A (4096) and B (256)
    let f = ptr_kernel(256, 0);
    let e = errors(&f);
    assert!(e.is_empty(), "{e:?}");
    let s = asm(&f);
    assert!(s.lines().filter(|l| l.trim().starts_with("s_cselect_b64")).count() == 1, "one s_cselect_b64 for a pointer:\n{s}");
    // +1024: fine for A (4096) but out of B's 256 bytes: the B arm must be reported
    let e = errors(&ptr_kernel(256, 1024));
    assert!(e.iter().any(|x| x.contains("GV-GLB-OOB") && x.contains("`bb`")), "the B arm overruns its extent: {e:?}");
    assert!(!e.iter().any(|x| x.contains("`a`")), "the A arm is in bounds: {e:?}");
}

#[test]
fn a_non_uniform_condition_is_a_compile_error() {
    let mut b = FnBuilder::new("probe_select_bad", 256);
    let (pa, pb, po, _n, tid) = prologue(&mut b, 4096, 4096);
    let ra = b.buffer_rsrc_sized(pa, 4096);
    let rb = b.buffer_rsrc_sized(pb, 4096);
    let k = b.c_i32(32);
    let cond = b.cmp_lt_i32(tid, k); // per-lane condition
    let r = b.select_uniform_rsrc(cond, ra, rb);
    let voff = b.bind_i32(ev(tid) << 2);
    let x = b.buffer_load_f32(r, voff, Opnd::I32(0), 0);
    let p = b.ptr_add(po, voff);
    b.store_f32(p, 0, x);
    let f = b.finish();
    assert!(amdgpu_compiler::compile(&f).is_err(), "a per-lane condition cannot select a wave-uniform descriptor");
}

#[test]
fn wave_role_select_with_the_wave_id() {
    // the GEMM shape: waves 0/1 take A, waves 2/3 take B, cond = (tid >> 6) < 2 (wave-uniform under `waveuni`)
    let mut b = FnBuilder::new("probe_select_wave", 256);
    let (pa, pb, po, _n, tid) = prologue(&mut b, 4096, 4096);
    let ra = b.buffer_rsrc_sized(pa, 4096);
    let rb = b.buffer_rsrc_sized(pb, 4096);
    let wave = b.bind_i32(ev(tid).lshr(6));
    let two = b.c_i32(2);
    let cond = b.cmp_lt_i32(wave, two);
    let r = b.select_uniform_rsrc(cond, ra, rb);
    let voff = b.bind_i32((ev(tid) & 63) << 2);
    let x = b.buffer_load_f32(r, voff, Opnd::I32(0), 0);
    let vo = b.bind_i32(ev(tid) << 2);
    let p = b.ptr_add(po, vo);
    b.store_f32(p, 0, x);
    let f = b.finish();
    assert!(errors(&f).is_empty(), "{:?}", errors(&f));
    let s = amdgpu_compiler::lower::with_opts(&["dce", "cmpfold", "scalarize", "scalarize2", "waveuni"], || amdgpu_compiler::compile(&f)).expect("compile").asm;
    assert_eq!(s.lines().filter(|l| l.trim().starts_with("s_cselect_b64")).count(), 2, "{s}");
}
