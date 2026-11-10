//! `buffer_load_b128x2`: two `buffer_load_dwordx4` straight into the halves of one 8-dword fragment (no v_mov copies, two vmcnt entries).
use amdgpu_dsl::ev;
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op};

fn kernel(x2: bool) -> Function {
    let mut b = FnBuilder::new("probe_buf_b128x2", 64);
    b.arg("a", DType::Ptr, 0, 8);
    b.arg_tensor("a", Ext::c(64 * 64));
    let pa = b.let_(Op::KernargPtr { offset: 0 });
    let r = b.buffer_rsrc_sized(pa, 64 * 64);
    let tid = b.let_(Op::WorkitemId);
    let voff = b.bind_i32(ev(tid) << 5);
    let f = if x2 {
        b.buffer_load_b128x2(r, voff, 0, 0, 16)
    } else {
        let lo = b.buffer_load_b128(r, voff, 0, 0);
        let hi = b.buffer_load_b128(r, voff, 0, 16);
        b.cat_frag8(lo, hi)
    };
    let z = b.frag_zero(DType::Frag16);
    let _ = b.mfma_fp8_32x32x64(f, f, z);
    b.finish()
}

#[test]
fn the_two_halves_land_in_one_register_range_without_copies() {
    let f = kernel(true);
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
    let s = amdgpu_codegen::compile(&f).expect("compile").asm;
    let loads: Vec<&str> = s.lines().map(|l| l.trim()).filter(|l| l.starts_with("buffer_load_dwordx4")).collect();
    assert_eq!(loads.len(), 2, "{s}");
    let base = |l: &str| -> u32 { l.split_whitespace().nth(1).unwrap().trim_start_matches("v[").split(':').next().unwrap().parse().unwrap() };
    assert_eq!(base(loads[1]), base(loads[0]) + 4, "the high half is the next 4 registers: {loads:?}");
    assert!(loads[1].contains("offset:16"), "{}", loads[1]);
    let movs = s.lines().filter(|l| l.trim_start().starts_with("v_mov_b32")).count();
    let cat = amdgpu_codegen::compile(&kernel(false)).expect("compile").asm;
    let movs_cat = cat.lines().filter(|l| l.trim_start().starts_with("v_mov_b32")).count();
    assert!(movs + 8 <= movs_cat, "no copy into the fragment: {movs} v_mov vs {movs_cat} with cat_frag8");
    let r = amdgpu_codegen::gpu_verify::analyse(&f).expect("analyse");
    assert!(r.errors().is_empty(), "{:?}", r.errors().iter().map(|e| &e.msg).collect::<Vec<_>>());
}
