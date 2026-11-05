//! `dpp_quad_perm`: `v_mov_b32_dpp quad_perm` (lane l <- lane 4 (l / 4) + perm[l % 4]), with the VALU-write -> DPP-read hazard (2 wait states).
use amdgpu_dsl::ev;
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op};

fn kernel(perm: [u8; 4]) -> Function {
    let mut b = FnBuilder::new("probe_dpp", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg_tensor("o", Ext::c(64 * 4));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let x = b.bind_i32(ev(tid) * 3 + 7); // a VALU result read right away by the DPP move
    let y = b.dpp_quad_perm(x, perm);
    let off = b.bind_i32(ev(tid) << 2);
    let p = b.ptr_add(po, off);
    let yf = b.bitcast_f32(y);
    b.store_f32(p, 0, yf);
    b.finish()
}

#[test]
fn quad_perm_emits_dpp_with_the_hazard_padding() {
    let f = kernel([1, 0, 3, 2]);
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
    let s = amdgpu_compiler::compile(&f).expect("compile").asm;
    let lines: Vec<&str> = s.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with(';')).collect();
    let i = lines.iter().position(|l| l.starts_with("v_mov_b32_dpp")).unwrap_or_else(|| panic!("no dpp move:\n{s}"));
    assert!(lines[i].contains("quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"), "{}", lines[i]);
    // the source register's writer is at least 2 wait states before the DPP read
    let src = lines[i].split(", ").nth(1).unwrap().split(' ').next().unwrap().to_string();
    let w = lines[..i].iter().rposition(|l| l.starts_with("v_") && l.split_whitespace().nth(1).map_or(false, |d| d.trim_end_matches(',') == src)).expect("writer");
    let states: usize = lines[w + 1..i].iter().map(|l| if let Some(n) = l.strip_prefix("s_nop ") { n.trim().parse::<usize>().unwrap() + 1 } else { 1 }).sum();
    assert!(states >= 2, "VALU write -> DPP read needs 2 wait states: {:?}", &lines[w..=i]);
    std::fs::write(std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("dpp.s"), &s).ok();
    let e = amdgpu_compiler::gpu_verify::analyse(&f).expect("analyse");
    assert!(e.errors().is_empty(), "{:?}", e.errors().iter().map(|x| &x.msg).collect::<Vec<_>>());
}

#[test]
fn quad_perm_control_encoding() {
    for (perm, want) in [([2u8, 3, 0, 1], "quad_perm:[2,3,0,1]"), ([0, 0, 0, 0], "quad_perm:[0,0,0,0]"), ([3, 2, 1, 0], "quad_perm:[3,2,1,0]")] {
        let s = amdgpu_compiler::compile(&kernel(perm)).expect("compile").asm;
        assert!(s.contains(want), "{want}:\n{s}");
    }
}