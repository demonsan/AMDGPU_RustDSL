//! `permlane32_swap(a, b) -> (a', b')`: one `v_permlane32_swap_b32` on copies of the operands (both are written), with the VALU-write ->
//! permlane hazard padded.
use amdgpu_dsl::ev;
use amdgpu_dsl::{DType, Ext, FnBuilder, Op};

#[test]
fn swap_emits_one_permlane_with_the_hazard_padding() {
    let mut b = FnBuilder::new("probe_permlane_swap", 64);
    b.arg("o", DType::Ptr, 0, 8);
    b.arg_tensor("o", Ext::c(64 * 8));
    let po = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let x = b.bind_i32(ev(tid) * 3);
    let y = b.bind_i32(ev(tid) + 100);
    let (x2, y2) = b.permlane32_swap(x, y);
    let off = b.bind_i32(ev(tid) << 3);
    let p = b.ptr_add(po, off);
    let fx = b.bitcast_f32(x2);
    let fy = b.bitcast_f32(y2);
    b.store_f32(p, 0, fx);
    b.store_f32(p, 4, fy);
    let f = b.finish();
    assert!(amdgpu_dsl::verify(&f).is_empty(), "{:?}", amdgpu_dsl::verify(&f));
    let s = amdgpu_codegen::compile(&f).expect("compile").asm;
    let lines: Vec<&str> = s.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with(';')).collect();
    let i = lines.iter().position(|l| l.starts_with("v_permlane32_swap")).unwrap_or_else(|| panic!("{s}"));
    assert_eq!(lines.iter().filter(|l| l.starts_with("v_permlane32_swap")).count(), 1);
    // the two copies feed it: at least 2 wait states after the last VALU writer
    let w = lines[..i].iter().rposition(|l| l.starts_with("v_")).unwrap();
    let states: usize = lines[w + 1..i].iter().map(|l| if let Some(n) = l.strip_prefix("s_nop ") { n.trim().parse::<usize>().unwrap() + 1 } else { 1 }).sum();
    assert!(states >= 2, "{:?}", &lines[w..=i]);
    let r = amdgpu_codegen::gpu_verify::analyse(&f).expect("analyse");
    assert!(r.errors().is_empty(), "{:?}", r.errors().iter().map(|e| &e.msg).collect::<Vec<_>>());
}
