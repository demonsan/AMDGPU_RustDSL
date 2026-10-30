// (F8 variants) `var_asm` misuse: wrong loop-variable type and wrong input count must be rejected by DSL verify.
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
use amdgpu_safeasm::examples::rowsum_var_block;
use amdgpu_safeasm::*;
fn build(var_ty: DType, n_inputs: usize, name: &str) -> Vec<String> {
    let mut b = FnBuilder::new(name, 64);
    b.arg("o", DType::Ptr, 0, 8);
    let _o = b.let_(Op::KernargPtr { offset: 0 });
    let c0 = b.c_i32(0);
    let (c3, c4) = (b.c_i32(3), b.c_i32(4));
    let a_rs = b.frag4(c0, c0, c0, c0);
    let pa = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let pb = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let init = match var_ty {
        DType::Frag8 => Opnd::Val(pa),
        _ => Opnd::Val(a_rs),
    };
    let vars = b.begin_loop(&[(var_ty, init)]);
    let la = vars[0];
    let all = [Opnd::Val(a_rs), Opnd::Val(pa), Opnd::Val(pb), Opnd::Val(c3), Opnd::Val(c4)];
    b.var_asm(la, rowsum_var_block(), &all[..n_inputs]);
    let cond = b.cmp_lt_i32(c0, c0);
    b.break_if(cond);
    b.end_loop();
    let func = b.finish();
    amdgpu_dsl::verify(&func)
}
fn main() {
    println!("CASE pass_var_asm_misuse:");
    println!("  wrong var type  : {:?}", build(DType::Frag8, 5, "wrong_var").iter().take(3).collect::<Vec<_>>());
    println!("  4 inputs (need 5): {:?}", build(DType::Frag4, 4, "wrong_inputs").iter().take(3).collect::<Vec<_>>());
    println!("  correct         : {:?}", build(DType::Frag4, 5, "ok_loop").iter().take(3).collect::<Vec<_>>());
}
