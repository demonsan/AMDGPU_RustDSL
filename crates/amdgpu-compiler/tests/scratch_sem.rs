//! scratch: what does a read of a loop variable after set_var lower to?
use amdgpu_compiler::lower;
use amdgpu_dsl::{DType, FnBuilder, Opnd};

#[test]
fn scratch_semantics() {
    let mut b = FnBuilder::new("t", 64);
    b.arg("out", DType::Ptr, 0, 8);
    let c0 = b.c_i32(0);
    let c1 = b.c_i32(1);
    let c10 = b.c_i32(10);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(c0))]);
    let (x, i) = (vars[0], vars[1]);
    let y = b.add_i32(Opnd::Var(x), Opnd::Val(c1)); // y = x + 1
    b.set_var(x, Opnd::Val(y)); // x = y
    let r = b.add_i32(Opnd::Var(x), Opnd::Val(c10)); // read after set
    let _ = r;
    let inext = b.add_i32(Opnd::Var(i), Opnd::Val(c1));
    b.set_var(i, Opnd::Val(inext));
    let more = b.cmp_lt_i32(Opnd::Val(inext), Opnd::Val(c10));
    let nm = b.let_(amdgpu_dsl::Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(nm));
    b.end_loop();
    let f = b.finish();
    println!("{}", f.dump());
    for w in amdgpu_dsl::lint(&f) {
        println!("LINT {w}");
    }
    let m = lower::lower(&f).unwrap();
    for (bi, blk) in m.blocks.iter().enumerate() {
        for i in &blk.insts {
            println!("b{bi}: {:?} dst={:?} srcs={:?} imm={}", i.op, i.dst, i.srcs, i.imm);
        }
    }
}