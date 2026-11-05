//! Loop-variable semantics (docs/region-ir.md "Loop variables"): a read of a loop variable sees the last `set_var` executed on the path in the
//! current iteration (through an `end_if`: the merge), else the iteration-start value. The lowering gives the variable ONE register for the whole
//! loop, `set_var` writes it (copy or retargeted producer) and every read reads it; this test pins that on the final allocated MIR under each
//! compile profile: the read after the `if` uses exactly the physical register the `set_var` inside the `if` wrote, and nothing else writes that
//! register between the `if` and the read.
use amdgpu_compiler::compile_mir_alloc;
use amdgpu_compiler::mir::Src;
use amdgpu_dsl::{CompileProfile, DType, FnBuilder, Function, Op, Opnd};

/// loop { y = i + 1; if (i < 5) { set_var(x, y * 3) }; r = x + 1000; store r; set_var(i, i + 1); break if !(i + 1 < 10) }
fn kernel(p: Option<CompileProfile>) -> Function {
    let mut b = FnBuilder::new("var_semantics", 64);
    b.arg("out", DType::Ptr, 0, 8);
    if let Some(p) = p {
        b.compile_profile(p);
    }
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(Opnd::Val(tid), Opnd::Val(c4));
    let base = b.ptr_add(Opnd::Val(out), Opnd::Val(off));
    let c0 = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0)), (DType::I32, Opnd::Val(c0))]);
    let (x, i) = (vars[0], vars[1]);
    let c1 = b.c_i32(1);
    let y = b.add_i32(Opnd::Var(i), Opnd::Val(c1));
    let c5 = b.c_i32(5);
    let grow = b.cmp_lt_i32(Opnd::Var(i), Opnd::Val(c5));
    b.begin_if(Opnd::Val(grow));
    let c3 = b.c_i32(3);
    let nx = b.mul_i32(Opnd::Val(y), Opnd::Val(c3));
    b.set_var(x, Opnd::Val(nx));
    b.end_if();
    let k = b.c_i32(1000);
    let r = b.add_i32(Opnd::Var(x), Opnd::Val(k));
    let rf = b.let_(Op::BitcastF32(Opnd::Val(r)));
    b.store_f32(Opnd::Val(base), 0, Opnd::Val(rf));
    let inext = b.add_i32(Opnd::Var(i), Opnd::Val(c1));
    b.set_var(i, Opnd::Val(inext));
    let c10 = b.c_i32(10);
    let more = b.cmp_lt_i32(Opnd::Val(inext), Opnd::Val(c10));
    let stop = b.let_(Op::NotI1(Opnd::Val(more)));
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.finish()
}

#[test]
fn a_read_after_an_if_that_set_the_variable_reads_the_variable_register() {
    for p in [Some(CompileProfile::Vs128), Some(CompileProfile::T128), Some(CompileProfile::Gemm)] {
        let f = kernel(p);
        let (m, a) = compile_mir_alloc(&f).unwrap_or_else(|e| panic!("{p:?}: {e}"));
        let phys = |r| {
            let c = m.canon(r);
            (a.phys[c as usize], a.scalar[c as usize])
        };
        // the read: the add with the immediate 1000 (or a materialised 1000 register)
        let mut found = None;
        for (bi, blk) in m.blocks.iter().enumerate() {
            for (ii, ins) in blk.insts.iter().enumerate() {
                if ins.srcs.iter().any(|s| matches!(s, Src::Imm(1000))) {
                    let reg = ins.srcs.iter().find_map(|s| if let Src::Reg(r) = s { Some(*r) } else { None }).expect("register operand");
                    found = Some((bi, ii, phys(reg)));
                }
            }
        }
        let (rb, ri, xr) = found.unwrap_or_else(|| panic!("{p:?}: the read of x not found"));
        // writers of that register before the read: one must be in another block (the `if` body), and the read's own block writes it
        // nowhere before the read
        let writers: Vec<usize> = m
            .blocks
            .iter()
            .enumerate()
            .filter(|(bi, blk)| *bi != rb && blk.insts.iter().any(|i| i.dst.map_or(false, |d| phys(d) == xr)))
            .map(|(bi, _)| bi)
            .collect();
        assert!(!writers.is_empty(), "{p:?}: no block writes the variable register {xr:?}");
        assert!(
            !m.blocks[rb].insts[..ri].iter().any(|i| i.dst.map_or(false, |d| phys(d) == xr)),
            "{p:?}: the read's block overwrites the variable register before the read"
        );
    }
}
