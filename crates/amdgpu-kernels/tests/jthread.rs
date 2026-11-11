//! Two-tile loop with one exit per tile and a drain chosen by a flag set right before each exit (the vendor-FMHA two-tile pass,
//! docs/fmha-replica.md): the CFG pass `jthread` (lowering token) sends each exit edge straight to its drain.

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, Op, Opnd};

/// out[tid] = 1.0 when the loop left after an even tile (idx = 1 on that exit), 2.0 after an odd one; the loop walks n tiles two at a time.
pub fn two_exit_kernel(tokens: &[&str]) -> Function {
    let mut b = FnBuilder::new("probe_two_exit", 64);
    b.arg("o", DType::Ptr, 0x00, 8);
    b.arg("n", DType::I32, 0x08, 4);
    b.arg("pad", DType::I32, 0x0C, 4);
    b.arg_tensor("o", Ext::c(64 * 4));
    b.ka_range(0x08, 1, 64);
    b.lower_opts(tokens);
    let po = b.let_(Op::KernargPtr { offset: 0x00 });
    let n = b.let_(Op::KernargI32 { offset: 0x08 });
    let tid = b.let_(Op::WorkitemId);
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0)), (DType::I32, Opnd::I32(0))]);
    let (it, idx) = (vars[0], vars[1]);
    // tile it (slot 0)
    b.set_var(idx, Opnd::I32(1));
    let it1 = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    let c1 = b.cmp(CmpPred::Ge, CmpTy::I32, it1, n);
    b.break_if(c1);
    // tile it + 1 (slot 1)
    b.set_var(idx, Opnd::I32(0));
    let it2 = b.bind_i32(amdgpu_dsl::varx(it) + 2);
    b.set_var(it, it2);
    let c2 = b.cmp(CmpPred::Ge, CmpTy::I32, it2, n);
    b.break_if(c2);
    b.end_loop();
    let one = b.c_i32(1);
    let w = b.cmp(CmpPred::Eq, CmpTy::I32, Opnd::Var(idx), one);
    let off = b.bind_i32(ev(tid) << 2);
    let q = b.ptr_add(po, off);
    b.begin_if(w);
    let a = b.c_f32(1.0);
    b.store_f32(q, 0, a);
    b.else_();
    let c = b.c_f32(2.0);
    b.store_f32(q, 0, c);
    b.end_if();
    b.finish()
}

fn dump(m: &amdgpu_codegen::mir::MFunction) -> String {
    let mut s = String::new();
    for bl in &m.blocks {
        s.push_str(&format!("B{}:\n", bl.label));
        for i in &bl.insts {
            s.push_str(&format!("    {:?} dst={:?} srcs={:?} imm={}\n", i.op, i.dst, i.srcs, i.imm));
        }
        s.push_str(&format!("    -> {:?}\n", bl.term));
    }
    s
}

#[test]
fn dump_two_exit_mir() {
    let f = two_exit_kernel(&[]);
    let m = amdgpu_codegen::compile_mir(&f).unwrap_or_else(|e| panic!("{e}"));
    if std::env::var("JTHREAD_DUMP").is_ok() {
        println!("{}", dump(&m));
    }
    assert!(!m.blocks.is_empty());
}
