//! Register views of loop variables (lane extracts, `var_slice8`, bitcasts) have SSA semantics: a view used after a write of its variable is
//! snapshotted (copied) at its definition; a view used before any later write stays a zero-cost alias (docs/weave-capture.md).
use amdgpu_codegen::{compile_mir, mir::MOp};
use amdgpu_dsl::facts::stale_view_roots;
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd};

/// out[tid] = (view of the variable, used `after_write` or before the variable is incremented)
fn kernel(after_write: bool, in_loop_use: bool) -> Function {
    let mut b = FnBuilder::new("views", 64);
    b.arg("out", DType::Ptr, 0, 8);
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.let_(Op::MulI32(Opnd::Val(tid), Opnd::Val(c4)));
    let base = b.let_(Op::PtrAdd { base: Opnd::Val(out), byte_off: Opnd::Val(off) });
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(3))]); // a uniform loop (branches are wave-level)
    let _ = tid;
    let view = b.let_(Op::BitcastF32(Opnd::Var(vars[0])));
    if in_loop_use {
        // the view is defined in the loop and used in a NESTED loop after the write: the next inner iteration must not see the write
        let n = b.add_i32(Opnd::Var(vars[0]), 1);
        b.set_var(vars[0], n);
        let inner = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
        b.store_f32(base, 0, view);
        let k = b.add_i32(Opnd::Var(inner[0]), 1);
        b.set_var(inner[0], k);
        let d = b.let_(Op::CmpGeI32(Opnd::Val(k), Opnd::I32(2)));
        b.break_if(d);
        b.end_loop();
    } else if after_write {
        let n = b.add_i32(Opnd::Var(vars[0]), 1);
        b.set_var(vars[0], n);
        b.store_f32(base, 0, view);
    } else {
        b.store_f32(base, 0, view);
        let n = b.add_i32(Opnd::Var(vars[0]), 1);
        b.set_var(vars[0], n);
    }
    let done = b.let_(Op::CmpGeI32(Opnd::Var(vars[0]), Opnd::I32(100)));
    b.break_if(done);
    b.end_loop();
    b.finish()
}

/// The defining instructions of the register the (single) global store reads its data from.
fn store_src_defs(f: &Function) -> Vec<MOp> {
    let m = compile_mir(f).unwrap();
    let mut src = None;
    for b in &m.blocks {
        for i in &b.insts {
            if i.op == MOp::GlobalStoreF32 {
                if let Some(amdgpu_codegen::mir::Src::Reg(r)) = i.srcs.get(1) {
                    src = Some(m.canon(*r));
                }
            }
        }
    }
    let s = src.expect("a global store");
    m.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.dst.map(|d| m.canon(d)) == Some(s)).map(|i| i.op).collect()
}

#[test]
fn a_view_used_before_the_write_stays_an_alias() {
    let f = kernel(false, false);
    assert!(stale_view_roots(&f).is_empty());
}

#[test]
fn a_view_used_after_the_write_is_snapshotted() {
    amdgpu_codegen::without_scalar_defaults(view_after_write_body)
}

fn view_after_write_body() {
    let f = kernel(true, false);
    let roots = stale_view_roots(&f);
    assert_eq!(roots.len(), 1, "the bitcast of the variable is a stale view");
    // the store reads a snapshot: a register defined once, by the copy at the view's definition (not the loop variable's register,
    // which is written by the loop entry and the set_var)
    assert_eq!(store_src_defs(&f), vec![MOp::VMovB32]);
    let alias = kernel(false, false);
    assert!(store_src_defs(&alias).len() >= 2, "the alias case reads the variable's own register: {:?}", store_src_defs(&alias));
}

#[test]
fn a_view_used_in_a_nested_loop_after_the_write_is_snapshotted() {
    let f = kernel(false, true);
    assert_eq!(stale_view_roots(&f).len(), 1);
}

#[test]
fn facts_treat_view_uses_as_variable_reads() {
    let mut b = FnBuilder::new("viewfacts", 64);
    let tid = b.let_(Op::WorkitemId);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(tid))]);
    let (c, _) = b.capture(|b| {
        let v = b.let_(Op::BitcastF32(Opnd::Var(vars[0]))); //  0 view (forwarding: not a read)
        let w = b.let_(Op::AddF32(Opnd::Val(v), Opnd::F32(1.0))); //  1 reads the variable through the view
        let n = b.add_i32(Opnd::Var(vars[0]), 1); //                  2 reads the variable
        b.set_var(vars[0], n); //                                     3 writes it
        w
    });
    let fs = b.facts(&c);
    assert!(fs[1].var_reads.contains(&vars[0]), "{:?}", fs[1]);
    assert!(amdgpu_dsl::facts::must_order(&fs[1], &fs[3]), "a view use stays before the variable's next write");
    b.splice(c);
    let d = b.let_(Op::CmpGeI32(Opnd::Var(vars[0]), Opnd::I32(4)));
    b.break_if(d);
    b.end_loop();
    let _ = b.finish();
}
