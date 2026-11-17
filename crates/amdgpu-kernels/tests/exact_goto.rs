//! Exact-mode gotos (`Stmt::GotoIf` / `Stmt::Goto`): only forward, only inside one structured region (docs/exact-mode.md). The accepted
//! shape and the three rejected mutations: a backward goto, a goto into a loop, a goto out of an if arm.
use amdgpu_dsl::{verify, CmpPred, CmpTy, DType, FnBuilder, Function, Op, Opnd};

fn base(f: impl FnOnce(&mut FnBuilder, Opnd)) -> Function {
    let mut b = FnBuilder::new("goto_probe", 32);
    b.exact();
    let x = b.let_(Op::WorkitemId);
    let c = b.let_(Op::Cmp { pred: CmpPred::Eq, ty: CmpTy::U32, a: Opnd::Val(x), b: Opnd::I32(0) });
    f(&mut b, Opnd::Val(c));
    b.finish()
}

fn goto_errors(f: &Function) -> Vec<String> {
    verify(f).into_iter().filter(|e| e.contains("goto")).collect()
}

#[test]
fn a_forward_goto_in_one_region_is_accepted() {
    let f = base(|b, c| {
        b.goto_if(c, 2);
        b.layout_key(1);
        b.goto(3);
        b.layout_key(2);
        b.layout_key(3);
    });
    assert!(goto_errors(&f).is_empty(), "{:?}", goto_errors(&f));
}

#[test]
fn a_backward_goto_is_rejected() {
    let f = base(|b, c| {
        b.layout_key(1);
        b.goto_if(c, 1);
    });
    let e = goto_errors(&f);
    assert!(e.iter().any(|x| x.contains("backward")), "{e:?}");
}

#[test]
fn a_goto_into_a_loop_is_rejected() {
    let f = base(|b, c| {
        b.goto_if(c, 7);
        let _v = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
        b.layout_key(7);
        b.break_if(c);
        b.end_loop();
    });
    let e = goto_errors(&f);
    assert!(e.iter().any(|x| x.contains("leaves its structured region")), "{e:?}");
}

#[test]
fn a_goto_out_of_an_if_arm_is_rejected() {
    let f = base(|b, c| {
        b.begin_if(c);
        b.goto(9);
        b.end_if();
        b.layout_key(9);
    });
    let e = goto_errors(&f);
    assert!(e.iter().any(|x| x.contains("leaves its structured region")), "{e:?}");
}
