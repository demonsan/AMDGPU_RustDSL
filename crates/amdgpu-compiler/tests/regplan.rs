//! Register planning (docs/weaver-regs.md): pins, separate / share groups, rings and per-region caps are honoured by the allocator and
//! checked after allocation.
use amdgpu_compiler::{compile_mir_alloc, mir::MFunction, regalloc::Alloc, regplan};
use amdgpu_dsl::{DType, FnBuilder, Function, Op, Opnd, RegRef, ValueId};

/// out[tid] = tid + 1; out[tid + 64] = tid + 2; ... : `n` values with DISJOINT lifetimes (each stored right after it is computed), then
/// `wide` values live at the same time inside the weave region "r".
fn kernel(n: usize, wide: usize, plan: impl FnOnce(&mut FnBuilder, &[ValueId], &[ValueId])) -> Function {
    let mut b = FnBuilder::new("regplan_t", 64);
    b.arg("out", DType::Ptr, 0, 8);
    let out = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(Opnd::Val(tid), Opnd::Val(c4));
    let base = b.ptr_add(Opnd::Val(out), Opnd::Val(off));
    let mut seq = Vec::new();
    for k in 0..n {
        let c = b.c_i32(k as i32 + 1);
        let v = b.add_i32(Opnd::Val(tid), Opnd::Val(c));
        let f = b.let_(Op::BitcastF32(Opnd::Val(v)));
        b.store_f32(Opnd::Val(base), (256 * k) as u32, Opnd::Val(f));
        seq.push(v);
    }
    let mut par = Vec::new();
    b.weave_region_begin("r");
    for k in 0..wide {
        let c = b.c_i32(100 + k as i32);
        par.push(b.add_i32(Opnd::Val(tid), Opnd::Val(c)));
    }
    let mut acc = par.first().copied();
    for v in par.iter().skip(1) {
        acc = Some(b.add_i32(Opnd::Val(acc.unwrap()), Opnd::Val(*v)));
    }
    if let Some(a) = acc {
        let f = b.let_(Op::BitcastF32(Opnd::Val(a)));
        b.store_f32(Opnd::Val(base), 8192, Opnd::Val(f));
    }
    b.weave_region_end();
    plan(&mut b, &seq, &par);
    b.finish()
}

fn err(f: &Function) -> String {
    match compile_mir_alloc(f) {
        Ok(_) => panic!("expected a compile error"),
        Err(e) => e,
    }
}

fn reg(m: &MFunction, a: &Alloc, v: ValueId) -> u32 {
    let name = format!("%v{v}");
    let r = *m.reg_hints.by_name.get(&name).unwrap_or_else(|| panic!("{name} has no register"));
    a.phys[r as usize]
}

#[test]
fn no_plan_changes_nothing() {
    let f = kernel(3, 4, |_, _, _| {});
    assert!(f.reg_plan.is_empty());
    let (m, _) = compile_mir_alloc(&f).unwrap();
    assert!(!m.reg_hints.active());
}

#[test]
fn pin_is_honoured_and_conflicts_are_errors() {
    let f = kernel(2, 0, |b, s, _| b.pin_vgpr_at(RegRef::Val(s[0]), 40));
    let (m, a) = compile_mir_alloc(&f).unwrap();
    assert_eq!(reg(&m, &a, f_val(&f, 0)), 40);
    // two values live at the same time pinned to the same register
    let g = kernel(0, 3, |b, _, p| {
        b.pin_vgpr_at(RegRef::Val(p[0]), 50);
        b.pin_vgpr_at(RegRef::Val(p[1]), 50);
    });
    let e = err(&g);
    assert!(e.contains("register plan"), "{e}");
}

/// the value of the k-th sequential store (re-derived from the kernel: the add of tid and k+1)
fn f_val(f: &Function, k: usize) -> ValueId {
    let mut n = 0;
    for st in f.stmts() {
        if let amdgpu_dsl::Stmt::Let { dst, op: Op::AddI32(..) } = st {
            if n == k {
                return *dst;
            }
            n += 1;
        }
    }
    panic!("no add {k}")
}

#[test]
fn separate_never_shares_even_with_disjoint_lifetimes() {
    let base = kernel(2, 0, |b, s, _| b.reg_share(&[RegRef::Val(s[0]), RegRef::Val(s[1])]));
    let (m0, a0) = compile_mir_alloc(&base).unwrap();
    let (x0, y0) = (reg(&m0, &a0, f_val(&base, 0)), reg(&m0, &a0, f_val(&base, 1)));
    assert_eq!(x0, y0, "share hint: disjoint lifetimes end up in the same register");
    assert!(regplan::report(&m0, &a0).iter().any(|l| l.contains("shared")));
    let f = kernel(2, 0, |b, s, _| b.reg_separate(&[RegRef::Val(s[0]), RegRef::Val(s[1])]));
    let (m, a) = compile_mir_alloc(&f).unwrap();
    assert_ne!(reg(&m, &a, f_val(&f, 0)), reg(&m, &a, f_val(&f, 1)));
    let r = kernel(3, 0, |b, s, _| b.reg_ring("buf", &[RegRef::Val(s[0]), RegRef::Val(s[1]), RegRef::Val(s[2])]));
    let (m, a) = compile_mir_alloc(&r).unwrap();
    let regs: Vec<u32> = (0..3).map(|k| reg(&m, &a, f_val(&r, k))).collect();
    assert!(regs[0] != regs[1] && regs[1] != regs[2] && regs[0] != regs[2], "{regs:?}");
}

#[test]
fn region_caps_use_the_allocator_footprint() {
    let ok = kernel(0, 8, |b, _, _| b.reg_cap("r", 64));
    let (m, a) = compile_mir_alloc(&ok).unwrap();
    let rep = regplan::region_report(&m, &a);
    let r = rep.iter().find(|p| p.name == "r").expect("region r");
    assert!(r.insts > 0 && r.vgpr_footprint_peak >= 8 && r.vgpr_live_peak >= 8, "{r:?}");
    let tight = kernel(0, 8, |b, _, _| b.reg_cap("r", 4));
    let e = err(&tight);
    assert!(e.contains("register cap"), "{e}");
    let unknown = kernel(0, 2, |b, _, _| b.reg_cap("nope", 64));
    assert!(err(&unknown).contains("unknown region"));
}

#[test]
fn unbalanced_regions_are_rejected_by_the_verifier() {
    let mut b = FnBuilder::new("bad", 64);
    b.weave_region_begin("x");
    let f = b.finish();
    assert!(amdgpu_dsl::verify(&f).iter().any(|e| e.contains("never closed")));
}
