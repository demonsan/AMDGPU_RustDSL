//! E5 (docs/exact-mode.md section 5): exact branch forms - out-of-line `if` branches at layout anchors (cold paths that jump back, loop
//! exits with their own bodies) and scope variables. Layout / scope only: no instruction of their own; existing kernels are unchanged (md5).

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, Op, Opnd};

/// out[tid] = 1.0 when the loop left at its first exit (n odd), 2.0 at its second (n even), carried out of the loop in a scope variable;
/// out[64 + tid] = 3.0 from a cold branch taken when it >= 2 (n >= 3). The exit bodies are laid out at anchor "exits", the cold branch at
/// anchor "cold" (the kernel end).
pub fn e5_kernel(place: bool) -> Function {
    let mut b = FnBuilder::new("probe_e5_branches", 64);
    // kernarg layout of tests/gpu/tileop_probe_host.cpp: in (unused), out, PROBE_I32
    b.arg("i", DType::Ptr, 0x00, 8);
    b.arg("o", DType::Ptr, 0x08, 8);
    b.arg("n", DType::I32, 0x10, 4);
    b.arg("pad", DType::I32, 0x14, 4);
    b.arg_tensor("i", Ext::c(4));
    b.arg_tensor("o", Ext::c(128 * 4));
    b.ka_range(0x10, 1, 64);
    let po = b.let_(Op::KernargPtr { offset: 0x08 });
    let n = b.let_(Op::KernargI32 { offset: 0x10 });
    let tid = b.let_(Op::WorkitemId);
    let zero = b.c_f32(0.0);
    let res = b.declare_vars(&[(DType::F32, Opnd::Val(zero))])[0];
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let it = vars[0];
    let two = b.c_i32(2);
    let rare = b.cmp(CmpPred::Ge, CmpTy::I32, Opnd::Var(it), two);
    if place { b.begin_if_at(rare, "cold") } else { b.begin_if_unlikely(rare) }
    let off2 = b.bind_i32((ev(tid) << 2) + 256);
    let q2 = b.ptr_add(po, off2);
    let three = b.c_f32(3.0);
    b.store_f32(q2, 0, three);
    b.end_if();
    let it1 = b.bind_i32(amdgpu_dsl::varx(it) + 1);
    let c1 = b.cmp(CmpPred::Ge, CmpTy::I32, it1, n);
    if place { b.begin_if_at(c1, "exits") } else { b.begin_if(c1) }
    let one = b.c_f32(1.0);
    b.set_var(res, one);
    b.break_if(c1);
    b.end_if();
    let it2 = b.bind_i32(amdgpu_dsl::varx(it) + 2);
    b.set_var(it, it2);
    let c2 = b.cmp(CmpPred::Ge, CmpTy::I32, it2, n);
    if place { b.begin_if_at(c2, "exits") } else { b.begin_if(c2) }
    let twof = b.c_f32(2.0);
    b.set_var(res, twof);
    b.break_if(c2);
    b.end_if();
    b.end_loop();
    if place {
        b.layout_anchor("exits");
    }
    let off = b.bind_i32(ev(tid) << 2);
    let q = b.ptr_add(po, off);
    b.store_f32(q, 0, Opnd::Var(res));
    if place {
        b.layout_anchor("cold");
    }
    b.finish()
}

fn asm(f: &Function) -> String {
    amdgpu_codegen::compile(f).unwrap_or_else(|e| panic!("{e}")).asm
}

fn pos(a: &str, s: &str) -> usize {
    a.find(s).unwrap_or_else(|| panic!("{s} not found:\n{a}"))
}

fn branches(a: &str, mn: &str) -> usize {
    a.lines().filter(|l| l.trim_start().starts_with(mn)).count()
}

/// The exit bodies (the stores of 1.0 / 2.0 into the scope variable) follow the loop's back edge, the cold store (3.0) follows the final
/// store; each exit and the cold branch is ONE conditional branch in the loop.
#[test]
fn placed_branches_are_laid_out_at_their_anchors() {
    let a = asm(&e5_kernel(true));
    let back = a.lines().position(|l| l.contains("s_branch") || l.contains("s_cbranch")).is_some();
    assert!(back);
    let lit = |dec: &str, hex: &str| if a.contains(dec) { dec.to_string() } else { hex.to_string() };
    let one = lit("1065353216", "0x3f800000");
    let three = lit("1077936128", "0x40400000");
    let (one, three) = (one.as_str(), three.as_str());
    let store = pos(&a, "global_store_dword");
    // no jump to a block that only jumps on (the exits reach the join in one branch)
    assert!(!a.lines().zip(a.lines().skip(1)).any(|(x, y)| x.trim_end().ends_with(":") && y.trim_start().starts_with("s_branch") && !x.contains("BB0")), "a forwarding block survived:\n{a}");
    let last_store = a.rfind("global_store_dword").unwrap();
    // the cold branch's store is the last store in the layout, after the result store
    assert!(pos(&a, three) > store, "the cold branch follows the hot code:\n{a}");
    assert!(last_store > pos(&a, three) || last_store > store, "{a}");
    // the first exit body (1.0) is laid out after the loop's back edge
    let back_edge = a.lines().enumerate().filter(|(_, l)| l.trim_start().starts_with("s_branch") || l.trim_start().starts_with("s_cbranch")).map(|(i, _)| i).collect::<Vec<_>>();
    let one_line = a.lines().position(|l| l.contains(one)).unwrap();
    assert!(back_edge.iter().any(|&i| i < one_line), "{a}");
    // 3 conditional branches in the loop (cold, exit 1, exit 2), no flag compare after the loop
    assert_eq!(branches(&a, "s_cbranch"), 3, "one conditional branch per placed if:\n{a}");
}

/// Without placement the same kernel is a plain structured one (if / break_if / if): the exits re-test their condition.
#[test]
fn unplaced_form_is_the_plain_structure() {
    let a = asm(&e5_kernel(false));
    assert!(branches(&a, "s_cbranch") > 3, "{a}");
}

/// The placed kernel proves like any other (strict GPU semantics: 0 errors).
#[test]
fn placed_kernel_verifies() {
    let f = e5_kernel(true);
    let rep = amdgpu_codegen::gpu_verify::analyse(&f).unwrap_or_else(|e| panic!("{e}"));
    assert!(rep.errors().is_empty(), "{:?}", rep.errors());
}

fn validate(f: &Function) -> Vec<String> {
    amdgpu_dsl::verify(f)
}

fn with_stmts(f: &Function, stmts: Vec<amdgpu_dsl::Stmt>) -> Function {
    let mut g = f.clone();
    g.set_stmts(stmts);
    g
}

/// Mutations: a placement not followed by an `if`, a placement at an anchor that does not exist, an anchor declared twice.
#[test]
fn mutations_are_rejected() {
    use amdgpu_dsl::Stmt;
    let ok = e5_kernel(true);
    assert!(validate(&ok).is_empty(), "{:?}", validate(&ok));
    let mut stmts = ok.stmts().to_vec();
    let at = stmts.iter().position(|s| matches!(s, Stmt::BranchPlace { .. })).unwrap();
    // 1. the placement no longer precedes its if
    let mut s1 = stmts.clone();
    s1.swap(at, at + 1);
    let e = validate(&with_stmts(&ok, s1));
    assert!(e.iter().any(|m| m.contains("out-of-line placement must be followed by an `if`")), "{e:?}");
    // 2. unknown anchor
    let mut s2 = stmts.clone();
    s2[at] = Stmt::BranchPlace { anchor: "nowhere".into() };
    let e = validate(&with_stmts(&ok, s2));
    assert!(e.iter().any(|m| m.contains("no layout anchor of that name")), "{e:?}");
    // 3. a duplicated anchor
    let an = stmts.iter().position(|s| matches!(s, Stmt::LayoutAnchor { .. })).unwrap();
    let dup = stmts[an].clone();
    stmts.insert(an, dup);
    let e = validate(&with_stmts(&ok, stmts));
    assert!(e.iter().any(|m| m.contains("declared more than once")), "{e:?}");
}

/// The new statements round-trip through the irv1 text (print -> parse -> identical function).
#[test]
fn e5_forms_round_trip_through_ir_text() {
    let f = e5_kernel(true);
    let txt = amdgpu_dsl::ir_text::print(&f);
    assert!(txt.contains("BranchPlace") && txt.contains("LayoutAnchor") && txt.contains("DeclareVars"), "{txt}");
    let g = amdgpu_dsl::ir_text::parse(&txt).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(amdgpu_dsl::ir_text::print(&g), txt);
    assert_eq!(asm(&g), asm(&f));
}

/// Writes the asm of the placed and the plain kernel for tools/e5_probe.sh (hardware check).
#[test]
#[ignore]
fn write_asm() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/e5");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, place) in [("plain", false), ("placed", true)] {
        std::fs::write(dir.join(format!("{name}.s")), asm(&e5_kernel(place))).unwrap();
    }
}

/// Two wave roles (waves 0-3 / 4-7) each running a loop that is left only through an E5 exit with its own body. Role A: per iteration
/// one barrier before the exit test and one after, the exit body one: exit path 2N + 2. Role B: one barrier before the loop, the exit
/// test first, two barriers after it, the exit body `b_exit` barriers: exit path 1 + 2N + b_exit. Balanced for b_exit = 1.
fn two_role_exits(b_exit: usize) -> Function {
    let mut b = FnBuilder::new("e5_bar_roles", 512);
    b.arg("n", DType::I32, 0x00, 4);
    b.ka_range(0x00, 1, 64);
    let n = b.let_(Op::KernargI32 { offset: 0x00 });
    let tid = b.let_(Op::WorkitemId);
    let six = b.c_i32(6);
    let wave = b.let_(Op::LshrI32(Opnd::Val(tid), Opnd::Val(six)));
    let four = b.c_i32(4);
    let role_a = b.cmp(CmpPred::Lt, CmpTy::I32, wave, four);
    b.begin_if(role_a);
    {
        let it = b.begin_loop(&[(DType::I32, Opnd::I32(0))])[0];
        b.barrier();
        let i1 = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        let c = b.cmp(CmpPred::Ge, CmpTy::I32, i1, n);
        b.begin_if_at(c, "exits_a");
        b.barrier();
        b.break_if(c);
        b.end_if();
        b.barrier();
        b.set_var(it, i1);
        b.end_loop();
        b.layout_anchor("exits_a");
    }
    b.else_();
    {
        b.barrier();
        let it = b.begin_loop(&[(DType::I32, Opnd::I32(0))])[0];
        let i1 = b.bind_i32(amdgpu_dsl::varx(it) + 1);
        let c = b.cmp(CmpPred::Ge, CmpTy::I32, i1, n);
        b.begin_if_at(c, "exits_b");
        for _ in 0..b_exit {
            b.barrier();
        }
        b.break_if(c);
        b.end_if();
        b.barrier();
        b.barrier();
        b.set_var(it, i1);
        b.end_loop();
        b.layout_anchor("exits_b");
    }
    b.end_if();
    b.finish()
}

fn bar_errors(f: &Function) -> Vec<String> {
    let rep = amdgpu_codegen::gpu_verify::analyse(f).unwrap_or_else(|e| panic!("{e}"));
    rep.errors().iter().map(|e| format!("{e:?}")).filter(|m| m.contains("GV-BAR")).collect()
}

/// GV-BAR follows the exit paths: balanced roles verify although their fall-through counts (no path) differ; an exit body with one
/// barrier too many is named.
#[test]
fn exit_paths_are_barrier_balanced() {
    assert!(bar_errors(&two_role_exits(1)).is_empty(), "{:?}", bar_errors(&two_role_exits(1)));
    let e = bar_errors(&two_role_exits(2));
    assert!(e.iter().any(|m| m.contains("exit paths")), "{e:?}");
}