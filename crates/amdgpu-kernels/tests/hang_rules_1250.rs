//! G5 of docs/gfx1250-plan.md: the hang-safety rules (`amdgpu_codegen::gpu_verify_1250`, rule ids `GV-1250-*`, docs/gfx1250-hang-rules.md).
//!
//! Every rule has
//! * a mutation test `mut_<rule>`: the faulty kernel (or instruction stream) is REJECTED with the rule's id, and ACCEPTED when that rule alone is switched off - so the test is red as soon as
//!   the rule is removed or weakened (`with_rules_disabled`);
//! * a positive twin `pos_<rule>`: the nearest legal variant is accepted.
//! Nothing here runs on a GPU: the faulty kernels stop at the compiler, by construction.
//! `every_rule_is_covered` ties `RULES` to this file: a rule without its two tests fails the build.

use amdgpu_codegen::gpu_verify_1250::{verify_asm, verify_ir, with_rules_disabled, RULES};
use amdgpu_codegen::compile_target;
use amdgpu_dsl::{CmpPred, CmpTy, DType, FnBuilder, Function, Op, Opnd, TdmCpol};

fn gfx1250(f: &Function) -> Result<String, String> {
    compile_target(f, "gfx1250").map(|c| c.asm)
}

fn has(v: &[amdgpu_codegen::gpu_verify_1250::Violation], rule: &str) -> bool {
    v.iter().any(|x| x.rule == rule)
}

/// An IR mutation: `verify_ir` reports the rule; with the rule off it does not.
fn ir_mutation(f: &Function, rule: &str) {
    let v = verify_ir(f);
    assert!(has(&v, rule), "{rule}: the faulty kernel was not rejected: {v:?}");
    with_rules_disabled(&[rule], || {
        let v = verify_ir(f);
        assert!(!has(&v, rule), "{rule}: still reported with the rule switched off");
    });
}

/// An IR mutation that also reaches the compiler: `compile_target` fails with the rule's id, and compiles without it.
fn compile_mutation(f: &Function, rule: &str) {
    ir_mutation(f, rule);
    let e = gfx1250(f).err().unwrap_or_else(|| panic!("{rule}: the faulty kernel compiled"));
    assert!(e.contains(rule), "{rule}: wrong error: {e}");
    with_rules_disabled(&[rule], || {
        gfx1250(f).unwrap_or_else(|e| panic!("{rule}: does not compile even without the rule (the test is not about the rule): {e}"));
    });
}

fn ir_positive(f: &Function) {
    let v = verify_ir(f);
    assert!(v.is_empty(), "the legal kernel was rejected: {v:?}");
}

fn compile_positive(f: &Function) -> String {
    ir_positive(f);
    gfx1250(f).unwrap_or_else(|e| panic!("the legal kernel does not compile: {e}"))
}

/// An instruction-stream mutation: the text violates the rule; with the rule off it does not.
fn asm_mutation(asm: &str, rule: &str) {
    let v = verify_asm(asm);
    assert!(has(&v, rule), "{rule}: the faulty stream was not rejected: {v:?}\n{asm}");
    with_rules_disabled(&[rule], || {
        assert!(!has(&verify_asm(asm), rule), "{rule}: still reported with the rule switched off");
    });
}

fn asm_positive(asm: &str) {
    let v = verify_asm(asm);
    assert!(v.is_empty(), "the legal stream was rejected: {v:?}\n{asm}");
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// kernels

/// One tensor load with constant descriptor words: group 1 word 0 = `w0` (multicast mask in its low 16 bits), cluster launch of `cluster` workgroups (0 = none), a barrier after the wait.
fn tdm_kernel(w0: i32, cluster: i64) -> Function {
    let mut b = FnBuilder::new("hr_tdm", 64);
    b.lds(4096);
    let g0 = b.let_(Op::RsrcCat { words: [Opnd::I32(1), Opnd::I32(0), Opnd::I32(0), Opnd::I32((2u32 << 30) as i32)] });
    let g1 = b.sgpr_cat8([Opnd::I32(w0), Opnd::I32(64 << 16), Opnd::I32(1), Opnd::I32(64 << 16), Opnd::I32(1), Opnd::I32(64), Opnd::I32(0), Opnd::I32(0)]);
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default());
    b.wait_tensor(0);
    b.barrier();
    let mut f = b.finish();
    if cluster > 0 {
        f.kd.push(("cluster_dims".into(), cluster));
    }
    f
}

/// A tensor load with every descriptor word given: group 0 = (`g0w0`, LDS address `lds`, 0, `g0w3`), group 1 = (`w0`, dims, `tile0`, `tile1`), LDS of `lds_bytes`.
fn tdm_desc_kernel(g0w0: i32, g0w3: i32, lds: i32, w0: i32, tile0: i32, tile1: i32, lds_bytes: u32) -> Function {
    let mut b = FnBuilder::new("hr_desc", 64);
    b.lds(lds_bytes);
    let g0 = b.let_(Op::RsrcCat { words: [Opnd::I32(g0w0), Opnd::I32(lds), Opnd::I32(0), Opnd::I32(g0w3)] });
    let g1 = b.sgpr_cat8([Opnd::I32(w0), Opnd::I32(4096 << 16), Opnd::I32(0), Opnd::I32(tile0 << 16), Opnd::I32(tile1), Opnd::I32(4096), Opnd::I32(0), Opnd::I32(0)]);
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default());
    b.wait_tensor(0);
    b.barrier();
    b.finish()
}

const TYPE2: i32 = (2u32 << 30) as i32;

/// The group-1 word 0 comes from the kernarg segment (the host builds the descriptor); `host_values` is the host contract on it (empty: none).
fn tdm_host_kernel(host_values: &[i64]) -> Function {
    let mut b = FnBuilder::new("hr_tdm_host", 64);
    b.lds(4096);
    b.arg("d0", DType::I32, 0, 4);
    if !host_values.is_empty() {
        b.ka_one_of(0, host_values);
    }
    let w0 = b.let_(Op::KernargI32 { offset: 0 });
    let g0 = b.let_(Op::RsrcCat { words: [Opnd::I32(1), Opnd::I32(0), Opnd::I32(0), Opnd::I32((2u32 << 30) as i32)] });
    let g1 = b.sgpr_cat8([Opnd::Val(w0), Opnd::I32(64 << 16), Opnd::I32(1), Opnd::I32(64 << 16), Opnd::I32(1), Opnd::I32(64), Opnd::I32(0), Opnd::I32(0)]);
    b.tensor_load_to_lds(Opnd::Val(g0), Opnd::Val(g1), None, TdmCpol::default());
    b.wait_tensor(0);
    b.barrier();
    b.finish()
}

/// `body` runs under a condition that differs between the waves of the work-group (work-item id < 32).
fn divergent_if(then: fn(&mut FnBuilder), els: Option<fn(&mut FnBuilder)>) -> Function {
    let mut b = FnBuilder::new("hr_div", 64);
    let t = b.let_(Op::WorkitemId);
    let c = b.cmp(CmpPred::Lt, CmpTy::I32, t, Opnd::I32(32));
    b.begin_if(c);
    then(&mut b);
    if let Some(e) = els {
        b.else_();
        e(&mut b);
    }
    b.end_if();
    b.finish()
}

fn bar1(b: &mut FnBuilder) {
    b.barrier();
}
fn bar2(b: &mut FnBuilder) {
    b.barrier();
    b.barrier();
}
fn sig1(b: &mut FnBuilder) {
    b.barrier_signal(-1);
}

/// Barrier operations by id: `ops` are (signal?, id).
fn barrier_kernel(ops: &[(bool, i32)], kd: &[(&str, i64)]) -> Function {
    let mut b = FnBuilder::new("hr_bar", 64);
    for &(signal, id) in ops {
        if signal {
            b.barrier_signal(id);
        } else {
            b.barrier_wait(id);
        }
    }
    let mut f = b.finish();
    for (k, v) in kd {
        f.kd.push((k.to_string(), *v));
    }
    f
}

/// A kernel whose body is `body` (the named-barrier forms), a 64-work-item group (two waves), `named` allocated named barriers and `cluster` workgroups per cluster.
fn named_kernel(body: impl FnOnce(&mut FnBuilder), named: i64, cluster: i64) -> Function {
    let mut b = FnBuilder::new("hr_named", 64);
    body(&mut b);
    let mut f = b.finish();
    if named > 0 {
        f.kd.push(("named_barrier_count".into(), named));
    }
    if cluster > 0 {
        f.kd.push(("cluster_dims".into(), cluster));
    }
    f
}

/// Initialize named barrier `id` with `members`, then order the initialization with a work-group barrier.
fn init(b: &mut FnBuilder, id: i32, members: u8) {
    b.barrier_init(id, members);
    b.barrier();
}

fn base_asm() -> String {
    compile_positive(&tdm_kernel(0x20000, 0))
}

/// `base_asm` with `extra` inserted before the first instruction that contains `before`.
fn inject(before: &str, extra: &[&str]) -> String {
    let asm = base_asm();
    let mut out = Vec::new();
    let mut done = false;
    for l in asm.lines() {
        if !done && l.trim_start().starts_with(before) {
            for e in extra {
                out.push(format!("    {e}"));
            }
            done = true;
        }
        out.push(l.to_string());
    }
    assert!(done, "`{before}` not found in the base asm");
    out.join("\n")
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// GV-1250-TDM-CLAUSE

#[test]
fn mut_tdm_clause() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_clause 0x1"]), "GV-1250-TDM-CLAUSE");
    asm_mutation(&inject("tensor_load_to_lds", &["s_clause 0x0"]), "GV-1250-TDM-CLAUSE");
}

#[test]
fn pos_tdm_clause() {
    // the clause ends before the tensor operation
    asm_positive(&inject("tensor_load_to_lds", &["s_clause 0x0", "s_nop 0"]));
    asm_positive(&base_asm());
}

// GV-1250-SLEEP

#[test]
fn mut_sleep() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_sleep 0x8000"]), "GV-1250-SLEEP");
    asm_mutation(&inject("tensor_load_to_lds", &["s_monitor_sleep 0x8001"]), "GV-1250-SLEEP");
    asm_mutation(&inject("tensor_load_to_lds", &[".long 0xbf838000"]), "GV-1250-SLEEP");
}

#[test]
fn pos_sleep() {
    asm_positive(&inject("tensor_load_to_lds", &["s_sleep 0x5"]));
    asm_positive(&inject("tensor_load_to_lds", &[".long 0xbf830005"]));
}

// GV-1250-MCAST

#[test]
fn mut_mcast() {
    compile_mutation(&tdm_kernel(0x20003, 0), "GV-1250-MCAST");   // a constant mask, no cluster launch
    compile_mutation(&tdm_kernel(0x20003, 1), "GV-1250-MCAST");   // a cluster of one workgroup is not a cluster launch
    ir_mutation(&tdm_host_kernel(&[]), "GV-1250-MCAST");          // a host-built word with no host contract: not provably zero
    ir_mutation(&tdm_host_kernel(&[0x20000, 0x20003]), "GV-1250-MCAST");   // the contract admits a non-zero mask
}

#[test]
fn pos_mcast() {
    compile_positive(&tdm_kernel(0x20000, 0));
    compile_positive(&tdm_kernel(0x2000f, 4));   // a declared cluster launch of four workgroups
    ir_positive(&tdm_host_kernel(&[0x10000, 0x20000]));   // the host contract: the low 16 bits are zero
}

// GV-1250-BAR-ID

#[test]
fn mut_bar_id() {
    compile_mutation(&barrier_kernel(&[(true, 0), (false, 0)], &[]), "GV-1250-BAR-ID");     // the null barrier
    compile_mutation(&barrier_kernel(&[(true, 17), (false, 17)], &[]), "GV-1250-BAR-ID");   // beyond the named barriers
    compile_mutation(&barrier_kernel(&[(true, -5), (false, -5)], &[]), "GV-1250-BAR-ID");
    compile_mutation(&barrier_kernel(&[(true, -3), (false, -3)], &[]), "GV-1250-BAR-ID");   // the cluster barrier outside a cluster launch
    compile_mutation(&named_kernel(|b| { init(b, 2, 2); b.barrier_join(2); b.barrier_signal(2); b.barrier_wait(2); b.barrier_leave(); }, 0, 0), "GV-1250-BAR-ID");     // a named barrier that is not allocated
    compile_mutation(&named_kernel(|b| { init(b, 3, 2); b.barrier_join(3); b.barrier_signal(3); b.barrier_wait(3); b.barrier_leave(); }, 2, 0), "GV-1250-BAR-ID");   // beyond the allocated count
}

#[test]
fn mut_bar_id_asm() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_barrier_wait 0x11"]), "GV-1250-BAR-ID");
}

#[test]
fn pos_bar_id_asm() {
    // a disassembler prints the 16-bit immediate of -1 / -3 as unsigned
    asm_positive(&inject("tensor_load_to_lds", &["s_barrier_wait 0xffff", "s_barrier_wait 0xfffd"]));
}

#[test]
fn pos_bar_id() {
    compile_positive(&barrier_kernel(&[(true, -1), (false, -1)], &[]));
    compile_positive(&named_kernel(|b| { init(b, 2, 2); b.barrier_join(2); b.barrier_signal(2); b.barrier_wait(2); b.barrier_leave(); }, 2, 0));
    compile_positive(&barrier_kernel(&[(true, -3), (false, -3)], &[("cluster_dims", 4)]));
}

#[test]
fn mut_bar_id_forms() {
    // the cluster barrier in a launch that declares no cluster
    compile_mutation(&named_kernel(|b| b.cluster_barrier(), 0, 0), "GV-1250-BAR-ID");
    compile_mutation(&named_kernel(|b| b.cluster_barrier(), 0, 1), "GV-1250-BAR-ID");
    // an init / join of a barrier that is not allocated
    ir_mutation(&named_kernel(|b| b.barrier_init(2, 2), 1, 0), "GV-1250-BAR-ID");
    ir_mutation(&named_kernel(|b| b.barrier_join(3), 2, 0), "GV-1250-BAR-ID");
    // the first-arriver form on the cluster barrier
    asm_mutation(&inject("tensor_load_to_lds", &["s_barrier_signal_isfirst -3"]), "GV-1250-BAR-ID");
}

#[test]
fn pos_bar_id_forms() {
    compile_positive(&named_kernel(|b| b.cluster_barrier(), 0, 4));
    ir_positive(&named_kernel(|b| { init(b, 2, 2); b.barrier_join(2); b.barrier_leave(); }, 2, 0));
    asm_positive(&inject("tensor_load_to_lds", &["s_barrier_signal_isfirst -1"]));
}

// GV-1250-NBAR-COUNT

#[test]
fn mut_nbar_count() {
    ir_mutation(&named_kernel(|b| b.barrier_init(1, 0), 1, 0), "GV-1250-NBAR-COUNT");    // no members
    ir_mutation(&named_kernel(|b| b.barrier_init(1, 64), 1, 0), "GV-1250-NBAR-COUNT");   // beyond the field
}

#[test]
fn pos_nbar_count() {
    ir_positive(&named_kernel(|b| b.barrier_init(1, 2), 1, 0));
    ir_positive(&named_kernel(|b| b.barrier_init(1, 1), 1, 0));
    // completion is count-based: signal-only producers count too, so the member count may exceed the waves that join
    ir_positive(&named_kernel(|b| b.barrier_init(1, 4), 1, 0));
}

// GV-1250-BAR-JOIN

#[test]
fn mut_bar_join() {
    ir_mutation(&named_kernel(|b| { init(b, 1, 2); init(b, 2, 2); b.barrier_join(1); b.barrier_join(2); b.barrier_leave(); }, 2, 0), "GV-1250-BAR-JOIN");   // two joined at once
    ir_mutation(&named_kernel(|b| b.barrier_leave(), 1, 0), "GV-1250-BAR-JOIN");                                                                              // leave with nothing joined
    ir_mutation(&named_kernel(|b| { init(b, 1, 2); b.barrier_signal(1); b.barrier_wait(1); }, 1, 0), "GV-1250-BAR-JOIN");                                     // a wait while not joined
    ir_mutation(&named_kernel(|b| { init(b, 1, 2); init(b, 2, 2); b.barrier_join(1); b.barrier_signal(2); b.barrier_wait(2); b.barrier_leave(); }, 2, 0), "GV-1250-BAR-JOIN");   // a wait on another barrier than the joined one
    // the paths of a branch end differently joined (uniform condition)
    let mut b = FnBuilder::new("hr_join_if", 64);
    b.arg("n", DType::I32, 0, 4);
    init(&mut b, 1, 2);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let c = b.cmp(CmpPred::Gt, CmpTy::I32, n, Opnd::I32(0));
    b.begin_if(c);
    b.barrier_join(1);
    b.end_if();
    let mut f = b.finish();
    f.kd.push(("named_barrier_count".into(), 1));
    ir_mutation(&f, "GV-1250-BAR-JOIN");
    // a loop body that joins every iteration without leaving
    let mut b = FnBuilder::new("hr_join_loop", 64);
    b.arg("n", DType::I32, 0, 4);
    init(&mut b, 1, 2);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let c = b.cmp(CmpPred::Gt, CmpTy::I32, vars[0], n);
    b.break_if(c);
    b.barrier_join(1);
    b.end_loop();
    let mut f = b.finish();
    f.kd.push(("named_barrier_count".into(), 1));
    ir_mutation(&f, "GV-1250-BAR-JOIN");
}

#[test]
fn pos_bar_join() {
    ir_positive(&named_kernel(|b| { init(b, 1, 2); b.barrier_join(1); b.barrier_signal(1); b.barrier_wait(1); b.barrier_leave(); }, 1, 0));
    // join, use, leave, join another
    ir_positive(&named_kernel(|b| { init(b, 1, 2); init(b, 2, 2); b.barrier_join(1); b.barrier_signal(1); b.barrier_wait(1); b.barrier_leave(); b.barrier_join(2); b.barrier_signal(2); b.barrier_wait(2); b.barrier_leave(); }, 2, 0));
    // any wave may signal any barrier, joined or not (a signal-only producer)
    ir_positive(&named_kernel(|b| { init(b, 1, 3); b.barrier_signal(1); }, 1, 0));
    ir_positive(&named_kernel(|b| { init(b, 1, 3); init(b, 2, 2); b.barrier_join(1); b.barrier_signal(2); b.barrier_signal(1); b.barrier_wait(1); b.barrier_leave(); }, 2, 0));
    // a loop that joins and leaves inside its body
    let mut b = FnBuilder::new("hr_join_loop_ok", 64);
    b.arg("n", DType::I32, 0, 4);
    init(&mut b, 1, 2);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let c = b.cmp(CmpPred::Gt, CmpTy::I32, vars[0], n);
    b.break_if(c);
    b.barrier_join(1);
    b.barrier_signal(1);
    b.barrier_wait(1);
    b.barrier_leave();
    b.end_loop();
    let mut f = b.finish();
    f.kd.push(("named_barrier_count".into(), 1));
    ir_positive(&f);
}

// GV-1250-BAR-INIT

#[test]
fn mut_bar_init() {
    ir_mutation(&named_kernel(|b| { b.barrier_join(1); b.barrier_signal(1); b.barrier_wait(1); b.barrier_leave(); }, 1, 0), "GV-1250-BAR-INIT");   // never initialized
    ir_mutation(&named_kernel(|b| { b.barrier_init(1, 2); b.barrier_join(1); b.barrier_signal(1); b.barrier_wait(1); b.barrier_leave(); }, 1, 0), "GV-1250-BAR-INIT");   // no work-group barrier after the init
    ir_mutation(&named_kernel(|b| { b.barrier_init(1, 2); b.barrier_signal(1); }, 1, 0), "GV-1250-BAR-INIT");                                                       // a signal right after the init
}

#[test]
fn pos_bar_init() {
    ir_positive(&named_kernel(|b| { init(b, 1, 2); b.barrier_join(1); b.barrier_signal(1); b.barrier_wait(1); b.barrier_leave(); }, 1, 0));
    // the work-group barrier in its split form orders the init too
    ir_positive(&named_kernel(|b| { b.barrier_init(1, 2); b.barrier_signal(-1); b.barrier_wait(-1); b.barrier_join(1); b.barrier_signal(1); b.barrier_wait(1); b.barrier_leave(); }, 1, 0));
}

// GV-1250-TRAP-BAR

#[test]
fn mut_trap_bar() {
    compile_mutation(&barrier_kernel(&[(true, -2), (false, -2)], &[]), "GV-1250-TRAP-BAR");
    compile_mutation(&barrier_kernel(&[(true, -4), (false, -4)], &[("cluster_dims", 4)]), "GV-1250-TRAP-BAR");
    asm_mutation(&inject("tensor_load_to_lds", &["s_barrier_signal -2"]), "GV-1250-TRAP-BAR");
}

#[test]
fn pos_trap_bar() {
    compile_positive(&barrier_kernel(&[(true, -1), (false, -1)], &[]));
    asm_positive(&inject("tensor_load_to_lds", &["s_barrier_signal -1"]));
}

// GV-1250-BAR-WAIT-IMM

#[test]
fn mut_bar_wait_imm() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_barrier_wait m0"]), "GV-1250-BAR-WAIT-IMM");
    asm_mutation(&inject("tensor_load_to_lds", &["s_barrier_wait s4"]), "GV-1250-BAR-WAIT-IMM");
}

#[test]
fn pos_bar_wait_imm() {
    asm_positive(&inject("tensor_load_to_lds", &["s_barrier_wait -1"]));
    asm_positive(&inject("tensor_load_to_lds", &["s_barrier_signal m0"]));   // a named barrier selected through M0 on the signal side
}

// GV-1250-BAR-REACH

#[test]
fn mut_bar_reach() {
    // a barrier only on one side of a wave-divergent condition
    ir_mutation(&divergent_if(bar1, None), "GV-1250-BAR-REACH");
    // the arms arrive at different numbers of barriers
    ir_mutation(&divergent_if(bar2, Some(bar1)), "GV-1250-BAR-REACH");
    // split barriers: a signal in one arm only
    ir_mutation(&divergent_if(sig1, None), "GV-1250-BAR-REACH");
    // a loop whose exit differs between the waves, with a barrier in it
    let mut b = FnBuilder::new("hr_loop", 64);
    let t = b.let_(Op::WorkitemId);
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let c = b.cmp(CmpPred::Gt, CmpTy::I32, vars[0], t);
    b.break_if(c);
    b.barrier();
    b.end_loop();
    ir_mutation(&b.finish(), "GV-1250-BAR-REACH");
    // a wait with no signal of that barrier
    ir_mutation(&barrier_kernel(&[(false, -1)], &[]), "GV-1250-BAR-REACH");
    ir_mutation(&barrier_kernel(&[(true, 2), (false, 3)], &[("named_barrier_count", 3)]), "GV-1250-BAR-REACH");
}

#[test]
fn pos_bar_reach() {
    // both arms arrive at the same barrier
    ir_positive(&divergent_if(bar1, Some(bar1)));
    // a barrier under a condition every wave takes the same way (a kernarg scalar)
    let mut b = FnBuilder::new("hr_uni", 64);
    b.arg("n", DType::I32, 0, 4);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let c = b.cmp(CmpPred::Gt, CmpTy::I32, n, Opnd::I32(0));
    b.begin_if(c);
    b.barrier();
    b.end_if();
    ir_positive(&b.finish());
    // a loop with a uniform exit
    let mut b = FnBuilder::new("hr_uloop", 64);
    b.arg("n", DType::I32, 0, 4);
    let n = b.let_(Op::KernargI32 { offset: 0 });
    let vars = b.begin_loop(&[(DType::I32, Opnd::I32(0))]);
    let c = b.cmp(CmpPred::Gt, CmpTy::I32, vars[0], n);
    b.break_if(c);
    b.barrier();
    b.end_loop();
    ir_positive(&b.finish());
    // signal early, wait late
    ir_positive(&barrier_kernel(&[(true, -1), (false, -1)], &[]));
    ir_positive(&named_kernel(|b| { init(b, 2, 2); init(b, 3, 2); b.barrier_join(2); b.barrier_signal(2); b.barrier_wait(2); b.barrier_leave(); b.barrier_join(3); b.barrier_signal(3); b.barrier_wait(3); b.barrier_leave(); }, 3, 0));
}

// GV-1250-TDM-DESC

#[test]
fn mut_tdm_desc() {
    ir_mutation(&tdm_desc_kernel(1 | 8, TYPE2, 0, 0x20000, 64, 1, 4096), "GV-1250-TDM-DESC");        // the store bit set in a load descriptor
    ir_mutation(&tdm_desc_kernel(1 | 4, TYPE2, 0, 0x20000, 64, 1, 4096), "GV-1250-TDM-DESC");        // the context-restore bit
    ir_mutation(&tdm_desc_kernel(1 | 16, TYPE2, 0, 0x20000, 64, 1, 4096), "GV-1250-TDM-DESC");       // the volatile bit
    ir_mutation(&tdm_desc_kernel(3, TYPE2, 0, 0x20000, 64, 1, 4096), "GV-1250-TDM-DESC");            // a valid count of 3
    ir_mutation(&tdm_desc_kernel(1, 1 << 30, 0, 0x20000, 64, 1, 4096), "GV-1250-TDM-DESC");          // a wrong type
    ir_mutation(&tdm_desc_kernel(1, TYPE2, 0, 0x20000, 64, 32, 4096), "GV-1250-TDM-DESC");           // 64 x 32 x 4 bytes = 8 KiB in a 4 KiB LDS
    ir_mutation(&tdm_desc_kernel(1, TYPE2, 4000, 0x20000, 64, 1, 4096), "GV-1250-TDM-DESC");         // the tile starts too close to the end
}

#[test]
fn pos_tdm_desc() {
    compile_positive(&tdm_desc_kernel(1, TYPE2, 0, 0x20000, 64, 1, 4096));
    compile_positive(&tdm_desc_kernel(1, TYPE2, 2048, 0x20000, 64, 8, 4096));   // exactly to the end of the LDS
    compile_positive(&tdm_desc_kernel(0, TYPE2, 0, 0x20000, 0, 0, 4096));      // a null tensor (valid count 0): a no-op by design
}

// GV-1250-TDM-DRAIN

const T: &str = "    tensor_load_to_lds s[24:27], s[32:39]\n";

fn burst(n: usize) -> String {
    format!("k:\n{}    s_endpgm\n", T.repeat(n))
}

#[test]
fn mut_tdm_drain() {
    // 12 back to back: the 12th issues with 11 outstanding
    asm_mutation(&burst(12), "GV-1250-TDM-DRAIN");
    // an unbounded loop: it issues and never drains
    asm_mutation(&format!("k:\nBB0:\n{T}    s_cbranch_scc1 BB0\n    s_endpgm\n"), "GV-1250-TDM-DRAIN");
    // a wait that leaves more than 10 does not drain enough, and in a loop the bound grows past 10
    asm_mutation(&format!("k:\nBB0:\n{T}    s_wait_tensorcnt 0xb\n    s_cbranch_scc1 BB0\n    s_endpgm\n"), "GV-1250-TDM-DRAIN");
    // one path of an if accumulates 6, the join continues with 6 more: the 12th op on that path sees 11
    asm_mutation(&format!("k:\n    s_cbranch_scc1 BB2\n{}BB2:\n{}    s_endpgm\n", T.repeat(6), T.repeat(6)), "GV-1250-TDM-DRAIN");
    // far apart does not matter when nothing drains in between: 12 ops with a lot of other work between them
    asm_mutation(&format!("k:\n{}    s_endpgm\n", format!("{T}{}", "    s_nop 0\n".repeat(40)).repeat(12)), "GV-1250-TDM-DRAIN");
}

#[test]
fn pos_tdm_drain() {
    // 11 back to back: every issue sees at most 10 outstanding
    asm_positive(&burst(11));
    // the vendor-like shape: two operations far apart with nothing else outstanding
    asm_positive(&format!("k:\n{T}{}{T}    s_endpgm\n", "    s_nop 0\n".repeat(40)));
    // a loop that drains to at most 10 each iteration
    asm_positive(&format!("k:\nBB0:\n{T}    s_wait_tensorcnt 0xa\n    s_cbranch_scc1 BB0\n    s_endpgm\n"));
    asm_positive(&format!("k:\nBB0:\n{T}{T}    s_wait_tensorcnt 0x8\n    s_cbranch_scc1 BB0\n    s_endpgm\n"));
    // a full drain, then a burst
    asm_positive(&format!("k:\n{}    s_wait_tensorcnt 0x0\n{}    s_endpgm\n", T.repeat(11), T.repeat(11)));
    // both paths drain, whatever they accumulated
    asm_positive(&format!("k:\n    s_cbranch_scc1 BB2\n{}    s_wait_tensorcnt 0x0\n    s_branch BB3\nBB2:\n    s_wait_tensorcnt 0x2\nBB3:\n{}    s_endpgm\n", T.repeat(6), T.repeat(8)));
    asm_positive(&base_asm());
}

// GV-1250-DPP

#[test]
fn mut_dpp() {
    asm_mutation(&inject("tensor_load_to_lds", &["v_add_f32_dpp v1, 0x12345678, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf"]), "GV-1250-DPP");
    asm_mutation(&inject("tensor_load_to_lds", &["v_add_nc_u64_dpp v[0:1], v[2:3], v[4:5] row_shl:1"]), "GV-1250-DPP");
}

#[test]
fn pos_dpp() {
    asm_positive(&inject("tensor_load_to_lds", &["v_add_f32_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf"]));
    asm_positive(&inject("tensor_load_to_lds", &["v_add_f32_dpp v1, 1.0, v3 row_shl:1 row_mask:0xf bank_mask:0xf"]));   // an inline constant is not a literal
}

// GV-1250-PERMLANE-CMPX

#[test]
fn mut_permlane_cmpx() {
    asm_mutation(&inject("tensor_load_to_lds", &["v_cmpx_eq_u32 vcc_lo, v1, v2", "v_permlane16_b32 v1, v2, s0, s1"]), "GV-1250-PERMLANE-CMPX");
}

#[test]
fn pos_permlane_cmpx() {
    asm_positive(&inject("tensor_load_to_lds", &["v_cmpx_eq_u32 vcc_lo, v1, v2", "s_nop 0", "v_permlane16_b32 v1, v2, s0, s1"]));
}

// GV-1250-DVGPR

#[test]
fn mut_dvgpr() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_alloc_vgpr 0x40"]), "GV-1250-DVGPR");
}

#[test]
fn pos_dvgpr() {
    asm_positive(&base_asm());   // a fixed VGPR allocation in the kernel descriptor
}

// GV-1250-PRIV

#[test]
fn mut_priv() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_sethalt 1"]), "GV-1250-PRIV");
    asm_mutation(&inject("tensor_load_to_lds", &["s_setreg_imm32_b32 hwreg(HW_REG_STATE_PRIV, 0, 1), 1"]), "GV-1250-PRIV");
}

#[test]
fn pos_priv() {
    asm_positive(&inject("tensor_load_to_lds", &["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1", "s_nop 0"]));
}

// GV-1250-SCHED-MODE

#[test]
fn mut_sched_mode() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 1"]), "GV-1250-SCHED-MODE");   // the full relaxed mode
    asm_mutation(&inject("tensor_load_to_lds", &["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 3"]), "GV-1250-SCHED-MODE");   // the reserved value
    asm_mutation(&inject("tensor_load_to_lds", &["s_setreg_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), s5"]), "GV-1250-SCHED-MODE");        // a value the verifier cannot see
}

#[test]
fn pos_sched_mode() {
    // the normal mode, the partial mode (the program owns two dependency classes: exact replays of vendor code) and the unrelated field above the mode bits
    asm_positive(&inject("tensor_load_to_lds", &["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 0", "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2", "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1"]));
}

// GV-1250-SETREG-MSB

#[test]
fn mut_setreg_msb() {
    asm_mutation(&inject("tensor_load_to_lds", &["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1", "s_set_vgpr_msb 0x1"]), "GV-1250-SETREG-MSB");
}

#[test]
fn pos_setreg_msb() {
    asm_positive(&inject("tensor_load_to_lds", &["s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1", "s_nop 0", "s_set_vgpr_msb 0x1"]));
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// coverage: RULES <-> tests

/// rule id -> the stem of its two tests (`mut_<stem>`, `pos_<stem>`)
const COVERAGE: &[(&str, &str)] = &[
    ("GV-1250-TDM-CLAUSE", "tdm_clause"),
    ("GV-1250-SLEEP", "sleep"),
    ("GV-1250-MCAST", "mcast"),
    ("GV-1250-BAR-ID", "bar_id"),
    ("GV-1250-NBAR-COUNT", "nbar_count"),
    ("GV-1250-BAR-JOIN", "bar_join"),
    ("GV-1250-BAR-INIT", "bar_init"),
    ("GV-1250-BAR-WAIT-IMM", "bar_wait_imm"),
    ("GV-1250-TRAP-BAR", "trap_bar"),
    ("GV-1250-BAR-REACH", "bar_reach"),
    ("GV-1250-TDM-DESC", "tdm_desc"),
    ("GV-1250-TDM-DRAIN", "tdm_drain"),
    ("GV-1250-DPP", "dpp"),
    ("GV-1250-PERMLANE-CMPX", "permlane_cmpx"),
    ("GV-1250-DVGPR", "dvgpr"),
    ("GV-1250-PRIV", "priv"),
    ("GV-1250-SCHED-MODE", "sched_mode"),
    ("GV-1250-SETREG-MSB", "setreg_msb"),
];

#[test]
fn every_rule_is_covered() {
    let me = include_str!("hang_rules_1250.rs");
    for r in RULES {
        let stem = COVERAGE.iter().find(|(id, _)| *id == r.id).unwrap_or_else(|| panic!("{}: no entry in COVERAGE", r.id)).1;
        for pre in ["mut_", "pos_"] {
            assert!(me.contains(&format!("fn {pre}{stem}()")), "{}: the test `{pre}{stem}` does not exist", r.id);
        }
    }
    for (id, _) in COVERAGE {
        assert!(RULES.iter().any(|r| r.id == *id), "{id}: COVERAGE names a rule that does not exist");
    }
}

/// Every shipped gfx1250 probe passes the rules.
#[test]
fn the_probes_pass_the_rules() {
    for k in amdgpu_kernels::gfx1250_kernel_names() {
        let f = amdgpu_kernels::kernel_by_name(k).unwrap();
        let v = verify_ir(&f);
        assert!(v.is_empty(), "{k}: {v:?}");
        let asm = compile_target(&f, "gfx1250").unwrap_or_else(|e| panic!("{k}: {e}")).asm;
        let v = verify_asm(&asm);
        assert!(v.is_empty(), "{k}: {v:?}");
    }
}
