//! GPU-semantics verifier (`gpu_verify`): positive cases (shipped kernels, annotated toy kernels) and one NEGATIVE case per error class.
//! The error taxonomy and the model are documented in docs/gpu-semantics-verifier.md.
use amdgpu_compiler::gpu_verify::{analyse, verify_ir, verify_mir, Report, Severity};
use amdgpu_compiler::mir::{Block, MFunction, MInst, MOp, Pred, RClass, Term};
use amdgpu_compiler::regalloc::Alloc;
use amdgpu_dsl::{DType, Ext, FnBuilder, Function, Op, Opnd};

fn has(rep: &Report, code: &str) -> bool {
    rep.findings.iter().any(|f| f.code == code && f.severity == Severity::Error)
}

fn errors(rep: &Report) -> String {
    rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n")
}

fn tid(b: &mut FnBuilder) -> amdgpu_dsl::ValueId {
    b.let_(Op::WorkitemId)
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// LDS budget

fn lds_kernel(lds_bytes: u32, offset: i32) -> Function {
    let mut b = FnBuilder::new("lds_probe", 64);
    b.lds(lds_bytes);
    let t = tid(&mut b);
    let c16 = b.c_i32(16);
    let a = b.mul_i32(t, c16);
    let a = b.add_i32(a, Opnd::I32(offset));
    let z = b.c_i32(0);
    let v = b.frag4(z, z, z, z);
    b.store_lds_b128(a, v);
    b.finish()
}

#[test]
fn lds_inside_the_declared_segment_is_proven() {
    let r = verify_ir(&lds_kernel(1024, 0));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert_eq!(r.proven.get("lds"), Some(&1));
}

#[test]
fn lds_overflow_past_the_declared_segment_is_rejected() {
    // lane 63 writes [1008, 1024) but only 512 bytes are declared
    let r = verify_ir(&lds_kernel(512, 0));
    assert!(has(&r, "GV-LDS-OOB"), "{}", r.render());
}

#[test]
fn lds_wraparound_below_zero_is_rejected() {
    let r = verify_ir(&lds_kernel(4096, -16));
    assert!(has(&r, "GV-LDS-OOB"), "{}", r.render());
    assert!(r.errors()[0].msg.contains("lower bound not provably >= 0"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// global bounds

/// out[idx] = p[off] where `off` is built by `mk` from (tid*4, n); the extent of p is n*4 bytes.
fn global_kernel(range_n: Option<(i64, i64)>, extent: Option<Ext>, clamp: bool) -> Function {
    let mut b = FnBuilder::new("glb", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg("n", DType::I32, 8, 4);
    if let Some(e) = extent {
        b.arg_tensor("p", e);
    }
    if let Some((lo, hi)) = range_n {
        b.ka_range(8, lo, hi);
    }
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let n = b.let_(Op::KernargI32 { offset: 8 });
    let t = tid(&mut b);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(t, c4);
    let off = if clamp {
        let nm1 = b.add_i32(n, Opnd::I32(-1));
        let last = b.mul_i32(nm1, c4);
        b.min_i32(off, last)
    } else {
        off
    };
    let q = b.ptr_add(p, off);
    let v = b.load_f32(q, 0);
    b.store_f32(q, 0, v);
    b.finish()
}

fn n_bytes() -> Ext {
    Ext::ka(8) * Ext::c(4)
}

#[test]
fn global_access_proven_with_a_declared_extent_and_a_shape_bound() {
    // n >= 64 elements: lane 63 reads [252, 256) <= 4n
    let r = verify_ir(&global_kernel(Some((64, i64::MAX)), Some(n_bytes()), false));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert_eq!(r.proven.get("global"), Some(&2)); // load + store
}

#[test]
fn global_access_without_the_shape_bound_is_unprovable_and_prints_the_expression() {
    let r = verify_ir(&global_kernel(None, Some(n_bytes()), false));
    assert!(has(&r, "GV-GLB-OOB"), "{}", r.render());
    let m = &r.errors()[0].msg;
    assert!(m.contains("ka(0x8)"), "the symbolic expression must be printed: {m}");
}

#[test]
fn global_access_without_a_declared_extent_is_rejected() {
    let r = verify_ir(&global_kernel(Some((64, i64::MAX)), None, false));
    assert!(has(&r, "GV-GLB-NOEXT"), "{}", r.render());
}

#[test]
fn global_min_clamp_is_understood_but_only_with_a_nonempty_tensor() {
    // clamp to the last element: proven when n >= 1 ...
    let r = verify_ir(&global_kernel(Some((1, i64::MAX)), Some(n_bytes()), true));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    // ... and rejected when n may be 0 (the clamp bound (n-1)*4 would be negative)
    let r = verify_ir(&global_kernel(None, Some(n_bytes()), true));
    assert!(has(&r, "GV-GLB-OOB"), "{}", r.render());
}

#[test]
fn stores_are_checked_against_the_logical_extent_not_the_zeroed_slack() {
    let mut b = FnBuilder::new("slack", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg_tensor_slack("p", Ext::c(256), Ext::c(512));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = tid(&mut b);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(t, c4);
    let q = b.ptr_add(p, off);
    let v = b.load_f32(q, 256); // reads the slack: inside alloc (512)
    b.store_f32(q, 256, v); // writes the slack: outside logical (256)
    let f = b.finish();
    let r = verify_ir(&f);
    assert_eq!(r.proven.get("global"), Some(&1), "{}", r.render());
    assert!(has(&r, "GV-GLB-OOB"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// buffer descriptors

fn desc_kernel(extent: i64, num_records: u32, dynamic_over: Option<i32>) -> Function {
    let mut b = FnBuilder::new("desc", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg("n", DType::I32, 8, 4);
    if dynamic_over.is_some() {
        b.ka_range(8, 1, i64::MAX);
        b.arg_tensor("p", Ext::ka(8));
    } else {
        b.arg_tensor("p", Ext::c(extent));
    }
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let t = tid(&mut b);
    let c4 = b.c_i32(4);
    let off = b.mul_i32(t, c4);
    let z = b.c_i32(0);
    let d = if let Some(extra) = dynamic_over {
        let n = b.let_(Op::KernargI32 { offset: 8 });
        let nn = b.add_i32(n, Opnd::I32(extra));
        b.buffer_desc_dyn(p, nn)
    } else {
        b.buffer_desc(p, num_records)
    };
    let _v = b.buffer_load_f32(d, off, z, 0);
    b.finish()
}

#[test]
fn a_descriptor_within_the_allocation_is_proven() {
    let r = verify_ir(&desc_kernel(256, 256, None));
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert_eq!(r.proven.get("descriptor"), Some(&1));
    let r = verify_ir(&desc_kernel(0, 0, Some(0)));
    assert!(r.errors().is_empty(), "runtime num_records = n: {}", errors(&r));
}

#[test]
fn a_descriptor_that_overstates_its_allocation_is_rejected() {
    let r = verify_ir(&desc_kernel(256, 512, None));
    assert!(has(&r, "GV-BUF-OVER"), "{}", r.render());
    // runtime num_records = n + 16 over an allocation of n bytes
    let r = verify_ir(&desc_kernel(0, 0, Some(16)));
    assert!(has(&r, "GV-BUF-OVER"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// loop variables: assume is checked inductively

fn ring_kernel(update: &str) -> Function {
    let mut b = FnBuilder::new("ring", 64);
    b.lds(8192);
    let t = tid(&mut b);
    let z = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let ring = vars[0];
    let stride = b.c_i32(2048);
    let base = b.mul_i32(ring, stride);
    let c16 = b.c_i32(16);
    let lane = b.mul_i32(t, c16);
    let a = b.add_i32(base, lane);
    let zz = b.c_i32(0);
    let v = b.frag4(zz, zz, zz, zz);
    b.store_lds_b128(a, v);
    match update {
        "xor1" => {
            let nx = b.let_(Op::XorI32(Opnd::Var(ring), Opnd::I32(1)));
            b.set_var(ring, nx);
        }
        _ => {
            let nx = b.add_i32(ring, Opnd::I32(1)); // grows without bound
            b.set_var(ring, nx);
        }
    }
    let stop = b.cmp_lt_i32(zz, zz);
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.finish()
}

#[test]
fn a_toggling_ring_variable_is_bounded_by_the_loop_fixpoint() {
    // ring in {0,1}: addresses [0, 2048 + 1008 + 16) inside 8192
    let r = verify_ir(&ring_kernel("xor1"));
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn a_ring_variable_that_grows_without_bound_is_rejected() {
    let r = verify_ir(&ring_kernel("inc"));
    assert!(has(&r, "GV-LDS-OOB"), "{}", r.render());
    assert!(r.errors()[0].msg.contains("loop variable"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// role / barrier protocol

fn role_kernel(extra_barrier_in_else: bool) -> Function {
    let mut b = FnBuilder::new("roles", 128);
    let t = tid(&mut b);
    let c64 = b.c_i32(64);
    let is_a = b.cmp_lt_i32(t, c64);
    b.begin_if(is_a);
    b.barrier();
    b.barrier();
    b.else_();
    b.barrier();
    if extra_barrier_in_else {
        b.barrier();
    }
    b.end_if();
    b.finish()
}

#[test]
fn balanced_role_barriers_are_accepted() {
    let r = verify_ir(&role_kernel(true));
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn unbalanced_role_barriers_are_rejected() {
    let r = verify_ir(&role_kernel(false));
    assert!(has(&r, "GV-BAR"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// MIR: register legality

fn mfunc(blocks: Vec<(Vec<MInst>, Term)>, classes: Vec<RClass>) -> MFunction {
    let mut bs = Vec::new();
    for (i, (insts, term)) in blocks.into_iter().enumerate() {
        let mut b = Block::new(i as u32);
        b.insts = insts;
        b.term = term;
        bs.push(b);
    }
    MFunction { name: "gv".into(), args: Vec::new(), kernarg_size: 0, workgroup_size: 64, lds_bytes: 0, blocks: bs, vregs: classes, loops: Vec::new(), alias: Default::default(), elide_fallthrough: false, agpr_pin: Default::default() }
}

fn alloc(phys: Vec<u32>, classes: &[RClass]) -> Alloc {
    Alloc { phys, max_v: 64, max_s: 24, scalar: classes.iter().map(|c| !c.is_vector()).collect() }
}

fn dummy() -> Function {
    FnBuilder::new("dummy", 64).finish()
}

fn mfma_case(phys: Vec<u32>) -> Report {
    // 0 = A (V8), 1 = B (V8), 2 = D (V16)
    let classes = vec![RClass::V8, RClass::V8, RClass::V16];
    let m = mfunc(vec![(vec![MInst::new(MOp::VMfma32x32x64Fp8Z).def(2).srcreg(0).srcreg(1)], Term::Endpgm)], classes.clone());
    verify_mir(&dummy(), &m, &alloc(phys, &classes))
}

#[test]
fn legal_mfma_register_assignment_is_accepted() {
    let r = mfma_case(vec![16, 24, 32]);
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn misaligned_mfma_destination_is_rejected() {
    let r = mfma_case(vec![16, 24, 41]); // 16-dword accumulator at an ODD register: the assembler rejects it
    assert!(has(&r, "GV-REG-ALIGN"), "{}", r.render());
}

#[test]
fn isa_legal_but_width_unaligned_tuples_are_a_policy_note_not_an_error() {
    // measured with the gfx950 assembler: V16 / V8 tuples are legal at any EVEN start register (v[2:17], v[6:21]); our allocator aligns to the width (policy)
    let r = mfma_case(vec![18, 26, 42]);
    assert!(r.errors().is_empty(), "{}", errors(&r));
    assert!(r.findings.iter().any(|f| f.code == "GV-REG-POLICY"), "{}", r.render());
}

#[test]
fn mfma_multiplicand_overlapping_the_destination_is_rejected() {
    let r = mfma_case(vec![16, 32, 32]); // B = v[32:40) inside D = v[32:48)
    assert!(has(&r, "GV-REG-MFMA"), "{}", r.render());
}

#[test]
fn descriptor_quad_and_pair_alignment_and_limits_are_checked() {
    let classes = vec![RClass::S4, RClass::S2];
    let m = mfunc(vec![(vec![MInst::new(MOp::SBufferRsrc).def(0).srcreg(1).imm(0)], Term::Endpgm)], classes.clone());
    let r = verify_mir(&dummy(), &m, &alloc(vec![10, 8], &classes)); // S4 at s10: not a multiple of 4 (assembler-measured: s_load_dwordx4 / descriptors at s2, s10 are rejected)
    assert!(has(&r, "GV-REG-ALIGN"), "{}", r.render());
    let r = verify_mir(&dummy(), &m, &alloc(vec![200, 8], &classes)); // beyond the SGPR file
    assert!(has(&r, "GV-REG-LIMIT"), "{}", r.render());
    let classes = vec![RClass::V4];
    let m = mfunc(vec![(vec![MInst::new(MOp::VMovB32).def(0).imm(0)], Term::Endpgm)], classes.clone());
    let r = verify_mir(&dummy(), &m, &alloc(vec![254], &classes)); // v[254:258) beyond 256 VGPRs
    assert!(has(&r, "GV-REG-LIMIT") || has(&r, "GV-REG-ALIGN"), "{}", r.render());
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// MIR: path-sensitive use-before-wait

fn wait_cfg(wait_on_taken: bool, wait_on_fall: bool) -> Report {
    // 0 = V2 address, 1 = V1 loaded value, 2 = V1 consumer
    let classes = vec![RClass::V2, RClass::V1, RClass::V1];
    let load = MInst::new(MOp::GlobalLoadF32).def(1).srcreg(0).imm(0);
    let mut b1 = Vec::new();
    if wait_on_taken {
        b1.push(MInst::new(MOp::SWaitVm).imm(0));
    }
    let mut b2 = Vec::new();
    if wait_on_fall {
        b2.push(MInst::new(MOp::SWaitVm).imm(0));
    }
    let m = mfunc(
        vec![
            (vec![load], Term::CondGoto { on: Pred::Vccnz, target: 1, fall: 2 }),
            (b1, Term::Goto(3)),
            (b2, Term::Goto(3)),
            (vec![MInst::new(MOp::VMovB32).def(2).srcreg(1)], Term::Endpgm),
        ],
        classes.clone(),
    );
    verify_mir(&dummy(), &m, &alloc(vec![2, 6, 7], &classes))
}

#[test]
fn a_wait_on_every_path_is_accepted() {
    let r = wait_cfg(true, true);
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn a_wait_missing_on_one_branch_path_is_rejected() {
    let r = wait_cfg(true, false);
    assert!(has(&r, "GV-WAIT"), "{}", r.render());
    let r = wait_cfg(false, true);
    assert!(has(&r, "GV-WAIT"), "{}", r.render());
}

fn loop_cfg(wait_before_use: bool) -> Report {
    // 0 = V2 address, 1 = V1 loaded value (loop carried), 2 = V1 consumer
    let classes = vec![RClass::V2, RClass::V1, RClass::V1];
    let mut body = Vec::new();
    if wait_before_use {
        body.push(MInst::new(MOp::SWaitVm).imm(0));
    }
    body.push(MInst::new(MOp::VMovB32).def(2).srcreg(1)); // consume the value loaded by the previous iteration
    body.push(MInst::new(MOp::GlobalLoadF32).def(1).srcreg(0).imm(0)); // issue the next load (in flight across the back edge)
    let m = mfunc(
        vec![
            (vec![MInst::new(MOp::VMovB32).def(1).imm(0)], Term::Goto(1)),
            (body, Term::CondGoto { on: Pred::Vccnz, target: 1, fall: 2 }),
            (vec![], Term::Endpgm),
        ],
        classes.clone(),
    );
    verify_mir(&dummy(), &m, &alloc(vec![2, 6, 7], &classes))
}

#[test]
fn a_loop_carried_in_flight_value_needs_a_wait_before_its_use_in_the_next_iteration() {
    let r = loop_cfg(false);
    assert!(has(&r, "GV-WAIT"), "{}", r.render());
    let r = loop_cfg(true);
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// shipped kernels: the register / wait checks (enforced by default) pass on every one

#[test]
fn shipped_kernels_pass_the_enforced_mir_level_checks() {
    for name in [
        "attention_fwd",
        "probe_lds",
        "probe_buffer_rw",
        "probe_mfma_unscaled",
        "fmha_fwd_hd128_mxfp8_t",
        "fmha_fwd_hd128_mxfp8_t128",
        "fmha_fwd_hd128_mxfp8_vs128",
        "fmha_fwd_hd128_mxfp8_kvsplit",
        "fmha_fwd_hd128_mxfp8_kvsplit_combine8h",
        "fmha_fwd_hd128_mxfp8_dsl_vs128",
        "fmha_fwd_hd128_mxfp8_dsl_t128",
        "fmha_fwd_hd128_mxfp8_dsl_kvsplit",
        "fmha_fwd_hd128_mxfp8_dsl_kvsplit_combine8h",
        "fmha_fwd_hd128_mxfp8_v3_vs128",
        "fmha_fwd_hd128_mxfp8_v3_t128",
        "fmha_fwd_hd128_mxfp8_v3_kvsplit",
        "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h",
    ] {
        let f = amdgpu_compiler::kernel_by_name(name).unwrap_or_else(|| panic!("unknown kernel {name}"));
        let rep = analyse(&f).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mir_errors: Vec<_> = rep.errors().into_iter().filter(|e| e.code.starts_with("GV-REG") || e.code == "GV-WAIT").collect();
        assert!(mir_errors.is_empty(), "{name}: {mir_errors:?}");
        // and the default-mode compile (which enforces them) succeeds
        amdgpu_compiler::compile(&f).unwrap_or_else(|e| panic!("{name}: compile failed: {e}"));
    }
}
// ---------------------------------------------------------------------------------------------------------------------------------------------
// assume soundness (kernel-occ's gv_attack_assume_loop): a loop-carried `assume` must be inductive, otherwise it is a proof hole

fn assume_loop(trusted: bool, init: i32) -> Function {
    let mut b = FnBuilder::new("assume_loop", 64);
    b.lds(64);
    let z = b.c_i32(init);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let i = vars[0];
    if trusted {
        b.assume_trusted(i, Ext::c(0), Ext::c(0), "test: bounded by a guard the analyzer cannot see");
    } else {
        b.assume(i, Ext::c(0), Ext::c(0));
    }
    let c16 = b.c_i32(16);
    let a = b.mul_i32(i, c16);
    let zero = b.c_i32(0);
    let v = b.frag4(zero, zero, zero, zero);
    b.store_lds_b128(a, v);
    let nx = b.add_i32(i, Opnd::I32(1)); // the loop DOES leave the assumed range [0, 0]
    b.set_var(i, nx);
    let stop = b.cmp_lt_i32(zero, zero);
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    b.finish()
}

#[test]
fn a_non_inductive_loop_assume_is_an_error_not_a_certification() {
    let r = verify_ir(&assume_loop(false, 0));
    assert!(has(&r, "GV-ASSUME-VIOLATED"), "{}", r.render());
    assert!(r.errors().iter().any(|e| e.msg.contains("not inductive")), "{}", r.render());
}

#[test]
fn an_assume_whose_initial_value_is_outside_the_range_is_an_error() {
    let r = verify_ir(&assume_loop(false, 5));
    assert!(has(&r, "GV-ASSUME-VIOLATED"), "{}", r.render());
}

#[test]
fn assume_trusted_is_explicit_and_listed_not_an_error() {
    let r = verify_ir(&assume_loop(true, 0));
    assert!(!has(&r, "GV-ASSUME-VIOLATED"), "{}", r.render());
    assert!(r.findings.iter().any(|f| f.code == "GV-ASSUME" && f.msg.contains("TRUSTED")), "{}", r.render());
}

#[test]
fn an_inductive_assume_is_proven() {
    // i in [0, 1], toggled by xor 1: the step stays inside
    let mut b = FnBuilder::new("assume_ok", 64);
    b.lds(64);
    let z = b.c_i32(0);
    let vars = b.begin_loop(&[(DType::I32, Opnd::Val(z))]);
    let i = vars[0];
    b.assume(i, Ext::c(0), Ext::c(1));
    let nx = b.let_(Op::XorI32(Opnd::Var(i), Opnd::I32(1)));
    b.set_var(i, nx);
    let stop = b.cmp_lt_i32(z, z);
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    let r = verify_ir(&b.finish());
    assert!(r.errors().is_empty(), "{}", errors(&r));
}

#[test]
fn a_kernarg_scalar_restricted_to_a_set_uses_its_min_and_max() {
    // splits in {2, 4, 8}: an access n*4 bytes into a tensor of 8 bytes is in bounds only because splits >= 2
    let mut b = FnBuilder::new("oneof", 64);
    b.arg("p", DType::Ptr, 0, 8);
    b.arg("s", DType::I32, 8, 4);
    b.ka_one_of(8, &[2, 4, 8]);
    b.arg_tensor("p", Ext::ka(8) * Ext::c(4));
    let p = b.let_(Op::KernargPtr { offset: 0 });
    let _v = b.load_f32(p, 4);
    let r = verify_ir(&b.finish());
    assert!(r.errors().is_empty(), "{}", errors(&r));
}
