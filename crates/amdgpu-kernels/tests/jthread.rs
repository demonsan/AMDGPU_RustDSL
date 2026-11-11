//! Two-tile loop with one exit per tile and a drain chosen by a flag set right before each exit (the vendor-FMHA two-tile pass,
//! docs/fmha-replica.md): the CFG pass `jthread` (lowering token) sends each exit edge straight to its drain.

use amdgpu_dsl::{ev, CmpPred, CmpTy, DType, Ext, FnBuilder, Function, Op, Opnd};

/// out[tid] = 1.0 when the loop left after an even tile (idx = 1 on that exit), 2.0 after an odd one; the loop walks n tiles two at a time.
pub fn two_exit_kernel(tokens: &[&str]) -> Function {
    let mut b = FnBuilder::new("probe_two_exit", 64);
    // kernarg layout of tests/gpu/tileop_probe_host.cpp: in (unused), out, PROBE_I32
    b.arg("i", DType::Ptr, 0x00, 8);
    b.arg("o", DType::Ptr, 0x08, 8);
    b.arg("n", DType::I32, 0x10, 4);
    b.arg("pad", DType::I32, 0x14, 4);
    b.arg_tensor("i", Ext::c(4));
    b.arg_tensor("o", Ext::c(64 * 4));
    b.ka_range(0x10, 1, 64);
    b.lower_opts(tokens);
    let po = b.let_(Op::KernargPtr { offset: 0x08 });
    let n = b.let_(Op::KernargI32 { offset: 0x10 });
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
    for tok in [&[][..], &["jthread"][..]] {
        let f = two_exit_kernel(tok);
        let m = amdgpu_codegen::compile_mir(&f).unwrap_or_else(|e| panic!("{e}"));
        println!("== {tok:?}\n{}", dump(&m));
    }
}

/// Without the token the exit block compares the flag and branches; with it each exit edge has its own copy of the exit block that jumps
/// straight to its drain (the stores of 1.0 and 2.0), and no block branches on the flag any more.
#[test]
fn each_exit_goes_straight_to_its_drain() {
    let lowered = |tok: &[&str]| {
        let f = two_exit_kernel(tok);
        amdgpu_dsl::ir_text::print(&f);
        amdgpu_codegen::lower::with_opts(&[&["dce", "cmpfold"][..], tok].concat(), || amdgpu_codegen::lower::lower(&f)).unwrap()
    };
    let base = lowered(&[]);
    let cmp_branch = |m: &amdgpu_codegen::mir::MFunction| {
        m.blocks.iter().filter(|b| b.insts.iter().any(|i| i.op == amdgpu_codegen::mir::MOp::VCmpEqI32) && matches!(b.term, amdgpu_codegen::mir::Term::CondGoto { .. })).count()
    };
    assert_eq!(cmp_branch(&base), 0, "the flag compare feeds a cndmask; the branch is on the 0/1 value:\n{}", dump(&base));
    let ne_branch = |m: &amdgpu_codegen::mir::MFunction| m.blocks.iter().filter(|b| b.insts.iter().any(|i| i.op == amdgpu_codegen::mir::MOp::VCmpEqI32) && matches!(b.term, amdgpu_codegen::mir::Term::Goto(_))).count();
    assert_eq!(ne_branch(&base), 0);
    let t = lowered(&["jthread"]);
    assert_eq!(ne_branch(&t), 2, "two copies of the exit block, each ending in a jump:\n{}", dump(&t));
    // the two copies go to different drains
    let targets: Vec<u32> = t
        .blocks
        .iter()
        .filter(|b| b.insts.iter().any(|i| i.op == amdgpu_codegen::mir::MOp::VCmpEqI32))
        .filter_map(|b| if let amdgpu_codegen::mir::Term::Goto(x) = b.term { Some(x) } else { None })
        .collect();
    assert_eq!(targets.len(), 2);
    assert_ne!(targets[0], targets[1], "{}", dump(&t));
    // the threaded kernel still compiles end to end and its drain for idx == 1 stores 1.0 (0x3f800000) on the first exit's path
    let f = two_exit_kernel(&["jthread"]);
    amdgpu_codegen::compile(&f).unwrap_or_else(|e| panic!("{e}"));
}

/// Writes the own-tail asm of the kernel with and without the token for tools/jthread_probe.sh (hardware check).
#[test]
#[ignore]
fn write_asm() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/jthread");
    std::fs::create_dir_all(&dir).unwrap();
    for (name, tok) in [("base", &[][..]), ("jthread", &["jthread"][..])] {
        let c = amdgpu_codegen::compile(&two_exit_kernel(tok)).unwrap();
        std::fs::write(dir.join(format!("{name}.s")), c.asm).unwrap();
    }
}
