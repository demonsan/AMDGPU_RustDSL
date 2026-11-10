//! Safe-asm sequences through the compiler: standalone assembly, hazard padding, equality with the DSL-built path, embedding via `Op::Asm`
//! and the DS/VMEM scoreboard (`Pending` tokens).

use amdgpu_compiler::asm::{block_insts, block_to_asm, block_to_mfunction};
use amdgpu_compiler::mir::{MFunction, MInst, MOp, Src, VReg};
use amdgpu_compiler::{compile, lower};
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
use amdgpu_safeasm::examples::{pv_rowsum_phase, rowsum_chain_block};
use amdgpu_safeasm::*;
use std::collections::HashMap;

#[test]
fn safeasm_crate_forbids_unsafe_and_contains_none() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../amdgpu-safeasm/src");
    let mut saw_forbid = false;
    for f in std::fs::read_dir(dir).unwrap() {
        let p = f.unwrap().path();
        let t = std::fs::read_to_string(&p).unwrap();
        for (n, l) in t.lines().enumerate() {
            let l = l.trim();
            if l.starts_with("//") {
                continue;
            }
            if l.contains("#![forbid(unsafe_code)]") {
                saw_forbid = true;
                continue;
            }
            assert!(!l.contains("unsafe"), "{}:{}: `unsafe` found: {l}", p.display(), n + 1);
        }
    }
    assert!(saw_forbid);
}

fn assembles(asm: &str) {
    let t = amdgpu_compiler::backend::LlvmTools::locate();
    let clang = t.tool("clang");
    if !std::process::Command::new(&clang).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIPPED: clang not runnable");
        return;
    }
    let dir = std::env::temp_dir().join(format!("safeasm-{}-{}", std::process::id(), asm.len()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.s"), asm).unwrap();
    let out = std::process::Command::new(&clang)
        .args(amdgpu_compiler::target::T.llvm.asm_args)
        .args(["-c", "-o"])
        .arg(dir.join("k.o"))
        .arg(dir.join("k.s"))
        .output()
        .unwrap();
    assert!(out.status.success(), "assembler rejected:\n{}\n{asm}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn pv_phase_assembles() {
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let asm = block_to_asm(&s.finish().unwrap()).unwrap();
    assert!(asm.contains("v_mfma_scale_f32_32x32x64_f8f6f4") && asm.contains("v_smfmac_f32_16x16x128_fp8_fp8"), "{asm}");
    assembles(&asm);
}

#[test]
fn waits_are_inserted_by_construction() {
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let b = s.finish().unwrap();
    let ops: Vec<&AInst> = b.insts().iter().collect();
    let smf: Vec<usize> = ops.iter().enumerate().filter(|(_, i)| i.op == "VSmfmac16x16x128Fp8").map(|(k, _)| k).collect();
    assert_eq!(smf.len(), 4);
    for w in smf.windows(2) {
        assert_eq!(w[1], w[0] + 1, "chained SMFMACs must be back to back");
    }
    let last = *smf.last().unwrap();
    let reader = ops.iter().enumerate().skip(last + 1).find(|(_, i)| i.op == "VAddF32").map(|(k, _)| k).unwrap();
    let states: i64 = ops[last + 1..reader].iter().map(|i| if i.op == "SNop" { i.imm + 1 } else { 1 }).sum();
    assert!(states >= 20, "only {states} wait states between the SMFMAC chain and the reader");
}

// ---- identical instruction sequence vs the existing kernel path ----------------------------------------------------------------------

fn kernel_path() -> MFunction {
    let mut b = FnBuilder::new("pv_phase_dsl", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let c0 = b.c_i32(0);
    let (c1, c2, c3, c4) = (b.c_i32(1), b.c_i32(2), b.c_i32(3), b.c_i32(4));
    let pa = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let pb = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let v0 = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let v1 = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let z16 = b.frag_zero(DType::Frag16);
    let o1 = b.mfma_scale_fp8(pa, v0, z16, c1, c2);
    let o2 = b.mfma_scale_fp8(pb, v1, o1, c1, c2);
    let a_rs = b.frag4(c0, c0, c0, c0);
    let z4 = b.frag_zero(DType::Frag4);
    let r1 = b.smfmac_fp8_16x16x128(a_rs, pa, c3, z4);
    let r2 = b.smfmac_fp8_16x16x128(a_rs, pb, c3, r1);
    let r3 = b.smfmac_fp8_16x16x128(a_rs, pa, c4, r2);
    let r4 = b.smfmac_fp8_16x16x128(a_rs, pb, c4, r3);
    let lu = b.frag4_lane(r4, 2);
    let lf = b.bitcast_f32(lu);
    let zf = b.c_f32(0.0);
    let l_new = b.add_f32(zf, lf);
    let o_lane = b.frag16_lane(o2, 0);
    let of = b.bitcast_f32(o_lane);
    b.store_f32(o, 0, l_new);
    b.store_f32(o, 4, of);
    lower::lower(&b.finish()).unwrap()
}

type Norm = Vec<(MOp, Option<(u32, u8)>, Vec<(u32, u8)>)>;

fn normalise(m: &MFunction, insts: &[MInst], keep: impl Fn(MOp) -> bool) -> Norm {
    let mut ids: HashMap<VReg, u32> = HashMap::new();
    let mut id_of = |m: &MFunction, r: VReg| -> (u32, u8) {
        let (base, off) = m.alias.get(&r).copied().unwrap_or((r, 0));
        let n = ids.len() as u32;
        (*ids.entry(base).or_insert(n), off)
    };
    insts
        .iter()
        .filter(|i| keep(i.op))
        .map(|i| {
            let d = i.dst.map(|d| id_of(m, d));
            let s: Vec<(u32, u8)> = i.srcs.iter().filter_map(|s| if let Src::Reg(r) = s { Some(id_of(m, *r)) } else { None }).collect();
            (i.op, d, s)
        })
        .collect()
}

#[test]
fn matrix_sequence_equals_the_existing_kernel_path() {
    let keep = |op: MOp| matches!(op, MOp::VMfmaScale32x32x64Fp8 | MOp::VMfmaScale32x32x64Fp8Z | MOp::VSmfmac16x16x128Fp8 | MOp::VAddF32);
    let kernel = kernel_path();
    let kinsts: Vec<MInst> = kernel.blocks.iter().flat_map(|b| b.insts.clone()).collect();
    let k = normalise(&kernel, &kinsts, keep);
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let m = block_to_mfunction(&s.finish().unwrap(), "pv_safe").unwrap();
    let safe = normalise(&m, &m.blocks[0].insts, keep);
    assert_eq!(k.len(), 7);
    assert_eq!(safe, k, "safe-DSL sequence differs from the kernel path");
}

// ---- embedding through Op::Asm: identical MIR and asm to the DSL-built chain ------------------------------------------------------------

fn chain_kernel(use_asm: bool) -> amdgpu_dsl::Function {
    let mut b = FnBuilder::new(if use_asm { "chain_asm" } else { "chain_dsl" }, 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let c0 = b.c_i32(0);
    let (c3, c4) = (b.c_i32(3), b.c_i32(4));
    let a_rs = b.frag4(c0, c0, c0, c0);
    let pa = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let pb = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let r = if use_asm {
        b.asm(rowsum_chain_block(), &[Opnd::Val(a_rs), Opnd::Val(pa), Opnd::Val(pb), Opnd::Val(c3), Opnd::Val(c4)])
    } else {
        let z4 = b.frag_zero(DType::Frag4);
        let r1 = b.smfmac_fp8_16x16x128(a_rs, pa, c3, z4);
        let r2 = b.smfmac_fp8_16x16x128(a_rs, pb, c3, r1);
        let r3 = b.smfmac_fp8_16x16x128(a_rs, pa, c4, r2);
        b.smfmac_fp8_16x16x128(a_rs, pb, c4, r3)
    };
    let lu = b.frag4_lane(r, 2);
    let lf = b.bitcast_f32(lu);
    b.store_f32(o, 0, lf);
    b.finish()
}

#[test]
fn embedded_block_lowers_to_the_same_mir_as_the_dsl_chain() {
    // the DSL path also builds an unused zero tuple (4 dead VWriteSub) before the SMFMAC chain: compare everything except zero-initialisation moves
    let all = |op: MOp| !matches!(op, MOp::VWriteSub | MOp::VMovB32 | MOp::SNop); // the safe path's own padding is explicit s_nop; the DSL path is padded later by the hazard pass
    let fa = lower::lower(&chain_kernel(true)).unwrap();
    let fd = lower::lower(&chain_kernel(false)).unwrap();
    let na = normalise(&fa, &fa.blocks.iter().flat_map(|b| b.insts.clone()).collect::<Vec<_>>(), all);
    let nd = normalise(&fd, &fd.blocks.iter().flat_map(|b| b.insts.clone()).collect::<Vec<_>>(), all);
    let show = |n: &Norm| n.iter().map(|(o, d, s)| format!("{o:?} d={d:?} s={s:?}")).collect::<Vec<_>>().join("\n");
    assert_eq!(na, nd, "--- asm\n{}\n--- dsl\n{}", show(&na), show(&nd));
}

#[test]
fn embedded_block_emits_identical_assembly() {
    let a = compile(&chain_kernel(true)).unwrap().asm;
    let d = compile(&chain_kernel(false)).unwrap().asm;
    // matrix instructions, waits and the store are identical; the safe path is leaner by the DSL path's 4 dead zero-init moves
    let core = |s: &str| s.lines().filter(|l| l.contains("v_smfmac") || l.contains("s_waitcnt") || l.contains("global_store")).map(|l| l.trim().to_string()).collect::<Vec<_>>();
    assert_eq!(core(&a), core(&d));
    let movs = |s: &str| s.lines().filter(|l| l.contains("v_mov_b32")).count();
    assert_eq!(movs(&a) + 4, movs(&d), "safe path should drop exactly the DSL path's dead zero tuple");
    assembles(&a);
}

#[test]
fn input_type_mismatch_is_rejected_by_verify() {
    let mut b = FnBuilder::new("bad", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let c0 = b.c_i32(0);
    let a_rs = b.frag4(c0, c0, c0, c0);
    // port 1 wants a Frag8, a Frag4 is bound
    let r = b.asm(rowsum_chain_block(), &[Opnd::Val(a_rs), Opnd::Val(a_rs), Opnd::Val(a_rs), Opnd::Val(c0), Opnd::Val(c0)]);
    let _ = r;
    let f = b.finish();
    let errs = amdgpu_dsl::verify(&f);
    assert!(errs.iter().any(|e| e.contains("asm")), "{errs:?}");
}

// ---- DS / VMEM scoreboard -------------------------------------------------------------------------------------------------------------

#[test]
fn pending_loads_emit_partial_waits_from_the_scoreboard() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let (d0, d1, d2) = (s.vgpr::<4>(), s.vgpr::<4>(), s.vgpr::<4>());
    let r0 = s.ds_read_b128::<0, _>(d0, &addr);
    let r1 = s.ds_read_b128::<16, _>(d1, &addr);
    let r2 = s.ds_read_b128::<32, _>(d2, &addr);
    let v0 = r0.wait(&mut s); // two later loads may stay in flight: lgkmcnt(2)
    let v2 = r2.wait(&mut s); // drains everything up to r2: lgkmcnt(0)
    let v1 = r1.wait(&mut s); // already covered: no instruction
    s.output(v0);
    s.output(v1);
    s.output(v2);
    let b = s.finish().unwrap();
    let waits: Vec<i64> = b.insts().iter().filter(|i| i.op == "SWaitLgkm").map(|i| i.imm).collect();
    assert_eq!(waits, vec![2, 0], "{:?}", b.insts().iter().map(|i| (i.op, i.imm)).collect::<Vec<_>>());
}

#[test]
fn vm_and_lgkm_queues_are_independent() {
    let mut s = Seq::new();
    let a2 = s.vgpr::<2>();
    let addr = s.splat(0);
    let (dg, dl) = (s.vgpr::<4>(), s.vgpr::<4>());
    let g = s.global_load_dwordx4(dg, &a2, 0);
    let l = s.ds_read_b128::<0, _>(dl, &addr);
    let vl = l.wait(&mut s); // lgkm queue holds only this load: lgkmcnt(0), the global load stays in flight
    let vg = g.wait(&mut s); // vmcnt(0)
    s.output(vl);
    s.output(vg);
    let b = s.finish().unwrap();
    let w: Vec<(&str, i64)> = b.insts().iter().filter(|i| i.op.starts_with("SWait")).map(|i| (i.op, i.imm)).collect();
    assert_eq!(w, vec![("SWaitLgkm", 0), ("SWaitVm", 0)]);
}

#[test]
fn block_insts_round_trip_names() {
    let b = rowsum_chain_block();
    assert!(block_insts(&b).unwrap().iter().any(|i| i.op == MOp::VSmfmac16x16x128Fp8));
}
// ---- attacks: every bypass of the type discipline must be refused by the poison flag / the hazard verifier -----------------------------------

use amdgpu_safeasm::OP_MFMA_SCALE_ZERO;

fn mfma_setup(s: &mut Seq) -> (Vgpr<8>, Vgpr<8>, Vgpr<1>, Vgpr<1>, Vgpr<16>) {
    (s.zeroed::<8>(), s.zeroed::<8>(), s.splat(127), s.splat(127), s.vgpr::<16>())
}

/// (a) F1: drop the in-flight token (`let _ = t`) and go on. The accumulator moved into the token, so it cannot even be read (compile error,
/// `fail_read_inflight_mfma_d.rs`); what remains is the abandoned token, which `finish` reports.
#[test]
fn attack_dropped_token_is_refused() {
    let mut s = Seq::new();
    let (a, b, sa, sb, o) = mfma_setup(&mut s);
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    let _ = t; // dropped immediately
    let e = s.finish().unwrap_err();
    assert!(e.contains("unretired matrix result"), "{e}");
}

/// (b) F2 (WAR): drop the token to release the operand borrows, then overwrite an operand the MFMA is still reading.
#[test]
fn attack_war_after_dropped_token_is_refused() {
    let mut s = Seq::new();
    let (a, mut b, sa, sb, o) = mfma_setup(&mut s);
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb);
    drop(t);
    let z = s.splat(0);
    let b0 = s.lane_mut::<8, 0>(&mut b);
    s.v_mov_b32(b0, &z);
    let e = s.finish().unwrap_err();
    assert!(e.contains("unretired matrix result") && e.contains("WAR"), "{e}");
}

/// The verifier is independent of the token discipline: a hand-written (forged) instruction list with the same holes is rejected too.
#[test]
fn verifier_rejects_forged_sequences() {
    // MFMA then an immediate reader of its result, 2 wait states VALU->matrix operand, WAW, WAR
    let mut s = Seq::new();
    let (a, b, sa, sb, o) = mfma_setup(&mut s);
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let o = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb).retire(&mut s);
    let ok = s.finish().unwrap();
    // forge: remove the retire padding
    let ops_no_pad: Vec<AInst> = ok.insts().iter().filter(|i| i.op != "SNop").cloned().collect();
    let _ = o;
    let forged = amdgpu_safeasm::forge_for_tests(&ok, ops_no_pad.clone());
    // the reader is missing in `ok`; append a VALU reader of the accumulator and a writer of an operand
    let acc = ok.insts().iter().find(|i| i.op == "VMfmaScale32x32x64Fp8Z").unwrap().dst.unwrap();
    let mut with_reader = ops_no_pad.clone();
    with_reader.push(AInst { op: "VAddF32", dst: Some(0), srcs: vec![ASrc::Reg(acc), ASrc::Reg(acc)], imm: 0 });
    let e1 = amdgpu_safeasm::forge_for_tests(&ok, with_reader).verify().unwrap_err().join("; ");
    assert!(e1.contains("unretired matrix result"), "{e1}");
    let mut waw = ops_no_pad.clone();
    waw.push(AInst { op: "VMovB32", dst: Some(acc), srcs: vec![], imm: 0 });
    let e2 = amdgpu_safeasm::forge_for_tests(&ok, waw).verify().unwrap_err().join("; ");
    assert!(e2.contains("WAW"), "{e2}");
    let mfma = ok.insts().iter().find(|i| i.op == "VMfmaScale32x32x64Fp8Z").unwrap();
    let operand = if let ASrc::Reg(r) = mfma.srcs[0] { r } else { unreachable!() };
    let mut war = ops_no_pad.clone();
    war.push(AInst { op: "VMovB32", dst: Some(operand), srcs: vec![], imm: 0 });
    let e3 = amdgpu_safeasm::forge_for_tests(&ok, war).verify().unwrap_err().join("; ");
    assert!(e3.contains("WAR"), "{e3}");
    // VALU write -> matrix operand with 0 wait states
    let mut valu = vec![AInst { op: "VMovB32", dst: Some(operand), srcs: vec![], imm: 7 }];
    valu.extend(ops_no_pad.iter().filter(|i| i.op == "VMfmaScale32x32x64Fp8Z").cloned());
    let e4 = amdgpu_safeasm::forge_for_tests(&ok, valu).verify().unwrap_err().join("; ");
    assert!(e4.contains("needs 2"), "{e4}");
    let _ = forged;
}

/// F4: `chain` exists only for the accumulate opcode; the `_zero` token needs `retire` (checked at compile time in `fail_chain_after_zero.rs`).
#[test]
fn zero_form_then_accumulate_form_gets_the_full_latency() {
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let b = s.finish().unwrap();
    let ops = b.insts();
    let z = ops.iter().position(|i| i.op == "VMfmaScale32x32x64Fp8Z").unwrap();
    let acc = ops.iter().position(|i| i.op == "VMfmaScale32x32x64Fp8").unwrap();
    let states: i64 = ops[z + 1..acc].iter().map(|i| if i.op == "SNop" { i.imm + 1 } else { 1 }).sum();
    assert!(states >= 20, "{states}");
    let _ = OP_MFMA_SCALE_ZERO;
}

/// F6: a second `vcc()` is a value, not a panic.
#[test]
fn second_vcc_is_none() {
    let mut s = Seq::new();
    assert!(s.vcc().is_some());
    assert!(s.vcc().is_none());
}
// ---- round 2: F8 inputs read-only, F9 outputs defined, F10 ds_write, F12 verifier robustness ------------------------------------------------

/// F9: an output port that nothing writes is refused.
#[test]
fn output_must_be_written() {
    let mut s = Seq::new();
    let x = s.vgpr::<1>();
    s.output(x);
    let e = s.finish().unwrap_err();
    assert!(e.contains("never written"), "{e}");
}

/// F10: a store followed by a wait finishes clean (no abandoned token), and without the wait the sequence is refused.
#[test]
fn ds_write_then_wait_all_is_clean() {
    let mut s = Seq::new();
    let a = s.splat(0);
    let d = s.splat(1);
    s.ds_write_b32::<0>(&a, &d);
    s.wait_lgkm_all();
    let mut x = s.vgpr::<1>();
    s.v_mov_b32(&mut x, Imm::<1>::new());
    s.output(x);
    s.finish().unwrap();
    let mut s = Seq::new();
    let a = s.splat(0);
    let d = s.splat(1);
    s.ds_write_b32::<0>(&a, &d);
    assert!(s.finish().unwrap_err().contains("outstanding"));
}

/// F12: a forged stream with a malformed immediate is an Err, never a panic or a wrapped counter.
#[test]
fn verifier_survives_malformed_input() {
    let mut s = Seq::new();
    let mut x = s.vgpr::<1>();
    s.v_mov_b32(&mut x, Imm::<1>::new());
    s.output(x);
    let ok = s.finish().unwrap();
    for imm in [-5i64, 99, i64::MIN, i64::MAX] {
        let mut v = ok.insts().to_vec();
        v.insert(0, AInst { op: "SNop", dst: None, srcs: vec![], imm });
        v.push(AInst { op: "SWaitLgkm", dst: None, srcs: vec![], imm });
        let r = amdgpu_safeasm::forge_for_tests(&ok, v).verify();
        assert!(r.is_err(), "imm {imm} accepted");
    }
}

/// F8: a stream that defines an input register is refused by the verifier, and a DSL kernel embedding it fails to lower.
#[test]
fn writing_an_input_port_is_refused() {
    let ok = rowsum_chain_block();
    let input = ok.inputs()[3].reg; // idx01
    let mut v = ok.insts().to_vec();
    v.insert(0, AInst { op: "VMovB32", dst: Some(input), srcs: vec![], imm: 2 });
    let forged = amdgpu_safeasm::forge_for_tests(&ok, v);
    let e = forged.verify().unwrap_err().join("; ");
    assert!(e.contains("INPUT port"), "{e}");
    let mut b = FnBuilder::new("clobber", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let c0 = b.c_i32(0);
    let a_rs = b.frag4(c0, c0, c0, c0);
    let pa = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let r = b.asm(forged, &[Opnd::Val(a_rs), Opnd::Val(pa), Opnd::Val(pa), Opnd::Val(c0), Opnd::Val(c0)]);
    let _ = r;
    let err = lower::lower(&b.finish()).unwrap_err();
    assert!(err.contains("INPUT port") || err.contains("hazard verifier"), "{err}");
}
// ---- coverage: which MOps have typed templates ---------------------------------------------------------------------------------------------

/// Prints the coverage matrix used in docs/safe-asm-design.md (`cargo test -p amdgpu-kernels --test safeasm coverage -- --nocapture`) and checks
/// that every typed template names an op the backend table knows.
#[test]
fn coverage_matrix() {
    use amdgpu_safeasm::{HAND_WRITTEN_OPS, TYPED_OPS};
    let tbl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../amdgpu-compiler/src/target/gfx950/llvm/llvm24.tbl");
    assert!(std::path::Path::new(tbl_path).exists(), "{tbl_path} not found (the LLVM MIR tables moved? update this path)");
    let tbl = std::fs::read_to_string(tbl_path).unwrap();
    let keys: Vec<String> = tbl
        .lines()
        .filter(|l| !l.starts_with('#') && l.contains('|'))
        .map(|l| l.split('|').next().unwrap().trim().split(':').next().unwrap().to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let typed: std::collections::BTreeSet<&str> = TYPED_OPS.iter().map(|t| t.1).chain(HAND_WRITTEN_OPS.iter().map(|t| t.1)).collect();
    for t in &typed {
        assert!(keys.iter().any(|k| k == t) || ["VCmpMaskLtI32", "VCndmaskM", "VWriteSub"].contains(t) || tbl.contains(t), "typed op {t} unknown to the backend table");
    }
    let raw: Vec<&String> = keys.iter().filter(|k| !typed.contains(k.as_str())).collect();
    eprintln!("COVERAGE typed(generated)={} typed(hand-written)={} raw-only={} of {} backend rows", TYPED_OPS.len(), HAND_WRITTEN_OPS.len(), raw.len(), keys.len());
    eprintln!("COVERAGE raw: {}", raw.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "));
    assert!(TYPED_OPS.len() >= 20);
}

/// Round 3: an explicit partial wait must wait for something; counts beyond the outstanding operations are rejected.
#[test]
fn explicit_wait_counts_are_checked() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let (d0, d1, d2) = (s.vgpr::<4>(), s.vgpr::<4>(), s.vgpr::<4>());
    let r0 = s.ds_read_b128::<0, _>(d0, &addr);
    let r1 = s.ds_read_b128::<16, _>(d1, &addr);
    let r2 = s.ds_read_b128::<32, _>(d2, &addr);
    assert!(s.wait_lgkm_cnt(3).is_err(), "3 outstanding: lgkmcnt(3) waits for nothing");
    s.wait_lgkm_cnt(1).unwrap(); // r0 and r1 complete, r2 may stay in flight
    let v0 = r0.wait(&mut s); // covered: no instruction
    let v1 = r1.wait(&mut s);
    let v2 = r2.wait(&mut s); // lgkmcnt(0)
    s.output(v0);
    s.output(v1);
    s.output(v2);
    let b = s.finish().unwrap();
    let w: Vec<i64> = b.insts().iter().filter(|i| i.op == "SWaitLgkm").map(|i| i.imm).collect();
    assert_eq!(w, vec![1, 0]);
}

/// The generated templates (opspec.tbl) carry the same class rules as the hand-written ones they replaced.
#[test]
fn generated_templates_cover_the_documented_set() {
    let names: Vec<&str> = amdgpu_safeasm::TYPED_OPS.iter().map(|t| t.0).collect();
    for n in ["v_add_u32", "v_add_f32", "v_mul_lo_u32", "v_cmp_lt_i32", "v_perm_b32", "v_fma_f32", "global_store_dword", "v_lshrrev_b32"] {
        assert!(names.contains(&n), "{n} missing from the generated set");
    }
}

// ---- round 4: typed DS / conversion / pack ops and the loop-body contract ----------------------------------------------------------------

#[test]
fn ds_ops_with_partial_counts_and_transposed_read() {
    let mut s = Seq::new();
    let addr = s.splat(0);
    let (d0, d1, d2, d3) = (s.vgpr::<4>(), s.vgpr::<4>(), s.vgpr::<2>(), s.vgpr::<1>());
    let r0 = s.ds_read_b128::<0, _>(d0, &addr);
    let r1 = s.ds_read_b128::<16, _>(d1, &addr);
    let r2 = s.ds_read_b64_tr_b8::<8, _>(d2, &addr);
    let r3 = s.ds_read_b32::<4, _>(d3, &addr);
    let w = s.zeroed::<4>();
    s.ds_write_b128::<32>(&addr, &w);
    let v0 = r0.wait(&mut s); // 4 younger DS ops outstanding (r1, r2, r3, the write): lgkmcnt(4)
    let v1 = r1.wait(&mut s); // lgkmcnt(3)
    let v2 = r2.wait(&mut s);
    let v3 = r3.wait(&mut s);
    s.wait_lgkm_all(); // the write
    for r in [v0, v1] {
        s.output(r);
    }
    s.output(v2);
    s.output(v3);
    let b = s.finish().unwrap();
    let w: Vec<i64> = b.insts().iter().filter(|i| i.op == "SWaitLgkm").map(|i| i.imm).collect();
    assert_eq!(w, vec![4, 3, 2, 1, 0], "{:?}", b.insts().iter().map(|i| (i.op, i.imm)).collect::<Vec<_>>());
}

#[test]
fn transcendental_results_are_padded_and_pk_feeds_exp_with_a_gap() {
    let mut s = Seq::new();
    let a = s.zeroed::<2>();
    let b = s.zeroed::<2>();
    let mut p = s.vgpr::<2>();
    s.v_pk_add_f32(&mut p, &a, &b);
    let l = s.lane::<2, 0>(&p);
    let d = s.vgpr::<1>();
    let e = s.v_exp_f32(d, l); // reads a packed-fp32 result: the builder pads the 1 wait state
    let e = e.settle(&mut s); // 4 wait states
    let mut y = s.vgpr::<1>();
    s.v_add_f32(&mut y, &e, &e);
    s.output(y);
    let blk = s.finish().unwrap();
    let ops = blk.insts();
    let pk = ops.iter().position(|i| i.op == "VPkAddF32").unwrap();
    let ex = ops.iter().position(|i| i.op == "VExpF32").unwrap();
    let add = ops.iter().position(|i| i.op == "VAddF32").unwrap();
    let gap = |a: usize, b: usize| -> i64 { ops[a + 1..b].iter().map(|i| if i.op == "SNop" { i.imm + 1 } else { 1 }).sum() };
    assert!(gap(pk, ex) >= 1, "pk -> exp");
    assert!(gap(ex, add) >= 4, "exp -> reader");
}

#[test]
fn partial_write_converts_build_a_word_and_the_verifier_rejects_a_half_read() {
    let mut s = Seq::new();
    let (a, b, c, e) = (s.splat(1), s.splat(2), s.splat(3), s.splat(4));
    let scale = s.splat(127);
    let d = s.vgpr::<1>();
    let pw = s.partial(d);
    let pw = s.v_cvt_scalef32_pk_fp8_bf16_lo(pw, &a, &scale);
    let pw = s.v_cvt_scalef32_pk_fp8_bf16_hi(pw, &b, &scale);
    let word = pw.done();
    let d2 = s.vgpr::<1>();
    let pw2 = s.partial(d2);
    let pw2 = s.v_cvt_pk_fp8_f32_lo(pw2, &c, &e);
    let pw2 = s.v_cvt_pk_fp8_f32_hi(pw2, &c, &e);
    let word2 = pw2.done();
    let mut y = s.vgpr::<1>();
    s.v_add_u32(&mut y, &word, &word2);
    s.output(y);
    let ok = s.finish().unwrap();
    // forge: drop the high-half convert of the first word, then read the dword
    let hi = ok.insts().iter().position(|i| i.op == "VCvtPkFp8Hi").unwrap();
    let mut v = ok.insts().to_vec();
    v.remove(hi);
    let e = amdgpu_safeasm::forge_for_tests(&ok, v).verify().unwrap_err().join("; ");
    assert!(e.contains("only the low half was written"), "{e}");
}

#[test]
fn loop_body_contract_is_drain_closed() {
    // legitimate: matrix retired, loads waited, nothing exported in flight
    let mut s = Seq::new();
    let a = s.input::<8>();
    let b = s.input::<8>();
    let sa = s.splat(127);
    let sb = s.splat(127);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let o = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb).retire(&mut s);
    s.output(o);
    assert!(s.finish_loop_body().is_ok());

    // an in-flight matrix result at the boundary is refused
    let mut s = Seq::new();
    let a = s.input::<8>();
    let b = s.input::<8>();
    let sa = s.splat(127);
    let sb = s.splat(127);
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let ex = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb).exit();
    s.output_exit(ex);
    let e = s.finish_loop_body().unwrap_err();
    assert!(e.contains("drain-closed"), "{e}");

    // an outstanding DS load at the boundary is refused
    let mut s = Seq::new();
    let addr = s.input::<1>();
    let d = s.vgpr::<4>();
    let p = s.ds_read_b128::<0, _>(d, &addr);
    drop(p);
    let e = s.finish_loop_body().unwrap_err();
    assert!(e.contains("outstanding") || e.contains("abandoned"), "{e}");
}

#[test]
fn raw_only_ops_are_not_allowed_in_a_sequence() {
    let mut s = Seq::new();
    let mut x = s.vgpr::<1>();
    s.v_mov_b32(&mut x, Imm::<1>::new());
    s.output(x);
    let ok = s.finish().unwrap();
    let mut v = ok.insts().to_vec();
    v.push(AInst { op: "VAddCoU32", dst: Some(ok.outputs()[0].reg), srcs: vec![ASrc::Imm(1), ASrc::Reg(ok.outputs()[0].reg)], imm: 0 });
    let e = amdgpu_safeasm::forge_for_tests(&ok, v).verify().unwrap_err().join("; ");
    assert!(e.contains("raw-only"), "{e}");
}
// ---- Block C enablers: writable views as DS destinations, multi-inout variables -----------------------------------------------------------

/// Four `ds_read_b64_tr_b8` fill one 8-dword MFMA operand through disjoint pair views (all four in flight together); `ready` needs the borrows (and
/// therefore the waits) to be over, and the MFMA is padded only by the DS waits (a DS-written register has no VALU write to wait for).
#[test]
fn four_transposed_reads_assemble_one_mfma_operand() {
    let mut s = Seq::new();
    let addr = s.input::<1>();
    let b = s.input::<8>();
    let sa = s.splat(127);
    let sb = s.splat(127);
    let mut v8 = s.vgpr::<8>();
    {
        let [d0, d1, d2, d3] = s.split_pairs_mut(&mut v8);
        let p0 = s.ds_read_b64_tr_b8::<0, _>(d0, &addr);
        let p1 = s.ds_read_b64_tr_b8::<2048, _>(d1, &addr);
        let p2 = s.ds_read_b64_tr_b8::<4096, _>(d2, &addr);
        let p3 = s.ds_read_b64_tr_b8::<6144, _>(d3, &addr);
        let _ = p0.wait(&mut s); // lgkmcnt(3)
        let _ = p1.wait(&mut s); // lgkmcnt(2)
        let _ = p2.wait(&mut s); // lgkmcnt(1)
        let _ = p3.wait(&mut s); // lgkmcnt(0)
    }
    let o = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&v8), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let o = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsb).retire(&mut s);
    s.output(o);
    let blk = s.finish().unwrap();
    let waits: Vec<i64> = blk.insts().iter().filter(|i| i.op == "SWaitLgkm").map(|i| i.imm).collect();
    assert_eq!(waits, vec![3, 2, 1, 0]);
    let mf = blk.insts().iter().position(|i| i.op == "VMfmaScale32x32x64Fp8Z").unwrap();
    let last_wait = blk.insts().iter().rposition(|i| i.op == "SWaitLgkm").unwrap();
    assert!(last_wait < mf);
}

/// Multi-inout: a block updating two loop variables in place through two in/out ports (`Stmt::VarAsmN`), lowered into the same MIR as two separate
/// DSL in-place accumulations.
#[test]
fn var_asm_n_binds_inout_ports_in_declaration_order() {
    use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
    let mut s = Seq::new();
    let acc0 = s.inout::<4>();
    let acc1 = s.inout::<4>();
    let a_rs = s.input::<4>();
    let pa = s.input::<8>();
    let idx = s.input::<1>();
    let (ra, rpa, ri) = (s.ready(&a_rs), s.ready(&pa), s.ready(&idx));
    let e0 = s.v_smfmac_f32_16x16x128_fp8_fp8(acc0, ra, rpa, ri).exit();
    let e1 = s.v_smfmac_f32_16x16x128_fp8_fp8(acc1, ra, rpa, ri).exit();
    s.output_exit(e0);
    s.output_exit(e1);
    let blk = s.finish().unwrap();

    let build = |use_asm: bool| {
        let mut b = FnBuilder::new(if use_asm { "n_asm" } else { "n_dsl" }, 64);
        b.arg("o", DType::Ptr, 0, 8);
        let o = b.let_(Op::KernargPtr { offset: 0 });
        let c0 = b.c_i32(0);
        let c3 = b.c_i32(3);
        let a_rs = b.frag4(c0, c0, c0, c0);
        let pa = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
        let z0 = b.frag4(c0, c0, c0, c0);
        let z1 = b.frag4(c0, c0, c0, c0);
        let vars = b.begin_loop(&[(DType::Frag4, Opnd::Val(z0)), (DType::Frag4, Opnd::Val(z1))]);
        if use_asm {
            b.var_asm_n(&[vars[0], vars[1]], blk.clone(), &[Opnd::Val(a_rs), Opnd::Val(pa), Opnd::Val(c3)]);
        } else {
            b.var_smfmac_fp8(vars[0], a_rs, pa, c3);
            b.var_smfmac_fp8(vars[1], a_rs, pa, c3);
        }
        let stop = b.cmp_lt_i32(c0, c3); // always true: one iteration
        b.break_if(Opnd::Val(stop));
        b.end_loop();
        let x = b.frag4_lane(Opnd::Var(vars[1]), 2);
        let xf = b.bitcast_f32(x);
        b.store_f32(o, 0, xf);
        b.finish()
    };
    let fa = amdgpu_compiler::lower::lower(&build(true)).unwrap();
    let fd = amdgpu_compiler::lower::lower(&build(false)).unwrap();
    let smf = |m: &MFunction| m.blocks.iter().flat_map(|b| b.insts.iter()).filter(|i| i.op == MOp::VSmfmac16x16x128Fp8).count();
    assert_eq!(smf(&fa), 2);
    assert_eq!(smf(&fa), smf(&fd));
}
/// The safe-asm conversion chain (pinned block: DS write/transposed read, pk_add/mul, pknorm, svcorr, scalef32 lo/hi) emits the SAME sequence of
/// chain instructions as the DSL path of the same phase (the DSL path adds its own address `v_add` and copies; the chain ops and their order match).
#[test]
fn cvt_chain_instruction_sequence_matches_the_dsl_path() {
    use amdgpu_compiler::backend::OwnTail;
    let chain = |name: &str| -> Vec<String> {
        let f = amdgpu_kernels::kernel_by_name(name).unwrap();
        let asm = amdgpu_compiler::compile_with(&f, &OwnTail).unwrap().asm;
        asm.lines()
            .map(|l| l.trim().split_whitespace().next().unwrap_or("").to_string())
            .filter(|m| {
                matches!(
                    m.as_str(),
                    "v_pk_add_f32" | "v_pk_mul_f32" | "v_cvt_pknorm_u16_f32_e32" | "v_cvt_pknorm_u16_f32" | "v_lshrrev_b32_e32" | "v_and_b32_e32" | "v_perm_b32" | "v_pk_sub_u16"
                        | "v_cvt_scalef32_pk_fp8_bf16" | "ds_write_b32" | "ds_read_b64_tr_b8"
                )
            })
            .collect()
    };
    let d = chain("probe_cvtchain_dsl");
    let a = chain("probe_cvtchain_asm");
    assert!(d.len() >= 30, "unexpectedly short chain: {d:?}");
    assert_eq!(a, d, "safe-asm chain differs from the DSL chain");
}
/// Bias MFMA (f16 32x32x8, zero C) followed by a scaled MFMA accumulating into its result: different opcode => the full result latency (12 wait
/// states for the f16 op) is padded by `retire`, then the scaled MFMA chains in place; per-lane VGPR shift via `v_lshrrev_b32_v`.
#[test]
fn bias_mfma_then_scaled_accumulate_and_vgpr_shift() {
    let mut s = Seq::new();
    let a2 = s.input::<2>();
    let b2 = s.input::<2>();
    let a8 = s.input::<8>();
    let b8 = s.input::<8>();
    let sa = s.input::<1>();
    let sb = s.input::<1>();
    let kd = s.input::<1>();
    let g8 = s.input::<1>();
    let acc = s.vgpr::<16>();
    let (ra, rb) = (s.ready(&a2), s.ready(&b2));
    let acc = s.v_mfma_f32_32x32x8_f16_zero(acc, ra, rb).retire(&mut s);
    let (r8a, r8b, rsa, rsb) = (s.ready(&a8), s.ready(&b8), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4(acc, r8a, r8b, rsa, rsb);
    let mut sk0 = s.vgpr::<1>();
    s.v_lshrrev_b32_v(&mut sk0, &g8, &kd); // per-lane shift count in a VGPR
    let acc = t.retire(&mut s);
    s.output(acc);
    s.output(sk0);
    let blk = s.finish().unwrap();
    let ops = blk.insts();
    let f = ops.iter().position(|i| i.op == "VMfma32x32x8F16Z").unwrap();
    let m = ops.iter().position(|i| i.op == "VMfmaScale32x32x64Fp8").unwrap();
    let gap: i64 = ops[f + 1..m].iter().map(|i| if i.op == "SNop" { i.imm + 1 } else { 1 }).sum();
    assert!(gap >= 12, "{gap}");
    assert!(ops.iter().any(|i| i.op == "VLshrB32" && matches!(i.srcs[0], ASrc::Reg(_))));
}
#[test]
fn wide_global_stores_are_typed_and_use_a_vm_slot() {
    let mut s = Seq::new();
    let addr = s.vgpr::<2>();
    let d2 = s.zeroed::<2>();
    let d4 = s.zeroed::<4>();
    s.global_store_dwordx2(&addr, &d2, 8);
    s.global_store_dwordx4(&addr, &d4, 0);
    assert_eq!(s.outstanding(), (0, 2));
    s.wait_vm_all();
    let mut x = s.vgpr::<1>();
    s.v_mov_b32(&mut x, Imm::<1>::new());
    s.output(x);
    s.finish().unwrap();
}

// ---- scaled MFMA with an explicit C operand (persistent bias accumulator form) ---------------------------------------------------------------

#[test]
fn scaled_mfma_with_c_operand_reads_c_and_writes_a_different_register() {
    let mut s = Seq::new();
    let (a, b, c) = (s.input::<8>(), s.input::<8>(), s.input::<16>());
    let (sa, sb) = (s.splat(127), s.splat(127));
    let dst = s.vgpr::<16>();
    let (ra, rb, rc, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&c), s.ready(&sa), s.ready(&sb));
    let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_c(dst, rc, ra, rb, rsa, rsb);
    let t = t.chain(&mut s, ra, rb, rsa, rsb); // a second scaled MFMA accumulates in place on dst: same opcode, no wait
    let d = t.retire(&mut s);
    s.output(d);
    let blk = s.finish().unwrap();
    let m: Vec<_> = blk.insts().iter().filter(|i| i.op == "VMfmaScale32x32x64Fp8").collect();
    assert_eq!(m.len(), 2);
    let (d0, c0) = (m[0].dst.unwrap(), if let ASrc::Reg(r) = m[0].srcs[2] { r } else { unreachable!() });
    assert_ne!(d0, c0, "dst and C are different registers");
    // chain member is back to back
    let pos: Vec<usize> = blk.insts().iter().enumerate().filter(|(_, i)| i.op == "VMfmaScale32x32x64Fp8").map(|(k, _)| k).collect();
    assert_eq!(pos[1], pos[0] + 1);
}

/// The verifier treats C as a read of a matrix result: a C produced by an MFMA less than 20 states earlier is rejected (forged stream: the
/// type system already refuses it, see ui/fail_mfma_c_still_in_flight.rs).
#[test]
fn verifier_rejects_a_c_operand_younger_than_20_states() {
    let mut s = Seq::new();
    let (a, b) = (s.input::<8>(), s.input::<8>());
    let (sa, sb) = (s.splat(127), s.splat(127));
    let o1 = s.vgpr::<16>();
    let o2 = s.vgpr::<16>();
    let (ra, rb, rsa, rsb) = (s.ready(&a), s.ready(&b), s.ready(&sa), s.ready(&sb));
    let o1 = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o1, ra, rb, rsa, rsb).retire(&mut s);
    let rc = s.ready(&o1);
    let o2 = s.v_mfma_scale_f32_32x32x64_f8f6f4_c(o2, rc, ra, rb, rsa, rsb).retire(&mut s);
    s.output(o2);
    let ok = s.finish().unwrap();
    // forge: remove the padding between the producer of C and the C-form MFMA
    let v: Vec<AInst> = ok.insts().iter().filter(|i| i.op != "SNop").cloned().collect();
    let e = amdgpu_safeasm::forge_for_tests(&ok, v).verify().unwrap_err().join("; ");
    assert!(e.contains("unretired matrix result"), "{e}");
}

/// DSL level: `var_mfma_scale_c` reads a separate C and writes the loop variable (dst != C); the C operand of a persistent accumulator that a
/// previous scaled MFMA wrote is NOT padded (same-opcode interlock, in-order chain), one written by a VALU copy is padded 2 states.
#[test]
fn dsl_var_mfma_scale_c_lowers_with_distinct_dst_and_c() {
    use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};
    let mut b = FnBuilder::new("mfma_c", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let c0 = b.c_i32(0);
    let c127 = b.c_i32(127);
    let a8 = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let b8 = b.frag8(c0, c0, c0, c0, c0, c0, c0, c0);
    let z = b.frag_zero(DType::Frag16);
    let bias = b.frag_zero(DType::Frag16);
    let vars = b.begin_loop(&[(DType::Frag16, Opnd::Val(z))]);
    b.var_mfma_scale_c(vars[0], a8, b8, bias, c127, c127);
    let stop = b.cmp_lt_i32(c0, c127);
    b.break_if(Opnd::Val(stop));
    b.end_loop();
    let l = b.frag16_lane(Opnd::Var(vars[0]), 0);
    let lf = b.bitcast_f32(l);
    b.store_f32(o, 0, lf);
    let m = amdgpu_compiler::lower::lower(&b.finish()).unwrap();
    let mf = m.blocks.iter().flat_map(|bl| bl.insts.iter()).find(|i| i.op == MOp::VMfmaScale32x32x64Fp8).unwrap();
    let d = mf.dst.unwrap();
    let c = if let Src::Reg(r) = mf.srcs[2] { r } else { panic!() };
    assert_ne!(d, c);
}
// ---- unscaled fp8 MFMA (v_mfma_f32_32x32x64_f8f6f4) ---------------------------------------------------------------------------------------

#[test]
fn unscaled_fp8_mfma_chain_zero_and_c_forms() {
    let mut s = Seq::new();
    let (a, b, c) = (s.input::<8>(), s.input::<8>(), s.input::<16>());
    let (o1, o2) = (s.vgpr::<16>(), s.vgpr::<16>());
    let (ra, rb, rc) = (s.ready(&a), s.ready(&b), s.ready(&c));
    // zero-C form, then the in-place accumulate form needs the full result latency (different opcode), then chain back to back
    let o1 = s.v_mfma_f32_32x32x64_f8f6f4_zero(o1, ra, rb).retire(&mut s);
    let t = s.v_mfma_f32_32x32x64_f8f6f4(o1, ra, rb);
    let t = t.chain(&mut s, ra, rb);
    let o1 = t.retire(&mut s);
    // separate-C form reading a register the sequence only reads
    let o2 = s.v_mfma_f32_32x32x64_f8f6f4_c(o2, rc, ra, rb).retire(&mut s);
    s.output(o1);
    s.output(o2);
    let blk = s.finish().unwrap();
    let ops: Vec<&str> = blk.insts().iter().filter(|i| i.op.starts_with("VMfma32x32x64")).map(|i| i.op).collect();
    assert_eq!(ops, vec!["VMfma32x32x64Fp8Z", "VMfma32x32x64Fp8", "VMfma32x32x64Fp8", "VMfma32x32x64Fp8"]);
    let pos: Vec<usize> = blk.insts().iter().enumerate().filter(|(_, i)| i.op == "VMfma32x32x64Fp8").map(|(k, _)| k).collect();
    assert_eq!(pos[1], pos[0] + 1, "in-place chain of the same opcode is back to back");
}

#[test]
fn own_tail_chains_unscaled_mfma_without_nops_from_the_zero_form_on() {
    use amdgpu_compiler::hazards::fix_value_hazards_opts;
    use amdgpu_compiler::mir::{Block, MFunction, RClass};
    let mk = |insts: Vec<MInst>| {
        let mut b = Block::new(0);
        b.insts = insts;
        MFunction { name: "u".into(), args: Vec::new(), kernarg_size: 0, workgroup_size: 64, lds_bytes: 0, blocks: vec![b], vregs: vec![RClass::V8, RClass::V8, RClass::V16], loops: Vec::new(), alias: Default::default(), elide_fallthrough: false, edge_pad: false, agpr_pin: Default::default(), reg_hints: Default::default() }
    };
    let chain = |op| MInst::new(op).def(2).srcreg(0).srcreg(1).srcreg(2);
    // in-place accumulate then accumulate: hardware-interlocked, no s_nop
    let mut m = mk(vec![chain(MOp::VMfma32x32x64Fp8), chain(MOp::VMfma32x32x64Fp8)]);
    fix_value_hazards_opts(&mut m, true);
    assert!(m.blocks[0].insts.iter().all(|i| i.op != MOp::SNop), "{:?}", m.blocks[0].insts.iter().map(|i| i.op).collect::<Vec<_>>());
    // zero form then accumulate: the same machine instruction (srcC = inline 0), so the in-place chain is interlocked too (LLVM compares the
    // opcode, which the zero form shares; hardware: the bf16 / fp8 chains of the op checks start with the zero form)
    let mut m = mk(vec![MInst::new(MOp::VMfma32x32x64Fp8Z).def(2).srcreg(0).srcreg(1), chain(MOp::VMfma32x32x64Fp8)]);
    fix_value_hazards_opts(&mut m, true);
    assert!(m.blocks[0].insts.iter().all(|i| i.op != MOp::SNop));
    // ... but a VALU reader of the accumulator waits for the result latency
    let mut m = mk(vec![chain(MOp::VMfma32x32x64Fp8), MInst::new(MOp::VMovB32).def(0).srcreg(2)]);
    fix_value_hazards_opts(&mut m, true);
    assert!(m.blocks[0].insts.iter().any(|i| i.op == MOp::SNop));
}
