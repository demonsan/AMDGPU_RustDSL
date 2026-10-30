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
        .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6", "-c", "-o"])
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
    let r0 = s.ds_read_b128(d0, &addr, 0);
    let r1 = s.ds_read_b128(d1, &addr, 16);
    let r2 = s.ds_read_b128(d2, &addr, 32);
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
    let l = s.ds_read_b128(dl, &addr, 0);
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
    s.ds_write_b32(&a, &d, 0);
    s.wait_lgkm_all();
    let mut x = s.vgpr::<1>();
    s.v_mov_b32(&mut x, Imm::<1>::new());
    s.output(x);
    s.finish().unwrap();
    let mut s = Seq::new();
    let a = s.splat(0);
    let d = s.splat(1);
    s.ds_write_b32(&a, &d, 0);
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

/// Prints the coverage matrix used in docs/safe-asm-design.md (`cargo test -p amdgpu-compiler --test safeasm coverage -- --nocapture`) and checks
/// that every typed template names an op the backend table knows.
#[test]
fn coverage_matrix() {
    use amdgpu_safeasm::{HAND_WRITTEN_OPS, TYPED_OPS};
    let tbl = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/backend/llvm24.tbl")).unwrap();
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
