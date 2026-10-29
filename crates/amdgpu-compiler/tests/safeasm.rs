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
    let asm = block_to_asm(&s.finish()).unwrap();
    assert!(asm.contains("v_mfma_scale_f32_32x32x64_f8f6f4") && asm.contains("v_smfmac_f32_16x16x128_fp8_fp8"), "{asm}");
    assembles(&asm);
}

#[test]
fn waits_are_inserted_by_construction() {
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let b = s.finish();
    let ops: Vec<&AInst> = b.insts.iter().collect();
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
    let m = block_to_mfunction(&s.finish(), "pv_safe").unwrap();
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
    let all = |op: MOp| !matches!(op, MOp::VWriteSub | MOp::VMovB32);
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
    let b = s.finish();
    let waits: Vec<i64> = b.insts.iter().filter(|i| i.op == "SWaitLgkm").map(|i| i.imm).collect();
    assert_eq!(waits, vec![2, 0], "{:?}", b.insts.iter().map(|i| (i.op, i.imm)).collect::<Vec<_>>());
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
    let b = s.finish();
    let w: Vec<(&str, i64)> = b.insts.iter().filter(|i| i.op.starts_with("SWait")).map(|i| (i.op, i.imm)).collect();
    assert_eq!(w, vec![("SWaitLgkm", 0), ("SWaitVm", 0)]);
}

#[test]
fn block_insts_round_trip_names() {
    let b = rowsum_chain_block();
    assert!(block_insts(&b).unwrap().iter().any(|i| i.op == MOp::VSmfmac16x16x128Fp8));
}