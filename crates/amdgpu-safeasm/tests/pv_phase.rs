//! The PV + row-sum phase written in the safe DSL: (a) the crate has no `unsafe`, (b) the emitted asm assembles, (c) the matrix/VALU
//! instruction sequence equals the one the existing kernel path (DSL -> lowering) produces for the same phase.

use amdgpu_compiler::lower;
use amdgpu_compiler::mir::{MFunction, MInst, MOp, Src, VReg};
use amdgpu_dsl::{DType, FnBuilder, Op};
use amdgpu_safeasm::examples::pv_rowsum_phase;
use amdgpu_safeasm::*;
use std::collections::HashMap;

#[test]
fn crate_forbids_unsafe_and_contains_none() {
    let src = std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).unwrap();
    let mut saw_forbid = false;
    for f in src {
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
    assert!(saw_forbid, "lib.rs must carry #![forbid(unsafe_code)]");
}

#[test]
fn pv_phase_assembles() {
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let asm = s.to_asm().unwrap();
    assert!(asm.contains("v_mfma_scale_f32_32x32x64_f8f6f4") && asm.contains("v_smfmac_f32_16x16x128_fp8_fp8"), "{asm}");
    let t = amdgpu_compiler::backend::LlvmTools::locate();
    let clang = t.tool("clang");
    if !std::process::Command::new(&clang).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIPPED: clang not runnable");
        return;
    }
    let dir = std::env::temp_dir().join(format!("safeasm-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.s"), &asm).unwrap();
    let out = std::process::Command::new(&clang)
        .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6", "-c", "-o"])
        .arg(dir.join("k.o"))
        .arg(dir.join("k.s"))
        .output()
        .unwrap();
    assert!(out.status.success(), "assembler rejected the safe sequence:\n{}\n{asm}", String::from_utf8_lossy(&out.stderr));
}

/// Hazard padding is exact: the SMFMAC chain needs no wait between its members, the first reader of the result gets the 20 wait states
/// (hardware-confirmed rule), operands written by VALU get the 2 wait states right before the matrix op.
#[test]
fn waits_are_inserted_by_construction() {
    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let ops: Vec<&MInst> = s.insts().iter().collect();
    let idx = |op: MOp| ops.iter().position(|i| i.op == op).unwrap();
    let first_smfmac = idx(MOp::VSmfmac16x16x128Fp8);
    // chained SMFMACs are adjacent: no s_nop between members of the chain
    let smf: Vec<usize> = ops.iter().enumerate().filter(|(_, i)| i.op == MOp::VSmfmac16x16x128Fp8).map(|(k, _)| k).collect();
    assert_eq!(smf.len(), 4);
    for w in smf.windows(2) {
        assert_eq!(w[1], w[0] + 1, "chained SMFMACs must be back to back");
    }
    // 20 wait states between the last SMFMAC and the first read of its result
    let last = *smf.last().unwrap();
    let reader = ops.iter().enumerate().skip(last + 1).find(|(_, i)| i.op == MOp::VAddF32).map(|(k, _)| k).unwrap();
    let states: u32 = ops[last + 1..reader].iter().map(|i| if i.op == MOp::SNop { i.imm as u32 + 1 } else { 1 }).sum();
    assert!(states >= 20, "only {states} wait states between the SMFMAC chain and the reader");
    // VALU-written operands: >= 2 wait states before the first SMFMAC (the C accumulator was zeroed right before)
    let before: u32 = ops[..first_smfmac].iter().rev().take_while(|i| !matches!(i.op, MOp::VWriteSub)).map(|i| if i.op == MOp::SNop { i.imm as u32 + 1 } else { 1 }).sum();
    assert!(before >= 2, "{before} wait states between the accumulator zeroing and the SMFMAC");
}

// ---- (c) identical instruction sequence vs the existing kernel path ------------------------------------------------------------------

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

/// Matrix + float-add instructions with registers renumbered by first appearance (alias views = (base, dword offset)).
fn normalise(m: &MFunction, insts: &[MInst]) -> Norm {
    let keep = |op: MOp| matches!(op, MOp::VMfmaScale32x32x64Fp8 | MOp::VMfmaScale32x32x64Fp8Z | MOp::VSmfmac16x16x128Fp8 | MOp::VAddF32);
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
    let kernel = kernel_path();
    let kinsts: Vec<MInst> = kernel.blocks.iter().flat_map(|b| b.insts.clone()).collect();
    let k = normalise(&kernel, &kinsts);

    let mut s = Seq::new();
    pv_rowsum_phase(&mut s);
    let m = s.to_mfunction("pv_safe");
    let safe = normalise(&m, s.insts());

    let show = |n: &Norm| n.iter().map(|(o, d, s)| format!("{o:?} d={d:?} s={s:?}")).collect::<Vec<_>>().join("\n");
    assert_eq!(k.len(), 8, "expected 2 MFMA + 4 SMFMAC + 1 add (+ the operand moves filtered):\n{}", show(&k));
    assert_eq!(safe, k, "safe-DSL sequence differs from the kernel path:\n--- safe\n{}\n--- kernel\n{}", show(&safe), show(&k));
}