//! AGPR (accumulation VGPR) support: classes, allocation, instruction forms, conversion pass, hazards, GV-REG checks, assembler acceptance.
//! See docs/agpr.md.

mod common;

use amdgpu_codegen::agpr::{self, AGPR_ANY};
use amdgpu_codegen::backend::LlvmTools;
use amdgpu_codegen::hazards;
use amdgpu_codegen::mir::*;
use amdgpu_codegen::regalloc::{self, Alloc};
use amdgpu_codegen::{compile};
use amdgpu_kernels::{kernel_by_name};
use common::func;

fn one_block(insts: Vec<MInst>, vregs: Vec<RClass>) -> MFunction {
    let mut b = Block::new(0);
    b.insts = insts;
    func("t", vec![b], vregs)
}

fn assemble_ok(asm: &str, tag: &str) -> Result<(), String> {
    let clang = LlvmTools::locate().tool("clang");
    if !std::process::Command::new(&clang).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
        eprintln!("SKIPPED assembler: clang not runnable");
        return Ok(());
    }
    let dir = std::env::temp_dir().join(format!("amdc-agpr-{}-{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    let s = dir.join("k.s");
    std::fs::write(&s, asm).unwrap();
    let out = std::process::Command::new(&clang)
        .args(amdgpu_codegen::target::T.llvm.asm_args)
        .args(["-c", "-o"])
        .arg(dir.join("k.o"))
        .arg(&s)
        .output()
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).lines().filter(|l| l.contains("error")).take(4).collect::<Vec<_>>().join(" | "))
    }
}

#[test]
fn unified_budget_numbers() {
    // the vendor 4-wave GEMM: 244 arch VGPRs + 212 AGPRs, accum_offset 244, vgpr_count 456, one wave per SIMD
    assert_eq!(regalloc::unified_budget(244, 212), (244, 456, 1));
    // no AGPRs: accum_offset == next_free (old behaviour), 2 waves per SIMD at 256
    assert_eq!(regalloc::unified_budget(256, 0), (256, 256, 2));
    // rounding: arch 33 -> accum 36; 36 + 3 AGPRs -> 39 -> 40
    assert_eq!(regalloc::unified_budget(33, 3), (36, 40, 8));
}

#[test]
fn probe_kernels_emit_agpr_forms_and_assemble() {
    for (k, want_dsread, want_acd) in [("probe_agpr_acc", false, true), ("probe_agpr_full", true, true), ("probe_agpr_pinned", true, true)] {
        let f = kernel_by_name(k).unwrap();
        let c = compile(&f).unwrap_or_else(|e| panic!("{k}: {e}"));
        let asm = &c.asm;
        assert!(asm.contains(".agpr_count:"), "{k}: metadata lacks .agpr_count");
        // stores may read an AGPR directly (assembler-verified), so no v_accvgpr_read is needed for the epilogue stores of these probes
        assert!(asm.lines().any(|l| l.trim_start().starts_with("global_store") && l.contains(", a")), "{k}: the epilogue stores the accumulator lanes straight from AGPRs");
        assert_eq!(asm.contains("ds_read_b128 a["), want_dsread, "{k}: ds_read into AGPR");
        let acd = asm.lines().any(|l| l.trim_start().starts_with("v_mfma") && l.contains(" a["));
        assert_eq!(acd, want_acd, "{k}: acc_cd form MFMA");
        let accum: u32 = asm.lines().find_map(|l| l.trim().strip_prefix(".amdhsa_accum_offset").and_then(|r| r.trim().parse().ok())).unwrap();
        assert_eq!(accum % 4, 0);
        assemble_ok(asm, k).unwrap_or_else(|e| panic!("{k}: assembler rejected: {e}"));
    }
    let c = compile(&kernel_by_name("probe_agpr_ref").unwrap()).unwrap();
    assert!(!c.asm.contains(".agpr_count") && !c.asm.contains(" a["), "the unpinned reference must not use AGPRs");
}

#[test]
fn pinned_probe_lands_on_its_exact_registers() {
    let c = compile(&kernel_by_name("probe_agpr_pinned").unwrap()).unwrap();
    // the accumulator is pinned at a[64:79]: an in-place MFMA names it as both D and C
    assert!(c.asm.lines().any(|l| l.contains("v_mfma") && l.contains("a[64:79], a[") && l.contains(", a[64:79],")), "{}", c.asm);
}

#[test]
fn every_new_form_assembles() {
    let asm = "
.text
.globl k
.p2align 8
.type k,@function
k:
    v_accvgpr_read_b32 v0, a1
    v_accvgpr_write_b32 a1, v0
    v_accvgpr_write_b32 a1, 0
    v_accvgpr_write_b32 a1, s3
    v_accvgpr_mov_b32 a2, a3
    ds_read_b128 a[0:3], v1 offset:16
    ds_read_b64 a[2:3], v1
    ds_read_b32 a0, v1
    ds_read_b64_tr_b8 a[2:3], v1
    ds_write_b128 v1, a[0:3]
    global_load_dwordx4 a[0:3], v[2:3], off
    global_store_dword v[2:3], a0, off
    global_store_dwordx4 v[2:3], a[0:3], off
    buffer_load_dwordx4 a[0:3], v1, s[4:7], 0 offen
    buffer_store_dwordx4 a[0:3], v1, s[4:7], 0 offen
    v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], a[16:23], a[24:31], v[0:3], v1, v2
    v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[16:23], a[24:31], a[0:3], v1, v2
    v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[16:23], v[24:31], 0, v1, v2
    v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], a[16:23], a[24:31], a[0:15], v1, v2
    v_mfma_f32_32x32x64_f8f6f4 a[0:15], v[16:23], a[24:31], a[0:15]
    v_mfma_f32_32x32x16_fp8_fp8 a[0:15], a[16:17], a[18:19], a[0:15]
    v_mfma_f32_16x16x32_fp8_fp8 v[0:3], a[16:17], a[18:19], v[0:3]
    v_mfma_f32_32x32x8_f16 a[0:15], v[16:17], v[18:19], a[0:15]
    v_mfma_f32_16x16x16_f16 a[0:3], v[16:17], v[18:19], a[0:3]
    s_endpgm
.size k, .-k
.amdhsa_kernel k
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 32
.end_amdhsa_kernel
";
    assemble_ok(asm, "forms").unwrap_or_else(|e| panic!("{e}"));
    // and the rejections the compiler relies on (assembler-measured, docs/agpr.md): odd tuple start, VALU reading an AGPR, mixed vdst/srcC files
    for bad in ["ds_read_b128 a[1:4], v1", "v_add_f32 v0, a1, v2", "v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[16:17], v[18:19], a[0:3]"] {
        let a = format!(".text\n.globl k\n.type k,@function\nk:\n    {bad}\n    s_endpgm\n.size k, .-k\n.amdhsa_kernel k\n    .amdhsa_next_free_vgpr 64\n    .amdhsa_next_free_sgpr 16\n    .amdhsa_accum_offset 32\n.end_amdhsa_kernel\n");
        if std::process::Command::new(LlvmTools::locate().tool("clang")).arg("--version").output().map(|o| o.status.success()).unwrap_or(false) {
            assert!(assemble_ok(&a, "bad").is_err(), "the assembler accepted `{bad}`: the AGPR legality rules in docs/agpr.md are out of date");
        }
    }
}

#[test]
fn allocator_pins_alignment_overlap_and_budget() {
    let a4 = |v: &mut Vec<RClass>| {
        v.push(RClass::A4);
        (v.len() - 1) as VReg
    };
    // two A4 values live together, pinned at the same place -> error; at different places -> exact
    let mut v: Vec<RClass> = Vec::new();
    let x = a4(&mut v);
    let y = a4(&mut v);
    let body = |x: VReg, y: VReg| -> Vec<MInst> {
        vec![
            MInst::new(MOp::VAccvgprWrite).def(x).src(Src::Imm(0)),
            MInst::new(MOp::VAccvgprWrite).def(y).src(Src::Imm(0)),
            MInst::new(MOp::VAccvgprMov).def(x).srcreg(y),
        ]
    };
    // (the instructions only need to give both vregs live ranges; class legality is not under test here)
    let mut m = one_block(body(x, y), v.clone());
    m.agpr_pin.insert(x, 8);
    m.agpr_pin.insert(y, 8);
    let e = regalloc::allocate(&m).err().expect("overlapping pins must fail");
    assert!(e.contains("overlaps"), "{e}");
    let mut m = one_block(body(x, y), v.clone());
    m.agpr_pin.insert(x, 8);
    m.agpr_pin.insert(y, 12);
    let a = regalloc::allocate(&m).unwrap();
    assert_eq!(a.phys[x as usize], AGPR_TAG + 8);
    assert_eq!(a.phys[y as usize], AGPR_TAG + 12);
    assert_eq!(regalloc::agpr_count(&m, &a), 16);
    // misaligned pin
    let mut m = one_block(body(x, y), v.clone());
    m.agpr_pin.insert(x, 6);
    assert!(regalloc::allocate(&m).err().unwrap().contains("not aligned"));
    // budget: 70 simultaneously live A4 values = 280 AGPRs > 256
    let mut v: Vec<RClass> = Vec::new();
    let regs: Vec<VReg> = (0..70).map(|_| a4(&mut v)).collect();
    let mut insts: Vec<MInst> = regs.iter().map(|r| MInst::new(MOp::VAccvgprWrite).def(*r).src(Src::Imm(0))).collect();
    insts.extend(regs.iter().map(|r| MInst::new(MOp::VAccvgprMov).def(regs[0]).srcreg(*r)));
    let e = regalloc::allocate(&one_block(insts, v)).err().expect("280 AGPRs cannot fit");
    assert!(e.contains("out of AGPRs"), "{e}");
}

#[test]
fn verify_rejects_valu_on_agpr_and_mixed_mfma_files() {
    // VALU reading an A1
    let m = one_block(
        vec![MInst::new(MOp::VAccvgprWrite).def(0).src(Src::Imm(1)), MInst::new(MOp::VAddF32).def(1).srcreg(0).srcreg(0)],
        vec![RClass::A1, RClass::V1],
    );
    let errs = verify(&m);
    assert!(errs.iter().any(|e| e.contains("cannot read one")), "{errs:?}");
    // MFMA with vdst in V and srcC in A
    let m = one_block(
        vec![MInst::new(MOp::VMfmaScale16x16x128Fp8).def(0).srcreg(1).srcreg(1 + 1).srcreg(3).srcreg(4).srcreg(4)],
        vec![RClass::V4, RClass::V8, RClass::V8, RClass::A4, RClass::V1],
    );
    let errs = verify(&m);
    assert!(errs.iter().any(|e| e.contains("different register files")), "{errs:?}");
    // v_accvgpr_read must read an A1 into a V1
    let m = one_block(vec![MInst::new(MOp::VAccvgprRead).def(0).srcreg(1)], vec![RClass::V1, RClass::V1]);
    assert!(verify(&m).iter().any(|e| e.contains("v_accvgpr_read_b32")));
}

#[test]
fn assign_converts_movs_lane_reads_and_vector_consumers() {
    // v0 = 1.0f (V1 const), t (V4) built lane by lane, y = t[1] + v0 (VALU), store y; request t in AGPRs
    let mut m = one_block(
        vec![
            MInst::new(MOp::VMovB32).def(0).imm(0x3f800000),
            MInst::new(MOp::VWriteSub).def(1).srcreg(0).imm(0),
            MInst::new(MOp::VWriteSub).def(1).srcreg(0).imm(1),
            MInst::new(MOp::VWriteSub).def(1).srcreg(0).imm(2),
            MInst::new(MOp::VWriteSub).def(1).srcreg(0).imm(3),
            MInst::new(MOp::VAddF32).def(3).srcreg(2).srcreg(0),
            MInst::new(MOp::GlobalStoreF32).srcreg(4).srcreg(3),
        ],
        vec![RClass::V1, RClass::V4, RClass::V1, RClass::V1, RClass::V2],
    );
    m.alias.insert(2, (1, 1)); // r2 = lane 1 of r1
    m.agpr_pin.insert(1, AGPR_ANY);
    let st = agpr::assign(&mut m).unwrap();
    assert_eq!(m.vreg_class(1), RClass::A4);
    assert_eq!(m.vreg_class(2), RClass::A1, "alias views follow their tuple");
    assert!(m.agpr_pin.is_empty(), "'any' requests are consumed");
    assert_eq!(st.writes, 4, "four lane writes become v_accvgpr_write_b32");
    assert_eq!(st.reads, 1, "the VALU consumer gets one v_accvgpr_read_b32");
    assert!(verify(&m).is_empty(), "{:?}", verify(&m));
    let ops: Vec<MOp> = m.blocks[0].insts.iter().map(|i| i.op).collect();
    assert!(ops.iter().filter(|o| **o == MOp::VAccvgprWrite).count() == 4 && ops.contains(&MOp::VAccvgprRead));
}

fn mfma16(dst: VReg, c: VReg) -> MInst {
    // regs: 1 = A frag (V8), 2 = B frag (V8), 3/4 = scales (V1)
    MInst::new(MOp::VMfmaScale16x16x128Fp8).def(dst).srcreg(1).srcreg(2).srcreg(c).srcreg(3).srcreg(4)
}

fn nops(m: &MFunction) -> usize {
    m.blocks[0].insts.iter().filter(|i| i.op == MOp::SNop).map(|i| i.imm as usize + 1).sum()
}

#[test]
fn hazards_mfma_result_then_accvgpr_read_and_in_place_chain() {
    // classes: 0 = acc (A4, in place), 1/2 = A/B frags (A8), 3/4 scales (V1), 5 = V1 result of the read, 6 = alias lane 0 of 0
    let vregs = vec![RClass::A4, RClass::A8, RClass::A8, RClass::V1, RClass::V1, RClass::V1, RClass::A1];
    let mk = |insts: Vec<MInst>| {
        let mut m = one_block(insts, vregs.clone());
        m.alias.insert(6, (0, 0));
        m
    };
    // in-place same-opcode chain: no padding (hardware forwards the exact-C path)
    let mut chain = mk(vec![mfma16(0, 0), mfma16(0, 0), mfma16(0, 0)]);
    hazards::fix_value_hazards(&mut chain);
    assert_eq!(nops(&chain), 0, "in-place accumulation chain (acc_cd form) needs no wait states");
    // MFMA then immediately v_accvgpr_read of its result: padded like a VALU read of a VGPR result (20 wait states, 21 slots incl. the read)
    let mut rd = mk(vec![mfma16(0, 0), MInst::new(MOp::VAccvgprRead).def(5).srcreg(6)]);
    hazards::fix_value_hazards(&mut rd);
    assert!(nops(&rd) >= 20, "accvgpr_read right after an MFMA needs the result latency, got {} nop slots", nops(&rd));
    // enough independent VALU work in between: no padding
    let mut fill: Vec<MInst> = vec![mfma16(0, 0)];
    for _ in 0..21 {
        fill.push(MInst::new(MOp::VMovB32).def(5).imm(0));
    }
    fill.push(MInst::new(MOp::VAccvgprRead).def(3).srcreg(6));
    let mut f = mk(fill);
    hazards::fix_value_hazards(&mut f);
    assert_eq!(nops(&f), 0);
}

#[test]
fn hazards_accvgpr_write_then_mfma_read() {
    // v_accvgpr_write of an MFMA operand / accumulator, then the MFMA: VALU write -> MFMA read = 2 wait states
    let vregs = vec![RClass::A4, RClass::A8, RClass::A8, RClass::V1, RClass::V1, RClass::A1];
    let mut m = one_block(
        vec![MInst::new(MOp::VAccvgprWrite).def(5).src(Src::Imm(0)), mfma16(0, 0)],
        vregs,
    );
    m.alias.insert(5, (0, 2)); // the written dword is a lane of the accumulator
    hazards::fix_value_hazards(&mut m);
    assert!(nops(&m) >= 2, "v_accvgpr_write -> MFMA reading it needs 2 wait states, got {}", nops(&m));
}

#[test]
fn physical_war_covers_agpr_srcc() {
    // MFMA m reads srcC a[0:3] (separate dst a[4:7]); a v_accvgpr_write overwriting a[1] must wait for the running MFMA's source read
    let vregs = vec![RClass::A4, RClass::A4, RClass::A8, RClass::A8, RClass::V1, RClass::V1, RClass::A1];
    let mut m = one_block(
        vec![mfma16_sep(), MInst::new(MOp::VAccvgprWrite).def(6).src(Src::Imm(0))],
        vregs,
    );
    m.alias.insert(6, (0, 1));
    let a = Alloc { phys: vec![AGPR_TAG, AGPR_TAG + 4, AGPR_TAG + 8, AGPR_TAG + 16, 8, 9, AGPR_TAG + 1], max_v: 16, max_s: 16, scalar: vec![false; 7] };
    hazards::pad_physical_war(&mut m, &a);
    let n: usize = m.blocks[0].insts.iter().filter(|i| i.op == MOp::SNop).map(|i| i.imm as usize + 1).sum();
    assert!(n >= 8, "a write to an AGPR still read as srcC by a running MFMA must be padded, got {n}");
}

fn mfma16_sep() -> MInst {
    // dst = r1 (A4), A = r2, B = r3, C = r0, scales r4/r5
    MInst::new(MOp::VMfmaScale16x16x128Fp8).def(1).srcreg(2).srcreg(3).srcreg(0).srcreg(4).srcreg(5)
}

#[test]
fn hazards_accvgpr_read_then_pointer_add() {
    // LLVM oracle (hazard matrix, AGPR rows): v_accvgpr_read result consumed by v_add_co/v_addc pointer arithmetic needs 2 wait states
    let mut m = one_block(
        vec![MInst::new(MOp::VAccvgprRead).def(1).srcreg(0), MInst::new(MOp::VPtrAdd).def(3).srcreg(2).srcreg(1)],
        vec![RClass::A1, RClass::V1, RClass::V2, RClass::V2],
    );
    hazards::fix_value_hazards(&mut m);
    assert!(nops(&m) >= 2, "got {}", nops(&m));
}