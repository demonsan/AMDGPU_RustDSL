//! `sched_group_barrier`-style phase scheduling (sched/phase.rs): order, dependence pulling, s_setprio anchoring, marker removal.

mod common;

use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::phase;
use amdgpu_compiler::sched::Options;
use common::func;

fn marker(mask: u32, size: u32) -> MInst {
    MInst::new(MOp::SchedGroup).imm((mask as i64) | ((size as i64) << 16))
}

// registers: 0 = lds address (V1), 1..=4 = A frags (V8, loaded by two ds_read_b128 each is overkill: use V4 halves of a V8 tuple), see below
// layout used by every test: per MFMA k (0..4): frag A_k = V8 reg (1 + 2k), frag B_k = V8 reg (2 + 2k), acc_k = V4 reg (9 + k), scales V1 reg 13
fn mk_fn(insts: Vec<MInst>) -> MFunction {
    let mut v = vec![RClass::V1];
    for _ in 0..4 {
        v.push(RClass::V8);
        v.push(RClass::V8);
    }
    for _ in 0..4 {
        v.push(RClass::V4);
    }
    v.push(RClass::V1);
    // alias halves of A_k / B_k for the ds_reads: regs 14.. (V4 views)
    for _ in 0..16 {
        v.push(RClass::V4);
    }
    let mut b = Block::new(0);
    b.insts = insts;
    let mut m = func("t", vec![b], v);
    for k in 0..8u32 {
        let t = 1 + k;
        m.alias.insert(14 + 2 * k, (t, 0));
        m.alias.insert(15 + 2 * k, (t, 4));
    }
    m
}

fn mfma(k: u32) -> MInst {
    MInst::new(MOp::VMfmaScale16x16x128Fp8).def(9 + k).srcreg(1 + 2 * k).srcreg(2 + 2 * k).srcreg(9 + k).srcreg(13).srcreg(13)
}

fn ds_read(k: u32, half: u32) -> MInst {
    // loads half `half` of frag A_k (frag index 0..8 = A_0,B_0,A_1,... flattened as reg 1 + frag)
    MInst::new(MOp::LdsReadB128).def(14 + 2 * k + half).srcreg(0).imm((k * 32 + half * 16) as i64)
}

fn names(insts: &[MInst]) -> String {
    insts
        .iter()
        .map(|i| match i.op {
            MOp::VMfmaScale16x16x128Fp8 => format!("M{}", i.dst.unwrap() - 9),
            MOp::LdsReadB128 => format!("R{}", (i.dst.unwrap() - 14) / 2),
            MOp::SSetPrio => format!("P{}", i.imm),
            MOp::SchedGroup => "G".to_string(),
            other => format!("{other:?}"),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn run(insts: Vec<MInst>) -> (String, MFunction) {
    let m = mk_fn(insts.clone());
    let term = Term::Endpgm;
    let o = Options::default();
    let out = phase::schedule_block(&m, &insts, &term, &o).expect("verified").map(|r| r.0).unwrap_or_else(|| insts.iter().filter(|i| i.op != MOp::SchedGroup).cloned().collect());
    (names(&out), m)
}

#[test]
fn interleaves_independent_mfmas_and_reads_in_the_requested_order() {
    // source: R0..R3 (frags 0..3 of the A side) then M0..M3 (MFMA k reads frag regs 1+2k, 2+2k = ds_read-written tuples only for k < 2 here)
    // -> use reads that feed nothing the MFMAs read: reads write A_2/B_2.. views (k = 2, 3), MFMA 0/1 are independent of them
    let mut insts = vec![ds_read(2, 0), ds_read(2, 1), ds_read(3, 0), ds_read(3, 1), mfma(0), mfma(1)];
    // pattern: MFMA, DS read, MFMA, DS read x3
    insts.push(marker(phase::MFMA, 1));
    insts.push(marker(phase::DS_READ, 1));
    insts.push(marker(phase::MFMA, 1));
    insts.push(marker(phase::DS_READ, 3));
    let (got, _) = run(insts);
    assert_eq!(got, "M0 R1 M1 R1 R2 R2", "{got}");
}

#[test]
fn a_selected_mfma_pulls_the_reads_it_consumes() {
    // M0 consumes A_0/B_0 views loaded by R0 (halves 0,1 of frag 0: regs 14,15 = alias of tuple 1) - ds_read(0,*) writes tuple 1 = MFMA 0's A operand
    let mut insts = vec![ds_read(0, 0), ds_read(0, 1), ds_read(2, 0), ds_read(2, 1), mfma(0)];
    insts.push(marker(phase::MFMA, 1));
    let (got, _) = run(insts);
    // M0 needs R0 (both halves) only; the unrelated reads follow in program order
    assert_eq!(got, "R0 R0 M0 R1 R1", "{got}");
}

#[test]
fn s_setprio_stays_anchored_to_the_mfma_stream() {
    let mut insts = vec![ds_read(2, 0), mfma(0), MInst::new(MOp::SSetPrio).imm(1), mfma(1), MInst::new(MOp::SSetPrio).imm(0)];
    insts.push(marker(phase::DS_READ, 1));
    insts.push(marker(phase::MFMA, 2));
    let (got, _) = run(insts);
    assert_eq!(got, "R1 M0 P1 M1 P0", "{got}");
}

#[test]
fn markers_are_removed_and_unmarked_blocks_are_untouched() {
    let mut m = mk_fn(vec![ds_read(2, 0), mfma(0), marker(phase::MFMA, 1)]);
    assert!(phase::has_markers(&m.blocks[0].insts));
    phase::strip(&mut m);
    assert!(!phase::has_markers(&m.blocks[0].insts));
    assert_eq!(m.blocks[0].insts.len(), 2);
    // run() on a function without markers changes nothing
    let mut m2 = mk_fn(vec![ds_read(2, 0), mfma(0)]);
    let before = names(&m2.blocks[0].insts);
    phase::run(&mut m2, &Options::default()).unwrap();
    assert_eq!(names(&m2.blocks[0].insts), before);
}