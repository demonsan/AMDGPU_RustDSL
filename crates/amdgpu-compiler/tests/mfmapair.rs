//! `mfmapair` (sched/mfmapair.rs): dependent same-accumulator MFMA pairs adjacent, non-MFMA work only after the second MFMA of a pair.

mod common;

use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::{mfmapair, Options};
use common::func;

// registers: 0..16 accumulators (V4), 16 A (V8), 17 B (V8), 18 / 20 scales (V1), 44.. V1 scratch
fn mk(insts: Vec<MInst>) -> MFunction {
    let mut v = vec![RClass::V4; 16];
    v.push(RClass::V8);
    v.push(RClass::V8);
    v.push(RClass::V1);
    v.push(RClass::V1);
    v.push(RClass::V1);
    v.push(RClass::V1); // 20
    for _ in 0..30 {
        v.push(RClass::V1);
    }
    let mut b = Block::new(0);
    b.insts = insts;
    func("t", vec![b], v)
}

fn mfma(acc: u32, scale: u32) -> MInst {
    MInst::new(MOp::VMfmaScale16x16x128Fp8).def(acc).srcreg(16).srcreg(17).srcreg(acc).srcreg(scale).srcreg(scale)
}

fn name(i: &MInst) -> String {
    match i.op {
        MOp::VMfmaScale16x16x128Fp8 => format!("M{}", i.dst.unwrap()),
        MOp::VMovB32 => format!("mov{}", i.dst.unwrap()),
        MOp::VAddF32 => format!("add{}", i.dst.unwrap()),
        o => format!("{o:?}"),
    }
}

fn run_with(insts: Vec<MInst>, budget: f64) -> String {
    let mut m = mk(insts);
    let o = Options { verify: true, mfmapair: true, mfmapair_budget: budget, ..Options::default() };
    mfmapair::run(&mut m, &o).expect("verified");
    m.blocks[0].insts.iter().map(name).collect::<Vec<_>>().join(" ")
}
fn run(insts: Vec<MInst>) -> String {
    run_with(insts, 1.0)
}

#[test]
fn interleaved_accumulators_become_adjacent_pairs() {
    // M0 M1 M0 M1 (two accumulators, each accumulated twice) -> M0 M0 M1 M1
    assert_eq!(run(vec![mfma(0, 18), mfma(1, 18), mfma(0, 18), mfma(1, 18)]), "M0 M0 M1 M1");
}

#[test]
fn a_valu_between_the_pair_moves_behind_the_second_mfma() {
    assert_eq!(run(vec![mfma(0, 18), MInst::new(MOp::VMovB32).def(44).imm(1), mfma(0, 18)]), "M0 M0 mov44");
}

#[test]
fn what_the_second_mfma_needs_is_hoisted_in_front_of_the_pair() {
    // the second MFMA reads scale reg 20 produced between the two
    assert_eq!(run(vec![mfma(0, 18), MInst::new(MOp::VMovB32).def(20).imm(7), mfma(0, 20)]), "mov20 M0 M0");
}

#[test]
fn a_producer_that_consumes_the_first_result_blocks_the_pair() {
    // reg 20 is computed from the first MFMA's result and read by the second one: they cannot be adjacent
    let src = vec![mfma(0, 18), MInst::new(MOp::VMovB32).def(20).srcreg(0), mfma(0, 20)];
    assert_eq!(run(src), "M0 mov20 M0");
}

#[test]
fn more_non_mfma_work_than_the_shadow_hides_keeps_the_original_order() {
    let mut src = vec![mfma(0, 18)];
    for k in 0..24 {
        src.push(MInst::new(MOp::VMovB32).def(30 + k % 10).imm(k as i64));
    }
    src.push(mfma(0, 18));
    src.push(mfma(1, 18));
    let n = src.len();
    let out = run(src.clone());
    assert!(out.starts_with("M0 mov"), "the pair stays split: {out}");
    assert_eq!(out.split(' ').count(), n);
    // a larger budget lets the 24 instructions move behind the pair
    let big = run_with(src, 100.0);
    assert!(big.starts_with("M0 M0 mov"), "{big}");
}

#[test]
fn already_adjacent_pairs_and_blocks_without_pairs_are_untouched() {
    assert_eq!(run(vec![mfma(0, 18), mfma(0, 18), mfma(1, 18)]), "M0 M0 M1");
    assert_eq!(run(vec![mfma(0, 18), MInst::new(MOp::VMovB32).def(44).imm(1), mfma(1, 18)]), "M0 mov44 M1");
}

#[test]
fn a_run_of_three_same_accumulator_mfmas_becomes_one_adjacent_chain() {
    assert_eq!(run(vec![mfma(0, 18), mfma(1, 18), mfma(0, 18), mfma(1, 18), mfma(0, 18)]), "M0 M0 M0 M1 M1");
}