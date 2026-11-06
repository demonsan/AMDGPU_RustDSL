//! `ldahead` (sched/ldahead.rs): ds_reads move up until D MFMAs sit between them and their first consumer; dependences and the register budget stop them.

mod common;

use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::{ldahead, Options};
use common::func;

// registers: 0..16 accumulators (V4), 16 / 17 fragments (V8), 18 scale (V1), 19 lds address (V1), 20.. ds_read destinations (V4) x 8, 28.. V1 scratch
fn mk(insts: Vec<MInst>) -> MFunction {
    let mut v = vec![RClass::V4; 16];
    v.push(RClass::V8);
    v.push(RClass::V8);
    v.push(RClass::V1);
    v.push(RClass::V1);
    for _ in 0..8 {
        v.push(RClass::V4);
    }
    for _ in 0..8 {
        v.push(RClass::V1);
    }
    let mut b = Block::new(0);
    b.insts = insts;
    func("t", vec![b], v)
}
fn mfma(k: u32) -> MInst {
    MInst::new(MOp::VMfmaScale16x16x128Fp8).def(k % 16).srcreg(16).srcreg(17).srcreg(k % 16).srcreg(18).srcreg(18)
}
fn read(d: u32) -> MInst {
    MInst::new(MOp::LdsReadB128).def(20 + d).srcreg(19).imm((d * 16) as i64)
}
fn user(d: u32) -> MInst {
    MInst::new(MOp::VMovB32).def(28 + d).srcreg(20 + d)
}
fn name(i: &MInst) -> String {
    match i.op {
        MOp::VMfmaScale16x16x128Fp8 => format!("M{}", i.dst.unwrap()),
        MOp::LdsReadB128 => format!("R{}", i.dst.unwrap() - 20),
        MOp::VMovB32 => format!("u{}", i.dst.unwrap() - 28),
        MOp::VAddU32 => "a".into(),
        o => format!("{o:?}"),
    }
}
fn run(insts: Vec<MInst>, d: usize) -> String {
    let mut m = mk(insts);
    let o = Options { verify: true, ldahead: d, ..Options::default() };
    ldahead::run(&mut m, &o);
    m.blocks[0].insts.iter().map(name).collect::<Vec<_>>().join(" ")
}

#[test]
fn a_read_moves_up_until_d_mfmas_sit_between_it_and_its_consumer() {
    let src = vec![mfma(0), mfma(1), mfma(2), mfma(3), mfma(4), mfma(5), read(0), user(0)];
    assert_eq!(run(src.clone(), 3), "M0 M1 M2 R0 M3 M4 M5 u0");
    assert_eq!(run(src.clone(), 1), "M0 M1 M2 M3 M4 R0 M5 u0");
    assert_eq!(run(src, 0), "M0 M1 M2 M3 M4 M5 R0 u0", "off");
}

#[test]
fn the_address_producer_stops_the_read() {
    let add = MInst::new(MOp::VAddU32).def(19).srcreg(19).srcreg(19);
    let src = vec![mfma(0), mfma(1), mfma(2), add, read(0), mfma(3), user(0)];
    // R0 is already 1 MFMA from its user; it cannot pass the address update
    assert_eq!(run(src, 3), "M0 M1 M2 a R0 M3 u0");
}

#[test]
fn reads_keep_their_relative_order_and_each_reaches_its_distance() {
    let src = vec![mfma(0), mfma(1), mfma(2), mfma(3), mfma(4), read(0), read(1), mfma(5), user(0), user(1)];
    let out = run(src, 2);
    let p = |s: &str| out.split(' ').position(|x| x == s).unwrap();
    assert!(p("R0") < p("R1"), "{out}");
    // between R0 / R1 and their consumers at least two MFMAs
    let m_between = |a: &str, b: &str| out.split(' ').enumerate().filter(|(i, x)| *i > p(a) && *i < p(b) && x.starts_with('M')).count();
    assert!(m_between("R0", "u0") >= 2 && m_between("R1", "u1") >= 2, "{out}");
}

#[test]
fn ldahead32_moves_only_the_32_bit_scale_reads() {
    let b32 = |d: u32| MInst::new(MOp::LdsReadB32).def(28 + d).srcreg(19).imm((d * 4) as i64);
    let user32 = |d: u32| MInst::new(MOp::VMovB32).def(32 + d).srcreg(28 + d);
    let src = vec![mfma(0), mfma(1), mfma(2), mfma(3), b32(0), user32(0), read(0), user(0)];
    let mut m = mk(src);
    let o = Options { verify: true, ldahead32: 2, ..Options::default() };
    ldahead::run(&mut m, &o);
    let names: Vec<String> = m.blocks[0].insts.iter().map(|i| match i.op {
        MOp::LdsReadB32 => "B".into(),
        MOp::LdsReadB128 => "R".into(),
        MOp::VMfmaScale16x16x128Fp8 => "M".into(),
        _ => "u".into(),
    }).collect();
    // the b32 read hops up over two MFMAs, the b128 read (distance 0 requested) stays
    assert_eq!(names.join(" "), "M M B M M u R u");
}

#[test]
fn wsink_moves_an_explicit_wait_past_compute_but_not_past_memory_or_barriers() {
    use amdgpu_compiler::sched::ldahead::sink_explicit_waits;
    let wait = MInst::new(MOp::SWaitLgkm).imm(0);
    let src = vec![mfma(0), wait, mfma(1), mfma(2), MInst::new(MOp::SBarrier)];
    let mut m = mk(src);
    let o = Options { verify: true, ..Options::default() };
    let n = sink_explicit_waits(&mut m, &o);
    let names: Vec<String> = m.blocks[0].insts.iter().map(|i| match i.op {
        MOp::SWaitLgkm => "W".into(),
        MOp::SBarrier => "|".into(),
        _ => "M".into(),
    }).collect();
    assert!(n > 0, "the wait must move");
    assert_eq!(names.join(" "), "M M M W |");
}

#[test]
fn ovl_spreads_independent_valu_evenly_over_the_mfmas_of_a_stretch() {
    use amdgpu_compiler::sched::ldahead::spread_valu;
    let mv = |d: u32| MInst::new(MOp::VMovB32).def(28 + d).imm(d as i64);
    let src = vec![mfma(0), mfma(1), mfma(2), mfma(3), mv(0), mv(1), mv(2), mv(3), MInst::new(MOp::SBarrier)];
    let mut m = mk(src);
    let o = Options { verify: true, ovl: 1, ..Options::default() };
    let moved = spread_valu(&mut m, &o);
    let names: Vec<String> = m.blocks[0].insts.iter().map(|i| match i.op {
        MOp::VMfmaScale16x16x128Fp8 => "M".into(),
        MOp::VMovB32 => "v".into(),
        _ => "|".into(),
    }).collect();
    assert!(moved > 0);
    assert_eq!(names.join(" "), "M v M v M v M v |");
}

#[test]
fn ovl_leaves_a_valu_that_depends_on_an_mfma_behind_it() {
    use amdgpu_compiler::sched::ldahead::spread_valu;
    // v consumes MFMA 3's accumulator (reg 3): it cannot hop above that MFMA
    let dep = MInst::new(MOp::VMovB32).def(28).srcreg(3);
    let src = vec![mfma(0), mfma(1), mfma(2), mfma(3), dep, MInst::new(MOp::VMovB32).def(29).imm(1), MInst::new(MOp::SBarrier)];
    let mut m = mk(src);
    let o = Options { verify: true, ovl: 1, ..Options::default() };
    spread_valu(&mut m, &o);
    let p_mfma3 = m.blocks[0].insts.iter().position(|i| i.op == MOp::VMfmaScale16x16x128Fp8 && i.dst == Some(3)).unwrap();
    let p_dep = m.blocks[0].insts.iter().position(|i| i.op == MOp::VMovB32 && i.dst == Some(28)).unwrap();
    assert!(p_dep > p_mfma3);
}
