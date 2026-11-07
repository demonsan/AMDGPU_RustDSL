//! `ldahead32` / `wsink` (sched/ldahead.rs): the 32-bit scale-word reads move up until D MFMAs sit between them and their first consumer; explicit lgkm waits sink past compute.

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
