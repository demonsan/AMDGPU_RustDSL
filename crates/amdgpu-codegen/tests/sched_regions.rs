//! Weave regions (sched/region.rs): per-region scheduler strategies. Without regions nothing changes; with them each region's options apply only to its own instructions and no pass
//! crosses a region boundary.

mod common;

use amdgpu_codegen::mir::*;
use amdgpu_codegen::sched::region::{runs, RegionSpec};
use amdgpu_codegen::sched::{self, Options};
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
fn in_region(mut i: MInst, r: u16) -> MInst {
    i.region = r;
    i
}
fn name(i: &MInst) -> String {
    match i.op {
        MOp::VMfmaScale16x16x128Fp8 => format!("M{}", i.dst.unwrap()),
        MOp::LdsReadB128 => format!("R{}", i.dst.unwrap() - 20),
        MOp::VMovB32 => format!("u{}", i.dst.unwrap() - 28),
        o => format!("{o:?}"),
    }
}
fn spec(name: &str, authored: bool, tokens: &str) -> RegionSpec {
    RegionSpec { name: name.into(), authored, tokens: tokens.into() }
}
/// run the whole scheduling pass; the op sequence without the waits it inserted
fn run(insts: Vec<MInst>, o: &Options) -> Vec<String> {
    let mut m = mk(insts);
    sched::run(&mut m, o).expect("scheduled");
    m.blocks[0].insts.iter().filter(|i| !matches!(i.op, MOp::SWaitLgkm | MOp::SWaitVm)).map(name).collect()
}
fn seq(s: &str) -> Vec<String> {
    s.split(' ').map(|x| x.to_string()).collect()
}

#[test]
fn runs_partition_a_block_by_region() {
    let v = vec![read(0), in_region(mfma(0), 1), in_region(mfma(1), 1), in_region(mfma(2), 2), read(1)];
    let mut o = Options::default();
    o.regions = vec![spec("a", true, ""), spec("b", false, "")];
    assert_eq!(runs(&o, &v), vec![(0, 1, 0), (1, 3, 1), (3, 4, 2), (4, 5, 0)]);
    // lowered region ids (one per scope instance) map to strategies by name: ids 1 and 3 are two instances of declaration 1, id 2 is undeclared (inherits the default)
    o.region_of = vec![0, 1, 0, 1];
    let w = vec![in_region(mfma(0), 1), in_region(mfma(1), 3), in_region(mfma(2), 2), in_region(mfma(3), 1)];
    assert_eq!(runs(&o, &w), vec![(0, 2, 1), (2, 3, 0), (3, 4, 1)]);
}

#[test]
fn an_unused_region_table_leaves_the_schedule_unchanged() {
    // all instructions in region 0: the region table must not change anything
    let prog = || vec![read(0), user(0), mfma(0), mfma(1), mfma(2), mfma(3), read(1), user(1), mfma(4), mfma(5), mfma(6), mfma(7)];
    let plain = Options::default();
    let mut with_table = Options::default();
    with_table.regions = vec![spec("a", true, ""), spec("b", false, "ldsink=1")];
    assert_eq!(run(prog(), &plain), run(prog(), &with_table));
}

#[test]
fn an_authored_region_keeps_its_order_while_a_free_region_is_list_scheduled() {
    // the same pattern in both halves: a read, its consumer right behind it, then independent MFMAs - the list scheduler hoists the consumer away from the read
    let half = |rd: u32, k0: u32, r: u16| vec![in_region(read(rd), r), in_region(user(rd), r), in_region(mfma(k0), r), in_region(mfma(k0 + 1), r), in_region(mfma(k0 + 2), r), in_region(mfma(k0 + 3), r)];
    let mut prog = half(0, 0, 1);
    prog.extend(half(1, 4, 2));
    let mut o = Options::default(); // kernel default: full list scheduling
    o.regions = vec![spec("authored", true, ""), spec("free", false, "")];
    let out = run(prog, &o);
    assert_eq!(&out[..6], &seq("R0 u0 M0 M1 M2 M3")[..], "the authored region keeps the emitted order");
    assert_ne!(&out[6..], &seq("R1 u1 M4 M5 M6 M7")[..], "the free region is list scheduled");
    // and the kernel default (region 0) still schedules: the same program without regions reorders the first half too
    let mut plain = half(0, 0, 0);
    plain.extend(half(1, 4, 0));
    let out0 = run(plain, &Options::default());
    assert_ne!(&out0[..6], &seq("R0 u0 M0 M1 M2 M3")[..]);
}

#[test]
fn a_token_applies_only_inside_its_region() {
    // ldsink=1 in region 1 only; the kernel default only recomputes waits
    let half = |rd: u32, k0: u32, r: u16| vec![in_region(read(rd), r), in_region(mfma(k0), r), in_region(mfma(k0 + 1), r), in_region(mfma(k0 + 2), r), in_region(mfma(k0 + 3), r), in_region(user(rd), r)];
    let mut prog = half(0, 0, 1);
    prog.extend(half(1, 4, 2));
    let mut o = Options::default();
    o.reorder = false;
    o.regions = vec![spec("sink", true, "ldsink=1"), spec("plain", true, "")];
    let out = run(prog, &o);
    assert_eq!(&out[..6], &seq("M0 M1 M2 R0 M3 u0")[..], "ldsink=1: the read sinks to one MFMA before its consumer");
    assert_eq!(&out[6..], &seq("R1 M4 M5 M6 M7 u1")[..], "the other region is untouched");
}

#[test]
fn no_pass_crosses_a_region_boundary() {
    // the read ends region 1, its consumer sits in region 2 behind four MFMAs: without the boundary ldsink would sink it next to the consumer
    let mut prog: Vec<MInst> = (0..4).map(|k| in_region(mfma(k), 1)).collect();
    prog.push(in_region(read(0), 1));
    prog.extend((4..8).map(|k| in_region(mfma(k), 2)));
    prog.push(in_region(user(0), 2));
    let mut o = Options::default();
    o.reorder = false;
    o.regions = vec![spec("a", true, "ldsink=1"), spec("b", true, "ldsink=1")];
    let out = run(prog.clone(), &o);
    assert_eq!(out, seq("M0 M1 M2 M3 R0 M4 M5 M6 M7 u0"), "the read stays at the end of its region");
    // the same stream as ONE region does sink the read
    let one: Vec<MInst> = prog.into_iter().map(|i| in_region(i, 1)).collect();
    let mut o1 = o.clone();
    o1.regions = vec![spec("a", true, "ldsink=1")];
    let out1 = run(one, &o1);
    assert_eq!(out1, seq("M0 M1 M2 M3 M4 M5 M6 R0 M7 u0"));
}

#[test]
fn an_authored_region_cannot_be_unlocked_by_its_tokens() {
    let mut o = Options::default();
    o.regions = vec![spec("a", true, "reorder,slotfill,ldshoist,ldsink=2"), spec("f", false, "rewait")];
    let a = o.for_region(1);
    assert!(!a.reorder && !a.slotfill && !a.ldshoist, "authored: no list scheduling / slotfill / ldshoist");
    assert_eq!(a.ldsink, 2, "the order-preserving post-passes stay available");
    let f = o.for_region(2);
    assert!(!f.reorder, "a free region follows its tokens");
    assert!(o.for_region(0).reorder && o.for_region(9).reorder, "region 0 / an unknown id use the kernel default");
}