//! Regression (MLA decode, kernel-opt): loads in flight on a path where their result is dead; the allocator hands their registers to
//! another value, whose write the late load clobbers. `pad_inflight_waw` must wait for the load before such a write - also across blocks.
use amdgpu_codegen::hazards::pad_inflight_waw;
use amdgpu_codegen::mir::{Block, MFunction, MInst, MOp, RClass, Src, Term};
use amdgpu_codegen::regalloc::Alloc;

fn func(blocks: Vec<Vec<MInst>>, classes: Vec<RClass>) -> MFunction {
    let mut bs = Vec::new();
    for (i, insts) in blocks.into_iter().enumerate() {
        let mut b = Block::new(i as u32);
        b.insts = insts;
        bs.push(b);
    }
    MFunction {
        name: "waw".into(),
        args: Vec::new(),
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: bs,
        vregs: classes,
        loops: Vec::new(),
        alias: Default::default(),
        elide_fallthrough: false, edge_pad: false,
        agpr_pin: Default::default(), reg_hints: Default::default(),
    }
}


// vregs: 0 = S4 (descriptor), 1 = V1 (voffset), 2 = V4 (the load, dead on this path), 3 = V1 (temp: reuses the load's registers), 4 = V1 (unrelated)
fn classes() -> Vec<RClass> {
    vec![RClass::S4, RClass::V1, RClass::V4, RClass::V1, RClass::V1]
}

fn alloc(phys: Vec<u32>) -> Alloc {
    let n = phys.len();
    let mut scalar = vec![false; n];
    scalar[0] = true;
    Alloc { phys, max_v: 256, max_s: 16, scalar }
}

fn load() -> MInst {
    MInst::new(MOp::BufferLoadDwordX4).def(2).srcreg(0).srcreg(1).src(Src::Imm(0))
}

fn waits_before(m: &MFunction, bi: usize, op: MOp) -> Vec<(MOp, i64)> {
    let ins = &m.blocks[bi].insts;
    let k = ins.iter().position(|i| i.op == op).unwrap();
    ins[..k].iter().filter(|i| matches!(i.op, MOp::SWaitVm | MOp::SWaitLgkm)).map(|i| (i.op, i.imm)).collect()
}

#[test]
fn a_write_into_an_in_flight_loads_registers_in_a_successor_block_waits() {
    let mut m = func(vec![vec![load()], vec![MInst::new(MOp::VMovB32).def(3).src(Src::Imm(7))]], classes());
    m.blocks[0].term = Term::Goto(1);
    let n = pad_inflight_waw(&mut m, &alloc(vec![8, 1, 20, 21, 40]));
    assert_eq!(n, 1);
    assert_eq!(waits_before(&m, 1, MOp::VMovB32), vec![(MOp::SWaitVm, 0)]);
}

#[test]
fn unrelated_registers_and_already_waited_loads_are_left_alone() {
    // disjoint register
    let mut m = func(vec![vec![load()], vec![MInst::new(MOp::VMovB32).def(4).src(Src::Imm(7))]], classes());
    m.blocks[0].term = Term::Goto(1);
    assert_eq!(pad_inflight_waw(&mut m, &alloc(vec![8, 1, 20, 21, 40])), 0);
    // a wait already retired the load
    let mut m = func(vec![vec![load(), MInst::new(MOp::SWaitVm).imm(0)], vec![MInst::new(MOp::VMovB32).def(3).src(Src::Imm(7))]], classes());
    m.blocks[0].term = Term::Goto(1);
    assert_eq!(pad_inflight_waw(&mut m, &alloc(vec![8, 1, 20, 21, 40])), 0);
}

#[test]
fn the_wait_counts_only_the_younger_loads() {
    // two loads after the clobbered one: vmcnt(2) suffices
    let l2 = MInst::new(MOp::BufferLoadDwordX1).def(4).srcreg(0).srcreg(1).src(Src::Imm(0));
    let l3 = MInst::new(MOp::BufferLoadDwordX1).def(4).srcreg(0).srcreg(1).src(Src::Imm(0));
    let mut m = func(vec![vec![load(), l2, l3], vec![MInst::new(MOp::VMovB32).def(3).src(Src::Imm(7))]], classes());
    m.blocks[0].term = Term::Goto(1);
    pad_inflight_waw(&mut m, &alloc(vec![8, 1, 20, 21, 40]));
    assert_eq!(waits_before(&m, 1, MOp::VMovB32), vec![(MOp::SWaitVm, 2)]);
}
