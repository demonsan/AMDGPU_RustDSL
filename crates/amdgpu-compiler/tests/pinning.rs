//! Pinned (hand-authored) instruction runs survive the list scheduler verbatim; the unpinned control is reordered.
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched;
use amdgpu_safeasm::*;

/// A straight-line block: independent ALU work followed by a global load and a consumer chain. The list scheduler hoists the long-latency load
/// above the ALU work when nothing pins it.
fn block(pin_middle: bool) -> MFunction {
    let mut vregs = vec![RClass::V2];
    let mut v1 = |c: &mut Vec<RClass>| {
        c.push(RClass::V1);
        (c.len() - 1) as VReg
    };
    let mut insts: Vec<MInst> = Vec::new();
    let a: Vec<VReg> = (0..6).map(|_| v1(&mut vregs)).collect();
    for (k, r) in a.iter().enumerate() {
        let mut i = MInst::new(MOp::VMovB32).def(*r).imm(k as i64 + 1);
        i.pin = pin_middle;
        insts.push(i);
    }
    let sums: Vec<VReg> = (0..3).map(|_| v1(&mut vregs)).collect();
    for k in 0..3 {
        let mut i = MInst::new(MOp::VAddU32).def(sums[k]).srcreg(a[2 * k]).srcreg(a[2 * k + 1]);
        i.pin = pin_middle;
        insts.push(i);
    }
    let l = v1(&mut vregs);
    insts.push(MInst::new(MOp::GlobalLoadF32).def(l).srcreg(0).imm(0));
    let out = v1(&mut vregs);
    insts.push(MInst::new(MOp::VAddF32).def(out).srcreg(l).srcreg(l));
    insts.push(MInst::new(MOp::GlobalStoreF32).srcreg(0).srcreg(out).imm(0));
    let mut b = Block::new(0);
    b.insts = insts;
    MFunction {
        name: "pin_test".into(),
        args: Vec::new(),
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: vec![b],
        vregs,
        loops: Vec::new(),
        alias: Default::default(),
        elide_fallthrough: false,
    }
}

fn ops(m: &MFunction) -> Vec<MOp> {
    m.blocks[0].insts.iter().map(|i| i.op).collect()
}

fn run(mut m: MFunction) -> MFunction {
    let o = sched::Options::default(); // full list scheduler (reorder = true)
    sched::run(&mut m, &o).expect("scheduler");
    m
}

#[test]
fn unpinned_control_is_reordered() {
    let before = ops(&block(false));
    let after = ops(&run(block(false)));
    assert_ne!(before, after, "the control block must be reordered by the list scheduler (the load is hoisted)");
}

#[test]
fn pinned_run_is_preserved_verbatim_and_acts_as_a_barrier() {
    let m = block(true);
    let before: Vec<(MOp, Option<u32>, Vec<Src>, i64)> = m.blocks[0].insts.iter().map(|i| (i.op, i.dst, i.srcs.clone(), i.imm)).collect();
    let after_m = run(block(true));
    let pinned: Vec<(MOp, Option<u32>, Vec<Src>, i64)> = after_m.blocks[0].insts.iter().filter(|i| i.pin).map(|i| (i.op, i.dst, i.srcs.clone(), i.imm)).collect();
    let want: Vec<_> = before.iter().take(9).cloned().collect();
    assert_eq!(pinned, want, "the pinned instructions must keep their exact order and operands");
    // nothing moved into the middle of the pinned run: the nine pinned instructions are contiguous
    let pos: Vec<usize> = after_m.blocks[0].insts.iter().enumerate().filter(|(_, i)| i.pin).map(|(k, _)| k).collect();
    assert_eq!(pos.last().unwrap() - pos.first().unwrap(), 8, "pinned run must stay contiguous: {pos:?}");
}

#[test]
fn safeasm_block_pin_flag_reaches_the_mir() {
    let mut s = Seq::new();
    s.pin();
    let mut x = s.vgpr::<1>();
    s.v_mov_b32(&mut x, Imm::<1>::new());
    s.output(x);
    let b = s.finish().unwrap();
    assert!(b.is_pinned());
}