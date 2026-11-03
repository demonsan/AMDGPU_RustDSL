//! Regression (found by the biasacc,vpre determinism failure): register allocation may hand the registers of an MFMA result that is DEAD (or consumed
//! early) to a different value; the MFMA keeps writing its destination for ~19-21 wait states, so a VALU write (or read) of an overlapping register by
//! another vreg is a physical WAW/RAW race against the late MFMA write. `pad_physical_war` must wait for the MFMA's completion in that case, across
//! block boundaries too, and must not pad unrelated registers or the MFMA's own consumers (value-level pass).
use amdgpu_compiler::hazards::pad_physical_war;
use amdgpu_compiler::mir::{Block, MFunction, MInst, MOp, RClass, Term};
use amdgpu_compiler::regalloc::Alloc;

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
        elide_fallthrough: false,
        agpr_pin: Default::default(),
    }
}

fn alloc(phys: Vec<u32>) -> Alloc {
    let n = phys.len();
    Alloc { phys, max_v: 256, max_s: 8, scalar: vec![false; n] }
}

/// wait states between the MFMA and the first later instruction `op` after padding
fn gap(m: &MFunction, bi: usize, first: MOp, then: MOp) -> usize {
    let ins = &m.blocks[bi].insts;
    let a = ins.iter().position(|i| i.op == first).unwrap();
    let b = ins.iter().rposition(|i| i.op == then).unwrap();
    ins[a + 1..b].iter().map(|i| if i.op == MOp::SNop { i.imm as usize + 1 } else { 1 }).sum()
}

// vregs: 0 = V2 (mfma A), 1 = V2 (B), 2 = V16 (mfma dst, dead), 3 = V1 (temp that reuses the MFMA's registers), 4 = V1 (unrelated temp)
fn classes() -> Vec<RClass> {
    vec![RClass::V2, RClass::V2, RClass::V16, RClass::V1, RClass::V1]
}

#[test]
fn write_to_the_registers_of_an_inflight_mfma_result_waits_for_its_completion() {
    let mfma = MInst::new(MOp::VMfma32x32x8F16Z).def(2).srcreg(0).srcreg(1);
    let tmp = MInst::new(MOp::VMovB32).def(3).imm(7); // vreg 3 lives in v146, inside the MFMA's v[144:159]
    let mut m = func(vec![vec![mfma, tmp]], classes());
    pad_physical_war(&mut m, &alloc(vec![2, 4, 144, 146, 20]));
    assert!(gap(&m, 0, MOp::VMfma32x32x8F16Z, MOp::VMovB32) >= 18, "gap {}", gap(&m, 0, MOp::VMfma32x32x8F16Z, MOp::VMovB32));
}

#[test]
fn unrelated_registers_and_the_mfma_own_vreg_are_not_padded() {
    let mfma = MInst::new(MOp::VMfma32x32x8F16Z).def(2).srcreg(0).srcreg(1);
    let tmp = MInst::new(MOp::VMovB32).def(4).imm(7); // v20: disjoint from v[144:159]
    let mut m = func(vec![vec![mfma, tmp]], classes());
    pad_physical_war(&mut m, &alloc(vec![2, 4, 144, 146, 20]));
    assert_eq!(gap(&m, 0, MOp::VMfma32x32x8F16Z, MOp::VMovB32), 0);
}

#[test]
fn a_late_mfma_write_also_clobbers_a_value_read_later_from_the_same_registers() {
    // vreg 3 was written (by something earlier) in v146, the MFMA lands over it, then vreg 3 is read: the read must wait for the MFMA as well
    let mfma = MInst::new(MOp::VMfma32x32x8F16Z).def(2).srcreg(0).srcreg(1);
    let rd = MInst::new(MOp::VAddU32).def(4).srcreg(3).srcreg(3);
    let mut m = func(vec![vec![mfma, rd]], classes());
    pad_physical_war(&mut m, &alloc(vec![2, 4, 144, 146, 20]));
    assert!(gap(&m, 0, MOp::VMfma32x32x8F16Z, MOp::VAddU32) >= 18);
}

#[test]
fn the_wait_carries_across_a_block_boundary() {
    let mfma = MInst::new(MOp::VMfma32x32x8F16Z).def(2).srcreg(0).srcreg(1);
    let tmp = MInst::new(MOp::VMovB32).def(3).imm(7);
    let mut m = func(vec![vec![mfma], vec![tmp]], classes());
    m.blocks[0].term = Term::Goto(1);
    pad_physical_war(&mut m, &alloc(vec![2, 4, 144, 146, 20]));
    let pad: usize = m.blocks[1].insts.iter().take_while(|i| i.op == MOp::SNop).map(|i| i.imm as usize + 1).sum::<usize>()
        + m.blocks[0].insts.iter().rev().take_while(|i| i.op == MOp::SNop).map(|i| i.imm as usize + 1).sum::<usize>();
    assert!(pad >= 18, "only {pad} wait states of padding between the MFMA and the reuse in the successor");
}
// ---- VMEM store data hazard: a global_store_dwordx4 followed by a VALU write of its data registers ------------------------------------------

// vregs: 0 = V2 address, 1 = V4 store data (phys v4..v7), 2 = V1 temp reusing v4 (VALU writer), 3 = V4 LDS-read destination reusing v4..v7, 4 = V1 unrelated
fn store_classes() -> Vec<RClass> {
    vec![RClass::V2, RClass::V4, RClass::V1, RClass::V4, RClass::V1]
}

#[test]
fn a_valu_write_of_the_wide_store_data_registers_waits_two_wait_states() {
    let st = MInst::new(MOp::GlobalStoreF32x4).srcreg(0).srcreg(1).imm(0);
    let wr = MInst::new(MOp::VMovB32).def(2).imm(1); // v4 = ...
    let mut m = func(vec![vec![st, MInst::new(MOp::SWaitLgkm).imm(0), wr]], store_classes());
    pad_physical_war(&mut m, &alloc(vec![2, 4, 4, 4, 40]));
    // s_waitcnt counts as one wait state: one more is needed
    assert!(gap(&m, 0, MOp::GlobalStoreF32x4, MOp::VMovB32) >= 2, "gap {}", gap(&m, 0, MOp::GlobalStoreF32x4, MOp::VMovB32));
}

#[test]
fn the_store_data_hazard_does_not_touch_unrelated_registers_lds_reads_or_narrow_stores() {
    let wide = MInst::new(MOp::GlobalStoreF32x4).srcreg(0).srcreg(1).imm(0);
    // unrelated VALU destination (v40): no padding
    let mut m = func(vec![vec![wide.clone(), MInst::new(MOp::VMovB32).def(4).imm(1)]], store_classes());
    pad_physical_war(&mut m, &alloc(vec![2, 4, 4, 4, 40]));
    assert_eq!(gap(&m, 0, MOp::GlobalStoreF32x4, MOp::VMovB32), 0);
    // an LDS read returning into the data registers is not a VALU write (a nop in between changed nothing on hardware)
    let rd = MInst::new(MOp::LdsReadB128).def(3).srcreg(4).imm(0);
    let mut m = func(vec![vec![wide, rd]], store_classes());
    pad_physical_war(&mut m, &alloc(vec![2, 4, 4, 4, 40]));
    assert_eq!(gap(&m, 0, MOp::GlobalStoreF32x4, MOp::LdsReadB128), 0);
    // an 8-byte store has no such hazard
    let narrow = MInst::new(MOp::GlobalStoreF32x2).srcreg(0).srcreg(1).imm(0);
    let wr = MInst::new(MOp::VMovB32).def(2).imm(1);
    let mut m = func(vec![vec![narrow, wr]], store_classes());
    pad_physical_war(&mut m, &alloc(vec![2, 4, 4, 4, 40]));
    assert_eq!(gap(&m, 0, MOp::GlobalStoreF32x2, MOp::VMovB32), 0);
}
#[test]
fn buffer_store_dwordx4_data_gets_the_same_two_wait_states() {
    // vregs: 0 = V4 store data (v4..v7), 1 = S4 descriptor, 2 = V1 offset, 3 = V1 temp reusing v4
    let classes = vec![RClass::V4, RClass::S4, RClass::V1, RClass::V1];
    let st = MInst::new(MOp::BufferStoreDwordX4).srcreg(0).srcreg(1).srcreg(2).src(amdgpu_compiler::mir::Src::Imm(0)).imm(0);
    let wr = MInst::new(MOp::VMovB32).def(3).imm(1);
    let mut m = func(vec![vec![st.clone(), MInst::new(MOp::SWaitLgkm).imm(0), wr.clone()]], classes.clone());
    pad_physical_war(&mut m, &alloc(vec![4, 8, 20, 4]));
    assert!(gap(&m, 0, MOp::BufferStoreDwordX4, MOp::VMovB32) >= 2, "gap {}", gap(&m, 0, MOp::BufferStoreDwordX4, MOp::VMovB32));
    // a narrow buffer store has no such hazard
    let narrow = MInst::new(MOp::BufferStoreDwordX2).srcreg(0).srcreg(1).srcreg(2).src(amdgpu_compiler::mir::Src::Imm(0)).imm(0);
    let mut m = func(vec![vec![narrow, wr]], classes);
    pad_physical_war(&mut m, &alloc(vec![4, 8, 20, 4]));
    assert_eq!(gap(&m, 0, MOp::BufferStoreDwordX2, MOp::VMovB32), 0);
}
