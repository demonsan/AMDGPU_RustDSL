//! The final-MIR check of pinned blocks (value-hazard pass, wait pass, partial writes) against the compiler's own models.
use amdgpu_compiler::mir::{Block, MFunction, MInst, MOp, RClass};
use amdgpu_compiler::pin_check::check;

fn func(insts: Vec<MInst>, classes: Vec<RClass>) -> MFunction {
    let mut b = Block::new(0);
    b.insts = insts;
    MFunction {
        name: "p".into(),
        args: Vec::new(),
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: vec![b],
        vregs: classes,
        loops: Vec::new(),
        alias: Default::default(),
        elide_fallthrough: false, edge_pad: false,
        agpr_pin: Default::default(),
    }
}

fn pinned(mut i: MInst) -> MInst {
    i.pin = true;
    i
}

#[test]
fn unpinned_code_is_not_checked() {
    let m = func(vec![MInst::new(MOp::VMovB32).def(0).imm(1), MInst::new(MOp::VReadFirstLane).def(1).srcreg(0)], vec![RClass::V1, RClass::S1]);
    assert!(check(&m).is_empty());
}

#[test]
fn pinned_value_hazard_without_padding_is_reported_and_with_padding_is_clean() {
    let w = pinned(MInst::new(MOp::VMovB32).def(0).imm(1));
    let r = pinned(MInst::new(MOp::VReadFirstLane).def(1).srcreg(0));
    let bad = func(vec![w.clone(), r.clone()], vec![RClass::V1, RClass::S1]);
    let e = check(&bad);
    assert!(e.iter().any(|x| x.contains("value-hazard pass would pad")), "{e:?}");
    let nop = pinned(MInst::new(MOp::SNop).imm(1)); // 2 wait states
    let ok = func(vec![w, nop, r], vec![RClass::V1, RClass::S1]);
    assert!(check(&ok).is_empty(), "{:?}", check(&ok));
}

#[test]
fn pinned_read_of_an_outstanding_ds_load_without_a_wait_is_reported() {
    let rd = pinned(MInst::new(MOp::LdsReadB32).def(1).srcreg(0).imm(0));
    let use1 = pinned(MInst::new(MOp::VAddU32).def(2).srcreg(1).srcreg(1));
    let bad = func(vec![rd.clone(), use1.clone()], vec![RClass::V1, RClass::V1, RClass::V1]);
    let e = check(&bad);
    assert!(e.iter().any(|x| x.contains("automatic")), "{e:?}");
    let wait = pinned(MInst::new(MOp::SWaitLgkm).imm(0)); // explicit
    let ok = func(vec![rd, wait, use1], vec![RClass::V1, RClass::V1, RClass::V1]);
    assert!(check(&ok).is_empty(), "{:?}", check(&ok));
}

#[test]
fn pinned_read_of_a_half_written_dword_is_reported() {
    let lo = pinned(MInst::new(MOp::VCvtPkFp8Lo).def(0).srcreg(1).srcreg(2));
    let rd = pinned(MInst::new(MOp::VAddU32).def(3).srcreg(0).srcreg(0));
    let bad = func(vec![lo.clone(), pinned(MInst::new(MOp::SNop).imm(1)), rd.clone()], vec![RClass::V1; 4]);
    let e = check(&bad);
    assert!(e.iter().any(|x| x.contains("only the low half")), "{e:?}");
    let hi = pinned(MInst::new(MOp::VCvtPkFp8Hi).def(0).srcreg(1).srcreg(2).srcreg(0));
    let ok = func(vec![lo, hi, pinned(MInst::new(MOp::SNop).imm(1)), rd], vec![RClass::V1; 4]);
    assert!(check(&ok).is_empty(), "{:?}", check(&ok));
}