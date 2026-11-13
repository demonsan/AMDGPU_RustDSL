//! The reaching-writer check guards the two-phase (hole-filling) register allocation. It must accept a register shared with a value that is not live on
//! the zero-trip path of a loop (a loop-carried value defined inside the loop body and read after it; `FMHA_OPT=constres1..6` shape) and still reject a
//! genuine clobber (the value was written, overwritten by another vreg on the same path, then read).
use amdgpu_codegen::mir::{Block, MFunction, MInst, MOp, Pred, RClass, Term};
use amdgpu_codegen::regalloc::{check_reaching, Alloc};

fn func(blocks: Vec<(Vec<MInst>, Term)>, nv: usize) -> MFunction {
    let mut bs = Vec::new();
    for (i, (insts, term)) in blocks.into_iter().enumerate() {
        let mut b = Block::new(i as u32);
        b.insts = insts;
        b.term = term;
        bs.push(b);
    }
    MFunction { name: "r".into(), args: Vec::new(), kernarg_size: 0, workgroup_size: 64, lds_bytes: 0, fp_mode: None, exact: false, blocks: bs, vregs: vec![RClass::V1; nv], loops: Vec::new(), alias: Default::default(), elide_fallthrough: false, edge_pad: false, agpr_pin: Default::default(), reg_hints: Default::default() }
}

fn alloc(phys: Vec<u32>) -> Alloc {
    let n = phys.len();
    Alloc { phys, max_v: 64, max_s: 8, scalar: vec![false; n] }
}

// vreg 0 = foreign value (prologue only), vreg 1 = loop-carried own value (defined in the loop body), vreg 2 = result; vregs 0 and 1 share v10.
//   B0: v0 = 1            -> B1
//   B1: loop test         -> B2 (body) | B3 (exit)        <- the zero-trip path B0 -> B1 -> B3 never writes v1
//   B2: v1 = 2            -> B1
//   B3: v2 = v1 + v1      -> end
fn loop_shape() -> MFunction {
    func(
        vec![
            (vec![MInst::new(MOp::VMovB32).def(0).imm(1)], Term::Goto(1)),
            (vec![], Term::CondGoto { on: Pred::Vccnz, target: 2, fall: 3 }),
            (vec![MInst::new(MOp::VMovB32).def(1).imm(2)], Term::Goto(1)),
            (vec![MInst::new(MOp::VAddU32).def(2).srcreg(1).srcreg(1)], Term::Endpgm),
        ],
        3,
    )
}

#[test]
fn shared_register_on_the_zero_trip_path_is_accepted() {
    let m = loop_shape();
    let v = check_reaching(&m, &alloc(vec![10, 10, 20]));
    assert!(v.is_empty(), "{v:?}");
}

#[test]
fn a_clobber_after_the_own_write_is_still_rejected() {
    // B2: v1 = 2; v0 = 3 (foreign value lands on v10 while v1 is still needed); B3 reads v1
    let m = func(
        vec![
            (vec![MInst::new(MOp::VMovB32).def(0).imm(1)], Term::Goto(1)),
            (vec![], Term::CondGoto { on: Pred::Vccnz, target: 2, fall: 3 }),
            (vec![MInst::new(MOp::VMovB32).def(1).imm(2), MInst::new(MOp::VMovB32).def(0).imm(3)], Term::Goto(1)),
            (vec![MInst::new(MOp::VAddU32).def(2).srcreg(1).srcreg(1)], Term::Endpgm),
        ],
        3,
    );
    let v = check_reaching(&m, &alloc(vec![10, 10, 20]));
    assert!(!v.is_empty(), "the clobber of v1 by v0 must be reported");
    assert!(v[0].contains("overwritten"), "{v:?}");
}

#[test]
fn a_value_whose_own_write_never_arrives_is_still_rejected() {
    // v1 is never written, but the foreign v0 sits on its register: the reader would see another value's data
    let m = func(
        vec![
            (vec![MInst::new(MOp::VMovB32).def(0).imm(1)], Term::Goto(1)),
            (vec![MInst::new(MOp::VAddU32).def(2).srcreg(1).srcreg(1)], Term::Endpgm),
        ],
        3,
    );
    let v = check_reaching(&m, &alloc(vec![10, 10, 20]));
    assert!(!v.is_empty() && v[0].contains("never arrives"), "{v:?}");
}

#[test]
fn distinct_registers_and_in_place_accumulation_are_clean() {
    // an accumulator updated in place inside the loop (v1 = v1 + v1) with its init before the loop
    let m = func(
        vec![
            (vec![MInst::new(MOp::VMovB32).def(1).imm(1)], Term::Goto(1)),
            (vec![], Term::CondGoto { on: Pred::Vccnz, target: 2, fall: 3 }),
            (vec![MInst::new(MOp::VAddU32).def(1).srcreg(1).srcreg(1)], Term::Goto(1)),
            (vec![MInst::new(MOp::VAddU32).def(2).srcreg(1).srcreg(1)], Term::Endpgm),
        ],
        3,
    );
    assert!(check_reaching(&m, &alloc(vec![10, 11, 20])).is_empty());
}