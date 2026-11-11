//! W3 B5 (docs/w3-b5-b6.md section 3): the LDS-DMA dependences of a ring are memory dependences of the wait pass. A DMA writes LDS, not a
//! register, so the register scoreboard cannot see that the first LDS read of a slot needs the slot's DMA complete (vmcnt), or that a
//! DMA overwriting a slot needs the slot's earlier LDS reads complete (lgkmcnt). With a plan whose Free phase moves a DMA across
//! iterations, the plan's rings give each access a (ring, relative slot) token. Mutation tests: dropping either wait must be caught by
//! `verify_waits`.

mod common;

use amdgpu_codegen::mir::*;
use amdgpu_codegen::sched::waits::{self, RingModel};
use amdgpu_dsl::plan::*;
use common::func;

/// One phase (Free), a ring of `slots` slots produced and consumed in it: unit 0 = the DMA of iteration `d`, unit 1 = the reads of the
/// current iteration.
fn plan(slots: u32, d: i32) -> KernelPlan {
    let u = |id, name: &str, class, iter| UnitDecl { id, name: name.into(), phase: 0, class, anchor: false, iter, latency: 0, roles: 0 };
    KernelPlan {
        phases: vec![Phase { id: 0, name: "qk".into(), parent: None, kind: PhaseKind::Straight, order: OrderSpec::Free, budget: Budget::default() }],
        rings: vec![RingDecl { id: 0, name: "k".into(), slots, bytes_per_slot: 16384, cadence: Cadence::TwoPhaseOffset, producer_phase: 0, consumer_phase: 0, owner: RingOwner::Cta }],
        units: vec![u(0, "dma", UnitClass::Dma, d), u(1, "read", UnitClass::LdsRead, 0)],
        ..Default::default()
    }
}

fn tagged(mut i: MInst, unit_id: u32) -> MInst {
    i.unit = unit_id + 1;
    i
}

fn dma() -> MInst {
    tagged(MInst::new(MOp::BufferLoadDwordX4Lds).srcreg(2).srcreg(1).srcreg(3).imm(0), 0)
}

fn read() -> MInst {
    tagged(MInst::new(MOp::LdsReadB32).def(4).srcreg(0).imm(0), 1)
}

/// r0 LDS address, r1 voffset, r2 descriptor, r3 soffset, r4 read result
fn mf(insts: Vec<MInst>) -> MFunction {
    let mut b = Block::new(0);
    b.insts = insts;
    func("t", vec![b], vec![RClass::V1, RClass::V1, RClass::S4, RClass::S1, RClass::V1])
}

fn ops(m: &MFunction) -> Vec<MOp> {
    m.blocks[0].insts.iter().map(|i| i.op).collect()
}

#[test]
fn the_model_is_active_only_for_a_moved_dma_in_a_free_phase() {
    assert!(RingModel::of(&plan(2, 2)).is_some());
    assert!(RingModel::of(&plan(2, 0)).is_none(), "no DMA moved across iterations");
    let mut p = plan(2, 2);
    p.phases[0].order = OrderSpec::Authored;
    assert!(RingModel::of(&p).is_none(), "an authored phase keeps its explicit waits");
}

#[test]
fn raw_the_first_read_of_a_slot_waits_for_its_dma() {
    // depth 2, distance 2: the DMA writes the slot the current iteration reads
    let model = RingModel::of(&plan(2, 2));
    let mut m = mf(vec![dma(), read()]);
    // mutation: without the wait the verifier must object
    let err = waits::with_ring_model(model.clone(), || waits::verify_waits(&m));
    assert!(err.is_err(), "verify_waits must reject a read of a slot whose DMA is in flight");
    let n = waits::with_ring_model(model.clone(), || waits::insert_waits(&mut m));
    assert_eq!(n, 1);
    assert_eq!(ops(&m), vec![MOp::BufferLoadDwordX4Lds, MOp::SWaitVm, MOp::LdsReadB32]);
    waits::with_ring_model(model.clone(), || waits::verify_waits(&m)).expect("the repaired block verifies");
    // mutation: drop the inserted wait again
    m.blocks[0].insts.retain(|i| i.op != MOp::SWaitVm);
    assert!(waits::with_ring_model(model, || waits::verify_waits(&m)).is_err());
    // without a ring model (today's kernels) the scoreboard does not see the dependence
    let mut m2 = mf(vec![dma(), read()]);
    assert_eq!(waits::insert_waits(&mut m2), 0);
}

#[test]
fn war_a_dma_overwriting_a_slot_waits_for_its_reads() {
    let model = RingModel::of(&plan(2, 2));
    let mut m = mf(vec![read(), dma()]);
    assert!(waits::with_ring_model(model.clone(), || waits::verify_waits(&m)).is_err(), "a DMA overwriting a slot with reads in flight");
    let n = waits::with_ring_model(model.clone(), || waits::insert_waits(&mut m));
    assert_eq!(n, 1);
    assert_eq!(ops(&m), vec![MOp::LdsReadB32, MOp::SWaitLgkm, MOp::BufferLoadDwordX4Lds]);
    waits::with_ring_model(model, || waits::verify_waits(&m)).unwrap();
}

#[test]
fn a_deeper_ring_separates_the_slots() {
    // depth 3, distance 2: the DMA writes (t + 2) mod 3, the reads touch t mod 3: no wait either way
    let model = RingModel::of(&plan(3, 2));
    for insts in [vec![dma(), read()], vec![read(), dma()]] {
        let mut m = mf(insts);
        assert_eq!(waits::with_ring_model(model.clone(), || waits::insert_waits(&mut m)), 0, "{:?}", ops(&m));
        waits::with_ring_model(model.clone(), || waits::verify_waits(&m)).unwrap();
    }
}

#[test]
fn a_loop_back_edge_ages_the_tokens() {
    // depth 3, distance 2, one iteration per trip: the DMA of trip t is read two trips later. Body: read (t), dma (t + 2), back edge.
    // On entry to the header, a DMA issued one trip ago has relative slot 1, two trips ago 0 = the slot read now: the read must wait.
    let model = RingModel::of(&plan(3, 2));
    let mut b0 = Block::new(0);
    b0.term = Term::Goto(1);
    let mut b1 = Block::new(1);
    b1.insts = vec![read(), dma()];
    b1.term = Term::CondGoto { on: Pred::Scc1, target: 1, fall: 2 };
    let b2 = Block::new(2);
    let mut m = func("t", vec![b0, b1, b2], vec![RClass::V1, RClass::V1, RClass::S4, RClass::S1, RClass::V1]);
    m.loops.push((1, 1));
    assert!(waits::with_ring_model(model.clone(), || waits::verify_waits(&m)).is_err(), "the read of trip t + 2 meets the DMA of trip t");
    let n = waits::with_ring_model(model.clone(), || waits::insert_waits(&mut m));
    assert!(n >= 1);
    assert_eq!(m.blocks[1].insts[0].op, MOp::SWaitVm, "{:?}", m.blocks[1].insts.iter().map(|i| i.op).collect::<Vec<_>>());
    waits::with_ring_model(model, || waits::verify_waits(&m)).unwrap();
}
