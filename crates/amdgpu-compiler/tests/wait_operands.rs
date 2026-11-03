//! Regression: a register operand of an LDS-DMA load (`buffer_load ... lds` voffset / soffset) must wait for an in-flight LDS read that writes it.
//! The DMA ops are fences for the list scheduler and used to skip the operand check of `sched::waits` (insert and verify), so a `ds_read` result feeding a DMA
//! voffset - e.g. an LDS-spilled address - was consumed without `s_waitcnt lgkmcnt` (silent wrong results in the GEMM when the register-pressure ladder reached the LDS spill).

mod common;

use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::waits;
use common::func;

fn mk(with_wait: bool) -> MFunction {
    // r0 = LDS address (V1), r1 = loaded voffset (V1), r2 = buffer descriptor (S4), r3 = soffset (S1)
    let mut insts = vec![MInst::new(MOp::LdsReadB32).def(1).srcreg(0).imm(0)];
    if with_wait {
        insts.push(MInst::new(MOp::SWaitLgkm).imm(0));
    }
    insts.push(MInst::new(MOp::BufferLoadDwordX4Lds).srcreg(2).srcreg(1).srcreg(3).imm(0));
    let mut b = Block::new(0);
    b.insts = insts;
    func("t", vec![b], vec![RClass::V1, RClass::V1, RClass::S4, RClass::S1])
}

#[test]
fn dma_voffset_waits_for_a_pending_lds_read() {
    let mut m = mk(false);
    assert!(waits::verify_waits(&m).is_err(), "verify_waits must reject a DMA that consumes the voffset of an LDS read still in flight");
    let n = waits::insert_waits(&mut m);
    assert_eq!(n, 1, "one wait inserted");
    let ops: Vec<MOp> = m.blocks[0].insts.iter().map(|i| i.op).collect();
    assert_eq!(ops, vec![MOp::LdsReadB32, MOp::SWaitLgkm, MOp::BufferLoadDwordX4Lds]);
    waits::verify_waits(&m).expect("the repaired block verifies");
    // with an explicit wait already present nothing is added and the block verifies
    let mut m2 = mk(true);
    assert_eq!(waits::insert_waits(&mut m2), 0);
    waits::verify_waits(&m2).unwrap();
}