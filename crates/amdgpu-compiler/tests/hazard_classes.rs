//! The own tail's value-hazard pass must pad "VALU write -> hazard-sensitive reader" for every VALU writer class (audit of the writer list).
use amdgpu_compiler::hazards::fix_value_hazards_opts;
use amdgpu_compiler::mir::{Block, MFunction, MInst, MOp, RClass};

fn func(insts: Vec<MInst>, nvregs: usize) -> MFunction {
    let mut b = Block::new(0);
    b.insts = insts;
    MFunction {
        name: "h".into(),
        args: Vec::new(),
        kernarg_size: 0,
        workgroup_size: 64,
        lds_bytes: 0,
        blocks: vec![b],
        vregs: vec![RClass::V1; nvregs],
        loops: Vec::new(),
        alias: Default::default(),
        elide_fallthrough: false, edge_pad: false,
        agpr_pin: Default::default(),
    }
}

/// wait states between instruction `from` and the first later `reader` op in the padded block
fn gap(m: &MFunction, writer: MOp, reader: MOp) -> usize {
    let ops = &m.blocks[0].insts;
    let w = ops.iter().position(|i| i.op == writer).unwrap();
    let r = ops.iter().rposition(|i| i.op == reader).unwrap();
    ops[w + 1..r].iter().map(|i| if i.op == MOp::SNop { i.imm as usize + 1 } else { 1 }).sum()
}

#[test]
fn scalef32_converts_get_two_wait_states_before_a_matrix_operand_or_readfirstlane() {
    for writer in [MOp::VCvtPkFp8Lo, MOp::VCvtPkFp8Hi, MOp::VPermlane32Swap, MOp::VReadPhys, MOp::VReadSub] {
        // v0 = writer(v1, v2); mfma uses v0 as an operand right away
        let w = MInst::new(writer).def(0).srcreg(1).srcreg(2);
        let mfma = MInst::new(MOp::VMfmaScale32x32x64Fp8Z).def(3).srcreg(0).srcreg(1).srcreg(2).srcreg(1);
        let mut m = func(vec![w.clone(), mfma], 5);
        fix_value_hazards_opts(&mut m, true);
        assert!(gap(&m, writer, MOp::VMfmaScale32x32x64Fp8Z) >= 2, "{writer:?}: only {} wait states before the MFMA operand", gap(&m, writer, MOp::VMfmaScale32x32x64Fp8Z));
        let rfl = MInst::new(MOp::VReadFirstLane).def(4).srcreg(0);
        let mut m = func(vec![w, rfl], 5);
        fix_value_hazards_opts(&mut m, true);
        assert!(gap(&m, writer, MOp::VReadFirstLane) >= 2, "{writer:?}: only {} wait states before v_readfirstlane", gap(&m, writer, MOp::VReadFirstLane));
    }
}

#[test]
fn a_high_half_scalef32_convert_gets_one_wait_state_before_any_consumer_of_its_result() {
    // found on hardware (the 4-wave prototype, scheduler-reordered orders): `v_cvt_scalef32_pk_fp8_bf16 ... op_sel:[0,0,1]` (the high-half partial-dword write) immediately followed by a
    // consumer of its result read a stale value - a `v_accvgpr_write_b32` (first seen) and a `v_mov_b32` copy; one `s_nop` fixes it. The low half and the accvgpr_read -> accvgpr_write copy
    // pairs of the shipped kernels need nothing, and the partner conversion of the other half (which merges into the same dword) is interlocked.
    let after = |writer: MOp, consumer: MInst| {
        let w = MInst::new(writer).def(0).srcreg(1).srcreg(2);
        let cop = consumer.op;
        let mut m = func(vec![w, consumer], 6);
        fix_value_hazards_opts(&mut m, false);
        gap(&m, writer, cop)
    };
    let accw = || MInst::new(MOp::VAccvgprWrite).def(3).srcreg(0);
    let mov = || MInst::new(MOp::VMovB32).def(3).srcreg(0);
    assert!(after(MOp::VCvtPkFp8Hi, accw()) >= 1, "high-half convert -> v_accvgpr_write needs a wait state");
    assert!(after(MOp::VCvtPkFp8Hi, mov()) >= 1, "high-half convert -> v_mov_b32 needs a wait state");
    assert!(after(MOp::VCvtPkFp8F32Hi, mov()) >= 1, "the f32 high-half convert too");
    assert_eq!(after(MOp::VCvtPkFp8Lo, accw()), 0, "the low half stays unpadded");
    assert_eq!(after(MOp::VCvtPkFp8Lo, mov()), 0, "the low half stays unpadded");
    assert_eq!(after(MOp::VAccvgprRead, accw()), 0, "accvgpr_read -> accvgpr_write copies stay unpadded");
    // the partner conversion reads the dword it merges into: interlocked, no padding (the shipped Lo / Hi pairs)
    let partner = || MInst::new(MOp::VCvtPkFp8Lo).def(0).srcreg(0).srcreg(2);
    assert_eq!(after(MOp::VCvtPkFp8Hi, partner()), 0, "the partner half-convert is exempt");
}
