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
fn a_high_half_scalef32_convert_gets_one_wait_state_before_a_dependent_accvgpr_write() {
    // found on hardware (the 4-wave prototype, a search-scheduled order): `v_cvt_scalef32_pk_fp8_bf16 ... op_sel:[0,0,1]` (the high-half partial write) immediately followed by
    // `v_accvgpr_write_b32` of its result read a stale value; one `s_nop` fixes it. The low half and the accvgpr_read -> accvgpr_write copy pairs of the shipped kernels need nothing.
    let mk = |writer: MOp| {
        let w = MInst::new(writer).def(0).srcreg(1).srcreg(2);
        let aw = MInst::new(MOp::VAccvgprWrite).def(3).srcreg(0);
        let mut m = func(vec![w, aw], 5);
        fix_value_hazards_opts(&mut m, false);
        gap(&m, writer, MOp::VAccvgprWrite)
    };
    assert!(mk(MOp::VCvtPkFp8Hi) >= 1, "the high-half convert needs a wait state before the accvgpr_write");
    assert_eq!(mk(MOp::VCvtPkFp8Lo), 0, "the low half stays unpadded");
    assert_eq!(mk(MOp::VAccvgprRead), 0, "accvgpr_read -> accvgpr_write copies stay unpadded");
}
