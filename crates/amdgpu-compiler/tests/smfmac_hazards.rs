//! SMFMAC hazards (docs/vendor-spec.md 4.5.2, CDNA4 ISA 7.6):
//! * the accumulator C is a real source of the MIR op (it is D, in place);
//! * a VALU write of A, B, SRC2 or C needs 2 wait states (8 clocks) before the SMFMAC;
//! * a dependent same-opcode SMFMAC chain through D is interlocked by the hardware (no padding wanted).

use amdgpu_compiler::mir::*;
use amdgpu_compiler::{compile, compile_mir};
use amdgpu_dsl::{DType, FnBuilder, Op, Opnd};

fn vrange(tok: &str) -> Option<(u32, u32)> {
    let t = tok.trim().trim_end_matches(',');
    let rest = t.strip_prefix('v')?;
    if let Some(r) = rest.strip_prefix('[') {
        let r = r.trim_end_matches(']');
        let (a, b) = r.split_once(':')?;
        Some((a.parse().ok()?, b.parse::<u32>().ok()? + 1))
    } else {
        let n: u32 = rest.parse().ok()?;
        Some((n, n + 1))
    }
}

fn valu_inputs_kernel() -> amdgpu_dsl::Function {
    let mut b = FnBuilder::new("smfmac_valu_inputs", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let lane_off = b.mul_i32(tid, c16);
    let base = b.ptr_add(o, lane_off);
    let one = b.c_i32(1);
    // every operand produced by VALU instructions right before the SMFMAC
    let x = b.add_i32(tid, one);
    let y = b.add_i32(x, tid);
    let a = b.frag4(x, y, x, y);
    let bb = b.frag8(x, y, x, y, x, y, x, y);
    let idx = b.add_i32(y, y);
    let c = b.frag4(x, y, x, y);
    let d = b.smfmac_fp8_16x16x128(a, bb, idx, c);
    let d0 = b.frag4_lane(d, 0);
    let f0 = b.bitcast_f32(d0);
    b.store_f32(base, 0, f0);
    b.finish()
}

#[test]
fn smfmac_mir_op_has_the_accumulator_as_a_source() {
    let f = valu_inputs_kernel();
    let m = compile_mir(&f).unwrap();
    let mut seen = 0;
    for blk in &m.blocks {
        for i in &blk.insts {
            if i.op == MOp::VSmfmac16x16x128Fp8 {
                seen += 1;
                assert_eq!(i.srcs.len(), 4, "A, B, SRC2 and C");
                assert!(matches!(i.srcs[3], Src::Reg(c) if Some(c) == i.dst), "C must be D (in place)");
            }
        }
    }
    assert_eq!(seen, 1);
    assert!(amdgpu_compiler::mir::verify(&m).is_empty());
}

#[test]
fn valu_written_smfmac_operands_get_two_wait_states() {
    let f = valu_inputs_kernel();
    let asm = compile(&f).unwrap().asm;
    let lines: Vec<&str> = asm.lines().map(|l| l.split(';').next().unwrap().trim()).filter(|l| !l.is_empty() && !l.starts_with('.')).collect();
    let i = lines.iter().position(|l| l.starts_with("v_smfmac")).expect("smfmac in the asm");
    let ops: Vec<&str> = lines[i].split_whitespace().skip(1).collect();
    let ranges: Vec<(u32, u32)> = ops.iter().filter_map(|t| vrange(t)).collect(); // D(=C), A, B, SRC2
    assert_eq!(ranges.len(), 4, "{}", lines[i]);
    let mut states = 0usize;
    for j in (0..i).rev() {
        let l = lines[j];
        if l.starts_with("s_nop") {
            states += l.trim_start_matches("s_nop").trim().parse::<usize>().unwrap_or(0) + 1;
            continue;
        }
        if l.starts_with("v_") && !l.starts_with("v_mfma") && !l.starts_with("v_smfmac") {
            if let Some(dst) = l.split_whitespace().nth(1).and_then(vrange) {
                if ranges.iter().any(|r| dst.0 < r.1 && r.0 < dst.1) {
                    assert!(states >= 2, "`{l}` writes an SMFMAC operand with only {states} wait states before `{}`:\n{asm}", lines[i]);
                }
            }
        }
        states += 1;
        if states >= 8 {
            break;
        }
    }
}

#[test]
fn dependent_smfmac_chain_is_not_padded() {
    let mut b = FnBuilder::new("smfmac_chain", 64);
    b.arg("o", DType::Ptr, 0, 8);
    let o = b.let_(Op::KernargPtr { offset: 0 });
    let tid = b.let_(Op::WorkitemId);
    let c16 = b.c_i32(16);
    let lane_off = b.mul_i32(tid, c16);
    let base = b.ptr_add(o, lane_off);
    let ones = b.c_i32(0x38383838u32 as i32);
    let i01 = b.c_i32(0x44444444u32 as i32);
    let i23 = b.c_i32(0xeeeeeeeeu32 as i32);
    let a = b.frag4(ones, ones, ones, ones);
    let bb = b.frag8(ones, ones, ones, ones, ones, ones, ones, ones);
    let c0 = b.frag_zero(DType::Frag4);
    let d1 = b.smfmac_fp8_16x16x128(a, bb, i01, c0);
    let d2 = b.smfmac_fp8_16x16x128(a, bb, i23, d1);
    let l = b.frag4_lane(d2, 0);
    let lf = b.bitcast_f32(l);
    b.store_f32(base, 0, lf);
    let _ = Opnd::I32(0);
    let asm = compile(&b.finish()).unwrap().asm;
    let lines: Vec<&str> = asm.lines().map(|l| l.split(';').next().unwrap().trim()).filter(|l| !l.is_empty() && !l.starts_with('.')).collect();
    let idx: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.starts_with("v_smfmac")).map(|(i, _)| i).collect();
    assert_eq!(idx.len(), 2);
    let between = &lines[idx[0] + 1..idx[1]];
    assert!(between.iter().all(|l| !l.starts_with("s_nop")), "chained SMFMACs were padded: {between:?}");
}