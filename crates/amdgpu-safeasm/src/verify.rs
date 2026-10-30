//! Independent hazard verifier. It walks the finished instruction list with its OWN clock (it shares no state with `Seq`), so it is the net under
//! the type-level discipline: a dropped token, a forged sequence or any future hole in the API still has to pass this check.
//!
//! Rules (CDNA4 ISA 7.6 as used by this project; the numbers are wait states between the two instructions):
//! * VALU write of a VGPR -> matrix operand / `v_readfirstlane` / mask read: >= 2
//! * MFMA / SMFMAC write -> any reader of the result: >= 20, except a same-opcode in-place chain (hardware interlocked)
//! * WAW: a write to a register with a matrix write in flight (< 20), except the same-opcode in-place chain
//! * WAR: a write to a register a matrix instruction is still reading (< 20 after it issued), except its own accumulator chain
//! * a register with an outstanding DS/VMEM load must not be read before the `s_waitcnt` that retires the load
//! * the sequence must not end with loads outstanding, and may end with matrix writes in flight only if declared (`output_exit`)

use crate::block::*;
use std::collections::HashMap;

const MATRIX_RESULT: u32 = 20;
const VALU_TO_MATRIX_READ: u32 = 2;
const VCC: u32 = u32::MAX;

fn is_matrix(op: &str) -> bool {
    op.starts_with("VMfma") || op.starts_with("VSmfmac")
}

fn is_load(op: &str) -> Option<bool> {
    // Some(true) = lgkm queue, Some(false) = vm queue
    match op {
        "LdsReadB32" | "LdsReadB128" | "LdsReadB64" => Some(true),
        o if o.starts_with("GlobalLoad") => Some(false),
        _ => None,
    }
}

fn is_valu_writer(op: &str) -> bool {
    op.starts_with('V') && !is_matrix(op) && is_load(op).is_none()
}

pub(crate) fn verify_block(b: &AsmBlock) -> Result<(), Vec<String>> {
    let mut errs: Vec<String> = Vec::new();
    let alias: HashMap<u32, u32> = b.alias.iter().map(|(a, base, _)| (*a, *base)).collect();
    let canon = |r: u32| alias.get(&r).copied().unwrap_or(r);

    let mut slot: u32 = 0;
    let mut valu_write: HashMap<u32, u32> = HashMap::new();
    let mut matrix_write: HashMap<u32, (u32, &'static str)> = HashMap::new();
    // (register read by a matrix instruction, slot of that instruction, index)
    let mut matrix_reads: Vec<(u32, u32, usize)> = Vec::new();
    let mut lgkm: Vec<Option<u32>> = Vec::new();
    let mut vm: Vec<Option<u32>> = Vec::new();

    for (k, i) in b.insts.iter().enumerate() {
        let here = |m: String| format!("inst {k} ({}): {m}", i.op);
        let matrix = is_matrix(i.op);
        let reads: Vec<u32> = i.srcs.iter().filter_map(|s| if let ASrc::Reg(r) = s { Some(*r) } else { None }).collect();
        let dst = i.dst.map(canon);

        // in-place same-opcode chain: reads and rewrites its own accumulator
        let chain_reg = if matrix {
            dst.filter(|d| matrix_write.get(d).map_or(false, |(_, o)| *o == i.op) && reads.iter().any(|r| canon(*r) == *d))
        } else {
            None
        };

        let has_vcc_read = matches!(i.op, "VCndmask" | "VAddcCoU32");
        let mut rd: Vec<(u32, bool)> = reads.iter().map(|r| (canon(*r), matches!(i.op, "VCndmaskM") && Some(*r) == reads.last().copied())).collect();
        if has_vcc_read {
            rd.push((VCC, true));
        }
        for (r, is_mask) in rd {
            if lgkm.iter().chain(vm.iter()).any(|x| *x == Some(r)) {
                errs.push(here(format!("reads r{r} while its load is outstanding (no covering s_waitcnt)")));
            }
            if let Some((w, wop)) = matrix_write.get(&r).copied() {
                if Some(r) != chain_reg {
                    let have = slot - w - 1;
                    if have < MATRIX_RESULT {
                        errs.push(here(format!("reads r{r}, the result of {wop} issued {have} wait states ago: needs {MATRIX_RESULT} (unretired matrix result)")));
                    }
                }
            }
            if let Some(w) = valu_write.get(&r).copied() {
                if matrix || i.op == "VReadFirstLane" || is_mask || r == VCC {
                    let have = slot - w - 1;
                    if have < VALU_TO_MATRIX_READ {
                        errs.push(here(format!("reads r{r} written by a VALU instruction {have} wait states ago: needs {VALU_TO_MATRIX_READ}")));
                    }
                }
            }
        }

        if let Some(d) = dst {
            if let Some((w, wop)) = matrix_write.get(&d).copied() {
                if Some(d) != chain_reg && slot - w - 1 < MATRIX_RESULT {
                    errs.push(here(format!("writes r{d} while the {wop} writing it is still in flight (WAW)")));
                }
            }
            for (r, at, ki) in &matrix_reads {
                if *r == d && Some(d) != chain_reg && slot - at - 1 < MATRIX_RESULT && *ki != k {
                    errs.push(here(format!("writes r{d}, an operand instruction {ki} is still reading (WAR, issued {} wait states ago)", slot - at - 1)));
                }
            }
        }

        // state updates
        match i.op {
            "SNop" => {
                slot += i.imm as u32 + 1;
                continue;
            }
            "SWaitLgkm" => {
                let keep = i.imm as usize;
                if lgkm.len() > keep {
                    let n = lgkm.len() - keep;
                    lgkm.drain(..n);
                }
            }
            "SWaitVm" => {
                let keep = i.imm as usize;
                if vm.len() > keep {
                    let n = vm.len() - keep;
                    vm.drain(..n);
                }
            }
            _ => {}
        }
        if let Some(q) = is_load(i.op) {
            let d = dst;
            if q {
                lgkm.push(d);
            } else {
                vm.push(d);
            }
            if let Some(d) = d {
                valu_write.remove(&d);
            }
        } else if i.op == "LdsWriteB32" {
            lgkm.push(None);
        } else if matrix {
            if let Some(d) = dst {
                matrix_write.insert(d, (slot, static_name(i.op)));
                valu_write.remove(&d);
                for r in &reads {
                    let rc = canon(*r);
                    if rc != d {
                        matrix_reads.push((rc, slot, k));
                    }
                }
            }
        } else if is_valu_writer(i.op) {
            if let Some(d) = dst {
                valu_write.insert(d, slot);
            }
            if i.op.starts_with("VCmp") && dst.is_none() {
                valu_write.insert(VCC, slot);
            }
        }
        slot += 1;
    }

    if lgkm.iter().chain(vm.iter()).next().is_some() {
        errs.push("sequence ends with DS/VMEM loads outstanding".to_string());
    }
    for (r, (w, wop)) in &matrix_write {
        if slot - w - 1 < MATRIX_RESULT {
            let declared = b.exit_in_flight.iter().any(|(x, _)| canon(*x) == *r) && b.outputs.iter().any(|p| canon(p.reg) == *r);
            if !declared {
                errs.push(format!("end of sequence: r{r}, the result of {wop}, is still in flight and not exported with output_exit (unretired matrix result)"));
            }
        }
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

fn static_name(op: &str) -> &'static str {
    match op {
        "VMfmaScale32x32x64Fp8" => "VMfmaScale32x32x64Fp8",
        "VMfmaScale32x32x64Fp8Z" => "VMfmaScale32x32x64Fp8Z",
        "VSmfmac16x16x128Fp8" => "VSmfmac16x16x128Fp8",
        _ => "matrix op",
    }
}