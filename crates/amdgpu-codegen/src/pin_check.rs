//! Final-MIR validation of PINNED (hand-authored, safe-asm) instruction runs, run on the scheduled MIR before the value-hazard pass.
//!
//! The safe-asm verifier proves a block against its own wait-state clock; this check proves the block against the COMPILER's models, in the
//! context it was lowered into (so a model mismatch cannot hide behind the two-model split):
//!  * value hazards: the own value-hazard pass is run on a copy; any `s_nop` it would add before a pinned instruction means the sequence's own
//!    padding is weaker than the pass's rules (the sequence is then padded by the pass - reported, because it changes the hand-authored schedule);
//!  * DS / VMEM counts: the wait pass is re-run on a copy with every automatic wait removed; an automatic wait it inserts before a pinned
//!    instruction means the sequence's explicit `s_waitcnt` do not cover a read of an outstanding load;
//!  * partial writes: a pinned reader of a dword of which only one 16-bit half was written by the scalef32 / fp8 half-write converts.

use crate::mir::*;
use std::collections::HashMap;

pub fn check(m: &MFunction) -> Vec<String> {
    let mut errs = Vec::new();
    if !m.blocks.iter().any(|b| b.insts.iter().any(|i| i.pin)) {
        return errs;
    }
    // value hazards: what would the pass pad before a pinned instruction?
    let mut c = m.clone();
    crate::hazards::fix_value_hazards_opts(&mut c, true);
    for (bi, (orig, padded)) in m.blocks.iter().zip(c.blocks.iter()).enumerate() {
        let mut oi = 0usize;
        let mut pending_nops = 0usize;
        for p in &padded.insts {
            if p.op == MOp::SNop && p.auto {
                pending_nops += 1;
                continue;
            }
            if p.op == MOp::SNop && !orig.insts.get(oi).map_or(false, |o| o.op == MOp::SNop) {
                pending_nops += 1;
                continue;
            }
            if let Some(o) = orig.insts.get(oi) {
                // padding at the BOUNDARY of a run (against an instruction outside it) is the pass's normal work; padding INSIDE the run changes the schedule
                let inside = oi > 0 && orig.insts[oi - 1].pin;
                if o.pin && inside && pending_nops > 0 {
                    errs.push(format!("block {bi} inst {oi} ({:?}): the value-hazard pass would pad {pending_nops} s_nop before this pinned instruction (the sequence's own padding is weaker than the pass's rules)", o.op));
                }
            }
            pending_nops = 0;
            oi += 1;
        }
    }
    // DS / VMEM counts: strip the automatic waits and see where the wait pass wants one before a pinned instruction
    let mut w = m.clone();
    for b in &mut w.blocks {
        b.insts.retain(|i| !(i.auto && matches!(i.op, MOp::SWaitVm | MOp::SWaitLgkm)));
    }
    let before: Vec<usize> = w.blocks.iter().map(|b| b.insts.len()).collect();
    crate::sched::waits::insert_waits(&mut w);
    for (bi, b) in w.blocks.iter().enumerate() {
        let _ = before[bi];
        for k in 1..b.insts.len() {
            // an automatic wait at the boundary of a run (before its first instruction) covers loads issued outside; one INSIDE the run is a violation
            let mut back = k;
            while back > 0 && b.insts[back - 1].auto && matches!(b.insts[back - 1].op, MOp::SWaitVm | MOp::SWaitLgkm) {
                back -= 1;
            }
            let inside = back > 0 && b.insts[back - 1].pin;
            if inside && b.insts[k - 1].auto && matches!(b.insts[k - 1].op, MOp::SWaitVm | MOp::SWaitLgkm) && b.insts[k].pin {
                errs.push(format!("block {bi}: pinned {:?} needs an automatic {:?}({}) the sequence did not provide (its explicit s_waitcnt do not cover an outstanding load it reads)", b.insts[k].op, b.insts[k - 1].op, b.insts[k - 1].imm));
            }
        }
    }
    // partial writes
    for (bi, b) in m.blocks.iter().enumerate() {
        let mut halves: HashMap<VReg, u8> = HashMap::new();
        for (ii, i) in b.insts.iter().enumerate() {
            if !i.pin {
                if let Some(d) = i.dst {
                    halves.remove(&d);
                }
                continue;
            }
            let lo = crate::op_attrs::attrs(i.op).half == Some(crate::op_attrs::Half::Lo);
            let hi = crate::op_attrs::attrs(i.op).half == Some(crate::op_attrs::Half::Hi);
            for s in &i.srcs {
                if let Src::Reg(r) = s {
                    if let Some(h) = halves.get(r) {
                        let tied = hi && Some(*r) == i.dst;
                        if *h != 3 && !tied {
                            errs.push(format!("block {bi} inst {ii} ({:?}): reads r{r} of which only the {} half was written", i.op, if *h == 1 { "low" } else { "high" }));
                        }
                    }
                }
            }
            if let Some(d) = i.dst {
                if lo || hi {
                    let h = halves.entry(d).or_insert(0);
                    *h |= if lo { 1 } else { 2 };
                } else if halves.contains_key(&d) {
                    halves.insert(d, 3);
                }
            }
        }
    }
    errs
}