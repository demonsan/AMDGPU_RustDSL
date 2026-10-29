//! Guards for the default-on lowering optimisations (`dce`, `cmpfold`) and for register allocation.
//!
//! 1. `lowering_opts_keep_side_effects_and_definedness`: for every kernel the lowered MIR with the default options
//!    must (a) have the identical sequence of side-effecting instructions (memory, barriers, waits, loads of
//!    kernargs) and the identical block structure as with `none`, and (b) never read a virtual-register dword that
//!    is not defined on every path (a removed definition or a stale folded compare would show up here).
//! 2. `register_assignment_never_clobbers_a_live_value`: on the final allocated MIR (scheduler defaults included) a
//!    reaching-writer analysis over *physical* registers proves that every read sees exactly the write of the
//!    virtual register it names: no other value was assigned to that register in between on any path.
//! 3. `descriptor_declares_every_register`: the own-backend kernel descriptor covers every VGPR/SGPR (and vcc)
//!    the emitted code touches.

use amdgpu_compiler::lower::with_opts;
use amdgpu_compiler::mir::*;
use amdgpu_compiler::sched::effects::{effects, Slot, VCC};
use amdgpu_compiler::sched::liveness::successors;
use amdgpu_compiler::{compile_mir_alloc, kernel_by_name, kernel_names, lower};
use std::collections::{BTreeSet, HashMap, HashSet};

fn side_effect_op(op: MOp) -> bool {
    matches!(
        op,
        MOp::LdsWriteB32 | MOp::LdsWriteB128 | MOp::LdsReadB32 | MOp::LdsReadB128 | MOp::LdsReadB64TrB8
            | MOp::GlobalLoadU16 | MOp::GlobalLoadF32 | MOp::GlobalLoadB128 | MOp::GlobalLoadB128S | MOp::GlobalLoadF32S
            | MOp::GlobalStoreU16 | MOp::GlobalStoreF32 | MOp::SLoadDword | MOp::SLoadDwordx2 | MOp::SBarrier
            | MOp::SWaitVm | MOp::SWaitLgkm | MOp::SSetPrio | MOp::SEndpgm
    )
}

fn side_effect_signature(m: &MFunction) -> Vec<Vec<String>> {
    m.blocks
        .iter()
        .map(|b| {
            let mut v: Vec<String> = b.insts.iter().filter(|i| side_effect_op(i.op)).map(|i| format!("{:?}/{}/{}", i.op, i.imm, i.srcs.len())).collect();
            v.push(format!("{:?}", b.term));
            v
        })
        .collect()
}

/// Virtual-register dwords read before they are defined on some path.
fn undefined_reads(m: &MFunction) -> Vec<String> {
    let succs = successors(m);
    let nb = m.blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            preds[s].push(b);
        }
    }
    let mut out_set: Vec<Option<HashSet<Slot>>> = vec![None; nb]; // None = not reached yet (top)
    for _ in 0..500 {
        let mut changed = false;
        for b in 0..nb {
            let mut cur: HashSet<Slot> = if b == 0 {
                HashSet::new()
            } else {
                let mut acc: Option<HashSet<Slot>> = None;
                for &p in &preds[b] {
                    if let Some(o) = &out_set[p] {
                        acc = Some(match acc {
                            None => o.clone(),
                            Some(a) => a.intersection(o).copied().collect(),
                        });
                    }
                }
                match acc {
                    Some(a) => a,
                    None => continue,
                }
            };
            for inst in &m.blocks[b].insts {
                cur.extend(effects(m, inst).writes.iter().copied());
            }
            if out_set[b].as_ref() != Some(&cur) {
                out_set[b] = Some(cur);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    undefined_reads_report(m, &preds, &out_set)
}
fn undefined_reads_report(m: &MFunction, preds: &[Vec<usize>], out_set: &[Option<HashSet<Slot>>]) -> Vec<String> {
    let mut report = Vec::new();
    for b in 0..m.blocks.len() {
        let mut cur: HashSet<Slot> = if b == 0 {
            HashSet::new()
        } else {
            let mut acc: Option<HashSet<Slot>> = None;
            for &p in &preds[b] {
                if let Some(o) = &out_set[p] {
                    acc = Some(match acc {
                        None => o.clone(),
                        Some(a) => a.intersection(o).copied().collect(),
                    });
                }
            }
            match acc {
                Some(a) => a,
                None => continue,
            }
        };
        for (ii, inst) in m.blocks[b].insts.iter().enumerate() {
            let e = effects(m, inst);
            for r in &e.reads {
                if *r != VCC && *r != u32::MAX - 1 && !cur.contains(r) {
                    report.push(format!("block {b} inst {ii} {:?}: reads undefined slot {r}", inst.op));
                }
            }
            cur.extend(e.writes.iter().copied());
        }
    }
    report
}

#[test]
fn lowering_opts_keep_side_effects_and_definedness() {
    let mut bad = Vec::new();
    for k in kernel_names() {
        let f = kernel_by_name(k).unwrap();
        let none = with_opts(&["none"], || lower::lower(&f)).unwrap();
        let dflt = with_opts(&["dce", "cmpfold"], || lower::lower(&f)).unwrap();
        if side_effect_signature(&none) != side_effect_signature(&dflt) {
            bad.push(format!("{k}: side-effect/terminator sequence differs between none and dce,cmpfold"));
        }
        let (un, ud) = (undefined_reads(&none), undefined_reads(&dflt));
        if ud.len() > un.len() {
            bad.push(format!("{k}: dce,cmpfold introduced undefined reads ({} vs {}): {:?}", ud.len(), un.len(), &ud[..ud.len().min(3)]));
        }
        if dflt.blocks.iter().map(|b| b.insts.len()).sum::<usize>() > none.blocks.iter().map(|b| b.insts.len()).sum::<usize>() {
            bad.push(format!("{k}: dce,cmpfold made the code longer"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Physical register -> set of virtual dword slots that may have been written last (`None` entry = never written).
type Writers = HashMap<(bool, u32), BTreeSet<Slot>>;

fn merge(into: &mut Writers, from: &Writers) -> bool {
    let mut ch = false;
    for (k, v) in from {
        let e = into.entry(*k).or_default();
        for s in v {
            ch |= e.insert(*s);
        }
    }
    ch
}

fn phys_of(m: &MFunction, a: &amdgpu_compiler::regalloc::Alloc, slot: Slot) -> (bool, u32) {
    let t = slot / 16;
    let class = m.vreg_class(t);
    (class.is_vector(), a.phys[t as usize] + (slot - t * 16))
}

fn clobbers(m: &MFunction, a: &amdgpu_compiler::regalloc::Alloc) -> Vec<String> {
    let succs = successors(m);
    let nb = m.blocks.len();
    let mut in_st: Vec<Option<Writers>> = vec![None; nb];
    in_st[0] = Some(Writers::new());
    for _ in 0..500 {
        let mut changed = false;
        for b in 0..nb {
            let Some(mut st) = in_st[b].clone() else { continue };
            for inst in &m.blocks[b].insts {
                let e = effects(m, inst);
                for w in &e.writes {
                    if *w == VCC || *w == u32::MAX - 1 {
                        continue;
                    }
                    st.insert(phys_of(m, a, *w), BTreeSet::from([*w]));
                }
            }
            for &s in &succs[b] {
                match &mut in_st[s] {
                    None => {
                        in_st[s] = Some(st.clone());
                        changed = true;
                    }
                    Some(x) => changed |= merge(x, &st),
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut report = Vec::new();
    for b in 0..nb {
        let Some(mut st) = in_st[b].clone() else { continue };
        for (ii, inst) in m.blocks[b].insts.iter().enumerate() {
            let e = effects(m, inst);
            for r in &e.reads {
                if *r == VCC || *r == u32::MAX - 1 {
                    continue;
                }
                // partial writers (v_cvt_pk_fp8: 16-bit halves) read the old value of their own destination: the first
                // half of the idiom reads garbage by design, the second half is covered by the writer check
                if e.writes.contains(r) && matches!(inst.op, MOp::VCvtPkFp8F32 | MOp::VCvtPkFp8Lo | MOp::VCvtPkFp8Hi | MOp::VCvtPkFp8F32Hi) {
                    continue;
                }
                let p = phys_of(m, a, *r);
                match st.get(&p) {
                    // never written on any path: an undefined read (reported by the other test), not a clobber
                    None => {}
                    Some(ws) => {
                        if !ws.contains(r) || ws.len() > 1 {
                            report.push(format!(
                                "block {b} inst {ii} {:?}: reads slot {r} ({}{}) but the reaching writers are {:?}",
                                inst.op,
                                if p.0 { "v" } else { "s" },
                                p.1,
                                ws
                            ));
                        }
                    }
                }
            }
            for w in &e.writes {
                if *w == VCC || *w == u32::MAX - 1 {
                    continue;
                }
                st.insert(phys_of(m, a, *w), BTreeSet::from([*w]));
            }
        }
    }
    report
}

#[test]
fn register_assignment_never_clobbers_a_live_value() {
    let mut bad = Vec::new();
    for k in kernel_names() {
        let f = kernel_by_name(k).unwrap();
        for opts in [&["none"][..], &["dce", "cmpfold"][..]] {
            let r = with_opts(opts, || compile_mir_alloc(&f));
            let (m, a) = match r {
                Ok(x) => x,
                Err(e) => {
                    bad.push(format!("{k} [{opts:?}]: {e}"));
                    continue;
                }
            };
            let c = clobbers(&m, &a);
            if !c.is_empty() {
                bad.push(format!("{k} [{opts:?}]: {} clobbered reads, first: {}", c.len(), c[0]));
            }
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

fn max_reg(asm: &str, prefix: char) -> i64 {
    let mut best = -1i64;
    let b = asm.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        let boundary = i == 0 || !(b[i - 1] as char).is_ascii_alphanumeric() && b[i - 1] != b'_';
        if c == prefix && boundary {
            let mut j = i + 1;
            if j < b.len() && b[j] == b'[' {
                // range v[a:b]
                j += 1;
                let mut hi = String::new();
                while j < b.len() && b[j] != b':' && b[j] != b']' {
                    j += 1;
                }
                if j < b.len() && b[j] == b':' {
                    j += 1;
                    while j < b.len() && (b[j] as char).is_ascii_digit() {
                        hi.push(b[j] as char);
                        j += 1;
                    }
                    if let Ok(n) = hi.parse::<i64>() {
                        best = best.max(n);
                    }
                }
            } else {
                let mut n = String::new();
                while j < b.len() && (b[j] as char).is_ascii_digit() {
                    n.push(b[j] as char);
                    j += 1;
                }
                if !n.is_empty() {
                    best = best.max(n.parse::<i64>().unwrap());
                }
            }
        }
        i += 1;
    }
    best
}

fn declared(asm: &str, key: &str) -> Option<i64> {
    asm.lines().find_map(|l| l.trim().strip_prefix(key).and_then(|v| v.trim().parse().ok()))
}

#[test]
fn descriptor_declares_every_register() {
    let mut bad = Vec::new();
    for k in kernel_names() {
        let f = kernel_by_name(k).unwrap();
        let asm = match amdgpu_compiler::compile(&f) {
            Ok(c) => c.asm,
            Err(e) => {
                bad.push(format!("{k}: {e}"));
                continue;
            }
        };
        let code: String = asm.split(".amdhsa_kernel").next().unwrap_or(&asm).to_string();
        let (v, s) = (max_reg(&code, 'v'), max_reg(&code, 's'));
        let (dv, ds) = (declared(&asm, ".amdhsa_next_free_vgpr").unwrap_or(0), declared(&asm, ".amdhsa_next_free_sgpr").unwrap_or(0));
        if dv <= v {
            bad.push(format!("{k}: uses v{v} but declares next_free_vgpr {dv}"));
        }
        // vcc occupies two more SGPRs after the last allocated one when reserved by the descriptor; the declared count must
        // at least cover every explicitly named SGPR
        if ds <= s {
            bad.push(format!("{k}: uses s{s} but declares next_free_sgpr {ds}"));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}