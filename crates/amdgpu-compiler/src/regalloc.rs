//! Linear-scan register allocation over virtual registers.
//!
//! Only the classes and rules this backend needs are modelled: one VGPR file
//! (with 64-bit pairs requiring an even base) and one SGPR file. ABI registers
//! are reserved. Spilling is not implemented on purpose: a kernel that does not
//! fit is a compile error rather than a silent performance cliff.

use std::collections::HashSet;

use crate::mir::*;

pub struct Alloc {
    pub phys: Vec<u32>,
    pub max_v: u32,
    pub max_s: u32,
}

const VGPR_COUNT: usize = 256;
const SGPR_COUNT: usize = 104;

/// Reserved: v0 holds workitem id; s0:s1 kernarg pointer; s2..s4 workgroup ids.
const V_RESERVED: usize = 1;
const S_RESERVED: usize = 8;

pub fn allocate(m: &MFunction) -> Result<Alloc, String> {
    let n = m.vregs.len();
    let mut start = vec![usize::MAX; n];
    let mut end = vec![0usize; n];

    let mut idx_of: Vec<Vec<usize>> = Vec::with_capacity(m.blocks.len());
    let mut next = 0usize;
    for b in &m.blocks {
        let mut v = Vec::with_capacity(b.insts.len());
        for _ in 0..b.insts.len() {
            v.push(next);
            next += 1;
        }
        idx_of.push(v);
    }

    // --- CFG ---------------------------------------------------------
    let label_to_block: std::collections::HashMap<u32, usize> =
        m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); m.blocks.len()];
    for (i, b) in m.blocks.iter().enumerate() {
        match &b.term {
            Term::Goto(t) => {
                if let Some(j) = label_to_block.get(t) {
                    succs[i].push(*j);
                }
            }
            Term::CondGoto { target, fall, .. } => {
                if let Some(j) = label_to_block.get(target) {
                    succs[i].push(*j);
                }
                if let Some(j) = label_to_block.get(fall) {
                    succs[i].push(*j);
                }
            }
            Term::Endpgm => {}
        }
    }

    // --- block-level liveness ---------------------------------------
    let mut use_set: Vec<HashSet<VReg>> = vec![HashSet::new(); m.blocks.len()];
    let mut def_set: Vec<HashSet<VReg>> = vec![HashSet::new(); m.blocks.len()];
    for (i, b) in m.blocks.iter().enumerate() {
        for inst in &b.insts {
            for r in inst.uses() {
                if !def_set[i].contains(&r) {
                    use_set[i].insert(r);
                }
            }
            if let Some(d) = inst.dst {
                def_set[i].insert(d);
            }
        }
    }

    let mut live_in: Vec<HashSet<VReg>> = vec![HashSet::new(); m.blocks.len()];
    let mut live_out: Vec<HashSet<VReg>> = vec![HashSet::new(); m.blocks.len()];
    let mut changed = true;
    let mut guard = 0usize;
    while changed {
        changed = false;
        guard += 1;
        if guard > 10_000 {
            return Err("liveness did not converge".to_string());
        }
        for i in (0..m.blocks.len()).rev() {
            let mut out: HashSet<VReg> = HashSet::new();
            for s in &succs[i] {
                for r in &live_in[*s] {
                    out.insert(*r);
                }
            }
            let mut inn = use_set[i].clone();
            for r in &out {
                if !def_set[i].contains(r) {
                    inn.insert(*r);
                }
            }
            if inn != live_in[i] || out != live_out[i] {
                live_in[i] = inn;
                live_out[i] = out;
                changed = true;
            }
        }
    }

    // --- live intervals ---------------------------------------------
    let mut bump = |r: VReg, at: usize, start: &mut Vec<usize>, end: &mut Vec<usize>| {
        if at < start[r as usize] {
            start[r as usize] = at;
        }
        if at > end[r as usize] {
            end[r as usize] = at;
        }
    };

    for (bi, b) in m.blocks.iter().enumerate() {
        let mut live = live_out[bi].clone();
        for i in (0..b.insts.len()).rev() {
            let at = idx_of[bi][i];
            let snapshot: Vec<VReg> = live.iter().copied().collect();
            for r in snapshot {
                bump(r, at, &mut start, &mut end);
            }
            let inst = &b.insts[i];
            if let Some(d) = inst.dst {
                bump(d, at, &mut start, &mut end);
                live.remove(&d);
            }
            for r in inst.uses() {
                bump(r, at, &mut start, &mut end);
                live.insert(r);
            }
        }
        // Values live across the block boundary are live at block entry.
        let entry = idx_of[bi].first().copied().unwrap_or(0);
        let snapshot: Vec<VReg> = live.iter().copied().collect();
        for r in snapshot {
            bump(r, entry, &mut start, &mut end);
        }
    }

    // --- extend values that cross a loop backedge --------------------
    // A value live at a loop header is re-read on the next iteration, so its
    // interval must reach the latch instead of ending at its last use.
    for (header, latch) in &m.loops {
        let (hb, lb) = match (label_to_block.get(header), label_to_block.get(latch)) {
            (Some(h), Some(l)) => (*h, *l),
            _ => continue,
        };
        let latch_end = idx_of[lb].last().copied().unwrap_or(0);
        let vals: Vec<VReg> = live_in[hb].iter().copied().collect();
        for r in vals {
            if latch_end > end[r as usize] {
                end[r as usize] = latch_end;
            }
            if start[r as usize] == usize::MAX {
                start[r as usize] = 0;
            }
        }
    }

    // --- linear scan -------------------------------------------------
    let mut phys = vec![u32::MAX; n];
    let mut used_v = vec![false; VGPR_COUNT];
    let mut used_s = vec![false; SGPR_COUNT];
    for i in 0..V_RESERVED {
        used_v[i] = true;
    }
    for i in 0..S_RESERVED {
        used_s[i] = true;
    }

    // VGPRs: V1 and V2 share one file.
    let mut vregs: Vec<(usize, usize, VReg, RClass)> = Vec::new();
    for (r, c) in m.vregs.iter().enumerate() {
        if matches!(c, RClass::V1 | RClass::V2) {
            let s = if start[r] == usize::MAX { 0 } else { start[r] };
            vregs.push((s, end[r], r as VReg, *c));
        }
    }
    vregs.sort_by_key(|x| (x.0, x.1));
    let mut active: Vec<(usize, VReg, RClass)> = Vec::new();
    for (s, e, r, c) in vregs {
        active.retain(|(ae, ar, ac)| {
            if *ae < s {
                free_phys(&mut used_v, &mut used_s, phys[*ar as usize], *ac);
                false
            } else {
                true
            }
        });
        let slot = find_slot(&used_v, c).ok_or_else(|| {
            format!(
                "out of VGPRs allocating r{r} ({c:?}, live {}..{})",
                s, e
            )
        })?;
        phys[r as usize] = slot as u32;
        active.push((e, r, c));
    }

    // SGPRs.
    let mut sregs: Vec<(usize, usize, VReg, RClass)> = Vec::new();
    for (r, c) in m.vregs.iter().enumerate() {
        if matches!(c, RClass::S1 | RClass::S2) {
            let s = if start[r] == usize::MAX { 0 } else { start[r] };
            sregs.push((s, end[r], r as VReg, *c));
        }
    }
    sregs.sort_by_key(|x| (x.0, x.1));
    let mut sactive: Vec<(usize, VReg, RClass)> = Vec::new();
    for (s, e, r, c) in sregs {
        sactive.retain(|(ae, ar, ac)| {
            if *ae < s {
                free_phys(&mut used_v, &mut used_s, phys[*ar as usize], *ac);
                false
            } else {
                true
            }
        });
        let slot = find_slot(&used_s, c).ok_or_else(|| {
            format!(
                "out of SGPRs allocating r{r} ({c:?}, live {}..{})",
                s, e
            )
        })?;
        phys[r as usize] = slot as u32;
        sactive.push((e, r, c));
    }

    let mut max_v = 0usize;
    let mut max_s = 0usize;
    for (i, u) in used_v.iter().enumerate() {
        if *u {
            max_v = i + 1;
        }
    }
    for (i, u) in used_s.iter().enumerate() {
        if *u {
            max_s = i + 1;
        }
    }

    for (r, p) in phys.iter().enumerate() {
        if *p == u32::MAX {
            // Never-used virtual register: give it a harmless slot.
            let _ = r;
        }
    }
    let mut patched = Vec::with_capacity(n);
    for (r, p) in phys.into_iter().enumerate() {
        if p == u32::MAX {
            patched.push(0u32);
            let _ = r;
        } else {
            patched.push(p);
        }
    }

    Ok(Alloc { phys: patched, max_v: max_v as u32, max_s: max_s as u32 })
}

fn free_phys(used_v: &mut [bool], used_s: &mut [bool], phys: u32, c: RClass) {
    let p = phys as usize;
    match c {
        RClass::V1 => used_v[p] = false,
        RClass::V2 => {
            used_v[p] = false;
            used_v[p + 1] = false;
        }
        RClass::S1 => used_s[p] = false,
        RClass::S2 => {
            used_s[p] = false;
            used_s[p + 1] = false;
        }
    }
}

fn find_slot(used: &[bool], c: RClass) -> Option<usize> {
    let (lo, width, align) = match c {
        RClass::V1 => (V_RESERVED, 1usize, 1usize),
        RClass::V2 => (2usize, 2usize, 2usize),
        RClass::S1 => (S_RESERVED, 1, 1),
        RClass::S2 => (S_RESERVED, 2, 2),
    };
    let n = used.len();
    let mut i = lo;
    while i + width <= n {
        if i % align == 0 && (0..width).all(|k| !used[i + k]) {
            return Some(i);
        }
        i += 1;
    }
    None
}
