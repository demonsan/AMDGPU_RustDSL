//! Linear-scan register allocation over virtual registers.
//!
//! Only the classes and rules this backend needs are modelled: one VGPR file
//! (with 64-bit pairs requiring an even base) and one SGPR file. ABI registers
//! are reserved. Spilling is not implemented on purpose: a kernel that does not
//! fit is a compile error rather than a silent performance cliff.

use std::collections::{HashMap, HashSet};

use crate::mir::*;

pub struct Alloc {
    pub phys: Vec<u32>,
    pub max_v: u32,
    pub max_s: u32,
    /// Per vreg: lives in an SGPR (so a VALU source operand prints `sN`).
    pub scalar: Vec<bool>,
}

const VGPR_COUNT: usize = crate::target::T.regs.vgpr.count as usize;
/// AGPRs per wave (a0..a255); arch VGPR + AGPR together must fit the unified 512-register file.
const AGPR_COUNT: usize = crate::target::agpr_count(crate::target::T) as usize;
pub const UNIFIED_FILE: u32 = crate::target::unified_total(crate::target::T);
// 102 usable SGPRs (the descriptor's next_free_sgpr must stay <= 102; vcc is separate)
const SGPR_COUNT: usize = crate::target::T.regs.sgpr.count as usize;

/// Reserved: v0 holds workitem id; s0:s1 kernarg pointer; s2..s4 workgroup ids.
const V_RESERVED: usize = crate::target::T.regs.vgpr.reserved as usize;
const S_RESERVED: usize = crate::target::T.regs.sgpr.reserved as usize;

pub fn allocate(m: &MFunction) -> Result<Alloc, String> {
    let a = allocate_any(m)?;
    if m.reg_hints.active() {
        crate::regplan::check(m, &a)?;
    }
    Ok(a)
}

fn allocate_any(m: &MFunction) -> Result<Alloc, String> {
    // First try the compact policy. If the kernel runs out of registers only because wide
    // tuples cannot find an aligned hole between scattered singles, retry with the singles
    // packed from the top of the file so the low registers stay available for tuples.
    match allocate_with(m, false) {
        Err(e) if e.contains("out of VGPRs") => match allocate_with(m, true) {
            // last resort: offline two-phase packing (wide tuples first, singles into the remaining holes in time)
            Err(e2) if e2.contains("out of VGPRs") => allocate_two_phase(m).and_then(|a| checked(m, a, "two-phase")).map_err(|e3| if e3.contains("out of VGPRs") { format!("{e2}  ||  {e3}") } else { e3 }),
            other => other,
        },
        other => other,
    }
}

/// Reject an allocation whose reaching-writer check fails (the hole-filling packing must never ship a clobbered value).
fn checked(m: &MFunction, a: Alloc, what: &str) -> Result<Alloc, String> {
    let v = check_reaching(m, &a);
    if v.is_empty() {
        Ok(a)
    } else {
        Err(format!("{what} allocation failed the reaching-writer check ({} violations), e.g. {}", v.len(), v[0]))
    }
}

fn allocate_with(m: &MFunction, singles_from_top: bool) -> Result<Alloc, String> {
    allocate_policy(m, Policy::Scan { singles_from_top })
}

fn allocate_two_phase(m: &MFunction) -> Result<Alloc, String> {
    allocate_policy(m, Policy::TwoPhase)
}

#[derive(Clone, Copy)]
enum Policy {
    Scan { singles_from_top: bool },
    TwoPhase,
}

fn allocate_policy(m: &MFunction, policy: Policy) -> Result<Alloc, String> {
    let singles_from_top = matches!(policy, Policy::Scan { singles_from_top: true });
    let n = m.vregs.len();
    let mut start = vec![usize::MAX; n];
    let mut end = vec![0usize; n];

    let mut idx_of: Vec<Vec<usize>> = Vec::with_capacity(m.blocks.len());
    // Linear position of each block's first slot. An empty block still sits at a definite
    // position; using 0 for it would stretch every value live across it back to the start of
    // the function.
    let mut block_start: Vec<usize> = Vec::with_capacity(m.blocks.len());
    let mut next = 0usize;
    for b in &m.blocks {
        block_start.push(next);
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
                let r = m.canon(r);
                if !def_set[i].contains(&r) {
                    use_set[i].insert(r);
                }
            }
            if let Some(d) = inst.dst {
                def_set[i].insert(m.canon(d));
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
    let bump = |r: VReg, at: usize, start: &mut Vec<usize>, end: &mut Vec<usize>| {
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
                let d = m.canon(d);
                bump(d, at, &mut start, &mut end);
                live.remove(&d);
            }
            for r in inst.uses() {
                let r = m.canon(r);
                bump(r, at, &mut start, &mut end);
                live.insert(r);
            }
        }
        // Values live across the block boundary are live at block entry.
        let entry = idx_of[bi].first().copied().unwrap_or(block_start[bi]);
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
        let latch_end = idx_of[lb].last().copied().unwrap_or(block_start[lb]);
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

    // --- precise live segments (dword-level liveness, holes included) ------------------------------------------
    // Used by the two-phase policy: a register may be reused inside a hole of another value's live range. Liveness is
    // tracked per register DWORD (the scheduler's effects model: lane writes, partial writers, in-place operands), so a
    // fragment built lane by lane stays live between its lane writes.
    let segs: Vec<Vec<(usize, usize)>> = if matches!(policy, Policy::TwoPhase) { precise_segments(m, &idx_of, &succs) } else { vec![Vec::new(); n] };
    // --- register plan (MFunction::reg_hints; regplan.rs) ----------------
    let plan = Plan::new(m);
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
        // A vreg that is never defined or used (for example the original destination of a
        // producer that was retargeted into a fragment) needs no register.
        if c.is_vector() && !m.alias.contains_key(&(r as VReg)) && start[r] != usize::MAX {
            let s = start[r];
            vregs.push((s, end[r], r as VReg, *c));
        }
    }
    vregs.sort_by_key(|x| (x.0, x.1));
    let mut active: Vec<(usize, VReg, RClass)> = Vec::new();
    // Highest slot ever handed out, not just the ones still live at the end:
    // the descriptor must cover every register the code touches.
    let mut max_v = 0usize;
    if matches!(policy, Policy::TwoPhase) {
        // Per physical register: the (start, end) intervals already placed on it. Wide tuples first (widest, then longest),
        // every placement checked for overlap in time, so a single may sit in a register between two phases of a tuple.
        let mut occ: Vec<Vec<(usize, usize)>> = vec![Vec::new(); VGPR_COUNT];
        for i in 0..V_RESERVED {
            occ[i].push((0, usize::MAX));
        }
        let seg_len = |r: VReg| -> usize { segs[r as usize].iter().map(|(a, b)| b - a + 1).sum() };
        let mut order: Vec<(usize, usize, VReg, RClass)> = vregs.clone();
        // pinned values first (their registers are fixed: placing them before the free values keeps the pin honourable), then wide tuples
        order.sort_by_key(|x| (plan.pin(x.2).is_none(), std::cmp::Reverse(x.3.dwords()), std::cmp::Reverse(seg_len(x.2)), x.0));
        for (s, e, r, c) in order {
            if phys[r as usize] != u32::MAX {
                // placed together with its rotation class
                continue;
            }
            let own = |q: VReg| -> Vec<(usize, usize)> { if segs[q as usize].is_empty() { vec![(start[q as usize], end[q as usize])] } else { segs[q as usize].clone() } };
            // a rotation class is placed as ONE value: the union of its members' live segments on one tuple (members must not overlap)
            let class = plan.rot_class(r);
            for (k, &a) in class.iter().enumerate() {
                for &b in class.iter().skip(k + 1) {
                    if own(a).iter().any(|(as_, ae)| own(b).iter().any(|(bs, be)| !(*ae < *bs || *be < *as_))) {
                        let (rn, pa) = plan.rot_pos(a);
                        let (_, pb) = plan.rot_pos(b);
                        return Err(format!(
                            "register plan: rotation `{rn}`: {} (fragment {pa}) and {} (fragment {pb}) share a buffer but are live at the same time (a read ran ahead of the previous fragment's last use)",
                            plan.name(a), plan.name(b)
                        ));
                    }
                }
            }
            let my: Vec<(usize, usize)> = if class.len() > 1 { class.iter().flat_map(|q| own(*q)).collect() } else if segs[r as usize].is_empty() { vec![(s, e)] } else { segs[r as usize].clone() };
            let width = c.dwords() as usize;
            let lo = if width == 1 { V_RESERVED } else { width };
            let forbid = plan.forbidden(r, s, e, &start, &end, &phys, m);
            // a rotation class shares one tuple: a member placed earlier fixes the registers
            let forced = plan.rot_target(r, &phys);
            let mut found = None;
            let mut i = lo;
            while i + width <= VGPR_COUNT {
                let pin_ok = plan.pin(r).map_or(true, |b| b as usize == i) && forced.as_ref().map_or(true, |f| f.0 as usize == i);
                // tuple alignment: the allocator's own rule is width alignment; an explicit pin needs only the ISA's even alignment (gfx90a+:
                // 64-bit aligned VGPR tuples; the vendor FMHA keeps its S tiles at v[52:67])
                let aligned = i % width == 0 || (plan.pin(r).is_some() && i % 2 == 0);
                if pin_ok && aligned && (0..width).all(|k| !forbid[i + k] && occ[i + k].iter().all(|(os, oe)| my.iter().all(|(ms, me)| *oe < *ms || *me < *os))) {
                    found = Some(i);
                    break;
                }
                i += 1;
            }
            if found.is_none() {
                if let Some((b, q, rn, pos)) = &forced {
                    let overlap = segs[*q as usize].iter().any(|(qs, qe)| my.iter().any(|(ms, me)| !(*qe < *ms || *me < *qs)));
                    return Err(if overlap {
                        format!("register plan: rotation `{rn}`: {} (fragment {pos}, live {s}..{e}) and {} share a buffer but are live at the same time (a read ran ahead of the previous fragment's last use)", plan.name(r), plan.name(*q))
                    } else {
                        format!("register plan: rotation `{rn}`: the buffer of {} (fragment {pos}) at v{b} is taken by another value live at the same time", plan.name(r))
                    });
                }
                if let Some(b) = plan.pin(r) {
                    let bi = b as usize;
                    let why = if (bi..bi + width).any(|k| forbid[k]) {
                        let others: Vec<String> = plan
                            .pins
                            .iter()
                            .filter(|(&p, &pb)| p != r && start[p as usize] != usize::MAX && start[p as usize] <= e && s <= end[p as usize] && (pb as usize) < bi + width && bi < pb as usize + m.vreg_class(p).dwords() as usize)
                            .map(|(&p, &pb)| format!("{} at v{pb} [{}..{}]", plan.name(p), start[p as usize], end[p as usize]))
                            .collect();
                        format!("forbidden by overlapping pins: {}", others.join(", "))
                    } else {
                        format!("occupied: {:?}", (bi..bi + width).map(|k| occ[k].iter().filter(|(os, oe)| my.iter().any(|(ms, me)| !(*oe < *ms || *me < *os))).count()).collect::<Vec<_>>())
                    };
                    return Err(format!("register plan: pin of {} [{s}..{e}] at v{b} cannot be honoured ({why})", plan.name(r)));
                }
            }
            let slot = found.ok_or_else(|| {
                // numeric lower bound: the largest number of dwords live at one position
                let mut ev: Vec<(usize, i64)> = Vec::new();
                for (_, _, r2, c2) in &vregs {
                    // dword-level live segments of the value (not the [first, last] hull)
                    let sg = &segs[*r2 as usize];
                    if sg.is_empty() {
                        continue;
                    }
                    for (a, b) in sg {
                        ev.push((*a, c2.dwords() as i64));
                        ev.push((b.saturating_add(1), -(c2.dwords() as i64)));
                    }
                }
                ev.sort();
                let (mut cur, mut peak, mut at) = (V_RESERVED as i64, 0i64, 0usize);
                for (p, d) in ev {
                    cur += d;
                    if cur > peak {
                        peak = cur;
                        at = p;
                    }
                }
                format!("out of VGPRs allocating r{r} ({c:?}, live {s}..{e}) in the two-phase packing; peak live dwords {peak} at position {at} (limit {VGPR_COUNT})")
            })?;
            for k in 0..width {
                occ[slot + k].extend(my.iter().copied());
            }
            phys[r as usize] = slot as u32;
            for &q in &class {
                phys[q as usize] = slot as u32;
            }
            max_v = max_v.max(slot + width);
        }
        for (i, o) in occ.iter().enumerate() {
            if !o.is_empty() {
                used_v[i] = true;
            }
        }
        vregs = Vec::new();
    }
    for (s, e, r, c) in vregs {
        active.retain(|(ae, ar, ac)| {
            if *ae < s {
                free_phys(&mut used_v, &mut used_s, phys[*ar as usize], *ac);
                false
            } else {
                true
            }
        });
        let slot = if plan.active {
            plan.slot(r, c, s, e, &used_v, &start, &end, &phys, m, singles_from_top)?
        } else {
            find_slot(&used_v, c, singles_from_top)
        };
        let slot = slot.ok_or_else(|| {
            let mut dwords = 0usize;
            let mut by_class = [0usize; 5];
            for (_, ar, ac) in &active {
                dwords += ac.dwords() as usize;
                match ac {
                    RClass::V1 => by_class[0] += 1,
                    RClass::V2 => by_class[1] += 1,
                    RClass::V4 => by_class[2] += 1,
                    RClass::V8 => by_class[3] += 1,
                    RClass::V16 => by_class[4] += 1,
                    _ => {}
                }
                let _ = ar;
            }
            let map: String = used_v.iter().map(|u| if *u { '#' } else { '.' }).collect();
            let wide: Vec<String> = active
                .iter()
                .filter(|(_, _, ac)| ac.dwords() >= 8)
                .map(|(ae, ar, ac)| {
                    let first_block = (0..m.blocks.len()).find(|bi| live_in[*bi].contains(ar));
                    format!("r{ar}:{ac:?}@{}..{ae} livein_first_block={first_block:?}", start[*ar as usize])
                })
                .collect();
            format!(
                "out of VGPRs allocating r{r} ({c:?}, live {}..{}); wide {wide:?}; map {map}; {} live vregs holding {} dwords \
                 (V1 {} V2 {} V4 {} V8 {} V16 {})",
                s,
                e,
                active.len(),
                dwords,
                by_class[0],
                by_class[1],
                by_class[2],
                by_class[3],
                by_class[4]
            )
        })?;
        phys[r as usize] = slot as u32;
        mark_used(&mut used_v, slot, c);
        max_v = max_v.max(slot + c.dwords() as usize);
        active.push((e, r, c));
    }

    // SGPRs.
    let mut sregs: Vec<(usize, usize, VReg, RClass)> = Vec::new();
    for (r, c) in m.vregs.iter().enumerate() {
        if c.is_scalar() {
            let s = if start[r] == usize::MAX { 0 } else { start[r] };
            sregs.push((s, end[r], r as VReg, *c));
        }
    }
    sregs.sort_by_key(|x| (x.0, x.1));
    let mut sactive: Vec<(usize, VReg, RClass)> = Vec::new();
    let mut max_s = 0usize;
    for (s, e, r, c) in sregs {
        sactive.retain(|(ae, ar, ac)| {
            if *ae < s {
                free_phys(&mut used_v, &mut used_s, phys[*ar as usize], *ac);
                false
            } else {
                true
            }
        });
        let slot = find_slot(&used_s, c, false).ok_or_else(|| {
            format!(
                "out of SGPRs allocating r{r} ({c:?}, live {}..{})",
                s, e
            )
        })?;
        phys[r as usize] = slot as u32;
        mark_used(&mut used_s, slot, c);
        max_s = max_s.max(slot + c.dwords() as usize);
        sactive.push((e, r, c));
    }

    // AGPRs: a separate register file (one unified 512-register budget with the arch VGPRs). Explicit pins first, then first fit
    // over per-register occupancy intervals (hulls, like the scan above).
    let mut max_a = 0usize;
    {
        let mut avs: Vec<(usize, usize, VReg, RClass)> = Vec::new();
        for (r, c) in m.vregs.iter().enumerate() {
            if c.is_agpr() && !m.alias.contains_key(&(r as VReg)) && start[r] != usize::MAX {
                avs.push((start[r], end[r], r as VReg, *c));
            }
        }
        avs.sort_by_key(|x| (x.0, x.1));
        let mut occ: Vec<Vec<(usize, usize)>> = vec![Vec::new(); AGPR_COUNT];
        let fits = |occ: &Vec<Vec<(usize, usize)>>, i: usize, w: usize, s: usize, e: usize| -> bool {
            i + w <= AGPR_COUNT && (0..w).all(|k| occ[i + k].iter().all(|(os, oe)| *oe < s || e < *os))
        };
        for &(s, e, r, c) in avs.iter().filter(|x| m.agpr_pin.contains_key(&x.2)) {
            let w = c.dwords() as usize;
            let base = m.agpr_pin[&r] as usize;
            if base % w != 0 {
                return Err(format!("AGPR pin of r{r} ({c:?}) at a{base} is not aligned to its width {w}"));
            }
            if !fits(&occ, base, w, s, e) {
                return Err(format!("AGPR pin of r{r} ({c:?}) at a{base}..a{} overlaps another value live at the same time (or leaves the file)", base + w - 1));
            }
            for k in 0..w {
                occ[base + k].push((s, e));
            }
            phys[r as usize] = AGPR_TAG + base as u32;
            max_a = max_a.max(base + w);
        }
        for &(s, e, r, c) in avs.iter().filter(|x| !m.agpr_pin.contains_key(&x.2)) {
            let w = c.dwords() as usize;
            let mut i = 0usize;
            let slot = loop {
                if i + w > AGPR_COUNT {
                    return Err(format!("out of AGPRs allocating r{r} ({c:?}, live {s}..{e})"));
                }
                if i % w == 0 && fits(&occ, i, w, s, e) {
                    break i;
                }
                i += 1;
            };
            for k in 0..w {
                occ[slot + k].push((s, e));
            }
            phys[r as usize] = AGPR_TAG + slot as u32;
            max_a = max_a.max(slot + w);
        }
    }
    {
        // unified budget: accum_offset (arch VGPRs rounded to 4) + AGPRs <= 512; arch VGPRs alone <= 256
        let arch = (max_v + 3) / 4 * 4;
        if max_a > 0 && arch + max_a > UNIFIED_FILE as usize {
            return Err(format!("out of AGPRs: {arch} arch VGPRs + {max_a} AGPRs exceed the unified {UNIFIED_FILE}-register file"));
        }
    }
    // Sub-register views live inside their tuple.
    for (a, (t, off)) in &m.alias {
        // (a tuple that was never allocated - e.g. a load coalesced into a loop variable - leaves its views unallocated too)
        if phys[*t as usize] != u32::MAX {
            phys[*a as usize] = phys[*t as usize] + *off as u32;
        }
    }

    for (i, u) in used_v.iter().enumerate() {
        if *u {
            max_v = max_v.max(i + 1);
        }
    }
    for (i, u) in used_s.iter().enumerate() {
        if *u {
            max_s = max_s.max(i + 1);
        }
    }
    // (a second loop here used to ASSIGN max_s = highest still-live slot + 1, discarding the running maximum: kernels whose last live
    // scalar registers were low under-declared .amdhsa_next_free_sgpr and the hardware aliased the registers above it)

    for (r, p) in phys.iter().enumerate() {
        if *p == u32::MAX {
            // Never-used virtual register: give it a harmless slot.
            let _ = r;
        }
    }
    check_mfma_operand_aliasing(m, &phys)?;

    let mut patched = Vec::with_capacity(n);
    for (r, p) in phys.into_iter().enumerate() {
        if p == u32::MAX {
            patched.push(0u32);
            let _ = r;
        } else {
            patched.push(p);
        }
    }

    Ok(Alloc { phys: patched, max_v: max_v as u32, max_s: max_s as u32, scalar: m.vregs.iter().map(|c| c.is_scalar()).collect() })
}

/// Number of AGPRs a function uses (highest allocated AGPR + 1), from an allocation.
pub fn agpr_count(m: &MFunction, a: &Alloc) -> u32 {
    let mut n = 0u32;
    for (r, c) in m.vregs.iter().enumerate() {
        if c.is_agpr() && is_agpr_phys(a.phys[r]) {
            n = n.max(a.phys[r] - AGPR_TAG + c.dwords());
        }
    }
    n
}

/// Unified-file accounting of an allocation: (accum_offset, next_free_vgpr, waves per SIMD). accum_offset is the arch VGPR count rounded to 4
/// (.amdhsa_accum_offset), next_free_vgpr = accum_offset + AGPRs rounded to 8; waves per SIMD = floor(512 / next_free_vgpr) (at most 8).
pub fn unified_budget(max_v: u32, agprs: u32) -> (u32, u32, u32) {
    let t = crate::target::T;
    let (ag, tg, maxw) = match &t.regs.unified {
        Some(u) => (u.accum_granule, u.total_granule, u.max_waves_per_simd),
        None => (t.emit.vgpr_granule, t.regs.vgpr.alloc_granule, 1),
    };
    let accum = ((max_v + ag - 1) / ag * ag).max(t.emit.vgpr_min);
    let total = if agprs == 0 { accum } else { ((accum + agprs + tg - 1) / tg) * tg };
    let waves = (UNIFIED_FILE / total.max(1)).min(maxw).max(1);
    (accum, total, waves)
}

fn mark_used(used: &mut [bool], slot: usize, c: RClass) {
    used[slot] = true;
    for k in 1..c.dwords() as usize {
        used[slot + k] = true;
    }
}

fn free_phys(used_v: &mut [bool], used_s: &mut [bool], phys: u32, c: RClass) {
    let p = phys as usize;
    let table = if c.is_vector() { used_v } else { used_s };
    for k in 0..c.dwords() as usize {
        table[p + k] = false;
    }
}

fn find_slot(used: &[bool], c: RClass, singles_from_top: bool) -> Option<usize> {
    // A register tuple must start at a multiple of its own width: the ISA
    // requires 64-bit pairs at even indices and 128-bit tuples at multiples of
    // four.
    let width = c.dwords() as usize;
    let align = width;
    let lo = if c.is_vector() {
        if width == 1 {
            V_RESERVED
        } else {
            width
        }
    } else {
        S_RESERVED
    };
    let n = used.len();
    if singles_from_top && c.is_vector() && width < 16 {
        // Narrow operands pack from the top (aligned), leaving the low registers as dense
        // sixteen-dword blocks for the accumulators.
        let mut i = (n - width) / align * align;
        loop {
            if i >= lo && (0..width).all(|k| !used[i + k]) {
                return Some(i);
            }
            if i < align || i - align < lo {
                break;
            }
            i -= align;
        }
        return None;
    }
    let mut i = lo;
    while i + width <= n {
        if i % align == 0 && (0..width).all(|k| !used[i + k]) {
            return Some(i);
        }
        i += 1;
    }
    None
}
/// The MFMA instructions require the accumulator (D, C) and the multiplicands
/// (A, B) to live in disjoint register ranges; D and C may either be the same
/// range (in-place accumulation) or disjoint. Violating this silently produces
/// wrong results, so it is checked after allocation.
fn check_mfma_operand_aliasing(m: &MFunction, phys: &[u32]) -> Result<(), String> {
    fn range(phys: &[u32], m: &MFunction, r: VReg) -> (u32, u32) {
        let base = phys[r as usize];
        (base, base + m.vreg_class(r).dwords() as u32)
    }
    fn overlaps(a: (u32, u32), b: (u32, u32)) -> bool {
        a.0 < b.1 && b.0 < a.1
    }
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if inst.op == MOp::VSmfmac16x16x128Fp8 {
                // D (4) is the in-place accumulator; A (4) and B (8) must be aligned to their
                // width and disjoint from D; SRC2 is a single dword.
                let d = inst
                    .dst
                    .ok_or_else(|| format!("block {bi} inst {ii}: SMFMAC without destination"))?;
                let regs: Vec<VReg> = inst
                    .srcs
                    .iter()
                    .map(|s| match s {
                        Src::Reg(r) => Ok(*r),
                        _ => Err(format!("block {bi} inst {ii}: SMFMAC operands must be registers")),
                    })
                    .collect::<Result<_, String>>()?;
                if regs.len() < 3 {
                    return Err(format!("block {bi} inst {ii}: SMFMAC needs 3 source operands"));
                }
                let rd = range(phys, m, d);
                let ra = range(phys, m, regs[0]);
                let rb = range(phys, m, regs[1]);
                // gfx90a+ ISA: register tuples are 64-bit (even) aligned; no width alignment (the vendor FMHA's v_smfmac reads v[52:59])
                for (name, r) in [("A", ra), ("B", rb), ("D", rd)] {
                    let width = r.1 - r.0;
                    if r.0 % 2 != 0 {
                        return Err(format!(
                            "block {bi} inst {ii}: SMFMAC operand {name} at v{} is not aligned to its width {}",
                            r.0, width
                        ));
                    }
                }
                for (name, x, y) in [("A/B", ra, rb), ("A/D", ra, rd), ("B/D", rb, rd)] {
                    if overlaps(x, y) {
                        return Err(format!(
                            "block {bi} inst {ii}: SMFMAC operands {name} overlap (v{}..v{}, v{}..v{})",
                            x.0,
                            x.1 - 1,
                            y.0,
                            y.1 - 1
                        ));
                    }
                }
                continue;
            }
            // MFMA with a C operand (op_attrs: non-sparse, not zero-C)
            let is_mfma = crate::op_attrs::attrs(inst.op).mfma.map_or(false, |m| !m.sparse && !m.zero_c);
            if !is_mfma || inst.srcs.len() < 3 {
                continue;
            }
            let d = inst
                .dst
                .ok_or_else(|| format!("block {bi} inst {ii}: MFMA without destination"))?;
            let (a, b_src, c) = match (&inst.srcs[0], &inst.srcs[1], &inst.srcs[2]) {
                (Src::Reg(a), Src::Reg(b), Src::Reg(c)) => (*a, *b, *c),
                _ => return Err(format!("block {bi} inst {ii}: MFMA operands must be registers")),
            };
            let rd = range(phys, m, d);
            let ra = range(phys, m, a);
            let rb = range(phys, m, b_src);
            let rc = range(phys, m, c);
            // ISA (gfx90a+): operands are contiguous register tuples, 64-bit (even) aligned. The allocator places its own tuples at their width,
            // but the hardware needs only the even alignment: the vendor MXFP8 FMHA (aiter, public) runs v_mfma_scale ... v[52:67], v[20:27] /
            // v_mfma_f32_32x32x64_f8f6f4 v[116:131] (docs/fmha-replica.md R3 step 3).
            for (name, r) in [("A", ra), ("B", rb), ("C/D", rd)] {
                let width = r.1 - r.0;
                if r.0 % 2 != 0 {
                    return Err(format!(
                        "block {bi} inst {ii}: MFMA operand {name} at v{} is not aligned to its width {}",
                        r.0, width
                    ));
                }
            }
            // The accumulator is in-place (C == D on gfx9), so only A and B have
            // to stay clear of the destination.
            // C may be a different register than D (gfx950 `v_mfma* vdst, a, b, srcC`), but a partial overlap is illegal
            if rc != rd && overlaps(rc, rd) {
                return Err(format!(
                    "block {bi} inst {ii}: MFMA accumulator C must be identical to or disjoint from D (v{}..v{} vs v{}..v{})",
                    rc.0,
                    rc.1 - 1,
                    rd.0,
                    rd.1 - 1
                ));
            }
            let pairs = [("A/B", ra, rb), ("A/D", ra, rd), ("B/D", rb, rd)];
            for (name, x, y) in pairs {
                if overlaps(x, y) {
                    return Err(format!(
                        "block {bi} inst {ii}: MFMA operands {name} overlap (v{}..v{}, v{}..v{})",
                        x.0,
                        x.1 - 1,
                        y.0,
                        y.1 - 1
                    ));
                }
            }

        }
    }
    Ok(())
}
/// Per virtual register (canonical tuple): the instruction-position segments in which any of its dwords is live.
fn precise_segments(m: &MFunction, idx_of: &[Vec<usize>], succs: &[Vec<usize>]) -> Vec<Vec<(usize, usize)>> {
    use crate::sched::effects::{effects, Slot, SCC, VCC};
    let n = m.vregs.len();
    let nb = m.blocks.len();
    let keep = |s: &Slot| *s != VCC && *s != SCC;
    // reads / writes per instruction (slots), computed once
    let mut rw: Vec<Vec<(Vec<Slot>, Vec<Slot>)>> = Vec::with_capacity(nb);
    for b in &m.blocks {
        rw.push(
            b.insts
                .iter()
                .map(|i| {
                    let e = effects(m, i);
                    (e.reads.into_iter().filter(keep).collect(), e.writes.into_iter().filter(keep).collect())
                })
                .collect(),
        );
    }
    let mut gen: Vec<HashSet<Slot>> = vec![HashSet::new(); nb];
    let mut kill: Vec<HashSet<Slot>> = vec![HashSet::new(); nb];
    for bi in 0..nb {
        for (r, w) in &rw[bi] {
            for s in r {
                if !kill[bi].contains(s) {
                    gen[bi].insert(*s);
                }
            }
            for s in w {
                kill[bi].insert(*s);
            }
        }
    }
    let mut live_in: Vec<HashSet<Slot>> = gen.clone();
    let mut live_out: Vec<HashSet<Slot>> = vec![HashSet::new(); nb];
    loop {
        let mut changed = false;
        for bi in (0..nb).rev() {
            let mut out: HashSet<Slot> = HashSet::new();
            for s in &succs[bi] {
                out.extend(live_in[*s].iter().copied());
            }
            let mut inn = gen[bi].clone();
            for s in &out {
                if !kill[bi].contains(s) {
                    inn.insert(*s);
                }
            }
            if inn != live_in[bi] || out != live_out[bi] {
                live_in[bi] = inn;
                live_out[bi] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // Pruned liveness: a slot is only live where some definition of it can reach (forward may-defined sets). A slot that is
    // live without any reaching definition is a read of an uninitialised value on an infeasible path (for example two
    // correlated `if`s); keeping it live would only inflate the pressure.
    let mut def_in: Vec<HashSet<Slot>> = vec![HashSet::new(); nb];
    let mut def_out: Vec<HashSet<Slot>> = kill.clone();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nb];
    for (b, ss) in succs.iter().enumerate() {
        for &s in ss {
            preds[s].push(b);
        }
    }
    loop {
        let mut changed = false;
        for bi in 0..nb {
            let mut inn: HashSet<Slot> = HashSet::new();
            for &p in &preds[bi] {
                inn.extend(def_out[p].iter().copied());
            }
            let mut out = inn.clone();
            out.extend(kill[bi].iter().copied());
            if inn != def_in[bi] || out != def_out[bi] {
                def_in[bi] = inn;
                def_out[bi] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for bi in 0..nb {
        live_in[bi].retain(|s| def_in[bi].contains(s));
        live_out[bi].retain(|s| def_out[bi].contains(s));
    }
    let mut raw: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for (bi, b) in m.blocks.iter().enumerate() {
        if b.insts.is_empty() {
            continue;
        }
        let first = idx_of[bi][0];
        let last = *idx_of[bi].last().unwrap();
        let mut open: std::collections::HashMap<Slot, usize> = std::collections::HashMap::new();
        for s in &live_out[bi] {
            open.insert(*s, last);
        }
        for i in (0..b.insts.len()).rev() {
            let at = idx_of[bi][i];
            let (r, w) = &rw[bi][i];
            for s in w {
                let v = (*s / 16) as usize;
                match open.remove(s) {
                    Some(e) => raw[v].push((at, e)),
                    None => raw[v].push((at, at)),
                }
            }
            for s in r {
                open.entry(*s).or_insert(at);
                // a use keeps the position itself occupied even when the slot is also written here
                raw[(*s / 16) as usize].push((at, at));
            }
        }
        for (s, e) in open {
            if def_in[bi].contains(&s) {
                raw[(s / 16) as usize].push((first, e));
            } else {
                // a use with no reaching definition: keep just the use position
                raw[(s / 16) as usize].push((e, e));
            }
        }
    }
    // merge overlapping / touching segments per vreg
    for segs in raw.iter_mut() {
        segs.sort();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(segs.len());
        for &(a, b) in segs.iter() {
            match merged.last_mut() {
                Some(l) if a <= l.1 + 1 => l.1 = l.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        *segs = merged;
    }
    raw
}
/// Peak vector-register pressure (dwords, v0 included) from dword-level liveness: (peak, linear position, block index).
pub fn pressure_profile(m: &MFunction) -> (u32, usize, usize) {
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
    let label_to_block: std::collections::HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); m.blocks.len()];
    for (i, b) in m.blocks.iter().enumerate() {
        match &b.term {
            Term::Goto(t) => {
                if let Some(j) = label_to_block.get(t) {
                    succs[i].push(*j);
                }
            }
            Term::CondGoto { target, fall, .. } => {
                for t in [target, fall] {
                    if let Some(j) = label_to_block.get(t) {
                        succs[i].push(*j);
                    }
                }
            }
            Term::Endpgm => {}
        }
    }
    let segs = precise_segments(m, &idx_of, &succs);
    let mut ev: Vec<(usize, i64)> = Vec::new();
    for (r, sg) in segs.iter().enumerate() {
        let c = m.vreg_class(r as VReg);
        if !c.is_vector() || m.alias.contains_key(&(r as VReg)) {
            continue;
        }
        for (a, b) in sg {
            ev.push((*a, c.dwords() as i64));
            ev.push((b.saturating_add(1), -(c.dwords() as i64)));
        }
    }
    ev.sort();
    let (mut cur, mut peak, mut at) = (V_RESERVED as i64, 0i64, 0usize);
    for (p, d) in ev {
        cur += d;
        if cur > peak {
            peak = cur;
            at = p;
        }
    }
    let blk = idx_of.iter().position(|v| v.last().map_or(false, |l| *l >= at) && v.first().map_or(false, |f| *f <= at)).unwrap_or(0);
    (peak as u32, at, blk)
}
/// Physical reaching-writer check (the same analysis as tests/mir_invariants.rs, made PATH-AWARE): a register dword read must see the write of the
/// virtual register dword it names; a foreign value (another vreg placed on the same physical register) must not have overwritten it on any path
/// on which the reader's own value had been written.
///
/// The state per physical register maps the CURRENT writer to the set of writers that preceded it on the paths reaching here ("history"). A read of
/// slot `r` is a violation when
///  * no path delivers `r` at all but a foreign value reaches it (the own value never arrives), or
///  * some foreign current writer has `r` in its history: `r` was written, then overwritten by the foreign value, on one and the same path (a clobber).
/// A foreign writer whose paths never wrote `r` is NOT a clobber: on those paths the read is of an undefined value whatever the allocation is (typical
/// shape: a loop-carried value defined inside the loop body and read after it, with the zero-trip path through the loop test - the precise-liveness
/// packing deliberately lets another value use the register there). This is exactly the case the flat "set of reaching writers" test rejected, which
/// made the two-phase packing fail on any persistent loop-carried slot (`FMHA_OPT=constres1..6`).
/// Returns the violations; used to guard the hole-filling packing and for allocator experiments.
pub fn check_reaching(m: &MFunction, a: &Alloc) -> Vec<String> {
    use crate::sched::effects::{effects, Slot, SCC, VCC};
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    // physical register -> current writer -> writers that preceded it on the paths ending in that writer
    type Writers = HashMap<(bool, u32), BTreeMap<Slot, BTreeSet<Slot>>>;
    let phys = |slot: Slot| -> (bool, u32) {
        let t = slot / 16;
        (m.vreg_class(t).is_vector(), a.phys[t as usize] + (slot - t * 16))
    };
    let write = |st: &mut Writers, w: Slot| {
        let e = st.entry(phys(w)).or_default();
        let mut hist: BTreeSet<Slot> = BTreeSet::new();
        for (cur, prev) in e.iter() {
            hist.insert(*cur);
            hist.extend(prev.iter().copied());
        }
        e.clear();
        e.insert(w, hist);
    };
    let succs = crate::sched::liveness::successors(m);
    let nb = m.blocks.len();
    let mut in_st: Vec<Option<Writers>> = vec![None; nb];
    in_st[0] = Some(Writers::new());
    for _ in 0..500 {
        let mut changed = false;
        for b in 0..nb {
            let Some(mut st) = in_st[b].clone() else { continue };
            for inst in &m.blocks[b].insts {
                for w in &effects(m, inst).writes {
                    if *w != VCC && *w != SCC {
                        write(&mut st, *w);
                    }
                }
            }
            for &s in &succs[b] {
                match &mut in_st[s] {
                    None => {
                        in_st[s] = Some(st.clone());
                        changed = true;
                    }
                    Some(x) => {
                        for (k, v) in &st {
                            let e = x.entry(*k).or_default();
                            for (cur, prev) in v {
                                let slot = e.entry(*cur).or_insert_with(|| {
                                    changed = true;
                                    BTreeSet::new()
                                });
                                for p in prev {
                                    changed |= slot.insert(*p);
                                }
                            }
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    let mut report = Vec::new();
    // where each slot is written (diagnostics: names the instructions behind a reported writer)
    let mut wloc: HashMap<Slot, Vec<(usize, usize, MOp)>> = HashMap::new();
    for (b, blk) in m.blocks.iter().enumerate() {
        for (ii, inst) in blk.insts.iter().enumerate() {
            for w in &effects(m, inst).writes {
                wloc.entry(*w).or_default().push((b, ii, inst.op));
            }
        }
    }
    for b in 0..nb {
        let Some(mut st) = in_st[b].clone() else { continue };
        for (ii, inst) in m.blocks[b].insts.iter().enumerate() {
            let e = effects(m, inst);
            for r in &e.reads {
                if *r == VCC || *r == SCC {
                    continue;
                }
                if e.writes.contains(r) && crate::op_attrs::attrs(inst.op).regfx == crate::op_attrs::RegFx::PartialWrite {
                    continue;
                }
                let p = phys(*r);
                if let Some(ws) = st.get(&p) {
                    let own_arrives = ws.contains_key(r);
                    let clobbered: Vec<Slot> = ws.iter().filter(|(w, prev)| *w != r && prev.contains(r)).map(|(w, _)| *w).collect();
                    if !own_arrives || !clobbered.is_empty() {
                        let list: Vec<Slot> = if !clobbered.is_empty() { clobbered.clone() } else { ws.keys().copied().collect() };
                        let at: Vec<String> = list.iter().map(|w| format!("{w}={:?}", wloc.get(w).map(|v| v.iter().take(3).collect::<Vec<_>>()).unwrap_or_default())).collect();
                        report.push(format!(
                            "block {b} inst {ii} {:?}: reads slot {r} ({}{}) but {} [{}]",
                            inst.op,
                            if p.0 { "v" } else { "s" },
                            p.1,
                            if !clobbered.is_empty() { "it was overwritten by another value on a path where it had been written, writers" } else { "its own value never arrives; the reaching writers are" },
                            at.join("; ")
                        ));
                    }
                }
            }
            for w in &e.writes {
                if *w != VCC && *w != SCC {
                    write(&mut st, *w);
                }
            }
        }
    }
    if let Ok(path) = std::env::var("AMDC_RA_DEBUG_FILE") {
        let _ = std::fs::write(path, report.join("\n"));
    }
    report
}

/// The register plan as the allocator consumes it (regplan.rs checks the result).
struct Plan {
    active: bool,
    pins: HashMap<VReg, u32>,
    /// vreg -> the vregs it must never share registers with (separate groups and rings)
    sep: HashMap<VReg, Vec<VReg>>,
    /// vreg -> the vregs it should share registers with (hint)
    share: HashMap<VReg, Vec<VReg>>,
    /// vreg -> (rotation name, its residue-class partners, its position): the class shares ONE register tuple (hard)
    rot: HashMap<VReg, (String, Vec<VReg>, usize)>,
    names: HashMap<VReg, String>,
}

impl Plan {
    fn new(m: &MFunction) -> Plan {
        let h = &m.reg_hints;
        let mut p = Plan { active: h.active(), pins: HashMap::new(), sep: HashMap::new(), share: HashMap::new(), rot: HashMap::new(), names: h.names.clone() };
        if !p.active {
            return p;
        }
        for (r, b) in &h.pins_v {
            p.pins.insert(m.canon(*r), *b);
        }
        for g in h.separate.iter().chain(h.rings.iter().map(|x| &x.1)) {
            for a in g {
                for b in g {
                    if m.canon(*a) != m.canon(*b) {
                        p.sep.entry(m.canon(*a)).or_default().push(m.canon(*b));
                    }
                }
            }
        }
        for (name, bufs, frags) in &h.rotations {
            let bufs = (*bufs).max(1) as usize;
            for (i, &a) in frags.iter().enumerate() {
                if a == u32::MAX {
                    continue;
                }
                let a = m.canon(a);
                let mut class: Vec<VReg> = frags.iter().enumerate().filter(|(j, x)| j % bufs == i % bufs && **x != u32::MAX && *j != i).map(|(_, x)| m.canon(*x)).filter(|x| *x != a).collect();
                class.dedup();
                p.rot.entry(a).or_insert((name.clone(), class, i));
                // the classes never share registers
                for (j, &b) in frags.iter().enumerate() {
                    if b != u32::MAX && j % bufs != i % bufs && m.canon(b) != a {
                        let e = p.sep.entry(a).or_default();
                        if !e.contains(&m.canon(b)) {
                            e.push(m.canon(b));
                        }
                    }
                }
            }
        }
        for g in &h.share {
            for a in g {
                for b in g {
                    if m.canon(*a) != m.canon(*b) {
                        p.share.entry(m.canon(*a)).or_default().push(m.canon(*b));
                    }
                }
            }
        }
        p
    }
    fn pin(&self, r: VReg) -> Option<u32> {
        if self.active {
            self.pins.get(&r).copied()
        } else {
            None
        }
    }
    /// `r` and its rotation-class partners (just `r` outside a rotation).
    fn rot_class(&self, r: VReg) -> Vec<VReg> {
        match self.rot.get(&r) {
            Some((_, class, _)) if self.active => std::iter::once(r).chain(class.iter().copied()).collect(),
            _ => vec![r],
        }
    }
    /// (rotation name, fragment position) of a rotation member.
    fn rot_pos(&self, r: VReg) -> (String, usize) {
        self.rot.get(&r).map(|(n, _, p)| (n.clone(), *p)).unwrap_or_default()
    }
    /// The registers a rotation partner of `r` already holds (registers, partner, rotation name, r's fragment position).
    fn rot_target(&self, r: VReg, phys: &[u32]) -> Option<(u32, VReg, String, usize)> {
        if !self.active {
            return None;
        }
        let (name, class, pos) = self.rot.get(&r)?;
        class.iter().find(|q| phys[**q as usize] != u32::MAX).map(|q| (phys[*q as usize], *q, name.clone(), *pos))
    }
    fn name(&self, r: VReg) -> String {
        self.names.get(&r).cloned().unwrap_or_else(|| format!("r{r}"))
    }
    /// VGPR slots `r` (live [s, e]) may not use: other pins live at the same time, assigned separate partners.
    #[allow(clippy::too_many_arguments)]
    fn forbidden(&self, r: VReg, s: usize, e: usize, start: &[usize], end: &[usize], phys: &[u32], m: &MFunction) -> Vec<bool> {
        let mut f = vec![false; VGPR_COUNT];
        if !self.active {
            return f;
        }
        let mut block = |base: u32, w: u32| {
            for k in base..(base + w).min(VGPR_COUNT as u32) {
                f[k as usize] = true;
            }
        };
        for (&p, &b) in &self.pins {
            if p != r && start[p as usize] != usize::MAX && start[p as usize] <= e && s <= end[p as usize] && m.vreg_class(p).is_vector() {
                block(b, m.vreg_class(p).dwords());
            }
        }
        if let Some(ps) = self.sep.get(&r) {
            for &q in ps {
                let pq = phys[q as usize];
                if pq != u32::MAX && m.vreg_class(q).is_vector() {
                    block(pq, m.vreg_class(q).dwords());
                }
            }
        }
        f
    }
    /// Linear-scan slot for `r` under the plan: the exact pin, else a free share partner's registers, else the first free slot avoiding the
    /// forbidden ranges. Err for a pin that cannot be honoured.
    #[allow(clippy::too_many_arguments)]
    fn slot(&self, r: VReg, c: RClass, s: usize, e: usize, used: &[bool], start: &[usize], end: &[usize], phys: &[u32], m: &MFunction, from_top: bool) -> Result<Option<usize>, String> {
        let forbid = self.forbidden(r, s, e, start, end, phys, m);
        let mask: Vec<bool> = used.iter().zip(&forbid).map(|(a, b)| *a || *b).collect();
        let w = c.dwords() as usize;
        let free = |i: usize| i % w == 0 && i + w <= VGPR_COUNT && (i..i + w).all(|k| !mask[k]);
        if let Some(b) = self.pins.get(&r) {
            let b = *b as usize;
            // an explicit pin needs only the ISA's even tuple alignment (see the TwoPhase path)
            if b % 2 == 0 && b + w <= VGPR_COUNT && (b..b + w).all(|k| !mask[k]) {
                return Ok(Some(b));
            }
            return Err(format!("register plan: pin of {} ({c:?}) at v{b} cannot be honoured: the registers are taken by a value live at the same time or by a separate partner", self.name(r)));
        }
        // a rotation class shares one register tuple: the first member placed decides, every later member must take it
        if let Some((name, class, pos)) = self.rot.get(&r) {
            for &q in class {
                let pq = phys[q as usize];
                if pq == u32::MAX {
                    continue;
                }
                if m.vreg_class(q) != c {
                    return Err(format!("register plan: rotation `{name}`: {} ({c:?}, fragment {pos}) and {} ({:?}) are in one buffer but have different register classes", self.name(r), self.name(q), m.vreg_class(q)));
                }
                if start[q as usize] <= e && s <= end[q as usize] {
                    return Err(format!(
                        "register plan: rotation `{name}`: {} (fragment {pos}, live {s}..{e}) and {} (live {}..{}) share a buffer but are live at the same time (a read ran ahead of the previous fragment's last use)",
                        self.name(r), self.name(q), start[q as usize], end[q as usize]
                    ));
                }
                if free(pq as usize) {
                    return Ok(Some(pq as usize));
                }
                return Err(format!("register plan: rotation `{name}`: the buffer of {} (fragment {pos}) at v{pq} is taken by another value live at the same time", self.name(r)));
            }
        }
        if let Some(ps) = self.share.get(&r) {
            for &q in ps {
                let pq = phys[q as usize];
                if pq != u32::MAX && m.vreg_class(q) == c && free(pq as usize) {
                    return Ok(Some(pq as usize));
                }
            }
        }
        Ok(find_slot(&mask, c, from_top))
    }
}

/// The launch-time register budget of a workgroup: its waves are spread over the 4 SIMDs of a CU, so ceil(waves / 4) of them share one
/// SIMD's 512-entry (unified VGPR + AGPR) file; each wave allocates its register count in granules of 8. A kernel over that budget assembles
/// and verifies but fails to launch (`HSA_STATUS_ERROR_INVALID_ISA`). Err with an explanation when `m` does not fit.
pub fn check_workgroup_budget(m: &crate::mir::MFunction, a: &Alloc) -> Result<(), String> {
    let n_agpr = agpr_count(m, a);
    let (_, total, _) = unified_budget(a.max_v, n_agpr);
    let t = crate::target::T;
    let (g, simds) = match &t.regs.unified {
        Some(u) => (u.total_granule, u.simds),
        None => (t.regs.vgpr.alloc_granule, 4),
    };
    let lanes = t.wave.lanes;
    let per_wave = (total + g - 1) / g * g;
    let waves = (m.workgroup_size + lanes - 1) / lanes;
    let per_simd = ((waves + simds - 1) / simds).max(1);
    let limit = UNIFIED_FILE / per_simd / g * g;
    if per_wave > limit {
        return Err(format!(
            "register budget: workgroup_size {} = {waves} waves -> {per_simd} per SIMD, so each wave may use at most {limit} registers (VGPR + AGPR, \
             granule 8), but `{}` needs {per_wave} ({} VGPRs{}); the kernel would fail to launch (HSA_STATUS_ERROR_INVALID_ISA). Use fewer threads \
             per workgroup or fewer live registers",
            m.workgroup_size,
            m.name,
            a.max_v,
            if n_agpr > 0 { format!(" + {n_agpr} AGPRs") } else { String::new() }
        ));
    }
    Ok(())
}
