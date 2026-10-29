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
    /// Per vreg: lives in an SGPR (so a VALU source operand prints `sN`).
    pub scalar: Vec<bool>,
}

const VGPR_COUNT: usize = 256;
const SGPR_COUNT: usize = 104;

/// Reserved: v0 holds workitem id; s0:s1 kernarg pointer; s2..s4 workgroup ids.
const V_RESERVED: usize = 1;
const S_RESERVED: usize = 8;

pub fn allocate(m: &MFunction) -> Result<Alloc, String> {
    // AMDC_RA_FORCE_TWOPHASE=1: use the precise-liveness packing first (testing; it is normally the last resort)
    if std::env::var("AMDC_RA_FORCE_TWOPHASE").map(|v| !v.is_empty() && v != "0").unwrap_or(false) {
        if let Ok(a) = allocate_two_phase(m) {
            return Ok(a);
        }
    }
    // First try the compact policy. If the kernel runs out of registers only because wide
    // tuples cannot find an aligned hole between scattered singles, retry with the singles
    // packed from the top of the file so the low registers stay available for tuples.
    match allocate_with(m, false) {
        Err(e) if e.contains("out of VGPRs") => match allocate_with(m, true) {
            // last resort: offline two-phase packing (wide tuples first, singles into the remaining holes in time)
            Err(e2) if e2.contains("out of VGPRs") => allocate_two_phase(m).map_err(|e3| if e3.contains("out of VGPRs") { format!("{e2}  ||  {e3}") } else { e3 }),
            other => other,
        },
        other => other,
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
        order.sort_by_key(|x| (std::cmp::Reverse(x.3.dwords()), std::cmp::Reverse(seg_len(x.2)), x.0));
        for (s, e, r, c) in order {
            let my: Vec<(usize, usize)> = if segs[r as usize].is_empty() { vec![(s, e)] } else { segs[r as usize].clone() };
            let width = c.dwords() as usize;
            let lo = if width == 1 { V_RESERVED } else { width };
            let mut found = None;
            let mut i = lo;
            while i + width <= VGPR_COUNT {
                if i % width == 0 && (0..width).all(|k| occ[i + k].iter().all(|(os, oe)| my.iter().all(|(ms, me)| *oe < *ms || *me < *os))) {
                    found = Some(i);
                    break;
                }
                i += 1;
            }
            let slot = found.ok_or_else(|| {
                // numeric lower bound: the largest number of dwords live at one position
                let mut ev: Vec<(usize, i64)> = Vec::new();
                for (s2, e2, _, c2) in &vregs {
                    ev.push((*s2, c2.dwords() as i64));
                    ev.push((e2.saturating_add(1), -(c2.dwords() as i64)));
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
        let slot = find_slot(&used_v, c, singles_from_top).ok_or_else(|| {
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
        if !c.is_vector() {
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

    // Sub-register views live inside their tuple.
    for (a, (t, off)) in &m.alias {
        phys[*a as usize] = phys[*t as usize] + *off as u32;
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

    Ok(Alloc { phys: patched, max_v: max_v as u32, max_s: max_s as u32, scalar: m.vregs.iter().map(|c| !c.is_vector()).collect() })
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
                for (name, r) in [("A", ra), ("B", rb), ("D", rd)] {
                    let width = r.1 - r.0;
                    if r.0 % width != 0 || r.0 % 2 != 0 {
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
            let is_mfma = matches!(
                inst.op,
                MOp::VMfma16x16x16F16
                    | MOp::VMfma32x32x8F16
                    | MOp::VMfma16x16x32Fp8
                    | MOp::VMfma32x32x16Fp8
                    | MOp::VMfmaScale32x32x64Fp8
            );
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
            // ISA: operands are contiguous and the base register is aligned to
            // the operand width (and at least even aligned).
            for (name, r) in [("A", ra), ("B", rb), ("C/D", rd)] {
                let width = r.1 - r.0;
                if r.0 % width != 0 || r.0 % 2 != 0 {
                    return Err(format!(
                        "block {bi} inst {ii}: MFMA operand {name} at v{} is not aligned to its width {}",
                        r.0, width
                    ));
                }
            }
            // The accumulator is in-place (C == D on gfx9), so only A and B have
            // to stay clear of the destination.
            if rc != rd {
                return Err(format!(
                    "block {bi} inst {ii}: MFMA accumulator C must be in place with D (v{}..v{} vs v{}..v{})",
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
            raw[(s / 16) as usize].push((first, e));
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