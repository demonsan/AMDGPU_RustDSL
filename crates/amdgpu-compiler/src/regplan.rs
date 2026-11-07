//! Register planning (the weaver's step 3, docs/weaver-regs.md): the post-allocation check of the kernel's register plan
//! (`MFunction::reg_hints`, from the DSL `RegMark`s) and the per-region pressure report measured on the REAL allocation.
//!
//! * pins (`pin_vgpr_at`): the tuple sits exactly at its base;
//! * separate groups / rings: no two members overlap in physical registers (at any time);
//! * caps: at every instruction of the region (nested regions included) the arch-VGPR footprint (highest live physical VGPR + 1) and the AGPR
//!   footprint stay within the cap;
//! * share groups are hints: the report says which members ended up in the same registers.

use crate::mir::*;
use crate::regalloc::Alloc;
use std::collections::{HashMap, HashSet};

/// The peak of one region (allocator truth).
#[derive(Clone, Debug, Default)]
pub struct RegionPeak {
    pub name: String,
    /// instructions tagged with the region (or a nested one)
    pub insts: usize,
    /// largest number of live arch-VGPR dwords
    pub vgpr_live_peak: u32,
    /// largest (highest live physical arch VGPR + 1): what the region needs from the file
    pub vgpr_footprint_peak: u32,
    /// (block, instruction) of the footprint peak and the values live there (DSL names where known), highest register first
    pub at: (usize, usize),
    pub live_at_peak: Vec<String>,
    pub agpr_footprint_peak: u32,
}

/// Canonical tuple vreg, physical base and width of a live vreg; `None` for scalars and unallocated vregs.
fn place(m: &MFunction, a: &Alloc, r: VReg) -> Option<(bool, u32, u32)> {
    let c = m.canon(r);
    let cls = m.vreg_class(c);
    if cls.is_scalar() || a.scalar.get(c as usize).copied().unwrap_or(false) {
        return None;
    }
    let p = a.phys[c as usize];
    let w = cls.dwords();
    if is_agpr_phys(p) {
        Some((true, p - AGPR_TAG, w))
    } else {
        Some((false, p, w))
    }
}

/// Block-level liveness of canonical vregs (live-out sets).
fn live_out(m: &MFunction) -> Vec<HashSet<VReg>> {
    let succs = crate::sched::liveness::successors(m);
    let n = m.blocks.len();
    let mut uses: Vec<HashSet<VReg>> = vec![HashSet::new(); n];
    let mut defs: Vec<HashSet<VReg>> = vec![HashSet::new(); n];
    for (i, b) in m.blocks.iter().enumerate() {
        for inst in &b.insts {
            for r in inst.uses() {
                let r = m.canon(r);
                if !defs[i].contains(&r) {
                    uses[i].insert(r);
                }
            }
            if let Some(d) = inst.dst {
                defs[i].insert(m.canon(d));
            }
        }
    }
    let mut lin: Vec<HashSet<VReg>> = vec![HashSet::new(); n];
    let mut lout: Vec<HashSet<VReg>> = vec![HashSet::new(); n];
    loop {
        let mut changed = false;
        for i in (0..n).rev() {
            let mut out = HashSet::new();
            for s in &succs[i] {
                out.extend(lin[*s].iter().copied());
            }
            let mut inn = uses[i].clone();
            inn.extend(out.iter().filter(|r| !defs[i].contains(r)).copied());
            if inn != lin[i] || out != lout[i] {
                lin[i] = inn;
                lout[i] = out;
                changed = true;
            }
        }
        if !changed {
            return lout;
        }
    }
}

/// Region ids an instruction counts for: its own region and every enclosing one.
fn chain(m: &MFunction, id: u16) -> Vec<u16> {
    let mut v = Vec::new();
    let mut k = id;
    while k != 0 && (k as usize) <= m.reg_hints.regions.len() {
        v.push(k);
        k = m.reg_hints.regions[k as usize - 1].1;
    }
    v
}

/// Per-region peaks of the allocated kernel (regions in declaration order; regions with the same name are merged).
pub fn region_report(m: &MFunction, a: &Alloc) -> Vec<RegionPeak> {
    let regions = &m.reg_hints.regions;
    let mut by_name: Vec<RegionPeak> = Vec::new();
    let mut idx_of: HashMap<String, usize> = HashMap::new();
    for (n, _) in regions {
        if !idx_of.contains_key(n) {
            idx_of.insert(n.clone(), by_name.len());
            by_name.push(RegionPeak { name: n.clone(), ..Default::default() });
        }
    }
    if regions.is_empty() {
        return by_name;
    }
    let lout = live_out(m);
    for (bi, b) in m.blocks.iter().enumerate() {
        let mut live: HashSet<VReg> = lout[bi].clone();
        for ii in (0..b.insts.len()).rev() {
            let inst = &b.insts[ii];
            // registers occupied while the instruction executes: what is live after it, its destination and its sources
            let mut occ: Vec<VReg> = live.iter().copied().collect();
            if let Some(d) = inst.dst {
                occ.push(m.canon(d));
            }
            for r in inst.uses() {
                occ.push(m.canon(r));
            }
            if inst.region != 0 {
                occ.sort_unstable();
                occ.dedup();
                let (mut vfoot, mut vlive, mut afoot) = (0u32, 0u32, 0u32);
                let mut seen: HashSet<(bool, u32)> = HashSet::new();
                for &r in &occ {
                    if let Some((ag, p, w)) = place(m, a, r) {
                        if !seen.insert((ag, p)) {
                            continue;
                        }
                        if ag {
                            afoot = afoot.max(p + w);
                        } else {
                            vfoot = vfoot.max(p + w);
                            vlive += w;
                        }
                    }
                }
                for id in chain(m, inst.region) {
                    let k = idx_of[&regions[id as usize - 1].0];
                    let rp = &mut by_name[k];
                    rp.insts += 1;
                    rp.vgpr_live_peak = rp.vgpr_live_peak.max(vlive);
                    rp.agpr_footprint_peak = rp.agpr_footprint_peak.max(afoot);
                    if vfoot > rp.vgpr_footprint_peak {
                        rp.vgpr_footprint_peak = vfoot;
                        rp.at = (bi, ii);
                        let mut l: Vec<(u32, String)> = occ
                            .iter()
                            .filter_map(|&r| place(m, a, r).filter(|x| !x.0).map(|(_, p, w)| (p, format!("{}@v{}{}", m.reg_hints.names.get(&r).cloned().unwrap_or_else(|| format!("r{r}")), p, if w > 1 { format!("..v{}", p + w - 1) } else { String::new() }))))
                            .collect();
                        l.sort_by(|x, y| y.0.cmp(&x.0));
                        l.dedup();
                        rp.live_at_peak = l.into_iter().map(|x| x.1).collect();
                    }
                }
            }
            if let Some(d) = inst.dst {
                live.remove(&m.canon(d));
            }
            for r in inst.uses() {
                live.insert(m.canon(r));
            }
        }
    }
    by_name
}

fn overlap(a: (bool, u32, u32), b: (bool, u32, u32)) -> bool {
    a.0 == b.0 && a.1 < b.1 + b.2 && b.1 < a.1 + a.2
}

/// The plan's report lines (honoured / not) without failing.
pub fn report(m: &MFunction, a: &Alloc) -> Vec<String> {
    let h = &m.reg_hints;
    let name = |r: VReg| h.names.get(&m.canon(r)).cloned().unwrap_or_else(|| format!("r{r}"));
    let at = |r: VReg| match place(m, a, r) {
        Some((ag, p, w)) => format!("{}{}{}", if ag { "a" } else { "v" }, p, if w > 1 { format!("..{}", p + w - 1) } else { String::new() }),
        None => "-".to_string(),
    };
    let mut out = Vec::new();
    for (r, b) in &h.pins_v {
        out.push(format!("pin {} at v{b}: {} ({})", name(*r), at(*r), if place(m, a, *r).map_or(false, |x| !x.0 && x.1 == *b) { "honoured" } else { "VIOLATED" }));
    }
    for g in &h.separate {
        out.push(format!("separate [{}]", g.iter().map(|r| format!("{}={}", name(*r), at(*r))).collect::<Vec<_>>().join(", ")));
    }
    for (n, g) in &h.rings {
        out.push(format!("ring {n} [{}]", g.iter().map(|r| format!("{}={}", name(*r), at(*r))).collect::<Vec<_>>().join(", ")));
    }
    for g in &h.share {
        let shared = g.windows(2).all(|w| a.phys[m.canon(w[0]) as usize] == a.phys[m.canon(w[1]) as usize]);
        out.push(format!("share [{}]: {}", g.iter().map(|r| format!("{}={}", name(*r), at(*r))).collect::<Vec<_>>().join(", "), if shared { "shared" } else { "not shared (lifetimes overlap or the registers were taken)" }));
    }
    for u in &h.unresolved {
        out.push(format!("unresolved target {u} (no register: unused or folded value)"));
    }
    out
}

/// Check the hard parts of the plan on an allocation.
pub fn check(m: &MFunction, a: &Alloc) -> Result<(), String> {
    let h = &m.reg_hints;
    let name = |r: VReg| h.names.get(&m.canon(r)).cloned().unwrap_or_else(|| format!("r{r}"));
    for (r, b) in &h.pins_v {
        match place(m, a, *r) {
            Some((false, p, _)) if p == *b => {}
            Some((ag, p, _)) => return Err(format!("register plan: {} pinned at v{b} but allocated at {}{p}", name(*r), if ag { "a" } else { "v" })),
            None => {}
        }
    }
    for g in h.separate.iter().chain(h.rings.iter().map(|x| &x.1)) {
        for (i, x) in g.iter().enumerate() {
            for y in &g[i + 1..] {
                if let (Some(px), Some(py)) = (place(m, a, *x), place(m, a, *y)) {
                    if m.canon(*x) != m.canon(*y) && overlap(px, py) {
                        return Err(format!("register plan: {} and {} must be separate but share registers", name(*x), name(*y)));
                    }
                }
            }
        }
    }
    if !h.caps.is_empty() {
        let peaks = region_report(m, a);
        for (region, vcap, acap) in &h.caps {
            let Some(p) = peaks.iter().find(|p| &p.name == region) else {
                return Err(format!("register plan: cap on unknown region `{region}` (no weave region / stage of that name has instructions)"));
            };
            if let Some(c) = vcap {
                if p.vgpr_footprint_peak > *c {
                    return Err(format!(
                        "register cap: region `{region}` needs {} arch VGPRs at block {} inst {} (cap {c}); live there (highest first): {}",
                        p.vgpr_footprint_peak,
                        p.at.0,
                        p.at.1,
                        p.live_at_peak.iter().take(24).cloned().collect::<Vec<_>>().join(" ")
                    ));
                }
            }
            if let Some(c) = acap {
                if p.agpr_footprint_peak > *c {
                    return Err(format!("register cap: region `{region}` needs {} AGPRs (cap {c})", p.agpr_footprint_peak));
                }
            }
        }
    }
    Ok(())
}
