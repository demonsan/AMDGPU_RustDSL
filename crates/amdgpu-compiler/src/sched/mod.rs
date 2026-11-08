//! Instruction scheduling framework (own compiler; LLVM's schedulers work per basic block only).
//! See docs/scheduling.md.
//!
//! Pipeline position: after lowering, before hazards/RA. The pass
//! 1. strips the waits that lowering inserted (`MInst::auto`),
//! 2. reorders each region (stage 1: basic blocks) with the list scheduler, every result proven
//!    legal by the independent verifier,
//! 3. re-inserts waits from a whole-function scoreboard over the final order.
//! Default OFF (`--sched` / `AMDC_SCHED=1`).

pub mod cfg;
pub mod depgraph;
pub mod effects;
pub mod hoist;
pub mod list;
pub mod liveness;
pub mod m0;
pub mod model;
pub mod order;
pub mod ldahead;
pub mod phase;
pub mod region;
pub mod slotfill;
pub mod sim;
pub mod verify;
pub mod waits;

use crate::mir::*;

#[derive(Clone, Debug)]
pub struct Options {
    /// Reorder instructions (false: only strip and re-insert waits, "rewait" mode).
    pub reorder: bool,
    /// Let non-memory instructions cross `s_barrier` (memory ops never do).
    pub alu_across_barrier: bool,
    /// Remove empty latch blocks / elide fall-through branches (default on with the scheduler).
    pub cfg_cleanup: bool,
    /// Pressure limit in VGPR dwords (never below the original peak of the block).
    pub max_vgprs: u32,
    /// Run the verifiers (always on in tests).
    pub verify: bool,
    pub stats: bool,
    /// `ldshoist`: hoist `ds_read` above independent instructions (`hoist=N`: max distance).
    pub ldshoist: bool,
    pub hoist_dist: usize,
    /// `slotfill`: the slot-filling scheduler (sched/slotfill.rs): MFMAs are the skeleton, the other instructions are spread over their deadline windows
    pub slotfill: bool,
    /// `ldsink=D` (sched/ldahead.rs): the opposite direction - sink every ds_read down until at most D MFMAs sit between it and its first consumer (0 = off)
    pub ldsink: usize,
    /// `ldahead32=D` (sched/ldahead.rs): post-pass moving every 32-bit scale-word read (`ds_read_b32`) up until D MFMAs sit between it and its first consumer (0 = off)
    pub ldahead32: usize,
    /// `wsink` (sched/ldahead.rs): sink the kernel's EXPLICIT `s_waitcnt lgkmcnt(n)` down past the compute that does not need them, so the MFMAs in front of a closing `wait_lds(0)` get the counted waits of the scoreboard
    pub waitsink: bool,
    /// Weave regions (`regions[id - 1]`, `MInst::region` = id): per-region strategies, see `sched::region`. Empty = the scheduler is byte-identical to the region-less one.
    pub regions: Vec<region::RegionSpec>,
    /// Lowered region id (`MInst::region`) -> strategy id (`regions[id - 1]`; 0 = the kernel default); filled by the compiler from the region names. Empty: the ids are the strategy ids.
    pub region_of: Vec<u16>,
    /// `hzfill` (hzfill.rs): post-RA filling of the hazard `s_nop` windows with independent instructions from below
    pub hzfill: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { reorder: true, alu_across_barrier: false, cfg_cleanup: true, max_vgprs: 240, verify: true, stats: false, ldshoist: false, hoist_dist: 64, slotfill: false, ldsink: 0, ldahead32: 0, waitsink: false, regions: Vec::new(), region_of: Vec::new(), hzfill: false }
    }
}

impl Options {
    /// Debug / bisect override: `AMDC_SCHED=1` (full scheduler) or `ld` (waits + LDS-read hoisting), `AMDC_SCHED_OPTS=rewait,xbar,noverify,stats,limit=N,..`
    /// tunes it (`AMDC_SCHED=0`: no scheduler).
    pub fn from_env() -> Option<Options> {
        let mode = std::env::var("AMDC_SCHED").ok();
        if !matches!(mode.as_deref(), Some("1") | Some("ld")) {
            return None;
        }
        let mut o = Options::default();
        if mode.as_deref() == Some("ld") {
            // preset: scoreboard waits + LDS-read hoisting, no instruction reordering
            o.reorder = false;
            o.ldshoist = true;
        }
        if let Ok(s) = std::env::var("AMDC_SCHED_OPTS") {
            o.apply_tokens(&s);
        }
        Some(o)
    }

    /// Apply comma-separated scheduler option tokens (AMDC_SCHED_OPTS vocabulary; also Function::sched_opts).
    pub fn apply_tokens(&mut self, s: &str) {
        let o = self;
        for t in s.split(',') {
                let t = t.trim();
                match t {
                    "rewait" => o.reorder = false,
                    "reorder" => o.reorder = true,
                    "xbar" => o.alu_across_barrier = true,
                    "noverify" => o.verify = false,
                    "stats" => o.stats = true,
                    "nocfg" => o.cfg_cleanup = false,
                    "ldshoist" => o.ldshoist = true,
                    "slotfill" => o.slotfill = true,
                    "wsink" => o.waitsink = true,
                    "hzfill" => o.hzfill = true,
                    _ => {
                        if let Some(v) = t.strip_prefix("ldahead32=") {
                            if let Ok(n) = v.parse() {
                                o.ldahead32 = n;
                            }
                        }
                        if let Some(v) = t.strip_prefix("ldsink=") {
                            if let Ok(n) = v.parse() {
                                o.ldsink = n;
                            }
                        }
                        if let Some(v) = t.strip_prefix("hoist=") {
                            if let Ok(n) = v.parse() {
                                o.hoist_dist = n;
                                o.ldshoist = true;
                            }
                        }
                        if let Some(v) = t.strip_prefix("limit=") {
                            if let Ok(n) = v.parse() {
                                o.max_vgprs = n;
                            }
                        }
                    }
                }
            }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub blocks: usize,
    pub scheduled: usize,
    pub skipped: usize,
    pub moved: usize,
    pub waits_inserted: usize,
    pub est_before: u64,
    pub est_after: u64,
    pub peak_before: i64,
    pub peak_after: i64,
}

fn apply_perm(insts: &[MInst], perm: &[usize]) -> Vec<MInst> {
    perm.iter().map(|&k| insts[k].clone()).collect()
}


/// Reorder every basic block with the list scheduler (verified).
pub(crate) fn schedule_blocks(m: &mut MFunction, o: &Options, stats: &mut Stats) -> Result<(), String> {
    if !o.reorder && !o.any_region(|r| r.reorder) {
        return Ok(());
    }
        let lv = liveness::compute(m);
        for bi in 0..m.blocks.len() {
            stats.blocks += 1;
            let insts = m.blocks[bi].insts.clone();
            let regional = region::has_regions(o, &insts);
            if !o.reorder && !regional {
                continue;
            }
            if insts.len() < 2 {
                continue;
            }
            if regional {
                schedule_block_runs(m, bi, o, &lv, stats)?;
                continue;
            }
            // Explicit partial waits count outstanding operations by program order: do not touch.
            if insts.iter().any(|i| matches!(i.op, MOp::SWaitLgkm | MOp::SWaitVm) && i.imm != 0) {
                stats.skipped += 1;
                continue;
            }
            let term = m.blocks[bi].term.clone();
            let xb = o.alu_across_barrier;
            let g = depgraph::build(m, &insts, &term, xb);
            let r = list::schedule_best(m, &insts, &g, &lv.live_in[bi], &lv.live_out[bi], o.max_vgprs as i64, false);
            if o.verify {
                verify::check_perm(m, &insts, &r.perm, &term, xb)
                    .map_err(|e| format!("schedule verification failed in block {bi}: {e}"))?;
            }
            stats.scheduled += 1;
            stats.moved += r.perm.iter().enumerate().filter(|(k, p)| *k != **p).count();
            stats.est_before += r.est_before as u64;
            stats.est_after += r.est_after as u64;
            stats.peak_before = stats.peak_before.max(r.peak_before);
            stats.peak_after = stats.peak_after.max(r.peak_after);
            m.blocks[bi].insts = apply_perm(&insts, &r.perm);
        }
    Ok(())
}

/// A block with weave regions: every maximal run of one region is scheduled by itself with the region's options (an authored / `rewait` run keeps its order); the live sets at the ends of
/// a run come from the block's liveness, so the register-pressure model stays exact. Nothing moves across a region boundary.
fn schedule_block_runs(m: &mut MFunction, bi: usize, o: &Options, lv: &liveness::Liveness, stats: &mut Stats) -> Result<(), String> {
    let insts = m.blocks[bi].insts.clone();
    let n = insts.len();
    let term = m.blocks[bi].term.clone();
    let xb = o.alu_across_barrier;
    // live sets before each instruction (the same rules as `liveness::compute`: any destination write kills)
    let mut live: std::collections::HashSet<VReg> = lv.live_out[bi].clone();
    let mut before: Vec<std::collections::HashSet<VReg>> = vec![std::collections::HashSet::new(); n + 1];
    before[n] = live.clone();
    for i in (0..n).rev() {
        if let Some(d) = insts[i].dst {
            live.remove(&m.canon(d));
        }
        for r in insts[i].uses() {
            live.insert(m.canon(r));
        }
        before[i] = live.clone();
    }
    let ro = o.region_options();
    let mut out: Vec<MInst> = Vec::with_capacity(n);
    for (lo, hi, rid) in region::runs(o, &insts) {
        let opt = &ro[region::region_index(o, rid)];
        let run: Vec<MInst> = insts[lo..hi].to_vec();
        // explicit partial waits count outstanding operations by program order; markers belong to the phase pass
        if !opt.reorder || run.len() < 2 || run.iter().any(|i| matches!(i.op, MOp::SWaitLgkm | MOp::SWaitVm) && i.imm != 0) || phase::has_markers(&run) {
            if opt.reorder {
                stats.skipped += 1;
            }
            out.extend(run);
            continue;
        }
        let t_run = if hi == n { term.clone() } else { Term::Endpgm };
        let g = depgraph::build(m, &run, &t_run, xb);
        let r = list::schedule_best(m, &run, &g, &before[lo], &before[hi], opt.max_vgprs as i64, false);
        if o.verify {
            verify::check_perm(m, &run, &r.perm, &t_run, xb).map_err(|e| format!("schedule verification failed in block {bi} (region {rid}): {e}"))?;
        }
        stats.scheduled += 1;
        stats.moved += r.perm.iter().enumerate().filter(|(k, p)| *k != **p).count();
        stats.est_before += r.est_before as u64;
        stats.est_after += r.est_after as u64;
        stats.peak_before = stats.peak_before.max(r.peak_before);
        stats.peak_after = stats.peak_after.max(r.peak_after);
        out.extend(apply_perm(&run, &r.perm));
    }
    m.blocks[bi].insts = out;
    Ok(())
}

/// Run the scheduling pass over the whole function.
pub fn run(m: &mut MFunction, o: &Options) -> Result<Stats, String> {
    // The register budget every pass schedules against may not exceed what the workgroup can launch with: ceil(waves / 4) waves share a SIMD's
    // 512-entry file (a 1024-thread block leaves 128 per wave). Moving loads up past that (e.g. later passes' loads hoisted above an earlier
    // reduction) would make the kernel unlaunchable; keep 8 registers of headroom for temporaries.
    let launch = launch_vgpr_budget(m.workgroup_size);
    let clamped;
    let o = if o.max_vgprs > launch {
        clamped = Options { max_vgprs: launch, ..o.clone() };
        &clamped
    } else {
        o
    };
    let mut stats = Stats::default();
    if o.cfg_cleanup {
        cfg::cleanup(m);
    }
    waits::strip_auto_waits(m);
    cfg::dce_dead_selects(m);
    if o.any_region(|r| r.ldshoist) {
        let n = hoist::run(m, o);
        if o.stats {
            eprintln!("ldshoist: {n} loads hoisted");
        }
    }
    if o.any_region(|r| r.slotfill) {
        let n = slotfill::run(m, o)?;
        if o.stats {
            eprintln!("slotfill: {n} instructions moved");
        }
    }
    // `sched_group_barrier` phases (blocks with directives; the directives are removed)
    let phased = phase::run(m, o)?;
    if o.stats && phased > 0 {
        eprintln!("phase: {phased} instructions moved");
    }
    schedule_blocks(m, o, &mut stats)?;
    if o.any_region(|r| r.waitsink) {
        let n = ldahead::sink_explicit_waits(m, o);
        if o.stats {
            eprintln!("wsink: {n} explicit waits moved");
        }
    }
    if o.any_region(|r| r.ldahead32 > 0 || r.ldsink > 0) {
        let n = ldahead::run(m, o);
        if o.stats {
            eprintln!("ldahead: {n} blocks changed");
        }
    }
    stats.waits_inserted = waits::insert_waits(m);
    if o.verify {
        waits::verify_waits(m).map_err(|e| format!("wait verification failed: {e}"))?;
    }
    if o.stats {
        eprintln!("sched: {stats:?}");
    }
    Ok(stats)
}

#[cfg(test)]
mod tests;

/// The per-wave register budget a scheduler may plan for: 512 / ceil(waves / 4) registers per wave, minus 8 of headroom, capped at 240 (the
/// default arch-VGPR budget).
pub fn launch_vgpr_budget(workgroup_size: u32) -> u32 {
    let waves = (workgroup_size + 63) / 64;
    let per_simd = ((waves + 3) / 4).max(1);
    (512 / per_simd).saturating_sub(8).min(240)
}
