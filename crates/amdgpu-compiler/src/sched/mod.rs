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
    /// Debug bisect: ALU may cross barriers only in this block index.
    pub xbar_only: Option<usize>,
    /// Debug: full-drain waits only.
    pub drain_only: bool,
    /// Remove empty latch blocks / elide fall-through branches (default on with the scheduler).
    pub cfg_cleanup: bool,
    /// Debug bisect: schedule only this block index.
    pub only_block: Option<Vec<usize>>,
    /// Pressure limit in VGPR dwords (never below the original peak of the block).
    pub max_vgprs: u32,
    /// Run the verifiers (always on in tests).
    pub verify: bool,
    pub stats: bool,
    /// `ldshoist`: hoist `ds_read` above independent instructions (`hoist=N`: max distance).
    pub ldshoist: bool,
    /// `hoistchain`: ldshoist also moves the pure ALU producers of a load's address (and of theirs) up with it, so a load whose address is computed right before it can still be issued early
    pub hoist_chain: bool,
    pub hoist_dist: usize,
    /// `slotfill`: the slot-filling scheduler (sched/slotfill.rs): MFMAs are the skeleton, the other instructions are spread over their deadline windows
    pub slotfill: bool,
    /// `ldahead=D` (sched/ldahead.rs): post-pass moving every ds_read up until D MFMAs sit between it and its first consumer (0 = off)
    pub ldahead: usize,
    /// `ldslack=S`: extra live VGPR dwords `ldahead` may use above the block's original peak
    pub ldslack: i64,
    /// `ldsink=D` (sched/ldahead.rs): the opposite direction - sink every ds_read down until at most D MFMAs sit between it and its first consumer (0 = off)
    pub ldsink: usize,
    /// `ldahead32=D`: like `ldahead`, for the 32-bit scale-word reads (`ds_read_b32`) only
    pub ldahead32: usize,
    /// `wsink` (sched/ldahead.rs): sink the kernel's EXPLICIT `s_waitcnt lgkmcnt(n)` down past the compute that does not need them, so the MFMAs in front of a closing `wait_lds(0)` get the counted waits of the scoreboard
    pub waitsink: bool,
    /// `ovl=N` (sched/ldahead.rs `spread_valu`): inside every barrier-free stretch with >= 2 MFMAs, VALU / SALU work that is independent of the MFMAs in between is hoisted up so that it is spread evenly over the stretch's MFMAs (a stretch needs at least N such instructions; 0 = off)
    pub ovl: usize,
}

impl Default for Options {
    fn default() -> Self {
        Options { reorder: true, alu_across_barrier: false, xbar_only: None, drain_only: false, cfg_cleanup: true, only_block: None, max_vgprs: 240, verify: true, stats: false, ldshoist: false, hoist_chain: false, hoist_dist: 64, slotfill: false, ldahead: 0, ldslack: 0, ldsink: 0, ldahead32: 0, waitsink: false, ovl: 0 }
    }
}

impl Options {
    /// Debug / bisect override: `AMDC_SCHED=1` (full scheduler) or `ld` (waits + LDS-read hoisting), `AMDC_SCHED_OPTS=rewait,xbar,noverify,stats,limit=N,only=..`
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
                    "xbar" => o.alu_across_barrier = true,
                    "noverify" => o.verify = false,
                    "stats" => o.stats = true,
                    "drain" => o.drain_only = true,
                    "nocfg" => o.cfg_cleanup = false,
                    "ldshoist" => o.ldshoist = true,
                    "slotfill" => o.slotfill = true,
                    "wsink" => o.waitsink = true,
                    "ovl" => o.ovl = 8,
                    "hoistchain" => {
                        o.ldshoist = true;
                        o.hoist_chain = true;
                    }
                    _ => {
                        if let Some(v) = t.strip_prefix("only=") {
                            o.only_block = Some(v.split('+').filter_map(|x| x.parse().ok()).collect());
                        }
                        if let Some(v) = t.strip_prefix("ovl=") {
                            if let Ok(n) = v.parse() {
                                o.ovl = n;
                            }
                        }
                        if let Some(v) = t.strip_prefix("ldahead32=") {
                            if let Ok(n) = v.parse() {
                                o.ldahead32 = n;
                            }
                        }
                        if let Some(v) = t.strip_prefix("ldahead=") {
                            if let Ok(n) = v.parse() {
                                o.ldahead = n;
                            }
                        }
                        if let Some(v) = t.strip_prefix("ldsink=") {
                            if let Ok(n) = v.parse() {
                                o.ldsink = n;
                            }
                        }
                        if let Some(v) = t.strip_prefix("ldslack=") {
                            if let Ok(n) = v.parse() {
                                o.ldslack = n;
                            }
                        }
                        if let Some(v) = t.strip_prefix("hoist=") {
                            if let Ok(n) = v.parse() {
                                o.hoist_dist = n;
                                o.ldshoist = true;
                            }
                        }
                        if let Some(v) = t.strip_prefix("xbar=") {
                            o.xbar_only = v.parse().ok();
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
    if !o.reorder {
        return Ok(());
    }
        let lv = liveness::compute(m);
        for bi in 0..m.blocks.len() {
            stats.blocks += 1;
            let insts = m.blocks[bi].insts.clone();
            if insts.len() < 2 || o.only_block.as_ref().map_or(false, |x| !x.contains(&bi)) {
                continue;
            }
            // Explicit partial waits count outstanding operations by program order: do not touch.
            if insts.iter().any(|i| matches!(i.op, MOp::SWaitLgkm | MOp::SWaitVm) && i.imm != 0) {
                stats.skipped += 1;
                continue;
            }
            let term = m.blocks[bi].term.clone();
            let xb = o.alu_across_barrier || o.xbar_only == Some(bi);
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

/// Run the scheduling pass over the whole function.
pub fn run(m: &mut MFunction, o: &Options) -> Result<Stats, String> {
    let mut stats = Stats::default();
    if o.cfg_cleanup {
        cfg::cleanup(m);
    }
    waits::DRAIN_ONLY.store(o.drain_only, std::sync::atomic::Ordering::Relaxed);
    waits::strip_auto_waits(m);
    cfg::dce_dead_selects(m);
    if o.ldshoist {
        let n = hoist::run(m, o);
        if o.stats {
            eprintln!("ldshoist: {n} loads hoisted");
        }
    }
    if o.slotfill {
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
    if o.stats {
        let (pk, at, blk) = crate::regalloc::pressure_profile(m);
        eprintln!("pressure (allocator measure, authored order): peak {pk} at position {at} (block {blk})");
    }
    if o.stats && std::env::var("AMDC_PRESSURE_DEBUG").is_ok() {
        let (pk, at, blk) = crate::regalloc::pressure_profile(m);
        let alloc_live: Vec<VReg> = crate::regalloc::live_at(m, at);
        let lv = liveness::compute(m);
        let start: usize = m.blocks[..blk].iter().map(|b| b.insts.len()).sum();
        let upto = at.saturating_sub(start);
        // model live set at `upto`: live-in plus defs so far, minus values with no later use (backwards from live-out)
        let insts = &m.blocks[blk].insts;
        let mut live: std::collections::HashSet<VReg> = lv.live_out[blk].iter().map(|&r| m.canon(r)).collect();
        for i in (upto..insts.len()).rev() {
            if let Some(d) = insts[i].dst {
                if !m.alias.contains_key(&d) {
                    live.remove(&m.canon(d));
                }
            }
            for r in insts[i].uses() {
                live.insert(m.canon(r));
            }
        }
        let w = |r: VReg| if m.vreg_class(r).is_vector() { m.vreg_class(r).dwords() as i64 } else { 0 };
        let model: i64 = live.iter().map(|&r| w(r)).sum();
        let only_alloc: Vec<String> = alloc_live.iter().filter(|r| !live.contains(&m.canon(**r))).map(|r| format!("r{r}:{:?}", m.vreg_class(*r))).collect();
        let only_model: Vec<String> = live.iter().filter(|r| !alloc_live.iter().any(|a| m.canon(*a) == **r)).map(|r| format!("r{r}:{:?}", m.vreg_class(*r))).collect();
        eprintln!("PDBG block {blk} pos {at}: allocator peak {pk}; backward-liveness model dwords {model}; allocator live vregs {}; ONLY allocator: {:?}; ONLY model: {:?}", alloc_live.len(), only_alloc, only_model);
    }
    schedule_blocks(m, o, &mut stats)?;
    if o.stats {
        let (pk, at, blk) = crate::regalloc::pressure_profile(m);
        eprintln!("pressure (allocator measure, scheduled order): peak {pk} at position {at} (block {blk}); model peak_before {} peak_after {}", stats.peak_before, stats.peak_after);
    }
    if o.ovl > 0 {
        let n = ldahead::spread_valu(m, o);
        if o.stats {
            eprintln!("ovl: {n} instructions moved");
        }
    }
    if o.waitsink {
        let n = ldahead::sink_explicit_waits(m, o);
        if o.stats {
            eprintln!("wsink: {n} explicit waits moved");
        }
    }
    if o.ldahead > 0 || o.ldahead32 > 0 || o.ldsink > 0 {
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