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
pub mod list;
pub mod liveness;
pub mod model;
pub mod order;
pub mod sim;
pub mod swp;
pub mod taildup;
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
    /// Software-pipeline eligible counted loops (stage 3).
    pub swp: bool,
    /// Tail-duplicate join prefixes into the arms of if/else diamonds inside loops (stage 2).
    pub taildup: bool,
    /// Remove empty latch blocks / elide fall-through branches (default on with the scheduler).
    pub cfg_cleanup: bool,
    /// Debug bisect: schedule only this block index.
    pub only_block: Option<Vec<usize>>,
    /// Pressure limit in VGPR dwords (never below the original peak of the block).
    pub max_vgprs: u32,
    /// Run the verifiers (always on in tests).
    pub verify: bool,
    pub stats: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { reorder: true, alu_across_barrier: false, xbar_only: None, drain_only: false, swp: false, taildup: false, cfg_cleanup: true, only_block: None, max_vgprs: 240, verify: true, stats: false }
    }
}

impl Options {
    /// `AMDC_SCHED=1` enables; `AMDC_SCHED_OPTS=rewait,xbar,noverify,stats,limit=N` tunes.
    pub fn from_env() -> Option<Options> {
        if std::env::var("AMDC_SCHED").ok().as_deref() != Some("1") {
            return None;
        }
        let mut o = Options::default();
        if let Ok(s) = std::env::var("AMDC_SCHED_OPTS") {
            for t in s.split(',') {
                let t = t.trim();
                match t {
                    "rewait" => o.reorder = false,
                    "xbar" => o.alu_across_barrier = true,
                    "noverify" => o.verify = false,
                    "stats" => o.stats = true,
                    "drain" => o.drain_only = true,
                    "nocfg" => o.cfg_cleanup = false,
                    "swp" => o.swp = true,
                    "taildup" => o.taildup = true,
                    _ => {
                        if let Some(v) = t.strip_prefix("only=") {
                            o.only_block = Some(v.split('+').filter_map(|x| x.parse().ok()).collect());
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
        Some(o)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub blocks: usize,
    pub scheduled: usize,
    pub skipped: usize,
    pub moved: usize,
    pub swp_loops: usize,
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
            let r = list::schedule_best(m, &insts, &g, &lv.live_in[bi], &lv.live_out[bi], o.max_vgprs as i64);
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
    if o.taildup {
        let st = taildup::run(m, o);
        if o.stats {
            for l in &st.log {
                eprintln!("taildup: {l}");
            }
        }
    }
    if o.swp {
        let st = swp::run(m, o);
        stats.swp_loops = st.loops_pipelined;
        if o.stats {
            for l in &st.log {
                eprintln!("swp: {l}");
            }
        }
    }
    schedule_blocks(m, o, &mut stats)?;
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