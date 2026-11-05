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
pub mod mfmapair;
pub mod phase;
pub mod slotfill;
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
    /// Cheaper scheduling (fewer windows): used while evaluating transformation candidates.
    pub fast: bool,
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
    /// AMDC_SCHED=auto: decide per loop with the multi-wave simulator (see uto_plan).
    pub auto: bool,
    /// `trustmodel`: let `auto` also consider loops with barriers.
    pub trust_model: bool,
    /// `ldshoist`: hoist `ds_read` above independent instructions (`hoist=N`: max distance).
    pub ldshoist: bool,
    /// `hoistchain`: ldshoist also moves the pure ALU producers of a load's address (and of theirs) up with it, so a load whose address is computed right before it can still be issued early
    pub hoist_chain: bool,
    pub hoist_dist: usize,
    /// `slotfill`: the slot-filling scheduler (sched/slotfill.rs): MFMAs are the skeleton, the other instructions are spread over their deadline windows
    pub slotfill: bool,
    /// `mfmapair` (sched/mfmapair.rs): dependent same-accumulator MFMA pairs adjacent, non-MFMA work only after the second MFMA of a pair
    pub mfmapair: bool,
    /// shadow budget of a pair in units of the second MFMA's pipe occupancy (`mfmapair=1.5`); default 1.0
    pub mfmapair_budget: f64,
    /// Restrict block scheduling to blocks whose label is in the set or above the floor (blocks created
    /// by swp / taildup). Used by uto_plan.
    pub sched_labels: Option<(std::collections::HashSet<u32>, u32)>,
}

impl Default for Options {
    fn default() -> Self {
        Options { reorder: true, alu_across_barrier: false, xbar_only: None, drain_only: false, swp: false, fast: false, taildup: false, cfg_cleanup: true, only_block: None, max_vgprs: 240, verify: true, stats: false, auto: false, trust_model: false, ldshoist: false, hoist_chain: false, hoist_dist: 64, slotfill: false, mfmapair: false, mfmapair_budget: 1.0, sched_labels: None }
    }
}

impl Options {
    /// `AMDC_SCHED=1` enables; `AMDC_SCHED_OPTS=rewait,xbar,noverify,stats,limit=N` tunes.
    pub fn from_env() -> Option<Options> {
        let mode = std::env::var("AMDC_SCHED").ok();
        if !matches!(mode.as_deref(), Some("1") | Some("auto") | Some("ld")) {
            return None;
        }
        let mut o = Options::default();
        o.auto = mode.as_deref() == Some("auto");
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
                    "swp" => o.swp = true,
                    "taildup" => o.taildup = true,
                    "trustmodel" => o.trust_model = true,
                    "ldshoist" => o.ldshoist = true,
                    "slotfill" => o.slotfill = true,
                    "mfmapair" => o.mfmapair = true,
                    "hoistchain" => {
                        o.ldshoist = true;
                        o.hoist_chain = true;
                    }
                    _ => {
                        if let Some(v) = t.strip_prefix("only=") {
                            o.only_block = Some(v.split('+').filter_map(|x| x.parse().ok()).collect());
                        }
                        if let Some(v) = t.strip_prefix("mfmapair=") {
                            if let Ok(b) = v.parse() {
                                o.mfmapair = true;
                                o.mfmapair_budget = b;
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
            if let Some((set, floor)) = &o.sched_labels {
                let lb = m.blocks[bi].label;
                if !(set.contains(&lb) || lb > *floor) {
                    continue;
                }
            }
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
            let r = list::schedule_best(m, &insts, &g, &lv.live_in[bi], &lv.live_out[bi], o.max_vgprs as i64, o.fast);
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
    schedule_blocks(m, o, &mut stats)?;
    if o.mfmapair {
        let n = mfmapair::run(m, o)?;
        if o.stats {
            eprintln!("mfmapair: {n} pairs made adjacent");
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

/// Simulated cycles per iteration (average of 1 and 2 waves per SIMD) of every loop, by header label.
fn loop_costs(m: &MFunction) -> std::collections::HashMap<u32, f64> {
    let mut c = m.clone();
    crate::hazards::fix_value_hazards_opts(&mut c, true);
    sim::report(&c, &[1, 2])
        .into_iter()
        .filter_map(|r| {
            let w1 = r.cycles_per_iter.iter().find(|x| x.0 == 1)?.1;
            let w2 = r.cycles_per_iter.iter().find(|x| x.0 == 2)?.1;
            Some((r.header_label, (w1 + w2) / 2.0))
        })
        .collect()
}

/// Gain threshold shared with swp: a transformation is applied only if the model predicts >= 3%.
pub const AUTO_GAIN: f64 = 0.97;

/// `AMDC_SCHED=auto`. Returns the options to schedule `base` with, or `None` when the model predicts no gain
/// anywhere (the kernel is then compiled completely unscheduled).
/// * in-block scheduling is enabled per loop: only the hot-path blocks of loops whose simulated
///   cycles/iteration drop by >= 3% are scheduled; straight-line code is never touched;
/// * software pipelining is enabled (it applies its own >= 3% model gate per loop);
/// * taildup is enabled only if, compared with the plan without it, the loops that exist in both
///   results improve by >= 3% in total.
pub fn auto_plan(base: &MFunction, o: &Options) -> Option<Options> {
    let mut m = base.clone();
    if o.cfg_cleanup {
        cfg::cleanup(&mut m);
    }
    let log = |s: String| {
        if o.stats {
            eprintln!("auto: {s}");
        }
    };
    let base_cost = loop_costs(&m);
    let floor = m.blocks.iter().map(|b| b.label).max().unwrap_or(0);
    let idx = |l: u32| m.blocks.iter().position(|b| b.label == l);
    let mut selected = std::collections::HashSet::new();
    let mut eligible: Vec<u32> = Vec::new();
    for &(hl, ll) in &m.loops {
        let (Some(hi), Some(li)) = (idx(hl), idx(ll)) else { continue };
        let path = sim::hot_path(&m, hi, li);
        // The model has no cross-wave barrier term: on the FMHA kernels it predicts -23% for in-block scheduling
        // while hardware measured 221.7 vs 225.0 TFLOP/s unscheduled. Loops with `s_barrier` are therefore not
        // eligible unless AMDC_SCHED_OPTS contains `trustmodel`.
        if !o.trust_model && path.iter().any(|&b| m.blocks[b].insts.iter().any(|i| i.op == MOp::SBarrier)) {
            log(format!("loop {hl}: contains s_barrier (model blind to cross-wave sync): keep unscheduled"));
            continue;
        }
        eligible.push(hl);
        let mut c = m.clone();
        waits::strip_auto_waits(&mut c);
        let mut o2 = o.clone();
        o2.fast = true;
        o2.auto = false;
        o2.sched_labels = None;
        o2.only_block = Some(path.clone());
        let mut st = Stats::default();
        if schedule_blocks(&mut c, &o2, &mut st).is_err() {
            continue;
        }
        waits::insert_waits(&mut c);
        let (Some(b), Some(s)) = (base_cost.get(&hl), loop_costs(&c).get(&hl).copied()) else { continue };
        let take = s < b * AUTO_GAIN;
        log(format!("loop {hl}: model {b:.1} -> {s:.1} cycles/iteration ({:+.1}%) {}", (s / b - 1.0) * 100.0, if take { "SCHEDULE" } else { "keep unscheduled" }));
        if take {
            for &p in &path {
                selected.insert(m.blocks[p].label);
            }
        }
    }
    if eligible.is_empty() {
        log("no eligible loop: kernel stays unscheduled".to_string());
        return None;
    }
    let mut plan = o.clone();
    plan.auto = false;
    plan.swp = true;
    plan.reorder = true;
    plan.taildup = false;
    plan.sched_labels = Some((selected.clone(), floor));
    // Trial run: what do the plan (and the plan + taildup) actually produce?
    let trial = |opts: &Options| -> Option<(Stats, std::collections::HashMap<u32, f64>)> {
        let mut c = base.clone();
        let mut q = opts.clone();
        q.stats = false;
        let st = run(&mut c, &q).ok()?;
        Some((st, loop_costs(&c)))
    };
    let a = trial(&plan);
    let applied = a.as_ref().map_or(false, |(st, _)| st.scheduled > 0 || st.swp_loops > 0);
    let mut chosen = if applied { Some(plan.clone()) } else { None };
    let mut td = plan.clone();
    td.taildup = true;
    if let Some((stb, cb)) = trial(&td) {
        let reference = a.as_ref().map(|x| x.1.clone()).unwrap_or_else(|| base_cost.clone());
        let common: Vec<u32> = cb.keys().filter(|k| reference.contains_key(k) && eligible.contains(k)).copied().collect();
        let (sa, sb): (f64, f64) = common.iter().fold((0.0, 0.0), |acc, k| (acc.0 + reference[k], acc.1 + cb[k]));
        let gain = !common.is_empty() && sb < sa * AUTO_GAIN;
        log(format!("taildup trial: loops {common:?} model {sa:.1} -> {sb:.1} ({} scheduled blocks) {}", stb.scheduled, if gain { "ENABLE" } else { "off" }));
        if gain {
            chosen = Some(td);
        }
    }
    if chosen.is_none() {
        log("no modelled gain: kernel stays unscheduled".to_string());
    }
    chosen
}

#[cfg(test)]
mod tests;