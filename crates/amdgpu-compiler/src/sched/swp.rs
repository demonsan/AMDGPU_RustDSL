//! Software pipelining (modulo scheduling by 2x-unrolled kernel with register renaming) of
//! single-block counted loops with a runtime trip count.
//!
//! Loop shape (after `cfg::cleanup`): block H, `CondGoto { Vccnz, target: H, fall: exit }`, whose
//! last vcc write is `v_cmp_lt_i32 (it + 1), n` with `it` the induction register (step 1) and `n`
//! loop invariant (do-while: m = max(1, n - it0) iterations).
//!
//! The body is cut in program order into S0 (prefix) and S1 (rest). With m iterations:
//!   P : S0(0)
//!   K : repeated kt = (m-1)/2 times:  S1(i) S0(i+1) S1(i+1) S0(i+2)   (renamed so the two halves use
//!       different registers for values that cross the cut; S0(i+1) can then overlap S1(i))
//!   E : S1(2kt) ; if (m-1) is odd: S0 S1 once more (plain)
//! No instruction is ever executed for an iteration that does not exist (no speculation), short trip
//! counts (m = 1, 2) skip the kernel. Correctness is program-order equivalence of the concatenated
//! iterations; the block scheduler then reorders S1(i) against S0(i+1) under the usual dependences.

use std::collections::{HashMap, HashSet};

use super::effects::*;
use crate::mir::*;

pub struct LoopInfo {
    pub header: usize,
    pub pre: usize,
    pub exit_label: u32,
    pub it: VReg,
    /// Body without the exit-test instructions.
    pub body: Vec<MInst>,
    /// vreg holding the bound `n` and (if defined inside the loop) a clone of its definition.
    pub bound: VReg,
    pub bound_def: Option<MInst>,
}

fn writes_vcc(i: &MInst) -> bool {
    i.op.defines_vcc() || i.op == MOp::VPtrAdd
}

pub fn analyze(m: &MFunction, hi: usize) -> Option<LoopInfo> {
    let h = &m.blocks[hi];
    let Term::CondGoto { on: Pred::Vccnz, target, fall } = h.term.clone() else { return None };
    if target != h.label || hi == 0 {
        return None;
    }
    let pre = hi - 1;
    if !matches!(m.blocks[pre].term, Term::Goto(t) if t == h.label) {
        return None;
    }
    // no other predecessor of H
    for (bi, b) in m.blocks.iter().enumerate() {
        if bi == hi || bi == pre {
            continue;
        }
        let refs = match &b.term {
            Term::Goto(t) => *t == h.label,
            Term::CondGoto { target, fall, .. } => *target == h.label || *fall == h.label,
            Term::Endpgm => false,
        };
        if refs {
            return None;
        }
    }
    let insts = &h.insts;
    if insts.iter().any(|i| {
        matches!(i.op, MOp::SBarrier | MOp::SSetPrio | MOp::SWaitLgkm | MOp::SWaitVm | MOp::SNop | MOp::SEndpgm)
    }) {
        return None;
    }
    // exit test
    let ic = insts.iter().rposition(|i| writes_vcc(i))?;
    let cmp = &insts[ic];
    if cmp.op != MOp::VCmpLtI32 {
        return None;
    }
    let (Some(Src::Reg(a)), Some(Src::Reg(b))) = (cmp.srcs.first().copied(), cmp.srcs.get(1).copied()) else { return None };
    // a = it + 1
    let da = insts[..ic].iter().rposition(|i| i.dst == Some(a))?;
    let add = &insts[da];
    if add.op != MOp::VAddU32 {
        return None;
    }
    let (Some(Src::Reg(it)), Some(c1)) = (add.srcs.first().copied(), add.srcs.get(1).copied()) else { return None };
    let one = match c1 {
        Src::Imm(1) => true,
        Src::Reg(c) => insts[..da]
            .iter()
            .rposition(|i| i.dst == Some(c))
            .map_or(false, |k| insts[k].op == MOp::VMovB32 && insts[k].srcs.is_empty() && insts[k].imm == 1),
        _ => false,
    };
    if !one || m.alias.contains_key(&it) || m.vreg_class(it) != RClass::V1 {
        return None;
    }
    // `it` is written exactly once in the body: by the add itself or by a following copy
    let writers: Vec<usize> = insts.iter().enumerate().filter(|(_, i)| i.dst == Some(it)).map(|(k, _)| k).collect();
    let ok_writer = writers.len() == 1
        && (writers[0] == da
            || (insts[writers[0]].op == MOp::VMovB32
                && matches!(insts[writers[0]].srcs.first(), Some(Src::Reg(r)) if *r == a)
                && writers[0] > da));
    if !ok_writer {
        return None;
    }
    // bound: defined before the loop, or by a loop-invariant instruction inside it
    let written: HashSet<VReg> = insts.iter().filter_map(|i| i.dst).map(|d| m.canon(d)).collect();
    let db = insts[..ic].iter().rposition(|i| i.dst == Some(b));
    let (bound_def, bound_idx) = match db {
        None => (None, None),
        Some(k) => {
            let d = &insts[k];
            let inv = matches!(d.op, MOp::VMovFromS | MOp::VMovB32)
                && d.srcs.iter().all(|s| match s {
                    Src::Reg(r) => !written.contains(&m.canon(*r)),
                    Src::Imm(_) => true,
                })
                && insts.iter().filter(|i| i.dst == Some(b)).count() == 1;
            if !inv {
                return None;
            }
            (Some(d.clone()), Some(k))
        }
    };
    // body = everything except the compare (and the bound definition if nobody else reads it)
    let mut body: Vec<MInst> = Vec::new();
    for (k, i) in insts.iter().enumerate() {
        if k == ic {
            continue;
        }
        if Some(k) == bound_idx {
            let used_elsewhere = insts.iter().enumerate().any(|(j, x)| j != ic && j != k && x.uses().contains(&b));
            if !used_elsewhere {
                continue;
            }
        }
        body.push(i.clone());
    }
    if body.len() < 4 {
        return None;
    }
    // vcc must not be read by anything in the body (it only feeds the removed compare)
    if body.iter().any(|i| i.op.consumes_vcc()) {
        return None;
    }
    Some(LoopInfo { header: hi, pre, exit_label: fall, it, body, bound: b, bound_def })
}

struct Renamer<'a> {
    m: &'a mut MFunction,
    rename: HashMap<VReg, VReg>, // canonical tuple -> new tuple
    alias_cache: HashMap<(VReg, u8, RClass), VReg>,
}

impl<'a> Renamer<'a> {
    fn map(&mut self, r: VReg) -> VReg {
        let canon = self.m.canon(r);
        let Some(&nt) = self.rename.get(&canon) else { return r };
        if r == canon {
            return nt;
        }
        let off = self.m.alias[&r].1;
        let class = self.m.vreg_class(r);
        if let Some(a) = self.alias_cache.get(&(nt, off, class)) {
            return *a;
        }
        let a = self.m.vregs.len() as VReg;
        self.m.vregs.push(class);
        self.m.alias.insert(a, (nt, off));
        self.alias_cache.insert((nt, off, class), a);
        a
    }

    fn inst(&mut self, i: &MInst) -> MInst {
        let mut x = i.clone();
        if let Some(d) = x.dst {
            x.dst = Some(self.map(d));
        }
        for s in x.srcs.iter_mut() {
            if let Src::Reg(r) = s {
                *r = self.map(*r);
            }
        }
        x
    }
}

/// Cross-stage temporaries: canonical vector vregs whose every read slot is first written in the same
/// iteration, defined in S0, read in S1, not written in S1 and not live after the loop.
fn crossing(m: &MFunction, body: &[MInst], k: usize, live_after: &HashSet<VReg>) -> Vec<VReg> {
    let effs: Vec<Effects> = body.iter().map(|i| effects(m, i)).collect();
    let mut first_is_write: HashMap<Slot, bool> = HashMap::new();
    for e in &effs {
        for s in &e.reads {
            first_is_write.entry(*s).or_insert(false);
        }
        for s in &e.writes {
            first_is_write.entry(*s).or_insert(true);
        }
    }
    let mut defs0: HashSet<VReg> = HashSet::new();
    let mut uses1: HashSet<VReg> = HashSet::new();
    let mut defs1: HashSet<VReg> = HashSet::new();
    let mut live_in: HashSet<VReg> = HashSet::new();
    for (idx, e) in effs.iter().enumerate() {
        for s in &e.writes {
            if *s >= SCC {
                continue;
            }
            if idx < k {
                defs0.insert(s / 16);
            } else {
                defs1.insert(s / 16);
            }
        }
        if idx >= k {
            for s in &e.reads {
                if *s < SCC {
                    uses1.insert(s / 16);
                }
            }
        }
    }
    for (s, w) in &first_is_write {
        if !*w && *s < SCC {
            live_in.insert(s / 16);
        }
    }
    let mut out: Vec<VReg> = defs0
        .into_iter()
        .filter(|v| {
            uses1.contains(v)
                && !defs1.contains(v)
                && !live_in.contains(v)
                && !live_after.contains(v)
                && m.vreg_class(*v).is_vector()
                && !m.alias.contains_key(v)
        })
        .collect();
    out.sort();
    out
}

/// Apply the transformation with cut `k`; returns the new function (header block replaced by
/// kernel/epilogue blocks) or None if the cut is not applicable.
pub fn apply(m: &MFunction, li: &LoopInfo, k: usize, live_after: &HashSet<VReg>) -> Option<MFunction> {
    if k == 0 || k >= li.body.len() {
        return None;
    }
    let mut out = m.clone();
    let x = crossing(&out, &li.body, k, live_after);
    let mut rn = Renamer { m: &mut out, rename: HashMap::new(), alias_cache: HashMap::new() };
    for v in &x {
        let c = rn.m.vreg_class(*v);
        let nv = rn.m.vregs.len() as VReg;
        rn.m.vregs.push(c);
        rn.rename.insert(*v, nv);
    }
    let s0: Vec<MInst> = li.body[..k].to_vec();
    let s1: Vec<MInst> = li.body[k..].to_vec();
    let ren_all = |rn: &mut Renamer, v: &[MInst]| -> Vec<MInst> { v.iter().map(|i| rn.inst(i)).collect() };
    let s0_b = ren_all(&mut rn, &s0);
    let s1_b = ren_all(&mut rn, &s1);
    let m2: &mut MFunction = rn.m;

    let label = |m: &MFunction| m.blocks.iter().map(|b| b.label).max().unwrap_or(0) + 1;
    let k_label = label(m2);
    let e_label = k_label + 1;
    let e2_label = k_label + 2;
    let new_v1 = |m: &mut MFunction| -> VReg {
        m.vregs.push(RClass::V1);
        (m.vregs.len() - 1) as VReg
    };
    // ---- preheader additions
    let mut pre: Vec<MInst> = Vec::new();
    let nreg = match &li.bound_def {
        Some(d) => {
            let nb = new_v1(m2);
            let mut c = d.clone();
            c.dst = Some(nb);
            pre.push(c);
            nb
        }
        None => li.bound,
    };
    let notit = new_v1(m2);
    pre.push(MInst::new(MOp::VXorB32).def(notit).src(Src::Imm(-1)).srcreg(li.it));
    let t0 = new_v1(m2);
    pre.push(MInst::new(MOp::VAddU32).def(t0).srcreg(nreg).srcreg(notit));
    let zero = new_v1(m2);
    pre.push(MInst::new(MOp::VMovB32).def(zero).imm(0));
    pre.push(MInst::new(MOp::VCmpGeI32).srcreg(t0).srcreg(zero));
    let t = new_v1(m2);
    pre.push(MInst::new(MOp::VCndmask).def(t).src(Src::Imm(0)).srcreg(t0));
    let kt = new_v1(m2);
    pre.push(MInst::new(MOp::VLshrB32).def(kt).src(Src::Imm(1)).srcreg(t));
    let odd = new_v1(m2);
    pre.push(MInst::new(MOp::VAndB32).def(odd).src(Src::Imm(1)).srcreg(t));
    let cnt = new_v1(m2);
    pre.push(MInst::new(MOp::VMovB32).def(cnt).imm(0));
    pre.extend(s0.iter().cloned()); // P = S0(0), original names
    pre.push(MInst::new(MOp::VCmpLtI32).src(Src::Imm(0)).srcreg(kt));

    // ---- kernel: S1(A) S0(B) S1(B) S0(A)
    let mut kern: Vec<MInst> = Vec::new();
    kern.extend(s1.iter().cloned());
    kern.extend(s0_b.iter().cloned());
    kern.extend(s1_b.iter().cloned());
    kern.extend(s0.iter().cloned());
    kern.push(MInst::new(MOp::VAddU32).def(cnt).src(Src::Imm(1)).srcreg(cnt));
    kern.push(MInst::new(MOp::VCmpLtI32).srcreg(cnt).srcreg(kt));

    // ---- epilogue
    let mut ep: Vec<MInst> = s1.to_vec();
    ep.push(MInst::new(MOp::VCmpNeI32).src(Src::Imm(0)).srcreg(odd));
    let mut ep2: Vec<MInst> = s0.to_vec();
    ep2.extend(s1.iter().cloned());

    let hl = m2.blocks[li.header].label;
    let mut pre_b = m2.blocks[li.pre].clone();
    pre_b.insts.extend(pre);
    pre_b.term = Term::CondGoto { on: Pred::Vccnz, target: k_label, fall: e_label };
    let mut kb = Block::new(k_label);
    kb.insts = kern;
    kb.term = Term::CondGoto { on: Pred::Vccnz, target: k_label, fall: e_label };
    let mut eb = Block::new(e_label);
    eb.insts = ep;
    eb.term = Term::CondGoto { on: Pred::Vccnz, target: e2_label, fall: li.exit_label };
    let mut e2b = Block::new(e2_label);
    e2b.insts = ep2;
    e2b.term = Term::Goto(li.exit_label);
    // replace [pre, header] by [pre', K, E, E2]
    let mut blocks: Vec<Block> = Vec::new();
    for (bi, b) in m2.blocks.iter().enumerate() {
        if bi == li.pre {
            blocks.push(pre_b.clone());
            blocks.push(kb.clone());
            blocks.push(eb.clone());
            blocks.push(e2b.clone());
        } else if bi == li.header {
            continue;
        } else {
            blocks.push(b.clone());
        }
    }
    m2.blocks = blocks;
    m2.loops.retain(|l| l.0 != hl);
    m2.loops.push((k_label, k_label));
    Some(out)
}
/// Cost used to choose between the plain loop and a candidate: simulated cycles per original
/// iteration, averaged over 1 and 2 waves per SIMD, for the loop headed by `label`.
pub(super) fn eval_cost(m: &MFunction, label: u32, per_trip: f64, o: &super::Options) -> Option<f64> {
    let mut c = m.clone();
    let mut stats = super::Stats::default();
    super::schedule_blocks(&mut c, o, &mut stats).ok()?;
    super::waits::insert_waits(&mut c);
    crate::hazards::fix_value_hazards_opts(&mut c, true);
    let rep = super::sim::report(&c, &[1, 2]);
    let r = rep.iter().find(|r| r.header_label == label)?;
    let w1 = r.cycles_per_iter.iter().find(|x| x.0 == 1)?.1;
    let w2 = r.cycles_per_iter.iter().find(|x| x.0 == 2)?.1;
    Some((w1 + w2) / 2.0 / per_trip)
}

#[derive(Debug, Clone, Default)]
pub struct SwpStats {
    pub loops_seen: usize,
    pub loops_pipelined: usize,
    pub log: Vec<String>,
}

/// Pipeline every eligible loop of `m` whose modelled cost improves by at least 3%.
pub fn run(m: &mut MFunction, o: &super::Options) -> SwpStats {
    let mut st = SwpStats::default();
    let loops: Vec<(u32, u32)> = m.loops.clone();
    for (hl, ll) in loops {
        if hl != ll {
            continue;
        }
        st.loops_seen += 1;
        let Some(hi) = m.blocks.iter().position(|b| b.label == hl) else { continue };
        let Some(li) = analyze(m, hi) else {
            st.log.push(format!("loop {hl}: not a canonical counted single-block loop"));
            continue;
        };
        let lv = super::liveness::compute(m);
        let exit_idx = m.blocks.iter().position(|b| b.label == li.exit_label);
        let live_after: HashSet<VReg> = exit_idx.map(|e| lv.live_in[e].clone()).unwrap_or_default();
        let Some(base) = eval_cost(m, hl, 1.0, o) else { continue };
        let n = li.body.len();
        let step = (n / 40).max(1);
        let mut best: Option<(f64, usize, MFunction)> = None;
        let mut k = 1;
        while k < n {
            if let Some(cand) = apply(m, &li, k, &live_after) {
                let kl = cand.loops.last().map(|l| l.0).unwrap_or(0);
                if let Some(c) = eval_cost(&cand, kl, 2.0, o) {
                    if best.as_ref().map_or(true, |b| c < b.0) {
                        best = Some((c, k, cand));
                    }
                }
            }
            k += step;
        }
        match best {
            Some((c, k, cand)) if c < base * 0.97 => {
                st.log.push(format!("loop {hl}: pipelined at cut {k}/{n}: model {base:.1} -> {c:.1} cycles/iteration"));
                *m = cand;
                st.loops_pipelined += 1;
            }
            Some((c, k, _)) => st.log.push(format!("loop {hl}: not pipelined (best cut {k}: {c:.1} vs {base:.1})")),
            None => {}
        }
    }
    st
}