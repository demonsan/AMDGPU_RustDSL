//! Stage 2: tail duplication across if/else diamonds.
//!
//! A join block J reached only from the two arms A and B (both ending in `Goto J`) can have a
//! prefix of its instructions copied to the end of both arms. The scheduler then sees the join's
//! independent work together with the arm's long-latency tail (an MFMA draining, a load in flight)
//! and fills the stall; J keeps the rest. Each path executes exactly one copy, so semantics and
//! barrier counts per path are unchanged. The prefix length is chosen with the simulator on the hot
//! path of the enclosing loop (only done inside loops).

use crate::mir::*;

use super::swp::eval_cost;

#[derive(Debug, Clone, Default)]
pub struct TdStats {
    pub diamonds: usize,
    pub duplicated: usize,
    pub log: Vec<String>,
}

fn preds_of(m: &MFunction, label: u32) -> Vec<usize> {
    let mut v = Vec::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        let hit = match &b.term {
            Term::Goto(t) => *t == label,
            Term::CondGoto { target, fall, .. } => *target == label || *fall == label,
            Term::Endpgm => false,
        };
        if hit {
            v.push(bi);
        }
    }
    v
}

fn apply(m: &MFunction, a: usize, b: usize, j: usize, k: usize) -> MFunction {
    let mut c = m.clone();
    let pre: Vec<MInst> = c.blocks[j].insts[..k].to_vec();
    c.blocks[a].insts.extend(pre.iter().cloned());
    c.blocks[b].insts.extend(pre);
    c.blocks[j].insts.drain(..k);
    c
}

pub fn run(m: &mut MFunction, o: &super::Options) -> TdStats {
    let mut st = TdStats::default();
    let loops: Vec<(u32, u32)> = m.loops.clone();
    let nb = m.blocks.len();
    for j in 0..nb {
        let jl = m.blocks[j].label;
        let preds = preds_of(m, jl);
        if preds.len() != 2 || preds.iter().any(|&p| p >= j) {
            continue;
        }
        let (a, b) = (preds[0], preds[1]);
        if !matches!(m.blocks[a].term, Term::Goto(t) if t == jl) || !matches!(m.blocks[b].term, Term::Goto(t) if t == jl) {
            continue;
        }
        // inside a loop: the loop header must come before the arms and the latch at or after J
        let Some(&(hl, ll)) = loops.iter().find(|&&(h, l)| {
            let hi = m.blocks.iter().position(|x| x.label == h);
            let li = m.blocks.iter().position(|x| x.label == l);
            matches!((hi, li), (Some(hi), Some(li)) if hi < a.min(b) && li >= j)
        }) else {
            continue;
        };
        st.diamonds += 1;
        let len = m.blocks[j].insts.len();
        if len == 0 {
            continue;
        }
        let Some(base) = eval_cost(m, hl, 1.0, o) else { continue };
        let _ = ll;
        let mut best: Option<(f64, usize)> = None;
        let mut cands: Vec<usize> = (1..=len.min(16)).collect();
        let mut k = 20;
        while k < len {
            cands.push(k);
            k += 8;
        }
        cands.push(len);
        cands.sort();
        cands.dedup();
        for k in cands {
            let cand = apply(m, a, b, j, k);
            if let Some(c) = eval_cost(&cand, hl, 1.0, o) {
                if best.map_or(true, |x| c < x.0) {
                    best = Some((c, k));
                }
            }
        }
        match best {
            Some((c, k)) if c < base * 0.98 => {
                st.log.push(format!("diamond at block {j}: duplicated {k}/{len} join instructions into both arms: model {base:.1} -> {c:.1}"));
                *m = apply(m, a, b, j, k);
                st.duplicated += 1;
            }
            Some((c, k)) => st.log.push(format!("diamond at block {j}: not duplicated (best k={k}: {c:.1} vs {base:.1})")),
            None => {}
        }
    }
    st
}