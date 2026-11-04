//! Stage-boundary RESOURCE CONTRACTS of a fused kernel (program item G step 3, docs/fusion-layout-handoff.md §3).
//!
//! A fused kernel is a sequence of stages (GEMM 1 -> elementwise -> GEMM 2, ...). At each boundary the producing stage declares what it hands over
//! and the consuming stage declares what it expects; `check_boundary` proves they agree. The checks run at build time today (assertions in the
//! builder); when region IR lands (item B) the same `StageBoundary` hangs off the Region node and the verifier checks it. A boundary is not a
//! scheduling barrier: nothing here emits an instruction.
//!
//! What a boundary carries:
//! * live-out fragments: name, TV layout over the tensor tile, element bits, storage (VGPR / AGPR / an LDS region), the K permutation it is in;
//! * pinned register ranges (e.g. AGPR accumulators that the next stage must not clobber);
//! * LDS regions the producer frees and the consumer reuses (a reused region must have been freed);
//! * the outstanding vm / lgkm counts allowed across the boundary (the consumer's tolerance >= the producer's worst case);
//! * the barrier phase (cross-wave stages must agree on how many barriers have been passed) and the wave roles.

use amdgpu_layout::{plan_convert_tables, Layout, Movement};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Storage {
    Vgpr,
    Agpr,
    Lds(String),
}

/// One fragment crossing the boundary: its element table (`[t + threads v]` -> tensor element, after the K permutation) and register facts.
#[derive(Clone, Debug)]
pub struct LiveFrag {
    pub name: String,
    pub threads: usize,
    pub values: usize,
    pub table: Vec<i64>,
    pub elem_bits: u32,
    pub storage: Storage,
}

impl LiveFrag {
    pub fn from_tv(name: &str, tv: &Layout, elem_bits: u32, storage: Storage) -> LiveFrag {
        let t = tv.get(0).size().st() as usize;
        let v = tv.get(1).size().st() as usize;
        LiveFrag { name: name.into(), threads: t, values: v, table: (0..t * v).map(|i| tv.at(i as i64)).collect(), elem_bits, storage }
    }
    pub fn from_pairs(name: &str, per_lane: &[Vec<i64>], elem_bits: u32, storage: Storage) -> LiveFrag {
        let t = per_lane.len();
        let v = per_lane[0].len();
        let table = (0..t * v).map(|i| per_lane[i % t][i / t]).collect();
        LiveFrag { name: name.into(), threads: t, values: v, table, elem_bits, storage }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StageBoundary {
    pub live: Vec<LiveFrag>,
    /// (what, first register, count): ranges that must survive the boundary untouched
    pub pinned: Vec<(String, u32, u32)>,
    pub lds_freed: Vec<String>,
    pub lds_reused: Vec<String>,
    /// producer: worst-case outstanding counts at the boundary; consumer: the counts it tolerates
    pub vm: u8,
    pub lgkm: u8,
    pub barrier_phase: u32,
    pub roles: Vec<String>,
}

/// What the boundary costs / proves (for the report).
#[derive(Clone, Debug, Default)]
pub struct BoundaryReport {
    /// per consumed fragment: (name, movement of the hand-off, conversion instructions)
    pub frags: Vec<(String, Movement, usize)>,
}

/// Proves that `prod` (what the producing stage hands over) satisfies `cons` (what the consuming stage expects).
pub fn check_boundary(prod: &StageBoundary, cons: &StageBoundary) -> Result<BoundaryReport, String> {
    let mut rep = BoundaryReport::default();
    for want in &cons.live {
        let have = prod.live.iter().find(|f| f.name == want.name).ok_or_else(|| format!("boundary: '{}' is not handed over", want.name))?;
        if (have.threads, have.values) != (want.threads, want.values) {
            return Err(format!("boundary: '{}' shape {}x{} vs expected {}x{}", want.name, have.threads, have.values, want.threads, want.values));
        }
        if have.elem_bits != want.elem_bits {
            return Err(format!("boundary: '{}' is {} bits, expected {}", want.name, have.elem_bits, want.elem_bits));
        }
        let regs_per = (32 / want.elem_bits.min(32)) as usize;
        let p = plan_convert_tables(&have.table, &want.table, want.threads, want.values, regs_per.max(1));
        if have.storage != want.storage && !matches!((&have.storage, &want.storage), (Storage::Vgpr, Storage::Agpr) | (Storage::Agpr, Storage::Vgpr)) {
            return Err(format!("boundary: '{}' storage {:?} vs expected {:?}", want.name, have.storage, want.storage));
        }
        rep.frags.push((want.name.clone(), p.movement, p.cost));
    }
    for r in &cons.lds_reused {
        if !prod.lds_freed.contains(r) {
            return Err(format!("boundary: LDS region '{r}' reused but not freed by the producer"));
        }
    }
    for (what, lo, n) in &cons.pinned {
        if !prod.pinned.iter().any(|(w, l2, n2)| w == what && l2 == lo && n2 == n) {
            return Err(format!("boundary: pinned range '{what}' {lo}+{n} not provided"));
        }
    }
    if prod.vm > cons.vm || prod.lgkm > cons.lgkm {
        return Err(format!("boundary: outstanding vm {} / lgkm {} exceed the consumer's tolerance {} / {}", prod.vm, prod.lgkm, cons.vm, cons.lgkm));
    }
    if prod.barrier_phase != cons.barrier_phase {
        return Err(format!("boundary: barrier phase {} vs {}", prod.barrier_phase, cons.barrier_phase));
    }
    if prod.roles != cons.roles {
        return Err(format!("boundary: roles {:?} vs {:?}", prod.roles, cons.roles));
    }
    Ok(rep)
}
