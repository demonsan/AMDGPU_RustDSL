//! Protocol engine (gpu_verify v4): LDS tiles / rings / private regions AND global-memory regions. The four axes - ARRIVAL (s_barrier), COMPLETION (wait_vm / wait_lds /
//! fence_release), VISIBILITY (barrier for LDS; release -> barrier -> acquire for global) and GENERATION (version algebra, periodic) - are separate events here; see
//! docs/gpu-semantic-protocol.md. Legacy GV-TILE-* codes stay the primary names of the LDS instance; `proto_alias` maps every verdict onto GV-PROTO-*.
//! (earlier header: LDS tile lifetime / versioning, gpu_verify v1.1.) Design: docs/gpu-semantics-verifier.md section 3.x.
//!
//! The kernel declares tiles / rings (`lds_tile`, `lds_ring`) and marks, with no-code statements, which LDS writes / DMA produce which VERSION of which slot
//! (`tile_produce`) and which reads expect which version (`tile_consume`). This module executes the kernel's control skeleton concretely for each wave ROLE
//! (the two branches of the wave-divergent `if`), unrolling every loop for 1..=4 iterations, evaluating slot / version expressions (ring counters are compile-time
//! sequences), and checks, per slot, the producer -> consumer protocol across the barrier-delimited phases:
//!   * a consumer must see exactly the expected version, published (DMA / DS writes COMPLETED by the producing wave, then a barrier) before its phase;
//!   * no other wave may overwrite (or still be landing) a slot while a consumer reads it (read of a slot overwritten by the next iteration);
//!   * concurrent writes of one slot by different waves are only allowed for the same version (cooperative fill);
//!   * reads must be complete (`wait_lds(0)`) before the next barrier so that the barrier really separates them from the next writer.

use std::collections::{BTreeSet, HashMap, HashSet};

use amdgpu_dsl::{AtomicOp, Function, Op, Opnd, Stmt, ValueId, VarId};

use crate::gpu_verify::{Report, Severity};

#[derive(Clone, Copy, Debug, PartialEq)]
enum CV {
    K(i64),
    U,
}

#[derive(Clone, Debug)]
enum Item {
    Barrier { ord: u32 },
    /// `s_waitcnt vmcnt(count)`: completes the VMEM operations of this wave except the `count` most recent ones (loads return in issue order)
    WaitVm { ord: u32, count: u8 },
    /// `s_waitcnt lgkmcnt(0)`: completes every LDS operation this wave issued before it. (A counted `wait_lds(n > 0)` is not a completion
    /// point for the protocol: the scheduler may reorder independent DS reads, so "n later operations" is not stable through compilation.)
    WaitLds { ord: u32 },
    /// `s_waitcnt lgkmcnt(n > 0)` of an EXACT kernel (the author's order is final, so the in-order DS returns make it a completion point for every
    /// DS operation followed by at least n later DS operations)
    WaitLdsN { ord: u32, count: u8 },
    /// a tile marker opened an event here (the event's `start`)
    Open,
    Access { dma: bool },
    /// `fence_release`: waits vmcnt(0)/lgkmcnt(0) (completion) and, with `wb`, writes L2 back (device-scope visibility of plain stores)
    Release { ord: u32 },
    /// `fence_acquire`: `inv` invalidates the non-coherent caches
    Acquire { ord: u32, inv: bool },
}

#[derive(Clone, Debug)]
struct Ev {
    produce: bool,
    tile: usize,
    slot: i64,
    ver: i64,
    ord: u32,
    stmt: usize,
    start: usize,
    last: usize,
    dma: bool,
    ds: bool,
    n_acc: usize,
    /// a global-memory region event (stores / loads), not LDS
    global: bool,
    /// every read of a consume event was a coherent (sc1) load
    coherent: bool,
    /// the iteration of the OUTERMOST enclosing loop when the event opened (-1 outside loops): the period claim is checked per outer iteration
    outer: i64,
}

struct Trace {
    items: Vec<Item>,
    evs: Vec<Ev>,
    /// the outcomes taken at wave-uniform runtime ifs: (stmt, outcome, workgroup-uniform)
    path: Vec<(usize, bool, bool)>,
    /// `analyse_trace` of every event, computed once per trace (a trace is checked against several partner traces)
    infos: Vec<TraceInfo>,
    /// `bar_from[k]`: a barrier sits at item index k or later (suffix flags, len = items + 1)
    bar_from: Vec<bool>,
}

/// Regions whose markers the intra-workgroup simulator must not interpret: counted-trust regions and inter-block (last-arriver) regions.
fn region_skipped(f: &Function, name: &str) -> bool {
    f.contract.buf_trusts.iter().any(|(n, _)| n == name) || f.contract.last_arrivers.iter().any(|l| l.region == name)
}

/// The v4 name of a verdict (obligation O1 complete-before-publish, O2 publish-before-read, O3 read-before-overwrite, O4 generation-consistency).
pub(crate) fn proto_alias(code: &str, msg: &str) -> &'static str {
    match code {
        "GV-TILE-STALE" | "GV-PROTO-UNPUBLISHED" | "GV-PROTO-NOT-VISIBLE" | "GV-PROTO-NO-COMPLETION" | "GV-PROTO-STALE-GEN" => {
            if msg.contains("NOT released") || msg.contains("no acquire") || msg.contains("before the acquire") {
                "GV-PROTO-NOT-VISIBLE"
            } else if msg.contains("never completed") || msg.contains("not completed") || msg.contains("not completed by this wave") {
                "GV-PROTO-NO-COMPLETION"
            } else if msg.contains("expects version") || msg.contains("generation") {
                "GV-PROTO-STALE-GEN"
            } else {
                "GV-PROTO-UNPUBLISHED"
            }
        }
        "GV-TILE-RACE" | "GV-PROTO-RACE" => "GV-PROTO-RACE",
        "GV-TILE-PENDING" => "GV-PROTO-NO-COMPLETION",
        "GV-TILE-INDEX" => "GV-PROTO-STALE-GEN",
        "GV-BAR" => "GV-PROTO-ARRIVAL-MISMATCH",
        "GV-TILE-UNROLL-LIMIT" | "GV-TILE-UNSUPPORTED" => "GV-PROTO-UNROLL-LIMIT",
        "GV-ASSUME" if msg.contains("TRUSTED") => "GV-PROTO-TRUST",
        _ => "",
    }
}

pub(crate) fn divergent_ifs(f: &Function) -> HashSet<usize> {
    let mut tv: HashSet<ValueId> = HashSet::new();
    let mut tvars: HashSet<VarId> = HashSet::new();
    let tainted = |o: &Opnd, tv: &HashSet<ValueId>, tvars: &HashSet<VarId>| match o {
        Opnd::Val(v) => tv.contains(v),
        Opnd::Var(v) => tvars.contains(v),
        _ => false,
    };
    loop {
        let mut changed = false;
        for st in f.stmts() {
            match st {
                Stmt::Let { dst, op } => {
                    let t = matches!(op, Op::WorkitemId) || op.operands().iter().any(|o| tainted(o, &tv, &tvars));
                    if t && tv.insert(*dst) {
                        changed = true;
                    }
                }
                Stmt::SetVar { var, value } => {
                    if tainted(value, &tv, &tvars) && tvars.insert(*var) {
                        changed = true;
                    }
                }
                Stmt::BeginLoop { vars, init } | Stmt::DeclareVars { vars, init } => {
                    for (v, i0) in vars.iter().zip(init.iter()) {
                        if tainted(i0, &tv, &tvars) && tvars.insert(*v) {
                            changed = true;
                        }
                    }
                }
                _ => {}
            }
        }
        if !changed {
            break;
        }
    }
    let mut out = HashSet::new();
    for (i, st) in f.stmts().iter().enumerate() {
        if let Stmt::BeginIf { cond } = st {
            if tainted(cond, &tv, &tvars) {
                out.insert(i);
            }
        }
    }
    out
}

fn matching(f: &Function, begin: usize) -> usize {
    let mut depth = 0usize;
    for (k, s) in f.stmts().iter().enumerate().skip(begin) {
        match s {
            Stmt::BeginLoop { .. } => depth += 1,
            Stmt::EndLoop => {
                depth -= 1;
                if depth == 0 {
                    return k;
                }
            }
            _ => {}
        }
    }
    f.stmts().len()
}

/// (else index or end index, end index) of the if starting at `begin`.
fn if_span(f: &Function, begin: usize) -> (Option<usize>, usize) {
    let mut depth = 0usize;
    let mut els = None;
    for (k, s) in f.stmts().iter().enumerate().skip(begin) {
        match s {
            Stmt::BeginIf { .. } => depth += 1,
            Stmt::Else if depth == 1 => els = Some(k),
            Stmt::EndIf => {
                depth -= 1;
                if depth == 0 {
                    return (els, k);
                }
            }
            _ => {}
        }
    }
    (els, f.stmts().len())
}

/// (is_write, is_coherent) of a global-memory access statement.
fn is_global_access(st: &Stmt) -> Option<(bool, bool)> {
    match st {
        Stmt::StoreU16 { .. } | Stmt::StoreF32 { .. } | Stmt::StoreF32x2 { .. } | Stmt::StoreF32x4 { .. } => Some((true, false)),
        Stmt::StoreF32Wt { .. } => Some((true, true)),
        // buffer forms: write-through / coherent with `sc1`
        Stmt::BufferStoreN { cache, .. } => Some((true, cache.sc1)),
        Stmt::BufferStore { .. } => Some((true, false)),
        Stmt::StoreGlobalN { .. } | Stmt::AtomicGlobalNoRet { .. } => Some((true, false)),
        Stmt::Let { op, .. } => match op {
            Op::LoadU16 { .. } | Op::LoadF32 { .. } | Op::LoadGlobalB128 { .. } => Some((false, false)),
            Op::LoadF32Coherent { .. } => Some((false, true)),
            Op::BufferLoadN { cache, .. } => Some((false, cache.sc1)),
            Op::BufferLoad { .. } => Some((false, false)),
            Op::LoadGlobalN { .. } | Op::LoadGlobalD16 { .. } => Some((false, false)),
            Op::AtomicGlobal { .. } => Some((true, false)),
            _ => None,
        },
        _ => None,
    }
}

fn is_lds_access(st: &Stmt) -> Option<(bool, bool)> {
    // (dma, is_write)
    match st {
        Stmt::LoadGlobalLds { .. } | Stmt::BufferLoadLds { .. } => Some((true, true)),
        Stmt::StoreLdsB32 { .. } | Stmt::StoreLdsB128 { .. } | Stmt::StoreLdsB64 { .. } | Stmt::StoreLdsN { .. } | Stmt::StoreLds2 { .. } | Stmt::AtomicLdsNoRet { .. } => Some((false, true)),
        Stmt::Let { op, .. } => match op {
            Op::LoadLdsB32 { .. } | Op::LoadLdsB128 { .. } | Op::LoadLdsB64 { .. } | Op::LoadLdsTrB8 { .. } | Op::LoadLdsTr4B8 { .. } | Op::LoadLdsB128x2 { .. } | Op::LoadLdsN { .. } | Op::LoadLdsD16 { .. } | Op::LoadLds2 { .. } | Op::LoadLdsTrB16 { .. } => Some((false, false)),
            Op::AtomicLds { .. } => Some((false, true)),
            _ => None,
        },
        _ => None,
    }
}

struct Sim<'a> {
    f: &'a Function,
    role: usize,
    n: usize,
    div: &'a HashSet<usize>,
    vals: HashMap<ValueId, CV>,
    vars: HashMap<VarId, CV>,
    items: Vec<Item>,
    evs: Vec<Ev>,
    ord: u32,
    open: Option<usize>,
    diags: &'a mut Vec<(&'static str, Severity, usize, String)>,
    untagged: &'a mut BTreeSet<usize>,
    steps: usize,
    break_taken: bool,
    decisions: Vec<bool>,
    dpos: usize,
    taken: Vec<(usize, bool, bool)>,
    loop_depth: usize,
    /// iteration index of the outermost loop being simulated (-1 outside loops)
    outer: i64,
    in_role: bool,
    unsupported: &'a mut Vec<(usize, String)>,
    /// runtime `break_if`s met in the current iteration of the outermost loop (their ordinal keys the exit decision, so that the two roles' loops,
    /// built alike, pair exit for exit)
    brk_ord: usize,
    /// inside the body of an E5 exit taken on this path (runs once: its runtime ifs are decisions)
    exit_body: bool,
}

impl<'a> Sim<'a> {
    fn cv(&self, o: &Opnd) -> CV {
        match o {
            Opnd::I32(c) => CV::K(*c as i64),
            Opnd::Val(v) => self.vals.get(v).copied().unwrap_or(CV::U),
            Opnd::Var(v) => self.vars.get(v).copied().unwrap_or(CV::U),
            Opnd::F32(_) => CV::U,
        }
    }

    fn eval(&self, op: &Op) -> CV {
        use CV::*;
        let bin = |a: &Opnd, b: &Opnd, f: &dyn Fn(i64, i64) -> Option<i64>| match (self.cv(a), self.cv(b)) {
            (K(x), K(y)) => f(x, y).map_or(U, K),
            _ => U,
        };
        match op {
            Op::ConstI32(c) => K(*c as i64),
            Op::MovS { a, .. } | Op::MovV { a, .. } | Op::BitcastI32(a) => self.cv(a),
            Op::AddK { a, k } => match self.cv(a) {
                K(x) => K((x + *k as i64) as i32 as i64),
                U => U,
            },
            Op::Cmp { pred, ty, a, b } if matches!(ty, amdgpu_dsl::CmpTy::I32 | amdgpu_dsl::CmpTy::U32) => {
                let u = matches!(ty, amdgpu_dsl::CmpTy::U32);
                bin(a, b, &|x, y| {
                    let (x, y) = if u { ((x as i32 as u32) as i64, (y as i32 as u32) as i64) } else { (x as i32 as i64, y as i32 as i64) };
                    use amdgpu_dsl::CmpPred::*;
                    Some(match pred {
                        Lt => x < y,
                        Le => x <= y,
                        Gt => x > y,
                        Ge => x >= y,
                        Eq => x == y,
                        Ne => x != y,
                        #[allow(unreachable_patterns)]
                        _ => return None,
                    } as i64)
                })
            }
            Op::AddI32(a, b) => bin(a, b, &|x, y| Some((x + y) as i32 as i64)),
            Op::MulI32(a, b) => bin(a, b, &|x, y| Some((x * y) as i32 as i64)),
            Op::ShlI32(a, b) => bin(a, b, &|x, y| if (0..32).contains(&y) { Some(((x as i32) << y) as i64) } else { None }),
            Op::LshrI32(a, b) => bin(a, b, &|x, y| if (0..32).contains(&y) { Some(((x as i32 as u32) >> y) as i64) } else { None }),
            Op::AndI32(a, b) => bin(a, b, &|x, y| Some(x & y)),
            Op::XorI32(a, b) => bin(a, b, &|x, y| Some(x ^ y)),
            Op::MinI32(a, b) => bin(a, b, &|x, y| Some(x.min(y))),
            Op::CmpLtI32(a, b) => bin(a, b, &|x, y| Some((x < y) as i64)),
            Op::CmpGeI32(a, b) => bin(a, b, &|x, y| Some((x >= y) as i64)),
            Op::NotI1(a) => match self.cv(a) {
                K(x) => K((x == 0) as i64),
                U => U,
            },
            Op::Select { ty, cond, a, b } if matches!(ty, amdgpu_dsl::DType::I32 | amdgpu_dsl::DType::U32) => match self.cv(cond) {
                K(c) => {
                    if c != 0 {
                        self.cv(a)
                    } else {
                        self.cv(b)
                    }
                }
                U => U,
            },
            _ => U,
        }
    }

    fn diag(&mut self, code: &'static str, sev: Severity, stmt: usize, msg: String) {
        if !self.diags.iter().any(|d| d.0 == code && d.2 == stmt) {
            self.diags.push((code, sev, stmt, msg));
        }
    }

    fn close(&mut self) {
        self.open = None;
    }

    /// Outcome of a wave-uniform runtime if: replay the recorded prefix, then take `true` and remember that `false` is still to be explored.
    fn choose(&mut self, stmt: usize, uniform: bool) -> bool {
        let d = if self.dpos < self.decisions.len() { self.decisions[self.dpos] } else { true };
        self.dpos += 1;
        self.taken.push((stmt, d, uniform));
        d
    }

    fn open_marker(&mut self, i: usize, produce: bool, tile: &str, slot: &Opnd, ver: &Opnd) {
        let Some(ti) = self.f.contract.tiles.iter().position(|t| t.name == tile) else { return };
        // a private (per-wave) region: the slot operand is the WAVE KEY, checked symbolically against the access address by the interval analysis; the simulation needs no value for it
        let private = self.f.contract.tiles[ti].private;
        let global = self.f.contract.tiles[ti].global;
        if region_skipped(self.f, tile) {
            self.open = None; // a TRUSTED region (buf_trust) or an inter-block last-arriver region (verify_last_arrivers): not simulated here
            return;
        }
        let slot_cv = if private { CV::K(0) } else { self.cv(slot) };
        match (slot_cv, self.cv(ver)) {
            (CV::K(s), CV::K(v)) => {
                // a slot given as its byte offset (`slot_by_offset`): base + slot * slot_size
                let t = &self.f.contract.tiles[ti];
                let s = if t.slot_by_offset && !private {
                    let d = s;
                    if t.slot_size <= 0 || d.rem_euclid(t.slot_size) != 0 {
                        self.diag("GV-TILE-INDEX", Severity::Error, i, format!("stmt {i}: tile `{tile}`: the slot offset {s} (from the base) is not a multiple of the slot size {}", t.slot_size));
                        self.open = None;
                        return;
                    }
                    d.div_euclid(t.slot_size)
                } else {
                    s
                };
                let slots = self.f.contract.tiles[ti].slots as i64;
                if !private && (s < 0 || s >= slots) {
                    self.diag("GV-TILE-INDEX", Severity::Error, i, format!("stmt {i}: tile `{tile}` slot {s} is outside the declared ring of {slots} slot(s)"));
                    self.open = None;
                    return;
                }
                let ev = Ev { produce, tile: ti, slot: s, ver: v, ord: self.ord, stmt: i, start: self.items.len(), last: self.items.len(), dma: false, ds: false, n_acc: 0, global, coherent: true, outer: self.outer };
                self.evs.push(ev);
                let idx = self.evs.len() - 1;
                self.items.push(Item::Open);
                self.open = Some(idx);
            }
            _ => {
                self.diag("GV-TILE-INDEX", Severity::Error, i, format!("stmt {i}: the slot / version of the `{tile}` marker is not a compile-time function of the loop counters (the protocol cannot be simulated; derive it from ring counters or constants)"));
                self.open = None;
            }
        }
    }

    fn exec(&mut self, start: usize, end: usize) {
        let f = self.f;
        let mut i = start;
        while i < end {
            self.steps += 1;
            if self.steps > 2_000_000 {
                return;
            }
            let st = &f.stmts()[i];
            if let Some((write, coh)) = is_global_access(st) {
                if let Some(e) = self.open {
                    if self.evs[e].global {
                        if self.evs[e].produce != write {
                            self.diag("GV-TILE-KIND", Severity::Error, i, format!("stmt {i}: a global {} is issued under a buf_{} marker (stmt {})", if write { "write" } else { "read" }, if self.evs[e].produce { "produce" } else { "consume" }, self.evs[e].stmt));
                        } else {
                            // a global-region access: not an LDS-direct DMA (the counted-vmcnt rule only counts in-order DMA loads)
                            self.items.push(Item::Access { dma: false });
                            let pos = self.items.len() - 1;
                            let ev = &mut self.evs[e];
                            ev.last = pos;
                            ev.n_acc += 1;
                            ev.dma = true;
                            if !write && !coh {
                                ev.coherent = false;
                            }
                        }
                    }
                }
            }
            if let Some((dma, write)) = is_lds_access(st) {
                match self.open {
                    Some(e) if self.evs[e].global => {
                        self.diag("GV-TILE-KIND", Severity::Error, i, format!("stmt {i}: an LDS access is issued under a buf_ (global region) marker"));
                    }
                    Some(e) => {
                        let produce = self.evs[e].produce;
                        if produce != write {
                            self.diag("GV-TILE-KIND", Severity::Error, i, format!("stmt {i}: an LDS {} is issued under a tile_{} marker (stmt {})", if write { "write / DMA" } else { "read" }, if produce { "produce" } else { "consume" }, self.evs[e].stmt));
                        } else {
                            self.items.push(Item::Access { dma });
                            let pos = self.items.len() - 1;
                            let ev = &mut self.evs[e];
                            ev.last = pos;
                            ev.n_acc += 1;
                            if dma {
                                ev.dma = true;
                            } else {
                                ev.ds = true;
                            }
                        }
                    }
                    None => {
                        self.untagged.insert(i);
                    }
                }
            }
            match st {
                Stmt::Let { dst, op } => {
                    let v = self.eval(op);
                    self.vals.insert(*dst, v);
                }
                Stmt::SetVar { var, value } => {
                    let v = self.cv(value);
                    self.vars.insert(*var, v);
                }
                // E5 scope variables: their initial values (an Undef init is unknown until set)
                Stmt::DeclareVars { vars, init } => {
                    for (v, i0) in vars.iter().zip(init.iter()) {
                        let k = self.cv(i0);
                        self.vars.insert(*v, k);
                    }
                }
                Stmt::Barrier => {
                    self.items.push(Item::Barrier { ord: self.ord });
                    self.ord += 1;
                    self.close();
                }
                Stmt::FenceRelease { .. } => self.items.push(Item::Release { ord: self.ord }),
                Stmt::FenceAcquire { inv } => self.items.push(Item::Acquire { ord: self.ord, inv: *inv }),
                Stmt::WaitVm { count } => self.items.push(Item::WaitVm { ord: self.ord, count: *count }),
                Stmt::WaitLds { count: 0 } => self.items.push(Item::WaitLds { ord: self.ord }),
                // the combined s_waitcnt (exact mode): both counters
                Stmt::WaitCnt { vm, lgkm } => {
                    self.items.push(Item::WaitVm { ord: self.ord, count: *vm });
                    if *lgkm == 0 {
                        self.items.push(Item::WaitLds { ord: self.ord });
                    } else if f.exact {
                        self.items.push(Item::WaitLdsN { ord: self.ord, count: *lgkm });
                    }
                }
                Stmt::WaitLds { count } if *count > 0 && f.exact => self.items.push(Item::WaitLdsN { ord: self.ord, count: *count }),
                Stmt::TileProduce { tile, slot, version } => self.open_marker(i, true, tile, slot, version),
                Stmt::TileConsume { tile, slot, version } => self.open_marker(i, false, tile, slot, version),
                Stmt::TileEnd => self.close(),
                Stmt::BeginLoop { vars, init } => {
                    self.close();
                    let close = matching(f, i);
                    for (v, i0) in vars.iter().zip(init.iter()) {
                        let k = self.cv(i0);
                        self.vars.insert(*v, k);
                    }
                    self.loop_depth += 1;
                    for it in 0..self.n {
                        if self.loop_depth == 1 {
                            self.outer = it as i64;
                            self.brk_ord = 0;
                        }
                        self.exec_loop_body(i + 1, close);
                        if self.break_taken {
                            break;
                        }
                    }
                    self.break_taken = false;
                    self.loop_depth -= 1;
                    if self.loop_depth == 0 {
                        self.outer = -1;
                    }
                    self.close();
                    i = close + 1;
                    continue;
                }
                Stmt::BreakIf { cond } => {
                    match self.cv(cond) {
                        CV::K(c) => {
                            if c != 0 {
                                self.break_taken = true;
                                return;
                            }
                        }
                        // a runtime exit of the outermost loop: in its LAST simulated iteration each exit point is a decision (taken / not taken), so the
                        // code after every exit (e.g. a per-exit drain of a loop unrolled by two) is simulated too; keyed by the exit's ordinal in the
                        // iteration (workgroup-uniform: both roles leave at the same exit)
                        // (an exit with nothing protocol-relevant after it in the body - the loop test at the end - leaves exactly like the last
                        // iteration ending: no decision)
                        CV::U if self.loop_depth == 1 && self.outer == self.n as i64 - 1 && self.exit_matters(i) => {
                            let key = usize::MAX - self.brk_ord;
                            self.brk_ord += 1;
                            if self.choose(key, true) {
                                self.break_taken = true;
                                return;
                            }
                        }
                        CV::U => {}
                    }
                }
                Stmt::BeginIf { cond } => {
                    self.close();
                    let (els, endif) = if_span(f, i);
                    let then_range = (i + 1, els.unwrap_or(endif));
                    let else_range = els.map(|e| (e + 1, endif));
                    // waits and fences count too: a role-specific completion point (e.g. the retiring wait_lds(0) of reads kept in flight across a barrier) must be
                    // simulated in that role, not dropped
                    let proto = |a: usize, b: usize| f.stmts()[a..b].iter().any(|s| matches!(s, Stmt::Barrier | Stmt::TileProduce { .. } | Stmt::TileConsume { .. } | Stmt::WaitLds { .. } | Stmt::WaitVm { .. } | Stmt::FenceRelease { .. } | Stmt::FenceAcquire { .. }));
                    let anything = |a: usize, b: usize| f.stmts()[a..b].iter().any(|s| matches!(s, Stmt::Barrier | Stmt::TileProduce { .. } | Stmt::TileConsume { .. }) || is_lds_access(s).is_some());
                    let has_proto = proto(then_range.0, then_range.1) || else_range.map_or(false, |r| proto(r.0, r.1));
                    let has_any = anything(then_range.0, then_range.1) || else_range.map_or(false, |r| anything(r.0, r.1));
                    let run_arm = |s: &mut Self, then: bool| {
                        if then {
                            s.exec(then_range.0, then_range.1);
                        } else if let Some(r) = else_range {
                            s.exec(r.0, r.1);
                        }
                    };
                    // an E5 loop exit with its own body (`if c { drain..; break_if(c) }`): like a runtime `break_if` - in the LAST simulated iteration of the
                    // outermost loop it is a decision (taken: the arm runs and leaves; keyed by the exit's ordinal, workgroup-uniform), otherwise it is
                    // not taken (the iteration continues)
                    let exit_arm = else_range.is_none()
                        && then_range.1 > then_range.0
                        && matches!(&f.stmts()[then_range.1 - 1], Stmt::BreakIf { cond: c2 } if c2 == cond)
                        && matches!(self.cv(cond), CV::U);
                    if exit_arm && self.loop_depth >= 1 && !(self.div.contains(&i) && !self.in_role) {
                        let take = if self.loop_depth == 1 && self.outer == self.n as i64 - 1 {
                            let key = usize::MAX - self.brk_ord;
                            self.brk_ord += 1;
                            self.choose(key, true)
                        } else {
                            false
                        };
                        if take {
                            // the arm body; its final break_if is a runtime condition, so leave explicitly. It runs ONCE on this path, so a
                            // runtime if inside it (e.g. a drain's tail test) is a decision like one outside the loop
                            let was = self.exit_body;
                            self.exit_body = true;
                            self.exec(then_range.0, then_range.1 - 1);
                            self.exit_body = was;
                            self.close();
                            self.break_taken = true;
                            return;
                        }
                        self.close();
                        i = endif + 1;
                        continue;
                    }
                    if self.div.contains(&i) && !self.in_role && has_proto {
                        // the wave ROLE split: role 0 takes the then arm, role 1 the else arm
                        self.in_role = true;
                        run_arm(self, self.role == 0);
                        self.in_role = false;
                    } else {
                        match self.cv(cond) {
                            CV::K(c) => run_arm(self, c != 0),
                            CV::U => {
                                if has_any {
                                    if self.loop_depth > 0 && !self.exit_body {
                                        self.unsupported.push((i, format!("stmt {i}: a runtime if INSIDE a loop contains barriers / tile markers / LDS accesses: its two outcomes are not explored per iteration (a 2^iterations path set); hoist it out of the loop, make both arms identical, or restructure")));
                                    } else {
                                        let uniform = !self.div.contains(&i);
                                        let d = self.choose(i, uniform);
                                        run_arm(self, d);
                                    }
                                }
                            }
                        }
                    }
                    self.close();
                    if self.break_taken {
                        return;
                    }
                    i = endif + 1;
                    continue;
                }                _ => {}
            }
            i += 1;
        }
    }

    /// Does anything protocol-relevant (barrier, tile marker, wait, fence, LDS access) follow the `break_if` at `i` in its loop body?
    fn exit_matters(&self, i: usize) -> bool {
        let f = self.f;
        let mut depth = 0usize;
        for s in &f.stmts()[i + 1..] {
            match s {
                Stmt::BeginLoop { .. } => depth += 1,
                Stmt::EndLoop if depth == 0 => return false,
                Stmt::EndLoop => depth -= 1,
                Stmt::Barrier | Stmt::TileProduce { .. } | Stmt::TileConsume { .. } | Stmt::WaitLds { .. } | Stmt::WaitVm { .. } | Stmt::FenceRelease { .. } | Stmt::FenceAcquire { .. } => return true,
                _ if is_lds_access(s).is_some() => return true,
                _ => {}
            }
        }
        false
    }

    fn exec_loop_body(&mut self, start: usize, end: usize) {
        self.exec(start, end);
    }
}

#[derive(Clone)]
struct TraceInfo {
    /// item index at which the event is complete in its own wave
    done: Option<usize>,
    /// first phase at which every other wave may rely on it (barrier after completion, + a release for global regions)
    publish: Option<u32>,
    /// phase at which a consumer's reads are complete
    done_ord: u32,
    /// global produce: a fence_release sits between the last write and the publishing barrier
    released: bool,
}

fn analyse_trace(items: &[Item], ev: &Ev) -> TraceInfo {
    let after = ev.last + 1;
    let find = |pred: &dyn Fn(&Item) -> bool| items.iter().enumerate().skip(after).find(|(_, it)| pred(it)).map(|(k, _)| k);
    let ord_of = |it: &Item, dflt: u32| match it {
        Item::WaitVm { ord, .. } | Item::WaitLds { ord } | Item::WaitLdsN { ord, .. } | Item::Release { ord } | Item::Acquire { ord, .. } | Item::Barrier { ord } => *ord,
        _ => dflt,
    };
    // the event's DS accesses are complete at the first wait_lds(0) after its last access. Barriers in between do not matter for completion (the access
    // window then spans them; the race check uses it). The scheduler never moves an LDS access below a later lgkmcnt wait (sched/order.rs).
    let ds_done = || -> Option<usize> {
        // lgkmcnt(0); in an exact kernel also lgkmcnt(n) once n later DS operations of this wave were issued (in-order DS returns)
        let mut later = 0usize;
        for (k, it) in items.iter().enumerate().skip(after) {
            match it {
                Item::WaitLds { .. } => return Some(k),
                Item::Access { dma: false } => later += 1,
                Item::WaitLdsN { count, .. } if later >= *count as usize => return Some(k),
                _ => {}
            }
        }
        None
    };
    if ev.produce {
        let done = if ev.global {
            // COMPLETION of global writes: wait_vm(0) or a fence_release (which waits vmcnt(0) itself)
            find(&|it| matches!(it, Item::WaitVm { count: 0, .. } | Item::Release { .. }))
        } else {
            // a DMA is complete at the first vmcnt(n) after which at least n LATER LDS-direct DMAs of this wave are in flight (in-order load returns; other
            // VMEM operations are not counted, which only makes the rule more conservative)
            let vm = if ev.dma {
                let mut later = 0usize;
                let mut hit = None;
                for (k, it) in items.iter().enumerate().skip(after) {
                    match it {
                        Item::Access { dma: true } => later += 1,
                        Item::WaitVm { count, .. } if later >= *count as usize => {
                            hit = Some(k);
                            break;
                        }
                        _ => {}
                    }
                }
                hit
            } else {
                None
            };
            let ds = if ev.ds { ds_done() } else { None };
            match (ev.dma, ev.ds) {
                (true, true) => match (vm, ds) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    _ => None,
                },
                (true, false) => vm,
                (false, true) => ds,
                _ => None,
            }
        };
        match done {
            Some(d) => {
                let ord = ord_of(&items[d], ev.ord);
                let bar = items.iter().enumerate().skip(d + 1).find_map(|(k, it)| if let Item::Barrier { ord } = it { Some((k, *ord)) } else { None });
                let released = if ev.global { items.iter().enumerate().skip(after).any(|(k, it)| matches!(it, Item::Release { .. }) && bar.map_or(true, |(bk, _)| k < bk)) } else { true };
                TraceInfo { done: Some(d), publish: bar.map(|(_, o)| o + 1), done_ord: ord, released }
            }
            None => TraceInfo { done: None, publish: None, done_ord: ev.ord, released: false },
        }
    } else {
        // consumer: its reads are complete at the retiring wait (wait_lds(0) for LDS; wait_vm(0) / a fence for global reads). The read's
        // lifetime is [issue phase, phase of the retiring wait]: reads may stay in flight ACROSS barriers (the vendor's V read-ahead) - the race check
        // then requires that no producer writes the slot in any phase of that window.
        let d = if ev.global { find(&|it| matches!(it, Item::WaitVm { count: 0, .. } | Item::Release { .. } | Item::Acquire { .. })) } else { ds_done() };
        match d {
            Some(d) => TraceInfo { done: Some(d), publish: None, done_ord: ord_of(&items[d], ev.ord), released: true },
            None => {
                let b = items.iter().skip(after).find_map(|it| if let Item::Barrier { ord } = it { Some(ord + 1) } else { None });
                TraceInfo { done: None, publish: None, done_ord: b.unwrap_or(ev.ord), released: true }
            }
        }
    }
}
/// Is every tile marker operand an AFFINE / PERIODIC function of loop counters? (constants, + - of counters, * << & ^ by constants; each counter used must be updated by
/// `+ const` or `^ const` of itself and start at a constant.) Only then does simulating a bounded number of iterations cover every trip count.
fn periodic_cone(f: &Function) -> Result<usize, String> {
    fn konst(f: &Function, o: &Opnd) -> Option<i64> {
        match o {
            Opnd::I32(c) => Some(*c as i64),
            Opnd::Val(v) => f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op: Op::ConstI32(c) } = s { if dst == v { Some(*c as i64) } else { None } } else { None }),
            _ => None,
        }
    }
    fn def<'a>(f: &'a Function, v: ValueId) -> Option<&'a Op> {
        f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if *dst == v { Some(op) } else { None } } else { None })
    }
    // a constant through authored moves (exact mode: `s_mov_b32 sN, k`)
    fn konst_mov(f: &Function, o: &Opnd) -> Option<i64> {
        match o {
            Opnd::I32(c) => Some(*c as i64),
            Opnd::Val(v) => match def(f, *v) {
                Some(Op::ConstI32(c)) => Some(*c as i64),
                Some(Op::MovS { a, .. } | Op::MovV { a, .. }) => konst_mov(f, a),
                Some(Op::AddK { a, k }) => konst_mov(f, a).map(|x| x + *k as i64),
                Some(Op::AddI32(a, b)) => Some(konst_mov(f, a)? + konst_mov(f, b)?),
                _ => None,
            },
            _ => None,
        }
    }
    fn walk(f: &Function, o: &Opnd, seen: &mut HashSet<(bool, u32)>) -> Result<(), String> {
        match o {
            Opnd::I32(_) => Ok(()),
            Opnd::F32(_) => Err("a float".to_string()),
            Opnd::Val(v) => {
                if !seen.insert((false, *v)) {
                    return Ok(());
                }
                let Some(op) = def(f, *v) else { return Err(format!("v{v} (no definition)")) };
                let name = format!("{op:?}");
                let name = name.split(|c: char| c == '(' || c == ' ').next().unwrap_or("?").to_string();
                match op {
                    Op::ConstI32(_) => Ok(()),
                    Op::AddI32(a, b) => {
                        walk(f, a, seen)?;
                        walk(f, b, seen)
                    }
                    // exact-mode views / moves / the author's s_addk: the operand's function
                    Op::MovS { a, .. } | Op::MovV { a, .. } | Op::BitcastI32(a) => walk(f, a, seen),
                    Op::AddK { a, .. } => walk(f, a, seen),
                    Op::MulI32(a, b) => {
                        if konst(f, a).is_none() && konst(f, b).is_none() {
                            return Err("a product of two counters".to_string());
                        }
                        walk(f, a, seen)?;
                        walk(f, b, seen)
                    }
                    Op::ShlI32(a, b) | Op::AndI32(a, b) | Op::XorI32(a, b) | Op::LshrI32(a, b) => {
                        if konst(f, b).is_none() && konst(f, a).is_none() {
                            return Err(format!("{name} of two non-constant operands"));
                        }
                        walk(f, a, seen)?;
                        walk(f, b, seen)
                    }
                    // a finite-state function (compare / select / not) of finite-state ring counters and constants only: periodic with their period
                    Op::Select { .. } | Op::CmpLtI32(..) | Op::CmpGeI32(..) | Op::NotI1(..) | Op::Cmp { .. } if fs_cone(f, &Opnd::Val(*v)) => Ok(()),
                    _ => Err(name),
                }
            }
            Opnd::Var(x) => {
                if !seen.insert((true, x.0)) {
                    return Ok(());
                }
                if finite_state_period(f, *x).is_some() {
                    return Ok(());
                }
                let mut ok_update = false;
                let mut depth = 0i32;
                // the last constant set outside every loop (the counter's real initial value); an earlier top-level set with no loop in between
                // is overwritten before any loop sees it (an author's register reused as a temporary first)
                let stmts = f.stmts();
                let mut init_at: Option<usize> = None;
                {
                    let mut d = 0i32;
                    for (k, s) in stmts.iter().enumerate() {
                        match s {
                            Stmt::BeginLoop { .. } => d += 1,
                            Stmt::EndLoop => d -= 1,
                            Stmt::SetVar { var, value } if var == x && d == 0 && konst_mov(f, value).is_some() => init_at = Some(k),
                            _ => {}
                        }
                    }
                }
                let dead_before = |k: usize| init_at.map_or(false, |ia| k < ia && !stmts[k..ia].iter().any(|s| matches!(s, Stmt::BeginLoop { .. })));
                for (sk, s) in stmts.iter().enumerate() {
                    match s {
                        Stmt::SetVar { var, .. } if var == x && depth == 0 && dead_before(sk) => {}
                        Stmt::BeginLoop { .. } if !matches!(s, Stmt::BeginLoop { vars, .. } if vars.contains(x)) => depth += 1,
                        Stmt::EndLoop => depth -= 1,
                        // an E5 scope variable (exact mode): set to a constant outside every loop = its initial value
                        Stmt::SetVar { var, value } if var == x && depth == 0 && konst_mov(f, value).is_some() => {}
                        Stmt::BeginLoop { vars, init } => {
                            if let Some(k) = vars.iter().position(|y| y == x) {
                                if konst(f, &init[k]).is_none() {
                                    return Err(format!("loop counter var{} with a non-constant initial value", x.0));
                                }
                            }
                        }
                        Stmt::SetVar { var, value } if var == x => {
                            let Opnd::Val(u) = value else { return Err(format!("loop counter var{} updated by a non-expression", x.0)) };
                            let good = match def(f, *u) {
                                Some(Op::AddI32(a, b)) | Some(Op::XorI32(a, b)) => (matches!(a, Opnd::Var(y) if y == x) && konst(f, b).is_some()) || (matches!(b, Opnd::Var(y) if y == x) && konst(f, a).is_some()),
                                // the author's s_addk_i32 (exact mode)
                                Some(Op::AddK { a: Opnd::Var(y), .. }) => y == x,
                                _ => false,
                            };
                            if !good {
                                return Err(format!("loop counter var{} updated by something other than `+ const` / `^ const` of itself", x.0));
                            }
                            ok_update = true;
                        }
                        _ => {}
                    }
                }
                if ok_update { Ok(()) } else { Err(format!("loop counter var{} (never updated by an affine step)", x.0)) }
            }
        }
    }
    let mut seen = HashSet::new();
    for (i, s) in f.stmts().iter().enumerate() {
        if let Stmt::TileProduce { tile, slot, version } | Stmt::TileConsume { tile, slot, version } = s {
            if region_skipped(f, tile) {
                continue;
            }
            // the slot of a private (per-wave) region is the wave key: it does not evolve with the loop iteration
            let private = f.contract.tiles.iter().any(|t| &t.name == tile && t.private);
            for o in if private { vec![version] } else { vec![slot, version] } {
                walk(f, o, &mut seen).map_err(|why| format!("stmt {i}: the tile marker depends on {why}"))?;
            }
        }
    }
    // the markers are periodic with lcm(periods of the finite-state counters in their cones) x the power-of-two bound of `period_bits`
    let mut l = 1usize;
    // (conservatively over every finite-state counter of the kernel: a longer claimed period only lengthens the simulated horizon, and the claim is verified)
    for s in f.stmts() {
        if let Stmt::BeginLoop { vars, .. } | Stmt::DeclareVars { vars, .. } = s {
            for x in vars {
                if let Some(p) = finite_state_period(f, *x) {
                    l = lcm(l, p);
                }
            }
        }
    }
    Ok(l)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}
fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

/// A FINITE-STATE ring counter: a loop variable with a constant initial value and exactly one update `x := F(x)` where F uses only `x` and constants
/// (+ ^ & * << >> compare select not). Its sequence x0, F(x0), ... is evaluated here; when it returns to x0 after p steps (a pure cycle, p <= 64) the
/// counter is periodic with period p (e.g. the 3-slot ring index `idx < 2 ? idx + 1 : 0` of the `_t128` cadence: p = 3). A counter stepped by `^ const`
/// is also finite-state, but stays on the affine path (`period_bits`) so the existing kernels' claims do not change.
fn finite_state_period(f: &Function, x: VarId) -> Option<usize> {
    let mut init: Option<i64> = None;
    for s in f.stmts() {
        if let Stmt::BeginLoop { vars, init: iv } = s {
            if let Some(k) = vars.iter().position(|y| *y == x) {
                init = fs_eval(f, &iv[k], x, 0);
            }
        }
    }
    // an E5 scope variable proven a ring counter / shadow by gpu_verify's ring pre-pass (no loop init): its orbit's period
    let Some(x0) = init else { return crate::gpu_verify::ring_period(f, x) };
    let ups: Vec<&Opnd> = f.stmts().iter().filter_map(|s| if let Stmt::SetVar { var, value } = s { if *var == x { Some(value) } else { None } } else { None }).collect();
    if ups.len() != 1 {
        return None;
    }
    // only a genuine selection counter: the plain `+ const` / `^ const` steps are handled (and bounded) by the affine analysis
    if let Opnd::Val(u) = ups[0] {
        if let Some(Op::AddI32(..) | Op::XorI32(..)) = f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if dst == u { Some(op) } else { None } } else { None }) {
            return None;
        }
    }
    let mut cur = x0;
    for p in 1..=64usize {
        cur = fs_eval(f, ups[0], x, cur)?;
        if cur == x0 {
            return Some(p);
        }
    }
    None
}

/// Does the cone of `o` contain only constants and finite-state counters (through finite-state ops)?
fn fs_cone(f: &Function, o: &Opnd) -> bool {
    match o {
        Opnd::I32(_) => true,
        Opnd::F32(_) => false,
        Opnd::Var(x) => finite_state_period(f, *x).is_some(),
        Opnd::Val(v) => match f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if dst == v { Some(op) } else { None } } else { None }) {
            Some(Op::ConstI32(_)) => true,
            Some(op @ (Op::AddI32(..) | Op::XorI32(..) | Op::AndI32(..) | Op::MulI32(..) | Op::ShlI32(..) | Op::LshrI32(..) | Op::CmpLtI32(..) | Op::CmpGeI32(..) | Op::NotI1(..) | Op::Select { .. } | Op::Cmp { .. } | Op::MovS { .. } | Op::BitcastI32(..))) => op.operands().iter().all(|u| fs_cone(f, u)),
            _ => false,
        },
    }
}

/// Host evaluation (i32 semantics) of a finite-state expression in which the variable `x` has the value `xv`; `None` for anything else.
fn fs_eval(f: &Function, o: &Opnd, x: VarId, xv: i64) -> Option<i64> {
    let w = |v: i64| v as i32 as i64;
    match o {
        Opnd::I32(c) => Some(*c as i64),
        Opnd::Var(y) if *y == x => Some(xv),
        Opnd::Val(v) => {
            let op = f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if dst == v { Some(op) } else { None } } else { None })?;
            let e = |u: &Opnd| fs_eval(f, u, x, xv);
            Some(match op {
                Op::ConstI32(c) => *c as i64,
                Op::AddI32(a, b) => w(e(a)? + e(b)?),
                Op::XorI32(a, b) => w(e(a)? ^ e(b)?),
                Op::AndI32(a, b) => w(e(a)? & e(b)?),
                Op::MulI32(a, b) => w(e(a)?.wrapping_mul(e(b)?)),
                Op::ShlI32(a, b) => w(((e(a)? as u32).wrapping_shl(e(b)? as u32)) as i64),
                Op::LshrI32(a, b) => w(((e(a)? as u32).wrapping_shr(e(b)? as u32)) as i64),
                Op::CmpLtI32(a, b) => (e(a)? < e(b)?) as i64,
                Op::MovS { a, .. } | Op::BitcastI32(a) => e(a)?,
                Op::Cmp { pred, ty, a, b } if matches!(ty, amdgpu_dsl::CmpTy::I32 | amdgpu_dsl::CmpTy::U32) => {
                    let (x, y) = (e(a)?, e(b)?);
                    let (x, y) = if matches!(ty, amdgpu_dsl::CmpTy::U32) { ((x as i32 as u32) as i64, (y as i32 as u32) as i64) } else { (x, y) };
                    use amdgpu_dsl::CmpPred::*;
                    (match pred {
                        Lt => x < y,
                        Le => x <= y,
                        Gt => x > y,
                        Ge => x >= y,
                        Eq => x == y,
                        Ne => x != y,
                        _ => return None,
                    }) as i64
                }
                Op::CmpGeI32(a, b) => (e(a)? >= e(b)?) as i64,
                Op::NotI1(a) => (e(a)? == 0) as i64,
                Op::Select { cond, a, b, .. } => {
                    if e(cond)? != 0 {
                        e(a)?
                    } else {
                        e(b)?
                    }
                }
                _ => return None,
            })
        }
        _ => None,
    }
}
/// Conservative bound M such that the tile markers are periodic in the loop iteration with a period dividing 2^M: every `& mask` contributes its width (msb + 1), every `>> k`
/// in the cone adds k (bit j of `x >> k` is bit j + k of x), and a counter toggled by `^ const` contributes 1. (`+ const` counters are affine: only their low bits matter for a
/// mask, and versions only matter through their differences.)
fn period_bits(f: &Function) -> (u32, u32) {
    fn konst(f: &Function, o: &Opnd) -> Option<i64> {
        match o {
            Opnd::I32(c) => Some(*c as i64),
            Opnd::Val(v) => f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op: Op::ConstI32(c) } = s { if dst == v { Some(*c as i64) } else { None } } else { None }),
            _ => None,
        }
    }
    fn def<'a>(f: &'a Function, v: ValueId) -> Option<&'a Op> {
        f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if *dst == v { Some(op) } else { None } } else { None })
    }
    struct Acc {
        mask_bits: u32,
        /// log2 of the product of all constant multipliers / left shifts in the cone (wrap-around guard)
        mul_log2: u32,
        shifts: u32,
        xor_counter: bool,
        seen: HashSet<(bool, u32)>,
    }
    fn walk(f: &Function, o: &Opnd, a: &mut Acc) {
        match o {
            Opnd::Val(v) => {
                if !a.seen.insert((false, *v)) {
                    return;
                }
                let Some(op) = def(f, *v) else { return };
                match op {
                    Op::AndI32(x, y) => {
                        for k in [x, y] {
                            if let Some(c) = konst(f, k) {
                                a.mask_bits = a.mask_bits.max(32 - (c as u32).leading_zeros());
                            }
                        }
                    }
                    // x ^ c: the bits of c flip bits of x, so the per-iteration relation of the (affine) value to its neighbours changes every 2^(width(c)) iterations
                    Op::XorI32(x, y) => {
                        for k in [x, y] {
                            if let Some(c) = konst(f, k) {
                                a.mask_bits = a.mask_bits.max(32 - (c as u32).leading_zeros());
                            }
                        }
                    }
                    Op::LshrI32(_, y) => {
                        if let Some(k) = konst(f, y) {
                            a.shifts += k.max(0) as u32;
                        }
                    }
                    Op::ShlI32(_, y) => {
                        if let Some(k) = konst(f, y) {
                            a.mul_log2 += k.max(0) as u32;
                        }
                    }
                    Op::MulI32(x, y) => {
                        for k in [x, y] {
                            if let Some(c) = konst(f, k) {
                                a.mul_log2 += 64 - (c.unsigned_abs().max(1) - 1).leading_zeros();
                            }
                        }
                    }
                    _ => {}
                }
                for x in op.operands() {
                    walk(f, &x, a);
                }
            }
            Opnd::Var(x) => {
                if !a.seen.insert((true, x.0)) {
                    return;
                }
                for s in f.stmts() {
                    if let Stmt::SetVar { var, value: Opnd::Val(u) } = s {
                        if var == x {
                            if let Some(Op::XorI32(..)) = def(f, *u) {
                                a.xor_counter = true;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // each marker operand is its own cone: the bounds are maxima over cones, not sums over unrelated nodes
    let (mut m_best, mut mul_best, mut xor_any) = (0u32, 0u32, false);
    for s in f.stmts() {
        if let Stmt::TileProduce { tile, slot, version } | Stmt::TileConsume { tile, slot, version } = s {
            if region_skipped(f, tile) {
                continue;
            }
            let private = f.contract.tiles.iter().any(|t| &t.name == tile && t.private);
            let ops: Vec<&Opnd> = if private { vec![version] } else { vec![slot, version] };
            for o in ops {
                let mut a = Acc { mask_bits: 0, mul_log2: 0, shifts: 0, xor_counter: false, seen: HashSet::new() };
                walk(f, o, &mut a);
                let m = if a.mask_bits > 0 { a.mask_bits + a.shifts } else { 0 };
                m_best = m_best.max(m);
                mul_best = mul_best.max(a.mul_log2);
                xor_any |= a.xor_counter;
            }
        }
    }
    (m_best.max(if xor_any { 1 } else { 0 }), mul_best)
}

/// The claimed period must be a TRUE period of what the simulator actually executes: for every marker statement the simulated slot sequence must repeat after `p` iterations
/// and the version must advance by a constant over `p` iterations.
fn verify_period_claim(f: &Function, traces: &[&Trace], p: usize) -> Result<(), String> {
    for (role, t) in traces.iter().enumerate() {
        let mut by_stmt: HashMap<usize, Vec<(i64, i64, i64)>> = HashMap::new();
        for ev in &t.evs {
            by_stmt.entry(ev.stmt).or_default().push((ev.slot, ev.ver, ev.outer));
        }
        for (stmt, execs) in by_stmt {
            // a marker inside an inner (data-dependent) loop runs 0..n times per outer iteration: when every execution within one outer iteration
            // has the same slot and version, the sequence to check is one entry per outer iteration (the period is a property of the outer loop);
            // otherwise every execution counts, as before
            let mut per_iter: Vec<(i64, (i64, i64))> = Vec::new();
            let mut uniform = true;
            for (s, v, o) in &execs {
                match per_iter.last() {
                    Some((lo, sv)) if *lo == *o => {
                        if *sv != (*s, *v) {
                            uniform = false;
                        }
                    }
                    _ => per_iter.push((*o, (*s, *v))),
                }
            }
            let seq: Vec<(i64, i64)> = if uniform && per_iter.len() < execs.len() { per_iter.iter().map(|x| x.1).collect() } else { execs.iter().map(|x| (x.0, x.1)).collect() };
            if seq.len() < p + 2 {
                continue;
            }
            let dv = seq[p].1 - seq[0].1;
            for i in 0..seq.len() - p {
                if seq[i + p].0 != seq[i].0 {
                    return Err(format!("stmt {stmt} (role {role}): the simulated slot sequence {:?}... is not periodic with the claimed period {p}", seq.iter().take(p * 2 + 2).map(|s| s.0).collect::<Vec<_>>()));
                }
                if seq[i + p].1 - seq[i].1 != dv {
                    return Err(format!("stmt {stmt} (role {role}): the simulated version advances by {} then {} over {p} iteration(s): not a constant step, so the claimed period {p} is not a true period", dv, seq[i + p].1 - seq[i].1));
                }
            }
        }
    }
    let _ = f;
    Ok(())
}

fn enumerate_traces(f: &Function, role: usize, n: usize, div: &HashSet<usize>, diags: &mut Vec<(&'static str, Severity, usize, String)>, untagged: &mut BTreeSet<usize>) -> Vec<Trace> {
    let mut out: Vec<Trace> = Vec::new();
    let mut work: Vec<Vec<bool>> = vec![Vec::new()];
    let mut unsupported: Vec<(usize, String)> = Vec::new();
    while let Some(prefix) = work.pop() {
        if out.len() + work.len() > 256 {
            diags.push(("GV-TILE-UNSUPPORTED", Severity::Error, 0, format!("more than 256 control-flow paths through the wave-uniform runtime ifs of role {role}: the tile protocol is not simulated for all of them")));
            break;
        }
        let mut sim = Sim { f, role, n, div, vals: HashMap::new(), vars: HashMap::new(), items: Vec::new(), evs: Vec::new(), ord: 0, open: None, outer: -1, diags, untagged, steps: 0, break_taken: false, decisions: prefix.clone(), dpos: 0, taken: Vec::new(), loop_depth: 0, in_role: false, unsupported: &mut unsupported, brk_ord: 0, exit_body: false };
        sim.exec(0, f.stmts().len());
        if sim.steps > 2_000_000 {
            sim.diags.push(("GV-TILE-UNSUPPORTED", Severity::Error, 0, "the tile simulation exceeded its step budget (2M statements): it is incomplete, so the protocol is NOT certified".to_string()));
        }
        for k in prefix.len()..sim.taken.len() {
            let mut alt: Vec<bool> = sim.taken[..k].iter().map(|t| t.1).collect();
            alt.push(!sim.taken[k].1);
            work.push(alt);
        }
        let taken = sim.taken.clone();
        let (items, evs) = (std::mem::take(&mut sim.items), std::mem::take(&mut sim.evs));
        let infos = evs.iter().map(|ev| analyse_trace(&items, ev)).collect();
        let mut bar_from = vec![false; items.len() + 1];
        for k in (0..items.len()).rev() {
            bar_from[k] = bar_from[k + 1] || matches!(items[k], Item::Barrier { .. });
        }
        out.push(Trace { items, evs, path: taken, infos, bar_from });
    }
    for (stmt, msg) in unsupported {
        if !diags.iter().any(|d| d.0 == "GV-TILE-UNSUPPORTED" && d.2 == stmt) {
            diags.push(("GV-TILE-UNSUPPORTED", Severity::Error, stmt, msg));
        }
    }
    out
}

fn compatible(a: &Trace, b: &Trace) -> bool {
    // a workgroup-uniform runtime condition has the same outcome in both roles
    a.path.iter().filter(|p| p.2).all(|pa| b.path.iter().filter(|p| p.2 && p.0 == pa.0).all(|pb| pb.1 == pa.1))
}

thread_local! {
    static REFERENCE_PAIR_CHECKS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `g` with the straightforward (quadratic) pair checks of `check_pair` instead of the indexed ones: the reference of the differential test
/// (`crates/amdgpu-kernels/tests/gpu_protocol_diff.rs`): the two must report the same diagnostics in the same order for every kernel.
#[doc(hidden)]
pub fn with_reference_pair_checks<R>(g: impl FnOnce() -> R) -> R {
    REFERENCE_PAIR_CHECKS.with(|c| c.set(true));
    let r = g();
    REFERENCE_PAIR_CHECKS.with(|c| c.set(false));
    r
}

fn check_pair(f: &Function, traces: [&Trace; 2], n: usize, diags: &mut Vec<(&'static str, Severity, usize, String)>, checked: &mut usize, own_reuse: &std::collections::BTreeMap<(usize, String), bool>, owned: &std::collections::BTreeMap<usize, bool>) {
    // OWNER events (lds_owned, every access proven in the issuing thread's own words): never conflict with each other across threads
    let is_owner = |ev: &Ev| f.contract.tiles[ev.tile].owned.is_some() && owned.get(&ev.stmt).copied().unwrap_or(false);
    struct Info {
        role: usize,
        ev: Ev,
        done: Option<usize>,
        publish: Option<u32>,
        done_ord: u32,
        released: bool,
        /// `is_owner(&ev)`, evaluated once (it looks the statement up in a map)
        owner: bool,
    }
    let mut infos: Vec<Info> = Vec::new();
    for (role, t) in traces.iter().enumerate() {
        for (k, ev) in t.evs.iter().enumerate() {
            if ev.n_acc == 0 {
                diags.push(("GV-TILE-EMPTY", Severity::Warn, ev.stmt, format!("stmt {}: tile marker `{}` covers no LDS access", ev.stmt, f.contract.tiles[ev.tile].name)));
                continue;
            }
            let ti = &t.infos[k];
            infos.push(Info { role, owner: is_owner(ev), ev: ev.clone(), done: ti.done, publish: ti.publish, done_ord: ti.done_ord, released: ti.released });
        }
    }
    // the producers per (tile, slot): every consumer looks up its own slot instead of scanning all events
    let mut prod_of: HashMap<(usize, i64), Vec<usize>> = HashMap::new();
    for (k, x) in infos.iter().enumerate() {
        if x.ev.produce {
            prod_of.entry((x.ev.tile, x.ev.slot)).or_default().push(k);
        }
    }
    // the producers of one tile (infos ascending): the own-wave lookup of a private region does not scan every (tile, slot) group
    let mut prod_by_tile: Vec<Vec<usize>> = vec![Vec::new(); f.contract.tiles.len()];
    for (k, x) in infos.iter().enumerate() {
        if x.ev.produce {
            prod_by_tile[x.ev.tile].push(k);
        }
    }
    // Per (tile, slot) group of producers (infos ascending), the indexes the clean-consumer pre-check uses: by phase with a prefix maximum of the publish phase (None = never), the latest first, and by version.
    struct Group {
        by_ord: Vec<usize>,
        prefmax: Vec<u64>,
        desc: Vec<usize>,
        by_ver: HashMap<i64, Vec<usize>>,
    }
    let groups: HashMap<(usize, i64), Group> = if reference {
        HashMap::new()
    } else {
        prod_of
            .iter()
            .map(|(k, ks)| {
                let mut by_ord = ks.clone();
                by_ord.sort_by_key(|&i| (infos[i].ev.ord, i));
                let mut prefmax = Vec::with_capacity(by_ord.len());
                let mut m = 0u64;
                for &i in &by_ord {
                    m = m.max(infos[i].publish.map_or(u64::MAX, |v| v as u64));
                    prefmax.push(m);
                }
                let mut desc = ks.clone();
                desc.sort_by_key(|&i| std::cmp::Reverse((infos[i].ev.ord, infos[i].ev.start, i)));
                let mut by_ver: HashMap<i64, Vec<usize>> = HashMap::new();
                for &i in ks {
                    by_ver.entry(infos[i].ev.ver as i64).or_default().push(i);
                }
                (*k, Group { by_ord, prefmax, desc, by_ver })
            })
            .collect()
    };
    let name = |ti: usize| f.contract.tiles[ti].name.clone();
    // the location suffixes of the messages, formatted once per pair (not per consumer)
    let wh_priv: [String; 2] = [0, 1].map(|r| format!("(role {r}, {n} iteration(s) simulated)"));
    let wh: [[String; 2]; 2] = [0, 1].map(|r| [false, true].map(|g| format!("(role {r}, {n} iteration(s) simulated){}", if g { " [global region]" } else { "" })));
    for c in infos.iter().filter(|x| !x.ev.produce) {
        *checked += 1;
        if f.contract.tiles[c.ev.tile].private {
            // own-wave semantics: the wave reads a version it produced AND completed (wait_vm/wait_lds(0)) itself; no other wave is involved, so no barrier is required
            let tname = f.contract.tiles[c.ev.tile].name.clone();
            let own_k: Vec<usize> = prod_by_tile[c.ev.tile].iter().copied().filter(|&k| infos[k].role == c.role).collect();
            let own: Vec<&Info> = own_k.iter().map(|&k| &infos[k]).collect();
            let completed: Vec<&&Info> = own.iter().filter(|p| p.done.map_or(false, |d| d < c.ev.start)).collect();
            let stmt = c.ev.stmt;
            let where_ = &wh_priv[c.role.min(1)];
            match completed.iter().max_by_key(|p| p.ev.start) {
                None => {
                    let why = match own.iter().find(|p| p.ev.start < c.ev.start) {
                        Some(p) => format!("the write of version {} (stmt {}) is not completed by this wave (wait_lds(0) / wait_vm(0)) before the read", p.ev.ver, p.ev.stmt),
                        None => "this wave has produced nothing in the region before the read".to_string(),
                    };
                    diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of the private region `{tname}` expecting version {} finds no completed own-wave version: {why} {where_}", c.ev.ver)));
                }
                Some(p) if p.ev.ver != c.ev.ver => {
                    diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of the private region `{tname}` expects version {} but this wave's latest completed version is {} (stmt {}) {where_}", c.ev.ver, p.ev.ver, p.ev.stmt)));
                }
                Some(_) => {}
            }
            continue;
        }
        let key = (c.ev.tile, c.ev.slot);
        // an owner read of a version this thread wrote itself before (owner write, program order): the thread's own words, no barrier needed
        let slot_prods: &[usize] = prod_of.get(&key).map(|v| v.as_slice()).unwrap_or(&[]);
        if c.owner && slot_prods.iter().map(|&k| &infos[k]).any(|p| p.owner && p.role == c.role && p.ev.ver == c.ev.ver && p.ev.start < c.ev.start) {
            for q in slot_prods.iter().map(|&k| &infos[k]).filter(|q| q.ev.ver != c.ev.ver && !q.owner) {
                let overlap = q.ev.ord <= c.done_ord && q.publish.map_or(true, |v| v > c.ev.ord);
                if overlap {
                    diags.push(("GV-TILE-RACE", Severity::Error, c.ev.stmt, format!("stmt {}: owner read of `{}` (version {}) overlaps the shared write of version {} (stmt {}): no barrier separates them", c.ev.stmt, name(c.ev.tile), c.ev.ver, q.ev.ver, q.ev.stmt)));
                }
            }
            continue;
        }
        // Pre-check (indexed): a consumer that certainly produces no finding below is skipped; anything else takes the full scan, which reports exactly as before. Clean means: no pending-read warning, not a global region,
        // the latest published version at this phase is the expected one, every producer of the expected version up to this phase is published, and no producer up to the end of the read window is still unpublished at
        // the start of the read (a superset of the overlap the race scan looks for).
        if !reference && !c.ev.global {
            if let Some(g) = groups.get(&key) {
                let vis = |p: &Info| p.publish.map_or(false, |v| v <= c.ev.ord);
                let pending_warn = c.done.is_none() && traces[c.role].bar_from.get(c.ev.last + 1).copied().unwrap_or(false);
                let latest_ok = g.desc.iter().find(|&&k| vis(&infos[k])).map_or(false, |&k| infos[k].ev.ver == c.ev.ver);
                let same_ver_ok = g.by_ver.get(&(c.ev.ver as i64)).map_or(true, |l| l.iter().all(|&k| { let p = &infos[k]; p.ev.ord > c.ev.ord || (c.owner && p.owner) || vis(p) }));
                let cnt = g.by_ord.partition_point(|&k| infos[k].ev.ord <= c.done_ord);
                let no_overlap = cnt == 0 || g.prefmax[cnt - 1] <= c.ev.ord as u64;
                if !pending_warn && latest_ok && same_ver_ok && no_overlap {
                    continue;
                }
            }
        }
        let prods: Vec<&Info> = slot_prods.iter().map(|&k| &infos[k]).collect();
        let stmt = c.ev.stmt;
        let gl = c.ev.global;
        let where_ = &wh[c.role.min(1)][gl as usize];
        // only a BARRIER after the uncompleted reads can order them against a next writer: reads still in flight when the wave's trace ends (the last tile of a
        // pipeline without a closing barrier, e.g. the `_t128` cadence) have no next writer to race with
        if c.done.is_none() && traces[c.role].bar_from.get(c.ev.last + 1).copied().unwrap_or(false) {
            diags.push(("GV-TILE-PENDING", Severity::Warn, stmt, format!("stmt {stmt}: reads of `{}`[{}] are not completed (wait_lds(0)) before the next barrier, so the barrier does not separate them from the next writer {where_}", name(c.ev.tile), c.ev.slot)));
        }
        // SHARED tile: the data is produced by OTHER waves (even of the same role) and consumed independently by every wave. A producer's own completion
        // (wait_vm / wait_lds(0)) is only what lets the next barrier PUBLISH it; visibility is always the barrier publication, for every role. Only private regions
        // (lds_private) may be self-produced and self-consumed.
        // (a global region additionally needs the producer's fence_release before the publishing barrier: the barrier alone does not push plain stores out of the wave's cache)
        let visible = |p: &Info| -> bool { p.publish.map_or(false, |v| v <= c.ev.ord) && (!gl || p.released) };
        let mut vis: Vec<&&Info> = prods.iter().filter(|p| visible(p)).collect();
        vis.sort_by_key(|p| (p.ev.ord, p.ev.start));
        match vis.last() {
            None => {
                let pending = prods.iter().find(|p| p.ev.ver == c.ev.ver);
                let why = match pending {
                    Some(p) if gl && p.done.is_some() && p.publish.map_or(false, |v| v <= c.ev.ord) && !p.released => format!("version {} (stmt {}) was completed and a barrier followed, but the write was NOT released (no fence_release between the write and the publishing barrier): the other waves may read stale lines", p.ev.ver, p.ev.stmt),
                    Some(p) if p.done.is_none() => format!("the {} of version {} (stmt {}) is never completed (no wait_vm(0) / wait_lds(0) / fence_release between the write and the next barrier), so no barrier can publish it", if p.ev.global { "global write" } else if p.ev.dma { "DMA" } else { "write" }, p.ev.ver, p.ev.stmt),
                    Some(p) if p.role == c.role && p.publish.map_or(true, |v| v > c.ev.ord) => format!("version {} (stmt {}) is produced by the waves of THIS role too, but a shared tile is only visible after a barrier publishes it (own wait_vm/wait_lds(0) is not enough: the other waves' parts may not be written yet); published at phase {}", p.ev.ver, p.ev.stmt, p.publish.map_or("never (no barrier after its completion)".to_string(), |v| v.to_string())),
                    Some(p) => format!("version {} is produced by role {} (stmt {}, phase {}) but is only published at phase {}", p.ev.ver, p.role, p.ev.stmt, p.ev.ord, p.publish.map_or("never (no barrier after its completion)".to_string(), |v| v.to_string())),
                    None => format!("no producer of `{}`[{}] with version {} exists in the simulated schedule", name(c.ev.tile), c.ev.slot, c.ev.ver),
                };
                diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] expecting version {} at phase {} finds NO published version: {why} {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord)));
                continue;
            }
            Some(p) => {
                if p.ev.ver != c.ev.ver {
                    diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] expects version {} but the latest published version at phase {} is {} (produced by role {}, stmt {}): the slot was overwritten before this read, or the next version is not yet published {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord, p.ev.ver, p.role, p.ev.stmt)));
                    continue;
                }
            }
        }
        if gl {
            // VISIBILITY on the consumer side: an acquire (invalidate) after the publishing barrier and before the first read, unless every read is a coherent (sc1) load
            let need = prods.iter().filter(|p| p.ev.ver == c.ev.ver && p.ev.ord <= c.ev.ord).filter_map(|p| p.publish).max();
            if let Some(pub_ord) = need {
                let ok = traces[c.role].items.iter().enumerate().take(c.ev.start).any(|(_, it)| match it {
                    Item::Acquire { ord, inv } => *ord >= pub_ord && (*inv || c.ev.coherent),
                    _ => false,
                });
                if !ok {
                    diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of the global region `{}` generation {} at phase {} runs before the acquire: no fence_acquire(inv) (or acquire + coherent sc1 loads) after the publishing barrier (phase {pub_ord}) and before the read, so the wave may still hit stale cache lines (no acquire) {where_}", name(c.ev.tile), c.ev.ver, c.ev.ord)));
                }
            }
        }
        for p in prods.iter().filter(|p| p.ev.ver == c.ev.ver && p.ev.ord <= c.ev.ord) {
            // another thread's owner write of this version touches only its own words, which an owner read never reads
            if c.owner && p.owner {
                continue;
            }
            if !visible(p) {
                diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] version {} at phase {}: the part produced by role {} (stmt {}) is not yet published (publish phase {}) {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord, p.role, p.ev.stmt, p.publish.map_or("never".to_string(), |v| v.to_string()))));
            }
        }
        for q in prods.iter().filter(|q| q.ev.ver != c.ev.ver && !(c.owner && q.owner)) {
            // the read window [c.ord, c.done_ord] against the write window [q.ord, q.publish) of ANY wave (same role included)
            let overlap = q.ev.ord <= c.done_ord && q.publish.map_or(true, |v| v > c.ev.ord);
            if overlap {
                diags.push(("GV-TILE-RACE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] (version {}, phases {}..{}) overlaps the write of version {} by role {} (stmt {}, issued at phase {}, published at {}): no barrier separates them {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord, c.done_ord, q.ev.ver, q.role, q.ev.stmt, q.ev.ord, q.publish.map_or("never".to_string(), |v| v.to_string()))));
            }
        }
    }
    // REUSED storage (`lds_reuse`: LdsPlan::tile_over / private(.., overlays)): the new declaration lives in bytes of the old one, so a write of one must
    // not overlap in time (barrier phases) any access of the other by another wave - the reused range must be dead for EVERY wave that reads it (its
    // reads completed and a barrier between) - unless the write provably stays in the writer's own slice of a private old region (gpu_verify's
    // `own_reuse` proof: the wave's own storage, dead for itself by program order). A private region counts as its whole extent (the simulation does
    // not track wave keys); plain `lds_alias` declarations (overlapping declared extents of interleaved rings) are not reuses and are not checked here.
    let reference = REFERENCE_PAIR_CHECKS.with(|c| c.get());
    let tiles = &f.contract.tiles;
    let reuse = |x: usize, y: usize| {
        let (a, b) = (&tiles[x].name, &tiles[y].name);
        f.contract.tile_reuses.iter().any(|(p, q)| (p == a && q == b) || (p == b && q == a))
    };
    let range = |ev: &Ev| -> (i64, i64) {
        let t = &tiles[ev.tile];
        if t.private {
            (t.base, t.base + t.slot_size * t.slots as i64)
        } else {
            (t.base + t.slot_size * ev.slot, t.base + t.slot_size * (ev.slot + 1))
        }
    };
    // a write of a shared reuse tile proven to stay in the writer's own slice of the private region `p`
    let own = |w: &Info, p: usize| w.ev.produce && !tiles[w.ev.tile].private && tiles[p].private && own_reuse.get(&(w.ev.stmt, tiles[p].name.clone())).copied().unwrap_or(false);
    // without a declared reuse every pair is skipped by `reuse` below: the quadratic scan has nothing to find
    let scan_reuse = reference || !f.contract.tile_reuses.is_empty();
    for (i, a) in infos.iter().enumerate().filter(|_| scan_reuse) {
        for b in infos.iter().skip(i + 1) {
            if a.ev.tile == b.ev.tile || !(a.ev.produce || b.ev.produce) || a.ev.global || b.ev.global || !reuse(a.ev.tile, b.ev.tile) {
                continue;
            }
            let (ta, tb) = (&tiles[a.ev.tile], &tiles[b.ev.tile]);
            if ta.private && tb.private && ta.base == tb.base && ta.slot_size == tb.slot_size {
                continue; // same per-wave slicing: each wave only touches its own slice of both
            }
            if own(a, b.ev.tile) || own(b, a.ev.tile) {
                continue;
            }
            let (ra, rb) = (range(&a.ev), range(&b.ev));
            if !(ra.0 < rb.1 && rb.0 < ra.1) {
                continue;
            }
            // access windows in barrier phases: a write lasts until it is published, a read until it is completed
            let win = |x: &Info| -> (u32, u32) { if x.ev.produce { (x.ev.ord, x.publish.unwrap_or(u32::MAX)) } else { (x.ev.ord, x.done_ord.saturating_add(1)) } };
            let (wa, wb) = (win(a), win(b));
            if wa.0 < wb.1 && wb.0 < wa.1 {
                let what = |x: &Info| if x.ev.produce { "write" } else { "read" };
                let (newn, oldn) = if f.contract.tile_reuses.iter().any(|(p, _)| *p == name(a.ev.tile)) { (name(a.ev.tile), name(b.ev.tile)) } else { (name(b.ev.tile), name(a.ev.tile)) };
                let msg = format!(
                        "stmt {} / stmt {}: the {} of `{}` (role {}, phases {}..{}) and the {} of `{}` (role {}, phases {}..{}) may touch the same LDS bytes in [{}, {}): `{}` reuses the storage of `{}` (tile_over / overlays) but no barrier separates them and the write is not provably inside the writer's own slice - the reused range may still be live for another wave (complete the last reads with wait_lds(0) and put a barrier before reusing it, or write only the own wave's slice)",
                        a.ev.stmt, b.ev.stmt, what(a), name(a.ev.tile), a.role, wa.0, wa.1, what(b), name(b.ev.tile), b.role, wb.0, wb.1, ra.0.max(rb.0), ra.1.min(rb.1), newn, oldn
                    );
                if !diags.iter().any(|d| d.3 == msg) {
                    diags.push(("GV-TILE-ALIAS-RACE", Severity::Error, a.ev.stmt.max(b.ev.stmt), msg));
                }
            }
        }
    }
    let pp_msg = |a: &Info, b: &Info, aw: (u32, u32), bw: (u32, u32)| format!("stmt {} / stmt {}: roles {} and {} write different versions ({} / {}) of `{}`[{}] in overlapping phases {}..{} / {}..{} (write-write race)", a.ev.stmt, b.ev.stmt, a.role, b.role, a.ev.ver, b.ev.ver, name(a.ev.tile), a.ev.slot, aw.0, aw.1, bw.0, bw.1);
    if reference {
        for (i, a) in infos.iter().enumerate().filter(|(_, x)| x.ev.produce) {
            for b in infos.iter().skip(i + 1).filter(|x| x.ev.produce) {
                if a.role != b.role && (a.ev.tile, a.ev.slot) == (b.ev.tile, b.ev.slot) && a.ev.ver != b.ev.ver && !f.contract.tiles[a.ev.tile].private && !(a.owner && b.owner) {
                    let (aw, bw) = ((a.ev.ord, a.publish.unwrap_or(u32::MAX)), (b.ev.ord, b.publish.unwrap_or(u32::MAX)));
                    if aw.0 < bw.1 && bw.0 < aw.1 {
                        diags.push(("GV-TILE-RACE", Severity::Error, a.ev.stmt, pp_msg(a, b, aw, bw)));
                    }
                }
            }
        }
    } else {
        // only the producers of one (tile, slot) can pair up: walk each group (indices ascending) and report in the order of the full scan (i, then j)
        let mut hits: Vec<(usize, usize)> = Vec::new();
        for ks in prod_of.values() {
            for (x, &i) in ks.iter().enumerate() {
                for &j in &ks[x + 1..] {
                    let (a, b) = (&infos[i], &infos[j]);
                    if a.role != b.role && a.ev.ver != b.ev.ver && !f.contract.tiles[a.ev.tile].private && !(a.owner && b.owner) {
                        let (aw, bw) = ((a.ev.ord, a.publish.unwrap_or(u32::MAX)), (b.ev.ord, b.publish.unwrap_or(u32::MAX)));
                        if aw.0 < bw.1 && bw.0 < aw.1 {
                            hits.push((i, j));
                        }
                    }
                }
            }
        }
        hits.sort_unstable();
        for (i, j) in hits {
            let (a, b) = (&infos[i], &infos[j]);
            let (aw, bw) = ((a.ev.ord, a.publish.unwrap_or(u32::MAX)), (b.ev.ord, b.publish.unwrap_or(u32::MAX)));
            diags.push(("GV-TILE-RACE", Severity::Error, a.ev.stmt, pp_msg(a, b, aw, bw)));
        }
    }
}

pub(crate) fn verify_tiles(f: &Function, rep: &mut Report) {
    verify_tiles_cap(f, rep, None)
}

/// `verify_tiles` simulating at most `cap` loop iterations (and not checking the period claim, which needs the full horizon): for mutation tests that
/// re-verify a kernel many times. A protocol error found within the cap is a real error; the full verification is `verify_tiles`.
/// W3 B5: the plan's ring-slot claims are checked against the simulated kernel, never trusted. The back end lets a ring DMA pass an LDS
/// read (and drops their waits) when the plan's iteration offsets put them in different slots (`sched::waits::RingModel`: producer offset
/// d, reader offset c, slots S: different iff d != c mod S). If the concrete simulation shows the two touching the SAME slot inside one
/// barrier interval, the plan lies and the reordered code could race: GV-PLAN-RING.
fn check_plan_rings(f: &Function, tr: &Trace, diags: &mut Vec<(&'static str, Severity, usize, String)>) {
    let Some(plan) = &f.plan else { return };
    let Some(rm) = crate::sched::waits::RingModel::of_function(f) else { return };
    let units = unit_of_stmt(f);
    let tile_of_ring: Vec<Option<usize>> = plan.rings.iter().map(|r| f.contract.tiles.iter().position(|t| t.name == r.name)).collect();
    for v in plan_ring_violations(&rm, &units, &tile_of_ring, &tr.evs) {
        diags.push(("GV-PLAN-RING", Severity::Error, v.0, v.1));
    }
}

/// The innermost plan unit tag (`id + 1`, as `MInst::unit`) of every statement (0 = none).
fn unit_of_stmt(f: &Function) -> Vec<u32> {
    let mut stack: Vec<u32> = Vec::new();
    f.stmts()
        .iter()
        .map(|s| match s {
            Stmt::UnitMark { id, begin: true } => {
                stack.push(*id + 1);
                *stack.last().unwrap()
            }
            Stmt::UnitMark { begin: false, .. } => {
                let u = stack.last().copied().unwrap_or(0);
                stack.pop();
                u
            }
            _ => stack.last().copied().unwrap_or(0),
        })
        .collect()
}

/// (statement, message) for every DMA produce / consume pair of one ring that the plan calls different slots while the simulated
/// events touch the same slot in the same barrier interval.
fn plan_ring_violations(rm: &crate::sched::waits::RingModel, units: &[u32], tile_of_ring: &[Option<usize>], evs: &[Ev]) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for p in evs.iter().filter(|e| e.produce && e.dma && !e.global) {
        let Some(prods) = rm.producers.get(&units.get(p.stmt).copied().unwrap_or(0)) else { continue };
        for q in evs.iter().filter(|e| !e.produce && !e.global && e.ord == p.ord && e.tile == p.tile && e.slot == p.slot) {
            let Some(cons) = rm.consumers.get(&units.get(q.stmt).copied().unwrap_or(0)) else { continue };
            for &(ri, s, d) in prods {
                if tile_of_ring.get(ri).copied().flatten() != Some(p.tile) {
                    continue;
                }
                for &(rj, _, c) in cons {
                    if rj == ri && (d - c).rem_euclid(s as i64) != 0 {
                        out.push((p.stmt, format!(
                            "the plan puts the DMA at stmt {} (iteration offset {d}) and the read at stmt {} (offset {c}) in different slots of ring {ri} ({s} slots), but both touch slot {} in the same barrier interval: the back end would let them pass each other",
                            p.stmt, q.stmt, p.slot
                        )));
                    }
                }
            }
        }
    }
    out
}

pub(crate) fn verify_tiles_cap(f: &Function, rep: &mut Report, cap: Option<usize>) {
    let tiles = &f.contract.tiles;
    if tiles.is_empty() {
        return;
    }
    // (a) declarations: inside the segment, pairwise disjoint unless aliased
    for t in tiles.iter().filter(|t| !t.global) {
        let end = t.base + t.slot_size * t.slots as i64;
        if t.base < 0 || end > f.lds_bytes as i64 {
            rep.add("GV-TILE-OVERLAP", Severity::Error, format!("tile `{}` [{}, {}) is not inside the declared {}-byte LDS segment", t.name, t.base, end, f.lds_bytes));
        }
    }
    for (i, a) in tiles.iter().enumerate().filter(|(_, t)| !t.global) {
        for b in tiles.iter().skip(i + 1).filter(|t| !t.global) {
            let (ae, be) = (a.base + a.slot_size * a.slots as i64, b.base + b.slot_size * b.slots as i64);
            let aliased = f.contract.tile_aliases.iter().any(|(x, y)| (x == &a.name && y == &b.name) || (x == &b.name && y == &a.name));
            if a.base < be && b.base < ae && !aliased {
                rep.add("GV-TILE-OVERLAP", Severity::Error, format!("live tiles `{}` [{}, {}) and `{}` [{}, {}) overlap and are not declared as aliases (lds_alias)", a.name, a.base, ae, b.name, b.base, be));
            }
        }
    }
    // horizon: the markers' period bound P = 2^M must fit twice into the simulated iterations
    let max_slots = tiles.iter().map(|t| t.slots as usize).max().unwrap_or(1);
    let base_h = 10usize.max(2 * max_slots + 4);
    let (m, mul_log2) = period_bits(f);
    let pbound = 1usize << m.min(20);
    const HORIZON_CAP: usize = 70;
    let nmax;
    let mut period_claim: Option<usize> = None;
    let cone = periodic_cone(f);
    let fs_period = *cone.as_ref().unwrap_or(&1);
    let pbound = if fs_period > 1 { lcm(pbound, fs_period) } else { pbound };
    let need = base_h.max(2 * pbound + 2);
    match (&f.contract.tile_trust, cone) {
        (Some(reason), _) => {
            nmax = base_h;
            rep.add("GV-ASSUME", Severity::Info, format!("tile_trust_periodic: TRUSTED that the protocol is periodic in the loop iteration ({reason}); {nmax} iterations are simulated"));
        }
        (None, Err(why)) => {
            nmax = base_h;
            rep.add(
                "GV-TILE-UNROLL-LIMIT",
                Severity::Error,
                format!("{why}, which is not an affine/periodic function of the loop counters: the protocol is simulated for at most {nmax} loop iterations, so a launch with more iterations is NOT covered (express slot/version with counters stepped by + const / ^ const, or declare b.tile_trust_periodic(reason) when the structure is periodic by construction)"),
            );
        }
        (None, Ok(_)) if mul_log2 >= 16 => {
            nmax = base_h;
            rep.add(
                "GV-TILE-UNROLL-LIMIT",
                Severity::Error,
                format!("the tile markers multiply / shift left by a total of 2^{mul_log2}: the 32-bit slot / version may wrap within about {} iterations, beyond the {nmax} simulated, so a longer launch is NOT covered (use smaller strides, or declare b.tile_trust_periodic(reason); i32 arithmetic is otherwise assumed not to wrap, A3)", 1u64 << (31 - mul_log2.min(31))),
            );
        }
        (None, Ok(_)) => {
            if need > HORIZON_CAP {
                nmax = base_h;
                rep.add(
                    "GV-TILE-UNROLL-LIMIT",
                    Severity::Error,
                    format!("the tile markers' slot/version expressions may have a period of up to {pbound} iterations (a shift/mask in their cone reaches bit {}), which needs {need} simulated iterations; the horizon is capped at {HORIZON_CAP}: a launch with more iterations than {nmax} is NOT covered (simplify the index arithmetic or declare b.tile_trust_periodic(reason))", m.saturating_sub(1)),
                );
            } else {
                nmax = need;
                period_claim = Some(pbound);
                rep.add("GV-TILE-PERIOD", Severity::Info, format!("tile markers claim a period dividing {pbound} iteration(s) (slot repeats, version advances by a constant; multiplications/shifts by constants assumed not to wrap, A3); {nmax} iterations (>= 2 x period + 2, >= 2 x ring slots + 4) are simulated for every exit parity and the claim is VERIFIED on the simulated sequences, which covers any trip count"));
            }
        }
    }
    let nmax = cap.map_or(nmax, |c| nmax.min(c));
    let period_claim = if cap.is_some() { None } else { period_claim };
    let div = divergent_ifs(f);
    let own_reuse = rep.own_reuse.clone();
    let owned = rep.owned.clone();
    let mut diags: Vec<(&'static str, Severity, usize, String)> = Vec::new();
    let mut untagged: BTreeSet<usize> = BTreeSet::new();
    let mut checked = 0usize;
    let mut paths = 0usize;
    for n in 1..=nmax {
        let t0 = enumerate_traces(f, 0, n, &div, &mut diags, &mut untagged);
        let t1 = enumerate_traces(f, 1, n, &div, &mut diags, &mut untagged);
        for tr in t0.iter().chain(t1.iter()) {
            check_plan_rings(f, tr, &mut diags);
        }
        for a in &t0 {
            for b in &t1 {
                if compatible(a, b) {
                    paths += 1;
                    check_pair(f, [a, b], n, &mut diags, &mut checked, &own_reuse, &owned);
                }
            }
        }
    }
    // the claimed period must be real: verify it on the longest simulated trace of each role
    if let Some(p) = period_claim {
        let ta = enumerate_traces(f, 0, nmax, &div, &mut diags, &mut untagged);
        let tb = enumerate_traces(f, 1, nmax, &div, &mut diags, &mut untagged);
        if let (Some(a), Some(b)) = (ta.first(), tb.first()) {
            if let Err(e) = verify_period_claim(f, &[a, b], p) {
                diags.push(("GV-TILE-UNROLL-LIMIT", Severity::Error, 0, format!("{e}; a launch with more iterations than {nmax} is NOT covered")));
            }
        }
    }
    let mut errs = 0usize;
    for (code, sev, _stmt, msg) in diags {
        if sev == Severity::Error {
            errs += 1;
        }
        // the global instance has no legacy names: its primary code is the protocol verdict itself
        let code = if msg.contains("[global region]") && !proto_alias(code, &msg).is_empty() { proto_alias(code, &msg) } else { code };
        rep.add(code, sev, msg);
    }
    for (name, reason) in &f.contract.buf_trusts {
        rep.add("GV-ASSUME", Severity::Info, format!("buf_trust `{name}`: TRUSTED, the protocol of this region is not checked ({reason})"));
    }
    if !untagged.is_empty() {
        rep.add("GV-TILE-UNTAGGED", Severity::Warn, format!("{} LDS access statement(s) executed outside any tile marker scope (first at stmt {}): their lifetime is not checked", untagged.len(), untagged.iter().next().unwrap()));
    }
    rep.add("GV-TILE-PERIOD", Severity::Info, format!("tile simulation: {paths} role-path combination(s) over {nmax} loop-iteration scenarios, {checked} read events checked"));
    if errs == 0 {
        for _ in 0..checked {
            rep.ok("tile-reads");
        }
    }
}
// --------------------------------------------------------------------------------------------------------------------------------------------
// the opt-in fused KV-split combine (last-arriver) pattern over a global region

fn ptr_arg(f: &Function, o: &Opnd, depth: usize) -> Option<String> {
    if depth > 16 {
        return None;
    }
    let Opnd::Val(v) = o else { return None };
    let op = f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if dst == v { Some(op) } else { None } } else { None })?;
    match op {
        Op::KernargPtr { offset } => f.args.iter().find(|a| a.kernarg_offset == *offset).map(|a| a.name.clone()),
        Op::PtrAdd { base, .. } => ptr_arg(f, base, depth + 1),
        _ => None,
    }
}

fn cone_has(f: &Function, o: &Opnd, pred: &dyn Fn(&Op) -> bool, seen: &mut HashSet<u32>) -> bool {
    match o {
        Opnd::Val(v) => {
            if !seen.insert(*v) {
                return false;
            }
            let Some(op) = f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if dst == v { Some(op) } else { None } } else { None }) else { return false };
            pred(op) || op.operands().iter().any(|x| cone_has(f, x, pred, seen))
        }
        Opnd::Var(x) => {
            // a loop counter: its initial value and every update
            if !seen.insert(1_000_000 + x.0) {
                return false;
            }
            f.stmts().iter().any(|s| match s {
                Stmt::SetVar { var, value } if var == x => cone_has(f, value, pred, seen),
                Stmt::BeginLoop { vars, init } | Stmt::DeclareVars { vars, init } => vars.iter().position(|y| y == x).map_or(false, |k| cone_has(f, &init[k], pred, seen)),
                _ => false,
            })
        }
        _ => false,
    }
}

/// O1 (stores complete + released before the ticket), O2 (the last arriver reads only after the acquire, inside the branch guarded by the ticket), O4 (generation = the block''s
/// split id, derived from the workgroup id; all splits'' slices are read). Cross-kernel (two-launch) split stays out of scope.
pub(crate) fn verify_last_arrivers(f: &Function, rep: &mut Report) {
    for la in &f.contract.last_arrivers {
        let region = &la.region;
        if !f.contract.tiles.iter().any(|t| &t.name == region && t.global) {
            rep.add("GV-CONTRACT", Severity::Error, format!("last-arriver pattern names `{region}` which is not a buf_region"));
            continue;
        }
        if f.contract.buf_trusts.iter().any(|(n, _)| n == region) {
            continue;
        }
        // scan: marker scopes, enclosing ifs, loop depth
        let mut open: Option<(bool, usize)> = None;
        let mut prod: Vec<(usize, bool)> = Vec::new();
        let mut prod_marker: Option<usize> = None;
        let mut cons: Vec<(usize, bool, Vec<usize>, usize)> = Vec::new(); // (stmt, coherent, enclosing ifs, loop depth)
        let mut ifs: Vec<usize> = Vec::new();
        let mut depth = 0usize;
        for (i, s) in f.stmts().iter().enumerate() {
            match s {
                Stmt::TileProduce { tile, .. } if tile == region => {
                    open = Some((true, i));
                    prod_marker.get_or_insert(i);
                }
                Stmt::TileConsume { tile, .. } if tile == region => open = Some((false, i)),
                Stmt::TileProduce { .. } | Stmt::TileConsume { .. } | Stmt::TileEnd | Stmt::Barrier => open = None,
                Stmt::BeginLoop { .. } => {
                    open = None;
                    depth += 1;
                }
                Stmt::EndLoop => {
                    open = None;
                    depth -= 1;
                }
                Stmt::BeginIf { .. } => {
                    open = None;
                    ifs.push(i);
                }
                Stmt::Else | Stmt::EndIf => {
                    open = None;
                    if matches!(s, Stmt::EndIf) {
                        ifs.pop();
                    }
                }
                _ => {}
            }
            if let (Some((is_prod, _)), Some((write, coh))) = (open, is_global_access(s)) {
                if is_prod && write {
                    prod.push((i, coh));
                } else if !is_prod && !write {
                    cons.push((i, coh, ifs.clone(), depth));
                }
            }
        }
        let fail = |rep: &mut Report, code: &'static str, msg: String| rep.add(code, Severity::Error, format!("last-arriver `{region}`: {msg}"));
        let Some(&(last_store, _)) = prod.last() else {
            fail(rep, "GV-PROTO-UNPUBLISHED", "no global store is marked buf_produce: nothing is published".to_string());
            continue;
        };
        // ticket: the atomic on the counter after the stores
        let ticket = f.stmts().iter().enumerate().skip(last_store + 1).find_map(|(i, s)| match s {
            Stmt::Let { dst, op: Op::AtomicAddU32 { base, .. } } if ptr_arg(f, base, 0).as_deref() == Some(la.counter_arg.as_str()) => Some((i, *dst)),
            // the buffer form: an atomic add through a descriptor over the counter pointer (one lane takes it, the others are out of range)
            Stmt::Let { dst, op: Op::BufferAtomicRet { op: AtomicOp::AddU32, rsrc, .. } } if rsrc_arg(f, rsrc).as_deref() == Some(la.counter_arg.as_str()) => Some((i, *dst)),
            _ => None,
        });
        let Some((ticket_idx, ticket_val)) = ticket else {
            fail(rep, "GV-PROTO-ARRIVAL-MISMATCH", format!("no ticket atomic_add on `{}` after the last produced store (stmt {last_store})", la.counter_arg));
            continue;
        };
        // O1: complete + release before the ticket
        let rel = f.stmts().iter().enumerate().skip(last_store + 1).take(ticket_idx - last_store - 1).find_map(|(i, s)| if let Stmt::FenceRelease { wb } = s { Some((i, *wb)) } else { None });
        match rel {
            None => fail(rep, "GV-PROTO-NO-COMPLETION", format!("O1: no fence_release between the last produced store (stmt {last_store}) and the ticket (stmt {ticket_idx}): the ticket can be taken before the data is complete / visible, so the last arriver may read stale or missing slices")),
            Some((ri, false)) if prod.iter().any(|(_, wt)| !*wt) => fail(rep, "GV-PROTO-NOT-VISIBLE", format!("O1: the release at stmt {ri} is the lite variant (no L2 write-back) but some produced stores are plain, not write-through (store_*_wt): their lines may stay in the producer's L2")),
            Some(_) => {}
        }
        // O2: the reads sit in the branch guarded by the ticket result, after the acquire
        if cons.is_empty() {
            fail(rep, "GV-PROTO-UNPUBLISHED", "no global load is marked buf_consume".to_string());
            continue;
        }
        let guarded = |ifs: &Vec<usize>| ifs.iter().any(|&bi| if let Stmt::BeginIf { cond } = &f.stmts()[bi] { cone_has(f, cond, &|op| matches!(op, Op::AtomicAddU32 { .. } | Op::BufferAtomicRet { op: AtomicOp::AddU32, .. }), &mut HashSet::new()) || matches!(cond, Opnd::Val(v) if *v == ticket_val) || cone_has_val(f, cond, ticket_val) } else { false });
        if let Some((i, ..)) = cons.iter().find(|(_, _, ifs, _)| !guarded(ifs)) {
            fail(rep, "GV-PROTO-ARRIVAL-MISMATCH", format!("the read at stmt {i} is not inside the branch guarded by the ticket result (stmt {ticket_idx}): a block that is not the last arriver may read slices other blocks have not written yet"));
        }
        let first_read = cons.iter().map(|c| c.0).min().unwrap();
        let acq = f.stmts().iter().enumerate().skip(ticket_idx + 1).take(first_read - ticket_idx - 1).find_map(|(i, s)| if let Stmt::FenceAcquire { inv } = s { Some((i, *inv)) } else { None });
        match acq {
            None => fail(rep, "GV-PROTO-NOT-VISIBLE", format!("O2: the first read (stmt {first_read}) runs before any fence_acquire after the ticket (stmt {ticket_idx}) - no acquire: the last arriver may hit stale cache lines")),
            Some((ai, false)) if cons.iter().any(|c| !c.1) => fail(rep, "GV-PROTO-NOT-VISIBLE", format!("O2: the acquire at stmt {ai} is the lite variant (no invalidate) but some reads are not coherent (sc1) loads")),
            Some(_) => {}
        }
        // O4: the produced generation is the block's split id
        if let Some(mi) = prod_marker {
            if let Stmt::TileProduce { version, .. } = &f.stmts()[mi] {
                let from_block = cone_has(f, version, &|op| matches!(op, Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ), &mut HashSet::new());
                if !from_block {
                    fail(rep, "GV-PROTO-STALE-GEN", format!("O4: the generation of the produce marker (stmt {mi}) is not derived from the workgroup id: the blocks do not agree on which slice (split) each of them owns"));
                }
            }
        }
        if cons.len() < 2 && !cons.iter().any(|c| c.3 > 0) {
            fail(rep, "GV-PROTO-STALE-GEN", "O4: the last arriver reads a single slice (no loop over the splits, fewer than two loads): not all splits'' slices are consumed".to_string());
        }
        if !rep.errors().iter().any(|e| e.msg.contains(&format!("last-arriver `{region}`"))) {
            rep.ok("last-arriver");
        }
    }
}

/// The pointer kernarg a buffer descriptor is built over (`BufferRsrc` / `BufferRsrcDyn` of a kernarg pointer, possibly offset).
fn rsrc_arg(f: &Function, o: &Opnd) -> Option<String> {
    let Opnd::Val(v) = o else { return None };
    let op = f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if dst == v { Some(op) } else { None } } else { None })?;
    match op {
        Op::BufferRsrc { base, .. } | Op::BufferRsrcDyn { base, .. } => ptr_arg(f, base, 0),
        _ => None,
    }
}

fn cone_has_val(f: &Function, o: &Opnd, target: u32) -> bool {
    fn go(f: &Function, o: &Opnd, target: u32, seen: &mut HashSet<u32>) -> bool {
        let Opnd::Val(v) = o else { return false };
        if *v == target {
            return true;
        }
        if !seen.insert(*v) {
            return false;
        }
        let Some(op) = f.stmts().iter().find_map(|s| if let Stmt::Let { dst, op } = s { if dst == v { Some(op) } else { None } } else { None }) else { return false };
        op.operands().iter().any(|x| go(f, x, target, seen))
    }
    go(f, o, target, &mut HashSet::new())
}

#[cfg(test)]
mod plan_ring_tests {
    use super::*;
    use crate::sched::waits::RingModel;

    fn ev(produce: bool, dma: bool, stmt: usize, slot: i64) -> Ev {
        Ev { produce, tile: 0, slot, ver: 0, ord: 3, stmt, start: 0, last: 0, dma, ds: !dma, n_acc: 1, global: false, coherent: true, outer: Default::default() }
    }

    fn model(slots: u32, dma_iter: i64, read_iter: i64) -> RingModel {
        let mut m = RingModel { slots: vec![slots], body_iters: 1, ..Default::default() };
        m.producers.insert(2, vec![(0, slots, dma_iter)]);
        m.consumers.insert(1, vec![(0, slots, read_iter)]);
        m
    }

    /// stmt 0 is in unit tag 1 (the reader), stmt 1 in unit tag 2 (the DMA)
    const UNITS: [u32; 2] = [1, 2];

    #[test]
    fn a_plan_that_separates_slots_the_kernel_shares_is_rejected() {
        // the kernel: the read and the DMA of the same barrier interval both touch slot 1
        let evs = vec![ev(false, false, 0, 1), ev(true, true, 1, 1)];
        // a lying plan: DMA offset 2, read offset 1, 2 slots -> "different slots"
        let v = plan_ring_violations(&model(2, 2, 1), &UNITS, &[Some(0)], &evs);
        assert_eq!(v.len(), 1, "{v:?}");
        // a truthful plan (same slot): nothing to report, the back end keeps the pair ordered
        assert!(plan_ring_violations(&model(2, 2, 0), &UNITS, &[Some(0)], &evs).is_empty());
    }

    #[test]
    fn different_slots_or_intervals_are_fine() {
        let other_slot = vec![ev(false, false, 0, 0), ev(true, true, 1, 1)];
        assert!(plan_ring_violations(&model(2, 2, 1), &UNITS, &[Some(0)], &other_slot).is_empty());
        let mut later = vec![ev(false, false, 0, 1), ev(true, true, 1, 1)];
        later[1].ord = 4; // separated by a barrier
        assert!(plan_ring_violations(&model(2, 2, 1), &UNITS, &[Some(0)], &later).is_empty());
    }
}
