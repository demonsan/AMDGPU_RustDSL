//! LDS tile lifetime / versioning (gpu_verify v1.1). Design: docs/gpu-semantics-verifier.md section 3.x.
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

use amdgpu_dsl::{Function, Op, Opnd, Stmt, ValueId, VarId};

use crate::gpu_verify::{Report, Severity};

#[derive(Clone, Copy, Debug, PartialEq)]
enum CV {
    K(i64),
    U,
}

#[derive(Clone, Debug)]
enum Item {
    Barrier { ord: u32 },
    WaitVm { ord: u32 },
    WaitLds { ord: u32 },
    Open { ev: usize },
    Access { dma: bool },
}

#[derive(Clone, Debug)]
struct Ev {
    produce: bool,
    tile: usize,
    slot: i64,
    ver: i64,
    ord: u32,
    stmt: usize,
    role: usize,
    start: usize,
    last: usize,
    dma: bool,
    ds: bool,
    n_acc: usize,
}

struct Trace {
    items: Vec<Item>,
    evs: Vec<Ev>,
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
        for st in &f.stmts {
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
                Stmt::BeginLoop { vars, init } => {
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
    for (i, st) in f.stmts.iter().enumerate() {
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
    for (k, s) in f.stmts.iter().enumerate().skip(begin) {
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
    f.stmts.len()
}

/// (else index or end index, end index) of the if starting at `begin`.
fn if_span(f: &Function, begin: usize) -> (Option<usize>, usize) {
    let mut depth = 0usize;
    let mut els = None;
    for (k, s) in f.stmts.iter().enumerate().skip(begin) {
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
    (els, f.stmts.len())
}

fn is_lds_access(st: &Stmt) -> Option<(bool, bool)> {
    // (dma, is_write)
    match st {
        Stmt::LoadGlobalLds { .. } | Stmt::BufferLoadLds { .. } => Some((true, true)),
        Stmt::StoreLdsB32 { .. } | Stmt::StoreLdsB128 { .. } => Some((false, true)),
        Stmt::Let { op, .. } => match op {
            Op::LoadLdsB32 { .. } | Op::LoadLdsB128 { .. } | Op::LoadLdsTrB8 { .. } | Op::LoadLdsTr4B8 { .. } | Op::LoadLdsB128x2 { .. } => Some((false, false)),
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

    fn open_marker(&mut self, i: usize, produce: bool, tile: &str, slot: &Opnd, ver: &Opnd) {
        let Some(ti) = self.f.contract.tiles.iter().position(|t| t.name == tile) else { return };
        match (self.cv(slot), self.cv(ver)) {
            (CV::K(s), CV::K(v)) => {
                let slots = self.f.contract.tiles[ti].slots as i64;
                if s < 0 || s >= slots {
                    self.diag("GV-TILE-INDEX", Severity::Error, i, format!("stmt {i}: tile `{tile}` slot {s} is outside the declared ring of {slots} slot(s)"));
                    self.open = None;
                    return;
                }
                let ev = Ev { produce, tile: ti, slot: s, ver: v, ord: self.ord, stmt: i, role: self.role, start: self.items.len(), last: self.items.len(), dma: false, ds: false, n_acc: 0 };
                self.evs.push(ev);
                let idx = self.evs.len() - 1;
                self.items.push(Item::Open { ev: idx });
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
            let st = &f.stmts[i];
            if let Some((dma, write)) = is_lds_access(st) {
                match self.open {
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
                Stmt::Barrier => {
                    self.items.push(Item::Barrier { ord: self.ord });
                    self.ord += 1;
                    self.close();
                }
                Stmt::WaitVm { count: 0 } => self.items.push(Item::WaitVm { ord: self.ord }),
                Stmt::WaitLds { count: 0 } => self.items.push(Item::WaitLds { ord: self.ord }),
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
                    for _ in 0..self.n {
                        self.exec_loop_body(i + 1, close);
                        if self.break_taken {
                            break;
                        }
                    }
                    self.break_taken = false;
                    self.close();
                    i = close + 1;
                    continue;
                }
                Stmt::BreakIf { cond } => {
                    if let CV::K(c) = self.cv(cond) {
                        if c != 0 {
                            self.break_taken = true;
                            return;
                        }
                    }
                }
                Stmt::BeginIf { cond } => {
                    self.close();
                    let (els, endif) = if_span(f, i);
                    let then_range = (i + 1, els.unwrap_or(endif));
                    let else_range = els.map(|e| (e + 1, endif));
                    if self.div.contains(&i) {
                        if self.role == 0 {
                            self.exec(then_range.0, then_range.1);
                        } else if let Some(r) = else_range {
                            self.exec(r.0, r.1);
                        }
                    } else {
                        match self.cv(cond) {
                            CV::K(c) => {
                                if c != 0 {
                                    self.exec(then_range.0, then_range.1);
                                } else if let Some(r) = else_range {
                                    self.exec(r.0, r.1);
                                }
                            }
                            CV::U => {
                                // a wave-uniform if with an unknown condition: its body must not carry protocol events
                                let interesting = |a: usize, b: usize| f.stmts[a..b].iter().any(|s| matches!(s, Stmt::Barrier | Stmt::TileProduce { .. } | Stmt::TileConsume { .. }) || is_lds_access(s).is_some());
                                let bad = interesting(then_range.0, then_range.1) || else_range.map_or(false, |r| interesting(r.0, r.1));
                                if bad {
                                    self.diag("GV-TILE-UNSUPPORTED", Severity::Warn, i, format!("stmt {i}: wave-uniform if with a runtime condition contains barriers / tile markers / LDS accesses: both outcomes are NOT modelled (restructure or hoist them)"));
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
                }
                _ => {}
            }
            i += 1;
        }
    }

    fn exec_loop_body(&mut self, start: usize, end: usize) {
        self.exec(start, end);
    }
}

fn analyse_trace(items: &[Item], ev: &Ev) -> (Option<usize>, Option<u32>, u32) {
    // (completion item, publish ordinal, completion ordinal)
    let after = ev.last + 1;
    let find = |pred: &dyn Fn(&Item) -> bool| items.iter().enumerate().skip(after).find(|(_, it)| pred(it)).map(|(k, _)| k);
    if ev.produce {
        let vm = if ev.dma { find(&|it| matches!(it, Item::WaitVm { .. })) } else { None };
        let ds = if ev.ds { find(&|it| matches!(it, Item::WaitLds { .. })) } else { None };
        let done = match (ev.dma, ev.ds) {
            (true, true) => match (vm, ds) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            },
            (true, false) => vm,
            (false, true) => ds,
            _ => None,
        };
        match done {
            Some(d) => {
                let ord = match &items[d] {
                    Item::WaitVm { ord } | Item::WaitLds { ord } => *ord,
                    _ => ev.ord,
                };
                let publish = items.iter().skip(d + 1).find_map(|it| if let Item::Barrier { ord } = it { Some(ord + 1) } else { None });
                (Some(d), publish, ord)
            }
            None => (None, None, ev.ord),
        }
    } else {
        // consumer: its reads are complete at the first wait_lds(0) after the last read
        match find(&|it| matches!(it, Item::WaitLds { .. })) {
            Some(d) => {
                let barrier_first = items.iter().enumerate().skip(after).find(|(_, it)| matches!(it, Item::Barrier { .. })).map(|(k, _)| k);
                if barrier_first.map_or(false, |b| b < d) {
                    (None, None, match items[barrier_first.unwrap()] { Item::Barrier { ord } => ord + 1, _ => ev.ord })
                } else {
                    let ord = if let Item::WaitLds { ord } = &items[d] { *ord } else { ev.ord };
                    (Some(d), None, ord)
                }
            }
            None => {
                let b = items.iter().skip(after).find_map(|it| if let Item::Barrier { ord } = it { Some(ord + 1) } else { None });
                (None, None, b.unwrap_or(ev.ord))
            }
        }
    }
}

pub(crate) fn verify_tiles(f: &Function, rep: &mut Report) {
    let tiles = &f.contract.tiles;
    if tiles.is_empty() {
        return;
    }
    // (a) declarations: inside the segment, pairwise disjoint unless aliased
    for t in tiles {
        let end = t.base + t.slot_size * t.slots as i64;
        if t.base < 0 || end > f.lds_bytes as i64 {
            rep.add("GV-TILE-OVERLAP", Severity::Error, format!("tile `{}` [{}, {}) is not inside the declared {}-byte LDS segment", t.name, t.base, end, f.lds_bytes));
        }
    }
    for (i, a) in tiles.iter().enumerate() {
        for b in tiles.iter().skip(i + 1) {
            let (ae, be) = (a.base + a.slot_size * a.slots as i64, b.base + b.slot_size * b.slots as i64);
            let aliased = f.contract.tile_aliases.iter().any(|(x, y)| (x == &a.name && y == &b.name) || (x == &b.name && y == &a.name));
            if a.base < be && b.base < ae && !aliased {
                rep.add("GV-TILE-OVERLAP", Severity::Error, format!("live tiles `{}` [{}, {}) and `{}` [{}, {}) overlap and are not declared as aliases (lds_alias)", a.name, a.base, ae, b.name, b.base, be));
            }
        }
    }
    let div = divergent_ifs(f);
    let mut diags: Vec<(&'static str, Severity, usize, String)> = Vec::new();
    let mut untagged: BTreeSet<usize> = BTreeSet::new();
    let mut checked = 0usize;
    for n in 1..=4usize {
        let mut traces: Vec<Trace> = Vec::new();
        for role in 0..2usize {
            let mut sim = Sim { f, role, n, div: &div, vals: HashMap::new(), vars: HashMap::new(), items: Vec::new(), evs: Vec::new(), ord: 0, open: None, diags: &mut diags, untagged: &mut untagged, steps: 0, break_taken: false };
            sim.exec(0, f.stmts.len());
            traces.push(Trace { items: sim.items, evs: sim.evs });
        }
        // per event derived data
        struct Info {
            role: usize,
            ev: Ev,
            done: Option<usize>,
            publish: Option<u32>,
            done_ord: u32,
        }
        let mut infos: Vec<Info> = Vec::new();
        for (role, t) in traces.iter().enumerate() {
            for ev in &t.evs {
                if ev.n_acc == 0 {
                    diags.push(("GV-TILE-EMPTY", Severity::Warn, ev.stmt, format!("stmt {}: tile marker `{}` covers no LDS access", ev.stmt, f.contract.tiles[ev.tile].name)));
                    continue;
                }
                let (done, publish, ord) = analyse_trace(&t.items, ev);
                infos.push(Info { role, ev: ev.clone(), done, publish, done_ord: ord });
            }
        }
        let name = |ti: usize| f.contract.tiles[ti].name.clone();
        for c in infos.iter().filter(|x| !x.ev.produce) {
            checked += 1;
            let key = (c.ev.tile, c.ev.slot);
            let prods: Vec<&Info> = infos.iter().filter(|p| p.ev.produce && (p.ev.tile, p.ev.slot) == key).collect();
            let stmt = c.ev.stmt;
            let where_ = format!("(role {}, {} iteration(s) simulated)", c.role, n);
            if c.done.is_none() && !matches!(traces[c.role].items.get(c.ev.last + 1), None) {
                diags.push(("GV-TILE-PENDING", Severity::Warn, stmt, format!("stmt {stmt}: reads of `{}`[{}] are not completed (wait_lds(0)) before the next barrier, so the barrier does not separate them from the next writer {where_}", name(c.ev.tile), c.ev.slot)));
            }
            let visible = |p: &Info| -> bool {
                if p.role == c.role {
                    p.done.map_or(false, |d| d < c.ev.start)
                } else {
                    p.publish.map_or(false, |v| v <= c.ev.ord)
                }
            };
            let mut vis: Vec<&&Info> = prods.iter().filter(|p| visible(p)).collect();
            vis.sort_by_key(|p| (p.ev.ord, p.ev.start));
            match vis.last() {
                None => {
                    // is there a producer of the expected version that is merely not yet published?
                    let pending = prods.iter().find(|p| p.ev.ver == c.ev.ver);
                    let why = match pending {
                        Some(p) if p.role == c.role => format!("the {} of version {} (stmt {}) by this wave is not completed (wait_vm(0)/wait_lds(0)) before the read", if p.ev.dma { "DMA" } else { "write" }, p.ev.ver, p.ev.stmt),
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
            // every co-producer of the expected version issued so far must be published too (cooperative fill)
            for p in prods.iter().filter(|p| p.ev.ver == c.ev.ver && p.ev.ord <= c.ev.ord) {
                if !visible(p) {
                    diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] version {} at phase {}: the part produced by role {} (stmt {}) is not yet published (publish phase {}) {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord, p.role, p.ev.stmt, p.publish.map_or("never".to_string(), |v| v.to_string()))));
                }
            }
            // no other version may be written / landing while this read is in progress
            for q in prods.iter().filter(|q| q.ev.ver != c.ev.ver) {
                let overlap = if q.role == c.role {
                    // same wave: a write issued BEFORE the read (program order) and not completed races with it; later writes are ordered after it
                    q.ev.start < c.ev.start && !q.done.map_or(false, |d| d < c.ev.start)
                } else {
                    q.ev.ord <= c.done_ord && q.publish.map_or(true, |v| v > c.ev.ord)
                };
                if overlap {
                    diags.push(("GV-TILE-RACE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] (version {}, phases {}..{}) overlaps the write of version {} by role {} (stmt {}, issued at phase {}, published at {}): no barrier separates them {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord, c.done_ord, q.ev.ver, q.role, q.ev.stmt, q.ev.ord, q.publish.map_or("never".to_string(), |v| v.to_string()))));
                }
            }
        }
        // write-write: different waves writing one slot concurrently must write the same version
        for (i, a) in infos.iter().enumerate().filter(|(_, x)| x.ev.produce) {
            for b in infos.iter().skip(i + 1).filter(|x| x.ev.produce) {
                if a.role != b.role && (a.ev.tile, a.ev.slot) == (b.ev.tile, b.ev.slot) && a.ev.ver != b.ev.ver {
                    let (aw, bw) = ((a.ev.ord, a.publish.unwrap_or(u32::MAX)), (b.ev.ord, b.publish.unwrap_or(u32::MAX)));
                    if aw.0 < bw.1 && bw.0 < aw.1 {
                        diags.push(("GV-TILE-RACE", Severity::Error, a.ev.stmt, format!("stmt {} / stmt {}: roles {} and {} write different versions ({} / {}) of `{}`[{}] in overlapping phases {}..{} / {}..{} (write-write race)", a.ev.stmt, b.ev.stmt, a.role, b.role, a.ev.ver, b.ev.ver, name(a.ev.tile), a.ev.slot, aw.0, aw.1, bw.0, bw.1)));
                    }
                }
            }
        }
    }
    let mut errs = 0usize;
    for (code, sev, _stmt, msg) in diags {
        if sev == Severity::Error {
            errs += 1;
        }
        rep.add(code, sev, msg);
    }
    if !untagged.is_empty() {
        rep.add("GV-TILE-UNTAGGED", Severity::Warn, format!("{} LDS access statement(s) executed outside any tile marker scope (first at stmt {}): their lifetime is not checked", untagged.len(), untagged.iter().next().unwrap()));
    }
    if errs == 0 {
        for _ in 0..checked {
            rep.ok("tile-reads");
        }
    }
}