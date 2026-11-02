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
    /// the outcomes taken at wave-uniform runtime ifs: (stmt, outcome, workgroup-uniform)
    path: Vec<(usize, bool, bool)>,
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
    decisions: Vec<bool>,
    dpos: usize,
    taken: Vec<(usize, bool, bool)>,
    loop_depth: usize,
    in_role: bool,
    unsupported: &'a mut Vec<(usize, String)>,
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
        let slot_cv = if private { CV::K(0) } else { self.cv(slot) };
        match (slot_cv, self.cv(ver)) {
            (CV::K(s), CV::K(v)) => {
                let slots = self.f.contract.tiles[ti].slots as i64;
                if !private && (s < 0 || s >= slots) {
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
                    self.loop_depth += 1;
                    for _ in 0..self.n {
                        self.exec_loop_body(i + 1, close);
                        if self.break_taken {
                            break;
                        }
                    }
                    self.break_taken = false;
                    self.loop_depth -= 1;
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
                    let proto = |a: usize, b: usize| f.stmts[a..b].iter().any(|s| matches!(s, Stmt::Barrier | Stmt::TileProduce { .. } | Stmt::TileConsume { .. }));
                    let anything = |a: usize, b: usize| f.stmts[a..b].iter().any(|s| matches!(s, Stmt::Barrier | Stmt::TileProduce { .. } | Stmt::TileConsume { .. }) || is_lds_access(s).is_some());
                    let has_proto = proto(then_range.0, then_range.1) || else_range.map_or(false, |r| proto(r.0, r.1));
                    let has_any = anything(then_range.0, then_range.1) || else_range.map_or(false, |r| anything(r.0, r.1));
                    let run_arm = |s: &mut Self, then: bool| {
                        if then {
                            s.exec(then_range.0, then_range.1);
                        } else if let Some(r) = else_range {
                            s.exec(r.0, r.1);
                        }
                    };
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
                                    if self.loop_depth > 0 {
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

/// Is every tile marker operand an AFFINE / PERIODIC function of loop counters? (constants, + - of counters, * << & ^ by constants; each counter used must be updated by
/// `+ const` or `^ const` of itself and start at a constant.) Only then does simulating a bounded number of iterations cover every trip count.
fn periodic_cone(f: &Function) -> Result<(), String> {
    fn konst(f: &Function, o: &Opnd) -> Option<i64> {
        match o {
            Opnd::I32(c) => Some(*c as i64),
            Opnd::Val(v) => f.stmts.iter().find_map(|s| if let Stmt::Let { dst, op: Op::ConstI32(c) } = s { if dst == v { Some(*c as i64) } else { None } } else { None }),
            _ => None,
        }
    }
    fn def<'a>(f: &'a Function, v: ValueId) -> Option<&'a Op> {
        f.stmts.iter().find_map(|s| if let Stmt::Let { dst, op } = s { if *dst == v { Some(op) } else { None } } else { None })
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
                    _ => Err(name),
                }
            }
            Opnd::Var(x) => {
                if !seen.insert((true, x.0)) {
                    return Ok(());
                }
                let mut ok_update = false;
                for s in &f.stmts {
                    match s {
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
    for (i, s) in f.stmts.iter().enumerate() {
        if let Stmt::TileProduce { tile, slot, version } | Stmt::TileConsume { tile, slot, version } = s {
            // the slot of a private (per-wave) region is the wave key: it does not evolve with the loop iteration
            let private = f.contract.tiles.iter().any(|t| &t.name == tile && t.private);
            for o in if private { vec![version] } else { vec![slot, version] } {
                walk(f, o, &mut seen).map_err(|why| format!("stmt {i}: the tile marker depends on {why}"))?;
            }
        }
    }
    Ok(())
}

/// Conservative bound M such that the tile markers are periodic in the loop iteration with a period dividing 2^M: every `& mask` contributes its width (msb + 1), every `>> k`
/// in the cone adds k (bit j of `x >> k` is bit j + k of x), and a counter toggled by `^ const` contributes 1. (`+ const` counters are affine: only their low bits matter for a
/// mask, and versions only matter through their differences.)
fn period_bits(f: &Function) -> u32 {
    fn konst(f: &Function, o: &Opnd) -> Option<i64> {
        match o {
            Opnd::I32(c) => Some(*c as i64),
            Opnd::Val(v) => f.stmts.iter().find_map(|s| if let Stmt::Let { dst, op: Op::ConstI32(c) } = s { if dst == v { Some(*c as i64) } else { None } } else { None }),
            _ => None,
        }
    }
    fn def<'a>(f: &'a Function, v: ValueId) -> Option<&'a Op> {
        f.stmts.iter().find_map(|s| if let Stmt::Let { dst, op } = s { if *dst == v { Some(op) } else { None } } else { None })
    }
    struct Acc {
        mask_bits: u32,
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
                    Op::LshrI32(_, y) => {
                        if let Some(k) = konst(f, y) {
                            a.shifts += k.max(0) as u32;
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
                for s in &f.stmts {
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
    let mut a = Acc { mask_bits: 0, shifts: 0, xor_counter: false, seen: HashSet::new() };
    for s in &f.stmts {
        if let Stmt::TileProduce { tile, slot, version } | Stmt::TileConsume { tile, slot, version } = s {
            if !f.contract.tiles.iter().any(|t| &t.name == tile && t.private) {
                walk(f, slot, &mut a);
            }
            walk(f, version, &mut a);
        }
    }
    let m = if a.mask_bits > 0 { a.mask_bits + a.shifts } else { 0 };
    m.max(if a.xor_counter { 1 } else { 0 })
}

fn enumerate_traces(f: &Function, role: usize, n: usize, div: &HashSet<usize>, diags: &mut Vec<(&'static str, Severity, usize, String)>, untagged: &mut BTreeSet<usize>) -> Vec<Trace> {
    let mut out: Vec<Trace> = Vec::new();
    let mut work: Vec<Vec<bool>> = vec![Vec::new()];
    let mut unsupported: Vec<(usize, String)> = Vec::new();
    while let Some(prefix) = work.pop() {
        if out.len() + work.len() > 64 {
            diags.push(("GV-TILE-UNSUPPORTED", Severity::Error, 0, format!("more than 64 control-flow paths through the wave-uniform runtime ifs of role {role}: the tile protocol is not simulated for all of them")));
            break;
        }
        let mut sim = Sim { f, role, n, div, vals: HashMap::new(), vars: HashMap::new(), items: Vec::new(), evs: Vec::new(), ord: 0, open: None, diags, untagged, steps: 0, break_taken: false, decisions: prefix.clone(), dpos: 0, taken: Vec::new(), loop_depth: 0, in_role: false, unsupported: &mut unsupported };
        sim.exec(0, f.stmts.len());
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
        out.push(Trace { items, evs, path: taken });
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

fn check_pair(f: &Function, traces: [&Trace; 2], n: usize, diags: &mut Vec<(&'static str, Severity, usize, String)>, checked: &mut usize) {
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
        *checked += 1;
        if f.contract.tiles[c.ev.tile].private {
            // own-wave semantics: the wave reads a version it produced AND completed (wait_vm/wait_lds(0)) itself; no other wave is involved, so no barrier is required
            let tname = f.contract.tiles[c.ev.tile].name.clone();
            let own: Vec<&Info> = infos.iter().filter(|p| p.ev.produce && p.ev.tile == c.ev.tile && p.role == c.role).collect();
            let completed: Vec<&&Info> = own.iter().filter(|p| p.done.map_or(false, |d| d < c.ev.start)).collect();
            let stmt = c.ev.stmt;
            let where_ = format!("(role {}, {} iteration(s) simulated)", c.role, n);
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
        let prods: Vec<&Info> = infos.iter().filter(|p| p.ev.produce && (p.ev.tile, p.ev.slot) == key).collect();
        let stmt = c.ev.stmt;
        let where_ = format!("(role {}, {} iteration(s) simulated)", c.role, n);
        if c.done.is_none() && traces[c.role].items.get(c.ev.last + 1).is_some() {
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
        for p in prods.iter().filter(|p| p.ev.ver == c.ev.ver && p.ev.ord <= c.ev.ord) {
            if !visible(p) {
                diags.push(("GV-TILE-STALE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] version {} at phase {}: the part produced by role {} (stmt {}) is not yet published (publish phase {}) {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord, p.role, p.ev.stmt, p.publish.map_or("never".to_string(), |v| v.to_string()))));
            }
        }
        for q in prods.iter().filter(|q| q.ev.ver != c.ev.ver) {
            let overlap = if q.role == c.role {
                q.ev.start < c.ev.start && !q.done.map_or(false, |d| d < c.ev.start)
            } else {
                q.ev.ord <= c.done_ord && q.publish.map_or(true, |v| v > c.ev.ord)
            };
            if overlap {
                diags.push(("GV-TILE-RACE", Severity::Error, stmt, format!("stmt {stmt}: read of `{}`[{}] (version {}, phases {}..{}) overlaps the write of version {} by role {} (stmt {}, issued at phase {}, published at {}): no barrier separates them {where_}", name(c.ev.tile), c.ev.slot, c.ev.ver, c.ev.ord, c.done_ord, q.ev.ver, q.role, q.ev.stmt, q.ev.ord, q.publish.map_or("never".to_string(), |v| v.to_string()))));
            }
        }
    }
    for (i, a) in infos.iter().enumerate().filter(|(_, x)| x.ev.produce) {
        for b in infos.iter().skip(i + 1).filter(|x| x.ev.produce) {
            if a.role != b.role && (a.ev.tile, a.ev.slot) == (b.ev.tile, b.ev.slot) && a.ev.ver != b.ev.ver && !f.contract.tiles[a.ev.tile].private {
                let (aw, bw) = ((a.ev.ord, a.publish.unwrap_or(u32::MAX)), (b.ev.ord, b.publish.unwrap_or(u32::MAX)));
                if aw.0 < bw.1 && bw.0 < aw.1 {
                    diags.push(("GV-TILE-RACE", Severity::Error, a.ev.stmt, format!("stmt {} / stmt {}: roles {} and {} write different versions ({} / {}) of `{}`[{}] in overlapping phases {}..{} / {}..{} (write-write race)", a.ev.stmt, b.ev.stmt, a.role, b.role, a.ev.ver, b.ev.ver, name(a.ev.tile), a.ev.slot, aw.0, aw.1, bw.0, bw.1)));
                }
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
    // horizon: the markers' period bound P = 2^M must fit twice into the simulated iterations
    let max_slots = tiles.iter().map(|t| t.slots as usize).max().unwrap_or(1);
    let base_h = 10usize.max(2 * max_slots + 4);
    let m = period_bits(f);
    let pbound = 1usize << m.min(20);
    let need = base_h.max(2 * pbound + 2);
    const HORIZON_CAP: usize = 70;
    let nmax;
    match (&f.contract.tile_trust, periodic_cone(f)) {
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
        (None, Ok(())) => {
            if need > HORIZON_CAP {
                nmax = base_h;
                rep.add(
                    "GV-TILE-UNROLL-LIMIT",
                    Severity::Error,
                    format!("the tile markers' slot/version expressions may have a period of up to {pbound} iterations (a shift/mask in their cone reaches bit {}), which needs {need} simulated iterations; the horizon is capped at {HORIZON_CAP}: a launch with more iterations than {nmax} is NOT covered (simplify the index arithmetic or declare b.tile_trust_periodic(reason))", m.saturating_sub(1)),
                );
            } else {
                nmax = need;
                rep.add("GV-TILE-PERIOD", Severity::Info, format!("tile markers are periodic with a period dividing {pbound} iteration(s); {nmax} iterations (>= 2 x period + 2, >= 2 x ring slots + 4) are simulated for every exit parity, which covers any trip count"));
            }
        }
    }
    let div = divergent_ifs(f);
    let mut diags: Vec<(&'static str, Severity, usize, String)> = Vec::new();
    let mut untagged: BTreeSet<usize> = BTreeSet::new();
    let mut checked = 0usize;
    let mut paths = 0usize;
    for n in 1..=nmax {
        let t0 = enumerate_traces(f, 0, n, &div, &mut diags, &mut untagged);
        let t1 = enumerate_traces(f, 1, n, &div, &mut diags, &mut untagged);
        for a in &t0 {
            for b in &t1 {
                if compatible(a, b) {
                    paths += 1;
                    check_pair(f, [a, b], n, &mut diags, &mut checked);
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
    rep.add("GV-TILE-PERIOD", Severity::Info, format!("tile simulation: {paths} role-path combination(s) over {nmax} loop-iteration scenarios, {checked} read events checked"));
    if errs == 0 {
        for _ in 0..checked {
            rep.ok("tile-reads");
        }
    }
}