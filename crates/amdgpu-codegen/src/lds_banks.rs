//! Static LDS bank-conflict report: for every LDS load / store whose per-lane address is an affine-ish function of the lane id (a lane vector
//! plus an unknown wave-uniform term), the bank cost under the LDS model of `amdgpu_dsl::tdm_pad` (64 banks of 4 bytes; a request served in
//! phases of 256 / bytes lanes; a phase takes as many cycles as the most distinct dwords one bank holds). A uniform offset only rotates the
//! banks, so the unknown wave-uniform terms (kernel arguments, loop counters, work-group ids) do not change the cost. Addresses that mix the
//! lane with an unknown value non-additively (a lane-varying load result, `and` of an unknown, ...) are reported as not analysed.
//!
//! `amdc lds-banks <kernel> | --all-gfx1250 [--min-ways N]` prints it; compiles for gfx1250 warn (stderr) about accesses of more than
//! `WARN_WAYS` ways.

use amdgpu_dsl::{BinOp, Function, MemTy, Op, Opnd, Stmt, TriOp};
use std::collections::HashMap;

/// accesses above this many ways get a compile warning (gfx1250)
pub const WARN_WAYS: u32 = 4;

#[derive(Clone, Debug)]
enum A {
    /// per-lane values plus (flag) an unknown wave-uniform additive term
    K(Vec<i64>, bool),
    /// not analysable
    X,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub stmt: usize,
    pub what: String,
    pub bytes: u32,
    /// the busiest bank's distinct dwords in the worst phase (1 = conflict-free)
    pub ways: u32,
    /// cycles (sum over phases) / the conflict-free minimum
    pub cost: u32,
    pub ideal: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub entries: Vec<Entry>,
    /// LDS accesses whose lane pattern could not be derived
    pub unanalysed: usize,
}

struct Ev<'a> {
    f: &'a Function,
    lanes: usize,
    defs: HashMap<u32, &'a Op>,
    vars: HashMap<u32, A>,
    memo: HashMap<u32, A>,
    why: Option<HashMap<String, usize>>,
}

impl<'a> Ev<'a> {
    fn uni(&self) -> A {
        A::K(vec![0; self.lanes], true)
    }
    fn konst(&self, c: i64) -> A {
        A::K(vec![c; self.lanes], false)
    }
    fn opnd(&mut self, o: &Opnd) -> A {
        match o {
            Opnd::I32(c) => self.konst(*c as i64),
            Opnd::F32(_) => A::X,
            Opnd::Var(v) => self.vars.get(&v.0).cloned().unwrap_or(A::X),
            Opnd::Val(v) => self.val(*v),
        }
    }
    fn val(&mut self, v: u32) -> A {
        if let Some(a) = self.memo.get(&v) {
            return a.clone();
        }
        // (a cycle guard: a value is evaluated once)
        self.memo.insert(v, A::X);
        let r = match self.defs.get(&v).copied() {
            Some(op) => {
                let r = self.op(op);
                // (AMDC_LDS_BANKS_WHY: the ops at which the lane pattern was lost, counted)
                if matches!(r, A::X) && self.why.is_some() {
                    let name = format!("{op:?}");
                    let k = name.split(|c: char| c == '(' || c == ' ' || c == '{').next().unwrap_or("?").to_string();
                    let operands_ok = op.operands().iter().all(|o| !matches!(o, Opnd::Val(x) if matches!(self.memo.get(x), Some(A::X))) && !matches!(o, Opnd::Var(x) if matches!(self.vars.get(&x.0), Some(A::X) | None)));
                    if operands_ok {
                        *self.why.as_mut().unwrap().entry(k).or_insert(0) += 1;
                    }
                }
                r
            }
            None => A::X,
        };
        self.memo.insert(v, r.clone());
        r
    }
    fn lane_const(a: &A) -> Option<i64> {
        match a {
            A::K(v, false) if v.iter().all(|x| *x == v[0]) => Some(v[0]),
            _ => None,
        }
    }
    fn bin(&mut self, a: &Opnd, b: &Opnd, f: &dyn Fn(i64, i64) -> i64, additive: bool, scale_ok: bool) -> A {
        let (x, y) = (self.opnd(a), self.opnd(b));
        match (&x, &y) {
            // any function of two wave-uniform values is wave-uniform (unknown when either is)
            (A::K(p, pu), A::K(q, qu)) if (*pu || *qu) && p.iter().all(|v| *v == p[0]) && q.iter().all(|v| *v == q[0]) => A::K(vec![0; self.lanes], true),
            (A::K(p, pu), A::K(q, qu)) => {
                if additive {
                    return A::K(p.iter().zip(q).map(|(a, b)| f(*a, *b)).collect(), *pu || *qu);
                }
                if !*pu && !*qu {
                    return A::K(p.iter().zip(q).map(|(a, b)| f(*a, *b)).collect(), false);
                }
                // a scaling (mul / shl) of an unknown uniform by a lane constant stays an unknown uniform term
                if scale_ok {
                    if let Some(c) = Self::lane_const(&y) {
                        return A::K(p.iter().map(|a| f(*a, c)).collect(), *pu);
                    }
                    if let Some(c) = Self::lane_const(&x) {
                        return A::K(q.iter().map(|b| f(c, *b)).collect(), *qu);
                    }
                }
                A::X
            }
            _ => A::X,
        }
    }
    fn op(&mut self, op: &Op) -> A {
        let w = |x: i64| x as i32 as i64;
        match op {
            Op::ConstI32(c) => self.konst(*c as i64),
            Op::WorkitemId => A::K((0..self.lanes as i64).collect(), false),
            Op::KernargI32 { .. } | Op::KernargPtr { .. } | Op::WorkgroupIdX | Op::ReadFirstLane(_) => self.uni(),
            Op::MovS { a, .. } | Op::MovV { a, .. } | Op::BitcastI32(a) => self.opnd(a),
            Op::AddK { a, k } => match self.opnd(a) {
                A::K(v, u) => A::K(v.iter().map(|x| w(x + *k as i64)).collect(), u),
                A::X => A::X,
            },
            // a select between two values with the same lane pattern (up to a uniform shift: e.g. the two slots' bases) keeps the pattern;
            // a wave-uniform condition between different patterns, or a lane-dependent one, does not
            Op::Select { cond, a, b, .. } => match (self.opnd(a), self.opnd(b)) {
                (A::K(p, _), A::K(q, _)) => {
                    let d = q[0] - p[0];
                    if p.iter().zip(&q).all(|(x, y)| y - x == d) {
                        let _ = cond;
                        A::K(p, d != 0 || true)
                    } else {
                        A::X
                    }
                }
                _ => A::X,
            },
            Op::AddI32(a, b) => self.bin(a, b, &|x, y| w(x + y), true, false),
            Op::MulI32(a, b) => self.bin(a, b, &|x, y| w(x * y), false, true),
            Op::ShlI32(a, b) => {
                // only a lane-constant shift amount scales; an unknown uniform amount of a uniform value stays uniform
                let s = self.opnd(b);
                if let (A::K(sv, true), A::K(av, _)) = (&s, &self.opnd(a)) {
                    if sv.iter().all(|v| *v == sv[0]) && av.iter().all(|v| *v == av[0]) {
                        return A::K(vec![0; self.lanes], true);
                    }
                }
                match Self::lane_const(&s) {
                    Some(c) if (0..32).contains(&c) => match self.opnd(a) {
                        A::K(v, u) => A::K(v.iter().map(|x| w(x << c)).collect(), u),
                        A::X => A::X,
                    },
                    _ => A::X,
                }
            }
            Op::LshrI32(a, b) => self.bin(a, b, &|x, y| if (0..32).contains(&y) { ((x as i32 as u32) >> y) as i64 } else { 0 }, false, false),
            Op::AndI32(a, b) => self.bin(a, b, &|x, y| x & y, false, false),
            Op::XorI32(a, b) => self.bin(a, b, &|x, y| x ^ y, false, false),
            Op::MinI32(a, b) => self.bin(a, b, &|x, y| x.min(y), false, false),
            Op::Bin(k, a, b) => match k {
                BinOp::AddNcI32 => self.bin(a, b, &|x, y| w(x + y), true, false),
                BinOp::SubU32 => match (self.opnd(a), self.opnd(b)) {
                    (A::K(p, pu), A::K(q, qu)) => A::K(p.iter().zip(&q).map(|(a, b)| w(a - b)).collect(), pu || qu),
                    _ => A::X,
                },
                BinOp::OrB32 => self.bin(a, b, &|x, y| x | y, false, false),
                BinOp::AshrI32 => self.bin(a, b, &|x, y| if (0..32).contains(&y) { (x as i32 >> y) as i64 } else { 0 }, false, false),
                BinOp::MaxI32 => self.bin(a, b, &|x, y| x.max(y), false, false),
                BinOp::MulU32U24 => self.bin(a, b, &|x, y| w((x & 0xff_ffff) * (y & 0xff_ffff)), false, true),
                // mbcnt with an all-ones mask: the lane index (+ the accumulator)
                BinOp::MbcntLo | BinOp::MbcntHi => {
                    let m = self.opnd(a);
                    match (Self::lane_const(&m), self.opnd(b)) {
                        (Some(-1), A::K(acc, u)) => {
                            let lo = matches!(k, BinOp::MbcntLo);
                            A::K(acc.iter().enumerate().map(|(l, x)| x + if lo { (l as i64).min(32) } else { (l as i64 - 32).max(0) }).collect(), u)
                        }
                        _ => A::X,
                    }
                }
                _ => A::X,
            },
            Op::Tri(k, a, b, c) => match k {
                TriOp::Add3U32 => {
                    let ab = self.bin(a, b, &|x, y| w(x + y), true, false);
                    match (ab, self.opnd(c)) {
                        (A::K(p, pu), A::K(q, qu)) => A::K(p.iter().zip(&q).map(|(a, b)| w(a + b)).collect(), pu || qu),
                        _ => A::X,
                    }
                }
                TriOp::LshlAddU32 => {
                    let s = self.opnd(b);
                    match (Self::lane_const(&s), self.opnd(a), self.opnd(c)) {
                        (Some(sh), A::K(p, pu), A::K(q, qu)) if (0..32).contains(&sh) => A::K(p.iter().zip(&q).map(|(x, y)| w((x << sh) + y)).collect(), pu || qu),
                        _ => A::X,
                    }
                }
                TriOp::MadU32U24 => {
                    let ab = self.bin(a, b, &|x, y| w((x & 0xff_ffff) * (y & 0xff_ffff)), false, true);
                    match (ab, self.opnd(c)) {
                        (A::K(p, pu), A::K(q, qu)) => A::K(p.iter().zip(&q).map(|(a, b)| w(a + b)).collect(), pu || qu),
                        _ => A::X,
                    }
                }
                _ => A::X,
            },
            _ => A::X,
        }
    }
}

fn mem_bytes(t: MemTy) -> u32 {
    match t {
        MemTy::U8 | MemTy::I8 => 1,
        MemTy::U16 | MemTy::I16 => 2,
        MemTy::B32 => 4,
        MemTy::B64 => 8,
        MemTy::B96 => 12,
        MemTy::B128 => 16,
    }
}

/// (busiest bank's distinct dwords in the worst phase, cycles, conflict-free cycles) of one request
fn cost(lanes: &[i64], bytes: u32) -> (u32, u32, u32) {
    let per_phase = (256 / bytes.max(4)) as usize;
    let (mut worst, mut total, mut ideal) = (1u32, 0u32, 0u32);
    for phase in lanes.chunks(per_phase.max(1)) {
        let mut banks: Vec<Vec<i64>> = vec![Vec::new(); 64];
        for off in phase {
            for k in 0..bytes.div_ceil(4) as i64 {
                let dw = off.div_euclid(4) + k;
                let b = &mut banks[dw.rem_euclid(64) as usize];
                if !b.contains(&dw) {
                    b.push(dw);
                }
            }
        }
        let m = banks.iter().map(|b| b.len() as u32).max().unwrap_or(1).max(1);
        worst = worst.max(m);
        total += m;
        ideal += 1;
    }
    (worst, total, ideal)
}

/// The report of `f` for a wave of `lanes` lanes (32 on gfx1250, 64 on gfx950).
pub fn report(f: &Function, lanes: usize) -> Report {
    let mut ev = Ev { f, lanes, defs: HashMap::new(), vars: HashMap::new(), memo: HashMap::new(), why: std::env::var("AMDC_LDS_BANKS_WHY").ok().map(|_| HashMap::new()) };
    for s in f.stmts() {
        if let Stmt::Let { dst, op } = s {
            ev.defs.insert(*dst, op);
        }
    }
    // loop variables: the lane pattern of their initial value when every update keeps it (a uniform counter / a fixed per-lane base);
    // otherwise not analysable
    let mut inits: Vec<(u32, Opnd)> = Vec::new();
    let mut sets: Vec<(u32, Opnd)> = Vec::new();
    for s in f.stmts() {
        match s {
            Stmt::BeginLoop { vars, init } => inits.extend(vars.iter().zip(init.iter()).map(|(v, i)| (v.0, *i))),
            Stmt::SetVar { var, value } => sets.push((var.0, *value)),
            _ => {}
        }
    }
    for (v, i) in &inits {
        let a = ev.opnd(i);
        ev.vars.insert(*v, a);
    }
    for (v, val) in &sets {
        let a = ev.opnd(val);
        let keep = match (ev.vars.get(v), &a) {
            // the update's lane part must equal the initial one up to a uniform shift
            (Some(A::K(p, _)), A::K(q, _)) => {
                let d = q[0] - p[0];
                p.iter().zip(q).all(|(x, y)| y - x == d)
            }
            _ => false,
        };
        if keep {
            if let Some(A::K(_, u)) = ev.vars.get_mut(v) {
                *u = true;
            }
        } else {
            ev.vars.insert(*v, A::X);
        }
    }
    // (memoised values were computed with the initial variable patterns: recompute with the final ones)
    ev.memo.clear();
    let _ = ev.f;
    let mut rep = Report::default();
    let mut add = |ev: &mut Ev, stmt: usize, what: &str, addr: &Opnd, extra: &[i64], bytes: u32, rep: &mut Report| {
        match ev.opnd(addr) {
            A::K(v, _) => {
                let mut lanes: Vec<i64> = Vec::new();
                for e in extra {
                    lanes.clear();
                    lanes.extend(v.iter().map(|x| x + e));
                    let (ways, c, i) = cost(&lanes, bytes);
                    if std::env::var("AMDC_LDS_BANKS_STMT").ok().and_then(|s| s.parse::<usize>().ok()) == Some(stmt) {
                        eprintln!("stmt {stmt} {what}: lane offsets {lanes:?}");
                    }
                    rep.entries.push(Entry { stmt, what: what.to_string(), bytes, ways, cost: c, ideal: i });
                }
            }
            A::X => rep.unanalysed += 1,
        }
    };
    let lost_vars: Vec<u32> = ev.vars.iter().filter(|(_, a)| matches!(a, A::X)).map(|(k, _)| *k).collect();
    if ev.why.is_some() && !lost_vars.is_empty() {
        eprintln!("lds_banks: loop variables without a lane pattern: {lost_vars:?}");
    }
    for (i, s) in f.stmts().iter().enumerate() {
        match s {
            Stmt::Let { op, .. } => match op {
                Op::LoadLdsB32 { addr } => add(&mut ev, i, "ds_load_b32", addr, &[0], 4, &mut rep),
                Op::LoadLdsB64 { addr } => add(&mut ev, i, "ds_load_b64", addr, &[0], 8, &mut rep),
                Op::LoadLdsB128 { addr } => add(&mut ev, i, "ds_load_b128", addr, &[0], 16, &mut rep),
                Op::LoadLdsB128x2 { lo, hi } => {
                    add(&mut ev, i, "ds_load_b128 (lo)", lo, &[0], 16, &mut rep);
                    add(&mut ev, i, "ds_load_b128 (hi)", hi, &[0], 16, &mut rep);
                }
                Op::LoadLdsN { addr, ty } => add(&mut ev, i, "ds_load", addr, &[0], mem_bytes(*ty), &mut rep),
                Op::LoadLdsD16 { addr, ty, .. } => add(&mut ev, i, "ds_load_d16", addr, &[0], if matches!(ty, amdgpu_dsl::D16Ty::B16) { 2 } else { 1 }, &mut rep),
                Op::LoadLds2 { addr, off0, off1, b64, st64 } => {
                    let unit = if *b64 { 8 } else { 4 } * if *st64 { 64 } else { 1 };
                    add(&mut ev, i, "ds_load_2addr", addr, &[*off0 as i64 * unit, *off1 as i64 * unit], if *b64 { 8 } else { 4 }, &mut rep);
                }
                _ => {}
            },
            Stmt::StoreLdsN { addr, ty, .. } => add(&mut ev, i, "ds_store", addr, &[0], mem_bytes(*ty), &mut rep),
            Stmt::StoreLdsB32 { addr, .. } => add(&mut ev, i, "ds_store_b32", addr, &[0], 4, &mut rep),
            Stmt::StoreLdsB64 { addr, .. } => add(&mut ev, i, "ds_store_b64", addr, &[0], 8, &mut rep),
            Stmt::StoreLdsB128 { addr, .. } => add(&mut ev, i, "ds_store_b128", addr, &[0], 16, &mut rep),
            Stmt::StoreLds2 { addr, off0, off1, b64, st64, .. } => {
                let unit = if *b64 { 8 } else { 4 } * if *st64 { 64 } else { 1 };
                add(&mut ev, i, "ds_store_2addr", addr, &[*off0 as i64 * unit, *off1 as i64 * unit], if *b64 { 8 } else { 4 }, &mut rep);
            }
            _ => {}
        }
    }
    if let Some(w) = &ev.why {
        let mut v: Vec<_> = w.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        eprintln!("lds_banks: lane pattern lost at {v:?}");
    }
    rep
}

/// A text table of the entries with `ways >= min_ways` (worst first) and a summary line.
pub fn render(name: &str, r: &Report, min_ways: u32) -> String {
    let mut es: Vec<&Entry> = r.entries.iter().filter(|e| e.ways >= min_ways).collect();
    es.sort_by(|a, b| b.ways.cmp(&a.ways).then(a.stmt.cmp(&b.stmt)));
    let mut s = String::new();
    for e in &es {
        s.push_str(&format!("{name}: stmt {}: {} ({} B/lane): {}-way, {} cycles vs {} conflict-free\n", e.stmt, e.what, e.bytes, e.ways, e.cost, e.ideal));
    }
    let worst = r.entries.iter().map(|e| e.ways).max().unwrap_or(0);
    let over = r.entries.iter().filter(|e| e.ways > 1).count();
    s.push_str(&format!("{name}: {} LDS access(es) analysed, {} with bank conflicts (worst {}-way), {} not analysable\n", r.entries.len(), over, worst, r.unanalysed));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use amdgpu_dsl::{FnBuilder, Opnd};

    fn one(stride: i32) -> Report {
        let mut b = FnBuilder::new("lb", 32);
        b.lds(65536);
        let t = b.let_(Op::WorkitemId);
        let a = b.let_(Op::MulI32(Opnd::Val(t), Opnd::I32(stride)));
        let ka = b.let_(Op::KernargI32 { offset: 0 });
        let a2 = b.let_(Op::AddI32(Opnd::Val(a), Opnd::Val(ka)));
        let _ = b.load_lds(Opnd::Val(a2), MemTy::B32);
        report(&b.finish(), 32)
    }

    #[test]
    fn unit_stride_is_conflict_free_and_a_256_byte_stride_is_32_way() {
        assert_eq!(one(4).entries[0].ways, 1);
        assert_eq!(one(256).entries[0].ways, 32);
        assert_eq!(one(260).entries[0].ways, 1);
        assert_eq!(one(8).entries[0].ways, 1); // 32 lanes x 8 B = 64 dwords: two lanes per bank? no: dwords 0, 2, .., 62: distinct banks
        assert_eq!(one(128).entries[0].ways, 16);
    }
}
