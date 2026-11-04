//! `amdgpu-sym`: the symbolic arithmetic shared by the kernel contracts (`amdgpu_dsl::Ext`), the layout algebra and the GPU-semantics verifier
//! (docs/sym.md). Exact rationals (`Q`), polynomials over contract symbols (`Poly`), the constraint context (`Ctx`: symbol bounds, constants,
//! substitutions) with its sound `p >= 0` tests (shifted-coefficient test and box test), and candidate-bound intervals (`Itv`).
//! Zero dependencies, no unsafe.

#![forbid(unsafe_code)]

pub mod bits;

use std::collections::{BTreeMap, HashMap};


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Q {
    pub n: i128,
    pub d: i128,
}

pub fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Q {
    pub fn new(n: i128, d: i128) -> Q {
        let g = gcd(n, d).max(1);
        let s = if d < 0 { -1 } else { 1 };
        Q { n: s * n / g, d: s * d / g }
    }
    pub fn int(n: i128) -> Q {
        Q { n, d: 1 }
    }
    pub fn add(self, o: Q) -> Q {
        Q::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }
    pub fn mul(self, o: Q) -> Q {
        Q::new(self.n * o.n, self.d * o.d)
    }
    pub fn is_zero(self) -> bool {
        self.n == 0
    }
    pub fn is_neg(self) -> bool {
        self.n < 0
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Sym {
    Ka(u32),
    Grid(u8),
    Named(String),
}

impl Sym {
    pub fn name(&self) -> String {
        match self {
            Sym::Ka(o) => format!("ka(0x{o:X})"),
            Sym::Grid(a) => ["grid_x", "grid_y", "grid_z"][(*a as usize).min(2)].to_string(),
            Sym::Named(n) => n.clone(),
        }
    }
}

/// A polynomial over the contract symbols with rational coefficients; monomials are sorted multisets of symbols.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Poly(pub BTreeMap<Vec<Sym>, Q>);

impl Poly {
    pub fn konst(q: Q) -> Poly {
        let mut m = BTreeMap::new();
        if !q.is_zero() {
            m.insert(Vec::new(), q);
        }
        Poly(m)
    }
    pub fn int(n: i128) -> Poly {
        Poly::konst(Q::int(n))
    }
    pub fn sym(s: Sym) -> Poly {
        let mut m = BTreeMap::new();
        m.insert(vec![s], Q::int(1));
        Poly(m)
    }
    pub fn add(&self, o: &Poly) -> Poly {
        let mut m = self.0.clone();
        for (k, v) in &o.0 {
            let e = m.entry(k.clone()).or_insert(Q::int(0));
            *e = e.add(*v);
        }
        m.retain(|_, v| !v.is_zero());
        Poly(m)
    }
    pub fn scale(&self, c: Q) -> Poly {
        let mut m = BTreeMap::new();
        for (k, v) in &self.0 {
            let x = v.mul(c);
            if !x.is_zero() {
                m.insert(k.clone(), x);
            }
        }
        Poly(m)
    }
    pub fn sub(&self, o: &Poly) -> Poly {
        self.add(&o.scale(Q::int(-1)))
    }
    pub fn mul(&self, o: &Poly) -> Option<Poly> {
        let mut m: BTreeMap<Vec<Sym>, Q> = BTreeMap::new();
        for (k1, v1) in &self.0 {
            for (k2, v2) in &o.0 {
                let mut k = k1.clone();
                k.extend(k2.iter().cloned());
                if k.len() > 8 {
                    return None;
                }
                k.sort();
                let e = m.entry(k).or_insert(Q::int(0));
                *e = e.add(v1.mul(*v2));
            }
        }
        m.retain(|_, v| !v.is_zero());
        Some(Poly(m))
    }
    /// floor of a constant polynomial (values are integers).
    pub fn floor_const(&self) -> Option<i128> {
        self.as_const().map(|q| q.n.div_euclid(q.d))
    }
    pub fn as_const(&self) -> Option<Q> {
        if self.0.is_empty() {
            return Some(Q::int(0));
        }
        if self.0.len() == 1 {
            if let Some(v) = self.0.get(&Vec::new()) {
                return Some(*v);
            }
        }
        None
    }
    pub fn show(&self) -> String {
        if self.0.is_empty() {
            return "0".to_string();
        }
        let mut parts = Vec::new();
        for (k, v) in &self.0 {
            let c = if v.d == 1 { format!("{}", v.n) } else { format!("{}/{}", v.n, v.d) };
            if k.is_empty() {
                parts.push(c);
            } else {
                let mono = k.iter().map(|s| s.name()).collect::<Vec<_>>().join("*");
                parts.push(if c == "1" { mono } else { format!("{c}*{mono}") });
            }
        }
        parts.join(" + ").replace("+ -", "- ")
    }
}

/// The constraint context: symbol lower / upper bounds, constants and substitutions (the kernel contract's facts).
#[derive(Clone, Debug, Default)]
pub struct Ctx {
    pub lb: HashMap<Sym, i64>,
    pub konst: HashMap<Sym, i64>,
    pub subst: HashMap<Sym, Poly>,
    /// declared upper bounds (`ka_range` / `ka_set` / `sym_range` maxima; absent = unbounded)
    pub ub: HashMap<Sym, i64>,
}

impl Ctx {
    pub fn sym_poly(&self, s: Sym) -> Poly {
        if let Some(k) = self.konst.get(&s) {
            return Poly::int(*k as i128);
        }
        if let Some(p) = self.subst.get(&s) {
            return p.clone();
        }
        Poly::sym(s)
    }
    pub fn lower_bound(&self, s: &Sym) -> i128 {
        match self.lb.get(s) {
            Some(v) => *v as i128,
            None => match s {
                Sym::Grid(_) | Sym::Named(_) => 1,
                Sym::Ka(_) => 0,
            },
        }
    }
    /// Sufficient test for `p >= 0` for all admissible symbol values: shift every symbol s = lb(s) + t (t >= 0) and require non-negative coefficients.
    pub fn nonneg(&self, p: &Poly) -> bool {
        let mut out: BTreeMap<Vec<Sym>, Q> = BTreeMap::new();
        for (mono, c) in &p.0 {
            let k = mono.len();
            if k > 10 {
                return false;
            }
            for mask in 0u32..(1u32 << k) {
                let mut coef = *c;
                let mut ts: Vec<Sym> = Vec::new();
                for (i, s) in mono.iter().enumerate() {
                    if mask >> i & 1 == 1 {
                        ts.push(s.clone());
                    } else {
                        coef = coef.mul(Q::int(self.lower_bound(s)));
                    }
                }
                ts.sort();
                let e = out.entry(ts).or_insert(Q::int(0));
                *e = e.add(coef);
            }
        }
        out.values().all(|v| !v.is_neg()) || self.nonneg_box(p)
    }
    /// Second sufficient test: the minimum of every monomial over the box [lb, ub] of its (non-negative) symbols - `c * prod lb` for c >= 0, `c * prod ub`
    /// for c < 0 (needs every symbol of the monomial bounded above) - sums to >= 0. Uses the declared maxima (`ka_range`, `ka_set`, `sym_range`).
    pub fn nonneg_box(&self, p: &Poly) -> bool {
        let mut sum = Q::int(0);
        for (mono, c) in &p.0 {
            let mut m = *c;
            for s in mono {
                let b = if c.is_neg() {
                    match self.ub.get(s) {
                        Some(u) => *u as i128,
                        None => return false,
                    }
                } else {
                    self.lower_bound(s)
                };
                if b < 0 {
                    return false;
                }
                m = m.mul(Q::int(b));
            }
            sum = sum.add(m);
        }
        !sum.is_neg()
    }
    pub fn le(&self, a: &Poly, b: &Poly) -> bool {
        self.nonneg(&b.sub(a))
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// intervals: sets of candidate lower / upper bounds (empty = unbounded)

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Itv {
    pub lo: Vec<Poly>,
    pub hi: Vec<Poly>,
}

pub const CAND: usize = 4;

pub fn cap(mut v: Vec<Poly>) -> Vec<Poly> {
    let mut out: Vec<Poly> = Vec::new();
    for p in v.drain(..) {
        if !out.contains(&p) {
            out.push(p);
        }
        if out.len() >= CAND {
            break;
        }
    }
    out
}

impl Itv {
    pub fn top() -> Itv {
        Itv::default()
    }
    pub fn point(p: Poly) -> Itv {
        Itv { lo: vec![p.clone()], hi: vec![p] }
    }
    pub fn range(lo: Poly, hi: Poly) -> Itv {
        Itv { lo: vec![lo], hi: vec![hi] }
    }
    pub fn konst(n: i128) -> Itv {
        Itv::point(Poly::int(n))
    }
    pub fn point_const(&self) -> Option<Q> {
        if self.lo.len() == 1 && self.hi.len() == 1 && self.lo[0] == self.hi[0] {
            return self.lo[0].as_const();
        }
        None
    }
    pub fn add(&self, o: &Itv) -> Itv {
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for a in &self.lo {
            for b in &o.lo {
                lo.push(a.add(b));
            }
        }
        for a in &self.hi {
            for b in &o.hi {
                hi.push(a.add(b));
            }
        }
        Itv { lo: cap(lo), hi: cap(hi) }
    }
    pub fn add_const(&self, c: i128) -> Itv {
        self.add(&Itv::konst(c))
    }
    pub fn scale(&self, c: Q) -> Itv {
        if c.is_neg() {
            Itv { lo: cap(self.hi.iter().map(|p| p.scale(c)).collect()), hi: cap(self.lo.iter().map(|p| p.scale(c)).collect()) }
        } else {
            Itv { lo: cap(self.lo.iter().map(|p| p.scale(c)).collect()), hi: cap(self.hi.iter().map(|p| p.scale(c)).collect()) }
        }
    }
    pub fn nonneg(&self, cx: &Ctx) -> bool {
        self.lo.iter().any(|p| cx.nonneg(p))
    }
    pub fn mul(&self, o: &Itv, cx: &Ctx) -> Itv {
        if let Some(c) = self.point_const() {
            return o.scale(c);
        }
        if let Some(c) = o.point_const() {
            return self.scale(c);
        }
        if self.nonneg(cx) && o.nonneg(cx) {
            let mut lo = Vec::new();
            let mut hi = Vec::new();
            for a in &self.lo {
                for b in &o.lo {
                    if let Some(p) = a.mul(b) {
                        lo.push(p);
                    }
                }
            }
            for a in &self.hi {
                for b in &o.hi {
                    if let Some(p) = a.mul(b) {
                        hi.push(p);
                    }
                }
            }
            return Itv { lo: cap(lo), hi: cap(hi) };
        }
        Itv::top()
    }
    /// Union (join) of two intervals: a bound survives only if it bounds both operands.
    pub fn join(&self, o: &Itv, cx: &Ctx) -> Itv {
        let mut hi = Vec::new();
        for x in &self.hi {
            if o.hi.iter().any(|y| cx.le(y, x)) {
                hi.push(x.clone());
            }
        }
        for y in &o.hi {
            if self.hi.iter().any(|x| cx.le(x, y)) {
                hi.push(y.clone());
            }
        }
        let mut lo = Vec::new();
        for x in &self.lo {
            if o.lo.iter().any(|y| cx.le(x, y)) {
                lo.push(x.clone());
            }
        }
        for y in &o.lo {
            if self.lo.iter().any(|x| cx.le(y, x)) {
                lo.push(y.clone());
            }
        }
        // two operands that are both provably non-negative: 0 bounds the join even when their symbolic lower bounds are incomparable
        if self.nonneg(cx) && o.nonneg(cx) {
            lo.push(Poly::int(0));
        }
        Itv { lo: cap(lo), hi: cap(hi) }
    }
    /// min(a, b): any upper bound of either operand bounds it; a lower bound must bound both.
    pub fn min(&self, o: &Itv, cx: &Ctx) -> Itv {
        let j = self.join(o, cx);
        let mut hi = self.hi.clone();
        hi.extend(o.hi.iter().cloned());
        Itv { lo: j.lo, hi: cap(hi) }
    }
    /// Intersection (assume): all bounds of both.
    pub fn meet(&self, o: &Itv) -> Itv {
        let mut lo = o.lo.clone();
        lo.extend(self.lo.iter().cloned());
        let mut hi = o.hi.clone();
        hi.extend(self.hi.iter().cloned());
        Itv { lo: cap(lo), hi: cap(hi) }
    }
    /// self is contained in o (every bound of o is implied).
    pub fn within(&self, o: &Itv, cx: &Ctx) -> bool {
        let lo_ok = o.lo.iter().all(|b| self.lo.iter().any(|a| cx.le(b, a)));
        let hi_ok = o.hi.iter().all(|b| self.hi.iter().any(|a| cx.le(a, b)));
        lo_ok && hi_ok
    }
    /// Drop dominated candidates (keep only the tightest provable bounds) so that fixpoint iteration compares canonical forms.
    pub fn norm(&self, cx: &Ctx) -> Itv {
        let mut hi: Vec<Poly> = Vec::new();
        for (i, x) in self.hi.iter().enumerate() {
            let dominated = self.hi.iter().enumerate().any(|(j, y)| j != i && cx.le(y, x) && (!cx.le(x, y) || j < i));
            if !dominated {
                hi.push(x.clone());
            }
        }
        let mut lo: Vec<Poly> = Vec::new();
        for (i, x) in self.lo.iter().enumerate() {
            let dominated = self.lo.iter().enumerate().any(|(j, y)| j != i && cx.le(x, y) && (!cx.le(y, x) || j < i));
            if !dominated {
                lo.push(x.clone());
            }
        }
        Itv { lo, hi }
    }
    pub fn show(&self) -> String {
        let l = if self.lo.is_empty() { "-inf".to_string() } else { self.lo.iter().map(|p| p.show()).collect::<Vec<_>>().join(" | ") };
        let h = if self.hi.is_empty() { "+inf".to_string() } else { self.hi.iter().map(|p| p.show()).collect::<Vec<_>>().join(" | ") };
        format!("[{l}, {h}]")
    }
}


impl Ctx {
    /// An empty context (no facts: every symbol has its default lower bound, no upper bound).
    pub fn new() -> Ctx {
        Ctx::default()
    }
    /// Fact `lo <= s` (keeps the tightest of repeated facts).
    pub fn lower(&mut self, s: Sym, lo: i64) {
        let e = self.lb.entry(s).or_insert(lo);
        *e = (*e).max(lo);
    }
    /// Fact `s <= hi` (keeps the tightest).
    pub fn upper(&mut self, s: Sym, hi: i64) {
        let e = self.ub.entry(s).or_insert(hi);
        *e = (*e).min(hi);
    }
    /// Fact `s == p` (substituted wherever `s` is read through `sym_poly`).
    pub fn define(&mut self, s: Sym, p: Poly) {
        self.subst.insert(s, p);
    }
    /// Fact `s == k`.
    pub fn constant(&mut self, s: Sym, k: i64) {
        self.konst.insert(s, k);
    }
}
