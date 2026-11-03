//! GPU-semantics verifier (v1): bounds / lifetime / legality proofs over the DSL IR and the final MIR. Design: `docs/gpu-semantics-verifier.md`.
//!
//! * `verify_ir`  : LDS budget, global + buffer bounds (symbolic interval analysis with the kernel's declared `Contract`), role-barrier balance.
//! * `verify_mir` : register-class legality (limits, tuple / descriptor alignment, MFMA contiguity) and path-sensitive wait / lifetime checks.
//! * `enforce`    : the `AMDC_GPU_VERIFY=off|warn|strict` policy wired into `compile_with`.
//!
//! Soundness statement: every `Error` finding is a property that could NOT be proven; a kernel without errors satisfies, under the host contract
//! (declared extents / ranges / grid relations, trusted `assume`s, no i32 wrap-around), the checked properties. Trusted assumptions are listed as `Info`.

use std::collections::{BTreeMap, HashMap, HashSet};

use amdgpu_dsl::{Ext, Function, Op, Opnd, Stmt, ValueId, VarId};

use crate::mir::{MFunction, MOp, RClass, Src};
use crate::regalloc::Alloc;

// --------------------------------------------------------------------------------------------------------------------------------------------
// findings

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warn,
    Info,
}

/// The counted trust LEDGER of a report: every escape hatch the certificate relies on, by kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ledger {
    /// `assume_trusted` loop-counter ranges + unproven value assumes
    pub assume_trusted: usize,
    /// `tile_trust_periodic`
    pub tile_trust: usize,
    /// `buf_trust` (region trust)
    pub region_trust: usize,
}

impl Ledger {
    pub fn total(&self) -> usize {
        self.assume_trusted + self.tile_trust + self.region_trust
    }
}

#[derive(Clone, Debug)]
pub struct Finding {
    pub code: &'static str,
    pub severity: Severity,
    pub msg: String,
    /// The v4 protocol-model name of this verdict (GV-PROTO-*; empty when the finding is not a protocol obligation).
    pub alias: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub findings: Vec<Finding>,
    /// Number of accesses proven in bounds, by space.
    pub proven: BTreeMap<&'static str, usize>,
    /// > 0 while the analyzer iterates a loop to a fixpoint: nothing is recorded.
    quiet: u32,
}

impl Report {
    pub(crate) fn add(&mut self, code: &'static str, severity: Severity, msg: String) {
        if self.quiet == 0 {
            let alias = crate::gpu_protocol::proto_alias(code, &msg);
            self.findings.push(Finding { code, severity, msg, alias });
        }
    }
    pub(crate) fn ok(&mut self, what: &'static str) {
        if self.quiet == 0 {
            *self.proven.entry(what).or_insert(0) += 1;
        }
    }
    pub fn errors(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.severity == Severity::Error).collect()
    }
    pub fn merge(&mut self, o: Report) {
        self.findings.extend(o.findings);
        for (k, v) in o.proven {
            *self.proven.entry(k).or_insert(0) += v;
        }
    }
    /// The trust ledger: how many escapes (by kind) this report relies on.
    pub fn ledger(&self) -> Ledger {
        let mut l = Ledger::default();
        for f in &self.findings {
            if f.code != "GV-ASSUME" || !f.msg.contains("TRUSTED") || f.msg.contains("set_var") {
                continue;
            }
            if f.msg.contains("tile_trust_periodic") {
                l.tile_trust += 1;
            } else if f.msg.contains("buf_trust") {
                l.region_trust += 1;
            } else {
                l.assume_trusted += 1;
            }
        }
        l
    }

    pub fn render(&self) -> String {
        let mut s = String::new();
        for f in &self.findings {
            if f.alias.is_empty() || f.alias == f.code {
                s.push_str(&format!("[{}] {:?}: {}\n", f.code, f.severity, f.msg));
            } else {
                s.push_str(&format!("[{}|{}] {:?}: {}\n", f.code, f.alias, f.severity, f.msg));
            }
        }
        s.push_str(&format!("proven accesses: {:?}\n", self.proven));
        s
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// exact rationals and symbolic polynomials

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Q {
    n: i128,
    d: i128,
}

fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Q {
    fn new(n: i128, d: i128) -> Q {
        let g = gcd(n, d).max(1);
        let s = if d < 0 { -1 } else { 1 };
        Q { n: s * n / g, d: s * d / g }
    }
    fn int(n: i128) -> Q {
        Q { n, d: 1 }
    }
    fn add(self, o: Q) -> Q {
        Q::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }
    fn mul(self, o: Q) -> Q {
        Q::new(self.n * o.n, self.d * o.d)
    }
    fn is_zero(self) -> bool {
        self.n == 0
    }
    fn is_neg(self) -> bool {
        self.n < 0
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
enum Sym {
    Ka(u32),
    Grid(u8),
    Named(String),
}

impl Sym {
    fn name(&self) -> String {
        match self {
            Sym::Ka(o) => format!("ka(0x{o:X})"),
            Sym::Grid(a) => ["grid_x", "grid_y", "grid_z"][(*a as usize).min(2)].to_string(),
            Sym::Named(n) => n.clone(),
        }
    }
}

/// A polynomial over the contract symbols with rational coefficients; monomials are sorted multisets of symbols.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
struct Poly(BTreeMap<Vec<Sym>, Q>);

impl Poly {
    fn konst(q: Q) -> Poly {
        let mut m = BTreeMap::new();
        if !q.is_zero() {
            m.insert(Vec::new(), q);
        }
        Poly(m)
    }
    fn int(n: i128) -> Poly {
        Poly::konst(Q::int(n))
    }
    fn sym(s: Sym) -> Poly {
        let mut m = BTreeMap::new();
        m.insert(vec![s], Q::int(1));
        Poly(m)
    }
    fn add(&self, o: &Poly) -> Poly {
        let mut m = self.0.clone();
        for (k, v) in &o.0 {
            let e = m.entry(k.clone()).or_insert(Q::int(0));
            *e = e.add(*v);
        }
        m.retain(|_, v| !v.is_zero());
        Poly(m)
    }
    fn scale(&self, c: Q) -> Poly {
        let mut m = BTreeMap::new();
        for (k, v) in &self.0 {
            let x = v.mul(c);
            if !x.is_zero() {
                m.insert(k.clone(), x);
            }
        }
        Poly(m)
    }
    fn sub(&self, o: &Poly) -> Poly {
        self.add(&o.scale(Q::int(-1)))
    }
    fn mul(&self, o: &Poly) -> Option<Poly> {
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
    fn floor_const(&self) -> Option<i128> {
        self.as_const().map(|q| q.n.div_euclid(q.d))
    }
    fn as_const(&self) -> Option<Q> {
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
    fn show(&self) -> String {
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

/// Symbol bounds and substitutions from the contract.
struct Ctx {
    lb: HashMap<Sym, i64>,
    konst: HashMap<Sym, i64>,
    subst: HashMap<Sym, Poly>,
    /// declared upper bounds (`ka_range` / `ka_set` / `sym_range` maxima; absent = unbounded)
    ub: HashMap<Sym, i64>,
}

impl Ctx {
    fn new(f: &Function) -> Ctx {
        let mut c = Ctx { lb: HashMap::new(), konst: HashMap::new(), subst: HashMap::new(), ub: HashMap::new() };
        for (off, lo, hi) in &f.contract.ka_ranges {
            let s = Sym::Ka(*off);
            let e = c.lb.entry(s.clone()).or_insert(0);
            *e = (*e).max(*lo);
            if *hi < i64::MAX {
                let u = c.ub.entry(s.clone()).or_insert(*hi);
                *u = (*u).min(*hi);
            }
            if lo == hi {
                c.konst.insert(s, *lo);
            }
        }
        for (off, vals) in &f.contract.ka_sets {
            if let (Some(lo), Some(hi)) = (vals.iter().min(), vals.iter().max()) {
                let s = Sym::Ka(*off);
                let e = c.lb.entry(s.clone()).or_insert(0);
                *e = (*e).max(*lo);
                let u = c.ub.entry(s.clone()).or_insert(*hi);
                *u = (*u).min(*hi);
                if lo == hi {
                    c.konst.insert(s, *lo);
                }
            }
        }
        for (n, lo, hi) in &f.contract.sym_ranges {
            let s = Sym::Named(n.clone());
            c.lb.insert(s.clone(), *lo);
            if *hi < i64::MAX {
                c.ub.insert(s.clone(), *hi);
            }
            if lo == hi {
                c.konst.insert(s, *lo);
            }
        }
        // kernarg definitions: substituted to a fixpoint, so the result does not depend on the declaration order (an acyclic set of n definitions is fully
        // substituted after n passes; for definitions already in dependency order every pass after the first reproduces the same polynomials)
        for _ in 0..f.contract.ka_defs.len().max(1) {
            for (off, e) in &f.contract.ka_defs {
                if let Some(p) = c.ext_poly(e) {
                    c.subst.insert(Sym::Ka(*off), p);
                }
            }
        }
        // grid definitions: substitute (the right-hand side is converted with the substitutions known so far)
        for (axis, e) in &f.contract.grid_defs {
            if let Some(p) = c.ext_poly(e) {
                c.subst.insert(Sym::Grid(*axis), p);
            }
        }
        c
    }
    fn sym_poly(&self, s: Sym) -> Poly {
        if let Some(k) = self.konst.get(&s) {
            return Poly::int(*k as i128);
        }
        if let Some(p) = self.subst.get(&s) {
            return p.clone();
        }
        Poly::sym(s)
    }
    fn lower_bound(&self, s: &Sym) -> i128 {
        match self.lb.get(s) {
            Some(v) => *v as i128,
            None => match s {
                Sym::Grid(_) | Sym::Named(_) => 1,
                Sym::Ka(_) => 0,
            },
        }
    }
    /// Exact symbolic value of an `Ext` (None for division, which is bounded elsewhere).
    fn ext_poly(&self, e: &Ext) -> Option<Poly> {
        Some(match e {
            Ext::Const(c) => Poly::int(*c as i128),
            Ext::Ka(o) => self.sym_poly(Sym::Ka(*o)),
            Ext::Grid(a) => self.sym_poly(Sym::Grid(*a)),
            Ext::Sym(n) => self.sym_poly(Sym::Named(n.clone())),
            Ext::Add(a, b) => self.ext_poly(a)?.add(&self.ext_poly(b)?),
            Ext::Sub(a, b) => self.ext_poly(a)?.sub(&self.ext_poly(b)?),
            Ext::Mul(a, b) => self.ext_poly(a)?.mul(&self.ext_poly(b)?)?,
            Ext::CeilDiv(..) | Ext::FloorDiv(..) => return None,
        })
    }
    /// Sufficient test for `p >= 0` for all admissible symbol values: shift every symbol s = lb(s) + t (t >= 0) and require non-negative coefficients.
    fn nonneg(&self, p: &Poly) -> bool {
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
    fn nonneg_box(&self, p: &Poly) -> bool {
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
    fn le(&self, a: &Poly, b: &Poly) -> bool {
        self.nonneg(&b.sub(a))
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// intervals: sets of candidate lower / upper bounds (empty = unbounded)

#[derive(Clone, Debug, Default, PartialEq)]
struct Itv {
    lo: Vec<Poly>,
    hi: Vec<Poly>,
}

const CAND: usize = 4;

fn cap(mut v: Vec<Poly>) -> Vec<Poly> {
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
    fn top() -> Itv {
        Itv::default()
    }
    fn point(p: Poly) -> Itv {
        Itv { lo: vec![p.clone()], hi: vec![p] }
    }
    fn range(lo: Poly, hi: Poly) -> Itv {
        Itv { lo: vec![lo], hi: vec![hi] }
    }
    fn konst(n: i128) -> Itv {
        Itv::point(Poly::int(n))
    }
    fn point_const(&self) -> Option<Q> {
        if self.lo.len() == 1 && self.hi.len() == 1 && self.lo[0] == self.hi[0] {
            return self.lo[0].as_const();
        }
        None
    }
    fn add(&self, o: &Itv) -> Itv {
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
    fn add_const(&self, c: i128) -> Itv {
        self.add(&Itv::konst(c))
    }
    fn scale(&self, c: Q) -> Itv {
        if c.is_neg() {
            Itv { lo: cap(self.hi.iter().map(|p| p.scale(c)).collect()), hi: cap(self.lo.iter().map(|p| p.scale(c)).collect()) }
        } else {
            Itv { lo: cap(self.lo.iter().map(|p| p.scale(c)).collect()), hi: cap(self.hi.iter().map(|p| p.scale(c)).collect()) }
        }
    }
    fn nonneg(&self, cx: &Ctx) -> bool {
        self.lo.iter().any(|p| cx.nonneg(p))
    }
    fn mul(&self, o: &Itv, cx: &Ctx) -> Itv {
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
    fn join(&self, o: &Itv, cx: &Ctx) -> Itv {
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
    fn min(&self, o: &Itv, cx: &Ctx) -> Itv {
        let j = self.join(o, cx);
        let mut hi = self.hi.clone();
        hi.extend(o.hi.iter().cloned());
        Itv { lo: j.lo, hi: cap(hi) }
    }
    /// Intersection (assume): all bounds of both.
    fn meet(&self, o: &Itv) -> Itv {
        let mut lo = o.lo.clone();
        lo.extend(self.lo.iter().cloned());
        let mut hi = o.hi.clone();
        hi.extend(self.hi.iter().cloned());
        Itv { lo: cap(lo), hi: cap(hi) }
    }
    /// self is contained in o (every bound of o is implied).
    fn within(&self, o: &Itv, cx: &Ctx) -> bool {
        let lo_ok = o.lo.iter().all(|b| self.lo.iter().any(|a| cx.le(b, a)));
        let hi_ok = o.hi.iter().all(|b| self.hi.iter().any(|a| cx.le(a, b)));
        lo_ok && hi_ok
    }
    /// Drop dominated candidates (keep only the tightest provable bounds) so that fixpoint iteration compares canonical forms.
    fn norm(&self, cx: &Ctx) -> Itv {
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
    fn show(&self) -> String {
        let l = if self.lo.is_empty() { "-inf".to_string() } else { self.lo.iter().map(|p| p.show()).collect::<Vec<_>>().join(" | ") };
        let h = if self.hi.is_empty() { "+inf".to_string() } else { self.hi.iter().map(|p| p.show()).collect::<Vec<_>>().join(" | ") };
        format!("[{l}, {h}]")
    }
}

fn ext_itv(cx: &Ctx, e: &Ext) -> Itv {
    match e {
        Ext::Const(_) | Ext::Ka(_) | Ext::Grid(_) | Ext::Sym(_) => match cx.ext_poly(e) {
            Some(p) => Itv::point(p),
            None => Itv::top(),
        },
        Ext::Add(a, b) => ext_itv(cx, a).add(&ext_itv(cx, b)),
        Ext::Sub(a, b) => ext_itv(cx, a).add(&ext_itv(cx, b).scale(Q::int(-1))),
        Ext::Mul(a, b) => ext_itv(cx, a).mul(&ext_itv(cx, b), cx),
        Ext::CeilDiv(a, k) => {
            let x = ext_itv(cx, a);
            let k = (*k).max(1) as i128;
            Itv { lo: cap(x.lo.iter().map(|p| p.scale(Q::new(1, k))).collect()), hi: cap(x.hi.iter().map(|p| p.add(&Poly::int(k - 1)).scale(Q::new(1, k))).collect()) }
        }
        Ext::FloorDiv(a, k) => {
            let x = ext_itv(cx, a);
            let k = (*k).max(1) as i128;
            Itv { lo: cap(x.lo.iter().map(|p| p.add(&Poly::int(-(k - 1))).scale(Q::new(1, k))).collect()), hi: cap(x.hi.iter().map(|p| p.scale(Q::new(1, k))).collect()) }
        }
    }
}
// --------------------------------------------------------------------------------------------------------------------------------------------
// IR analysis: LDS budget, global / buffer bounds

#[derive(Clone, Debug)]
enum AV {
    Int(Itv),
    /// A global pointer: the kernarg it derives from (index into `f.args`) and the byte offset interval from its base.
    Ptr { arg: Option<usize>, off: Itv },
    /// A buffer descriptor: the kernarg it was built from and its `num_records` (None = the 0xFFFFFFFF "no bounds" sentinel).
    Rsrc { arg: Option<usize>, num: Option<Itv> },
    Other,
}

struct An<'a> {
    f: &'a Function,
    cx: Ctx,
    env: HashMap<ValueId, AV>,
    vars: HashMap<VarId, Itv>,
    var_init: HashMap<VarId, Itv>,
    assumed: HashMap<VarId, Itv>,
    assumed_trusted: HashMap<VarId, String>,
    tag: Option<(usize, Itv, Opnd)>,
    /// the address operand (and constant offset) of the LDS access being checked: private-region checks need it symbolically
    cur_addr: Option<(Opnd, i128)>,
    acc: HashMap<VarId, Itv>,
    defs: HashMap<ValueId, Op>,
    /// values constrained by `assume`: their interval is a contract fact, so `recompute` (select-arm refinement) must not re-derive them from their operands
    assumed_vals: std::collections::HashSet<ValueId>,
    ovr: Option<(Opnd, Itv)>,
    if_stack: Vec<(HashMap<ValueId, AV>, Option<HashMap<ValueId, AV>>)>,
    rep: Report,
    idx: usize,
    has_asm: bool,
}

fn is_int_ty(t: amdgpu_dsl::DType) -> bool {
    matches!(t, amdgpu_dsl::DType::I32 | amdgpu_dsl::DType::U32 | amdgpu_dsl::DType::I1)
}

impl<'a> An<'a> {
    fn int_of(&self, o: &Opnd) -> Itv {
        if let Some((t, i)) = &self.ovr {
            if t == o {
                return i.clone();
            }
        }
        match o {
            Opnd::I32(c) => Itv::konst(*c as i128),
            Opnd::Val(v) => match self.env.get(v) {
                Some(AV::Int(i)) => i.clone(),
                _ => Itv::top(),
            },
            Opnd::Var(v) => self.vars.get(v).cloned().unwrap_or_default(),
            Opnd::F32(_) => Itv::top(),
        }
    }
    fn av_of(&self, o: &Opnd) -> AV {
        match o {
            Opnd::Val(v) => self.env.get(v).cloned().unwrap_or(AV::Other),
            _ => AV::Other,
        }
    }
    fn arg_extent(&self, arg: usize) -> Option<&amdgpu_dsl::ArgExtent> {
        let name = &self.f.args.get(arg)?.name;
        self.f.contract.extents.iter().find(|e| &e.arg == name)
    }
    fn arg_name(&self, arg: usize) -> String {
        self.f.args.get(arg).map(|a| a.name.clone()).unwrap_or_else(|| "?".to_string())
    }

    fn nonneg(&self, i: &Itv) -> bool {
        i.nonneg(&self.cx)
    }

    fn shape(&self, o: &Opnd, depth: usize) -> String {
        match o {
            Opnd::I32(c) => format!("{c}"),
            Opnd::Var(v) => format!("var{}", v.0),
            Opnd::Val(v) => match self.defs.get(v) {
                Some(Op::ConstI32(k)) => format!("{k}"),
                Some(op) if depth > 0 => {
                    let name = format!("{op:?}");
                    let name = name.split(|c: char| c == '(' || c == ' ').next().unwrap_or("?").to_string();
                    let ops: Vec<String> = op.operands().iter().map(|x| self.shape(x, depth - 1)).collect();
                    format!("{name}({})", ops.join(", "))
                }
                _ => format!("v{v}"),
            },
            Opnd::F32(_) => "f".to_string(),
        }
    }

    /// Where does the lack of a bound come from? Follows the first unbounded operand down to a leaf.
    fn top_origin(&self, o: &Opnd, depth: usize) -> String {
        match o {
            Opnd::Var(v) => {
                let upd: Vec<String> = self.f.stmts.iter().filter_map(|s| if let Stmt::SetVar { var, value } = s { if var == v { Some(self.shape(value, 3)) } else { None } } else { None }).take(2).collect();
                let init: Vec<String> = self.f.stmts.iter().filter_map(|s| if let Stmt::BeginLoop { vars, init } = s { vars.iter().position(|x| x == v).map(|k| self.shape(&init[k], 3)) } else { None }).take(1).collect();
                format!("loop variable var{} = {} (not bounded by the fixpoint; initial {} = {}; updated by {}; add b.assume(var, lo, hi) at the top of the loop body)", v.0, self.vars.get(v).map(|i| i.show()).unwrap_or_default(), init.join(""), self.var_init.get(v).map(|i| i.show()).unwrap_or_default(), upd.join(" / "))
            }
            Opnd::Val(v) => {
                let Some(op) = self.defs.get(v) else { return format!("v{v}") };
                let name = format!("{op:?}");
                let name = name.split(|c: char| c == '(' || c == ' ').next().unwrap_or("?").to_string();
                if depth == 0 {
                    return format!("v{v} = {name} ...");
                }
                for x in op.operands() {
                    let unbounded = match &x {
                        Opnd::Val(y) => !matches!(self.env.get(y), Some(AV::Int(i)) if !i.lo.is_empty() && !i.hi.is_empty()),
                        Opnd::Var(y) => self.vars.get(y).map_or(true, |i| i.lo.is_empty() || i.hi.is_empty()),
                        _ => false,
                    };
                    if unbounded {
                        return format!("v{v} = {name} <- {}", self.top_origin(&x, depth - 1));
                    }
                }
                format!("v{v} = {name} (no interval rule for this operation)")
            }
            _ => "constant".to_string(),
        }
    }

    /// `lo >= 0 && hi + size <= limit`; returns the failing reason.
    fn prove_range(&self, addr: &Itv, size: i128, limit: &Itv) -> Result<(), String> {
        if !self.nonneg(addr) {
            return Err(format!("lower bound not provably >= 0: address {}", addr.show()));
        }
        let ok = addr.hi.iter().any(|h| limit.lo.iter().any(|l| self.cx.nonneg(&l.sub(&h.add(&Poly::int(size))))));
        if ok {
            Ok(())
        } else {
            Err(format!("upper bound not provably <= limit - {size}: address {} vs limit {}", addr.show(), limit.show()))
        }
    }

    fn check_lds_at(&mut self, what: &str, o: &Opnd, size: i128) {
        self.cur_addr = Some((*o, 0));
        let a = self.int_of(o);
        let before = self.rep.findings.len();
        self.check_lds(what, &a, size);
        if self.rep.findings.len() > before {
            let why = if a.lo.is_empty() { format!(" | unbounded because: {}", self.top_origin(o, 6)) } else { String::new() };
            let sh = self.shape(o, 5);
            if let Some(last) = self.rep.findings.last_mut() {
                last.msg.push_str(&format!("{why} | address = {sh}"));
            }
        }
    }

    /// Under a tile marker, the access must lie inside the marked tile / ring (hull over the slot interval).
    fn tile_cover(&mut self, what: &str, addr: &Itv, size: i128) {
        let Some((ti, slot, slot_o)) = self.tag.clone() else {
            // untagged: must provably stay out of every private (per-wave) region
            for t in self.f.contract.tiles.iter().filter(|t| t.private) {
                let (lo, hi) = (t.base as i128, (t.base + t.slot_size * t.slots as i64) as i128);
                let below = addr.hi.iter().any(|h| self.cx.le(&h.add(&Poly::int(size)), &Poly::int(lo)));
                let above = addr.lo.iter().any(|l| self.cx.le(&Poly::int(hi), l));
                if !(below || above) {
                    let m = format!("stmt {}: {what}: LDS access {} (+{size}) outside any tile marker may touch the private per-wave region `{}` [{lo}, {hi}): an unmarked access cannot be proven to be the owning wave", self.idx, addr.show(), t.name);
                    self.rep.add("GV-TILE-PRIVATE-ESCAPE", Severity::Error, m);
                }
            }
            return;
        };
        let t = &self.f.contract.tiles[ti];
        if t.global {
            return; // an LDS access under a global-region marker is reported by the protocol engine (GV-TILE-KIND)
        }
        if t.private {
            // the access must lie in the slice of THIS marker's wave key: addr - key*size - base in [0, size - accsize]
            let (key_atoms, key_c) = self.lin(&slot_o).unwrap_or((std::collections::BTreeMap::new(), 0));
            let ok = match self.cur_addr.clone().and_then(|(o, imm)| self.lin(&o).map(|l| (l, imm))) {
                Some(((mut atoms, c), imm)) => {
                    for (k, v) in key_atoms {
                        *atoms.entry(k).or_insert(0) -= v * t.slot_size as i128;
                    }
                    atoms.retain(|_, v| *v != 0);
                    let mut r = Itv::konst(c + imm - key_c * t.slot_size as i128 - t.base as i128);
                    for ((is_var, id), coef) in atoms {
                        let o = if is_var { Opnd::Var(amdgpu_dsl::VarId(id)) } else { Opnd::Val(id) };
                        r = r.add(&self.int_of(&o).scale(Q::int(coef)));
                    }
                    let limit = Itv::konst(t.slot_size as i128);
                    match self.prove_range(&r, size, &limit) {
                        Ok(()) => true,
                        Err(e) => {
                            let m = format!("stmt {}: {what}: LDS access is not provably inside the private slice of wave key {slot_o:?} of `{}` (slice size {}): another wave's slice may be touched: {e}", self.idx, t.name, t.slot_size);
                            self.rep.add("GV-TILE-PRIVATE-ESCAPE", Severity::Error, m);
                            false
                        }
                    }
                }
                None => {
                    let m = format!("stmt {}: {what}: the address of an access to the private region `{}` is not a linear function of the wave key {slot_o:?} (cannot prove it stays in the own-wave slice)", self.idx, t.name);
                    self.rep.add("GV-TILE-PRIVATE-ESCAPE", Severity::Error, m);
                    false
                }
            };
            if ok {
                self.rep.ok("tile-cover");
            }
            return;
        }
        let reg = slot.scale(Q::int(t.slot_size as i128)).add_const(t.base as i128);
        let lo_ok = reg.lo.iter().any(|r| addr.lo.iter().any(|a| self.cx.le(r, a)));
        let hi_ok = addr.hi.iter().any(|h| reg.hi.iter().any(|rh| self.cx.le(&h.add(&Poly::int(size)), &rh.add(&Poly::int(t.slot_size as i128)))));
        if lo_ok && hi_ok {
            self.rep.ok("tile-cover");
        } else {
            let m = format!("stmt {}: {what}: LDS access {} (+{size}) is not provably inside tile `{}` (slot interval {}, slot size {}, base {})", self.idx, addr.show(), t.name, slot.show(), t.slot_size, t.base);
            self.rep.add("GV-TILE-ESCAPE", Severity::Error, m);
        }
    }

    /// Linear form over atoms (values / loop variables that are not +, constant*, << constant of others): atom coefficients and a constant.
    fn lin(&self, o: &Opnd) -> Option<(std::collections::BTreeMap<(bool, u32), i128>, i128)> {
        use std::collections::BTreeMap;
        fn go(s: &An, o: &Opnd, depth: usize) -> Option<(BTreeMap<(bool, u32), i128>, i128)> {
            if depth > 24 {
                return None;
            }
            match o {
                Opnd::I32(c) => Some((BTreeMap::new(), *c as i128)),
                Opnd::Var(v) => Some((BTreeMap::from([((true, v.0), 1)]), 0)),
                Opnd::Val(v) => match s.defs.get(v) {
                    Some(Op::ConstI32(c)) => Some((BTreeMap::new(), *c as i128)),
                    Some(Op::AddI32(a, b)) => {
                        let (mut x, c1) = go(s, a, depth + 1)?;
                        let (y, c2) = go(s, b, depth + 1)?;
                        for (k, v) in y {
                            *x.entry(k).or_insert(0) += v;
                        }
                        x.retain(|_, v| *v != 0);
                        Some((x, c1 + c2))
                    }
                    Some(Op::MulI32(a, b)) => {
                        let (x, c1) = go(s, a, depth + 1)?;
                        let (y, c2) = go(s, b, depth + 1)?;
                        let (m, (atoms, c), k) = if x.is_empty() { (c1, (y, c2), 0) } else if y.is_empty() { (c2, (x, c1), 0) } else { return Some((BTreeMap::from([((false, *v), 1)]), 0)) };
                        let _ = k;
                        Some((atoms.into_iter().map(|(a, v)| (a, v * m)).filter(|(_, v)| *v != 0).collect(), c * m))
                    }
                    Some(Op::ShlI32(a, b)) => {
                        let (x, c1) = go(s, a, depth + 1)?;
                        let (y, c2) = go(s, b, depth + 1)?;
                        if y.is_empty() && (0..31).contains(&c2) {
                            let m = 1i128 << c2;
                            Some((x.into_iter().map(|(a, v)| (a, v * m)).collect(), c1 * m))
                        } else {
                            Some((BTreeMap::from([((false, *v), 1)]), 0))
                        }
                    }
                    _ => Some((BTreeMap::from([((false, *v), 1)]), 0)),
                },
                Opnd::F32(_) => None,
            }
        }
        go(self, o, 0)
    }

    fn check_lds(&mut self, what: &str, addr: &Itv, size: i128) {
        self.tile_cover(what, addr, size);
        self.cur_addr = None;
        let limit = Itv::konst(self.f.lds_bytes as i128);
        match self.prove_range(addr, size, &limit) {
            Ok(()) => self.rep.ok("lds"),
            Err(e) => {
                let m = format!("stmt {}: {what}: LDS access [addr, addr+{size}) not provably inside the declared {}-byte segment: {e}", self.idx, self.f.lds_bytes);
                self.rep.add("GV-LDS-OOB", Severity::Error, m)
            }
        }
    }

    fn check_global(&mut self, what: &str, base: &AV, imm: i128, size: i128, store: bool) {
        let (arg, off) = match base {
            AV::Ptr { arg: Some(a), off } => (*a, off.clone()),
            _ => {
                let m = format!("stmt {}: {what}: pointer provenance unknown (not derived from a kernarg pointer by +offsets): cannot bound the access", self.idx);
                self.rep.add("GV-GLB-PROV", Severity::Error, m);
                return;
            }
        };
        let Some(ext) = self.arg_extent(arg).cloned() else {
            let m = format!("stmt {}: {what}: kernarg pointer `{}` has no declared extent (arg_tensor / arg_tensor_slack)", self.idx, self.arg_name(arg));
            self.rep.add("GV-GLB-NOEXT", Severity::Error, m);
            return;
        };
        // under a global-region marker the access must also lie inside the declared region of THAT argument
        if let Some((ti, _, _)) = self.tag.clone() {
            let tname = self.f.contract.tiles[ti].name.clone();
            if self.f.contract.tiles[ti].global {
                if let Some(br) = self.f.contract.buf_regions.iter().find(|r| r.name == tname).cloned() {
                    if br.arg != self.arg_name(arg) {
                        let m = format!("stmt {}: {what}: the access goes through `{}` but the marked global region `{}` belongs to `{}`", self.idx, self.arg_name(arg), tname, br.arg);
                        self.rep.add("GV-TILE-ESCAPE", Severity::Error, m);
                    } else {
                        let (bi, si) = (ext_itv(&self.cx, &br.base), ext_itv(&self.cx, &br.size));
                        let a = off.add_const(imm);
                        let lo_ok = bi.hi.iter().any(|bh| a.lo.iter().any(|al| self.cx.le(bh, al)));
                        let hi_ok = a.hi.iter().any(|ah| bi.lo.iter().any(|bl| si.lo.iter().any(|sl| self.cx.le(&ah.add(&Poly::int(size)), &bl.add(sl)))));
                        if lo_ok && hi_ok {
                            self.rep.ok("region-cover");
                        } else {
                            let m = format!("stmt {}: {what}: access {} (+{size}) is not provably inside the global region `{}` [{}, +{}) of `{}`", self.idx, a.show(), tname, bi.show(), si.show(), br.arg);
                            self.rep.add("GV-TILE-ESCAPE", Severity::Error, m);
                        }
                    }
                }
            }
        }
        let lim_e = if store { ext.logical.clone().unwrap_or(ext.alloc.clone()) } else { ext.alloc.clone() };
        let limit = ext_itv(&self.cx, &lim_e);
        let addr = off.add_const(imm);
        match self.prove_range(&addr, size, &limit) {
            Ok(()) => self.rep.ok("global"),
            Err(e) => {
                let m = format!("stmt {}: {what}: {} of {size} bytes through `{}` not provably inside its {} extent: {e}", self.idx, if store { "store" } else { "load" }, self.arg_name(arg), if store && ext.logical.is_some() { "logical" } else { "allocated" });
                self.rep.add("GV-GLB-OOB", Severity::Error, m)
            }
        }
    }

    fn check_buffer(&mut self, what: &str, rsrc: &AV, voff: &Option<Opnd>, soff: &Opnd, imm: i128, size: i128, store: bool) {
        let AV::Rsrc { arg, num } = rsrc else {
            let m = format!("stmt {}: {what}: descriptor operand is not a known buffer descriptor", self.idx);
            self.rep.add("GV-GLB-PROV", Severity::Error, m);
            return;
        };
        if num.is_some() {
            // hardware range check against num_records (loads read 0, stores are dropped): accepted; overstatement is checked at the descriptor
            self.rep.ok("buffer(hw-guarded)");
            return;
        }
        let v = match voff {
            Some(o) => self.int_of(o),
            None => Itv::konst(0),
        };
        let off = v.add(&self.int_of(soff));
        let base = AV::Ptr { arg: *arg, off };
        self.check_global(what, &base, imm, size, store);
    }

    /// POSSIBLE BITS of an i32 value seen as a 32-bit pattern (a superset of the bits any execution can set; `u32::MAX` = unknown). Structural on the
    /// value's definition (constants, and / xor / shifts / add / mul / select / min) intersected with the interval bound (a non-negative value <= h has
    /// no bit above h's). Sound for wrap-around: shifts truncate, sums and products that may exceed 32 bits give `u32::MAX`.
    fn bits(&self, o: &Opnd, depth: usize) -> u32 {
        let all = u32::MAX;
        let from_itv = |i: &Itv| -> u32 {
            if !i.nonneg(&self.cx) {
                return all;
            }
            match i.hi.iter().filter_map(|p| p.floor_const()).min() {
                Some(h) if (0..=i32::MAX as i128).contains(&h) => {
                    let h = h as u64;
                    ((h + 1).next_power_of_two() - 1) as u32
                }
                _ => all,
            }
        };
        let itv_m = from_itv(&self.int_of(o));
        let v = match o {
            Opnd::I32(c) => return *c as u32,
            Opnd::Val(v) => *v,
            _ => return itv_m,
        };
        if depth == 0 || self.assumed_vals.contains(&v) {
            return itv_m;
        }
        let Some(op) = self.defs.get(&v) else { return itv_m };
        let grow = |s: u64| -> u32 { if s > u32::MAX as u64 { all } else { ((s + 1).next_power_of_two() - 1).min(u32::MAX as u64) as u32 } };
        let k = |o: &Opnd| -> Option<u32> {
            match o {
                Opnd::I32(c) => Some(*c as u32),
                Opnd::Val(x) => match self.defs.get(x) {
                    Some(Op::ConstI32(c)) => Some(*c as u32),
                    _ => None,
                },
                _ => None,
            }
        };
        let st = match op {
            Op::ConstI32(c) => *c as u32,
            Op::AndI32(a, b) => self.bits(a, depth - 1) & self.bits(b, depth - 1),
            Op::XorI32(a, b) => self.bits(a, depth - 1) | self.bits(b, depth - 1),
            Op::ShlI32(a, b) => match k(b) {
                Some(s) if s < 32 => self.bits(a, depth - 1).wrapping_shl(s),
                _ => all,
            },
            Op::LshrI32(a, b) => match k(b) {
                Some(s) if s < 32 => self.bits(a, depth - 1) >> s,
                _ => all,
            },
            Op::AddI32(a, b) => {
                let (ma, mb) = (self.bits(a, depth - 1), self.bits(b, depth - 1));
                if ma & mb == 0 {
                    ma | mb
                } else {
                    grow(ma as u64 + mb as u64)
                }
            }
            Op::MulI32(a, b) => grow((self.bits(a, depth - 1) as u64).saturating_mul(self.bits(b, depth - 1) as u64)),
            Op::Select { a, b, .. } | Op::MinI32(a, b) => self.bits(a, depth - 1) | self.bits(b, depth - 1),
            _ => all,
        };
        st & itv_m
    }

    /// Narrows a bitwise result by its possible bits: a pattern with bit 31 clear and no bit above `m` is in `[0, m]`.
    fn narrow_bits(&self, r: AV, m: u32) -> AV {
        match r {
            AV::Int(mut i) if m >> 31 == 0 => {
                i.lo.push(Poly::int(0));
                i.hi.push(Poly::int(m as i128));
                AV::Int(i)
            }
            other => other,
        }
    }

    /// Re-evaluates an integer value under the current override (used for the arms of a `Select` whose condition refines a variable).
    fn recompute(&mut self, o: &Opnd, depth: usize) -> Itv {
        if depth == 0 {
            return self.int_of(o);
        }
        let Opnd::Val(v) = o else { return self.int_of(o) };
        if let Some((t, i)) = &self.ovr {
            if t == o {
                return i.clone();
            }
        }
        if self.assumed_vals.contains(v) {
            return self.int_of(o);
        }
        let Some(op) = self.defs.get(v).cloned() else { return self.int_of(o) };
        if !is_int_ty(op.result_ty()) || matches!(op, Op::Asm { .. }) {
            return self.int_of(o);
        }
        let mut saved: Vec<(ValueId, Option<AV>)> = Vec::new();
        for ch in op.operands() {
            if let Opnd::Val(c) = ch {
                if self.defs.get(&c).map_or(false, |d| is_int_ty(d.result_ty())) {
                    let ci = self.recompute(&ch, depth - 1);
                    saved.push((c, self.env.insert(c, AV::Int(ci))));
                }
            }
        }
        let r = match self.eval(&op) {
            AV::Int(i) => i,
            _ => Itv::top(),
        };
        for (c, old) in saved.into_iter().rev() {
            match old {
                Some(a) => {
                    self.env.insert(c, a);
                }
                None => {
                    self.env.remove(&c);
                }
            }
        }
        r
    }

    /// `Select(cond, a, b)` where `cond` compares a value/variable `x` with a bound: evaluate the arms under the refined `x`.
    fn eval_select(&mut self, ty: amdgpu_dsl::DType, cond: &Opnd, a: &Opnd, b: &Opnd) -> AV {
        if !is_int_ty(ty) {
            if let (AV::Ptr { arg: x, off: ox }, AV::Ptr { arg: y, off: oy }) = (self.av_of(a), self.av_of(b)) {
                if x == y {
                    return AV::Ptr { arg: x, off: ox.join(&oy, &self.cx) };
                }
            }
            return AV::Other;
        }
        let refine = match cond {
            Opnd::Val(c) => match self.defs.get(c) {
                Some(Op::CmpLtI32(x, k)) => {
                    // `c < v` with a constant on the left refines `v` (mode 2: v >= c + 1 / v <= c); otherwise `x < k` refines x (mode 0)
                    let xi = self.int_of(x);
                    if xi.point_const().is_some() && self.int_of(k).point_const().is_none() { Some((*k, xi, 2u8)) } else { Some((*x, self.int_of(k), 0u8)) }
                }
                Some(Op::CmpGeI32(x, k)) => Some((*x, self.int_of(k), 1u8)),
                _ => None,
            },
            _ => None,
        };
        let (ia, ib) = if let Some((x, k, mode)) = refine {
            let xi = self.int_of(&x);
            // x < k : x <= k.hi - 1 ; x >= k : x >= k.lo
            let below = Itv { lo: Vec::new(), hi: k.hi.iter().map(|p| p.add(&Poly::int(-1))).collect() };
            let above = Itv { lo: k.lo.clone(), hi: Vec::new() };
            let (t_itv, f_itv) = match mode {
                0 => (xi.meet(&below), xi.meet(&above)),
                1 => (xi.meet(&above), xi.meet(&below)),
                _ => {
                    // k is the constant c: true  => x >= c + 1 ; false => x <= c
                    let ge = Itv { lo: k.lo.iter().map(|p| p.add(&Poly::int(1))).collect(), hi: Vec::new() };
                    let le = Itv { lo: Vec::new(), hi: k.hi.clone() };
                    (xi.meet(&ge), xi.meet(&le))
                }
            };
            self.ovr = Some((x, t_itv));
            let ia = self.recompute(a, 3);
            self.ovr = Some((x, f_itv));
            let ib = self.recompute(b, 3);
            self.ovr = None;
            (ia, ib)
        } else {
            (self.int_of(a), self.int_of(b))
        };
        AV::Int(ia.join(&ib, &self.cx).norm(&self.cx))
    }

    fn eval(&mut self, op: &Op) -> AV {
        if let Op::Select { ty, cond, a, b } = op {
            return self.eval_select(*ty, cond, a, b);
        }
        // bitwise ops: the interval rules below, narrowed by the possible bits of the operands (xor / and of bounded patterns, masks of any sign)
        if let Op::XorI32(a, b) | Op::AndI32(a, b) = op {
            let (ma, mb) = (self.bits(a, 6), self.bits(b, 6));
            let m = if matches!(op, Op::XorI32(..)) { ma | mb } else { ma & mb };
            let r = self.eval_core(op);
            return self.narrow_bits(r, m);
        }
        self.eval_core(op)
    }

    fn eval_core(&mut self, op: &Op) -> AV {
        let cx = &self.cx;
        match op {
            Op::ConstI32(c) => AV::Int(Itv::konst(*c as i128)),
            Op::KernargI32 { offset } => AV::Int(Itv::point(cx.sym_poly(Sym::Ka(*offset)))),
            Op::KernargPtr { offset } => {
                let arg = self.f.args.iter().position(|a| a.kernarg_offset == *offset);
                AV::Ptr { arg, off: Itv::konst(0) }
            }
            Op::WorkitemId => AV::Int(Itv::range(Poly::int(0), Poly::int(self.f.workgroup_size as i128 - 1))),
            Op::WorkgroupIdX | Op::WorkgroupIdY | Op::WorkgroupIdZ => {
                let axis = match op {
                    Op::WorkgroupIdX => 0u8,
                    Op::WorkgroupIdY => 1,
                    _ => 2,
                };
                AV::Int(Itv::range(Poly::int(0), cx.sym_poly(Sym::Grid(axis)).sub(&Poly::int(1))))
            }
            Op::AddI32(a, b) => AV::Int(self.int_of(a).add(&self.int_of(b))),
            Op::MulI32(a, b) => AV::Int(self.int_of(a).mul(&self.int_of(b), cx)),
            Op::ShlI32(a, b) => match self.int_of(b).point_const() {
                Some(k) if k.d == 1 && (0..31).contains(&k.n) => AV::Int(self.int_of(a).scale(Q::int(1i128 << k.n))),
                _ => AV::Int(Itv::top()),
            },
            Op::LshrI32(a, b) => match self.int_of(b).point_const() {
                Some(k) if k.d == 1 && (0..31).contains(&k.n) => {
                    let x = self.int_of(a);
                    if !x.nonneg(cx) {
                        return AV::Int(Itv::top());
                    }
                    let div = Q::new(1, 1i128 << k.n);
                    let mut lo: Vec<Poly> = vec![Poly::int(0)];
                    lo.extend(x.lo.iter().map(|p| p.add(&Poly::int(-((1i128 << k.n) - 1))).scale(div)));
                    AV::Int(Itv { lo: cap(lo), hi: cap(x.hi.iter().map(|p| { let q = p.scale(div); match q.floor_const() { Some(k) => Poly::int(k), None => q } }).collect()) })
                }
                _ => AV::Int(Itv::top()),
            },
            Op::AndI32(a, b) => {
                let (x, y) = (self.int_of(a), self.int_of(b));
                let mut hi: Vec<Poly> = Vec::new();
                let mut ok = false;
                for (p, q) in [(&x, &y), (&y, &x)] {
                    if let Some(m) = p.point_const() {
                        if m.d == 1 && m.n >= 0 {
                            hi.push(Poly::int(m.n));
                            ok = true;
                            if q.nonneg(cx) {
                                hi.extend(q.hi.iter().cloned());
                            }
                        }
                    }
                }
                if !ok && x.nonneg(cx) && y.nonneg(cx) {
                    hi.extend(x.hi.iter().cloned());
                    hi.extend(y.hi.iter().cloned());
                    ok = true;
                }
                if ok {
                    AV::Int(Itv { lo: vec![Poly::int(0)], hi: cap(hi) })
                } else {
                    AV::Int(Itv::top())
                }
            }
            Op::XorI32(a, b) => {
                let (x, y) = (self.int_of(a), self.int_of(b));
                let num = |i: &Itv| -> Option<i128> {
                    if !i.nonneg(cx) || i.hi.is_empty() {
                        return None;
                    }
                    i.hi.iter().filter_map(|p| p.floor_const()).min()
                };
                // x ^ c for a constant c (any sign): the bits above the width of x are those of c, the low bits are free
                let with_const = |p: &Itv, q: &Itv| -> Option<Itv> {
                    let c = p.point_const()?;
                    if c.d != 1 {
                        return None;
                    }
                    let m = num(q)?;
                    let mut p2 = 1i128;
                    while p2 <= m {
                        p2 <<= 1;
                    }
                    let c_hi = c.n - c.n.rem_euclid(p2);
                    Some(Itv::range(Poly::int(c_hi), Poly::int(c_hi + p2 - 1)))
                };
                if let Some(r) = with_const(&y, &x).or_else(|| with_const(&x, &y)) {
                    return AV::Int(r);
                }
                match (num(&x), num(&y)) {
                    (Some(m1), Some(m2)) => {
                        let m = m1.max(m2);
                        let mut p2 = 1i128;
                        while p2 <= m {
                            p2 <<= 1;
                        }
                        AV::Int(Itv::range(Poly::int(0), Poly::int(p2 - 1)))
                    }
                    _ => AV::Int(Itv::top()),
                }
            }
            Op::MinI32(a, b) => AV::Int(self.int_of(a).min(&self.int_of(b), cx)),
            Op::ReadFirstLane(a) => AV::Int(self.int_of(a)),
            Op::PtrAdd { base, byte_off } => match self.av_of(base) {
                AV::Ptr { arg, off } => AV::Ptr { arg, off: off.add(&self.int_of(byte_off)) },
                _ => AV::Ptr { arg: None, off: Itv::top() },
            },
            Op::LoadU16 { .. } => AV::Int(Itv::range(Poly::int(0), Poly::int(65535))),
            Op::BufferRsrc { base, num_records } => {
                let arg = match self.av_of(base) {
                    AV::Ptr { arg, off } if off.point_const().map_or(false, |q| q.is_zero()) => arg,
                    _ => None,
                };
                let num = if *num_records == 0xFFFF_FFFF { None } else { Some(Itv::konst(*num_records as i128)) };
                AV::Rsrc { arg, num }
            }
            Op::BufferRsrcDyn { base, num_records } => {
                let arg = match self.av_of(base) {
                    AV::Ptr { arg, off } if off.point_const().map_or(false, |q| q.is_zero()) => arg,
                    _ => None,
                };
                AV::Rsrc { arg, num: Some(self.int_of(num_records)) }
            }
            Op::Asm { .. } => {
                self.has_asm = true;
                AV::Other
            }
            other => {
                if is_int_ty(other.result_ty()) {
                    AV::Int(Itv::top())
                } else {
                    AV::Other
                }
            }
        }
    }

    fn global_load_size(op: &Op) -> Option<i128> {
        Some(match op {
            Op::LoadU16 { .. } => 2,
            Op::LoadF32 { .. } | Op::LoadF32Coherent { .. } => 4,
            Op::LoadGlobalB128 { .. } | Op::LoadB128Coherent { .. } => 16,
            _ => return None,
        })
    }

    fn check_op(&mut self, op: &Op) {
        if let Some(sz) = Self::global_load_size(op) {
            let (base, off) = match op {
                Op::LoadU16 { base, byte_off } | Op::LoadF32 { base, byte_off } | Op::LoadGlobalB128 { base, byte_off } | Op::LoadF32Coherent { base, byte_off } | Op::LoadB128Coherent { base, byte_off } => (base, *byte_off),
                _ => return,
            };
            let b = self.av_of(base);
            self.check_global("global load", &b, off as i128, sz, false);
            return;
        }
        match op {
            Op::AtomicAddU32 { base, byte_off, .. } => {
                let b = self.av_of(base);
                self.check_global("global atomic add", &b, *byte_off as i128, 4, true);
            }
            Op::LoadLdsB32 { addr } => self.check_lds_at("ds_read_b32", addr, 4),
            Op::LoadLdsB128 { addr } => self.check_lds_at("ds_read_b128", addr, 16),
            Op::LoadLdsB64 { addr } => self.check_lds_at("ds_read_b64", addr, 8),
            Op::LoadLdsTrB8 { addr } => self.check_lds_at("ds_read_b64_tr_b8", addr, 8),
            Op::LoadLdsTr4B8 { addrs } => {
                for ad in addrs.iter() {
                    self.check_lds_at("ds_read_b64_tr_b8 (x4)", ad, 8);
                }
            }
            Op::LoadLdsB128x2 { lo, hi } => {
                for ad in [lo, hi] {
                    self.check_lds_at("ds_read_b128 (x2)", ad, 16);
                }
            }
            Op::BufferLoad { rsrc, voff, soff, imm, dwords } => {
                let r = self.av_of(rsrc);
                self.check_buffer("buffer load", &r, voff, soff, *imm as i128, 4 * *dwords as i128, false);
            }
            Op::BufferRsrc { .. } | Op::BufferRsrcDyn { .. } => {}
            _ => {}
        }
    }

    /// A descriptor must not overstate the allocation it was built from.
    fn check_descriptor(&mut self, dst: &AV) {
        if let AV::Rsrc { arg: Some(a), num: Some(n) } = dst {
            let Some(ext) = self.arg_extent(*a).cloned() else {
                let m = format!("stmt {}: buffer descriptor over `{}` has no declared extent to check num_records against", self.idx, self.arg_name(*a));
                self.rep.add("GV-GLB-NOEXT", Severity::Error, m);
                return;
            };
            let limit = ext_itv(&self.cx, &ext.alloc);
            let ok = n.hi.iter().any(|h| limit.lo.iter().any(|l| self.cx.le(h, l)));
            if ok {
                self.rep.ok("descriptor");
            } else {
                let m = format!("stmt {}: buffer descriptor over `{}` declares num_records {} which is not provably <= the allocation {} (an overstated descriptor turns out-of-bounds reads into wrong data instead of zeros)", self.idx, self.arg_name(*a), n.show(), limit.show());
                self.rep.add("GV-BUF-OVER", Severity::Error, m);
            }
        } else if let AV::Rsrc { arg: None, .. } = dst {
            let m = format!("stmt {}: buffer descriptor not built from a plain kernarg pointer (provenance unknown)", self.idx);
            self.rep.add("GV-GLB-PROV", Severity::Error, m);
        }
    }

    fn assume(&mut self, target: &Opnd, lo: &Ext, hi: &Ext, trusted: &Option<String>) {
        let a = Itv { lo: ext_itv(&self.cx, lo).lo, hi: ext_itv(&self.cx, hi).hi };
        match target {
            Opnd::Var(v) => {
                let init_ok = match self.var_init.get(v) {
                    Some(i) => i.within(&a, &self.cx),
                    None => false,
                };
                self.assumed.insert(*v, a.clone());
                self.vars.insert(*v, a.clone());
                if let Some(r) = trusted {
                    self.assumed_trusted.insert(*v, r.clone());
                    let m = format!("stmt {}: assume_trusted var{} in {} ({r}): TRUSTED, the induction is not checked", self.idx, v.0, a.show());
                    self.rep.add("GV-ASSUME", Severity::Info, m);
                } else if init_ok {
                    let m = format!("stmt {}: assume var{} in {}: initial value proven inside; every set_var is checked against it (an unproven step is an error)", self.idx, v.0, a.show());
                    self.rep.add("GV-ASSUME", Severity::Info, m);
                } else {
                    let init = self.var_init.get(v).map(|i| i.show()).unwrap_or_default();
                    let m = format!("stmt {}: assume var{} in {}: the initial value {init} is not provably inside the assumed range (use assume_trusted(.., reason) if it holds by a guard the analyzer cannot see)", self.idx, v.0, a.show());
                    self.rep.add("GV-ASSUME-VIOLATED", Severity::Error, m);
                }
            }
            Opnd::Val(v) => {
                let cur = self.int_of(target);
                let proved = cur.within(&a, &self.cx);
                let refined = cur.meet(&a);
                self.env.insert(*v, AV::Int(refined));
                self.assumed_vals.insert(*v);
                let m = format!("stmt {}: assume v{v} in {}: {}", self.idx, a.show(), if proved { "already implied by the analysis (redundant)" } else { "TRUSTED" });
                self.rep.add("GV-ASSUME", Severity::Info, m);
            }
            _ => {}
        }
    }

    fn matching_end(&self, begin: usize) -> usize {
        let mut depth = 0usize;
        for (k, s) in self.f.stmts.iter().enumerate().skip(begin) {
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
        self.f.stmts.len()
    }

    fn run(&mut self) {
        let n = self.f.stmts.len();
        self.exec(0, n);
    }

    fn exec(&mut self, start: usize, end: usize) {
        let f = self.f;
        let mut i = start;
        while i < end {
            let st = &f.stmts[i];
            self.idx = i;
            match st {
                Stmt::Let { dst, op } => {
                    self.check_op(op);
                    let v = self.eval(op);
                    if matches!(op, Op::BufferRsrc { .. } | Op::BufferRsrcDyn { .. }) {
                        self.check_descriptor(&v);
                    }
                    self.env.insert(*dst, v);
                }
                Stmt::Assume { target, lo, hi, trusted } => self.assume(target, lo, hi, trusted),
                Stmt::TileProduce { tile, slot, .. } | Stmt::TileConsume { tile, slot, .. } => {
                    self.tag = f.contract.tiles.iter().position(|t| &t.name == tile).map(|ti| (ti, self.int_of(slot), *slot));
                }
                Stmt::TileEnd | Stmt::Barrier | Stmt::EndLoop | Stmt::Else | Stmt::EndIf => self.tag = None,
                Stmt::StoreU16 { base, byte_off, .. } => {
                    let b = self.av_of(base);
                    self.check_global("store_u16", &b, *byte_off as i128, 2, true)
                }
                Stmt::StoreF32 { base, byte_off, .. } | Stmt::StoreF32Wt { base, byte_off, .. } => {
                    let b = self.av_of(base);
                    self.check_global("store_f32", &b, *byte_off as i128, 4, true)
                }
                Stmt::StoreF32x2 { base, byte_off, .. } => {
                    let b = self.av_of(base);
                    self.check_global("store_f32x2", &b, *byte_off as i128, 8, true)
                }
                Stmt::StoreF32x4 { base, byte_off, .. } | Stmt::StoreF32x4Wt { base, byte_off, .. } => {
                    let b = self.av_of(base);
                    self.check_global("store_f32x4", &b, *byte_off as i128, 16, true)
                }
                Stmt::VarLoadB128 { base, byte_off, .. } => {
                    let b = self.av_of(base);
                    self.check_global("var load_b128", &b, *byte_off as i128, 16, false)
                }
                Stmt::BufferStore { rsrc, voff, soff, imm, dwords, .. } => {
                    let r = self.av_of(rsrc);
                    self.check_buffer("buffer store", &r, voff, soff, *imm as i128, 4 * *dwords as i128, true);
                }
                Stmt::BufferLoadLds { rsrc, voff, soff, lds_addr, imm, dwords } => {
                    let r = self.av_of(rsrc);
                    self.check_buffer("buffer load ... lds (global side)", &r, voff, soff, *imm as i128, 4 * *dwords as i128, false);
                    let a = self.int_of(lds_addr).add_const(*imm as i128);
                    self.cur_addr = Some((*lds_addr, *imm as i128));
                    self.check_lds("buffer load ... lds (LDS side: M0 + imm + lane*size, 64 lanes)", &a, 64 * 4 * *dwords as i128);
                }
                Stmt::LoadGlobalLds { base, lds_addr, byte_off, dwords } => {
                    let b = self.av_of(base);
                    self.check_global("global_load_lds (global side)", &b, *byte_off as i128, 4 * *dwords as i128, false);
                    let a = self.int_of(lds_addr).add_const(*byte_off as i128);
                    self.cur_addr = Some((*lds_addr, *byte_off as i128));
                    let before = self.rep.findings.len();
                    self.check_lds("global_load_lds (LDS side: M0 + imm + lane*size, 64 lanes)", &a, 64 * 4 * *dwords as i128);
                    if self.rep.findings.len() > before {
                        let sh = self.shape(lds_addr, 5);
                        if let Some(last) = self.rep.findings.last_mut() {
                            last.msg.push_str(&format!(" | M0 = {sh} ; imm {byte_off}"));
                        }
                    }
                }
                Stmt::StoreLdsB32 { addr, .. } => self.check_lds_at("ds_write_b32", addr, 4),
                Stmt::StoreLdsB128 { addr, .. } => self.check_lds_at("ds_write_b128", addr, 16),
            Stmt::StoreLdsB64 { addr, .. } => self.check_lds_at("ds_write_b64", addr, 8),
                Stmt::BeginLoop { vars, init } => {
                    self.tag = None;
                    let close = self.matching_end(i);
                    let mut cur: Vec<Itv> = Vec::new();
                    for (v, i0) in vars.iter().zip(init.iter()) {
                        let it = self.int_of(i0);
                        self.var_init.insert(*v, it.clone());
                        self.assumed.remove(v);
                        cur.push(it.norm(&self.cx));
                    }
                    // fixpoint of the loop variables over the body (join of the initial value and every set_var value), widened to unbounded after 5 rounds
                    let mut converged = false;
                    for round in 0..6 {
                        for (v, it) in vars.iter().zip(cur.iter()) {
                            self.vars.insert(*v, it.clone());
                            self.acc.remove(v);
                        }
                        self.rep.quiet += 1;
                        self.exec(i + 1, close);
                        self.rep.quiet -= 1;
                        let mut changed = false;
                        for (k, v) in vars.iter().enumerate() {
                            // a variable restricted by `assume` keeps its assumed range (checked in the reporting pass)
                            if let Some(a) = self.assumed.get(v) {
                                if cur[k] != *a {
                                    cur[k] = a.clone();
                                    changed = true;
                                }
                                continue;
                            }
                            let joined = match self.acc.get(v) {
                                Some(x) => cur[k].join(x, &self.cx).norm(&self.cx),
                                None => cur[k].clone(),
                            };
                            if joined != cur[k] {
                                changed = true;
                                cur[k] = if round >= 4 { Itv::top() } else { joined };
                            }
                        }
                        if !changed {
                            converged = true;
                            break;
                        }
                    }
                    if !converged {
                        for (k, v) in vars.iter().enumerate() {
                            if !self.assumed.contains_key(v) {
                                cur[k] = Itv::top();
                            }
                        }
                    }
                    for (v, it) in vars.iter().zip(cur.iter()) {
                        self.vars.insert(*v, it.clone());
                        self.acc.remove(v);
                    }
                    self.exec(i + 1, close);
                    // after the loop a variable holds its initial value or a set_var value: the loop hull still bounds it
                    for (v, it) in vars.iter().zip(cur.iter()) {
                        self.vars.insert(*v, it.clone());
                    }
                    i = close + 1;
                    continue;
                }
                Stmt::SetVar { var, value } => {
                    let nv = self.int_of(value);
                    if self.rep.quiet > 0 {
                        let e = match self.acc.get(var) {
                            Some(x) => x.join(&nv, &self.cx),
                            None => nv.clone(),
                        };
                        self.acc.insert(*var, e);
                    }
                    if let Some(a) = self.assumed.get(var).cloned() {
                        if !nv.within(&a, &self.cx) {
                            if let Some(r) = self.assumed_trusted.get(var).cloned() {
                                let m = format!("stmt {i}: set_var var{} = {} is not provably inside the assumed range {} (TRUSTED: {r})", var.0, nv.show(), a.show());
                                self.rep.add("GV-ASSUME", Severity::Info, m);
                            } else {
                                let m = format!("stmt {i}: set_var var{} = {} is not provably inside the assumed range {}: the assumption is not inductive (a loop-carried assume that the loop itself can violate is a proof hole; weaken it, or use assume_trusted(.., reason) when a guard the analyzer cannot see bounds it)", var.0, nv.show(), a.show());
                                self.rep.add("GV-ASSUME-VIOLATED", Severity::Error, m);
                            }
                        }
                    }
                }
                Stmt::EndLoop => {}
                Stmt::BeginIf { .. } => {
                    self.tag = None;
                    self.if_stack.push((self.env.clone(), None))
                }
                Stmt::Else => {
                    if let Some(top) = self.if_stack.last_mut() {
                        top.1 = Some(self.env.clone());
                        self.env = top.0.clone();
                    }
                }
                Stmt::EndIf => {
                    if let Some((snap, then_env)) = self.if_stack.pop() {
                        let mut merged = then_env.unwrap_or_default();
                        for (k, v) in self.env.drain() {
                            merged.insert(k, v);
                        }
                        for (k, v) in snap {
                            merged.insert(k, v);
                        }
                        self.env = merged;
                    }
                }
                Stmt::VarAsm { .. } | Stmt::VarAsmN { .. } => self.has_asm = true,
                _ => {}
            }
            i += 1;
        }
    }
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// role / barrier protocol

/// Dynamic barrier count of a statement range as a polynomial in the loop trip counts: index d = coefficient of the product of the trip counts of d nested loops
/// (all loops at the same nesting depth are assumed to run the same number of iterations in both roles, which is how the role kernels are built).
fn bar_counts(stmts: &[Stmt], pos: &mut usize, until: &dyn Fn(&Stmt) -> bool) -> Vec<i64> {
    let mut out: Vec<i64> = vec![0];
    while *pos < stmts.len() {
        if until(&stmts[*pos]) {
            return out;
        }
        match &stmts[*pos] {
            Stmt::Barrier => {
                out[0] += 1;
                *pos += 1;
            }
            Stmt::BeginLoop { .. } => {
                *pos += 1;
                let inner = bar_counts(stmts, pos, &|s| matches!(s, Stmt::EndLoop));
                *pos += 1;
                if out.len() < inner.len() + 1 {
                    out.resize(inner.len() + 1, 0);
                }
                for (d, c) in inner.iter().enumerate() {
                    out[d + 1] += c;
                }
            }
            Stmt::BeginIf { .. } => {
                *pos += 1;
                let t = bar_counts(stmts, pos, &|s| matches!(s, Stmt::Else | Stmt::EndIf));
                if matches!(stmts.get(*pos), Some(Stmt::Else)) {
                    *pos += 1;
                    let _ = bar_counts(stmts, pos, &|s| matches!(s, Stmt::EndIf));
                }
                *pos += 1;
                // a nested if contributes its then-side (nested divergent ifs are checked on their own)
                if out.len() < t.len() {
                    out.resize(t.len(), 0);
                }
                for (d, c) in t.iter().enumerate() {
                    out[d] += c;
                }
            }
            _ => *pos += 1,
        }
    }
    out
}

fn trim(mut v: Vec<i64>) -> Vec<i64> {
    while v.len() > 1 && *v.last().unwrap() == 0 {
        v.pop();
    }
    v
}
fn barrier_balance(f: &Function, rep: &mut Report) {
    // lane/wave-dependent values: derived from the workitem id (different waves of a workgroup take different paths)
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
    // every wave-divergent if/else must have the same barrier signature on both sides
    let mut pos = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    while pos < f.stmts.len() {
        if let Stmt::BeginIf { cond } = &f.stmts[pos] {
            let divergent = tainted(cond, &tv, &tvars);
            let mut p = pos + 1;
            let t = trim(bar_counts(&f.stmts, &mut p, &|s| matches!(s, Stmt::Else | Stmt::EndIf)));
            let mut e = vec![0];
            if matches!(f.stmts.get(p), Some(Stmt::Else)) {
                p += 1;
                e = trim(bar_counts(&f.stmts, &mut p, &|s| matches!(s, Stmt::EndIf)));
            }
            if divergent && t != e {
                rep.add(
                    "GV-BAR",
                    Severity::Error,
                    format!("stmt {pos}: wave-divergent if/else (condition depends on the workitem id) executes a different number of barriers: then {t:?} vs else {e:?} (coefficients of 1, N, N*M, ... for the nested loop trip counts) - waves that run fewer barriers deadlock the workgroup"),
                );
            } else if divergent {
                rep.ok("barrier-balanced-ifs");
            }
        }
        stack.clear();
        pos += 1;
    }
}

/// IR-level verification: LDS budget, global / buffer bounds, role-barrier balance, contract sanity.
pub fn verify_ir(f: &Function) -> Report {
    let mut an = An { f, cx: Ctx::new(f), env: HashMap::new(), vars: HashMap::new(), var_init: HashMap::new(), assumed: HashMap::new(), assumed_trusted: HashMap::new(), tag: None, cur_addr: None, acc: HashMap::new(), ovr: None, assumed_vals: Default::default(), defs: f.stmts.iter().filter_map(|s| if let Stmt::Let { dst, op } = s { Some((*dst, op.clone())) } else { None }).collect(), if_stack: Vec::new(), rep: Report::default(), idx: 0, has_asm: false };
    for e in &f.contract.extents {
        if !f.args.iter().any(|a| a.name == e.arg && a.ty == amdgpu_dsl::DType::Ptr) {
            an.rep.add("GV-CONTRACT", Severity::Error, format!("contract: extent declared for `{}` which is not a pointer kernarg of `{}`", e.arg, f.name));
        }
    }
    for r in &f.contract.buf_regions {
        if !f.args.iter().any(|a| a.name == r.arg && a.ty == amdgpu_dsl::DType::Ptr) {
            an.rep.add("GV-CONTRACT", Severity::Error, format!("contract: buf_region `{}` names `{}` which is not a pointer kernarg of `{}`", r.name, r.arg, f.name));
        }
    }
    for la in &f.contract.last_arrivers {
        if !f.args.iter().any(|a| a.name == la.counter_arg && a.ty == amdgpu_dsl::DType::Ptr) {
            an.rep.add("GV-CONTRACT", Severity::Error, format!("contract: last-arriver counter `{}` is not a pointer kernarg of `{}`", la.counter_arg, f.name));
        }
    }
    an.run();
    let has_asm = an.has_asm;
    let mut rep = an.rep;
    if has_asm {
        rep.add("GV-ASM", Severity::Info, "safe-asm blocks present: their internal memory accesses are NOT bounds-checked by v1 (address inputs come from IR values that are)".to_string());
    }
    barrier_balance(f, &mut rep);
    crate::gpu_protocol::verify_tiles(f, &mut rep);
    crate::gpu_protocol::verify_last_arrivers(f, &mut rep);
    rep
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// MIR level: register legality + path-sensitive wait / lifetime

const VGPR_LIMIT: u32 = 256;
const SGPR_LIMIT: u32 = 102;
const LDS_LIMIT: u32 = 160 * 1024;

fn is_matrix_op(op: MOp) -> bool {
    matches!(
        op,
        MOp::VMfma16x16x16F16
            | MOp::VMfma32x32x8F16
            | MOp::VMfma32x32x8F16Z
            | MOp::VMfma16x16x32Fp8
            | MOp::VMfma32x32x16Fp8
            | MOp::VMfmaScale32x32x64Fp8
            | MOp::VMfmaScale16x16x128Fp8
            | MOp::VMfmaScale32x32x64Fp8Z
            | MOp::VMfma32x32x64Fp8
            | MOp::VMfma32x32x64Fp8Z
    )
}

/// Final-MIR verification: every allocated register obeys the class rules and the declared limits; matrix operands do not overlap their destination;
/// no instruction touches a register with a load in flight on ANY control-flow path (loop back edges included).
pub fn verify_mir(f: &Function, m: &MFunction, a: &Alloc) -> Report {
    let mut rep = Report::default();
    let mut used: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for b in &m.blocks {
        for i in &b.insts {
            if let Some(d) = i.dst {
                used.insert(d);
            }
            for s in &i.srcs {
                if let Src::Reg(r) = s {
                    used.insert(*r);
                }
            }
        }
    }
    for &r in &used {
        let c: RClass = m.vreg_class(r);
        let p = a.phys[r as usize];
        let n = c.dwords();
        if c.is_vector() {
            if p < 1 {
                rep.add("GV-REG-RESERVED", Severity::Error, format!("vreg {r} ({c:?}) allocated to v{p}: v0 holds the workitem id"));
            }
            // ISA (measured with the gfx950 assembler, 2025-11: every multi-dword VGPR tuple - MFMA A/B/C/D of any width, global/ds/buffer data, packed ops -
            // is accepted at EVEN start registers and rejected at odd ones: v[2:17], v[6:21], v[14:29] are legal V16 tuples). The allocator additionally aligns to the
            // tuple width: reported as a policy deviation (Info), not an error.
            let need = 2;
            if n >= 2 && p % need != 0 {
                rep.add("GV-REG-ALIGN", Severity::Error, format!("vreg {r} ({c:?}) at v{p}: a {n}-dword VGPR tuple must start at an even register (ISA, assembler-measured)"));
            } else if n >= 2 && p % n != 0 {
                rep.add("GV-REG-POLICY", Severity::Info, format!("vreg {r} ({c:?}) at v{p}: ISA-legal, but not aligned to the tuple width {n} (the allocator normally aligns to the width)"));
            }
            if p + n > VGPR_LIMIT {
                rep.add("GV-REG-LIMIT", Severity::Error, format!("vreg {r} ({c:?}) at v{p}..v{}: exceeds the {VGPR_LIMIT}-VGPR per-wave limit", p + n - 1));
            }
        } else {
            if p < 8 {
                rep.add("GV-REG-RESERVED", Severity::Error, format!("vreg {r} ({c:?}) allocated to s{p}: s0..s7 hold the kernarg pointer / workgroup ids"));
            }
            // measured: pairs at even registers; quads (buffer descriptors, s_load_dwordx4) and octets (s_load_dwordx8) at multiples of 4
            let need = if n >= 4 { 4 } else { 2 };
            if n >= 2 && p % need != 0 {
                rep.add("GV-REG-ALIGN", Severity::Error, format!("vreg {r} ({c:?}) at s{p}: an SGPR {} must start at a multiple of {need} (64-bit pairs even, 128-bit descriptors / wider multiple of 4; assembler-measured)", if n == 4 { "descriptor quad" } else { "tuple" }));
            }
            if p + n > SGPR_LIMIT {
                rep.add("GV-REG-LIMIT", Severity::Error, format!("vreg {r} ({c:?}) at s{p}..s{}: exceeds the {SGPR_LIMIT}-SGPR limit", p + n - 1));
            }
        }
    }
    if a.max_v > VGPR_LIMIT {
        rep.add("GV-REG-LIMIT", Severity::Error, format!("kernel uses {} VGPRs (limit {VGPR_LIMIT})", a.max_v));
    }
    if a.max_s > SGPR_LIMIT {
        rep.add("GV-REG-LIMIT", Severity::Error, format!("kernel uses {} SGPRs (limit {SGPR_LIMIT})", a.max_s));
    }
    if m.lds_bytes > LDS_LIMIT {
        rep.add("GV-LDS-OOB", Severity::Error, format!("declared + spill LDS {} bytes exceeds the {LDS_LIMIT}-byte per-workgroup limit", m.lds_bytes));
    }
    if m.lds_bytes < f.lds_bytes {
        rep.add("GV-LDS-OOB", Severity::Error, format!("emitted LDS segment {} bytes is smaller than the DSL-declared {}", m.lds_bytes, f.lds_bytes));
    }
    // matrix operand / destination overlap
    let range = |r: u32| -> (u32, u32) {
        let p = a.phys[r as usize];
        (p, p + m.vreg_class(r).dwords())
    };
    for (bi, b) in m.blocks.iter().enumerate() {
        for (ii, i) in b.insts.iter().enumerate() {
            if !is_matrix_op(i.op) {
                continue;
            }
            let Some(d) = i.dst else { continue };
            let rd = range(d);
            for k in 0..2usize {
                if let Some(Src::Reg(s)) = i.srcs.get(k) {
                    let rs = range(*s);
                    if rs.0 < rd.1 && rd.0 < rs.1 {
                        rep.add("GV-REG-MFMA", Severity::Error, format!("block {bi} inst {ii} ({:?}): multiplicand {} v[{}..{}) overlaps the destination v[{}..{})", i.op, ["A", "B"][k], rs.0, rs.1, rd.0, rd.1));
                    }
                }
            }
        }
    }
    if let Err(e) = crate::sched::waits::verify_waits(m) {
        rep.add("GV-WAIT", Severity::Error, format!("{e} (on at least one control-flow path, including loop back edges)"));
    } else {
        rep.ok("wait-paths");
    }
    rep
}

// --------------------------------------------------------------------------------------------------------------------------------------------
// policy

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    /// Collect and print findings, never fail.
    Warn,
    /// Any `Error` finding fails the compile.
    Strict,
    /// Default: findings are collected silently; MIR-level errors (registers, waits) fail the compile, IR-level ones do not (kernels need their contract first).
    Default,
}

pub fn mode_from_env() -> Mode {
    match std::env::var("AMDC_GPU_VERIFY").ok().as_deref() {
        Some("off") | Some("0") => Mode::Off,
        Some("warn") => Mode::Warn,
        Some("strict") | Some("1") => Mode::Strict,
        _ => Mode::Default,
    }
}

/// Full analysis of a kernel (compiles it to the final MIR first).
pub fn analyse(f: &Function) -> Result<Report, String> {
    let (m, a) = crate::compile_mir_alloc(f)?;
    let mut rep = verify_ir(f);
    rep.merge(verify_mir(f, &m, &a));
    Ok(rep)
}

/// Called by `compile_with` on the final MIR.
pub fn enforce(f: &Function, m: &MFunction, a: &Alloc) -> Result<(), String> {
    let mode = mode_from_env();
    if mode == Mode::Off {
        return Ok(());
    }
    let mut rep = verify_ir(f);
    let mir = verify_mir(f, m, a);
    let mir_errors: Vec<String> = mir.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
    rep.merge(mir);
    if mode == Mode::Warn && !rep.findings.is_empty() {
        eprintln!("gpu_verify({}):\n{}", f.name, rep.render());
    }
    match mode {
        Mode::Strict => {
            let errs: Vec<String> = rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect();
            if errs.is_empty() {
                Ok(())
            } else {
                Err(format!("gpu_verify (strict) rejected `{}`: {} error(s)\n{}", f.name, errs.len(), errs.join("\n")))
            }
        }
        Mode::Default => {
            if mir_errors.is_empty() {
                Ok(())
            } else {
                Err(format!("gpu_verify rejected `{}` (register / wait legality): {} error(s)\n{}", f.name, mir_errors.len(), mir_errors.join("\n")))
            }
        }
        _ => Ok(()),
    }
}
