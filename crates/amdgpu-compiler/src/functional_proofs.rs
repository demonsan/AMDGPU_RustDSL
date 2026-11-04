//! Item E2: real-arithmetic loop-invariant proof for the online-softmax rebase/rollback recipe.
//!
//! The proof is symbolic, zero dependencies: expressions over real-valued leaves with `exp`, and a
//! normalizer that uses only `exp(a) exp(b) = exp(a+b)`, additive cancellation and constant folding.
//! No floating-point evaluation is part of the proof; the numeric counterexample is a separate
//! executable check of the *unproved* variant.
//!
//! State of the streaming algorithm after consuming a prefix:
//!   m = max of the scores seen so far (a frame)
//!   d = sum exp(s_i - m),  o = sum exp(s_i - m) v_i
//! Invariant: o * exp(m) = sum exp(s_i) v_i and d * exp(m) = sum exp(s_i).
//! A new score x with new frame m' = max(m,x) updates by first rescaling both accumulators by
//! exp(m - m') and then adding exp(x - m') (times v). The two algebraic identities below are exactly
//! the induction step, for arbitrary frames m, m' and arbitrary previous state S.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
enum E {
    Const(f64),
    Var(&'static str),
    Add(Vec<E>),
    Neg(Box<E>),
    Mul(Vec<E>),
    Exp(Box<E>),
}

impl fmt::Display for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            E::Const(c) => write!(f, "{c}"),
            E::Var(v) => write!(f, "{v}"),
            E::Add(a) => { write!(f, "(")?; for (i, x) in a.iter().enumerate() { if i > 0 { write!(f, " + ")?; } write!(f, "{x}")?; } write!(f, ")") }
            E::Neg(x) => write!(f, "-{x}"),
            E::Mul(a) => { write!(f, "(")?; for (i, x) in a.iter().enumerate() { if i > 0 { write!(f, " * ")?; } write!(f, "{x}")?; } write!(f, ")") }
            E::Exp(x) => write!(f, "exp({x})"),
        }
    }
}

fn var(v: &'static str) -> E { E::Var(v) }
fn add(a: E, b: E) -> E { E::Add(vec![a, b]) }
fn sub(a: E, b: E) -> E { E::Add(vec![a, E::Neg(Box::new(b))]) }
fn mul(a: E, b: E) -> E { E::Mul(vec![a, b]) }
fn exp(a: E) -> E { E::Exp(Box::new(a)) }

/// Flatten sums and cancel `x + (-x)` pairs; combine constants. `Add(vec![])` -> 0.
fn norm_add(terms: Vec<E>) -> E {
    let mut flat: Vec<E> = Vec::new();
    let mut c = 0.0f64;
    fn push(t: E, flat: &mut Vec<E>, c: &mut f64) {
        match t {
            E::Const(x) => *c += x,
            E::Add(xs) => for x in xs { push(x, flat, c); },
            E::Neg(x) => match *x {
                E::Add(xs) => for x in xs { push(E::Neg(Box::new(x)), flat, c); },
                other => flat.push(E::Neg(Box::new(other))),
            },
            other => flat.push(other),
        }
    }
    for t in terms { push(t, &mut flat, &mut c); }
    // cancel exact negatives (linear terms and arbitrary expressions alike)
    let mut out: Vec<E> = Vec::new();
    'outer: for t in flat {
        for i in 0..out.len() {
            let combined = match (&out[i], &t) {
                (E::Neg(a), b) if **a == *b => Some(true),
                (a, E::Neg(b)) if *a == **b => Some(true),
                _ => None,
            };
            if combined.is_some() { out.remove(i); continue 'outer; }
        }
        out.push(t);
    }
    if c != 0.0 { out.insert(0, E::Const(c)); }
    match out.len() { 0 => E::Const(0.0), 1 => out.pop().unwrap(), _ => E::Add(out) }
}

fn norm_mul(factors: Vec<E>) -> E {
    let mut flat: Vec<E> = Vec::new();
    let mut c = 1.0f64;
    fn push(t: E, flat: &mut Vec<E>, c: &mut f64) {
        match t {
            E::Const(x) => *c *= x,
            E::Mul(xs) => for x in xs { push(x, flat, c); },
            other => flat.push(other),
        }
    }
    for t in factors { push(t, &mut flat, &mut c); }
    // exp(a) exp(b) = exp(a + b)
    let mut exps: Vec<E> = Vec::new();
    let mut rest: Vec<E> = Vec::new();
    for t in flat {
        match t {
            E::Exp(x) => exps.push(*x),
            other => rest.push(other),
        }
    }
    if !exps.is_empty() { rest.push(exp(norm_add(exps))); }
    if c != 1.0 { rest.insert(0, E::Const(c)); }
    match rest.len() { 0 => E::Const(1.0), 1 => rest.pop().unwrap(), _ => E::Mul(rest) }
}

fn norm(e: E) -> E {
    match e {
        E::Const(_) | E::Var(_) => e,
        E::Neg(x) => match norm(*x) {
            E::Const(c) => E::Const(-c),
            E::Neg(y) => *y,
            y => E::Neg(Box::new(y)),
        },
        E::Add(xs) => norm_add(xs.into_iter().map(norm).collect()),
        E::Mul(xs) => norm_mul(xs.into_iter().map(norm).collect()),
        E::Exp(x) => exp(norm(*x)),
    }
}

/// The two induction identities, proved by normalisation, for arbitrary symbols:
///   (S * exp(m - m') + exp(x - m') v) * exp(m') = S exp(m) + exp(x) v
///   (D * exp(m - m') + exp(x - m')) * exp(m') = D exp(m) + exp(x)
/// plus the rollback/rebase identity: exp(a - b) exp(b - a) = 1.
pub fn softmax_invariant_proof() -> Result<(), String> {
    let (s, d, v, m, mp, x) = (var("S_o"), var("D"), var("V"), var("m"), var("m'"), var("x"));
    let rescale = exp(sub(m.clone(), mp.clone()));
    // o' * exp(m') compared with the invariant numerator
    let lhs_o = mul(add(mul(s, rescale.clone()), mul(exp(sub(x.clone(), mp.clone())), v.clone())), exp(mp.clone()));
    let rhs_o = add(mul(var("S_o"), exp(m.clone())), mul(exp(x.clone()), v));
    let lhs_d = mul(add(mul(d, rescale.clone()), exp(sub(x.clone(), mp.clone()))), exp(mp.clone()));
    let rhs_d = add(mul(var("D"), exp(m.clone())), exp(x.clone()));
    let (l_o, r_o) = (norm(lhs_o.clone()), norm(rhs_o.clone()));
    if l_o != r_o { return Err(format!("online-softmax numerator invariant failed: {l_o} != {r_o}")); }
    let (l_d, r_d) = (norm(lhs_d.clone()), norm(rhs_d.clone()));
    if l_d != r_d { return Err(format!("online-softmax denominator invariant failed: {l_d} != {r_d}")); }
    // rebase/rollback: changing the frame from a to b and back is the identity
    let a = exp(sub(var("a"), var("b")));
    let b = exp(sub(var("b"), var("a")));
    if norm(mul(a, b)) != E::Const(1.0) { return Err("rebase/rollback frame factor is not 1".into()); }
    Ok(())
}

/// Concrete update of one prefix; `rescale` false models the broken recipe (no rescale when the max
/// increases). Returns the accumulated (sum exp(s)v, sum exp(s)).
fn online(scores: &[f64], vals: &[f64], rescale: bool) -> (f64, f64) {
    let mut m = f64::NEG_INFINITY;
    let (mut o, mut d) = (0.0, 0.0);
    for (&x, &v) in scores.iter().zip(vals) {
        let mp = m.max(x);
        let f = if mp > m {
            if rescale { (m - mp).exp() } else { 1.0 }
        } else { (m - mp).exp() };
        o *= f; d *= f;
        o += (x - mp).exp() * v;
        d += (x - mp).exp();
        m = mp;
    }
    (o, d)
}

/// Naive reference: (sum exp(s)v, sum exp(s)), with the global max factored out for stability.
pub fn naive(scores: &[f64], vals: &[f64]) -> (f64, f64) {
    let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (mut o, mut d) = (0.0, 0.0);
    for (&x, &v) in scores.iter().zip(vals) {
        let e = (x - m).exp();
        o += e * v; d += e;
    }
    (o, d)
}

/// The counterexample for the removed rescale term. Returns the scores/vals and the mismatch.
pub fn softmax_counterexample() -> (Vec<f64>, Vec<f64>, (f64, f64), (f64, f64)) {
    let scores = vec![0.0, 10.0, 2.0];
    let vals = vec![1.0, -2.0, 0.5];
    let good = online(&scores, &vals, true);
    let bad = online(&scores, &vals, false);
    (scores, vals, good, bad)
}
