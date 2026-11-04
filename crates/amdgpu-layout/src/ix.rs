//! Symbolic index expressions (`Ix`): the integers a layout computes with. A STATIC leaf is `Ix::C(k)`; a DYNAMIC leaf is `Ix::Sym(id)`, a value known only on
//! the GPU (a lane id, a kernarg, a loop counter) whose range and possible bits are recorded in a [`Dyns`] table. Arithmetic folds constants and keeps sums in a
//! canonical form (sorted terms with coefficients + one constant), so that two offsets that differ only by a constant split into the SAME dynamic part and a
//! static remainder - the basis of the immediate folding in `tv.rs`.

use std::fmt;

/// Symbolic integer expression (64-bit host arithmetic; the GPU side is i32, the tile layer checks ranges).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ix {
    C(i64),
    Sym(u32),
    /// `sum(coef * term) + k` with non-constant terms, sorted, coefficients != 0, at least one term
    Sum(Vec<(Ix, i64)>, i64),
    /// product of two non-constant expressions
    Mul(Box<Ix>, Box<Ix>),
    /// floor division by a positive constant
    Div(Box<Ix>, i64),
    /// remainder by a positive constant
    Mod(Box<Ix>, i64),
    Xor(Box<Ix>, Box<Ix>),
    /// bitwise and with a constant mask
    And(Box<Ix>, i64),
    Shl(Box<Ix>, u32),
    Shr(Box<Ix>, u32),
}

/// Range and bit support of one dynamic symbol.
#[derive(Clone, Debug)]
pub struct DynInfo {
    pub name: String,
    pub lo: i64,
    pub hi: i64,
    /// bits that may be 1 (for a non-negative symbol); derived from `hi` when not given
    pub bits: u64,
}

/// The dynamic symbols of one kernel build.
#[derive(Clone, Debug, Default)]
pub struct Dyns {
    pub syms: Vec<DynInfo>,
}

fn bits_upto(hi: i64) -> u64 {
    if hi <= 0 {
        0
    } else {
        amdgpu_sym::bits::upto(hi as u64)
    }
}

impl Dyns {
    pub fn new() -> Dyns {
        Dyns::default()
    }
    /// A symbol with values in `[lo, hi]`.
    pub fn sym(&mut self, name: &str, lo: i64, hi: i64) -> Ix {
        self.syms.push(DynInfo { name: name.to_string(), lo, hi, bits: if lo >= 0 { bits_upto(hi) } else { u64::MAX } });
        Ix::Sym((self.syms.len() - 1) as u32)
    }
    /// A symbol known to be a multiple of `align` in `[lo, hi]` (e.g. a slot offset): its low bits are zero.
    pub fn sym_aligned(&mut self, name: &str, lo: i64, hi: i64, align: i64) -> Ix {
        assert!(align > 0 && (align & (align - 1)) == 0, "alignment must be a power of two");
        let s = self.sym(name, lo, hi);
        if let Ix::Sym(i) = s {
            self.syms[i as usize].bits &= !((align as u64) - 1);
        }
        s
    }
    pub fn info(&self, id: u32) -> &DynInfo {
        &self.syms[id as usize]
    }
}

impl From<i64> for Ix {
    fn from(v: i64) -> Ix {
        Ix::C(v)
    }
}
impl From<i32> for Ix {
    fn from(v: i32) -> Ix {
        Ix::C(v as i64)
    }
}

fn sum_parts(x: &Ix) -> (Vec<(Ix, i64)>, i64) {
    match x {
        Ix::C(k) => (vec![], *k),
        Ix::Sum(t, k) => (t.clone(), *k),
        other => (vec![(other.clone(), 1)], 0),
    }
}

fn mk_sum(mut terms: Vec<(Ix, i64)>, k: i64) -> Ix {
    terms.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out: Vec<(Ix, i64)> = Vec::with_capacity(terms.len());
    for (t, c) in terms {
        match out.last_mut() {
            Some(last) if last.0 == t => last.1 += c,
            _ => out.push((t, c)),
        }
    }
    out.retain(|(_, c)| *c != 0);
    if out.is_empty() {
        Ix::C(k)
    } else if out.len() == 1 && out[0].1 == 1 && k == 0 {
        out.pop().expect("one term").0
    } else {
        Ix::Sum(out, k)
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// `x = g * x'` with g = gcd(coefficients, constant, k): returns (x', k / g).
fn reduce_gcd(x: &Ix, k: i64) -> (Ix, i64) {
    let (t, c) = sum_parts(x);
    if t.is_empty() {
        return (x.clone(), k);
    }
    let mut g = gcd(k, c);
    for (_, m) in &t {
        g = gcd(g, *m);
    }
    if g <= 1 {
        return (x.clone(), k);
    }
    (mk_sum(t.into_iter().map(|(y, m)| (y, m / g)).collect(), c / g), k / g)
}
impl Ix {
    pub fn c(&self) -> Option<i64> {
        if let Ix::C(k) = self {
            Some(*k)
        } else {
            None
        }
    }
    pub fn is_static(&self) -> bool {
        matches!(self, Ix::C(_))
    }
    /// The static value; panics for a dynamic expression (used where the algebra requires a static mode).
    pub fn st(&self) -> i64 {
        self.c().unwrap_or_else(|| panic!("layout: a static value was required, found the dynamic expression {self}"))
    }
    pub fn add(&self, o: &Ix) -> Ix {
        if let (Ix::C(a), Ix::C(b)) = (self, o) {
            return Ix::C(a + b);
        }
        let (mut t, k) = sum_parts(self);
        let (u, l) = sum_parts(o);
        t.extend(u);
        mk_sum(t, k + l)
    }
    pub fn sub(&self, o: &Ix) -> Ix {
        self.add(&o.mul_c(-1))
    }
    pub fn add_c(&self, k: i64) -> Ix {
        self.add(&Ix::C(k))
    }
    pub fn mul_c(&self, k: i64) -> Ix {
        match self {
            _ if k == 0 => Ix::C(0),
            _ if k == 1 => self.clone(),
            Ix::C(a) => Ix::C(a * k),
            Ix::Sum(t, c) => mk_sum(t.iter().map(|(x, m)| (x.clone(), m * k)).collect(), c * k),
            other => mk_sum(vec![(other.clone(), k)], 0),
        }
    }
    pub fn mul(&self, o: &Ix) -> Ix {
        match (self, o) {
            (Ix::C(a), _) => o.mul_c(*a),
            (_, Ix::C(b)) => self.mul_c(*b),
            _ => {
                let (a, b) = if self <= o { (self, o) } else { (o, self) };
                Ix::Mul(Box::new(a.clone()), Box::new(b.clone()))
            }
        }
    }
    /// floor(self / k), simplified with the symbol ranges.
    pub fn div_c(&self, k: i64, d: &Dyns) -> Ix {
        assert!(k > 0, "division by a non-positive constant");
        if k == 1 {
            return self.clone();
        }
        if let Ix::C(a) = self {
            return Ix::C(a.div_euclid(k));
        }
        // split the multiples of k out of a sum: sum(c_i x_i) + c = k * (q) + r with the remainder part r; if r is provably in [0, k), floor(.../k) = q
        let (t, c) = sum_parts(self);
        let (mut q, mut r): (Vec<(Ix, i64)>, Vec<(Ix, i64)>) = (vec![], vec![]);
        for (x, m) in t {
            if m % k == 0 {
                q.push((x, m / k));
            } else {
                r.push((x, m));
            }
        }
        let rest = mk_sum(r, c.rem_euclid(k));
        let qc = c.div_euclid(k);
        let (lo, hi) = rest.range(d);
        if lo >= 0 && hi < k {
            return mk_sum(q, qc);
        }
        if q.is_empty() && qc == 0 {
            if let Some((lo, hi)) = Some(self.range(d)) {
                if lo >= 0 && hi < k {
                    return Ix::C(0);
                }
            }
            let (r2, k2) = reduce_gcd(self, k);
            if k2 == 1 {
                return r2;
            }
            let (lo2, hi2) = r2.range(d);
            if lo2 >= 0 && hi2 < k2 {
                return Ix::C(0);
            }
            return Ix::Div(Box::new(r2), k2);
        }
        let (rest, k2) = reduce_gcd(&rest, k);
        if k2 == 1 {
            return mk_sum(q, qc).add(&rest);
        }
        let (lo2, hi2) = rest.range(d);
        if lo2 >= 0 && hi2 < k2 {
            return mk_sum(q, qc);
        }
        mk_sum(q, qc).add(&Ix::Div(Box::new(rest), k2))
    }
    /// self mod k (self >= 0), simplified with the symbol ranges.
    pub fn mod_c(&self, k: i64, d: &Dyns) -> Ix {
        assert!(k > 0, "remainder by a non-positive constant");
        if k == 1 {
            return Ix::C(0);
        }
        if let Ix::C(a) = self {
            return Ix::C(a.rem_euclid(k));
        }
        let (t, c) = sum_parts(self);
        let r: Vec<(Ix, i64)> = t.into_iter().filter(|(_, m)| m % k != 0).collect();
        let rest = mk_sum(r, c.rem_euclid(k));
        let (lo, hi) = rest.range(d);
        if lo >= 0 && hi < k {
            return rest;
        }
        if let Ix::Mod(x, m) = &rest {
            if k % m == 0 {
                return rest.clone();
            }
            let _ = x;
        }
        // g | every coefficient, the constant and k: (g x) mod k = g (x mod k/g)
        let (r2, k2) = reduce_gcd(&rest, k);
        let g = k / k2;
        if g > 1 {
            return r2.mod_c(k2, d).mul_c(g);
        }
        Ix::Mod(Box::new(rest), k)
    }
    pub fn xor(&self, o: &Ix) -> Ix {
        match (self, o) {
            (Ix::C(a), Ix::C(b)) => Ix::C(a ^ b),
            (_, Ix::C(0)) => self.clone(),
            (Ix::C(0), _) => o.clone(),
            _ => {
                let (a, b) = if self <= o { (self, o) } else { (o, self) };
                Ix::Xor(Box::new(a.clone()), Box::new(b.clone()))
            }
        }
    }
    pub fn and_c(&self, m: i64) -> Ix {
        match self {
            Ix::C(a) => Ix::C(a & m),
            _ if m == 0 => Ix::C(0),
            _ => Ix::And(Box::new(self.clone()), m),
        }
    }
    pub fn shl(&self, s: u32) -> Ix {
        match self {
            Ix::C(a) => Ix::C(a << s),
            _ if s == 0 => self.clone(),
            _ => self.mul_c(1i64 << s),
        }
    }
    pub fn shr(&self, s: u32) -> Ix {
        match self {
            Ix::C(a) => Ix::C(a >> s),
            _ if s == 0 => self.clone(),
            _ => Ix::Shr(Box::new(self.clone()), s),
        }
    }
    /// (dynamic part without its constant, constant).
    pub fn split_const(&self) -> (Ix, i64) {
        match self {
            Ix::C(k) => (Ix::C(0), *k),
            Ix::Sum(t, k) => (mk_sum(t.clone(), 0), *k),
            other => (other.clone(), 0),
        }
    }
    /// Replaces symbol `s` by `v` (re-simplifying with the ranges of `d`).
    pub fn subst(&self, s: u32, v: &Ix, d: &Dyns) -> Ix {
        match self {
            Ix::C(_) => self.clone(),
            Ix::Sym(x) => {
                if *x == s {
                    v.clone()
                } else {
                    self.clone()
                }
            }
            Ix::Sum(t, k) => t.iter().fold(Ix::C(*k), |acc, (x, m)| acc.add(&x.subst(s, v, d).mul_c(*m))),
            Ix::Mul(a, b) => a.subst(s, v, d).mul(&b.subst(s, v, d)),
            Ix::Div(a, k) => a.subst(s, v, d).div_c(*k, d),
            Ix::Mod(a, k) => a.subst(s, v, d).mod_c(*k, d),
            Ix::Xor(a, b) => a.subst(s, v, d).xor(&b.subst(s, v, d)),
            Ix::And(a, m) => a.subst(s, v, d).and_c(*m),
            Ix::Shl(a, k) => a.subst(s, v, d).shl(*k),
            Ix::Shr(a, k) => a.subst(s, v, d).shr(*k),
        }
    }
    /// Evaluates with concrete symbol values.
    pub fn eval(&self, env: &dyn Fn(u32) -> i64) -> i64 {
        match self {
            Ix::C(k) => *k,
            Ix::Sym(s) => env(*s),
            Ix::Sum(t, k) => t.iter().map(|(x, m)| m * x.eval(env)).sum::<i64>() + k,
            Ix::Mul(a, b) => a.eval(env) * b.eval(env),
            Ix::Div(a, k) => a.eval(env).div_euclid(*k),
            Ix::Mod(a, k) => a.eval(env).rem_euclid(*k),
            Ix::Xor(a, b) => a.eval(env) ^ b.eval(env),
            Ix::And(a, m) => a.eval(env) & m,
            Ix::Shl(a, s) => a.eval(env) << s,
            Ix::Shr(a, s) => a.eval(env) >> s,
        }
    }
    /// Conservative value range `[lo, hi]`.
    pub fn range(&self, d: &Dyns) -> (i64, i64) {
        match self {
            Ix::C(k) => (*k, *k),
            Ix::Sym(s) => (d.info(*s).lo, d.info(*s).hi),
            Ix::Sum(t, k) => {
                let (mut lo, mut hi) = (*k, *k);
                for (x, m) in t {
                    let (a, b) = x.range(d);
                    let (p, q) = (a * m, b * m);
                    lo += p.min(q);
                    hi += p.max(q);
                }
                (lo, hi)
            }
            Ix::Mul(a, b) => {
                let (a0, a1) = a.range(d);
                let (b0, b1) = b.range(d);
                let c = [a0 * b0, a0 * b1, a1 * b0, a1 * b1];
                (*c.iter().min().expect("4"), *c.iter().max().expect("4"))
            }
            Ix::Div(a, k) => {
                let (lo, hi) = a.range(d);
                (lo.div_euclid(*k), hi.div_euclid(*k))
            }
            Ix::Mod(a, k) => {
                let (lo, hi) = a.range(d);
                if lo >= 0 && hi < *k {
                    (lo, hi)
                } else {
                    (0, k - 1)
                }
            }
            Ix::Xor(..) | Ix::And(..) | Ix::Shl(..) | Ix::Shr(..) => {
                let b = self.bits(d);
                if b == u64::MAX {
                    (i64::MIN / 4, i64::MAX / 4)
                } else {
                    (0, b as i64)
                }
            }
        }
    }
    /// Bits that may be 1 (u64::MAX = unknown / may be negative).
    pub fn bits(&self, d: &Dyns) -> u64 {
        match self {
            Ix::C(k) => {
                if *k >= 0 {
                    *k as u64
                } else {
                    u64::MAX
                }
            }
            Ix::Sym(s) => d.info(*s).bits,
            Ix::Sum(t, k) => {
                // disjoint terms add like `or`; otherwise a carry may reach one bit above the highest
                let mut acc: u64 = if *k >= 0 { *k as u64 } else { return u64::MAX };
                for (x, m) in t {
                    if *m < 0 {
                        return u64::MAX;
                    }
                    let b = if (*m as u64).is_power_of_two() {
                        let s = (*m as u64).trailing_zeros();
                        let xb = x.bits(d);
                        if xb == u64::MAX || (xb.leading_zeros() as u32) < s {
                            return u64::MAX;
                        }
                        xb << s
                    } else {
                        let (lo, hi) = x.range(d);
                        if lo < 0 {
                            return u64::MAX;
                        }
                        bits_upto(hi.saturating_mul(*m))
                    };
                    if b == u64::MAX {
                        return u64::MAX;
                    }
                    if acc & b == 0 {
                        acc |= b;
                    } else {
                        acc = bits_upto(((acc | b) as i64).saturating_mul(2));
                    }
                }
                acc
            }
            Ix::Mul(..) | Ix::Div(..) => {
                let (lo, hi) = self.range(d);
                if lo < 0 {
                    u64::MAX
                } else {
                    bits_upto(hi)
                }
            }
            Ix::Mod(a, k) => {
                let ab = a.bits(d);
                if (*k as u64).is_power_of_two() && ab != u64::MAX {
                    ab & (*k as u64 - 1)
                } else {
                    bits_upto(k - 1)
                }
            }
            Ix::Xor(a, b) => {
                let (x, y) = (a.bits(d), b.bits(d));
                if x == u64::MAX || y == u64::MAX {
                    u64::MAX
                } else {
                    x | y
                }
            }
            Ix::And(a, m) => {
                let x = a.bits(d);
                if *m >= 0 {
                    if x == u64::MAX {
                        *m as u64
                    } else {
                        x & *m as u64
                    }
                } else {
                    x
                }
            }
            Ix::Shl(a, s) => {
                let x = a.bits(d);
                if x == u64::MAX || x.leading_zeros() < *s {
                    u64::MAX
                } else {
                    x << s
                }
            }
            Ix::Shr(a, s) => {
                let x = a.bits(d);
                if x == u64::MAX {
                    u64::MAX
                } else {
                    x >> s
                }
            }
        }
    }
}

impl fmt::Display for Ix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ix::C(k) => write!(f, "{k}"),
            Ix::Sym(s) => write!(f, "s{s}"),
            Ix::Sum(t, k) => {
                write!(f, "(")?;
                for (i, (x, m)) in t.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    if *m == 1 {
                        write!(f, "{x}")?;
                    } else {
                        write!(f, "{m}*{x}")?;
                    }
                }
                if *k != 0 {
                    write!(f, " + {k}")?;
                }
                write!(f, ")")
            }
            Ix::Mul(a, b) => write!(f, "({a} * {b})"),
            Ix::Div(a, k) => write!(f, "({a} / {k})"),
            Ix::Mod(a, k) => write!(f, "({a} % {k})"),
            Ix::Xor(a, b) => write!(f, "({a} ^ {b})"),
            Ix::And(a, m) => write!(f, "({a} & {m})"),
            Ix::Shl(a, s) => write!(f, "({a} << {s})"),
            Ix::Shr(a, s) => write!(f, "({a} >> {s})"),
        }
    }
}

/// The layout's dynamic symbols in the shared symbolic arithmetic (`amdgpu-sym`): symbol `s` is `Sym::Named("ix<s>:<name>")` with its declared range.
impl Dyns {
    pub fn sym_of(&self, s: u32) -> amdgpu_sym::Sym {
        amdgpu_sym::Sym::Named(format!("ix{s}:{}", self.info(s).name))
    }
    /// The constraint context of the declared ranges.
    pub fn ctx(&self) -> amdgpu_sym::Ctx {
        let mut c = amdgpu_sym::Ctx::new();
        let mut s = 0u32;
        while (s as usize) < self.syms.len() {
            let i = self.info(s);
            c.lower(self.sym_of(s), i.lo);
            c.upper(self.sym_of(s), i.hi);
            s += 1;
        }
        c
    }
}

impl Ix {
    /// The exact polynomial of the affine / product part (None when a division, remainder, xor, mask or shift is involved).
    pub fn to_poly(&self, d: &Dyns) -> Option<amdgpu_sym::Poly> {
        use amdgpu_sym::Poly;
        Some(match self {
            Ix::C(k) => Poly::int(*k as i128),
            Ix::Sym(s) => Poly::sym(d.sym_of(*s)),
            Ix::Sum(t, k) => {
                let mut p = Poly::int(*k as i128);
                for (x, m) in t {
                    p = p.add(&x.to_poly(d)?.scale(amdgpu_sym::Q::int(*m as i128)));
                }
                p
            }
            Ix::Mul(a, b) => a.to_poly(d)?.mul(&b.to_poly(d)?)?,
            _ => return None,
        })
    }
    /// Sound proof of `self <= o` for every admissible value of the dynamic symbols (shared constraint context; false = not proven).
    pub fn can_prove_le(&self, o: &Ix, d: &Dyns) -> bool {
        match (self.to_poly(d), o.to_poly(d)) {
            (Some(a), Some(b)) => d.ctx().le(&a, &b),
            _ => self.range(d).1 <= o.range(d).0,
        }
    }
}
