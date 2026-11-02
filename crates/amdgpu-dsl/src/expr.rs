//! Operator expression layer over the UNCHANGED IR.
//!
//! `I32`, `F32`, `Cond` and `Ptr` are pure DATA: reference-counted expression trees that hold no `&mut FnBuilder`, which is exactly what makes `a * 4 + b` legal Rust (the
//! operators build trees; nothing touches the builder until you `bind`). `FnBuilder::bind_i32/bind_f32/bind_cond/bind_ptr` lower a tree post-order (left operand, right
//! operand, then the operation; shared subtrees - the same `Rc` node - are lowered once per `bind` call) to EXACTLY the `Op`s the manual API emits:
//!
//! | expression | emitted op(s) |
//! |---|---|
//! | `a + b`, `a * b`, `a << b`, `a >> b` (LOGICAL), `a & b`, `a ^ b`, `a.min(b)` (I32) | `AddI32`, `MulI32`, `ShlI32`, `LshrI32`, `AndI32`, `XorI32`, `MinI32` |
//! | `a + b`, `a - b`, `a * b`, `a.max(b)` (F32) | `AddF32`, `SubF32`, `MulF32`, `MaxF32` |
//! | `a.fma(b, c)`, `a.exp2()`, `a.log2()`, `a.rcp()`, `a.exp()` | `FmaF32`, `Exp2F32`, `Log2F32`, `RcpF32`, `ExpF32` |
//! | `a.lt(b)`, `a.ge(b)` (I32), `a.lt(b)` (F32), `!c` | `CmpLtI32`, `CmpGeI32`, `CmpLtF32`, `NotI1` |
//! | `c.select_i32(a, b)`, `c.select_f32(a, b)` | `Select` |
//! | `p + off` (Ptr + I32) | `PtrAdd` |
//! | `a.bitcast_f32()`, `x.bitcast_i32()` | `BitcastF32`, `BitcastI32` |
//!
//! Operators the IR has NO single op for are DEFINED as a fixed combination of the existing ops (documented; correct modulo 2^32 / IEEE sign handling):
//! `a - b` (I32) = `a + b * -1` (`a + (-k)` for a literal), `-a` = `a * -1`, `!a` (bitwise) = `a ^ -1`, `a | b` = `(a ^ b) + (a & b)`, `a.max(b)` (I32) =
//! `select(a < b, b, a)`, `a > b` = `b < a`, `a <= b` = `b >= a`, `a.ne(b)` = `min(a - b, -(a - b)) < 0`, `a.eq(b)` = `!a.ne(b)`, `-x` (F32) = `x * -1.0`, `x > y` (F32) = `y < x`.
//! There is deliberately NO `/` and NO implicit fma: `a * b + c` stays `MulF32` + `AddF32` (a fused multiply-add changes rounding); write `a.fma(b, c)` to get `FmaF32`.
//! There is no expression-level optimisation or CSE beyond the per-`bind` sharing: what you write is what is emitted. Literals become `ConstI32/ConstF32` statements at their first use
//! (`hoist_i32` / `hoist_f32` place them up front, like manual code) and are reused afterwards in the same or an inner scope; the reuse table is cleared when an `if`/`else`/loop scope closes.

use std::collections::HashMap;
use std::ops::{Add, BitAnd, BitOr, BitXor, Mul, Neg, Not, Shl, Shr, Sub};
use std::rc::Rc;

use crate::{DType, FnBuilder, Op, Opnd, ValueId};

type N = Rc<Node>;

/// One node of an expression tree (leaves are already-bound values or constants).
#[derive(Debug)]
pub enum Node {
    Val(ValueId),
    CI(i32),
    CF(f32),
    AddI(N, N),
    MulI(N, N),
    ShlI(N, N),
    LshrI(N, N),
    AndI(N, N),
    XorI(N, N),
    MinI(N, N),
    AddF(N, N),
    SubF(N, N),
    MulF(N, N),
    MaxF(N, N),
    FmaF(N, N, N),
    ExpF(N),
    Exp2F(N),
    Log2F(N),
    RcpF(N),
    CmpLtI(N, N),
    CmpGeI(N, N),
    CmpLtF(N, N),
    NotC(N),
    Select(DType, N, N, N),
    PtrAdd(N, N),
    BitcastF(N),
    BitcastI(N),
}

/// An i32 expression.
#[derive(Clone, Debug)]
pub struct I32(pub(crate) N);
/// An f32 expression.
#[derive(Clone, Debug)]
pub struct F32(pub(crate) N);
/// A boolean (i1) expression.
#[derive(Clone, Debug)]
pub struct Cond(pub(crate) N);
/// A global pointer expression (base pointer + byte offsets).
#[derive(Clone, Debug)]
pub struct Ptr(pub(crate) N);

fn n(x: Node) -> N {
    Rc::new(x)
}

impl I32 {
    pub fn c(v: i32) -> I32 {
        I32(n(Node::CI(v)))
    }
    pub fn lshr(&self, r: impl Into<I32>) -> I32 {
        I32(n(Node::LshrI(self.0.clone(), r.into().0)))
    }
    pub fn min(&self, r: impl Into<I32>) -> I32 {
        I32(n(Node::MinI(self.0.clone(), r.into().0)))
    }
    /// `select(self < r, r, self)` (the IR has no MaxI32).
    pub fn max(&self, r: impl Into<I32>) -> I32 {
        let r = r.into();
        Cond(n(Node::CmpLtI(self.0.clone(), r.0.clone()))).select_i32(r, self.clone())
    }
    pub fn lt(&self, r: impl Into<I32>) -> Cond {
        Cond(n(Node::CmpLtI(self.0.clone(), r.into().0)))
    }
    pub fn ge(&self, r: impl Into<I32>) -> Cond {
        Cond(n(Node::CmpGeI(self.0.clone(), r.into().0)))
    }
    /// `r < self`
    pub fn gt(&self, r: impl Into<I32>) -> Cond {
        Cond(n(Node::CmpLtI(r.into().0, self.0.clone())))
    }
    /// `r >= self`
    pub fn le(&self, r: impl Into<I32>) -> Cond {
        Cond(n(Node::CmpGeI(r.into().0, self.0.clone())))
    }
    /// `min(d, -d) < 0` with `d = self - r` (5 ops).
    pub fn ne(&self, r: impl Into<I32>) -> Cond {
        let d = self.clone() - r.into();
        let m = d.min(-d.clone());
        m.lt(0)
    }
    pub fn eq(&self, r: impl Into<I32>) -> Cond {
        !self.ne(r)
    }
    pub fn bitcast_f32(&self) -> F32 {
        F32(n(Node::BitcastF(self.0.clone())))
    }
    /// Evaluates a tree on the host (wrapping i32 semantics; `env` supplies the leaf values). Used to test the documented decompositions.
    pub fn eval(&self, env: &dyn Fn(ValueId) -> i32) -> i32 {
        eval_i(&self.0, env)
    }
}

fn eval_i(x: &N, env: &dyn Fn(ValueId) -> i32) -> i32 {
    match &**x {
        Node::Val(v) => env(*v),
        Node::CI(c) => *c,
        Node::AddI(a, b) => eval_i(a, env).wrapping_add(eval_i(b, env)),
        Node::MulI(a, b) => eval_i(a, env).wrapping_mul(eval_i(b, env)),
        Node::ShlI(a, b) => (eval_i(a, env) as u32).wrapping_shl(eval_i(b, env) as u32) as i32,
        Node::LshrI(a, b) => (eval_i(a, env) as u32).wrapping_shr(eval_i(b, env) as u32) as i32,
        Node::AndI(a, b) => eval_i(a, env) & eval_i(b, env),
        Node::XorI(a, b) => eval_i(a, env) ^ eval_i(b, env),
        Node::MinI(a, b) => eval_i(a, env).min(eval_i(b, env)),
        Node::Select(_, c, a, b) => {
            if eval_c(c, env) {
                eval_i(a, env)
            } else {
                eval_i(b, env)
            }
        }
        other => panic!("not an i32 expression node: {other:?}"),
    }
}

fn eval_c(x: &N, env: &dyn Fn(ValueId) -> i32) -> bool {
    match &**x {
        Node::CmpLtI(a, b) => eval_i(a, env) < eval_i(b, env),
        Node::CmpGeI(a, b) => eval_i(a, env) >= eval_i(b, env),
        Node::NotC(a) => !eval_c(a, env),
        other => panic!("not a condition node: {other:?}"),
    }
}

impl Cond {
    pub fn select_i32(&self, a: impl Into<I32>, b: impl Into<I32>) -> I32 {
        I32(n(Node::Select(DType::I32, self.0.clone(), a.into().0, b.into().0)))
    }
    pub fn select_f32(&self, a: impl Into<F32>, b: impl Into<F32>) -> F32 {
        F32(n(Node::Select(DType::F32, self.0.clone(), a.into().0, b.into().0)))
    }
    pub fn eval(&self, env: &dyn Fn(ValueId) -> i32) -> bool {
        eval_c(&self.0, env)
    }
}

impl Not for Cond {
    type Output = Cond;
    fn not(self) -> Cond {
        Cond(n(Node::NotC(self.0)))
    }
}

impl F32 {
    pub fn c(v: f32) -> F32 {
        F32(n(Node::CF(v)))
    }
    pub fn max(&self, r: impl Into<F32>) -> F32 {
        F32(n(Node::MaxF(self.0.clone(), r.into().0)))
    }
    /// `self * b + c` as ONE fused op (different rounding from `self * b + c` written with operators).
    pub fn fma(&self, b: impl Into<F32>, c: impl Into<F32>) -> F32 {
        F32(n(Node::FmaF(self.0.clone(), b.into().0, c.into().0)))
    }
    pub fn exp(&self) -> F32 {
        F32(n(Node::ExpF(self.0.clone())))
    }
    pub fn exp2(&self) -> F32 {
        F32(n(Node::Exp2F(self.0.clone())))
    }
    pub fn log2(&self) -> F32 {
        F32(n(Node::Log2F(self.0.clone())))
    }
    pub fn rcp(&self) -> F32 {
        F32(n(Node::RcpF(self.0.clone())))
    }
    pub fn lt(&self, r: impl Into<F32>) -> Cond {
        Cond(n(Node::CmpLtF(self.0.clone(), r.into().0)))
    }
    /// `r < self`
    pub fn gt(&self, r: impl Into<F32>) -> Cond {
        Cond(n(Node::CmpLtF(r.into().0, self.0.clone())))
    }
    pub fn bitcast_i32(&self) -> I32 {
        I32(n(Node::BitcastI(self.0.clone())))
    }
}

impl From<i32> for I32 {
    fn from(v: i32) -> I32 {
        I32::c(v)
    }
}
impl From<&I32> for I32 {
    fn from(v: &I32) -> I32 {
        v.clone()
    }
}
impl From<f32> for F32 {
    fn from(v: f32) -> F32 {
        F32::c(v)
    }
}
impl From<&F32> for F32 {
    fn from(v: &F32) -> F32 {
        v.clone()
    }
}

// ---- operator plumbing -----------------------------------------------------------------------------------------------------------------------------------------------

fn add_i(a: &I32, b: &I32) -> I32 {
    I32(n(Node::AddI(a.0.clone(), b.0.clone())))
}
fn sub_i(a: &I32, b: &I32) -> I32 {
    match &*b.0 {
        Node::CI(k) => add_i(a, &I32::c(k.wrapping_neg())),
        _ => add_i(a, &mul_i(b, &I32::c(-1))),
    }
}
fn mul_i(a: &I32, b: &I32) -> I32 {
    I32(n(Node::MulI(a.0.clone(), b.0.clone())))
}
fn shl_i(a: &I32, b: &I32) -> I32 {
    I32(n(Node::ShlI(a.0.clone(), b.0.clone())))
}
fn shr_i(a: &I32, b: &I32) -> I32 {
    I32(n(Node::LshrI(a.0.clone(), b.0.clone())))
}
fn and_i(a: &I32, b: &I32) -> I32 {
    I32(n(Node::AndI(a.0.clone(), b.0.clone())))
}
fn xor_i(a: &I32, b: &I32) -> I32 {
    I32(n(Node::XorI(a.0.clone(), b.0.clone())))
}
/// `a | b = (a ^ b) + (a & b)` (the IR has no OrI32).
fn or_i(a: &I32, b: &I32) -> I32 {
    add_i(&xor_i(a, b), &and_i(a, b))
}

macro_rules! i32_binop {
    ($tr:ident, $m:ident, $f:ident) => {
        impl $tr<I32> for I32 {
            type Output = I32;
            fn $m(self, r: I32) -> I32 {
                $f(&self, &r)
            }
        }
        impl $tr<&I32> for I32 {
            type Output = I32;
            fn $m(self, r: &I32) -> I32 {
                $f(&self, r)
            }
        }
        impl $tr<I32> for &I32 {
            type Output = I32;
            fn $m(self, r: I32) -> I32 {
                $f(self, &r)
            }
        }
        impl $tr<&I32> for &I32 {
            type Output = I32;
            fn $m(self, r: &I32) -> I32 {
                $f(self, r)
            }
        }
        impl $tr<i32> for I32 {
            type Output = I32;
            fn $m(self, r: i32) -> I32 {
                $f(&self, &I32::c(r))
            }
        }
        impl $tr<i32> for &I32 {
            type Output = I32;
            fn $m(self, r: i32) -> I32 {
                $f(self, &I32::c(r))
            }
        }
        impl $tr<I32> for i32 {
            type Output = I32;
            fn $m(self, r: I32) -> I32 {
                $f(&I32::c(self), &r)
            }
        }
        impl $tr<&I32> for i32 {
            type Output = I32;
            fn $m(self, r: &I32) -> I32 {
                $f(&I32::c(self), r)
            }
        }
    };
}
i32_binop!(Add, add, add_i);
i32_binop!(Sub, sub, sub_i);
i32_binop!(Mul, mul, mul_i);
i32_binop!(Shl, shl, shl_i);
i32_binop!(Shr, shr, shr_i);
i32_binop!(BitAnd, bitand, and_i);
i32_binop!(BitOr, bitor, or_i);
i32_binop!(BitXor, bitxor, xor_i);

impl Neg for I32 {
    type Output = I32;
    fn neg(self) -> I32 {
        mul_i(&self, &I32::c(-1))
    }
}
impl Neg for &I32 {
    type Output = I32;
    fn neg(self) -> I32 {
        mul_i(self, &I32::c(-1))
    }
}
impl Not for &I32 {
    type Output = I32;
    fn not(self) -> I32 {
        xor_i(self, &I32::c(-1))
    }
}
impl Not for I32 {
    type Output = I32;
    fn not(self) -> I32 {
        xor_i(&self, &I32::c(-1))
    }
}

fn add_f(a: &F32, b: &F32) -> F32 {
    F32(n(Node::AddF(a.0.clone(), b.0.clone())))
}
fn sub_f(a: &F32, b: &F32) -> F32 {
    F32(n(Node::SubF(a.0.clone(), b.0.clone())))
}
fn mul_f(a: &F32, b: &F32) -> F32 {
    F32(n(Node::MulF(a.0.clone(), b.0.clone())))
}

macro_rules! f32_binop {
    ($tr:ident, $m:ident, $f:ident) => {
        impl $tr<F32> for F32 {
            type Output = F32;
            fn $m(self, r: F32) -> F32 {
                $f(&self, &r)
            }
        }
        impl $tr<&F32> for F32 {
            type Output = F32;
            fn $m(self, r: &F32) -> F32 {
                $f(&self, r)
            }
        }
        impl $tr<F32> for &F32 {
            type Output = F32;
            fn $m(self, r: F32) -> F32 {
                $f(self, &r)
            }
        }
        impl $tr<&F32> for &F32 {
            type Output = F32;
            fn $m(self, r: &F32) -> F32 {
                $f(self, r)
            }
        }
        impl $tr<f32> for F32 {
            type Output = F32;
            fn $m(self, r: f32) -> F32 {
                $f(&self, &F32::c(r))
            }
        }
        impl $tr<f32> for &F32 {
            type Output = F32;
            fn $m(self, r: f32) -> F32 {
                $f(self, &F32::c(r))
            }
        }
        impl $tr<F32> for f32 {
            type Output = F32;
            fn $m(self, r: F32) -> F32 {
                $f(&F32::c(self), &r)
            }
        }
    };
}
f32_binop!(Add, add, add_f);
f32_binop!(Sub, sub, sub_f);
f32_binop!(Mul, mul, mul_f);

impl Neg for F32 {
    type Output = F32;
    fn neg(self) -> F32 {
        mul_f(&self, &F32::c(-1.0))
    }
}

impl Add<I32> for Ptr {
    type Output = Ptr;
    fn add(self, off: I32) -> Ptr {
        Ptr(n(Node::PtrAdd(self.0, off.0)))
    }
}
impl Add<&I32> for Ptr {
    type Output = Ptr;
    fn add(self, off: &I32) -> Ptr {
        Ptr(n(Node::PtrAdd(self.0, off.0.clone())))
    }
}
impl Add<I32> for &Ptr {
    type Output = Ptr;
    fn add(self, off: I32) -> Ptr {
        Ptr(n(Node::PtrAdd(self.0.clone(), off.0)))
    }
}
impl Add<&I32> for &Ptr {
    type Output = Ptr;
    fn add(self, off: &I32) -> Ptr {
        Ptr(n(Node::PtrAdd(self.0.clone(), off.0.clone())))
    }
}
impl Add<i32> for Ptr {
    type Output = Ptr;
    fn add(self, off: i32) -> Ptr {
        Ptr(n(Node::PtrAdd(self.0, n(Node::CI(off)))))
    }
}

// ---- lowering ----------------------------------------------------------------------------------------------------------------------------------------------------

struct Lower<'a> {
    b: &'a mut FnBuilder,
    memo: HashMap<*const Node, ValueId>,
}

impl Lower<'_> {
    fn go(&mut self, x: &N) -> ValueId {
        if let Node::Val(v) = &**x {
            return *v;
        }
        let key = Rc::as_ptr(x);
        if let Some(v) = self.memo.get(&key) {
            return *v;
        }
        let v = match &**x {
            Node::Val(v) => *v,
            Node::CI(c) => match self.b.expr_consts.get(&(false, *c as u32)) {
                Some(v) => *v,
                None => {
                    let v = self.b.c_i32(*c);
                    self.b.expr_consts.insert((false, *c as u32), v);
                    v
                }
            },
            Node::CF(c) => match self.b.expr_consts.get(&(true, c.to_bits())) {
                Some(v) => *v,
                None => {
                    let v = self.b.c_f32(*c);
                    self.b.expr_consts.insert((true, c.to_bits()), v);
                    v
                }
            },
            Node::AddI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.add_i32(a, c)
            }
            Node::MulI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.mul_i32(a, c)
            }
            Node::ShlI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.shl_i32(a, c)
            }
            Node::LshrI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.let_(Op::LshrI32(Opnd::Val(a), Opnd::Val(c)))
            }
            Node::AndI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.let_(Op::AndI32(Opnd::Val(a), Opnd::Val(c)))
            }
            Node::XorI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.xor_i32(a, c)
            }
            Node::MinI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.min_i32(a, c)
            }
            Node::AddF(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.add_f32(a, c)
            }
            Node::SubF(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.sub_f32(a, c)
            }
            Node::MulF(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.mul_f32(a, c)
            }
            Node::MaxF(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.max_f32(a, c)
            }
            Node::FmaF(a, c, d) => {
                let (a, c, d) = (self.go(a), self.go(c), self.go(d));
                self.b.let_(Op::FmaF32(Opnd::Val(a), Opnd::Val(c), Opnd::Val(d)))
            }
            Node::ExpF(a) => {
                let a = self.go(a);
                self.b.let_(Op::ExpF32(Opnd::Val(a)))
            }
            Node::Exp2F(a) => {
                let a = self.go(a);
                self.b.exp2_f32(a)
            }
            Node::Log2F(a) => {
                let a = self.go(a);
                self.b.let_(Op::Log2F32(Opnd::Val(a)))
            }
            Node::RcpF(a) => {
                let a = self.go(a);
                self.b.rcp_f32(a)
            }
            Node::CmpLtI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.cmp_lt_i32(a, c)
            }
            Node::CmpGeI(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.let_(Op::CmpGeI32(Opnd::Val(a), Opnd::Val(c)))
            }
            Node::CmpLtF(a, c) => {
                let (a, c) = (self.go(a), self.go(c));
                self.b.let_(Op::CmpLtF32(Opnd::Val(a), Opnd::Val(c)))
            }
            Node::NotC(a) => {
                let a = self.go(a);
                self.b.let_(Op::NotI1(Opnd::Val(a)))
            }
            Node::Select(ty, c, a, d) => {
                let (c, a, d) = (self.go(c), self.go(a), self.go(d));
                self.b.let_(Op::Select { ty: *ty, cond: Opnd::Val(c), a: Opnd::Val(a), b: Opnd::Val(d) })
            }
            Node::PtrAdd(base, off) => {
                let (base, off) = (self.go(base), self.go(off));
                self.b.ptr_add(base, off)
            }
            Node::BitcastF(a) => {
                let a = self.go(a);
                self.b.bitcast_f32(a)
            }
            Node::BitcastI(a) => {
                let a = self.go(a);
                self.b.bitcast_i32(a)
            }
        };
        self.memo.insert(key, v);
        v
    }
}

impl FnBuilder {
    /// Number of statements emitted so far (lets tests count what a `bind` lowered to).
    #[doc(hidden)]
    pub fn stmt_count(&self) -> usize {
        self.f.stmts.len()
    }
    /// Materialises these i32 constants NOW, in this order, and lets every later expression reuse them (the manual style declares its constants up front; this keeps the
    /// emitted instruction order - and so the register allocation - identical to such a kernel). Without it a literal is materialised at its first use and then reused.
    pub fn hoist_i32(&mut self, vals: &[i32]) {
        for &v in vals {
            if !self.expr_consts.contains_key(&(false, v as u32)) {
                let id = self.c_i32(v);
                self.expr_consts.insert((false, v as u32), id);
            }
        }
    }
    pub fn hoist_f32(&mut self, vals: &[f32]) {
        for &v in vals {
            if !self.expr_consts.contains_key(&(true, v.to_bits())) {
                let id = self.c_f32(v);
                self.expr_consts.insert((true, v.to_bits()), id);
            }
        }
    }
    /// An already-bound i32 value as an expression leaf.
    pub fn i32v(&self, v: ValueId) -> I32 {
        I32(n(Node::Val(v)))
    }
    pub fn f32v(&self, v: ValueId) -> F32 {
        F32(n(Node::Val(v)))
    }
    pub fn condv(&self, v: ValueId) -> Cond {
        Cond(n(Node::Val(v)))
    }
    pub fn ptrv(&self, v: ValueId) -> Ptr {
        Ptr(n(Node::Val(v)))
    }
    /// Lowers the tree post-order to the same `Op`s the manual API emits and returns the value of its root.
    pub fn bind_i32(&mut self, e: impl Into<I32>) -> ValueId {
        let e = e.into();
        Lower { b: self, memo: HashMap::new() }.go(&e.0)
    }
    pub fn bind_f32(&mut self, e: impl Into<F32>) -> ValueId {
        let e = e.into();
        Lower { b: self, memo: HashMap::new() }.go(&e.0)
    }
    pub fn bind_cond(&mut self, e: Cond) -> ValueId {
        Lower { b: self, memo: HashMap::new() }.go(&e.0)
    }
    pub fn bind_ptr(&mut self, e: Ptr) -> ValueId {
        Lower { b: self, memo: HashMap::new() }.go(&e.0)
    }
    /// `bind_i32` returning the bound value as a leaf expression (so it can be reused in later expressions without re-lowering).
    pub fn let_i32(&mut self, e: impl Into<I32>) -> I32 {
        let v = self.bind_i32(e);
        self.i32v(v)
    }
    pub fn let_f32(&mut self, e: impl Into<F32>) -> F32 {
        let v = self.bind_f32(e);
        self.f32v(v)
    }
    pub fn let_ptr(&mut self, e: Ptr) -> Ptr {
        let v = self.bind_ptr(e);
        self.ptrv(v)
    }
    /// `select(cond, a, b)` on i32 expressions.
    pub fn select_i32(&self, cond: &Cond, a: impl Into<I32>, b: impl Into<I32>) -> I32 {
        cond.select_i32(a, b)
    }
    pub fn select_f32(&self, cond: &Cond, a: impl Into<F32>, b: impl Into<F32>) -> F32 {
        cond.select_f32(a, b)
    }
}