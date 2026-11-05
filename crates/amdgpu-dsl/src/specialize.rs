//! Shape specialization (`amdc emit --specialize name=value,...`, docs/jit.md): kernarg scalars bound to constants.
//!
//! * every `Op::KernargI32 / KernargF32` load of a bound argument becomes a constant (the ABI is unchanged: the host still passes the argument);
//! * the contract gets the exact fact (`ka_range(off, v, v)`) and every `Ext::Ka(off)` of a bound argument becomes the constant, so extents, grid
//!   definitions and kernarg relations are checked / used with the value;
//! * the body is constant-folded: integer arithmetic of constants becomes a constant, and an `if` whose condition is a compare of constants keeps
//!   only the branch that runs (tails and clamps that are provably dead disappear); the lowering then folds what remains (immediates, scalar
//!   compares against constants).
//!
//! Rejected (Err): unknown names, pointers, duplicates, values outside a declared `ka_range` / not in a `ka_set`, and kernarg relations
//! (`ka_is`) that become false.

use crate::{Contract, DType, Ext, Function, Node, Op, Opnd, Stmt, ValueId};
use std::collections::HashMap;

/// One bound argument.
#[derive(Clone, Debug, PartialEq)]
pub enum SpecValue {
    Int(i64),
    F32(f32),
}

impl SpecValue {
    pub fn to_json(&self) -> String {
        match self {
            SpecValue::Int(v) => v.to_string(),
            SpecValue::F32(v) => format!("{v:?}"),
        }
    }
}

/// Parses `a=1,b=2` (also accepts repeated flags joined by the caller).
pub fn parse_bindings(s: &str) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    for part in s.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
        let (k, v) = part.split_once('=').ok_or_else(|| format!("--specialize: `{part}` is not name=value"))?;
        let (k, v) = (k.trim().to_string(), v.trim().to_string());
        if out.iter().any(|(x, _)| *x == k) {
            return Err(format!("--specialize: `{k}` given twice"));
        }
        out.push((k, v));
    }
    Ok(out)
}

/// The kind of a kernarg as the JIT metadata names it.
pub fn arg_kind(t: DType) -> &'static str {
    match t {
        DType::Ptr => "ptr",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::F32 => "f32",
        _ => "other",
    }
}

pub fn specializable(t: DType) -> bool {
    matches!(t, DType::I32 | DType::U32 | DType::F32)
}

fn eval(e: &Ext) -> Option<i64> {
    Some(match e {
        Ext::Const(c) => *c,
        Ext::Add(a, b) => eval(a)? + eval(b)?,
        Ext::Sub(a, b) => eval(a)? - eval(b)?,
        Ext::Mul(a, b) => eval(a)? * eval(b)?,
        Ext::CeilDiv(a, k) => (eval(a)? + k - 1).div_euclid(*k),
        Ext::FloorDiv(a, k) => eval(a)?.div_euclid(*k),
        _ => return None,
    })
}

/// `e` with every `Ka(off)` of a bound kernarg replaced by its value.
pub fn subst_ext(e: &Ext, ka: &HashMap<u32, i64>) -> Ext {
    let b = |x: &Ext| Box::new(subst_ext(x, ka));
    match e {
        Ext::Ka(o) => match ka.get(o) {
            Some(v) => Ext::Const(*v),
            None => e.clone(),
        },
        Ext::Add(a, c) => Ext::Add(b(a), b(c)),
        Ext::Sub(a, c) => Ext::Sub(b(a), b(c)),
        Ext::Mul(a, c) => Ext::Mul(b(a), b(c)),
        Ext::CeilDiv(a, k) => Ext::CeilDiv(b(a), *k),
        Ext::FloorDiv(a, k) => Ext::FloorDiv(b(a), *k),
        other => other.clone(),
    }
}

/// Validates the bindings against the kernel's arguments and contract; returns the typed values (by argument name).
pub fn check(f: &Function, binds: &[(String, String)]) -> Result<Vec<(String, u32, SpecValue)>, String> {
    let mut out = Vec::new();
    for (k, v) in binds {
        let Some(a) = f.args.iter().find(|a| a.name == *k) else {
            let names: Vec<&str> = f.args.iter().filter(|a| specializable(a.ty)).map(|a| a.name.as_str()).collect();
            return Err(format!("--specialize: kernel `{}` has no argument `{k}` (specializable: {})", f.name, names.join(", ")));
        };
        let val = match a.ty {
            DType::Ptr => return Err(format!("--specialize: `{k}` is a pointer; only scalar kernargs can be specialized")),
            DType::I32 | DType::U32 => {
                let x: i64 = v.parse().map_err(|_| format!("--specialize: `{k}={v}`: not an integer"))?;
                let ok = if a.ty == DType::I32 { (i32::MIN as i64..=i32::MAX as i64).contains(&x) } else { (0..=u32::MAX as i64).contains(&x) };
                if !ok {
                    return Err(format!("--specialize: `{k}={v}` does not fit {}", a.ty.name()));
                }
                SpecValue::Int(x)
            }
            DType::F32 => SpecValue::F32(v.parse().map_err(|_| format!("--specialize: `{k}={v}`: not a number"))?),
            other => return Err(format!("--specialize: `{k}` has type {} (not specializable)", other.name())),
        };
        if let SpecValue::Int(x) = val {
            for (o, lo, hi) in &f.contract.ka_ranges {
                if *o == a.kernarg_offset && (x < *lo || x > *hi) {
                    return Err(format!("--specialize: `{k}={x}` violates the contract range [{lo}, {}]", if *hi == i64::MAX { "inf".to_string() } else { hi.to_string() }));
                }
            }
            for (o, vs) in &f.contract.ka_sets {
                if *o == a.kernarg_offset && !vs.contains(&x) {
                    return Err(format!("--specialize: `{k}={x}` is not one of the contract values {vs:?}"));
                }
            }
        }
        out.push((k.clone(), a.kernarg_offset, val));
    }
    // kernarg relations that become closed must hold
    let ka: HashMap<u32, i64> = out.iter().filter_map(|(_, o, v)| if let SpecValue::Int(x) = v { Some((*o, *x)) } else { None }).collect();
    for (o, e) in &f.contract.ka_defs {
        let lhs = subst_ext(&Ext::Ka(*o), &ka);
        let rhs = subst_ext(e, &ka);
        if let (Some(l), Some(r)) = (eval(&lhs), eval(&rhs)) {
            if l != r {
                return Err(format!("--specialize: the contract relation ka(0x{o:x}) == {e:?} is violated ({l} != {r})"));
            }
        }
    }
    Ok(out)
}

/// The specialized kernel (same name / ABI).
pub fn specialize(f: &Function, binds: &[(String, String)]) -> Result<Function, String> {
    let vals = check(f, binds)?;
    let mut g = f.clone();
    let ka: HashMap<u32, i64> = vals.iter().filter_map(|(_, o, v)| if let SpecValue::Int(x) = v { Some((*o, *x)) } else { None }).collect();
    let kf: HashMap<u32, f32> = vals.iter().filter_map(|(_, o, v)| if let SpecValue::F32(x) = v { Some((*o, *x)) } else { None }).collect();
    // contract
    let c: &mut Contract = &mut g.contract;
    for e in c.extents.iter_mut() {
        e.alloc = subst_ext(&e.alloc, &ka);
        e.logical = e.logical.as_ref().map(|x| subst_ext(x, &ka));
    }
    for (_, e) in c.grid_defs.iter_mut() {
        *e = subst_ext(e, &ka);
    }
    for (_, e) in c.ka_defs.iter_mut() {
        *e = subst_ext(e, &ka);
    }
    for (o, v) in &ka {
        c.ka_ranges.retain(|(x, _, _)| x != o);
        c.ka_sets.retain(|(x, _)| x != o);
        c.ka_ranges.push((*o, *v, *v));
    }
    c.ka_ranges.sort_by_key(|x| x.0);
    // body: kernarg loads -> constants, then fold
    let mut body = g.body().to_vec();
    rewrite_loads(&mut body, &ka, &kf);
    let mut consts: HashMap<ValueId, i64> = HashMap::new();
    let mut conds: HashMap<ValueId, bool> = HashMap::new();
    fold(&mut body, &mut consts, &mut conds);
    g.set_body(body);
    Ok(g)
}

fn rewrite_loads(nodes: &mut [Node], ka: &HashMap<u32, i64>, kf: &HashMap<u32, f32>) {
    for n in nodes.iter_mut() {
        match n {
            Node::S(Stmt::Let { op, .. }) => match op {
                Op::KernargI32 { offset } => {
                    if let Some(v) = ka.get(offset) {
                        *op = Op::ConstI32(*v as i32);
                    }
                }
                Op::KernargF32 { offset } => {
                    if let Some(v) = kf.get(offset) {
                        *op = Op::ConstF32(*v);
                    }
                }
                _ => {}
            },
            Node::S(_) => {}
            Node::Loop { body, .. } | Node::Region { body, .. } | Node::Stage { body, .. } => rewrite_loads(body, ka, kf),
            Node::If { then, els, .. } => {
                rewrite_loads(then, ka, kf);
                if let Some(e) = els {
                    rewrite_loads(e, ka, kf);
                }
            }
        }
    }
}

fn cint(o: &Opnd, consts: &HashMap<ValueId, i64>) -> Option<i64> {
    match o {
        Opnd::I32(c) => Some(*c as i64),
        Opnd::Val(v) => consts.get(v).copied(),
        _ => None,
    }
}

fn ccond(o: &Opnd, conds: &HashMap<ValueId, bool>) -> Option<bool> {
    match o {
        Opnd::Val(v) => conds.get(v).copied(),
        _ => None,
    }
}

/// Folds integer arithmetic of constants into `ConstI32` and resolves `if`s on constant compares (wave-uniform by construction: the condition is
/// the same in every lane). Values are 32-bit (wrap-around like the hardware).
fn fold(nodes: &mut Vec<Node>, consts: &mut HashMap<ValueId, i64>, conds: &mut HashMap<ValueId, bool>) {
    let mut out: Vec<Node> = Vec::with_capacity(nodes.len());
    for mut n in std::mem::take(nodes) {
        match &mut n {
            Node::S(Stmt::Let { dst, op }) => {
                let w = |x: i64| (x as i32) as i64;
                let k = |o: &Opnd| cint(o, consts);
                let folded: Option<i64> = match op {
                    Op::ConstI32(c) => Some(*c as i64),
                    Op::AddI32(a, b) => Some(w(k(a)? + k(b)?)),
                    Op::MulI32(a, b) => Some(w(k(a)?.wrapping_mul(k(b)?))),
                    Op::ShlI32(a, b) => Some(w(((k(a)? as u32).wrapping_shl(k(b)? as u32)) as i32 as i64)),
                    Op::LshrI32(a, b) => Some(((k(a)? as u32).wrapping_shr(k(b)? as u32)) as i32 as i64),
                    Op::AndI32(a, b) => Some(w(k(a)? & k(b)?)),
                    Op::XorI32(a, b) => Some(w(k(a)? ^ k(b)?)),
                    Op::MinI32(a, b) => Some(k(a)?.min(k(b)?)),
                    _ => None,
                };
                if let Some(v) = folded {
                    consts.insert(*dst, v);
                    *op = Op::ConstI32(v as i32);
                } else {
                    let c = match op {
                        Op::CmpLtI32(a, b) => k(a).zip(k(b)).map(|(x, y)| x < y),
                        Op::CmpGeI32(a, b) => k(a).zip(k(b)).map(|(x, y)| x >= y),
                        Op::NotI1(a) => ccond(a, conds).map(|x| !x),
                        _ => None,
                    };
                    if let Some(c) = c {
                        conds.insert(*dst, c);
                    }
                }
                out.push(n);
            }
            Node::If { cond, then, els } => {
                match ccond(cond, conds) {
                    Some(true) => {
                        let mut t = std::mem::take(then);
                        fold(&mut t, consts, conds);
                        out.extend(t);
                    }
                    Some(false) => {
                        if let Some(e) = els {
                            let mut e = std::mem::take(e);
                            fold(&mut e, consts, conds);
                            out.extend(e);
                        }
                    }
                    None => {
                        fold(then, consts, conds);
                        if let Some(e) = els {
                            fold(e, consts, conds);
                        }
                        out.push(n);
                    }
                }
            }
            Node::Loop { body, .. } | Node::Region { body, .. } | Node::Stage { body, .. } => {
                fold(body, consts, conds);
                out.push(n);
            }
            Node::S(_) => out.push(n),
        }
    }
    *nodes = out;
}
