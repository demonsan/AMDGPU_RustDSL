//! L1: kernarg tensors. A tensor is declared ONCE (pointer kernarg, dims, byte strides); the declaration emits the kernarg layout, loads the scalars and
//! generates the GPU-semantics host contract (`arg_tensor[_slack]`, `ka_is`, `ka_range`) from the same dims/strides the addresses are computed from.

use std::collections::{HashMap, HashSet};

use amdgpu_dsl::{ev};
use crate::{is_zero};
use amdgpu_dsl::expr::Node;
use amdgpu_dsl::{DType, Ext, FnBuilder, Op, ValueId, I32};

/// Kernarg table: declares each kernarg once and loads each scalar once (the loaded value is cached: call the loaders at the kernel's top-level scope,
/// as manual kernels do, so that the cached value is visible everywhere).
#[derive(Default)]
pub struct KArgs {
    declared: HashSet<u32>,
    i32s: HashMap<u32, ValueId>,
    ptrs: HashMap<u32, ValueId>,
}

impl KArgs {
    pub fn new() -> KArgs {
        KArgs::default()
    }
    /// Declares a kernarg (no load); a second declaration of the same offset is ignored.
    pub fn arg(&mut self, b: &mut FnBuilder, name: &str, ty: DType, off: u32, size: u32) {
        if self.declared.insert(off) {
            b.arg(name, ty, off, size);
        }
    }
    /// Declares and loads a 32-bit scalar.
    pub fn i32(&mut self, b: &mut FnBuilder, name: &str, off: u32) -> ValueId {
        self.arg(b, name, DType::I32, off, 4);
        if let Some(v) = self.i32s.get(&off) {
            return *v;
        }
        let v = b.let_(Op::KernargI32 { offset: off });
        self.i32s.insert(off, v);
        v
    }
    /// Declares and loads a pointer.
    pub fn ptr(&mut self, b: &mut FnBuilder, name: &str, off: u32) -> ValueId {
        self.arg(b, name, DType::Ptr, off, 8);
        if let Some(v) = self.ptrs.get(&off) {
            return *v;
        }
        let v = b.let_(Op::KernargPtr { offset: off });
        self.ptrs.insert(off, v);
        v
    }
    /// A runtime dimension read from the kernarg scalar at `off`, with the host contract `lo <= value <= hi` (`i64::MAX` = unbounded).
    pub fn dim(&mut self, b: &mut FnBuilder, name: &str, off: u32, lo: i64, hi: i64) -> Dim {
        self.arg(b, name, DType::I32, off, 4);
        b.ka_range(off, lo, hi);
        let val = self.i32(b, name, off);
        Dim::Ka { off, val }
    }
    /// A tensor declaration (see [`TensorDecl`]); nothing is emitted until `declare`.
    pub fn tensor(&mut self, name: &str, ptr_off: u32, elem_bytes: i64, dims: Vec<Dim>, strides: Vec<Stride>) -> TensorDecl {
        assert_eq!(dims.len(), strides.len(), "tensor {name}: one stride per dim");
        TensorDecl { name: name.to_string(), ptr_off, elem_bytes, dims, strides, packed: false, same_strides: None, alloc: None, slack: None, contract: true }
    }
}

/// A tensor dimension.
#[derive(Clone, Debug)]
pub enum Dim {
    /// a compile-time constant
    C(i64),
    /// a kernarg scalar (`Ext::ka(off)`), loaded as `val`
    Ka { off: u32, val: ValueId },
    /// a contract-only symbol (`Ext::sym`, e.g. the KV head count `HK` or the batch `B` of the split launch)
    Sym(String),
    /// a launch grid dimension (`Ext::Grid`, e.g. the batch = grid z)
    Grid(u8),
}

impl Dim {
    pub fn c(v: i64) -> Dim {
        Dim::C(v)
    }
    pub fn sym(name: &str) -> Dim {
        Dim::Sym(name.to_string())
    }
    pub fn grid(axis: u8) -> Dim {
        Dim::Grid(axis)
    }
    pub fn ext(&self) -> Ext {
        match self {
            Dim::C(c) => Ext::c(*c),
            Dim::Ka { off, .. } => Ext::ka(*off),
            Dim::Sym(s) => Ext::sym(s),
            Dim::Grid(a) => Ext::Grid(*a),
        }
    }
    /// The runtime value (constants and kernarg dims only).
    pub fn val(&self) -> Option<I32> {
        match self {
            Dim::C(c) => Some(I32::c(*c as i32)),
            Dim::Ka { val, .. } => Some(ev(*val)),
            _ => None,
        }
    }
}

/// A byte stride.
#[derive(Clone, Debug)]
pub enum Stride {
    /// the element size (innermost, contiguous dim)
    Unit,
    /// a kernarg scalar (name, offset)
    Ka(&'static str, u32),
    /// not a kernarg: the packed value implied by the inner dims (contract), computed at run time by the given expression
    Derived(I32),
}

/// A tensor declaration: pointer kernarg + dims + byte strides, plus how the host contract is generated.
pub struct TensorDecl {
    name: String,
    ptr_off: u32,
    elem_bytes: i64,
    dims: Vec<Dim>,
    strides: Vec<Stride>,
    packed: bool,
    same_strides: Option<Tensor>,
    alloc: Option<Ext>,
    slack: Option<Ext>,
    /// emit the host contract here (false: the caller derives it, e.g. from an `amdgpu_layout::Layout`)
    contract: bool,
}

/// The value of a stride in contract terms: a constant when known, else the kernarg itself, else (derived) the packed value.
#[derive(Clone, Debug)]
enum Cx {
    K(i64),
    E(Ext),
}

impl Cx {
    fn ext(&self) -> Ext {
        match self {
            Cx::K(k) => Ext::c(*k),
            Cx::E(e) => e.clone(),
        }
    }
    fn mul_dim(&self, d: &Dim) -> Cx {
        match (d, self) {
            (Dim::C(c), Cx::K(k)) => Cx::K(c * k),
            _ => Cx::E(d.ext() * self.ext()),
        }
    }
}

impl TensorDecl {
    /// Dense (BSHD-style) layout: every non-innermost stride equals the next dim times the next stride. Generates `ka_is(stride, dim * next stride)` for each
    /// kernarg stride, or `ka_range(stride, c, c)` when the packed value is the constant `c`.
    pub fn packed(mut self) -> TensorDecl {
        self.packed = true;
        self
    }
    /// Same strides as an already declared tensor (e.g. V shares the K strides): `ka_is(stride_i, ka(other stride_i))` / `ka_range` for constant ones.
    pub fn same_strides_as(mut self, other: &Tensor) -> TensorDecl {
        self.same_strides = Some(other.clone());
        self
    }
    /// Overrides the allocation size (default: outermost dim x outermost stride).
    pub fn alloc(mut self, bytes: Ext) -> TensorDecl {
        self.alloc = Some(bytes);
        self
    }
    /// Declares the kernargs and loads only: the host contract (`arg_tensor`, `ka_is`, `ka_range`) is emitted by the caller.
    pub fn external_contract(mut self) -> TensorDecl {
        self.contract = false;
        self
    }
    /// The host allocates `extra` zeroed bytes after the logical tensor: reads may use them, writes may not (`arg_tensor_slack`).
    pub fn slack(mut self, extra: Ext) -> TensorDecl {
        self.slack = Some(extra);
        self
    }

    /// Emits the kernarg declarations + loads and the contract; returns the runtime tensor.
    pub fn declare(self, ka: &mut KArgs, b: &mut FnBuilder) -> Tensor {
        let n = self.dims.len();
        let ptr = ka.ptr(b, &self.name, self.ptr_off);
        // contract value of every stride (innermost first)
        let mut cx: Vec<Cx> = vec![Cx::K(0); n];
        let mut packed_v: Vec<Cx> = vec![Cx::K(0); n];
        for i in (0..n).rev() {
            let pv = if i == n - 1 { Cx::K(self.elem_bytes) } else { cx[i + 1].mul_dim(&self.dims[i + 1]) };
            packed_v[i] = pv.clone();
            cx[i] = match &self.strides[i] {
                Stride::Unit => Cx::K(self.elem_bytes),
                Stride::Derived(_) => pv,
                Stride::Ka(_, off) => match (&pv, self.packed) {
                    (Cx::K(k), true) => Cx::K(*k),
                    _ => Cx::E(Ext::ka(*off)),
                },
            };
        }
        // loads in declaration order
        let mut vals: Vec<I32> = Vec::with_capacity(n);
        for s in &self.strides {
            match s {
                Stride::Unit => vals.push(I32::c(self.elem_bytes as i32)),
                Stride::Derived(e) => vals.push(e.clone()),
                Stride::Ka(sname, off) => vals.push(ev(ka.i32(b, sname, *off))),
            }
        }
        // contract relations innermost first (a relation refers only to inner strides: the definitions are emitted in dependency order)
        for i in (0..n).rev() {
            let Stride::Ka(_, off) = &self.strides[i] else { continue };
            if !self.contract {
                continue;
            }
            if self.packed && self.same_strides.is_none() {
                match &packed_v[i] {
                    Cx::K(k) => b.ka_range(*off, *k, *k),
                    Cx::E(e) => b.ka_is(*off, e.clone()),
                }
            }
            if let Some(o) = &self.same_strides {
                match (&o.contract_strides[i], o.stride_offs[i]) {
                    (Cx::K(k), _) => b.ka_range(*off, *k, *k),
                    (_, Some(ooff)) => b.ka_is(*off, Ext::ka(ooff)),
                    (Cx::E(e), None) => b.ka_is(*off, e.clone()),
                }
            }
        }
        let alloc = self.alloc.clone().unwrap_or_else(|| cx[0].mul_dim(&self.dims[0]).ext());
        match (&self.slack, self.contract) {
            (_, false) => {}
            (None, true) => b.arg_tensor(&self.name, alloc),
            (Some(extra), true) => b.arg_tensor_slack(&self.name, alloc.clone(), alloc + extra.clone()),
        }
        let stride_offs = self.strides.iter().map(|s| if let Stride::Ka(_, o) = s { Some(*o) } else { None }).collect();
        Tensor { name: self.name, ptr, elem_bytes: self.elem_bytes, dims: self.dims, strides: vals, contract_strides: cx, stride_offs }
    }
}

/// A declared tensor: the loaded pointer and the runtime byte strides.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: String,
    pub ptr: ValueId,
    pub elem_bytes: i64,
    pub dims: Vec<Dim>,
    /// runtime byte stride of every dim
    pub strides: Vec<I32>,
    contract_strides: Vec<Cx>,
    stride_offs: Vec<Option<u32>>,
}

fn mul_stride(i: &I32, s: &I32) -> I32 {
    match &*s.0 {
        Node::CI(1) => i.clone(),
        _ => i * s,
    }
}

impl Tensor {
    /// Byte offset of the element at `idx` (literal-zero indices contribute nothing; a unit byte stride is not multiplied).
    pub fn offset(&self, idx: &[I32]) -> I32 {
        assert_eq!(idx.len(), self.strides.len(), "tensor {}: index rank", self.name);
        let mut acc: Option<I32> = None;
        for (i, s) in idx.iter().zip(self.strides.iter()) {
            if is_zero(i) {
                continue;
            }
            let t = mul_stride(i, s);
            acc = Some(match acc {
                None => t,
                Some(a) => a + t,
            });
        }
        acc.unwrap_or_else(|| I32::c(0))
    }
    /// A tile view anchored at `origin` (one coordinate per dim).
    pub fn tile(&self, origin: Vec<I32>) -> TileView {
        assert_eq!(origin.len(), self.dims.len(), "tensor {}: origin rank", self.name);
        let n = origin.len();
        TileView { t: self.clone(), origin, clamps: vec![None; n] }
    }
}

/// How an out-of-range tile index is made safe: it is replaced by the last valid index (the rows of a partial tile recompute / re-read the last valid row
/// instead of touching memory past the tensor).
#[derive(Clone, Debug)]
pub struct Clamp {
    /// valid relative indices are `< limit`
    pub limit: I32,
    /// the replacement index (`limit - 1`, usually pre-bound once)
    pub last: I32,
    /// `min(idx, last)` (one `v_min_i32`) instead of `select(idx < limit, idx, last)`
    pub use_min: bool,
}

/// A window of a tensor: origin coordinates plus per-dim clamps applied to the RELATIVE index.
#[derive(Clone, Debug)]
pub struct TileView {
    pub t: Tensor,
    pub origin: Vec<I32>,
    pub clamps: Vec<Option<Clamp>>,
}

impl TileView {
    /// Clamp the relative index of `dim` to `[0, limit)` (`last` = limit - 1): `select(i < limit, i, last)`.
    pub fn clamp(mut self, dim: usize, limit: I32, last: I32) -> TileView {
        self.clamps[dim] = Some(Clamp { limit, last, use_min: false });
        self
    }
    /// The clamped relative index of `dim`.
    pub fn clamped(&self, dim: usize, rel: I32) -> I32 {
        match &self.clamps[dim] {
            None => rel,
            Some(c) if c.use_min => rel.min(&c.last),
            Some(c) => rel.lt(&c.limit).select_i32(rel.clone(), c.last.clone()),
        }
    }
    /// Absolute coordinates of the relative index `rel` (clamped).
    pub fn coords(&self, rel: &[I32]) -> Vec<I32> {
        assert_eq!(rel.len(), self.origin.len());
        (0..rel.len())
            .map(|d| {
                let r = self.clamped(d, rel[d].clone());
                if is_zero(&self.origin[d]) {
                    r
                } else if is_zero(&r) {
                    self.origin[d].clone()
                } else {
                    self.origin[d].clone() + r
                }
            })
            .collect()
    }
    /// Byte offset of the relative index `rel`.
    pub fn offset(&self, rel: &[I32]) -> I32 {
        self.t.offset(&self.coords(rel))
    }
    /// Emits the pointer of the relative index `rel` (`ptr_add(base, offset)`).
    pub fn ptr(&self, b: &mut FnBuilder, rel: &[I32]) -> ValueId {
        let off = b.bind_i32(self.offset(rel));
        b.ptr_add(self.t.ptr, off)
    }
}
