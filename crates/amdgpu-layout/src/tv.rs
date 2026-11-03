//! Thread-value layouts, the OPUS-style issue space and offset plans with immediate folding.
//!
//! A TV layout maps `(thread, value)` to the linear (column-major) element index of a tile. An access by one atom issues `vec` consecutive values per
//! instruction; the THREAD part of an offset is dynamic (lane / wave ids, computed once per kernel), the ISSUE part is static (OPUS's per-thread "y" dims).
//! `layout_to_offsets` turns the per-issue offsets into a small set of base expressions + per-issue immediates:
//! * linear layouts: the static part goes into the immediate when it fits the instruction's immediate field;
//! * swizzled layouts: `swz(a + c) = swz(a) ^ c'` (`c' = c ^ M c[src]`, linearity, when the bits of `a` and `c` are disjoint); the bits of `c'` that are
//!   disjoint from every bit `swz(a)` may set fold into the immediate (XOR = ADD on disjoint bits), the others stay in the base. This reproduces the
//!   four K-read bases of the hand `_vs128` (`kfx[0..3]`: the static chunk XOR `2k` collides with the swizzled chunk bits) and folds what the hand code
//!   still adds with VALU (`4096 s`, `2048 r`, `8192 hh`).

use crate::ix::{Dyns, Ix};
use crate::layout::{composition, Layout};
use crate::swizzle::Swizzle;
use crate::tuple::IntTuple;

/// Immediate offset field of an instruction (bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImmRange {
    pub lo: i64,
    pub hi: i64,
    pub align: i64,
}

impl ImmRange {
    /// `ds_read*/ds_write*`: unsigned 16-bit byte offset.
    pub const DS: ImmRange = ImmRange { lo: 0, hi: 65535, align: 1 };
    /// `global_load/store*` (and `global_load_lds`) on gfx950: signed 13-bit.
    pub const GLOBAL: ImmRange = ImmRange { lo: -4096, hi: 4095, align: 1 };
    /// `buffer_load/store*` (MUBUF): unsigned 12-bit.
    pub const BUFFER: ImmRange = ImmRange { lo: 0, hi: 4095, align: 1 };
    /// No immediate field: everything stays in the base.
    pub const NONE: ImmRange = ImmRange { lo: 0, hi: 0, align: 1 };
    pub fn fits(&self, k: i64) -> bool {
        k >= self.lo && k <= self.hi && k % self.align == 0
    }
}

/// The issue space of one atom over a TV layout: thread mode 0, value mode 1, `vec` values per instruction.
#[derive(Clone, Debug)]
pub struct IssuePlan {
    pub tv: Layout,
    pub vec: i64,
}

impl IssuePlan {
    pub fn new(tv: Layout, vec: i64) -> IssuePlan {
        let v = tv.get(1).size().st();
        assert!(vec > 0 && v % vec == 0, "IssuePlan: {v} values per thread are not a multiple of the vector width {vec}");
        IssuePlan { tv, vec }
    }
    pub fn threads(&self) -> i64 {
        self.tv.get(0).size().st()
    }
    pub fn values(&self) -> i64 {
        self.tv.get(1).size().st()
    }
    pub fn issues(&self) -> i64 {
        self.values() / self.vec
    }
    /// The per-issue offsets of thread `lane` (symbolic) for a destination layout `dst` (tile coordinates -> BYTE offset, static shapes; its strides
    /// may be dynamic, e.g. a global tensor). `extra` is added to every offset (a static tile position).
    pub fn offsets(&self, dst: &Layout, lane: &Ix, extra: &Ix, d: &Dyns) -> Vec<Ix> {
        let comp = composition(dst, &self.tv);
        (0..self.issues())
            .map(|i| comp.crd2idx(&IntTuple::T(vec![IntTuple::L(lane.clone()), IntTuple::L(Ix::C(i * self.vec))]), d).add(extra))
            .collect()
    }
}

/// Bases + per-issue (base index, immediate).
#[derive(Clone, Debug, PartialEq)]
pub struct OffsetPlan {
    pub bases: Vec<Ix>,
    pub issues: Vec<(usize, i64)>,
}

impl OffsetPlan {
    pub fn eval(&self, issue: usize, env: &dyn Fn(u32) -> i64) -> i64 {
        let (b, imm) = self.issues[issue];
        self.bases[b].eval(env) + imm
    }
}

/// Splits per-issue byte offsets into bases and immediates (see the module doc). `swz` is applied to every offset (use `Swizzle::none()` for none).
pub fn layout_to_offsets(exprs: &[Ix], swz: &Swizzle, imm: ImmRange, d: &Dyns) -> OffsetPlan {
    let mut bases: Vec<Ix> = Vec::new();
    let mut issues: Vec<(usize, i64)> = Vec::with_capacity(exprs.len());
    let push = |b: Ix, k: i64, bases: &mut Vec<Ix>| -> (usize, i64) {
        let (b, k) = if imm.fits(k) { (b, k) } else { (b.add_c(k), 0) };
        let i = match bases.iter().position(|x| *x == b) {
            Some(i) => i,
            None => {
                bases.push(b);
                bases.len() - 1
            }
        };
        (i, k)
    };
    for e in exprs {
        let (a, c) = e.split_const();
        if swz.is_identity() {
            issues.push(push(a, c, &mut bases));
            continue;
        }
        let ab = a.bits(d);
        if c < 0 || ab == u64::MAX || (ab & c as u64) != 0 {
            // no disjointness: the whole swizzled offset is a base
            issues.push(push(swz.apply_ix(e, d), 0, &mut bases));
            continue;
        }
        let c2 = c ^ swz.term(c);
        let mask = ab | swz.dst_mask();
        let cx = (c2 as u64 & mask) as i64;
        let cf = (c2 as u64 & !mask) as i64;
        let base = swz.apply_ix(&a, d).xor(&Ix::C(cx));
        issues.push(push(base, cf, &mut bases));
    }
    OffsetPlan { bases, issues }
}
