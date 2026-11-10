//! `amdgpu-tile`: the tile layer on the layout algebra (docs/dsl-layout-refactor-design.md §3).
//!
//! * [`Ctx`]: the dynamic symbols of a kernel build (lane, wave, ...) with their IR values - layout expressions ([`amdgpu_layout::Ix`]) are lowered to
//!   `amdgpu_dsl::I32` expressions with the IR's shift/and forms for powers of two;
//! * [`TiledCopy`] / [`make_tiled_copy`] / [`make_tiled_copy_a`] / [`make_tiled_copy_b`]: an atom + a thread-value layout; `partition_lds` derives the
//!   per-lane base registers (bound ONCE per kernel, like the hand readers' constructors) and per-operation immediates from the LDS view layout
//!   (`Swizzle o Layout`), `at` adds the slot offset once per tile, `read` emits the atom's instructions under the slot's read marker;
//! * [`TiledMma`] / [`gemm`]: an MMA atom (optionally with swapped A/B) and the in-place accumulate.
//!
//! The FMHA readers of `crate::frag` (`RowsReader`, `TrReader`) are reproduced address for address by `TiledCopy` (tests/address_equality.rs).

#![forbid(unsafe_code)]

pub mod components;
pub mod epilogue;
pub mod frag;
pub mod lane;
pub mod pipeline;
pub mod reduce;
pub mod rows;
pub mod shared;
pub mod softmax;
pub mod tensor;
pub mod bufdma;
pub mod contract;
pub mod contract_stage;
pub mod copy;
pub mod dma;
pub mod global;
pub mod handoff;
pub mod orient;
pub mod rowsum;
pub mod mma;
pub mod weave;
pub mod weave_auto;
pub mod opgraph;
pub mod stage;
pub mod prims;
pub mod tileop;
pub mod templates;

pub use copy::{make_tiled_copy, make_tiled_copy_a, make_tiled_copy_b, make_tiled_copy_bcast, LdsPart, LdsPrivPart, LdsSlotPart, TiledCopy};
pub use global::GlobalPart;
pub use bufdma::{partition_buffer_dma, BufDmaPart};
pub use dma::{partition_dma, DmaPart};
pub use mma::{gemm, make_tiled_mma, TiledMma};

use amdgpu_dsl::I32;
use amdgpu_layout::{Dyns, Ix};

/// The dynamic symbols of one kernel build and their IR values.
#[derive(Default)]
pub struct Ctx {
    pub dyns: Dyns,
    vals: Vec<Option<I32>>,
    /// the contract expression of each symbol (host-contract generation), when it has one
    exts: Vec<Option<amdgpu_dsl::Ext>>,
    /// bound subexpressions (CSE across every expression bound through `bind`; valid at the scope of the first bind - the kernel's top level)
    memo: std::cell::RefCell<std::collections::HashMap<Ix, amdgpu_dsl::ValueId>>,
    /// lower one-symbol bit-field expressions through their normal form (`amdgpu_layout::BitForm`): off by default (keeps the FMHA v3 asm)
    bitfield: std::cell::Cell<bool>,
}

impl Ctx {
    pub fn new() -> Ctx {
        Ctx::default()
    }
    /// A dynamic symbol with range `[lo, hi]` standing for the IR value `v`.
    pub fn sym(&mut self, name: &str, lo: i64, hi: i64, v: I32) -> Ix {
        let s = self.dyns.sym(name, lo, hi);
        self.vals.push(Some(v));
        self.exts.push(None);
        s
    }
    /// A symbol with an IR value (`None`: contract-only, e.g. `HK`) and a contract expression (a kernarg, a grid dimension, a named symbol).
    pub fn sym_ext(&mut self, name: &str, lo: i64, hi: i64, v: Option<I32>, ext: amdgpu_dsl::Ext) -> Ix {
        let s = self.dyns.sym(name, lo, hi);
        self.vals.push(v);
        self.exts.push(Some(ext));
        s
    }
    /// The contract expression of a layout expression (sums, products, floor divisions by constants of contract symbols).
    pub fn ext(&self, x: &Ix) -> amdgpu_dsl::Ext {
        use amdgpu_dsl::Ext;
        match x {
            Ix::C(k) => Ext::c(*k),
            Ix::Sym(s) => self.exts[*s as usize].clone().unwrap_or_else(|| panic!("Ctx::ext: symbol {} has no contract expression", self.dyns.info(*s).name)),
            Ix::Sum(t, k) => {
                let mut acc: Option<Ext> = None;
                for (term, m) in t {
                    let e = if *m == 1 { self.ext(term) } else { self.ext(term) * Ext::c(*m) };
                    acc = Some(match acc {
                        None => e,
                        Some(a) => a + e,
                    });
                }
                let a = acc.expect("terms");
                if *k == 0 {
                    a
                } else {
                    a + Ext::c(*k)
                }
            }
            Ix::Mul(a, b) => self.ext(a) * self.ext(b),
            Ix::Div(a, k) => self.ext(a).floor_div(*k),
            other => panic!("Ctx::ext: {other} has no contract form"),
        }
    }
    /// Lower every expression that is a GF(2)-linear bit map of one symbol (lane / thread index math: pow2 divisions, remainders, masks, XOR
    /// swizzles, carry-free sums) through its bit-field normal form: XOR over shift groups `(x & M) << d`, when that takes fewer IR operations.
    pub fn set_bitfield(&self, on: bool) {
        self.bitfield.set(on);
    }
    /// The bit-field rewrite of `x`, when enabled and cheaper.
    fn bit_rewrite(&self, x: &Ix) -> Option<Ix> {
        if !self.bitfield.get() || matches!(x, Ix::C(_) | Ix::Sym(_)) {
            return None;
        }
        let bf = x.bitform(&self.dyns)?;
        let s = bf.sym?;
        if bf.cost() >= x.op_cost() {
            return None;
        }
        let full = if bf.width >= 32 { u32::MAX } else { (1u32 << bf.width) - 1 };
        let mut acc: Option<Ix> = None;
        for g in bf.groups() {
            let mut t = Ix::Sym(s);
            if g.mask != full {
                t = t.and_c(g.mask as i64);
            }
            if g.shift > 0 {
                t = t.shl(g.shift as u32);
            } else if g.shift < 0 {
                t = t.shr((-g.shift) as u32);
            }
            acc = Some(match acc {
                None => t,
                Some(a) => a.xor(&t),
            });
        }
        let mut e = acc.unwrap_or(Ix::C(0));
        if bf.konst != 0 {
            e = e.xor(&Ix::C(bf.konst as i64));
        }
        Some(e)
    }
    /// Binds a layout expression, sharing every compound subexpression already bound through this context (one IR value per distinct subterm).
    pub fn bind(&self, b: &mut amdgpu_dsl::FnBuilder, x: &Ix) -> I32 {
        match x {
            Ix::C(_) | Ix::Sym(_) => return self.lower(x),
            _ => {}
        }
        if let Some(v) = self.memo.borrow().get(x) {
            return amdgpu_dsl::ev(*v);
        }
        if let Some(r) = self.bit_rewrite(x) {
            let e = self.bind(b, &r);
            let v = b.bind_i32(e);
            self.memo.borrow_mut().insert(x.clone(), v);
            return amdgpu_dsl::ev(v);
        }
        let e = match x {
            Ix::Sum(t, k) => {
                let mut acc: Option<I32> = None;
                for (term, m) in t {
                    let v = self.bind(b, term);
                    let v = if *m == 1 {
                        v
                    } else if *m > 0 && (*m as u64).is_power_of_two() {
                        v << ((*m as u64).trailing_zeros() as i32)
                    } else {
                        v * (*m as i32)
                    };
                    acc = Some(match acc {
                        None => v,
                        Some(a) => a + v,
                    });
                }
                let a = acc.expect("a sum has terms");
                if *k == 0 {
                    a
                } else {
                    a + (*k as i32)
                }
            }
            Ix::Mul(a, c) => self.bind(b, a) * self.bind(b, c),
            Ix::Div(a, k) => {
                assert!((*k as u64).is_power_of_two(), "Ctx::bind: division by {k}");
                self.bind(b, a).lshr((*k as u64).trailing_zeros() as i32)
            }
            Ix::Mod(a, k) => {
                assert!((*k as u64).is_power_of_two(), "Ctx::bind: remainder by {k}");
                self.bind(b, a) & ((*k - 1) as i32)
            }
            Ix::Xor(a, c) => self.bind(b, a) ^ self.bind(b, c),
            Ix::And(a, m) => self.bind(b, a) & (*m as i32),
            Ix::Shl(a, s) => self.bind(b, a) << (*s as i32),
            Ix::Shr(a, s) => self.bind(b, a).lshr(*s as i32),
            Ix::C(_) | Ix::Sym(_) => unreachable!(),
        };
        let v = b.bind_i32(e);
        self.memo.borrow_mut().insert(x.clone(), v);
        amdgpu_dsl::ev(v)
    }
    /// The IR expression of a layout expression. Divisions and remainders must be by powers of two (shift / and); other constants multiply.
    pub fn lower(&self, x: &Ix) -> I32 {
        if let Some(r) = self.bit_rewrite(x) {
            return self.lower(&r);
        }
        match x {
            Ix::C(k) => I32::c(*k as i32),
            Ix::Sym(s) => self.vals[*s as usize].clone().unwrap_or_else(|| panic!("Ctx::lower: {} is a contract-only symbol", self.dyns.info(*s).name)),
            Ix::Sum(t, k) => {
                let mut acc: Option<I32> = None;
                for (term, m) in t {
                    let v = self.lower(term);
                    let v = if *m == 1 {
                        v
                    } else if *m > 0 && (*m as u64).is_power_of_two() {
                        v << ((*m as u64).trailing_zeros() as i32)
                    } else {
                        v * (*m as i32)
                    };
                    acc = Some(match acc {
                        None => v,
                        Some(a) => a + v,
                    });
                }
                let a = acc.expect("a sum has terms");
                if *k == 0 {
                    a
                } else {
                    a + (*k as i32)
                }
            }
            Ix::Mul(a, b) => self.lower(a) * self.lower(b),
            Ix::Div(a, k) => {
                assert!((*k as u64).is_power_of_two(), "Ctx::lower: division by {k} (only powers of two have an IR form)");
                self.lower(a).lshr((*k as u64).trailing_zeros() as i32)
            }
            Ix::Mod(a, k) => {
                assert!((*k as u64).is_power_of_two(), "Ctx::lower: remainder by {k} (only powers of two have an IR form)");
                self.lower(a) & ((*k - 1) as i32)
            }
            Ix::Xor(a, b) => self.lower(a) ^ self.lower(b),
            Ix::And(a, m) => self.lower(a) & (*m as i32),
            Ix::Shl(a, s) => self.lower(a) << (*s as i32),
            Ix::Shr(a, s) => self.lower(a).lshr(*s as i32),
        }
    }
}

// the tile components formerly in `amdgpu_dsl::tile`
pub use rows::{issue_rows_shared, DwordRowsCopy, DwordRowsVgpr, RowSource, RowsCopy};
pub use lane::{versioned, Lane, Role, Roles, TileIt};
pub use pipeline::{Cadence, PipeBody, Pipeline};
pub use shared::{fmha_swz, LdsPlan, Layout, PrivRef, Shared, SlotIdx, SlotRef, Swz};
pub use tensor::{Clamp, Dim, KArgs, Stride, Tensor, TensorDecl, TileView};

pub use amdgpu_dsl::{ev, ox, varx};

/// Is the expression the literal 0?

pub fn is_zero(e: &I32) -> bool {

    matches!(&*e.0, amdgpu_dsl::expr::Node::CI(0))

}

/// `a + b`, or `a` when `b` is the literal 0 (no `add` emitted).

pub fn add0(a: I32, b: &I32) -> I32 {

    if is_zero(b) {

        a

    } else {

        a + b

    }

}