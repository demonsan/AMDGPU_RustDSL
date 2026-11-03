//! `amdgpu-tile`: the tile layer on the layout algebra (docs/dsl-layout-refactor-design.md §3).
//!
//! * [`Ctx`]: the dynamic symbols of a kernel build (lane, wave, ...) with their IR values - layout expressions ([`amdgpu_layout::Ix`]) are lowered to
//!   `amdgpu_dsl::I32` expressions with the IR's shift/and forms for powers of two;
//! * [`TiledCopy`] / [`make_tiled_copy`] / [`make_tiled_copy_a`] / [`make_tiled_copy_b`]: an atom + a thread-value layout; `partition_lds` derives the
//!   per-lane base registers (bound ONCE per kernel, like the hand readers' constructors) and per-operation immediates from the LDS view layout
//!   (`Swizzle o Layout`), `at` adds the slot offset once per tile, `read` emits the atom's instructions under the slot's read marker;
//! * [`TiledMma`] / [`gemm`]: an MMA atom (optionally with swapped A/B) and the in-place accumulate.
//!
//! The FMHA readers of `amdgpu_dsl::tile::frag` (`RowsReader`, `TrReader`) are reproduced address for address by `TiledCopy` (tests/address_equality.rs).

#![forbid(unsafe_code)]

pub mod bufdma;
pub mod copy;
pub mod dma;
pub mod fmha;
pub mod mma;

pub use copy::{make_tiled_copy, make_tiled_copy_a, make_tiled_copy_b, LdsPart, LdsSlotPart, TiledCopy};
pub use bufdma::{partition_buffer_dma, BufDmaPart};
pub use dma::{partition_dma, DmaPart};
pub use mma::{gemm, make_tiled_mma, TiledMma};

use amdgpu_dsl::I32;
use amdgpu_layout::{Dyns, Ix};

/// The dynamic symbols of one kernel build and their IR values.
#[derive(Default)]
pub struct Ctx {
    pub dyns: Dyns,
    vals: Vec<I32>,
    /// bound subexpressions (CSE across every expression bound through `bind`; valid at the scope of the first bind - the kernel's top level)
    memo: std::cell::RefCell<std::collections::HashMap<Ix, amdgpu_dsl::ValueId>>,
}

impl Ctx {
    pub fn new() -> Ctx {
        Ctx::default()
    }
    /// A dynamic symbol with range `[lo, hi]` standing for the IR value `v`.
    pub fn sym(&mut self, name: &str, lo: i64, hi: i64, v: I32) -> Ix {
        let s = self.dyns.sym(name, lo, hi);
        self.vals.push(v);
        s
    }
    /// Binds a layout expression, sharing every compound subexpression already bound through this context (one IR value per distinct subterm).
    pub fn bind(&self, b: &mut amdgpu_dsl::FnBuilder, x: &Ix) -> I32 {
        match x {
            Ix::C(_) | Ix::Sym(_) => return self.lower(x),
            _ => {}
        }
        if let Some(v) = self.memo.borrow().get(x) {
            return amdgpu_dsl::tile::ev(*v);
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
        amdgpu_dsl::tile::ev(v)
    }
    /// The IR expression of a layout expression. Divisions and remainders must be by powers of two (shift / and); other constants multiply.
    pub fn lower(&self, x: &Ix) -> I32 {
        match x {
            Ix::C(k) => I32::c(*k as i32),
            Ix::Sym(s) => self.vals[*s as usize].clone(),
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
