//! Safe-Rust authoring of gfx950 instruction sequences.
//!
//! ```no_run
//! use amdgpu_safeasm::*;
//! let mut s = Seq::new();
//! let mut x = s.splat(3);
//! let y = s.splat(4);
//! s.v_add_u32(&mut x, &y, &y);           // v_add_u32 x, y, y
//! let mut sg = s.sgpr::<1>();
//! let r = s.ready(&x);                  // pads the 2 wait states a v_readfirstlane needs
//! s.v_readfirstlane_b32(&mut sg, r);
//! println!("{}", s.to_asm().unwrap());
//! ```
//!
//! See `docs/safe-asm-design.md` for the rules the types enforce and what they cannot.
#![forbid(unsafe_code)]

pub mod ops;
pub mod regs;
pub mod seq;

pub use regs::*;
pub use seq::{Free, Live, Mfma, Seq, Smfmac, Vcc};