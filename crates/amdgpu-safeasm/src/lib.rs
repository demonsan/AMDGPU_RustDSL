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
//! let block = s.finish().unwrap();      // verified plain data: the DSL embeds it (`Op::Asm`), the compiler lowers it to MIR
//! assert!(!block.insts().is_empty());
//! ```
//!
//! See `docs/safe-asm-design.md` for the rules the types enforce and what they cannot.
#![forbid(unsafe_code)]

pub mod block;
pub mod examples;
pub mod ops;
pub mod regs;
pub mod seq;
mod verify;

pub use block::*;
pub use regs::*;
pub use seq::{Free, Live, Mfma, Pending, Seq, Smfmac, Vcc, OP_MFMA_SCALE, OP_MFMA_SCALE_ZERO, OP_SMFMAC_FP8};
/// Verify an alternative instruction list (same registers and ports) with the block's rules: lets the compiler check that a rewrite of the
/// stream (constant folding) keeps every wait-state distance.
pub fn verify_insts(like: &AsmBlock, insts: Vec<AInst>) -> Result<(), Vec<String>> {
    AsmBlock { insts, ..like.clone() }.verify()
}

/// Test support: a block with the same registers but a different instruction list, NOT verified. It exists so the verifier can be tested against
/// hand-forged sequences; `AsmBlock::verify` (and the compiler's lowering) still reject whatever it produces if it is illegal.
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn forge_for_tests(like: &AsmBlock, insts: Vec<AInst>) -> AsmBlock {
    AsmBlock { insts, ..like.clone() }
}