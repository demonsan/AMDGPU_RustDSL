//! The GEMM family body as an L2 template (docs/l2-family-bodies.md): the register-staged pipeline body (`body`), the block-scaled MMA helpers (`mma`) and the operand formats (`formats`).
//! The kernel crate keeps `GemmConfig` and its builders, the ABI / contract declarations, the epilogue, the skeleton slots and the registry.

pub mod body;
pub mod formats;
pub mod mma;

pub use formats::{Elem, Scales};
