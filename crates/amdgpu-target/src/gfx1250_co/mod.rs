//! In-process gfx1250 encoder (docs/gfx1250-plan.md G1): `enc` encodes the gfx1250 emitter's text, `tables` is generated from the gfx1250 llvm-mc oracle.

pub mod enc;
pub mod tables;
pub use enc::{encode_line, encode_text};
