//! `amdgpu-isa`: the instruction ATOMS of the tile layer per target (docs/dsl-layout-refactor-design.md §3, docs/target-design.md). P0 has one
//! target, re-exported at the crate root.

#![forbid(unsafe_code)]

pub mod gfx950;
/// gfx1250 (MI450) atoms; not re-exported (the shipped kernels target gfx950).
pub mod gfx1250;

pub use gfx950::*;
