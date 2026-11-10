//! `amdgpu-isa`: the instruction ATOMS of the tile layer per target (docs/dsl-layout-refactor-design.md §3, docs/target-design.md). P0 has one
//! target, re-exported at the crate root.

#![forbid(unsafe_code)]

pub mod gfx950;

pub use gfx950::*;
