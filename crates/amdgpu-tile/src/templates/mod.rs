//! L2 pipeline templates (docs/l2-templates.md) extracted from hand-written kernel code. Kept in one directory so the crate split
//! (mdgpu-stage, docs/crate-split.md M7) can move it as a unit.

pub mod pair_ring;
pub mod split;
