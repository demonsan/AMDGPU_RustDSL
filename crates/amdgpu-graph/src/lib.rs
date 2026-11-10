//! L1 graph layer (docs/refactor-4layer-plan.md): kernel fusion / op graph and the dispatch rules.
//!
//! * [`opgraph`]: the op graph of a fused kernel family (nodes, classed edges, the cut into kernels, the on-chip budget), its JSON form, the review text and the skeleton generator
//!   (`amdc opgraph`, `amdc skeleton`). Moved from `amdgpu-tile::opgraph`; it still names the stage registry of `amdgpu-tile` (the L2 pieces move to `amdgpu-stage` later).
//! * [`dispatch`]: the host-side rules that pick a kernel form and its parameters for a launch shape (measured same-run against the vendor kernels). Moved from `amdgpu-kernels::dispatch`;
//!   the rules only return names and numbers, the tests that check the names are registered live in `amdgpu-kernels/tests/dispatch_compile.rs`.

pub mod dispatch;
pub mod opgraph;