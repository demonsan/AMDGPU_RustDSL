//! L2 stage layer (docs/refactor-4layer-plan.md): pipeline composition over the L3 function layer `amdgpu-tile`.
//!
//! * [`stage`]: the stage instance / composition API and the stage registry (fusion stage library); the woven order comes from the Weaver (`amdgpu-codegen`).
//! * [`pipeline`]: the tile pipeline driver (`Cadence::{TwoPhaseOffset, OneWindow, RegStaged}`) with its body trait.
//! * [`contract_stage`]: producer / consumer fragment-layout contracts between stages.
//! * [`components`]: grid / split index maths shared by the attention families.
//! * [`templates`]: the L2 pipeline templates extracted from the kernel families (docs/l2-templates.md).
//!
//! Moved from `amdgpu-tile` (M7 of docs/crate-split.md); `TileIt` stays in `amdgpu_tile::lane` (L3 data).

pub mod components;
pub mod contract_stage;
pub mod gemm;
pub mod pipeline;
pub mod stage;
pub mod templates;

pub use pipeline::{Cadence, PipeBody, Pipeline};
