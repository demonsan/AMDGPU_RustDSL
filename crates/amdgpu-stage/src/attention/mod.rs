//! The attention family bodies as L2 templates (docs/l2-family-bodies.md): the K / V ring with layout-derived copies and readers (`attn3`: `KvRing3`, `AttnBody3`), the block setup and the epilogues
//! (`block3`: `Operands`, `LaunchAbi`, `block3*`, `store_out3*`, `store_partial3`) and the per-block state (`state`). The kernel crate keeps `FmhaIo` / `declare_io` (the ABI), the family skeleton slots and the registry.

pub mod attn3;
pub mod block3;
pub mod state;
