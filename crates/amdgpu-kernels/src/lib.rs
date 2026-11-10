//! Kernels written only on the route-B layers (docs/dsl-layout-refactor-design.md): layouts derive every address, atoms + TiledCopy/TiledMma emit the
//! instructions, `amdgpu_dsl::tile` provides the LDS plan and the pipeline driver.

#![forbid(unsafe_code)]

pub mod fmha;
pub mod gemm;
pub mod mlp;
pub mod moe;
pub mod cache;
pub mod conv;
pub mod dequant;
pub mod eltwise;
pub mod norm;
pub mod rope;
pub mod lds_decl;
pub mod sampling;
pub mod topk;
pub mod rows;
pub mod ir_proof;
pub mod probes;
pub mod verifier_cases;
pub mod registry;
pub mod reachable;

pub use registry::{kernel_by_name, kernel_names};
