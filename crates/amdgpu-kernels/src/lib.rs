//! Kernels written only on the route-B layers (docs/dsl-layout-refactor-design.md): layouts derive every address, atoms + TiledCopy/TiledMma emit the
//! instructions, `amdgpu_dsl::tile` provides the LDS plan and the pipeline driver.

#![forbid(unsafe_code)]

pub mod fmha;
pub mod gemm;
pub mod mlp;
pub mod cache;
pub mod eltwise;
pub mod norm;
pub mod rope;
pub mod topk;
pub mod rows;
pub mod ir_proof;
pub mod probes;
pub mod verifier_cases;
pub mod registry;

pub use registry::{kernel_by_name, kernel_names};
