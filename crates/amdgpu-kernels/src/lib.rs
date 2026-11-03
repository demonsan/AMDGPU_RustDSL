//! Kernels written only on the route-B layers (docs/dsl-layout-refactor-design.md): layouts derive every address, atoms + TiledCopy/TiledMma emit the
//! instructions, `amdgpu_dsl::tile` provides the LDS plan and the pipeline driver.

#![forbid(unsafe_code)]

pub mod gemm_mxfp8;
