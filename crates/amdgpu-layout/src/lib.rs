//! `amdgpu-layout`: CuTe / FlyDSL-style layout algebra for gfx950 kernels (docs/dsl-layout-refactor-design.md §2).
//!
//! * [`Ix`] / [`Dyns`]: symbolic integers - static constants or dynamic symbols with a range and a bit support;
//! * [`IntTuple`], [`Layout`]: nested shapes / strides; `crd2idx`, `idx2crd`, `size`, `cosize`, `rank`, `depth`;
//! * the algebra: `composition`, `complement`, `coalesce`, `right_inverse`, `left_inverse`, logical / zipped / tiled / flat / blocked / raked products,
//!   logical / zipped / tiled / flat divides (by layout, and by mode with dynamic outer modes), `local_tile`, `slice`, `dice`, `select`, `group`, `append`,
//!   `prepend`, `recast_layout`, `make_layout_tv`;
//! * [`Swizzle`] (GF(2)-linear; CuTe `Swizzle<B,M,S>` and the FMHA 128-byte-row swizzle) and [`ComposedLayout`] (`Swizzle o Layout`);
//! * [`IssuePlan`] / [`layout_to_offsets`]: the OPUS thread-dims x issue-dims split, per-issue offsets, base + immediate folding;
//! * `proofs`: coverage, injectivity, contiguity / vector width, `ds_read_b64_tr_b8` source validity, MFMA operand match (with a K permutation),
//!   swizzle bijectivity, LDS bank conflicts.
//!
//! Pure Rust, no dependencies; everything is host-testable.

#![forbid(unsafe_code)]

pub mod ix;
pub mod layout;
pub mod proofs;
pub mod swizzle;
pub mod tuple;
pub mod tv;

pub use ix::{DynInfo, Dyns, Ix};
pub use layout::*;
pub use swizzle::{ComposedLayout, Swizzle};
pub use tuple::{leaf, IntTuple};
pub use tv::{layout_to_offsets, ImmRange, IssuePlan, OffsetPlan};
