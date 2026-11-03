//! L1 on the layout algebra: the host contract of a global tensor DERIVED from its `amdgpu_layout::Layout` (bytes; row-major dims, the leaves are contract
//! symbols - kernargs, the grid, named host symbols - or constants).
//! * allocation = `cosize(layout)`; for a compact layout `cosize = shape_0 * stride_0` (every stride is the product of the inner extents), which is the
//!   form the verifier's substitution uses;
//! * compactness: `stride_i = shape_{i+1} * stride_{i+1}` for every kernarg stride (`ka_is`, or `ka_range(k, k)` when the stride is a constant of the
//!   layout), emitted innermost first (dependency order);
//! * a tensor sharing another's strides: `ka_is(stride_i, ka(other_i))` per kernarg stride.

use crate::Ctx;
use amdgpu_dsl::{Ext, FnBuilder};
use amdgpu_layout::{Ix, Layout};

/// The flat (shape, stride) modes of a row-major layout.
fn modes(l: &Layout) -> Vec<(Ix, Ix)> {
    l.flat_modes()
}

/// Emits the contract of a compact row-major tensor. `stride_ka[i]` = the kernarg holding stride i (None: not a kernarg). `slack` = zeroed bytes the host
/// allocates behind the tensor (`arg_tensor_slack`).
pub fn compact_tensor(b: &mut FnBuilder, ctx: &Ctx, name: &str, l: &Layout, stride_ka: &[Option<u32>], slack: Option<Ext>) {
    let m = modes(l);
    assert_eq!(m.len(), stride_ka.len(), "compact_tensor({name}): one kernarg entry per mode");
    for i in (0..m.len().saturating_sub(1)).rev() {
        let Some(off) = stride_ka[i] else { continue };
        let packed = m[i + 1].0.mul(&m[i + 1].1);
        match (m[i].1.c(), packed.c()) {
            (Some(k), Some(p)) => {
                assert_eq!(k, p, "compact_tensor({name}): constant stride {k} of mode {i} is not the inner extent {p}");
                b.ka_range(off, k, k);
            }
            (Some(k), None) => panic!("compact_tensor({name}): mode {i} has the constant stride {k} but a dynamic inner extent {packed}"),
            _ => b.ka_is(off, ctx.ext(&packed)),
        }
    }
    let alloc = ctx.ext(&m[0].0.mul(&m[0].1));
    match slack {
        None => b.arg_tensor(name, alloc),
        Some(extra) => b.arg_tensor_slack(name, alloc.clone(), alloc + extra),
    }
}

/// Emits the contract of a tensor whose kernarg strides equal those of `other` (`other_ka`), e.g. V sharing the K strides.
pub fn same_strides(b: &mut FnBuilder, ctx: &Ctx, name: &str, l: &Layout, stride_ka: &[Option<u32>], other: &Layout, other_ka: &[Option<u32>]) {
    let (m, o) = (modes(l), modes(other));
    for i in (0..m.len()).rev() {
        let Some(off) = stride_ka[i] else { continue };
        match (o[i].1.c(), other_ka[i]) {
            (Some(k), _) => b.ka_range(off, k, k),
            (None, Some(ooff)) => b.ka_is(off, Ext::ka(ooff)),
            (None, None) => b.ka_is(off, ctx.ext(&o[i].1)),
        }
    }
    b.arg_tensor(name, ctx.ext(&m[0].0.mul(&m[0].1)));
}
