//! Tile abstraction layer (docs/dsl-tile-abstractions.md): tensor views, shared (LDS) buffers, copies and the pipeline driver, register fragments with
//! layouts, fragment reductions and the FMHA library components. Everything here is ordinary safe Rust over the UNCHANGED IR: every component emits
//! `FnBuilder` statements (mostly through the `expr` layer) and returns plain values / variables, so a hand-written kernel can adopt one component at a time.
//!
//! What the kernel author no longer writes by hand, and where it is generated:
//! * kernarg declarations and the host contract (`arg_tensor`, `ka_is`, `ka_range`) - `tensor::KArgs` / `tensor::TensorDecl::declare`;
//! * row clamps of tile accesses - `tensor::TileView`;
//! * LDS offsets, `lds_ring/lds_tile/lds_private/lds_alias` and the total segment size - `shared::LdsPlan::commit`;
//! * `tile_produce/tile_consume/tile_end` markers with their (slot, generation) - `shared::SlotRef` / `shared::PrivRef`, the generation coming from `pipeline::TileIt`;
//! * the global->LDS DMA lane addressing (swizzle, M0, negative immediates) - `copy::RowsCopy` / `copy::DwordRowsCopy`;
//! * waits, barriers, the role phase offset, the tile counter contract and the peeled tail - `pipeline::Pipeline::run`.

pub mod copy;
pub mod epilogue;
pub mod frag;
pub mod lane;
pub mod pipeline;
pub mod reduce;
pub mod shared;
pub mod softmax;
pub mod tensor;

pub use copy::{DwordRowsCopy, RowSource, RowsCopy};
pub use lane::{versioned, Lane, Role, Roles};
pub use pipeline::{Cadence, PipeBody, Pipeline, TileIt};
pub use shared::{fmha_swz, LdsPlan, Layout, PrivRef, Shared, SlotIdx, SlotRef, Swz};
pub use tensor::{Clamp, Dim, KArgs, Stride, Tensor, TensorDecl, TileView};

use crate::expr::Node;
use crate::{Opnd, ValueId, VarId, F32, I32};
use std::rc::Rc;

/// An already-bound i32 value as an expression leaf.
pub fn ev(v: ValueId) -> I32 {
    I32(Rc::new(Node::Val(v)))
}
/// An already-bound f32 value as an expression leaf.
pub fn fv(v: ValueId) -> F32 {
    F32(Rc::new(Node::Val(v)))
}
/// A loop variable as an i32 expression leaf (lowered to an `Opnd::Var` operand).
pub fn varx(v: VarId) -> I32 {
    I32(Rc::new(Node::Var(v)))
}
/// An operand (value, variable or i32 immediate) as an i32 expression.
pub fn ox(o: Opnd) -> I32 {
    match o {
        Opnd::Val(v) => ev(v),
        Opnd::Var(v) => varx(v),
        Opnd::I32(c) => I32::c(c),
        Opnd::F32(_) => panic!("tile::ox: an f32 immediate is not an i32 expression"),
    }
}

/// Is the expression the literal 0?
pub(crate) fn is_zero(e: &I32) -> bool {
    matches!(&*e.0, Node::CI(0))
}
/// `a + b`, or `a` when `b` is the literal 0 (no `add` emitted).
pub(crate) fn add0(a: I32, b: &I32) -> I32 {
    if is_zero(b) {
        a
    } else {
        a + b
    }
}