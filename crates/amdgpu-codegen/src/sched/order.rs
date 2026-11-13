//! Ordering constraints that are not register dependences: memory, fences, `s_setprio`.
//! Shared by the dependence-graph builder and the schedule verifier.

use super::effects::*;

/// May the two accesses touch the same bytes? Same base register in the same value version gives
/// exact byte-range answers; anything else is conservatively "yes".
fn may_alias(a: &MemAccess, va: Option<usize>, b: &MemAccess, vb: Option<usize>) -> bool {
    if a.space != b.space {
        return false;
    }
    if let (Some(x), Some(y)) = (a.base, b.base) {
        if x == y && va == vb {
            return a.off < b.off + b.size && b.off < a.off + a.size;
        }
    }
    true
}

fn fence_then(f: Fence, x: &Effects, alu_across: bool) -> bool {
    match f {
        Fence::SetPrio => true,
        Fence::Barrier | Fence::Nop => !alu_across || x.mem.is_some(),
        // An explicit wait only constrains what was issued before it.
        Fence::WaitLgkm(_) | Fence::WaitVm(_) | Fence::WaitBoth(..) => false,
        Fence::None => false,
    }
}

fn then_fence(x: &Effects, f: Fence, alu_across: bool) -> bool {
    match f {
        Fence::SetPrio => true,
        Fence::Barrier | Fence::Nop => !alu_across || x.mem.is_some(),
        Fence::WaitLgkm(_) => x.smem || matches!(&x.mem, Some(m) if m.space == Space::Lds),
        Fence::WaitVm(_) => matches!(&x.mem, Some(m) if m.space == Space::Global),
        Fence::WaitBoth(..) => x.smem || x.mem.is_some(),
        Fence::None => false,
    }
}

// `slotfill` legality contract (sched/slotfill.rs): inside a barrier interval an LDS-DMA never writes what a `ds_read` of the same interval reads, so the two may swap.
// Set only for the duration of that pass (and its verification).
thread_local! {
    static RELAX_DMA_LDS_READ: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Thread-local (the compiler runs concurrently in tests and in multi-kernel builds): only the thread inside `slotfill::run` sees the relaxed rules.
pub fn set_relax_dma_lds_read(on: bool) {
    RELAX_DMA_LDS_READ.with(|c| c.set(on));
}

pub fn relax_dma_lds_read() -> bool {
    RELAX_DMA_LDS_READ.with(|c| c.get())
}

fn dma_vs_lds_read(f: &Effects, x: &Effects) -> bool {
    f.fence == Fence::Barrier && f.mem.is_some() && x.fence == Fence::None && matches!(&x.mem, Some(m) if m.space == Space::Lds && !m.write)
}

pub fn mem_must_order(a: &Effects, va: Option<usize>, b: &Effects, vb: Option<usize>, alu_across: bool) -> bool {
    if relax_dma_lds_read() && (dma_vs_lds_read(a, b) || dma_vs_lds_read(b, a)) {
        return false;
    }
    match (a.fence != Fence::None, b.fence != Fence::None) {
        (true, true) => true,
        (true, false) => fence_then(a.fence, b, alu_across),
        (false, true) => then_fence(a, b.fence, alu_across),
        (false, false) => match (&a.mem, &b.mem) {
            (Some(x), Some(y)) => (x.write || y.write) && may_alias(x, va, y, vb),
            _ => false,
        },
    }
}