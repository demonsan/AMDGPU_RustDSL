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
        Fence::WaitLgkm(_) | Fence::WaitVm(_) => false,
        Fence::None => false,
    }
}

fn then_fence(x: &Effects, f: Fence, alu_across: bool) -> bool {
    match f {
        Fence::SetPrio => true,
        Fence::Barrier | Fence::Nop => !alu_across || x.mem.is_some(),
        Fence::WaitLgkm(_) => x.smem || matches!(&x.mem, Some(m) if m.space == Space::Lds),
        // AMDC_PIN_VMWAIT=1: an explicit `s_waitcnt vmcnt` also stays after everything that precedes it in program order (the list scheduler would
        // otherwise hoist it above the independent MFMA work and re-serialise a deliberately deferred DMA wait)
        Fence::WaitVm(_) => pin_vm_wait() || matches!(&x.mem, Some(m) if m.space == Space::Global),
        Fence::None => false,
    }
}

/// Per-kernel switch (set by `compile_mir` for `_vs6`) or the env var `AMDC_PIN_VMWAIT=1`.
pub static PIN_VMWAIT_KERNEL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn pin_vm_wait() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    PIN_VMWAIT_KERNEL.load(std::sync::atomic::Ordering::Relaxed) || *V.get_or_init(|| std::env::var("AMDC_PIN_VMWAIT").map_or(false, |v| v == "1"))
}

/// Must `a` (earlier in program order) stay before `b` because of memory / fence semantics?
pub fn mem_must_order(a: &Effects, va: Option<usize>, b: &Effects, vb: Option<usize>, alu_across: bool) -> bool {
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