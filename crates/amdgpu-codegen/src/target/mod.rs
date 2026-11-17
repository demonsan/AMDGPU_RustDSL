//! The compilation target (docs/target-design.md). P0 has one target: every pass reads gfx950's facts through `T`; a second target
//! turns `T` into the compilation's target (carried by the `MFunction`). The per-instruction tables that name `MOp` live under
//! `target/<name>/`.

pub use amdgpu_target::Target;

pub mod gfx950;
pub mod gfx1250;

/// The gfx950 target (the passes that are not yet target-parameterized read it directly; gfx950 output must not change).
pub const T: &Target = &amdgpu_target::gfx950::GFX950;

thread_local! {
    static CUR: std::cell::Cell<&'static Target> = const { std::cell::Cell::new(&amdgpu_target::gfx950::GFX950) };
}

/// The target of the compilation running on this thread (gfx950 unless `with_target` selected another).
pub fn cur() -> &'static Target {
    CUR.with(|c| c.get())
}

/// Runs `f` with `t` as this thread's compilation target (`cur`), restoring the previous one after.
pub fn with_target<R>(t: &'static Target, f: impl FnOnce() -> R) -> R {
    let prev = CUR.with(|c| c.replace(t));
    let r = f();
    CUR.with(|c| c.set(prev));
    r
}

/// AGPRs per wave (0 without an AGPR segment).
pub const fn agpr_count(t: &Target) -> u32 {
    match &t.regs.agpr {
        Some(a) => a.count,
        None => 0,
    }
}

/// The unified VGPR + AGPR file (gfx950: 512; without one: the VGPR count).
pub const fn unified_total(t: &Target) -> u32 {
    match &t.regs.unified {
        Some(u) => u.total,
        None => t.regs.vgpr.count,
    }
}
