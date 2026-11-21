//! The compilation target (docs/target-design.md). P0 has one target: every pass reads gfx950's facts through `T`; a second target
//! turns `T` into the compilation's target (carried by the `MFunction`). The per-instruction tables that name `MOp` live under
//! `target/<name>/`.

pub use amdgpu_target::{by_name, Target, TARGET_NAMES};

pub mod gfx950;
pub mod gfx1250;

/// The gfx950 target (the passes that are not yet target-parameterized read it directly; gfx950 output must not change).
pub const T: &Target = &amdgpu_target::gfx950::GFX950;

thread_local! {
    /// test guard: consulting the compile-time default target (`dflt`) while another target is current panics
    static T_GUARD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The compile-time default target (gfx950) for the code paths that are gfx950-only by construction (its emitter, the LLVM tail). A
/// shared pass must use `cur()`; under `with_default_target_guard` this panics when the current target is another one, which is how the
/// audit test finds a shared pass that still reads gfx950 facts during a gfx1250 compile.
#[track_caller]
pub fn dflt() -> &'static Target {
    if T_GUARD.with(|g| g.get()) && cur().name != T.name {
        panic!("the default target ({}) was consulted during a {} compile", T.name, cur().name);
    }
    T
}

/// The target a shared pass reads its facts from: the current one (`AMDC_T950_COMPAT`: the old behaviour, the compile-time default
/// gfx950, for A/B measurements of the audit's changes only).
pub fn tgt() -> &'static Target {
    if std::env::var("AMDC_T950_COMPAT").is_ok() {
        return T;
    }
    cur()
}

/// The scheduler's latency model target (AMDC_T950_LAT: the gfx950 latencies, A/B).
pub fn tgt_lat() -> &'static Target {
    if std::env::var("AMDC_T950_LAT").is_ok() {
        return T;
    }
    tgt()
}

/// A wait-state rule of the SHARED hazard pass (hazards.rs), in wait states. On gfx950 the target's rule. On gfx1250 those dependencies
/// interlock (its software hazards are target/gfx1250/hazards.rs) and the gfx1250 emitter drops the shared pass's `s_nop`s, so the rule
/// is no longer a correctness fact there: what remains is the pass's filling of the gaps with independent instructions, a scheduling
/// heuristic. It keeps the gfx950 distances deliberately (measured, 2026-11-21: zero distances reorder 49 gfx1250 kernels and cost the
/// MLA ~1 %); a gfx1250 rule of the same name, if the target describes one, wins.
pub fn shared_hazard(name: &str) -> u32 {
    let t = tgt();
    if t.hazards.iter().any(|h| h.name == name) {
        return t.hazard(name).n();
    }
    T.hazard(name).n()
}

/// `shared_hazard` for the latency-scaled rules (`by_latency`).
pub fn shared_hazard_by_latency(name: &str, lat: u32) -> u32 {
    let t = tgt();
    if t.hazards.iter().any(|h| h.name == name) {
        return t.hazard(name).by_latency(lat);
    }
    T.hazard(name).by_latency(lat)
}

/// Runs `g` with the default-target guard on (see `dflt`).
#[doc(hidden)]
pub fn with_default_target_guard<R>(g: impl FnOnce() -> R) -> R {
    let prev = T_GUARD.with(|x| x.replace(true));
    let r = g();
    T_GUARD.with(|x| x.set(prev));
    r
}

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
