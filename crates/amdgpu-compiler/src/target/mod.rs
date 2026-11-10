//! The compilation target (docs/target-design.md). P0 has one target: every pass reads gfx950's facts through `T`; a second target
//! turns `T` into the compilation's target (carried by the `MFunction`). The per-instruction tables that name `MOp` live under
//! `target/<name>/`.

pub use amdgpu_target::Target;

pub mod gfx950;

/// The target every compilation uses (P0: gfx950).
pub const T: &Target = &amdgpu_target::gfx950::GFX950;

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
