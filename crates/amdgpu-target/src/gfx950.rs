//! gfx950 (MI350X / MI355X, CDNA4).

use crate::*;

pub const GFX950: Target = Target {
    name: "gfx950",
    wave: Wave { lanes: 64 },
    regs: RegFiles {
        // v0 holds the workitem id
        vgpr: RegFile { count: 256, reserved: 1, alloc_granule: 8 },
        agpr: Some(RegFile { count: 256, reserved: 0, alloc_granule: 8 }),
        // 102 usable SGPRs (the descriptor's next_free_sgpr must stay <= 102; vcc is separate); s0:s1 kernarg pointer, s2..s4 workgroup ids
        sgpr: RegFile { count: 102, reserved: 8, alloc_granule: 8 },
        unified: Some(Unified { total: 512, accum_granule: 4, total_granule: 8, max_waves_per_simd: 8, simds: 4 }),
        occupancy: Occupancy::UnifiedSteps,
    },
    lds: Lds { bytes_per_workgroup: 160 * 1024 },
    emit: Emission { sgpr_granule: 8, sgpr_cap: 102, sgpr_min: 8, vgpr_min: 4, vgpr_granule: 4, code_object_version: 6 },
    verify: VerifyLimits { vgpr_limit: 256, sgpr_limit: 102, lds_limit: 160 * 1024 },
};
