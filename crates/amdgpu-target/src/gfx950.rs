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
    // vmcnt 6 bits, lgkmcnt 4 bits, expcnt 3 bits; LDS DMA counts on vmcnt; SMEM and messages complete out of order on lgkmcnt
    counters: Counters {
        counters: &[Counter { name: "vmcnt", max: 63 }, Counter { name: "lgkmcnt", max: 15 }, Counter { name: "expcnt", max: 7 }],
        classes: &[
            (MemClass::VmemLoad, 0, Order::InOrder),
            (MemClass::VmemStore, 0, Order::InOrder),
            (MemClass::LdsDma, 0, Order::InOrder),
            (MemClass::Lds, 1, Order::InOrder),
            (MemClass::Smem, 1, Order::OutOfOrder),
            (MemClass::Msg, 1, Order::OutOfOrder),
            (MemClass::Export, 2, Order::InOrder),
        ],
        wait: WaitForm::Combined { mnemonic: "s_waitcnt" },
    },
    hazards: &[
        HazardRule { name: "trans-result", count: Count::N(5), unit: Unit::WaitStates, scope: Scope::Value, source: "LLVM TransForwardingHazard (1; 5 kept, conservative)" },
        HazardRule { name: "mfma-dst-write", count: Count::ByMfmaLatency(&[(20, 21)], 19), unit: Unit::WaitStates, scope: Scope::Physical, source: "CDNA4 ISA 7.6 XDL write VGPR -> VALU write; LLVM GFX940_XDL_N_PassWriteVgprVALUWawWaitStates" },
        HazardRule { name: "mfma-srcc-war", count: Count::ByMfmaLatency(&[(20, 21)], 12), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM checkMAIVALUHazards SrcC WAR (conservative)" },
        HazardRule { name: "wide-store-data", count: Count::N(3), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM 12DWordStore (issue distance 3 = 2 wait states)" },
        HazardRule { name: "valu-sgpr-vmem", count: Count::N(5), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM checkVMEMHazards; probe_valu_sgpr" },
        HazardRule { name: "valu-sgpr-valu", count: Count::N(2), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM checkVALUHazards hasVDecCoExecHazard" },
        HazardRule { name: "m0-lds-dma", count: Count::N(1), unit: Unit::WaitStates, scope: Scope::Value, source: "LLVM checkReadM0Hazards (LDS DMA)" },
    ],
    emit: Emission { sgpr_granule: 8, sgpr_cap: 102, sgpr_min: 8, vgpr_min: 4, vgpr_granule: 4, code_object_version: 6 },
    verify: VerifyLimits { vgpr_limit: 256, sgpr_limit: 102, lds_limit: 160 * 1024 },
};
