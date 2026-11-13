//! gfx950 (MI350X / MI355X, CDNA4).

use crate::*;

pub const GFX950: Target = Target {
    name: "gfx950",
    wave: Wave { lanes: 64 },
    regs: RegFiles {
        // v0 holds the workitem id
        vgpr: RegFile { count: 256, reserved: 1, alloc_granule: 8, window: None },
        agpr: Some(RegFile { count: 256, reserved: 0, alloc_granule: 8, window: None }),
        // 102 usable SGPRs (the descriptor's next_free_sgpr must stay <= 102; vcc is separate); s0:s1 kernarg pointer, s2..s4 workgroup ids
        sgpr: RegFile { count: 102, reserved: 8, alloc_granule: 8, window: None },
        unified: Some(Unified { total: 512, accum_granule: 4, total_granule: 8, max_waves_per_simd: 8, simds: 4 }),
        occupancy: Occupancy::UnifiedSteps,
    },
    lds: Lds {
        bytes_per_workgroup: 160 * 1024,
        // M0 = LDS base + instruction offset + lane * size; completion on vmcnt
        dma: &[
            LdsDma { form: "buffer_load_dword* ... lds", lds_addr: LdsAddr::M0BasePlusImm, counter: 0 },
            LdsDma { form: "global_load_lds_dword*", lds_addr: LdsAddr::M0BasePlusImm, counter: 0 },
        ],
    },
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
        // lgkmcnt: partial waits only while no SMEM is queued (sched/waits.rs)
        mixed_waits_zero: true,
        implied: None,
        alu_wait: None,
    },
    hazards: &[
        HazardRule { kind: Kind::Software, name: "trans-result", count: Count::N(5), unit: Unit::WaitStates, scope: Scope::Value, source: "LLVM TransForwardingHazard (1; 5 kept, conservative)" },
        HazardRule { kind: Kind::Software, name: "mfma-dst-write", count: Count::ByMfmaLatency(&[(20, 21)], 19), unit: Unit::WaitStates, scope: Scope::Physical, source: "CDNA4 ISA 7.6 XDL write VGPR -> VALU write; LLVM GFX940_XDL_N_PassWriteVgprVALUWawWaitStates" },
        HazardRule { kind: Kind::Software, name: "mfma-srcc-war", count: Count::ByMfmaLatency(&[(20, 21)], 12), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM checkMAIVALUHazards SrcC WAR (conservative)" },
        HazardRule { kind: Kind::Software, name: "wide-store-data", count: Count::N(3), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM 12DWordStore (issue distance 3 = 2 wait states)" },
        HazardRule { kind: Kind::Software, name: "valu-sgpr-vmem", count: Count::N(5), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM checkVMEMHazards; probe_valu_sgpr" },
        HazardRule { kind: Kind::Software, name: "valu-sgpr-valu", count: Count::N(2), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM checkVALUHazards hasVDecCoExecHazard" },
        HazardRule { kind: Kind::Software, name: "valu-readfirstlane", count: Count::N(1), unit: Unit::WaitStates, scope: Scope::Value, source: "tools/readfirstlane_hazard_probe.sh: a VALU VGPR write read by v_readfirstlane is stale at 0 wait states and correct at 1 / 2 (512 threads, 4 VGPRs, 3 writer kinds, 5 runs); LLVM asks 2" },
        HazardRule { kind: Kind::Interlock, name: "valu-sgpr-mask", count: Count::N(1), unit: Unit::WaitStates, scope: Scope::Physical, source: "tools/lanemask_hazard_probe.sh: a VALU lane-mask write (v_cmp e64 into an SGPR pair, e32 into vcc) read as the v_cndmask mask is correct at 0 / 1 / 2 states (512 threads, 4 SGPR pairs, 5 runs); kept at 1 (vendor FMHA practice); LLVM hasVDecCoExecHazard asks 2" },
        HazardRule { kind: Kind::Interlock, name: "valu-vcc-valu", count: Count::N(1), unit: Unit::WaitStates, scope: Scope::Value, source: "tools/vcc_hazard_probe.sh: correct at 0 / 1 / 2 states; kept at 1 (vendor FMHA practice); LLVM hasVDecCoExecHazard asks 2" },
        HazardRule { kind: Kind::Software, name: "soft-clause-xnack", count: Count::N(1), unit: Unit::WaitStates, scope: Scope::Physical, source: "LLVM checkSoftClauseHazards (xnack any code objects)" },
        HazardRule { kind: Kind::Software, name: "m0-lds-dma", count: Count::N(1), unit: Unit::WaitStates, scope: Scope::Value, source: "LLVM checkReadM0Hazards (LDS DMA)" },
    ],
    emit: Emission { sgpr_granule: 8, sgpr_cap: 102, sgpr_min: 8, vgpr_min: 4, vgpr_granule: 4, code_object_version: 6 },
    // branch costs: docs/scheduling.md (tools/sched_mb/RESULTS2.md, 1 wave); slotfill: docs/slotfill.md (DMA 512 cycles swept on GEMM v4)
    latency: LatencyModel {
        branch_taken: 24,
        branch_uncond: 18,
        branch_not_taken: 6,
        vcc_to_branch: 16,
        barrier: 12,
        trans_separate_pipe: false,
        slot_cycles: 32,
        slot_budget: 28,
        dma_latency_cycles: 512,
        weave: WeaveMachine {
            issue_gap: 4,
            mfma_cycles: 64,
            mfma16_cycles: 32,
            mfma_latency: 84,
            mfma16_latency: 48,
            valu_latency: 8,
            ds_b128_latency: 144,
            ds_tr_latency: 125,
            ds_b32_latency: 64,
            lds_queue_depth: 8,
            lds_queue_penalty: 200,
            dma_issue: 24,
            branch_cost: 24,
        },
    },
    llvm: LlvmTail {
        triple: "amdgcn-amd-amdhsa",
        llc_args: &["-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx950", "-O2"],
        asm_args: &["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6"],
    },
    mem: Mem { num_records_bits: 32, cache_policy: &["sc0", "sc1", "nt"] },
    sync: Sync { split_barrier: false, named_barriers: 0, max_cluster_workgroups: 0 },
    verify: VerifyLimits { vgpr_limit: 256, sgpr_limit: 102, lds_limit: 160 * 1024 },
};
