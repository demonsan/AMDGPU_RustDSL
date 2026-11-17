//! gfx1250: the description items that need the extension fields of docs/target-design.md section 4, as data, and the whole `GFX1250`
//! target (selectable with `--target gfx1250`; the gfx1250 pipeline is built in stages G1-G3, docs/gfx1250-plan.md). Every value is taken from docs/target-gfx1250.md (public sources only:
//! the CDNA5 ISA reference guide and upstream LLVM @274ab686); the section of that document is cited per item.

use crate::*;

/// VGPRs: 1024 per wave in blocks of 16; v0..v255 direct, higher registers through `s_set_vgpr_msb` (target-gfx1250.md section 3.2.1:
/// CDNA5 ISA section 3.3.2.3, LLVM AMDGPULowerVGPREncoding.cpp, VOP3PInstructions.td:1894-1895, MIMGInstructions.td:566).
pub const VGPR: RegFile = RegFile {
    count: 1024,
    reserved: 0,
    alloc_granule: 16,
    window: Some(RegWindow { direct: 256, setter: "s_set_vgpr_msb", low_only: &["VIMAGE vaddr3", "VIMAGE vaddr4", "WMMA scale src0", "WMMA scale src1", "v_wmma_ld_scale* operands"] }),
};

/// SGPRs: s0..s105, fixed (not an occupancy term) (section 3.2.3: CDNA5 ISA section 3.3.1.1).
pub const SGPR: RegFile = RegFile { count: 106, reserved: 0, alloc_granule: 0, window: None };

/// Occupancy: a pool of 64 blocks of 16 VGPRs (section 3.2.5: LLVM AMDGPUBaseInfo.cpp, GCNFeatures.td:1907; the ISA is silent).
pub const OCCUPANCY: Occupancy = Occupancy::BlockPool { block_vgprs: 16, blocks: 64 };

pub const LOADCNT: usize = 0;
pub const STORECNT: usize = 1;
pub const DSCNT: usize = 2;
pub const KMCNT: usize = 3;
pub const XCNT: usize = 4;
pub const ASYNCCNT: usize = 5;
pub const TENSORCNT: usize = 6;

/// The split counters, their classes and per-counter waits (section 4.1-4.4: CDNA5 ISA section 5.7, LLVM SIInsertWaitcnts.cpp:1568-1605,
/// AMDGPUUsage.md:14388-14390 (a wait implies XCNT), SIDefines.h / AMDGPUAsmUtils.cpp:66-74 (`s_wait_alu` fields)).
pub const COUNTERS: Counters = Counters {
    counters: &[
        Counter { name: "loadcnt", max: 63 },
        Counter { name: "storecnt", max: 63 },
        Counter { name: "dscnt", max: 63 },
        Counter { name: "kmcnt", max: 31 },
        Counter { name: "xcnt", max: 63 },
        Counter { name: "asynccnt", max: 63 },
        Counter { name: "tensorcnt", max: 63 },
    ],
    classes: &[
        (MemClass::VmemLoad, LOADCNT, Order::InOrder),
        (MemClass::CacheInv, LOADCNT, Order::InOrder),
        (MemClass::ClusterLoad, LOADCNT, Order::InOrder),
        (MemClass::VmemStore, STORECNT, Order::InOrder),
        (MemClass::CacheWb, STORECNT, Order::InOrder),
        (MemClass::Lds, DSCNT, Order::InOrder),
        (MemClass::FlatLds, DSCNT, Order::InOrder),
        (MemClass::Smem, KMCNT, Order::OutOfOrder),
        (MemClass::Msg, KMCNT, Order::OutOfOrder),
        (MemClass::BarrierQuery, KMCNT, Order::OutOfOrder),
        (MemClass::Async, ASYNCCNT, Order::InOrder),
        (MemClass::AsyncStore, ASYNCCNT, Order::InOrder),
        (MemClass::Tensor, TENSORCNT, Order::InOrder),
        (MemClass::VmemXlate, XCNT, Order::InOrder),
        (MemClass::SmemXlate, XCNT, Order::OutOfOrder),
    ],
    wait: WaitForm::PerCounter(&[
        (LOADCNT, "s_wait_loadcnt"),
        (STORECNT, "s_wait_storecnt"),
        (XCNT, "s_wait_xcnt"),
        (DSCNT, "s_wait_dscnt"),
        (KMCNT, "s_wait_kmcnt"),
        (ASYNCCNT, "s_wait_asynccnt"),
        (TENSORCNT, "s_wait_tensorcnt"),
    ]),
    // async loads vs async stores are two classes on ASYNCcnt: mixed, so only a wait for 0 (section 4.2)
    mixed_waits_zero: true,
    // every wait on a memory counter also bounds XCNT, except the tensor waits (section 4.2; kernel-opt's note on TDM)
    implied: Some((XCNT, &[TENSORCNT])),
    alu_wait: Some(AluWait {
        mnemonic: "s_wait_alu",
        fields: &[("sa_sdst", 0, 1), ("va_vcc", 1, 1), ("vm_vsrc", 2, 3), ("hold_cnt", 7, 1), ("va_ssrc", 8, 1), ("va_sdst", 9, 3), ("va_vdst", 12, 4)],
    }),
};

/// LDS: 320 KiB per workgroup; the DMA forms take the LDS address from a per-lane VGPR or the tensor descriptor, never from M0 (section 3.3:
/// CDNA5 ISA sections 10.7-10.11, LLVM FLATInstructions.td:391-418).
pub const LDS: Lds = Lds {
    bytes_per_workgroup: 327_680,
    dma: &[
        LdsDma { form: "global_load_async_to_lds_b*", lds_addr: LdsAddr::VgprPerLane, counter: ASYNCCNT },
        LdsDma { form: "global_store_async_from_lds_b*", lds_addr: LdsAddr::VgprPerLane, counter: ASYNCCNT },
        LdsDma { form: "cluster_load_async_to_lds_b*", lds_addr: LdsAddr::VgprPerLane, counter: ASYNCCNT },
        LdsDma { form: "tensor_load_to_lds / tensor_store_from_lds", lds_addr: LdsAddr::Descriptor, counter: TENSORCNT },
    ],
};

/// Memory: 45-bit buffer `num_records`; cache policy SCOPE / TH / NV / SCALE_OFFSET (section 6.5, 6.8: CDNA5 ISA sections 9-10, LLVM SIDefines.h).
pub const MEM: Mem = Mem { num_records_bits: 45, cache_policy: &["scope", "th", "nv", "scale_offset"] };

/// Synchronization: split barrier signal / wait, named barriers 1..16, workgroup clusters of up to 16 (section 4.7-4.8: CDNA5 ISA sections 2.3, 5).
pub const SYNC: Sync = Sync { split_barrier: true, named_barriers: 16, max_cluster_workgroups: 16 };

/// A software-mandatory WMMA rule counted in independent VALU instructions (section 5.1: LLVM GCNHazardRecognizer fixWMMAHazards); the
/// distance depends on the instruction group, so the count here is the instruction form the pass would emit.
pub const WMMA_COEXEC: HazardRule = HazardRule {
    name: "wmma-coexec",
    kind: Kind::Software,
    count: Count::Instr("v_nop"),
    unit: Unit::IndependentValu,
    scope: Scope::Value,
    source: "target-gfx1250.md section 5.1; LLVM GCNHazardRecognizer.cpp fixWMMAHazards",
};

/// The gfx1250 target (G0: the description the passes will read once the gfx1250 pipeline exists; `amdc --target gfx1250`). Values from
/// docs/target-gfx1250.md; the scheduler machine model is the LLVM speed model of section 5.2 mapped onto the weave fields (WMMA in the
/// matrix-pipe slots), to be replaced by hardware measurements (G4).
pub const GFX1250: Target = Target {
    name: "gfx1250",
    wave: Wave { lanes: 32 },
    regs: RegFiles {
        // v0 holds the packed workitem ids (section 3.2.1); the direct window is 256 (G1 allocates there only)
        vgpr: RegFile { count: 1024, reserved: 1, alloc_granule: 16, window: VGPR.window },
        agpr: None,
        // s0:s1 kernarg pointer; workgroup ids are architected (TTMP7 / TTMP9), not user SGPRs (section 3.2.3)
        sgpr: RegFile { count: 106, reserved: 2, alloc_granule: 0, window: None },
        unified: None,
        occupancy: OCCUPANCY,
    },
    lds: LDS,
    counters: COUNTERS,
    hazards: &[WMMA_COEXEC],
    // VGPRs in blocks of 16, the SGPR count field is 0 (fixed file), code object v5 (section 7.1)
    emit: Emission { sgpr_granule: 1, sgpr_cap: 106, sgpr_min: 0, vgpr_min: 16, vgpr_granule: 16, code_object_version: 5 },
    latency: LatencyModel {
        branch_taken: 24,
        branch_uncond: 18,
        branch_not_taken: 6,
        vcc_to_branch: 16,
        barrier: 12,
        trans_separate_pipe: true,
        slot_cycles: 32,
        slot_budget: 28,
        dma_latency_cycles: 512,
        weave: WeaveMachine {
            issue_gap: 1,
            // WMMA 16x16x32 bf16 / f16: 8 / 4 passes (section 5.2), latency estimates until measured
            mfma_cycles: 16,
            mfma16_cycles: 8,
            mfma_latency: 24,
            mfma16_latency: 16,
            valu_latency: 5,
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
        llc_args: &["-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-O2"],
        asm_args: &["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-mcode-object-version=5"],
    },
    mem: MEM,
    sync: SYNC,
    verify: VerifyLimits { vgpr_limit: 1024, sgpr_limit: 106, lds_limit: 327_680 },
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_class_rows_name_existing_counters() {
        for (_, c, _) in COUNTERS.classes {
            assert!(*c < COUNTERS.counters.len());
        }
        for d in LDS.dma {
            assert!(d.counter < COUNTERS.counters.len());
        }
        if let WaitForm::PerCounter(w) = COUNTERS.wait {
            assert_eq!(w.len(), COUNTERS.counters.len(), "one wait instruction per counter");
        } else {
            panic!("gfx1250 waits per counter");
        }
    }

    #[test]
    fn xcnt_is_in_order_for_vmem_and_out_of_order_for_smem() {
        let ord = |k: MemClass| COUNTERS.classes.iter().find(|x| x.0 == k).map(|x| (x.1, x.2)).unwrap();
        assert_eq!(ord(MemClass::VmemXlate), (XCNT, Order::InOrder));
        assert_eq!(ord(MemClass::SmemXlate), (XCNT, Order::OutOfOrder));
        assert_eq!(COUNTERS.of(MemClass::Smem).name, "kmcnt");
    }

    #[test]
    fn the_alu_wait_fields_fit_16_bits_without_overlap() {
        let w = COUNTERS.alu_wait.as_ref().unwrap();
        let mut used = 0u32;
        for (_, lo, n) in w.fields {
            let m = ((1u32 << n) - 1) << lo;
            assert_eq!(used & m, 0);
            used |= m;
            assert!(lo + n <= 16);
        }
    }

    #[test]
    fn no_dma_form_uses_m0_on_gfx1250_but_gfx950_does() {
        assert!(LDS.dma.iter().all(|d| d.lds_addr != LdsAddr::M0BasePlusImm));
        assert!(crate::gfx950::GFX950.lds.dma.iter().all(|d| d.lds_addr == LdsAddr::M0BasePlusImm));
    }
}
