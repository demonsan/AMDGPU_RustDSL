//! Target descriptions (docs/target-design.md): plain data, no dependencies. The passes of `amdgpu-codegen` read every
//! architecture fact from a `Target` instead of a constant of their own; the per-instruction tables that name `MOp` live in
//! `amdgpu-codegen/src/target/<name>/`.
#![forbid(unsafe_code)]

pub mod gfx950;

/// One target.
#[derive(Debug)]
pub struct Target {
    /// LLVM processor name (`-mcpu`)
    pub name: &'static str,
    pub wave: Wave,
    pub regs: RegFiles,
    pub lds: Lds,
    pub counters: Counters,
    /// the hazard rules whose distances are target facts (the per-instruction writer delays are in the instruction tables)
    pub hazards: &'static [HazardRule],
    pub latency: LatencyModel,
    pub emit: Emission,
    pub llvm: LlvmTail,
    pub verify: VerifyLimits,
}

/// The machine model of the schedulers (the per-instruction issue / latency costs are in the instruction tables).
#[derive(Debug)]
pub struct LatencyModel {
    /// taken branch, unconditional branch, not-taken branch, VALU compare -> branch on vcc, `s_barrier` (cycles)
    pub branch_taken: u32,
    pub branch_uncond: u32,
    pub branch_not_taken: u32,
    pub vcc_to_branch: u32,
    pub barrier: u32,
    /// whether `v_exp` / `v_rcp` issue on their own pipe next to the VALU
    pub trans_separate_pipe: bool,
    /// slotfill: cycles of one MFMA issue slot, the instruction cycles that fit into it, a global -> LDS DMA's latency
    pub slot_cycles: u32,
    pub slot_budget: u32,
    pub dma_latency_cycles: u32,
    /// the weave planner's coarse model (one wave per SIMD)
    pub weave: WeaveMachine,
}

/// The weave planner's machine numbers (cycles).
#[derive(Debug)]
pub struct WeaveMachine {
    pub issue_gap: i64,
    pub mfma_cycles: i64,
    pub mfma16_cycles: i64,
    pub mfma_latency: i64,
    pub mfma16_latency: i64,
    pub valu_latency: i64,
    pub ds_b128_latency: i64,
    pub ds_tr_latency: i64,
    pub ds_b32_latency: i64,
    /// LDS ops outstanding at issue from which the return slows down, and by how much
    pub lds_queue_depth: usize,
    pub lds_queue_penalty: i64,
    pub dma_issue: i64,
    pub branch_cost: i64,
}

/// The LLVM tools' settings for this target.
#[derive(Debug)]
pub struct LlvmTail {
    pub triple: &'static str,
    /// `llc` arguments of the post-RA tail (`-mtriple`, `-mcpu`, optimisation level)
    pub llc_args: &'static [&'static str],
    /// clang's assembler arguments before `-o` (the code object of this target)
    pub asm_args: &'static [&'static str],
}

#[derive(Debug)]
pub struct Wave {
    /// lanes per wave (64; wave32 targets 32)
    pub lanes: u32,
}

/// One register file.
#[derive(Debug)]
pub struct RegFile {
    /// registers the allocator may use
    pub count: u32,
    /// registers reserved at the bottom (kernel inputs): never allocated
    pub reserved: u32,
    /// granule of the descriptor's register count
    pub alloc_granule: u32,
}

#[derive(Debug)]
pub struct RegFiles {
    pub vgpr: RegFile,
    /// the accumulation registers (None: a target without an AGPR segment)
    pub agpr: Option<RegFile>,
    pub sgpr: RegFile,
    /// arch VGPRs + AGPRs share one file of this many registers per lane (gfx950: 512)
    pub unified: Option<Unified>,
    pub occupancy: Occupancy,
}

#[derive(Debug)]
pub struct Unified {
    pub total: u32,
    /// `accum_offset` (the arch VGPR count) granule
    pub accum_granule: u32,
    /// the combined count's granule (`next_free_vgpr` when AGPRs are used)
    pub total_granule: u32,
    /// at most this many waves per SIMD
    pub max_waves_per_simd: u32,
    /// SIMDs per CU (a workgroup's waves are spread over them)
    pub simds: u32,
}

/// How register usage limits the waves per SIMD.
#[derive(Debug)]
pub enum Occupancy {
    /// waves per SIMD = unified total / registers per wave (gfx950)
    UnifiedSteps,
    /// a pool of `blocks` blocks of `block_vgprs` VGPRs shared by the waves; SGPRs are not an occupancy term
    BlockPool { block_vgprs: u32, blocks: u32 },
}

/// The memory classes that bump a dependency counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemClass {
    VmemLoad,
    VmemStore,
    /// a global -> LDS DMA (`buffer_load ... lds`, `global_load_lds`)
    LdsDma,
    Lds,
    Smem,
    Msg,
    Export,
    /// a tensor data mover operation
    Tensor,
    /// an asynchronous LDS DMA
    Async,
}

/// Completion order of one class on its counter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// completes in issue order: `wait(n)` waits for all but the n most recent
    InOrder,
    /// completes out of order: only `wait(0)` is a completion point for it
    OutOfOrder,
}

#[derive(Debug)]
pub struct Counter {
    pub name: &'static str,
    /// the largest encodable count (a counter with this many younger operations queued has completed the older ones)
    pub max: u32,
}

/// How the wait instruction names counters.
#[derive(Debug)]
pub enum WaitForm {
    /// one instruction with a field per counter (gfx950 `s_waitcnt vmcnt(n) lgkmcnt(n) expcnt(n)`)
    Combined { mnemonic: &'static str },
    /// one instruction per counter (gfx12 in upstream LLVM: `s_wait_loadcnt` / `s_wait_storecnt` / `s_wait_dscnt` / ...): (counter, mnemonic)
    PerCounter(&'static [(usize, &'static str)]),
}

/// The dependency counters (docs/target-design.md section 2).
#[derive(Debug)]
pub struct Counters {
    pub counters: &'static [Counter],
    /// (class, counter index, order)
    pub classes: &'static [(MemClass, usize, Order)],
    pub wait: WaitForm,
}

impl Counters {
    /// The counter a memory class bumps.
    pub const fn of(&self, class: MemClass) -> &Counter {
        let mut i = 0;
        while i < self.classes.len() {
            if self.classes[i].0 as u8 == class as u8 {
                return &self.counters[self.classes[i].1];
            }
            i += 1;
        }
        panic!("memory class without a counter")
    }
}

/// The distance unit of a hazard rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// wait states: every issued instruction or `s_nop` state counts (gfx950)
    WaitStates,
    /// only independent VALU instructions count (upstream LLVM `GCNHazardRecognizer::fixWMMAHazards`)
    IndependentValu,
}

/// The required distance of a rule.
#[derive(Debug)]
pub enum Count {
    N(u32),
    /// by the producer's MFMA result latency: (latency, distance) pairs, else the default
    ByMfmaLatency(&'static [(u32, u32)], u32),
    /// a required instruction rather than a distance
    Instr(&'static str),
}

/// Where the rule is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// on values before register allocation
    Value,
    /// on physical registers after allocation
    Physical,
}

/// One hazard rule (docs/trust-inventory.md section 5.1 is the gfx950 audit against LLVM GCNHazardRecognizer).
#[derive(Debug)]
pub struct HazardRule {
    pub name: &'static str,
    pub count: Count,
    pub unit: Unit,
    pub scope: Scope,
    /// the evidence: LLVM function, ISA section, probe
    pub source: &'static str,
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

impl Target {
    /// The hazard rule `name` (panics at compile time in a const context when the target has no such rule).
    pub const fn hazard(&self, name: &str) -> &HazardRule {
        let mut i = 0;
        while i < self.hazards.len() {
            if str_eq(self.hazards[i].name, name) {
                return &self.hazards[i];
            }
            i += 1;
        }
        panic!("hazard rule not described by the target")
    }
}

impl HazardRule {
    /// The fixed distance (panics for other counts).
    pub const fn n(&self) -> u32 {
        match self.count {
            Count::N(n) => n,
            _ => panic!("not a fixed-distance rule"),
        }
    }

    /// The distance for a producer with MFMA result latency `lat`.
    pub fn by_latency(&self, lat: u32) -> u32 {
        match &self.count {
            Count::ByMfmaLatency(t, d) => t.iter().find(|(l, _)| *l == lat).map_or(*d, |x| x.1),
            Count::N(n) => *n,
            Count::Instr(_) => 0,
        }
    }
}

#[derive(Debug)]
pub struct Lds {
    /// bytes per workgroup
    pub bytes_per_workgroup: u32,
}

/// The kernel descriptor (`.amdhsa_*`) and code-object facts.
#[derive(Debug)]
pub struct Emission {
    /// the descriptor's SGPR count: rounded to this granule, capped at `sgpr_cap` (vcc is separate on gfx950)
    pub sgpr_granule: u32,
    pub sgpr_cap: u32,
    /// the smallest declared SGPR / arch-VGPR counts
    pub sgpr_min: u32,
    pub vgpr_min: u32,
    /// granule of the arch VGPR count without AGPRs
    pub vgpr_granule: u32,
    pub code_object_version: u32,
}

/// Limits the GPU-semantics verifier checks the final allocation against.
#[derive(Debug)]
pub struct VerifyLimits {
    pub vgpr_limit: u32,
    pub sgpr_limit: u32,
    pub lds_limit: u32,
}
