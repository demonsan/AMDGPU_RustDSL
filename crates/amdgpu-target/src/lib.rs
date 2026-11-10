//! Target descriptions (docs/target-design.md): plain data, no dependencies. The passes of `amdgpu-compiler` read every
//! architecture fact from a `Target` instead of a constant of their own; the per-instruction tables that name `MOp` live in
//! `amdgpu-compiler/src/target/<name>/`.
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
    pub emit: Emission,
    pub verify: VerifyLimits,
}

#[derive(Debug)]
pub struct Wave {
    /// lanes per wave (64; gfx1250 32)
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
    /// the accumulation registers (None: no AGPR segment, gfx1250)
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
    /// a pool of `blocks` blocks of `block_vgprs` VGPRs shared by the waves; SGPRs are fixed (gfx1250)
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
    /// gfx1250: tensor data mover
    Tensor,
    /// gfx1250: asynchronous LDS DMA
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
    /// one instruction per counter (gfx1250 `s_wait_loadcnt` / `s_wait_storecnt` / `s_wait_dscnt` / ...): (counter, mnemonic)
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
