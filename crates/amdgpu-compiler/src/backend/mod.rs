//! Tail backends: what happens to the final, allocated, scheduled `MFunction`.
//!
//! * `OwnTail`  - our own text emitter (`emit.rs`), the default.
//! * `LlvmTail` - export post-RA MIR and let LLVM's tail passes (waitcnt, hazard recognizer,
//!   delay_alu, ...) finish the job. See docs/llvm-tail-integration.md.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::mir::{MFunction, MOp};
use crate::regalloc::Alloc;

pub mod llvm20;
pub mod llvm_mir;
pub mod probe;

pub trait TailBackend {
    fn name(&self) -> &'static str;
    /// Final assembly text for an allocated function.
    fn finish(&self, m: &MFunction, a: &Alloc) -> Result<String, String>;
}

/// Our own emitter (byte-identical to the historical output).
pub struct OwnTail;

impl TailBackend for OwnTail {
    fn name(&self) -> &'static str {
        "own"
    }
    fn finish(&self, m: &MFunction, a: &Alloc) -> Result<String, String> {
        Ok(crate::emit::emit(m, a))
    }
}

/// Locates the pinned LLVM tools.
#[derive(Clone, Debug)]
pub struct LlvmTools {
    pub bin: PathBuf,
}

impl LlvmTools {
    /// `ROCM_LLVM_BIN`, else `$ROCM_PATH/llvm/bin`, else the ROCm 7.0.2 install on the node.
    pub fn locate() -> LlvmTools {
        if let Ok(p) = std::env::var("ROCM_LLVM_BIN") {
            return LlvmTools { bin: PathBuf::from(p) };
        }
        let roc = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/shared/apps/ubuntu/opt/rocm-7.0.2".to_string());
        LlvmTools { bin: Path::new(&roc).join("llvm").join("bin") }
    }

    pub fn tool(&self, name: &str) -> PathBuf {
        self.bin.join(name)
    }

    pub fn run(&self, tool: &str, args: &[&str]) -> Result<(String, String), String> {
        let out = Command::new(self.tool(tool))
            .args(args)
            .output()
            .map_err(|e| format!("cannot run {}: {e} (set ROCM_LLVM_BIN)", self.tool(tool).display()))?;
        let so = String::from_utf8_lossy(&out.stdout).into_owned();
        let se = String::from_utf8_lossy(&out.stderr).into_owned();
        if !out.status.success() {
            return Err(format!("{tool} failed ({}):\n{se}", out.status));
        }
        Ok((so, se))
    }
}

pub struct LlvmTail {
    pub tools: LlvmTools,
    /// Directory to keep the exported `.mir` in (else a temp file that is removed).
    pub keep_mir: Option<PathBuf>,
    /// Skip the backend.lock check (tests only).
    pub skip_lock: bool,
}

impl LlvmTail {
    pub fn new() -> LlvmTail {
        let keep = std::env::var("AMDC_KEEP_MIR").ok().map(PathBuf::from);
        LlvmTail { tools: LlvmTools::locate(), keep_mir: keep, skip_lock: false }
    }

    pub fn export_mir(&self, m: &MFunction, a: &Alloc) -> Result<String, String> {
        Ok(llvm_mir::export(m, a)?.mir)
    }

    /// Run the pinned llc tail over a MIR text.
    pub fn run_llc(&self, name: &str, mir: &str, start: &str) -> Result<String, String> {
        let dir = std::env::temp_dir().join(format!("amdc-llvm-tail-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let mir_path = dir.join(format!("{name}.mir"));
        let asm_path = dir.join(format!("{name}.s"));
        std::fs::write(&mir_path, mir).map_err(|e| e.to_string())?;
        if let Some(k) = &self.keep_mir {
            let _ = std::fs::create_dir_all(k);
            let _ = std::fs::write(k.join(format!("{name}.mir")), mir);
        }
        let start_arg = format!("-start-before={start}");
        let mut args: Vec<&str> = llvm20::LLC_ARGS.to_vec();
        args.push(&start_arg);
        args.push("-filetype=asm");
        let extra = std::env::var("AMDC_LLC_ARGS").unwrap_or_default();
        for e in extra.split_whitespace() {
            args.push(e);
        }
        let mp = mir_path.to_string_lossy().into_owned();
        let ap = asm_path.to_string_lossy().into_owned();
        args.push(&mp);
        args.push("-o");
        args.push(&ap);
        let r = self.tools.run("llc", &args);
        let asm = std::fs::read_to_string(&asm_path);
        let _ = std::fs::remove_dir_all(&dir);
        r?;
        asm.map_err(|e| e.to_string())
    }
}

impl Default for LlvmTail {
    fn default() -> Self {
        Self::new()
    }
}

impl TailBackend for LlvmTail {
    fn name(&self) -> &'static str {
        "llvm-tail"
    }
    fn finish(&self, m: &MFunction, a: &Alloc) -> Result<String, String> {
        let start = if self.skip_lock {
            llvm20::START_PASS_PREFERENCE[0].to_string()
        } else {
            let lock = probe::Lock::find_and_load()?;
            let ver = probe::llc_version(&self.tools)?;
            lock.check_version(&ver)?;
            lock.start_before.clone()
        };
        let mir = self.export_mir(m, a)?;
        self.run_llc(&m.name, &mir, &start)
    }
}

pub fn backend_by_name(name: &str) -> Option<Box<dyn TailBackend>> {
    match name {
        "own" => Some(Box::new(OwnTail)),
        "llvm-tail" => Some(Box::new(LlvmTail::new())),
        _ => None,
    }
}
/// Every MOp, for table coverage checks and the per-opcode probe.
pub const ALL_MOPS: &[crate::mir::MOp] = &[
    MOp::VMovB32,
    MOp::VMovFromS,
    MOp::VMovFromSPair,
    MOp::VPtrAdd,
    MOp::VReadPhys,
    MOp::VReadFirstLane,
    MOp::VCvtPknormU16,
    MOp::VCvtPkFp8F32,
    MOp::VCvtPkBf16F32,
    MOp::VCvtPkFp8Lo,
    MOp::VCvtPkFp8Hi,
    MOp::VCvtPkFp8F32Hi,
    MOp::VPkAddF32,
    MOp::VPermlane32Swap,
    MOp::VPkMulF32,
    MOp::VAddU32,
    MOp::VAddCoU32,
    MOp::VAddcCoU32,
    MOp::VAddF32,
    MOp::VSubF32,
    MOp::VMulF32,
    MOp::VMaxF32,
    MOp::VMax3F32,
    MOp::VFmaF32,
    MOp::VExpF32,
    MOp::VRcpF32,
    MOp::VCvtF32F16,
    MOp::VCvtF16F32,
    MOp::VAndB32,
    MOp::VOrB32,
    MOp::VXorB32,
    MOp::VLshrB32,
    MOp::VLshlB32,
    MOp::VMulLoU32,
    MOp::VCmpLtI32,
    MOp::VCmpGeI32,
    MOp::VCmpNeI32,
    MOp::VCmpLtF32,
    MOp::VCndmask,
    MOp::VWriteSub,
    MOp::VReadSub,
    MOp::VMfma16x16x16F16,
    MOp::VMfma32x32x8F16,
    MOp::VMfma16x16x32Fp8,
    MOp::VMfma32x32x16Fp8,
    MOp::VMfmaScale32x32x64Fp8,
    MOp::VMfmaScale32x32x64Fp8Z,
    MOp::LdsWriteB32,
    MOp::LdsReadB32,
    MOp::LdsWriteB128,
    MOp::LdsReadB128,
    MOp::LdsReadB64TrB8,
    MOp::SBarrier,
    MOp::SSetPrio,
    MOp::GlobalLoadU16,
    MOp::GlobalLoadF32,
    MOp::GlobalLoadB128,
    MOp::GlobalStoreU16,
    MOp::GlobalStoreF32,
    MOp::SLoadDword,
    MOp::SLoadDwordx2,
    MOp::SReadPhys,
    MOp::SMovB32,
    MOp::SAddU32,
    MOp::SAddcU32,
    MOp::SMulI32,
    MOp::SShlB32,
    MOp::SCmpLtI32,
    MOp::SNop,
    MOp::SWaitVm,
    MOp::SWaitLgkm,
    MOp::SEndpgm,
];
