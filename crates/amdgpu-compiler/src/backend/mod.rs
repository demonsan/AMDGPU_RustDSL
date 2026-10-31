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
pub mod mir_roundtrip;
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
        // Primary pin: llc built against the ROCm 10.1.0 libLLVM (ROCm 10 ships no llc binary).
        let pin = Path::new("/shared/amdgpu/home/blyu_qle/llvm-tools/rocm-10.1.0/bin");
        // An explicit ROCM_PATH is honoured; there is deliberately NO silent fallback to another LLVM.
        if let Ok(roc) = std::env::var("ROCM_PATH") {
            return LlvmTools { bin: Path::new(&roc).join("llvm").join("bin") };
        }
        LlvmTools { bin: pin.to_path_buf() }
    }

    pub fn tool(&self, name: &str) -> PathBuf {
        self.bin.join(name)
    }

    pub fn run(&self, tool: &str, args: &[&str]) -> Result<(String, String), String> {
        let out = Command::new(self.tool(tool))
            .args(args)
            .output()
            .map_err(|e| format!("the LLVM tail needs {} but it cannot be run: {e}. Set ROCM_LLVM_BIN to a tool directory (see backend.lock / docs/llvm-tail-conformance.md) or use --backend=own for our complete own tail.", self.tool(tool).display()))?;
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
    /// Do not export lowering's auto-inserted waits (experimental, see docs).
    pub drop_auto_waits: bool,
    /// Extra llc arguments (e.g. `-verify-machineinstrs`), in addition to `AMDC_LLC_ARGS`.
    pub extra_args: Vec<String>,
}

impl LlvmTail {
    pub fn new() -> LlvmTail {
        let keep = std::env::var("AMDC_KEEP_MIR").ok().map(PathBuf::from);
        let drop = std::env::var("AMDC_DROP_AUTO_WAITS").map(|v| v == "1").unwrap_or(false);
        LlvmTail { tools: LlvmTools::locate(), keep_mir: keep, skip_lock: false, drop_auto_waits: drop, extra_args: Vec::new() }
    }

    pub fn export_mir(&self, m: &MFunction, a: &Alloc) -> Result<String, String> {
        Ok(llvm_mir::export_opts(m, a, false, self.drop_auto_waits)?.mir)
    }

    /// Run the pinned llc tail over a MIR text.
    pub fn run_llc(&self, name: &str, mir: &str, start: &str) -> Result<String, String> {
        // unique per invocation: tests (and callers) compile the same kernel name concurrently
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("amdc-llvm-tail-{}-{}-{}", std::process::id(), seq, name));
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
        for e in &self.extra_args {
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
        let asm = self.run_llc(&m.name, &mir, &start)?;
        Ok(strip_pad_args(&asm))
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
    MOp::VPermB32,
    MOp::VPkSubU16,
    MOp::VFmaF32,
    MOp::VExpF32,
    MOp::VLogF32,
    MOp::VRcpF32,
    MOp::VCvtF32F16,
    MOp::VCvtF16F32,
    MOp::VAndB32,
    MOp::VMinI32,
    MOp::VOrB32,
    MOp::VXorB32,
    MOp::VLshrB32,
    MOp::VLshlB32,
    MOp::VMulLoU32,
    MOp::VCmpLtI32,
    MOp::VCmpGeI32,
    MOp::VCmpGtI32,
    MOp::VCmpNeI32,
    MOp::VCmpLtF32,
    MOp::VCndmask,
    MOp::VWriteSub,
    MOp::VReadSub,
    MOp::VMfma16x16x16F16,
    MOp::VMfma32x32x8F16,
    MOp::VMfma32x32x8F16Z,
    MOp::VMfma16x16x32Fp8,
    MOp::VMfma32x32x16Fp8,
    MOp::VMfmaScale32x32x64Fp8,
    MOp::VMfmaScale32x32x64Fp8Z,
    MOp::VMfma32x32x64Fp8,
    MOp::VMfma32x32x64Fp8Z,
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
    MOp::GlobalLoadB128S,
    MOp::GlobalLoadF32S,
    MOp::GlobalStoreU16,
    MOp::GlobalStoreF32,
    MOp::GlobalLoadF32Sc1,
    MOp::GlobalLoadB128Sc1,
    MOp::GlobalStoreF32Sc1,
    MOp::GlobalStoreF32x4Sc1,
    MOp::GlobalAtomicAddRet,
    MOp::BufferWbl2,
    MOp::BufferInv,
    MOp::GlobalStoreF32x4,
    MOp::GlobalStoreF32x2,
    MOp::SBufferRsrc,
    MOp::SBufferRsrcDyn,
    MOp::BufferLoadDwordX1,
    MOp::BufferLoadDwordX2,
    MOp::BufferLoadDwordX4,
    MOp::BufferStoreDwordX1,
    MOp::BufferStoreDwordX2,
    MOp::BufferStoreDwordX4,
    MOp::SLoadDword,
    MOp::SLoadDwordx2,
    MOp::SReadPhys,
    MOp::SMovB32,
    MOp::SAddU32,
    MOp::SAddcU32,
    MOp::SMulI32,
    MOp::SShlB32,
    MOp::VSmfmac16x16x128Fp8,
    MOp::SLshrB32,
    MOp::SAndB32,
    MOp::SXorB32,
    MOp::SCmpLtI32,
    MOp::SNop,
    MOp::SWaitVm,
    MOp::SWaitLgkm,
    MOp::SEndpgm,
];

/// Remove the dummy padding parameters (`%padN`, added by `llvm_mir::kernarg_ir` to give the IR shell the DSL's
/// kernarg layout) from the `.amdgpu_metadata` `.args` list. They only exist so that LLVM computes the right
/// `kernarg_segment_size`; listing them makes `hipModuleLaunchKernel` with `kernelParams` read parameters that the
/// caller does not have (host crash). `.kernarg_segment_size` and every real argument entry are untouched.
pub fn strip_pad_args(asm: &str) -> String {
    let lines: Vec<&str> = asm.lines().collect();
    let mut out: Vec<&str> = Vec::with_capacity(lines.len());
    let indent = |l: &str| l.len() - l.trim_start().len();
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        out.push(l);
        i += 1;
        if l.trim() == ".args:" {
            let base = indent(l);
            // items: lines starting with "- " at indent > base; the list ends at the first line with indent <= base
            let mut items: Vec<Vec<&str>> = Vec::new();
            while i < lines.len() && (lines[i].trim().is_empty() || indent(lines[i]) > base) {
                if lines[i].trim_start().starts_with("- ") {
                    items.push(vec![lines[i]]);
                } else if let Some(last) = items.last_mut() {
                    last.push(lines[i]);
                } else {
                    out.push(lines[i]);
                }
                i += 1;
            }
            for it in items {
                let is_pad = it.iter().any(|x| {
                    let t = x.trim().trim_start_matches("- ");
                    t.strip_prefix(".name:").map(|v| {
                        let v = v.trim();
                        v.strip_prefix("pad").map(|d| !d.is_empty() && d.chars().all(|c| c.is_ascii_digit())).unwrap_or(false)
                    }).unwrap_or(false)
                });
                if !is_pad {
                    out.extend(it);
                }
            }
        }
    }
    let mut s = out.join("\n");
    if asm.ends_with('\n') {
        s.push('\n');
    }
    s
}