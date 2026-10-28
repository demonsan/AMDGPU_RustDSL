//! Version probe and `backend.lock`.
//!
//! The lock pins the LLVM the MIR tail was validated against: the `llc --version` line, the
//! pass pipeline from the chosen start pass to the end, and the list of passes that must NOT
//! run after the start (they would reorder or re-allocate our schedule). Any mismatch is an
//! error: a new LLVM must be re-validated (opcode probe test, oracle diff) and the lock
//! re-written deliberately with `amdc backend write-lock`.

use std::path::PathBuf;

use super::{llvm20, LlvmTools};

/// Passes that must never run after the start point: they would change our instruction order
/// or register assignment.
pub const FORBIDDEN_AFTER_START: &[&str] = &[
    "postmisched",
    "machine-scheduler",
    "post-RA-sched",
    "block-placement",
    "tailduplication",
    "branch-folder",
    "machine-cp",
    "greedy",
    "virtregrewriter",
    "prologepilog",
    "machine-sink",
    "postra-machine-sink",
];

/// Passes whose presence in the tail is the whole point.
pub const REQUIRED_IN_TAIL: &[&str] = &["si-insert-waitcnts", "post-RA-hazard-rec"];

pub const PROBE_IR: &str = "target triple = \"amdgcn-amd-amdhsa\"\ndefine amdgpu_kernel void @k() {\n  ret void\n}\n";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub version: String,
    pub all_passes: Vec<String>,
    pub start_before: String,
    pub tail: Vec<String>,
}

pub fn llc_version(t: &LlvmTools) -> Result<String, String> {
    let (so, _) = t.run("llc", &["--version"])?;
    so.lines()
        .map(|l| l.trim())
        .find(|l| l.contains("LLVM version"))
        .map(|s| s.to_string())
        .ok_or_else(|| "cannot find 'LLVM version' in llc --version".to_string())
}

pub fn probe(t: &LlvmTools) -> Result<Report, String> {
    let version = llc_version(t)?;
    let dir = std::env::temp_dir().join(format!("amdc-probe-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let ll = dir.join("p.ll");
    std::fs::write(&ll, PROBE_IR).map_err(|e| e.to_string())?;
    let llp = ll.to_string_lossy().into_owned();
    let mut args: Vec<&str> = llvm20::LLC_ARGS.to_vec();
    args.extend(["-debug-pass=Structure", "-filetype=null", &llp]);
    let r = t.run("llc", &args);
    let _ = std::fs::remove_dir_all(&dir);
    let (_, se) = r?;
    let line = se
        .lines()
        .find(|l| l.starts_with("Pass Arguments:") && l.contains("si-insert-waitcnts"))
        .ok_or("no 'Pass Arguments:' line with the code generation pipeline in llc -debug-pass=Structure")?;
    let all: Vec<String> = line
        .trim_start_matches("Pass Arguments:")
        .split_whitespace()
        .map(|s| s.trim_start_matches('-').to_string())
        .collect();
    let start = llvm20::START_PASS_PREFERENCE
        .iter()
        .find(|p| all.iter().any(|a| a == **p))
        .ok_or_else(|| format!("none of {:?} exists in the pipeline", llvm20::START_PASS_PREFERENCE))?
        .to_string();
    let idx = all.iter().position(|a| *a == start).unwrap();
    Ok(Report { version, tail: all[idx..].to_vec(), all_passes: all, start_before: start })
}

impl Report {
    /// Structural safety of the tail (independent of the lock).
    pub fn validate(&self) -> Result<(), String> {
        for p in &self.tail {
            if FORBIDDEN_AFTER_START.contains(&p.as_str()) {
                return Err(format!(
                    "pass '{p}' runs after '{}': LLVM would reorder/reallocate our schedule; the tail start must move or the design changes",
                    self.start_before
                ));
            }
        }
        for r in REQUIRED_IN_TAIL {
            if !self.tail.iter().any(|p| p == r) {
                return Err(format!("required tail pass '{r}' is missing after '{}'", self.start_before));
            }
        }
        Ok(())
    }

    pub fn render_lock(&self) -> String {
        format!(
            "# backend.lock - LLVM the MIR tail was validated against (see docs/llvm-tail-integration.md).\n\
             # Regenerate deliberately with: amdc backend write-lock  (after re-running the opcode probe and the oracle).\n\
             dialect = {}\n\
             llvm_version = {}\n\
             start_before = {}\n\
             forbidden_after_start = {}\n\
             required_in_tail = {}\n\
             tail_passes = {}\n",
            llvm20::NAME,
            self.version,
            self.start_before,
            FORBIDDEN_AFTER_START.join(","),
            REQUIRED_IN_TAIL.join(","),
            self.tail.join(",")
        )
    }
}

#[derive(Clone, Debug)]
pub struct Lock {
    pub dialect: String,
    pub llvm_version: String,
    pub start_before: String,
    pub tail_passes: Vec<String>,
}

impl Lock {
    pub fn parse(text: &str) -> Result<Lock, String> {
        let mut kv = std::collections::HashMap::new();
        for l in text.lines() {
            let l = l.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let (k, v) = l.split_once('=').ok_or_else(|| format!("backend.lock: bad line '{l}'"))?;
            kv.insert(k.trim().to_string(), v.trim().to_string());
        }
        let get = |k: &str| kv.get(k).cloned().ok_or_else(|| format!("backend.lock: missing '{k}'"));
        Ok(Lock {
            dialect: get("dialect")?,
            llvm_version: get("llvm_version")?,
            start_before: get("start_before")?,
            tail_passes: get("tail_passes")?.split(',').map(|s| s.to_string()).collect(),
        })
    }

    pub fn find() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("AMDC_BACKEND_LOCK") {
            return Some(PathBuf::from(p));
        }
        let mut dir = std::env::current_dir().ok()?;
        loop {
            let c = dir.join("backend.lock");
            if c.exists() {
                return Some(c);
            }
            if !dir.pop() {
                break;
            }
        }
        let c = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../backend.lock");
        c.exists().then_some(c)
    }

    pub fn find_and_load() -> Result<Lock, String> {
        let p = Lock::find().ok_or("backend.lock not found (run from the repo or set AMDC_BACKEND_LOCK)")?;
        let text = std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        Lock::parse(&text)
    }

    pub fn check_version(&self, version: &str) -> Result<(), String> {
        if self.dialect != llvm20::NAME {
            return Err(format!("backend.lock dialect '{}' but this build only has '{}'", self.dialect, llvm20::NAME));
        }
        if self.llvm_version != version {
            return Err(format!(
                "LLVM changed: backend.lock pins '{}', llc reports '{}'. Re-validate (cargo test opcode probe + tools/hazard_oracle) then `amdc backend write-lock`.",
                self.llvm_version, version
            ));
        }
        Ok(())
    }

    pub fn check(&self, r: &Report) -> Result<(), String> {
        self.check_version(&r.version)?;
        if self.start_before != r.start_before {
            return Err(format!("start pass changed: locked '{}', now '{}'", self.start_before, r.start_before));
        }
        if self.tail_passes != r.tail {
            let added: Vec<&String> = r.tail.iter().filter(|p| !self.tail_passes.contains(p)).collect();
            let removed: Vec<&String> = self.tail_passes.iter().filter(|p| !r.tail.contains(p)).collect();
            return Err(format!(
                "LLVM tail pipeline changed vs backend.lock: added {added:?}, removed {removed:?} (or reordered). Review the new passes for hazards/reordering, then write-lock."
            ));
        }
        Ok(())
    }
}