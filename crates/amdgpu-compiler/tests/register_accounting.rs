//! Independent register-accounting audit of the **final assembly text**.
//!
//! Hardware allocates per-wave register windows from the kernel descriptor counts
//! (`.amdhsa_next_free_vgpr` / `.amdhsa_next_free_sgpr`), so an under-declared count silently makes a
//! wave read/write registers that belong to another wave. The allocator's own numbers cannot be
//! trusted to catch this class of bug (the number and the code are produced together); this test
//! re-derives the requirements from the emitted instruction stream instead.
//!
//! For every kernel in [`kernel_names`], on the own backend, the audit parses the assembly and checks:
//!
//! 1. **descriptor range**: every physical VGPR/SGPR index that appears in any operand - including
//!    multi-register operands (`v[4:19]`, `s[0:1]`) - lies below the declared `next_free_*` count.
//!    (The previous scan in `backend.rs` only recognised single-register operands such as `v12`, so a
//!    register that only ever appears inside a tuple was invisible to it.)
//! 2. **MFMA tuple shape**: every multi-register operand of an MFMA is aligned to its own width and
//!    inside the declared VGPR range (the post-RA check in `regalloc.rs` checks the same property on
//!    MIR; this audit re-checks the emitted text, and also catches tuple operands of other wide
//!    instructions going out of range).
//! 3. **defined-before-use on every path**: a must-assignment dataflow over the emitted CFG. Every
//!    VGPR/SGPR read must be written earlier on *every* path from the entry block (or be one of the
//!    ABI inputs: `v0` workitem id, `s0:s1` kernarg pointer, `s2..s4` workgroup ids). This is the
//!    "reads a stale/garbage register" half of the same bug class.
//!
//! The own backend is always checked. The LLVM tail is checked for (1)+(2) when `AMDC_TEST_LLVM=1`
//! and the pinned `llc` is runnable (its output is allowed to differ in scheduling; only the
//! register accounting must hold). `checker_detects_the_bug_class` feeds deliberately broken assembly
//! through the same code so the audit cannot pass vacuously.
//!
//! A negative result here is not hypothetical: the SGPR under-declaration found on this branch
//! (a loop after allocation overwrote `max_s` with "last still-live slot + 1") is exactly what check
//! (1) reports. See `docs/verification.md`.

use std::collections::HashSet;

use amdgpu_compiler::backend::{LlvmTail, LlvmTools};
use amdgpu_compiler::{compile, compile_with, kernel_by_name, kernel_names};

/// A physical register reference parsed out of an operand: `v12`, `v[4:7]`, `s8`, `s[0:1]`, `a[0:15]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Reg {
    scalar: bool,
    base: u32,
    width: u32,
}

impl Reg {
    fn keys(&self) -> impl Iterator<Item = u32> + '_ {
        let class = if self.scalar { 512 } else { 0 };
        (self.base..self.base + self.width).map(move |i| class + i)
    }
}

/// Every register reference in an operand string. `a[]` is the ISA spelling of an accumulator in
/// the VGPR file on older targets and is treated like `v[]`.
fn scan_regs(s: &str) -> Vec<Reg> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < b.len() {
        let c = b[i] as char;
        let reg_letter = c == 'v' || c == 's' || c == 'a';
        let prev_ok = i == 0 || !((b[i - 1] as char).is_ascii_alphanumeric() || b[i - 1] == b'_');
        if reg_letter && prev_ok {
            if i + 1 < b.len() && b[i + 1] == b'[' {
                let mut j = i + 2;
                let mut base: u64 = 0;
                let mut n1 = 0usize;
                while j < b.len() && (b[j] as char).is_ascii_digit() {
                    base = base * 10 + (b[j] - b'0') as u64;
                    j += 1;
                    n1 += 1;
                }
                if n1 > 0 && j < b.len() && b[j] == b':' {
                    let mut k = j + 1;
                    let mut end: u64 = 0;
                    let mut n2 = 0usize;
                    while k < b.len() && (b[k] as char).is_ascii_digit() {
                        end = end * 10 + (b[k] - b'0') as u64;
                        k += 1;
                        n2 += 1;
                    }
                    if n2 > 0 && k < b.len() && b[k] == b']' && end >= base {
                        out.push(Reg { scalar: c == 's', base: base as u32, width: (end - base + 1) as u32 });
                        i = k + 1;
                        continue;
                    }
                }
            } else if i + 1 < b.len() && (b[i + 1] as char).is_ascii_digit() {
                let mut j = i + 1;
                let mut val: u64 = 0;
                while j < b.len() && (b[j] as char).is_ascii_digit() {
                    val = val * 10 + (b[j] - b'0') as u64;
                    j += 1;
                }
                out.push(Reg { scalar: c == 's', base: val as u32, width: 1 });
                i = j;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Split one instruction line into mnemonic and top-level operands (commas inside `[]`/`()` do not
/// separate operands: `op_sel:[0,0,1]`, `instid0(VALU_DEP_1)`).
fn split_ops(line: &str) -> (String, Vec<String>) {
    let line = line.split(';').next().unwrap_or("").trim();
    let mut it = line.splitn(2, char::is_whitespace);
    let mnem = it.next().unwrap_or("").to_string();
    let rest = it.next().unwrap_or("");
    let mut ops = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for ch in rest.chars() {
        match ch {
            '[' | '(' => {
                depth += 1;
                cur.push(ch);
            }
            ']' | ')' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => {
                ops.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        ops.push(cur.trim().to_string());
    }
    (mnem, ops)
}

fn is_control(m: &str) -> bool {
    m == "s_branch"
        || m.starts_with("s_cbranch")
        || m == "s_endpgm"
        || m == "s_waitcnt"
        || m == "s_barrier"
        || m == "s_nop"
        || m == "s_setprio"
        || m == "s_sleep"
        || m == "s_sendmsg"
        || m == "s_ttracedata"
        || m.starts_with("s_delay_alu")
        || m.starts_with("s_icache_inv")
}

fn is_store(m: &str) -> bool {
    m.starts_with("ds_write")
        || m.starts_with("global_store")
        || m.starts_with("buffer_store")
        || m.starts_with("scratch_store")
        || m.starts_with("s_store")
        || m.starts_with("global_load_lds")
}

fn is_load(m: &str) -> bool {
    m.starts_with("ds_read")
        || m.starts_with("global_load")
        || m.starts_with("buffer_load")
        || m.starts_with("scratch_load")
        || m.starts_with("s_load")
        || m.starts_with("s_buffer_load")
}

/// Register reads/writes of one instruction, as defined by the emitter's own operand formats
/// (`crates/amdgpu-compiler/src/emit.rs`).
fn read_write(m: &str, ops: &[String]) -> (Vec<Reg>, Vec<Reg>) {
    if is_control(m) {
        return (Vec::new(), Vec::new());
    }
    if is_store(m) {
        // address/data are all reads: no destination
        return (Vec::new(), ops.iter().flat_map(|o| scan_regs(o)).collect());
    }
    if is_load(m) {
        let defs = ops.first().map(|o| scan_regs(o)).unwrap_or_default();
        let uses = ops.iter().skip(1).flat_map(|o| scan_regs(o)).collect();
        return (defs, uses);
    }
    if m.starts_with("v_permlane32_swap") {
        // writes both operands in place
        let all: Vec<Reg> = ops.iter().flat_map(|o| scan_regs(o)).collect();
        return (all.clone(), all);
    }
    if m.starts_with("v_mfma") || m.starts_with("v_smfmac") {
        // D is in place with C; if C is the literal 0 (zero-accumulator form) D is not read.
        let zero_acc = ops.iter().any(|o| o.trim() == "0");
        let defs = ops.first().map(|o| scan_regs(o)).unwrap_or_default();
        let mut uses: Vec<Reg> = Vec::new();
        for (i, o) in ops.iter().enumerate() {
            if i == 0 && zero_acc {
                continue;
            }
            uses.extend(scan_regs(o));
        }
        return (defs, uses);
    }
    if m.starts_with("v_cmp") || m.starts_with("s_cmp") {
        // writes vcc/scc only; those are not numbered registers
        return (Vec::new(), ops.iter().flat_map(|o| scan_regs(o)).collect());
    }
    // Generic form of this emitter: the first operand is the destination.
    let defs = ops.first().map(|o| scan_regs(o)).unwrap_or_default();
    let uses = ops.iter().skip(1).flat_map(|o| scan_regs(o)).collect();
    (defs, uses)
}

#[derive(Default, Clone)]
struct Block {
    label: Option<String>,
    insts: Vec<(String, Vec<String>)>,
    succs: Vec<usize>,
}

/// The instruction lines of `asm` for kernel `name`: between `.type name,@function` and `.size`.
/// Labels inside the region are kept; other directives (`.p2align` etc.) are not instructions.
fn instruction_region(asm: &str, name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut inside = false;
    let needle = format!(".type {name},@function");
    for raw in asm.lines() {
        let line = raw.trim();
        if !inside {
            if line.starts_with(&needle) {
                inside = true;
            }
            continue;
        }
        if line.starts_with(".size") {
            break;
        }
        if line.is_empty() {
            continue;
        }
        if line.starts_with('.') && !line.ends_with(':') {
            continue;
        }
        out.push(line.to_string());
    }
    out
}

fn parse_cfg(lines: &[String]) -> Vec<Block> {
    let mut blocks: Vec<Block> = Vec::new();
    for line in lines {
        if line.ends_with(':') && !line.starts_with(".amdhsa") {
            let mut b = Block::default();
            b.label = Some(line.trim_end_matches(':').to_string());
            blocks.push(b);
        } else {
            if blocks.is_empty() {
                blocks.push(Block::default());
            }
            let (m, ops) = split_ops(line);
            if !m.is_empty() {
                blocks.last_mut().unwrap().insts.push((m, ops));
            }
        }
    }
    let label_of: Vec<Option<String>> = blocks.iter().map(|b| b.label.clone()).collect();
    let find = |name: &str| -> Option<usize> { label_of.iter().position(|l| l.as_deref() == Some(name)) };
    for i in 0..blocks.len() {
        let n_inst = blocks[i].insts.len();
        if n_inst == 0 {
            if i + 1 < blocks.len() {
                blocks[i].succs.push(i + 1);
            }
            continue;
        }
        let last = blocks[i].insts[n_inst - 1].0.clone();
        let target = blocks[i].insts[n_inst - 1].1.first().cloned();
        // This emitter can put a conditional branch and an unconditional fallthrough branch in the
        // same block (`s_cbranch_* T` followed by `s_branch F`), so look one instruction further.
        let prev = if n_inst >= 2 { Some(blocks[i].insts[n_inst - 2].clone()) } else { None };
        if last == "s_branch" {
            if let Some(t) = target.as_deref().and_then(|t| find(t.trim())) {
                blocks[i].succs.push(t);
            }
            if let Some((pm, pops)) = prev {
                if pm.starts_with("s_cbranch") {
                    if let Some(t) = pops.first().and_then(|t| find(t.trim())) {
                        blocks[i].succs.push(t);
                    }
                }
            }
        } else if last.starts_with("s_cbranch") {
            if let Some(t) = target.as_deref().and_then(|t| find(t.trim())) {
                blocks[i].succs.push(t);
            }
            if i + 1 < blocks.len() {
                blocks[i].succs.push(i + 1);
            }
        } else if last == "s_endpgm" {
            // no successors
        } else if i + 1 < blocks.len() {
            blocks[i].succs.push(i + 1);
        }
    }
    blocks
}

fn declared(asm: &str, directive: &str) -> Option<u32> {
    asm.lines().find_map(|l| l.trim().strip_prefix(directive).and_then(|r| r.trim().parse::<u32>().ok()))
}

/// ABI inputs available at function entry: `v0` workitem id, `s0:s1` kernarg pointer, `s2..s4`
/// workgroup ids (the descriptor declares exactly those).
fn implicit_inputs() -> HashSet<u32> {
    let mut s = HashSet::new();
    s.insert(0u32); // v0
    for r in 0..5u32 {
        s.insert(512 + r); // s0..s4
    }
    s
}

const VGPR_KEYS: u32 = 512;
const SGPR_KEYS: u32 = 128;

fn universe() -> HashSet<u32> {
    (0..VGPR_KEYS + SGPR_KEYS).collect()
}

/// Run every check on one kernel's assembly text and return human-readable violations.
fn audit_asm(kernel: &str, asm: &str) -> Vec<String> {
    let mut bad = Vec::new();
    let decl_v = match declared(asm, ".amdhsa_next_free_vgpr") {
        Some(v) => v,
        None => {
            return vec![format!("{kernel}: .amdhsa_next_free_vgpr missing")];
        }
    };
    let decl_s = match declared(asm, ".amdhsa_next_free_sgpr") {
        Some(v) => v,
        None => {
            return vec![format!("{kernel}: .amdhsa_next_free_sgpr missing")];
        }
    };
    if decl_v == 0 || decl_v > 256 {
        bad.push(format!("{kernel}: implausible next_free_vgpr {decl_v}"));
    }
    if decl_s == 0 || decl_s > 128 {
        bad.push(format!("{kernel}: implausible next_free_sgpr {decl_s}"));
    }
    let blocks = parse_cfg(&instruction_region(asm, kernel));
    if blocks.is_empty() {
        return vec![format!("{kernel}: empty instruction region")];
    }
    // The kernel descriptor must start 64 bytes after the code it describes (same invariant the old
    // `descriptor_covers_all_registers` test asserted).
    if !asm.contains(".p2align 6\n.amdhsa_kernel") {
        bad.push(format!("{kernel}: descriptor alignment missing"));
    }

    // --- checks (1) descriptor range and (2) MFMA tuple shape, per instruction ---
    for b in &blocks {
        for (m, ops) in &b.insts {
            for (i, o) in ops.iter().enumerate() {
                for r in scan_regs(o) {
                    let (decl, class) = if r.scalar { (decl_s, "s") } else { (decl_v, "v") };
                    if r.base + r.width > decl {
                        bad.push(format!(
                            "{kernel}: {m} operand {} uses {class}[{}:{}] but next_free_{} = {decl}",
                            i + 1,
                            r.base,
                            r.base + r.width - 1,
                            if r.scalar { "sgpr" } else { "vgpr" },
                        ));
                    }
                    if m.starts_with("v_mfma") && r.width > 1 && !r.scalar {
                        if r.base % r.width != 0 || r.base % 2 != 0 {
                            bad.push(format!(
                                "{kernel}: {m} tuple v[{}:{}] (width {}) is not aligned to its width",
                                r.base,
                                r.base + r.width - 1,
                                r.width
                            ));
                        }
                    }
                }
            }
        }
    }

    // --- check (3): must-assignment dataflow, every read defined earlier on every path ---
    let mut defs: Vec<HashSet<u32>> = Vec::with_capacity(blocks.len());
    let mut uses_before_def: Vec<HashSet<u32>> = Vec::with_capacity(blocks.len());
    for b in &blocks {
        let mut defined_here: HashSet<u32> = HashSet::new();
        let mut ubd: HashSet<u32> = HashSet::new();
        for (m, ops) in &b.insts {
            let (d, u) = read_write(m, ops);
            for r in &u {
                for k in r.keys() {
                    if !defined_here.contains(&k) {
                        ubd.insert(k);
                    }
                }
            }
            for r in &d {
                for k in r.keys() {
                    defined_here.insert(k);
                }
            }
        }
        uses_before_def.push(ubd);
        defs.push(defined_here);
    }
    let n = blocks.len();
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, b) in blocks.iter().enumerate() {
        for &s in &b.succs {
            if s < n {
                preds[s].push(i);
            }
        }
    }
    let entry = 0usize;
    let univ = universe();
    let mut in_set: Vec<HashSet<u32>> = vec![univ.clone(); n];
    in_set[entry] = implicit_inputs();
    let mut out_set: Vec<HashSet<u32>> = vec![univ.clone(); n];
    for _ in 0..(n * 4 + 16) {
        let mut changed = false;
        for b in 0..n {
            let new_in = if b == entry {
                implicit_inputs()
            } else if preds[b].is_empty() {
                univ.clone()
            } else {
                let mut it = preds[b].iter();
                let first = it.next().unwrap();
                let mut acc = out_set[*first].clone();
                for &p in it {
                    acc = acc.intersection(&out_set[p]).copied().collect();
                }
                acc
            };
            let mut new_out = new_in.clone();
            new_out.extend(defs[b].iter().copied());
            if new_in != in_set[b] {
                in_set[b] = new_in;
                changed = true;
            }
            if new_out != out_set[b] {
                out_set[b] = new_out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // reachability
    let mut reach = vec![false; n];
    let mut stack = vec![entry];
    reach[entry] = true;
    while let Some(b) = stack.pop() {
        for &s in &blocks[b].succs {
            if s < n && !reach[s] {
                reach[s] = true;
                stack.push(s);
            }
        }
    }
    for b in 0..n {
        if !reach[b] {
            continue;
        }
        let mut v: Vec<u32> = uses_before_def[b].iter().copied().filter(|k| !in_set[b].contains(k)).collect();
        v.sort_unstable();
        for k in v {
            let (class, idx) = if k >= VGPR_KEYS { ("s", k - VGPR_KEYS) } else { ("v", k) };
            bad.push(format!(
                "{kernel}: block {} reads {class}{idx} before it is written on every path",
                blocks[b].label.clone().unwrap_or_else(|| format!("#{b}"))
            ));
        }
    }
    bad
}

fn audit_kernel(name: &str) -> Vec<String> {
    let f = kernel_by_name(name).unwrap_or_else(|| panic!("kernel {name} not registered"));
    let c = compile(&f).unwrap_or_else(|e| panic!("{name}: own backend: {e}"));
    audit_asm(name, &c.asm)
}

/// Build-time variants of the FMHA kernels read their options from the environment while the DSL
/// function is *constructed*, so the audit has to rebuild the kernels under each option set. The
/// known SGPR under-declaration only appeared with the DMA staging variant, so the environment is
/// process-global: both tests that build kernels take this lock.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The `probe_dma*` kernels are registered but deliberately not in `kernel_names()` (the LLVM tail
/// cannot emit their MIR), so a `kernel_names()` sweep alone would miss them.
const DMA_PROBES: &[&str] = &["probe_dma", "probe_dma_tile", "probe_dma_seq", "probe_dma_kread", "probe_dma_map"];

const DMA_OPTS: &str = "cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,svpk,svcorr,swz,dma";

#[test]
fn every_kernel_register_accounting_own_backend() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut bad = Vec::new();
    for name in kernel_names() {
        bad.extend(audit_kernel(name));
    }
    // Variants that change the lowering (and therefore the SGPR live ranges): the DMA path is the
    // one where the under-declared `next_free_sgpr` corrupted hardware before it was found.
    let variants: &[(&str, &str)] = &[("FMHA_OPT", DMA_OPTS), ("FMHA_OPT", "dma"), ("FMHA_LDSPAD", "8192")];
    let saved: Vec<(&str, Option<String>)> =
        ["FMHA_OPT", "FMHA_LDSPAD", "FMHA_ABLATE", "FMHA_PITCH"].iter().map(|v| (*v, std::env::var(v).ok())).collect();
    for (var, val) in variants {
        std::env::set_var(var, val);
        for name in kernel_names() {
            bad.extend(audit_kernel(name));
        }
        for name in DMA_PROBES {
            bad.extend(audit_kernel(name));
        }
        std::env::remove_var(var);
    }
    for (var, val) in saved {
        match val {
            Some(v) => std::env::set_var(var, v),
            None => std::env::remove_var(var),
        }
    }
    // The DMA probes are compiled (own backend only) but not enumerated by kernel_names().
    for name in DMA_PROBES {
        if kernel_by_name(name).is_some() {
            bad.extend(audit_kernel(name));
        }
    }
    if !bad.is_empty() {
        panic!(
            "{} register-accounting violation(s):\n{}",
            bad.len(),
            bad.join("\n")
        );
    }
}

fn llc_available() -> bool {
    let t = LlvmTools::locate();
    let ok = std::process::Command::new(t.tool("llc"))
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED: llc not runnable at {}", t.bin.display());
    }
    ok
}

#[test]
fn llvm_tail_register_accounting() {
    // Opt-in: the LLVM tail is covered by tail_invariance.rs; re-checking the descriptor/range
    // invariants on its assembly for every kernel costs another llc run per kernel.
    if std::env::var("AMDC_TEST_LLVM").is_err() {
        eprintln!("SKIPPED: set AMDC_TEST_LLVM=1 to audit the LLVM tail assembly too");
        return;
    }
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if !llc_available() {
        return;
    }
    let tail = LlvmTail::new();
    let mut bad = Vec::new();
    for name in kernel_names() {
        let f = kernel_by_name(name).unwrap();
        let c = compile_with(&f, &tail).unwrap_or_else(|e| panic!("{name}: llvm tail: {e}"));
        // checks (1)+(2) only: the def-use dataflow is written for our own emitter's operand forms.
        bad.extend(audit_asm(name, &c.asm).into_iter().filter(|m| {
            m.contains("next_free_") || m.contains("tuple") || m.contains("implausible") || m.contains("missing")
        }));
    }
    if !bad.is_empty() {
        panic!("{} LLVM-tail register-accounting violation(s):\n{}", bad.len(), bad.join("\n"));
    }
}

fn synthetic(instructions: &[&str], decl_v: u32, decl_s: u32) -> String {
    let mut s = String::from("// synthetic\n.text\n.globl k\n.p2align 8\n.type k,@function\nk:\nBB0:\n");
    for i in instructions {
        s.push_str("    ");
        s.push_str(i);
        s.push('\n');
    }
    s.push_str(".size k, .-k\n\n.p2align 6\n.amdhsa_kernel k\n");
    s.push_str(&format!("    .amdhsa_next_free_vgpr {decl_v}\n"));
    s.push_str(&format!("    .amdhsa_next_free_sgpr {decl_s}\n"));
    s.push_str(".end_amdhsa_kernel\n");
    s
}

/// The audit must actually detect the bug class it is meant to catch; otherwise a green run proves
/// nothing. Each case below is a different way the real bug can look.
#[test]
fn checker_detects_the_bug_class() {
    // (1) descriptor under-declares a VGPR that only appears as a tuple operand
    let asm = synthetic(&["ds_write_b128 v[2:3], v[60:67], off", "s_endpgm"], 64, 8);
    assert!(
        audit_asm("k", &asm).iter().any(|m| m.contains("next_free_vgpr") && m.contains("60:67")),
        "tuple-only VGPR use not detected: {:?}",
        audit_asm("k", &asm)
    );
    // (2) descriptor under-declares SGPRs by one
    let asm = synthetic(&["s_mov_b32 s64, 0", "s_endpgm"], 4, 64);
    assert!(
        audit_asm("k", &asm).iter().any(|m| m.contains("next_free_sgpr")),
        "SGPR overrun not detected: {:?}",
        audit_asm("k", &asm)
    );
    // (3) misaligned MFMA B operand (must be even)
    let asm = synthetic(
        &[
            "v_mfma_f32_16x16x16_f16 v[4:7], v[2:3], v[5:6], v[4:7]",
            "s_endpgm",
        ],
        8,
        8,
    );
    assert!(
        audit_asm("k", &asm).iter().any(|m| m.contains("not aligned")),
        "misaligned MFMA tuple not detected: {:?}",
        audit_asm("k", &asm)
    );
    // (4) read of a register that is not written on every path (stale read)
    let asm = String::from(
        "// synthetic\n.text\n.globl k\n.type k,@function\nk:\nBB0:\n    s_cbranch_scc1 BB2\n    s_branch BB1\nBB1:\n    v_mov_b32_e32 v2, 1\n    s_branch BB3\nBB2:\n    v_add_f32_e32 v3, v2, 1\n    s_endpgm\nBB3:\n    s_endpgm\n.size k, .-k\n.amdhsa_kernel k\n    .amdhsa_next_free_vgpr 8\n    .amdhsa_next_free_sgpr 8\n.end_amdhsa_kernel\n",
    );
    assert!(
        audit_asm("k", &asm).iter().any(|m| m.contains("before it is written")),
        "stale read not detected: {:?}",
        audit_asm("k", &asm)
    );
    // (5) a clean snippet must produce no violation (no false positives from the checker itself)
    let asm = synthetic(&["v_mov_b32_e32 v1, 1", "v_add_f32_e32 v2, v1, v1", "s_endpgm"], 8, 8);
    let clean = audit_asm("k", &asm);
    assert!(clean.is_empty(), "clean snippet flagged: {clean:?}");
}
