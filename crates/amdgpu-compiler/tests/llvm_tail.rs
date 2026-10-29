//! Tests for the LLVM-tail backend. Tests that need the pinned llc skip (loudly) when it cannot
//! be run; set ROCM_LLVM_BIN to the llvm/bin directory to force them.

use amdgpu_compiler::backend::llvm20::{self, SigClass, Table};
use amdgpu_compiler::backend::{probe, LlvmTail, LlvmTools, TailBackend, ALL_MOPS};
use amdgpu_compiler::mir::*;
use amdgpu_compiler::regalloc::Alloc;
use amdgpu_compiler::{compile, compile_with, kernel_by_name, kernel_names};
use std::collections::HashMap;

fn llc_available() -> bool {
    let t = LlvmTools::locate();
    let ok = std::process::Command::new(t.tool("llc")).arg("--version").output().map(|o| o.status.success()).unwrap_or(false);
    if !ok {
        eprintln!("SKIPPED: llc not runnable at {} (set ROCM_LLVM_BIN)", t.bin.display());
    }
    ok
}

#[test]
fn table_has_a_row_for_every_mop() {
    let t = Table::load();
    for op in ALL_MOPS {
        let name = format!("{op:?}");
        let has = t.rows.keys().any(|k| k == &name || k.starts_with(&format!("{name}:")));
        assert!(has, "llvm20.tbl lacks a row for {name}");
    }
}

#[test]
fn own_tail_is_the_default_and_unchanged() {
    for k in ["attention_fwd", "fmha_fwd_hd128_mxfp8_t"] {
        let f = kernel_by_name(k).unwrap();
        let a = compile(&f).unwrap().asm;
        let b = compile_with(&f, &amdgpu_compiler::backend::OwnTail).unwrap().asm;
        assert_eq!(a, b, "{k}");
    }
}

#[test]
fn every_kernel_exports_mir() {
    let tail = LlvmTail::new();
    for k in kernel_names() {
        let f = kernel_by_name(k).unwrap();
        let mir = amdgpu_compiler::export_mir(&f, &tail).unwrap_or_else(|e| panic!("{k}: {e}"));
        assert!(mir.contains("body:"), "{k}");
        assert!(mir.contains("S_ENDPGM"), "{k}");
        assert!(!mir.contains('{') || !mir.contains("{d}"), "{k}: unexpanded token");
    }
}

/// One tiny MIR per table row, through the pinned llc with -verify-machineinstrs.
#[test]
fn opcode_probe_every_row_assembles() {
    if !llc_available() {
        return;
    }
    let t = Table::load();
    let tail = LlvmTail::new();
    let mut failures = Vec::new();
    let mut keys: Vec<&String> = t.rows.keys().filter(|k| !k.starts_with("T_") && !k.starts_with("X_")).collect();
    keys.sort();
    for key in keys {
        let row = &t.rows[key];
        let opname = key.split(':').next().unwrap();
        let op = *ALL_MOPS.iter().find(|o| format!("{o:?}") == opname).expect("row for unknown MOp");
        if row.text.len() == 1 && row.text[0] == "DROP" {
            continue;
        }
        // Sample function: one vreg per signature slot.
        let mut vregs = Vec::new();
        let mut phys = Vec::new();
        let (mut vnext, mut snext) = (8u32, 8u32);
        let mut alloc = |c: RClass, vregs: &mut Vec<RClass>, phys: &mut Vec<u32>| -> u32 {
            let n = c.dwords();
            let p = if c.is_vector() {
                vnext = (vnext + n - 1) / n * n;
                let p = vnext;
                vnext += n;
                p
            } else {
                snext = (snext + n - 1) / n * n;
                let p = snext;
                snext += n;
                p
            };
            vregs.push(c);
            phys.push(p);
            (vregs.len() - 1) as u32
        };
        let mut inst = MInst::new(op);
        if let Some(c) = row.dst {
            inst.dst = Some(alloc(c, &mut vregs, &mut phys));
        }
        for s in &row.srcs {
            match s {
                SigClass::Reg(c) => {
                    let r = alloc(*c, &mut vregs, &mut phys);
                    inst.srcs.push(Src::Reg(r));
                }
                SigClass::Imm => inst.srcs.push(Src::Imm(3)),
            }
        }
        inst.imm = match op {
            MOp::VWriteSub | MOp::VReadSub => 1,
            MOp::SWaitVm | MOp::SWaitLgkm => 3,
            MOp::VReadPhys | MOp::SReadPhys => 0,
            _ => 16,
        };
        let mut b = Block::new(0);
        b.insts.push(inst);
        let scalar: Vec<bool> = vregs.iter().map(|c| !c.is_vector()).collect();
        let m = MFunction {
            name: "probe".into(),
            args: vec![],
            kernarg_size: 0,
            workgroup_size: 256,
            lds_bytes: 0,
            blocks: vec![b],
            vregs,
            loops: vec![],
            alias: HashMap::new(),
            elide_fallthrough: false,
        };
        let a = Alloc { scalar, phys, max_v: vnext, max_s: snext };
        let mir = match tail.export_mir(&m, &a) {
            Ok(x) => x,
            Err(e) => {
                failures.push(format!("{key}: export: {e}"));
                continue;
            }
        };
        let mut t2 = LlvmTail::new();
        t2.skip_lock = true;
        std::env::set_var("AMDC_LLC_ARGS", "-verify-machineinstrs");
        match t2.run_llc("probe", &mir, llvm20::START_PASS_PREFERENCE[0]) {
            Ok(_) => {}
            Err(e) => failures.push(format!("{key}: {}", e.lines().take(6).collect::<Vec<_>>().join(" | "))),
        }
    }
    assert!(failures.is_empty(), "{} opcode rows rejected by llc:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn backend_lock_matches_pinned_llvm() {
    if !llc_available() {
        return;
    }
    let tools = LlvmTools::locate();
    let report = probe::probe(&tools).unwrap();
    report.validate().unwrap();
    let lock = probe::Lock::find_and_load().unwrap();
    lock.check(&report).unwrap();
}

#[test]
fn llvm_tail_assembles_kernels() {
    if !llc_available() {
        return;
    }
    let only = std::env::var("AMDC_TEST_KERNELS").ok();
    let mut tail = LlvmTail::new();
    tail.skip_lock = false;
    let mut bad = Vec::new();
    for k in kernel_names() {
        if let Some(o) = &only {
            if !o.split(',').any(|x| x == *k) {
                continue;
            }
        }
        let f = kernel_by_name(k).unwrap();
        match compile_with(&f, &tail) {
            Ok(c) => {
                if !c.asm.contains("s_endpgm") {
                    bad.push(format!("{k}: no s_endpgm"));
                }
            }
            Err(e) => bad.push(format!("{k}: {}", e.lines().take(8).collect::<Vec<_>>().join(" | "))),
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
    let _: &dyn TailBackend = &tail;
}
#[test]
fn drop_auto_waits_removes_only_auto_waits() {
    let tail = LlvmTail::new();
    let mut total_dropped = 0usize;
    for k in ["attention_fwd", "fmha_fwd_hd128_mxfp8_t", "probe_lds_self2"] {
        let f = kernel_by_name(k).unwrap();
        let mut m = amdgpu_compiler::lower::lower(&f).unwrap();
        amdgpu_compiler::hazards::fix_value_hazards(&mut m);
        let a = amdgpu_compiler::regalloc::allocate(&m).unwrap();
        let keep = amdgpu_compiler::backend::llvm_mir::export_opts(&m, &a, false, false).unwrap().mir;
        let drop = amdgpu_compiler::backend::llvm_mir::export_opts(&m, &a, false, true).unwrap().mir;
        let n = |s: &str| s.lines().filter(|l| l.trim_start().starts_with("S_WAITCNT")).count();
        let explicit = m
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, MOp::SWaitVm | MOp::SWaitLgkm) && !i.auto)
            .count();
        assert_eq!(n(&drop), explicit, "{k}: only explicit waits may remain");
        assert!(n(&keep) >= n(&drop));
        total_dropped += n(&keep) - n(&drop);
    }
    assert!(total_dropped > 0);
    let _ = tail;
}
/// Kill/dead flags are exported from our liveness: the verifier must accept them on every kernel.
#[test]
fn exported_flags_pass_the_machine_verifier() {
    if !llc_available() {
        return;
    }
    let mut tail = LlvmTail::new();
    tail.extra_args = vec!["-verify-machineinstrs".to_string()];
    let only = std::env::var("AMDC_TEST_KERNELS").ok().filter(|s| !s.is_empty());
    let mut bad = Vec::new();
    for k in kernel_names() {
        if let Some(o) = &only {
            if !o.split(',').any(|x| x == *k) {
                continue;
            }
        }
        let f = kernel_by_name(k).unwrap();
        let (m, a) = match amdgpu_compiler::compile_mir_alloc(&f) {
            Ok(x) => x,
            Err(e) => {
                bad.push(format!("{k}: {e}"));
                continue;
            }
        };
        let mir = tail.export_mir(&m, &a).unwrap();
        if !mir.contains("killed") || !mir.contains("dead") {
            bad.push(format!("{k}: no kill/dead flags exported"));
        }
        if let Err(e) = tail.run_llc(k, &mir, llvm20::START_PASS_PREFERENCE[0]) {
            bad.push(format!("{k}: {}", e.lines().take(6).collect::<Vec<_>>().join(" | ")));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}