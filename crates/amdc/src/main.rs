//! `amdc` - command line driver for the AMDGPU Rust DSL compiler.

#![forbid(unsafe_code)]

use amdgpu_compiler::backend::{self, probe, LlvmTail, LlvmTools, TailBackend};
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!("usage:");
    eprintln!("  amdc list");
    eprintln!("  amdc dump <kernel>");
    eprintln!("  amdc emit <kernel> [--out <file.s>] [--backend llvm-tail|own] [--mir-out <file.mir>] [--llvm-tail-drop-auto-waits] [--sched | --sched=auto | --sched=ld]");
    eprintln!("      --backend llvm-tail runs the pinned llc tail (ROCM_LLVM_BIN overrides the tool dir);");
    eprintln!("      --mir-out writes the exported post-RA MIR only (no llc needed) and exits.");
    eprintln!("  amdc backend probe|check|write-lock   version probe / verify / rewrite backend.lock");
}

fn backend_cmd(args: &[String]) -> ExitCode {
    let tools = LlvmTools::locate();
    let sub = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let report = match probe::probe(&tools) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("probe failed: {e}");
            return ExitCode::from(1);
        }
    };
    match sub {
        "probe" => {
            print!("{}", report.render_lock());
            match report.validate() {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("UNSAFE TAIL: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "check" => {
            if let Err(e) = report.validate() {
                eprintln!("UNSAFE TAIL: {e}");
                return ExitCode::from(1);
            }
            let lock = match probe::Lock::find_and_load() {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::from(1);
                }
            };
            match lock.check(&report) {
                Ok(()) => {
                    println!("backend.lock OK: {} start={} tail={} passes", report.version, report.start_before, report.tail.len());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("BACKEND LOCK MISMATCH: {e}");
                    ExitCode::from(1)
                }
            }
        }
        "write-lock" => {
            if let Err(e) = report.validate() {
                eprintln!("UNSAFE TAIL: {e}");
                return ExitCode::from(1);
            }
            let path = probe::Lock::find().unwrap_or_else(|| "backend.lock".into());
            if let Err(e) = fs::write(&path, report.render_lock()) {
                eprintln!("cannot write {}: {e}", path.display());
                return ExitCode::from(1);
            }
            eprintln!("wrote {}", path.display());
            ExitCode::SUCCESS
        }
        _ => {
            usage();
            ExitCode::from(2)
        }
    }
}

/// `amdc mir-roundtrip <in.mir> <out.mir>`: parse all `body: |` sections and print them back.
fn mir_roundtrip_cmd(args: &[String]) -> ExitCode {
    let (Some(i), Some(o)) = (args.get(1), args.get(2)) else {
        usage();
        return ExitCode::from(2);
    };
    let text = match fs::read_to_string(i) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{i}: {e}");
            return ExitCode::from(1);
        }
    };
    use amdgpu_compiler::backend::mir_roundtrip::Level;
    let lv = match args.get(3).map(|s| s.as_str()) {
        Some("--level=flags") => Level::Flags,
        Some("--level=reflag") => Level::Reflag,
        Some("--level=all") => Level::All,
        _ => Level::Identity,
    };
    let (out, st) = amdgpu_compiler::backend::mir_roundtrip::roundtrip_text_level(&text, lv);
    if let Err(e) = fs::write(o, out) {
        eprintln!("{o}: {e}");
        return ExitCode::from(1);
    }
    println!(
        "STATS files=1 bodies={} insts={} parsed={} opaque={} bundles={} changed={}",
        st.bodies, st.insts, st.parsed, st.opaque, st.bundles, st.changed_lines
    );
    ExitCode::SUCCESS
}

/// `amdc mir-table-report <files...>`: how well does our opcode table match LLVM's own MIR?
fn mir_table_report_cmd(args: &[String]) -> ExitCode {
    let mut files = Vec::new();
    for f in &args[1..] {
        match fs::read_to_string(f) {
            Ok(t) => files.push((f.clone(), t)),
            Err(e) => eprintln!("{f}: {e}"),
        }
    }
    let t = amdgpu_compiler::backend::llvm20::Table::load();
    let r = amdgpu_compiler::backend::mir_roundtrip::table_report(&files, &t);
    println!("instructions={} matched_exact={} implicit_operands_differ={} explicit_shape_mismatch={} unknown_opcode={}", r.total, r.matched, r.implicit_differs, r.shape_mismatch, r.unknown_opcode);
    println!("matched opcodes: {}", r.matched_by_opcode.len());
    println!("-- explicit operand shape differs from the table row (opcode: example)");
    for (o, ex) in &r.mismatch_examples {
        println!("  {o}: {ex}");
    }
    let mut unk: Vec<_> = r.unknown_by_opcode.iter().collect();
    unk.sort_by(|a, b| b.1.cmp(a.1));
    println!("-- explicit operands match but the implicit operand set differs (opcode: instruction | implicit set)");
    for (o, ex) in &r.implicit_examples {
        println!("  {o}: {ex}");
    }
    println!("-- top unknown opcodes (not in the table)");
    for (o, n) in unk.iter().take(40) {
        println!("  {n:>6} {o}");
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "list" => {
            for k in amdgpu_compiler::kernel_names() {
                println!("{k}");
            }
            ExitCode::SUCCESS
        }
        "mops" => {
            // amdc mops <kernel>: the distinct machine opcodes (MOp variants) of the final MIR of a kernel, one per line (the coverage sweep tools/llvm_opcode_sweep.sh uses it)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let Some(f) = amdgpu_compiler::kernel_by_name(name) else {
                eprintln!("unknown kernel: {name}");
                return ExitCode::from(2);
            };
            match amdgpu_compiler::compile_mir(&f) {
                Ok(m) => {
                    let mut set = std::collections::BTreeSet::new();
                    for b in &m.blocks {
                        for i in &b.insts {
                            set.insert(format!("{:?}", i.op));
                        }
                    }
                    for s in set {
                        println!("{s}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "backend" => backend_cmd(&args),
        "mir-roundtrip" => mir_roundtrip_cmd(&args),
        "mir-table-report" => mir_table_report_cmd(&args),
        "simmodel" => {
            // amdc simmodel <kernel> [--sched] : model cycles/iteration of every loop for 1,2 waves per SIMD
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let Some(f) = amdgpu_compiler::kernel_by_name(name) else {
                eprintln!("unknown kernel: {name}");
                return ExitCode::from(2);
            };
            if args.iter().any(|a| a == "--sched=auto") {
                env::set_var("AMDC_SCHED", "auto");
            } else if args.iter().any(|a| a == "--sched") {
                env::set_var("AMDC_SCHED", "1");
            }
            match amdgpu_compiler::compile_mir(&f) {
                Ok(m) => {
                    for r in amdgpu_compiler::sched::sim::report(&m, &[1, 2]) {
                        println!("SIM {} loop@{} blocks={} insts={} {}", name, r.header_label, r.blocks, r.insts,
                            r.cycles_per_iter.iter().map(|(w, c)| format!("waves{}={:.1}", w, c)).collect::<Vec<_>>().join(" "));
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "dump" => {
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            match amdgpu_compiler::kernel_by_name(name) {
                Some(f) => {
                    print!("{}", f.dump());
                    ExitCode::SUCCESS
                }
                None => {
                    eprintln!("unknown kernel: {name}");
                    ExitCode::from(2)
                }
            }
        }
        "emit" => {
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let Some(f) = amdgpu_compiler::kernel_by_name(name) else {
                eprintln!("unknown kernel: {name}");
                return ExitCode::from(2);
            };
            for w in amdgpu_dsl::lint(&f) {
                eprintln!("warning: {name}: {w}");
            }
            let mut out_path: Option<String> = None;
            let mut mir_out: Option<String> = None;
            let mut backend_name = env::var("AMDC_BACKEND").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "auto".to_string());
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--out" {
                    out_path = args.get(i + 1).cloned();
                    i += 2;
                } else if args[i] == "--mir-out" {
                    mir_out = args.get(i + 1).cloned();
                    i += 2;
                } else if let Some(b) = args[i].strip_prefix("--backend=") {
                    backend_name = b.to_string();
                    i += 1;
                } else if args[i] == "--scalarize" {
                    let cur = env::var("AMDC_LOWER_OPTS").ok().filter(|v| !v.is_empty()).unwrap_or_else(|| "dce,cmpfold".to_string());
                    env::set_var("AMDC_LOWER_OPTS", format!("{cur},scalarize"));
                    i += 1;
                } else if args[i] == "--sched=ld" {
                    env::set_var("AMDC_SCHED", "ld");
                    i += 1;
                } else if args[i] == "--sched=auto" {
                    env::set_var("AMDC_SCHED", "auto");
                    i += 1;
                } else if args[i] == "--sched" {
                    env::set_var("AMDC_SCHED", "1");
                    i += 1;
                } else if args[i] == "--llvm-tail-drop-auto-waits" {
                    env::set_var("AMDC_DROP_AUTO_WAITS", "1");
                    i += 1;
                } else if args[i] == "--backend" {
                    backend_name = args.get(i + 1).cloned().unwrap_or_default();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if let Some(p) = mir_out {
                return match amdgpu_compiler::export_mir(&f, &LlvmTail::new()) {
                    Ok(mir) => match fs::write(&p, mir) {
                        Ok(()) => {
                            eprintln!("wrote {p}");
                            ExitCode::SUCCESS
                        }
                        Err(e) => {
                            eprintln!("cannot write {p}: {e}");
                            ExitCode::from(1)
                        }
                    },
                    Err(e) => {
                        eprintln!("compile error:\n{e}");
                        ExitCode::from(1)
                    }
                };
            }
            let (resolved, note) = amdgpu_compiler::resolve_backend(&backend_name, &f);
            if let Some(n) = note {
                eprintln!("note: {n}");
            }
            let backend_name = resolved;
            let Some(tail) = backend::backend_by_name(&backend_name) else {
                eprintln!("unknown backend: {backend_name} (own | llvm-tail | auto)");
                return ExitCode::from(2);
            };
            let tail: &dyn TailBackend = tail.as_ref();
            match amdgpu_compiler::compile_with(&f, tail) {
                Ok(c) => {
                    match out_path {
                        Some(p) => {
                            if let Err(e) = fs::write(&p, &c.asm) {
                                eprintln!("cannot write {p}: {e}");
                                return ExitCode::from(1);
                            }
                            eprintln!("wrote {p} (backend={} vgpr={} sgpr={})", tail.name(), c.vgpr, c.sgpr);
                        }
                        None => print!("{}", c.asm),
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        other => {
            eprintln!("unknown command: {other}");
            usage();
            ExitCode::from(2)
        }
    }
}