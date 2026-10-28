//! `amdc` - command line driver for the AMDGPU Rust DSL compiler.

use amdgpu_compiler::backend::{self, probe, LlvmTail, LlvmTools, TailBackend};
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!("usage:");
    eprintln!("  amdc list");
    eprintln!("  amdc dump <kernel>");
    eprintln!("  amdc emit <kernel> [--out <file.s>] [--backend own|llvm-tail] [--mir-out <file.mir>] [--llvm-tail-drop-auto-waits]");
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
        "backend" => backend_cmd(&args),
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
            let mut out_path: Option<String> = None;
            let mut mir_out: Option<String> = None;
            let mut backend_name = env::var("AMDC_BACKEND").unwrap_or_else(|_| "own".to_string());
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
            let Some(tail) = backend::backend_by_name(&backend_name) else {
                eprintln!("unknown backend: {backend_name} (own | llvm-tail)");
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