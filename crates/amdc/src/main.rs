//! `amdc` - command line driver for the AMDGPU Rust DSL compiler.

use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!("usage:");
    eprintln!("  amdc list");
    eprintln!("  amdc dump <kernel>");
    eprintln!("  amdc emit <kernel> [--out <file.s>]");
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
            match amdgpu_compiler::compile(&f) {
                Ok(c) => {
                    let mut out_path: Option<String> = None;
                    let mut i = 2;
                    while i < args.len() {
                        if args[i] == "--out" {
                            out_path = args.get(i + 1).cloned();
                            i += 2;
                        } else {
                            i += 1;
                        }
                    }
                    match out_path {
                        Some(p) => {
                            if let Err(e) = fs::write(&p, &c.asm) {
                                eprintln!("cannot write {p}: {e}");
                                return ExitCode::from(1);
                            }
                            eprintln!("wrote {p} (vgpr={} sgpr={})", c.vgpr, c.sgpr);
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
