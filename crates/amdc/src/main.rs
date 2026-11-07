//! `amdc` - command line driver for the AMDGPU Rust DSL compiler.

#![forbid(unsafe_code)]

use amdc::hostgen;

use amdgpu_compiler::backend::{self, probe, LlvmTail, LlvmTools, TailBackend};
use std::env;
use std::fs;
use std::process::ExitCode;

fn usage() {
    eprintln!("usage:");
    eprintln!("  amdc list");
    eprintln!("  amdc dump <kernel>");
    eprintln!("  amdc print-ir <kernel>|--from-ir <file> [--stage dsl|mir|mir-ra] [--out <file>]");
    eprintln!("  amdc emit <kernel> [--out <file.s>] [--backend llvm-tail|own] [--mir-out <file.mir>] [--sched | --sched=ld] [--specialize name=value[,..]]");
    eprintln!("      --specialize binds scalar kernargs to constants (shape specialization; docs/jit.md); the specialized kernel must verify strictly");
    eprintln!("  amdc regplan <kernel> [--specialize ..]   per weave-region / stage register peaks (allocator truth) + register-plan report");
    eprintln!("  amdc ir-check <file.ir> [--verify]      parse + check an irv1 DSL IR file (docs/ir-text.md); --from-ir <file> works with emit / gpu-verify / jit-info / print-ir / regplan");
    eprintln!("  amdc jit-info <kernel> [--specialize name=value[,..]] [--backend ..]   JSON metadata for the Python JIT (docs/jit.md)");
    eprintln!("  amdc gpu-verify <kernel>|--all|--shipped [-v] [--strict] [--specialize ..]");
    eprintln!("      --backend llvm-tail runs the pinned llc tail (ROCM_LLVM_BIN overrides the tool dir);");
    eprintln!("      --mir-out writes the exported post-RA MIR only (no llc needed) and exits.");
    eprintln!("  amdc gen-host <kernel>|--shipped [--out <dir>]   generate C++ ABI/validate/launch header + JSON metadata");
    eprintln!("  amdc backend probe|check|write-lock   version probe / verify / rewrite backend.lock");
}

/// The `--specialize` bindings of the command line (`--specialize a=1,b=2` or `--specialize=a=1`, repeatable).
fn spec_bindings(args: &[String]) -> Result<Vec<(String, String)>, String> {
    let mut s: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--specialize" {
            s.push(args.get(i + 1).cloned().ok_or("--specialize needs name=value[,..]")?);
            i += 2;
        } else if let Some(v) = args[i].strip_prefix("--specialize=") {
            s.push(v.to_string());
            i += 1;
        } else {
            i += 1;
        }
    }
    amdgpu_dsl::specialize::parse_bindings(&s.join(","))
}

/// The kernel `name`, specialized by the command line's bindings (exit code 2 on a bad binding).
type Spec = (Vec<(String, u32, amdgpu_dsl::specialize::SpecValue)>, Vec<(u32, i64)>);

/// `--from-ir <file>` / `--from-ir=<file>`: the kernel comes from an irv1 text file instead of the registry (docs/ir-text.md).
fn from_ir_path(args: &[String]) -> Option<String> {
    args.windows(2).find(|w| w[0] == "--from-ir").map(|w| w[1].clone()).or_else(|| args.iter().find_map(|a| a.strip_prefix("--from-ir=").map(|s| s.to_string())))
}

/// The kernel of the command line: `--from-ir <file>` (parsed and checked) or the registered kernel `name`.
fn load_kernel(name: &str, args: &[String]) -> Result<amdgpu_dsl::Function, ExitCode> {
    if let Some(path) = from_ir_path(args) {
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("cannot read {path}: {e}");
                return Err(ExitCode::from(2));
            }
        };
        return amdgpu_dsl::ir_text::parse(&text).map_err(|e| {
            eprintln!("{path}: {e}");
            ExitCode::from(2)
        });
    }
    amdgpu_kernels::kernel_by_name(name).ok_or_else(|| {
        eprintln!("unknown kernel: {name}");
        ExitCode::from(2)
    })
}

/// `--from-ir` kernels and specialized kernels are verified strictly before use (unless `--no-verify`).
fn must_verify(args: &[String], spec: &Spec) -> bool {
    (from_ir_path(args).is_some() || !spec.0.is_empty()) && !args.iter().any(|a| a == "--no-verify")
}

fn kernel_spec(name: &str, args: &[String]) -> Result<(amdgpu_dsl::Function, Spec), ExitCode> {
    let f = load_kernel(name, args)?;
    let binds = match spec_bindings(args) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            return Err(ExitCode::from(2));
        }
    };
    if binds.is_empty() {
        return Ok((f, (Vec::new(), Vec::new())));
    }
    let vals = match amdgpu_dsl::specialize::check(&f, &binds) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return Err(ExitCode::from(2));
        }
    };
    let derived = match amdgpu_dsl::specialize::close(&f, &vals) {
        Ok(c) => c.derived,
        Err(e) => {
            eprintln!("{e}");
            return Err(ExitCode::from(2));
        }
    };
    match amdgpu_dsl::specialize::specialize(&f, &binds) {
        Ok(g) => Ok((g, (vals, derived))),
        Err(e) => {
            eprintln!("{e}");
            Err(ExitCode::from(2))
        }
    }
}

/// The strict certificate of a specialized kernel (stderr); Err when it does not verify.
fn certify_specialized(f: &amdgpu_dsl::Function) -> Result<(), ExitCode> {
    match amdgpu_compiler::gpu_verify::analyse(f) {
        Ok(r) => {
            let l = r.ledger();
            let c = &f.contract;
            let contracts = c.extents.len() + c.ka_ranges.len() + c.grid_defs.len() + c.ka_defs.len() + c.ka_sets.len() + c.sym_ranges.len();
            let errs = r.errors();
            eprintln!("verified {}: certificate: {} trusted assumes, {contracts} trusted contracts, {} unproven", f.name, l.total(), errs.len());
            if errs.is_empty() {
                Ok(())
            } else {
                for e in errs {
                    eprintln!("  [{}] {}", e.code, e.msg);
                }
                Err(ExitCode::from(1))
            }
        }
        Err(e) => {
            eprintln!("compile error:\n{e}");
            Err(ExitCode::from(1))
        }
    }
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
        "gen-host" => {
            // amdc gen-host <kernel>|--shipped [--out <dir>]
            let mut out = "build/hostgen".to_string();
            let mut i = 1usize;
            let mut which: Vec<String> = Vec::new();
            while i < args.len() {
                match args[i].as_str() {
                    "--out" => { i += 1; if let Some(d) = args.get(i) { out = d.clone(); } }
                    "--shipped" => which = ["fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_t128", "fmha_fwd_hd128_mxfp8_v3_kvsplit", "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"].iter().map(|s| s.to_string()).collect(),
                    k => which.push(k.to_string()),
                }
                i += 1;
            }
            if which.is_empty() { usage(); return ExitCode::from(2); }
            if let Err(e) = fs::create_dir_all(&out) { eprintln!("{out}: {e}"); return ExitCode::from(1); }
            let mut includes = String::from("// Generated by amdc gen-host --shipped. DO NOT EDIT.\n#pragma once\n");
            let mut aggregate = String::from("[\n");
            for (n, k) in which.iter().enumerate() {
                let Some(f) = amdgpu_kernels::kernel_by_name(k) else { eprintln!("unknown kernel: {k}"); return ExitCode::from(2); };
                let g = match hostgen::generate(&f) { Ok(g) => g, Err(e) => { eprintln!("{e}"); return ExitCode::from(1); } };
                let h = format!("{out}/{}_abi.h", g.file_stem);
                let j = format!("{out}/{}_abi.json", g.file_stem);
                if let Err(e) = fs::write(&h, &g.header).and_then(|_| fs::write(&j, &g.json)) { eprintln!("{e}"); return ExitCode::from(1); }
                includes.push_str(&format!("#include \"{}_abi.h\"\n", g.file_stem));
                if n > 0 { aggregate.push_str(",\n"); }
                aggregate.push_str(g.json.trim_end());
                println!("wrote {h} and {j}");
            }
            aggregate.push_str("\n]\n");
            let sj = format!("{out}/shipped_abi.json");
            let sh = format!("{out}/shipped_abi.h");
            if let Err(e) = fs::write(&sj, &aggregate).and_then(|_| fs::write(&sh, &includes)) { eprintln!("{e}"); return ExitCode::from(1); }
            println!("wrote {sh} and {sj}");
            ExitCode::SUCCESS
        }
        "list" => {
            for k in amdgpu_kernels::kernel_names() {
                println!("{k}");
            }
            ExitCode::SUCCESS
        }
        "gpu-verify" => {
            // amdc gpu-verify <kernel|--all> [--strict]: the GPU-semantics report (docs/gpu-semantics-verifier.md); exit 1 when --strict and an error was found
            let strict = args.iter().any(|a| a == "--strict");
            let names: Vec<String> = match args.get(1).map(|s| s.as_str()) {
                Some("--all") => amdgpu_kernels::kernel_names().iter().map(|s| s.to_string()).collect(),
                // the shipped default path (`fmha_fwd_hd128_mxfp8_auto`): the four v3 kernels (crates/amdgpu-tile/src/fmha.rs)
                Some("--shipped") => ["fmha_fwd_hd128_mxfp8_v3_vs128", "fmha_fwd_hd128_mxfp8_v3_t128", "fmha_fwd_hd128_mxfp8_v3_kvsplit", "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"].iter().map(|s| s.to_string()).collect(),
                Some(n) => vec![n.to_string()],
                None => {
                    usage();
                    return ExitCode::from(2);
                }
            };
            let mut bad = false;
            let (mut tot_trusted, mut tot_contracts, mut tot_unproven) = (0usize, 0usize, 0usize);
            for name in names {
                let (f, _) = match kernel_spec(&name, &args) {
                    Ok(x) => x,
                    Err(c) => return c,
                };
                match amdgpu_compiler::gpu_verify::analyse(&f) {
                    Ok(r) => {
                        let errs = r.errors().len();
                        println!("== {}: {} finding(s), {errs} error(s); proven {:?}", f.name, r.findings.len(), r.proven);
                        for fd in &r.findings {
                            if args.iter().any(|a| a == "-v") || fd.severity != amdgpu_compiler::gpu_verify::Severity::Info {
                                println!("  [{}{}] {:?}: {}", fd.code, if fd.alias.is_empty() || fd.alias == fd.code { String::new() } else { format!("|{}", fd.alias) }, fd.severity, fd.msg);
                            }
                        }
                        // the trust budget at a glance: explicit trusted assumptions, declared host contracts, unproven properties
                        let trusted = r.findings.iter().filter(|x| x.code == "GV-ASSUME" && x.msg.contains("TRUSTED") && !x.msg.contains("set_var")).count();
                        let c = &f.contract;
                        let contracts = c.extents.len() + c.ka_ranges.len() + c.grid_defs.len() + c.ka_defs.len() + c.ka_sets.len() + c.sym_ranges.len();
                        // the trust LEDGER: every escape is counted by kind
                        let l = r.ledger();
                        let trusted = l.total();
                        let c = &f.contract;
                        let contracts = c.extents.len() + c.ka_ranges.len() + c.grid_defs.len() + c.ka_defs.len() + c.ka_sets.len() + c.sym_ranges.len();
                        println!("   certificate: {trusted} trusted assumes ({} assume_trusted, {} tile_trust_periodic, {} region_trust), {contracts} trusted contracts, {errs} unproven", l.assume_trusted, l.tile_trust, l.region_trust);                        tot_trusted += trusted;
                        tot_contracts += contracts;
                        tot_unproven += errs;
                        bad |= errs > 0;
                    }
                    Err(e) => {
                        println!("== {name}: compile error: {e}");
                        bad = true;
                    }
                }
            }
            println!("certificate (all kernels listed): {tot_trusted} trusted assumes, {tot_contracts} trusted contracts, {tot_unproven} unproven");
            if strict && bad {
                ExitCode::from(1)
            } else {
                ExitCode::SUCCESS
            }
        }
        "mops" => {
            // amdc mops <kernel>: the distinct machine opcodes (MOp variants) of the final MIR of a kernel, one per line (the coverage sweep tools/llvm_opcode_sweep.sh uses it)
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let Some(f) = amdgpu_kernels::kernel_by_name(name) else {
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
            let Some(f) = amdgpu_kernels::kernel_by_name(name) else {
                eprintln!("unknown kernel: {name}");
                return ExitCode::from(2);
            };
            if args.iter().any(|a| a == "--sched") {
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
        "print-ir" => {
            let name = args.get(1).cloned().unwrap_or_default();
            if name.is_empty() || (name.starts_with("--") && from_ir_path(&args).is_none()) { usage(); return ExitCode::from(2); }
            let stage = args.windows(2).find(|w| w[0] == "--stage").map(|w| w[1].as_str()).unwrap_or("dsl");
            let f = match load_kernel(&name, &args) { Ok(f) => f, Err(c) => return c };
            let text = match stage {
                "dsl" => amdgpu_compiler::ir_print::print_dsl(&f),
                "mir" => match amdgpu_compiler::lower::lower(&f) {
                    Ok(m) => amdgpu_compiler::ir_print::print_mir(&m, None),
                    Err(e) => { eprintln!("lower failed: {e}"); return ExitCode::from(1) }
                },
                "mir-ra" => match amdgpu_compiler::compile_mir_alloc(&f) {
                    Ok((m, a)) => amdgpu_compiler::ir_print::print_mir(&m, Some(&a)),
                    Err(e) => { eprintln!("compile failed: {e}"); return ExitCode::from(1) }
                },
                _ => { eprintln!("unknown stage {stage} (dsl|mir|mir-ra)"); return ExitCode::from(2) }
            };
            // --out <file>: write the file directly (large kernels through a container's stdout pipe can be cut short)
            match args.windows(2).find(|w| w[0] == "--out").map(|w| w[1].clone()) {
                Some(p) => {
                    if let Err(e) = fs::write(&p, &text) {
                        eprintln!("cannot write {p}: {e}");
                        return ExitCode::from(1);
                    }
                }
                None => {
                    use std::io::Write;
                    let mut o = std::io::stdout().lock();
                    let _ = o.write_all(text.as_bytes());
                    let _ = o.flush();
                }
            }
            ExitCode::SUCCESS
        }
        "dump" => {
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            match amdgpu_kernels::kernel_by_name(name) {
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
        "ir-check" => {
            // amdc ir-check <file.ir> [--verify]: parse + check an irv1 file, re-print it (text fixpoint); --verify adds the strict GPU-semantics certificate
            let Some(path) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let text = match fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("cannot read {path}: {e}");
                    return ExitCode::from(2);
                }
            };
            let f = match amdgpu_dsl::ir_text::parse(&text) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("{path}: {e}");
                    return ExitCode::from(2);
                }
            };
            let again = amdgpu_dsl::ir_text::print(&f);
            let fix = amdgpu_dsl::ir_text::parse(&again).map(|g| format!("{g:?}") == format!("{f:?}")).unwrap_or(false);
            println!(
                "{path}: ok - kernel `{}`, {} statements, {} values, {} loop variables, {} args{}{}",
                f.name,
                f.stmts().len(),
                f.value_tys.len(),
                f.var_tys.len(),
                f.args.len(),
                if fix { "; re-print round-trips" } else { "; WARNING: re-print does not round-trip" },
                if f.compile_profile.is_none() { "; note: profile None - the compile policy is derived from the kernel NAME" } else { "" }
            );
            if args.iter().any(|a| a == "--verify") {
                if let Err(c) = certify_specialized(&f) {
                    return c;
                }
            }
            if fix {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        "regplan" => {
            // amdc regplan <kernel> [--specialize ..]: per weave-region / stage register peaks of the allocated kernel + the plan's report
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, _) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            match amdgpu_compiler::compile_mir_alloc(&f) {
                Ok((m, a)) => {
                    println!("{}: arch VGPRs {} AGPRs {}", f.name, a.max_v, amdgpu_compiler::regalloc::agpr_count(&m, &a));
                    for p in amdgpu_compiler::regplan::region_report(&m, &a) {
                        println!(
                            "region {:<28} insts {:>5}  vgpr footprint {:>3}  live dwords {:>3}  agpr footprint {:>3}  peak at b{} i{}: {}",
                            p.name,
                            p.insts,
                            p.vgpr_footprint_peak,
                            p.vgpr_live_peak,
                            p.agpr_footprint_peak,
                            p.at.0,
                            p.at.1,
                            p.live_at_peak.iter().take(12).cloned().collect::<Vec<_>>().join(" ")
                        );
                    }
                    for l in amdgpu_compiler::regplan::report(&m, &a) {
                        println!("plan: {l}");
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "jit-info" => {
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, spec) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            if must_verify(&args, &spec) {
                if let Err(c) = certify_specialized(&f) {
                    return c;
                }
            }
            let req = args.windows(2).find(|w| w[0] == "--backend").map(|w| w[1].clone()).or_else(|| args.iter().find_map(|a| a.strip_prefix("--backend=").map(|s| s.to_string()))).unwrap_or_else(|| "auto".to_string());
            let (resolved, _) = amdgpu_compiler::resolve_backend(&req, &f);
            let Some(tail) = backend::backend_by_name(&resolved) else {
                eprintln!("unknown backend: {resolved} (own | llvm-tail | auto)");
                return ExitCode::from(2);
            };
            match amdgpu_compiler::compile_with(&f, tail.as_ref()) {
                Ok(c) => {
                    let source = match from_ir_path(&args).and_then(|p| fs::read_to_string(p).ok()) {
                        Some(t) => format!("ir:{}", amdc::jit::fnv_hex(&t)),
                        None => "registry".to_string(),
                    };
                    println!("{}", amdc::jit::info_json_src(&f, &spec.0, &spec.1, tail.name(), amdc::jit::resources(&c.asm), &source));
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("compile error:\n{e}");
                    ExitCode::from(1)
                }
            }
        }
        "emit" => {
            let Some(name) = args.get(1) else {
                usage();
                return ExitCode::from(2);
            };
            let (f, spec) = match kernel_spec(name, &args) {
                Ok(x) => x,
                Err(c) => return c,
            };
            if must_verify(&args, &spec) {
                if let Err(c) = certify_specialized(&f) {
                    return c;
                }
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
                } else if args[i] == "--sched=ld" {
                    env::set_var("AMDC_SCHED", "ld");
                    i += 1;
                } else if args[i] == "--sched" {
                    env::set_var("AMDC_SCHED", "1");
                    i += 1;
                } else if args[i] == "--backend" {
                    backend_name = args.get(i + 1).cloned().unwrap_or_default();
                    i += 2;
                } else if args[i] == "--specialize" {
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