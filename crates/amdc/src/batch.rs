//! `amdc compile-batch` and the `--time` report of `amdc emit` (docs/compile-time.md): compile many kernels on N threads and report where the time goes.
//!
//! The compiler is a pure function of the kernel (`Function` -> assembly), so a batch is embarrassingly parallel: a work-stealing counter over the kernel list, one `std::thread::scope` worker per job, per-thread
//! phase timers (`amdgpu_codegen::timing`) merged at the end. The optional `--assemble` step runs the external `clang` (assembler + linker) per kernel, which is not part of the compiler proper.

use amdgpu_codegen::backend::OwnTail;
use amdgpu_codegen::timing::{self, Entry};
use amdgpu_codegen::{compile_with_options, CompileOptions};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Assembles `asm` into a code object with the external clang (`$AMD_LLVM_BIN/clang`, else `clang` on PATH); returns the wall time.
pub fn assemble_co(asm: &str, tag: &str) -> Result<Duration, String> {
    let dir = std::env::temp_dir();
    let (s, co) = (dir.join(format!("amdc-{}-{tag}.s", std::process::id())), dir.join(format!("amdc-{}-{tag}.co", std::process::id())));
    std::fs::write(&s, asm).map_err(|e| format!("write {}: {e}", s.display()))?;
    let clang = std::env::var("AMD_LLVM_BIN").ok().filter(|v| !v.is_empty()).map(|d| format!("{d}/clang")).unwrap_or_else(|| "clang".into());
    let t = Instant::now();
    let out = std::process::Command::new(&clang)
        .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6", "-o"])
        .arg(&co)
        .arg(&s)
        .output()
        .map_err(|e| format!("cannot run {clang}: {e}"))?;
    let d = t.elapsed();
    let _ = std::fs::remove_file(&s);
    let _ = std::fs::remove_file(&co);
    if !out.status.success() {
        return Err(format!("{clang}: {}", String::from_utf8_lossy(&out.stderr).chars().take(300).collect::<String>()));
    }
    Ok(d)
}

struct Row {
    name: String,
    build: Duration,
    compile: Duration,
    assemble: Option<Duration>,
    ok: bool,
    lines: usize,
    /// register-allocation runs of this compile (> 1: the retry ladders ran) and the time of the runs after the first one's share
    ra_runs: u32,
    ra_ms: f64,
}

/// `amdc compile-batch [--jobs N] [--names a,b,c] [--no-verify] [--assemble] [--repeat R] [--time] [--asm-dir DIR]`
pub fn compile_batch(args: &[String]) -> ExitCode {
    let opt = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let jobs = opt("--jobs").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    let repeat = opt("--repeat").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    let verify = !args.iter().any(|a| a == "--no-verify");
    let assemble = args.iter().any(|a| a == "--assemble");
    let show_phases = args.iter().any(|a| a == "--time");
    let asm_dir = opt("--asm-dir");
    let mut names: Vec<String> = match opt("--names") {
        Some(l) => l.split(',').filter(|s| !s.is_empty()).map(|s| s.to_string()).collect(),
        None => amdgpu_kernels::reachable::manifest_names().into_iter().collect(),
    };
    let base = names.clone();
    for _ in 1..repeat {
        names.extend(base.iter().cloned());
    }
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::with_capacity(names.len()));
    let phases: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
    let wall = Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..jobs {
            sc.spawn(|| {
                timing::enable(true);
                let mut acc: Vec<Entry> = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(k) = names.get(i) else { break };
                    let t = Instant::now();
                    let f = amdgpu_kernels::kernel_by_name(k);
                    let build = t.elapsed();
                    timing::add("dsl-build", build);
                    let Some(f) = f else {
                        rows.lock().unwrap().push(Row { name: k.clone(), build, compile: Duration::ZERO, assemble: None, ok: false, lines: 0, ra_runs: 0, ra_ms: 0.0 });
                        continue;
                    };
                    let t = Instant::now();
                    let res = compile_with_options(&f, &OwnTail, &CompileOptions { verify });
                    let compile = t.elapsed();
                    let (ok, lines, asm_t) = match &res {
                        Ok(c) => {
                            if let Some(d) = &asm_dir {
                                let _ = std::fs::write(format!("{d}/{k}.s"), &c.asm);
                            }
                            let a = if assemble {
                                let d = assemble_co(&c.asm, &format!("{i}")).ok();
                                if let Some(d) = d {
                                    timing::add("assemble-clang", d);
                                }
                                d
                            } else {
                                None
                            };
                            (true, c.asm.lines().count(), a)
                        }
                        Err(_) => (false, 0, None),
                    };
                    let ents = timing::take();
                    let ra = ents.iter().find(|e| e.name == "regalloc");
                    let (ra_runs, ra_ms) = ra.map_or((0, 0.0), |e| (e.runs, e.dur.as_secs_f64() * 1e3));
                    timing::merge(&mut acc, &ents);
                    rows.lock().unwrap().push(Row { name: k.clone(), build, compile, assemble: asm_t, ok, lines, ra_runs, ra_ms });
                }
                timing::merge(&mut acc, &timing::take());
                timing::enable(false);
                timing::merge(&mut phases.lock().unwrap(), &acc);
            });
        }
    });
    let wall = wall.elapsed();
    let mut rows = rows.into_inner().unwrap();
    let phases = phases.into_inner().unwrap();
    let n = rows.len();
    let okn = rows.iter().filter(|r| r.ok).count();
    let cpu: Duration = rows.iter().map(|r| r.build + r.compile + r.assemble.unwrap_or_default()).sum();
    let comp_cpu: Duration = rows.iter().map(|r| r.compile).sum();
    println!(
        "compile-batch: {n} kernel compiles ({okn} ok) on {jobs} thread(s), verify={verify}, assemble={assemble}: wall {:.3} s, {:.1} kernels/s, mean {:.2} ms of compile per kernel, cpu {:.3} s (utilization {:.0} %)",
        wall.as_secs_f64(),
        n as f64 / wall.as_secs_f64().max(1e-9),
        comp_cpu.as_secs_f64() * 1e3 / n.max(1) as f64,
        cpu.as_secs_f64(),
        100.0 * cpu.as_secs_f64() / (wall.as_secs_f64().max(1e-9) * jobs as f64)
    );
    let retried: Vec<&Row> = rows.iter().filter(|r| r.ra_runs > 1).collect();
    println!("retry ladders: {} of {} kernels ran register allocation more than once ({} runs in all, {:.1} s of the allocation time)", retried.len(), n, retried.iter().map(|r| r.ra_runs as usize).sum::<usize>(), retried.iter().map(|r| r.ra_ms).sum::<f64>() / 1e3);
    rows.sort_by(|a, b| b.compile.cmp(&a.compile));
    println!("slowest compiles (ms, register-allocation runs): {}", rows.iter().take(14).map(|r| format!("{} {:.1} (ra x{})", r.name, r.compile.as_secs_f64() * 1e3, r.ra_runs)).collect::<Vec<_>>().join(", "));
    let mut by_ms: Vec<f64> = rows.iter().map(|r| r.compile.as_secs_f64() * 1e3).collect();
    by_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !by_ms.is_empty() {
        let q = |p: f64| by_ms[((by_ms.len() - 1) as f64 * p) as usize];
        println!("compile ms per kernel: min {:.2}  p25 {:.2}  median {:.2}  p75 {:.2}  p90 {:.2}  max {:.2}", by_ms[0], q(0.25), q(0.5), q(0.75), q(0.9), by_ms[by_ms.len() - 1]);
    }
    if show_phases {
        println!("\nphases (summed over all kernels and threads; the share is of the summed cpu time):");
        print!("{}", timing::table(&phases, cpu));
    }
    for r in rows.iter().filter(|r| !r.ok) {
        eprintln!("error: {} did not compile", r.name);
    }
    let _ = rows.iter().map(|r| r.lines).sum::<usize>();
    if okn == n {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// The `amdc emit --time` report of one kernel: `build` = the DSL build, `compile` = the whole compile, the optional external assemble, then the phase table.
pub fn emit_report(name: &str, build: Duration, compile: Duration, asm: Option<&str>, assemble: bool, entries: Vec<Entry>) {
    let mut entries = entries;
    let mut assemble_d = None;
    if assemble {
        if let Some(a) = asm {
            match assemble_co(a, "emit") {
                Ok(d) => {
                    assemble_d = Some(d);
                    entries.push(Entry { name: "assemble-clang", dur: d, runs: 1 });
                }
                Err(e) => eprintln!("note: assemble skipped: {e}"),
            }
        }
    }
    let total = build + compile + assemble_d.unwrap_or_default();
    eprintln!("emit --time: {name}: dsl-build {:.3} ms, compile {:.3} ms{}", build.as_secs_f64() * 1e3, compile.as_secs_f64() * 1e3, assemble_d.map(|d| format!(", assemble (clang) {:.3} ms", d.as_secs_f64() * 1e3)).unwrap_or_default());
    eprint!("{}", timing::table(&entries, total));
}

/// `amdc ladder-hints [--out FILE] [--jobs N]`: compiles every manifest kernel WITHOUT its hint and writes `amdgpu_kernels::ladder_hints` (the kernels whose first success is not rung (0, 0)).
pub fn ladder_hints(args: &[String]) -> ExitCode {
    let opt = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let jobs = opt("--jobs").and_then(|v| v.parse::<usize>().ok()).unwrap_or(8).max(1);
    let names: Vec<String> = amdgpu_kernels::reachable::manifest_names().into_iter().collect();
    let next = AtomicUsize::new(0);
    let rows: Mutex<Vec<(String, u8, u8)>> = Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..jobs {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(k) = names.get(i) else { break };
                let Some(mut f) = amdgpu_kernels::registry::kernel_by_name_raw(k) else { continue };
                f.ladder_hint = None;
                if amdgpu_codegen::compile(&f).is_ok() {
                    let (o, r) = amdgpu_codegen::last_rung();
                    if (o, r) != (0, 0) {
                        rows.lock().unwrap().push((k.clone(), o, r));
                    }
                }
            });
        }
    });
    let mut rows = rows.into_inner().unwrap();
    rows.sort();
    let mut src = String::from("//! GENERATED by `amdc ladder-hints --out crates/amdgpu-kernels/src/ladder_hints.rs` (tools/regen_ladder_hints.sh): the first retry-ladder rung each kernel compiles at, for the kernels that do not compile\n//! at the first one (docs/compile-time.md). Skipping the rungs before it only skips attempts that run out of registers, so the output is the one of the unhinted compile (tests/ladder_hints.rs checks it).\n\n/// (kernel, outer rung, inner rung).\npub static HINTS: &[(&str, u8, u8)] = &[\n");
    for (k, o, r) in &rows {
        src.push_str(&format!("    ({k:?}, {o}, {r}),\n"));
    }
    src.push_str("];\n\npub fn lookup(name: &str) -> Option<(u8, u8)> {\n    HINTS.iter().find(|h| h.0 == name).map(|h| (h.1, h.2))\n}\n");
    match opt("--out") {
        Some(p) => match std::fs::write(&p, src) {
            Ok(()) => {
                eprintln!("wrote {p}: {} hinted kernels", rows.len());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("cannot write {p}: {e}");
                ExitCode::from(1)
            }
        },
        None => {
            print!("{src}");
            ExitCode::SUCCESS
        }
    }
}

/// `amdc co-check <oracle.tsv>`: every line "asm<TAB>hex bytes" (llvm-mc's encoding) against the in-process encoder; prints the mismatches and errors grouped by mnemonic.
pub fn co_check(args: &[String]) -> ExitCode {
    let Some(path) = args.get(1) else {
        eprintln!("usage: amdc co-check <oracle.tsv>");
        return ExitCode::from(2);
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("cannot read {path}");
        return ExitCode::from(2);
    };
    let (mut total, mut ok) = (0usize, 0usize);
    let mut bad: std::collections::BTreeMap<String, (usize, Vec<String>)> = std::collections::BTreeMap::new();
    let no_label = |_: &str| None;
    for l in text.lines() {
        let Some((asm, hex)) = l.split_once('\t') else { continue };
        total += 1;
        let want: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap_or(0)).collect();
        let mn = asm.split_whitespace().next().unwrap_or("").to_string();
        match amdgpu_target::gfx950_co::enc::encode_line(asm, 0, &no_label) {
            Ok(got) if got == want => ok += 1,
            Ok(got) => {
                let e = bad.entry(mn).or_default();
                e.0 += 1;
                if e.1.len() < 3 {
                    let h = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
                    e.1.push(format!("{asm}\n      want {}\n      got  {}", h(&want), h(&got)));
                }
            }
            Err(msg) => {
                let e = bad.entry(mn).or_default();
                e.0 += 1;
                if e.1.len() < 3 {
                    e.1.push(format!("{asm}\n      error: {msg}"));
                }
            }
        }
    }
    println!("co-check: {ok} of {total} lines encode identically; {} mnemonics with differences", bad.len());
    let limit = args.iter().position(|a| a == "--show").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok()).unwrap_or(40);
    for (mn, (n, ex)) in bad.iter().take(limit) {
        println!("{mn}: {n} lines");
        for e in ex.iter().take(2) {
            println!("    {e}");
        }
    }
    if ok == total {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// The code object clang + lld produce for `asm` (the cross-check / fallback path).
pub fn clang_co(asm: &str, tag: &str) -> Result<Vec<u8>, String> {
    let dir = std::env::temp_dir();
    let (s, co) = (dir.join(format!("amdc-{}-{tag}.s", std::process::id())), dir.join(format!("amdc-{}-{tag}.co", std::process::id())));
    std::fs::write(&s, asm).map_err(|e| format!("write {}: {e}", s.display()))?;
    let clang = std::env::var("AMD_LLVM_BIN").ok().filter(|v| !v.is_empty()).map(|d| format!("{d}/clang")).unwrap_or_else(|| "clang".into());
    let out = std::process::Command::new(&clang)
        .args(["-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6", "-o"])
        .arg(&co)
        .arg(&s)
        .output()
        .map_err(|e| format!("cannot run {clang}: {e}"))?;
    let bytes = std::fs::read(&co);
    let _ = std::fs::remove_file(&s);
    let _ = std::fs::remove_file(&co);
    if !out.status.success() {
        return Err(format!("{clang}: {}", String::from_utf8_lossy(&out.stderr).chars().take(300).collect::<String>()));
    }
    bytes.map_err(|e| e.to_string())
}

/// `amdc assemble-co <file.s> --out <file.co>`
pub fn assemble_co_cmd(args: &[String]) -> ExitCode {
    let (Some(src), Some(out)) = (args.get(1), args.windows(2).find(|w| w[0] == "--out").map(|w| w[1].clone())) else {
        eprintln!("usage: amdc assemble-co <file.s> --out <file.co>");
        return ExitCode::from(2);
    };
    let Ok(text) = std::fs::read_to_string(src) else {
        eprintln!("cannot read {src}");
        return ExitCode::from(2);
    };
    match amdgpu_target::gfx950_co::assemble(&text) {
        Ok(b) => match std::fs::write(&out, b) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("cannot write {out}: {e}");
                ExitCode::from(1)
            }
        },
        Err(e) => {
            eprintln!("assemble-co: {e}");
            ExitCode::from(1)
        }
    }
}

/// `amdc co-verify <dir> [--show N]`: for every `*.s` in the directory, the in-process code object against clang's.
pub fn co_verify(args: &[String]) -> ExitCode {
    let Some(dir) = args.get(1) else {
        eprintln!("usage: amdc co-verify <dir>");
        return ExitCode::from(2);
    };
    let mut files: Vec<_> = std::fs::read_dir(dir).map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map_or(false, |x| x == "s")).collect()).unwrap_or_default();
    files.sort();
    let show = args.iter().position(|a| a == "--show").and_then(|i| args.get(i + 1)).and_then(|v| v.parse::<usize>().ok()).unwrap_or(15);
    let jobs = args.windows(2).find(|w| w[0] == "--jobs").and_then(|w| w[1].parse::<usize>().ok()).unwrap_or(8).max(1);
    let next = AtomicUsize::new(0);
    let report: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());
    let same = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..jobs {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(f) = files.get(i) else { break };
                let name = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                let asm = std::fs::read_to_string(f).unwrap_or_default();
                let msg = match (amdgpu_target::gfx950_co::assemble(&asm), clang_co(&asm, &format!("v{i}"))) {
                    (Ok(a), Ok(b)) if a == b => {
                        same.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    (Ok(a), Ok(b)) => {
                        let at = a.iter().zip(&b).position(|(x, y)| x != y);
                        format!("differs: ours {} bytes, clang {} bytes, first difference at {:?}", a.len(), b.len(), at)
                    }
                    (Err(e), _) => format!("ours failed: {e}"),
                    (_, Err(e)) => format!("clang failed: {e}"),
                };
                report.lock().unwrap().push((name, msg));
            });
        }
    });
    let mut r = report.into_inner().unwrap();
    r.sort();
    println!("co-verify: {} of {} code objects are byte-identical to clang's", same.load(Ordering::Relaxed), files.len());
    for (n, m) in r.iter().take(show) {
        println!("  {n}: {m}");
    }
    if r.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
