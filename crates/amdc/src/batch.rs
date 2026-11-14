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

/// `amdc compile-batch [--jobs N] [--names a,b,c] [--no-verify] [--assemble] [--repeat R] [--time]`
pub fn compile_batch(args: &[String]) -> ExitCode {
    let opt = |name: &str| args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone());
    let jobs = opt("--jobs").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    let repeat = opt("--repeat").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1).max(1);
    let verify = !args.iter().any(|a| a == "--no-verify");
    let assemble = args.iter().any(|a| a == "--assemble");
    let show_phases = args.iter().any(|a| a == "--time");
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
    println!("slowest compiles (ms): {}", rows.iter().take(8).map(|r| format!("{} {:.1}", r.name, r.compile.as_secs_f64() * 1e3)).collect::<Vec<_>>().join(", "));
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
