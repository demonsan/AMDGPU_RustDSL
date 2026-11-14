//! The JIT compile request, as a library (docs/compile-time.md): kernel (+ specialization) -> compile once -> the specialization certificate -> assembly, code object and `jit-info` JSON, with every message
//! returned as text (the CLI prints it, `amdc serve` sends it back). One implementation behind `amdc emit --co --info`, `amdc serve` and `amdc certify-batch`.
//!
//! The certificate of a specialized (or `--from-ir`) kernel is the IR-level analysis (`gpu_verify::verify_ir`, which depends on the IR only) plus the MIR-level one (`verify_mir`, on the final MIR of the very
//! compile that produces the assembly: no second compile). The IR-level part is cached **content-addressed**: the key is the md5 of the exact specialized IR (`format!("{:?}", Function)`, every field), the cache
//! directory carries the compiler's hash; a cached verdict is reused only for a byte-identical IR under the same compiler, and only passing verdicts are stored.

use crate::md5;
use amdgpu_codegen::backend::{self, TailBackend};
use amdgpu_codegen::gpu_verify::{verify_ir, verify_mir, Report};
use amdgpu_codegen::{compile_with_mir, CompileOptions};
use amdgpu_dsl::specialize::SpecValue;
use amdgpu_dsl::Function;
use std::fs;
use std::path::Path;

pub type Spec = (Vec<(String, u32, SpecValue)>, Vec<(u32, i64)>);

#[derive(Default, Clone, Debug)]
pub struct Req {
    pub kernel: String,
    /// `name=value` bindings
    pub spec: Vec<(String, String)>,
    pub backend: String,
    pub out_s: Option<String>,
    pub co: Option<String>,
    pub info: Option<String>,
    pub from_ir: Option<String>,
    pub cert_cache: Option<String>,
    /// the shape-template directory (`amdc template-build`); a verified template replaces the compile
    pub template_dir: Option<String>,
    pub no_verify: bool,
}

#[derive(Debug)]
pub struct Resp {
    /// 0 ok; 1 error; 2 bad request; 3 the in-process assembler cannot encode the kernel (assembly and info are written: use the external assembler)
    pub code: u8,
    pub msg: String,
}

/// The kernel of a request, specialized by its bindings.
pub fn load(kernel: &str, spec: &[(String, String)], from_ir: Option<&str>) -> Result<(Function, Spec), String> {
    let f = match from_ir {
        Some(path) => {
            let text = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
            amdgpu_dsl::ir_text::parse(&text).map_err(|e| format!("{path}: {e}"))?
        }
        None => amdgpu_kernels::kernel_by_name(kernel).ok_or_else(|| format!("unknown kernel: {kernel}"))?,
    };
    if spec.is_empty() {
        return Ok((f, (Vec::new(), Vec::new())));
    }
    let vals = amdgpu_dsl::specialize::check(&f, spec)?;
    let derived = amdgpu_dsl::specialize::close(&f, &vals)?.derived;
    let g = amdgpu_dsl::specialize::specialize(&f, spec)?;
    Ok((g, (vals, derived)))
}

/// The content address of a kernel's IR: md5 of every field of the (specialized) `Function`.
pub fn cert_key(f: &Function) -> String {
    md5::hex(format!("{f:?}").as_bytes())
}

type Ledger5 = [usize; 5];

fn ledger_of(r: &Report) -> Ledger5 {
    let l = r.ledger();
    [l.assume_trusted, l.tile_trust, l.region_trust, l.asm_opaque, l.timing_trust]
}

fn cache_read(dir: &str, key: &str) -> Option<Ledger5> {
    let t = fs::read_to_string(Path::new(dir).join(format!("{key}.cert"))).ok()?;
    let mut it = t.split_whitespace();
    if it.next()? != "ok" {
        return None;
    }
    let mut l = [0usize; 5];
    for x in l.iter_mut() {
        *x = it.next()?.parse().ok()?;
    }
    Some(l)
}

fn cache_write(dir: &str, key: &str, l: Ledger5, kernel: &str) {
    let _ = fs::create_dir_all(dir);
    let tmp = Path::new(dir).join(format!(".{key}.{}.tmp", std::process::id()));
    if fs::write(&tmp, format!("ok {} {} {} {} {}\n{kernel}\n", l[0], l[1], l[2], l[3], l[4])).is_ok() {
        let _ = fs::rename(&tmp, Path::new(dir).join(format!("{key}.cert")));
    }
}

/// The IR-level certificate of `f` for the offline pre-population (`amdc certify-batch`): Ok(true) = computed and stored, Ok(false) = already cached, Err = does not verify.
pub fn certify_ir_cached(f: &Function, dir: &str) -> Result<bool, String> {
    let key = cert_key(f);
    if cache_read(dir, &key).is_some() {
        return Ok(false);
    }
    let r = verify_ir(f);
    let errs = r.errors();
    if !errs.is_empty() {
        return Err(errs.iter().map(|e| format!("  [{}] {}", e.code, e.msg)).collect::<Vec<_>>().join("\n"));
    }
    cache_write(dir, &key, ledger_of(&r), &f.name);
    Ok(true)
}

fn bad(msg: String) -> Resp {
    Resp { code: 2, msg }
}

/// Compiles one request. Messages (the certificate line, notes, errors) are in `Resp::msg`.
pub fn compile_request(r: &Req) -> Resp {
    let (f, spec) = match load(&r.kernel, &r.spec, r.from_ir.as_deref()) {
        Ok(x) => x,
        Err(e) => return bad(e),
    };
    let certify = (r.from_ir.is_some() || !spec.0.is_empty()) && !r.no_verify;
    let (resolved, _) = amdgpu_codegen::resolve_backend(if r.backend.is_empty() { "auto" } else { &r.backend }, &f);
    let Some(tail) = backend::backend_by_name(&resolved) else { return bad(format!("unknown backend: {resolved} (own | llvm-tail | auto)")) };
    let tail: &dyn TailBackend = tail.as_ref();
    // the IR-level certificate: from the cache, or on its own thread while the kernel compiles
    let key = certify.then(|| cert_key(&f));
    let cached = match (&key, &r.cert_cache) {
        (Some(k), Some(d)) => cache_read(d, k),
        _ => None,
    };
    let ir_thread = if certify && cached.is_none() {
        let fc = f.clone();
        Some(std::thread::spawn(move || verify_ir(&fc)))
    } else {
        None
    };
    // a verified shape template (exhaustively checked byte-identical to the full compile at build time) replaces the compile; its register / wait checks ran with that compile
    let tpl_asm = match (&r.template_dir, resolved.as_str(), r.from_ir.is_none()) {
        (Some(d), "own", true) => crate::template::lookup(d, &r.kernel, &r.spec, "own"),
        _ => None,
    };
    let from_template = tpl_asm.is_some();
    let (c, mir_parts) = match tpl_asm {
        Some(asm) => (amdgpu_codegen::Compiled { asm, vgpr: 0, sgpr: 0 }, None),
        None => match compile_with_mir(&f, tail, &CompileOptions { verify: !certify }) {
            Ok((c, m, alloc)) => (c, Some((m, alloc))),
            Err(e) => return Resp { code: 1, msg: format!("compile error:\n{e}\n") },
        },
    };
    let mut msg = String::new();
    if certify {
        let mir = match &mir_parts {
            Some((m, alloc)) => verify_mir(&f, m, alloc),
            None => Report::default(),
        };
        let (ir_ledger, mut errs): (Ledger5, Vec<String>) = match (cached, ir_thread) {
            (Some(l), _) => (l, Vec::new()),
            (None, Some(h)) => match h.join() {
                Ok(rep) => (ledger_of(&rep), rep.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)).collect()),
                Err(_) => return Resp { code: 1, msg: "the certificate thread panicked\n".into() },
            },
            (None, None) => ([0; 5], Vec::new()),
        };
        let ir_ok = errs.is_empty();
        errs.extend(mir.errors().iter().map(|e| format!("[{}] {}", e.code, e.msg)));
        let ml = ledger_of(&mir);
        let total: usize = ir_ledger.iter().sum::<usize>() + ml.iter().sum::<usize>();
        let cn = &f.contract;
        let contracts = cn.extents.len() + cn.ka_ranges.len() + cn.grid_defs.len() + cn.ka_defs.len() + cn.ka_sets.len() + cn.ka_pow2.len() + cn.sym_ranges.len();
        msg.push_str(&format!("verified {}: certificate: {total} trusted assumes, {contracts} trusted contracts, {} unproven{}{}\n", f.name, errs.len(), if cached.is_some() { " (IR certificate from the cache)" } else { "" }, if from_template { " (assembly from a verified shape template)" } else { "" }));
        if !errs.is_empty() {
            for e in &errs {
                msg.push_str(&format!("  {e}\n"));
            }
            return Resp { code: 1, msg };
        }
        if let (Some(k), Some(d), true, None) = (&key, &r.cert_cache, ir_ok, cached) {
            cache_write(d, k, ir_ledger, &f.name);
        }
    }
    if let Some(p) = &r.out_s {
        if let Err(e) = fs::write(p, &c.asm) {
            return Resp { code: 1, msg: format!("cannot write {p}: {e}\n") };
        }
    }
    if let Some(p) = &r.info {
        let source = match r.from_ir.as_deref().and_then(|p| fs::read_to_string(p).ok()) {
            Some(t) => format!("ir:{}", crate::jit::fnv_hex(&t)),
            None => "registry".to_string(),
        };
        let j = crate::jit::info_json_src(&f, &spec.0, &spec.1, tail.name(), crate::jit::resources(&c.asm), &source);
        if let Err(e) = fs::write(p, j) {
            return Resp { code: 1, msg: format!("cannot write {p}: {e}\n") };
        }
    }
    if let Some(p) = &r.co {
        match amdgpu_target::gfx950_co::assemble(&c.asm) {
            Ok(b) => {
                if let Err(e) = fs::write(p, b) {
                    return Resp { code: 1, msg: format!("cannot write {p}: {e}\n") };
                }
            }
            Err(e) => {
                msg.push_str(&format!("note: in-process assembler cannot encode this kernel ({e}); use the external assembler\n"));
                return Resp { code: 3, msg };
            }
        }
    }
    Resp { code: 0, msg }
}

/// The certificate-cache directory the Python JIT uses for this compiler: `<cache>/certs/<sha256(amdc)[:24]>` (`AMDGPU_JIT_CACHE_DIR`, else `~/.cache/amdgpu_jit`).
pub fn default_cert_dir() -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let bytes = fs::read(&exe).map_err(|e| format!("{}: {e}", exe.display()))?;
    let root = match std::env::var("AMDGPU_JIT_CACHE_DIR").ok().filter(|v| !v.is_empty()) {
        Some(d) => d,
        None => format!("{}/.cache/amdgpu_jit", std::env::var("HOME").map_err(|_| "HOME is not set".to_string())?),
    };
    Ok(format!("{root}/certs/{}", &crate::sha256::hex(&bytes)[..24]))
}

/// `amdc certify-batch FILE [--cert-cache DIR] [--jobs N]`: the IR-level certificate of every `kernel<TAB>name=v,name=v` line of FILE (`#` comments), stored content-addressed so that the JIT's cold compile of those shapes skips it.
/// Returns (shapes, stored, already cached, failures).
pub fn certify_batch(file: &str, dir: &str, jobs: usize) -> (usize, usize, usize, Vec<String>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let text = fs::read_to_string(file).unwrap_or_default();
    let lines: Vec<(String, String)> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let (k, s) = l.split_once('\t').unwrap_or((l, ""));
            (k.trim().to_string(), s.trim().to_string())
        })
        .collect();
    let next = AtomicUsize::new(0);
    let (stored, cached) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let failed = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..jobs.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((k, s)) = lines.get(i) else { break };
                let spec = if s.is_empty() { Ok(Vec::new()) } else { amdgpu_dsl::specialize::parse_bindings(s) };
                let r = spec.and_then(|sp| load(k, &sp, None)).and_then(|(f, _)| certify_ir_cached(&f, dir));
                match r {
                    Ok(true) => {
                        stored.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(false) => {
                        cached.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => failed.lock().unwrap().push(format!("{k} {s}: {e}")),
                }
            });
        }
    });
    (lines.len(), stored.into_inner(), cached.into_inner(), failed.into_inner().unwrap())
}

// ------------------------------------------------------------------------------------------------------------------------------ the server

fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\n', "\\n").replace('\t', "\\t")
}

fn opt(s: &str) -> Option<String> {
    if s == "-" || s.is_empty() {
        None
    } else {
        Some(unesc(s))
    }
}

/// One request line: `id TAB kernel TAB name=v,name=v|- TAB backend TAB out_s|- TAB co|- TAB info|- TAB from_ir|- TAB cert_cache|- TAB flags` (flags: `no-verify`).
pub fn parse_line(line: &str) -> Result<(String, Req), String> {
    let f: Vec<&str> = line.split('\t').collect();
    if f.len() < 10 {
        return Err(format!("request needs 10 (or 11) tab-separated fields, got {}", f.len()));
    }
    let spec = match opt(f[2]) {
        Some(s) => amdgpu_dsl::specialize::parse_bindings(&s)?,
        None => Vec::new(),
    };
    Ok((f[0].to_string(), Req { kernel: unesc(f[1]), spec, backend: unesc(f[3]), out_s: opt(f[4]), co: opt(f[5]), info: opt(f[6]), from_ir: opt(f[7]), cert_cache: opt(f[8]), template_dir: f.get(10).and_then(|x| opt(x)), no_verify: f[9].contains("no-verify") }))
}

pub fn response_line(id: &str, r: &Resp) -> String {
    format!("{id}\t{}\t{}\n", r.code, esc(&r.msg))
}

/// `amdc serve [--jobs N]`: a persistent compile server on stdin / stdout (one request per line, one response per line, any order; EOF ends it). Std only; each request is compiled on one of N worker threads.
pub fn serve(jobs: usize) -> std::io::Result<()> {
    use std::io::{BufRead, Write};
    use std::sync::{mpsc, Arc, Mutex};
    let (tx, rx) = mpsc::channel::<String>();
    let rx = Arc::new(Mutex::new(rx));
    let out = Arc::new(Mutex::new(std::io::stdout()));
    let mut workers = Vec::new();
    for _ in 0..jobs.max(1) {
        let (rx, out) = (rx.clone(), out.clone());
        workers.push(std::thread::spawn(move || loop {
            let line = { rx.lock().unwrap().recv() };
            let Ok(line) = line else { break };
            let resp = match parse_line(&line) {
                Ok((id, req)) => {
                    let r = std::panic::catch_unwind(|| compile_request(&req)).unwrap_or_else(|_| Resp { code: 1, msg: "the compiler panicked\n".into() });
                    response_line(&id, &r)
                }
                Err(e) => {
                    let id = line.split('\t').next().unwrap_or("?").to_string();
                    response_line(&id, &bad(e))
                }
            };
            let mut o = out.lock().unwrap();
            let _ = o.write_all(resp.as_bytes());
            let _ = o.flush();
        }));
    }
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let _ = tx.send(line);
        }
    }
    drop(tx);
    for w in workers {
        let _ = w.join();
    }
    Ok(())
}
