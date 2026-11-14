//! Shape templates (docs/compile-time.md section 10): the specialized assembly of a kernel for one varying shape parameter, as the assembly of one value with its integer literals that vary as affine holes
//! `a * v + b`, valid on the values for which the instantiated text was verified **byte-identical to the full compile**.
//!
//! The verification is exhaustive over the declared domain at build time (`amdc template-build`): every value of the domain is compiled with the full pipeline (the allocator's and the wait / hazard verifier's
//! checks included), the values are grouped by assembly skeleton (every integer literal masked), the holes are fitted on two members of a group, and a member counts as valid only if instantiating the template
//! at that value reproduces the full compile's text exactly. A lookup succeeds only for a value in a verified set (under the same compiler, the same kernel, the same other bindings, the same backend): there is
//! no extrapolation, special values (powers of two, small values) simply stay outside the verified sets and take the full path.

use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};

/// The integer literals of a line: (byte start, byte end, value, hex width when written in hex).
fn literals(line: &str) -> Vec<(usize, usize, i128, Option<usize>)> {
    let b = line.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let neg = c == b'-' && i + 1 < b.len() && b[i + 1].is_ascii_digit();
        if c.is_ascii_digit() || neg {
            let prev = if i == 0 { b' ' } else { b[i - 1] };
            let start = i;
            let mut j = if neg { i + 1 } else { i };
            let hex = j + 1 < b.len() && b[j] == b'0' && (b[j + 1] == b'x' || b[j + 1] == b'X');
            let (digits_start, mut val): (usize, i128);
            if hex {
                j += 2;
                digits_start = j;
                while j < b.len() && b[j].is_ascii_hexdigit() {
                    j += 1;
                }
                val = i128::from_str_radix(&line[digits_start..j], 16).unwrap_or(0);
            } else {
                digits_start = j;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                val = line[digits_start..j].parse::<i128>().unwrap_or(0);
            }
            if neg {
                val = -val;
            }
            let next = if j < b.len() { b[j] } else { b' ' };
            let ident_before = prev.is_ascii_alphanumeric() || prev == b'_' || prev == b'[' || prev == b'.';
            let ident_after = next.is_ascii_alphanumeric() || next == b'_' || next == b':' || next == b']' || next == b'.';
            if !ident_before && !ident_after && j > digits_start {
                out.push((start, j, val, if hex { Some(j - digits_start) } else { None }));
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    out
}

fn mask(line: &str) -> String {
    let mut s = String::new();
    let mut at = 0;
    for (a, b, _, _) in literals(line) {
        s.push_str(&line[at..a]);
        s.push('#');
        at = b;
    }
    s.push_str(&line[at..]);
    s
}

#[derive(Clone, Debug)]
pub struct Hole {
    pub line: usize,
    pub ordinal: usize,
    pub a: i128,
    pub b: i128,
}

/// One verified group: the base text, its holes and the values it is valid for.
#[derive(Clone, Debug)]
pub struct Class {
    pub lines: Vec<String>,
    pub holes: Vec<Hole>,
    /// (lo, hi, step): the verified values are lo, lo + step, .. hi
    pub ranges: Vec<(i64, i64, i64)>,
}

#[derive(Clone, Debug)]
pub struct Template {
    pub kernel: String,
    pub param: String,
    /// the other bindings, sorted
    pub fixed: Vec<(String, String)>,
    pub backend: String,
    pub classes: Vec<Class>,
}

impl Class {
    pub fn contains(&self, v: i64) -> bool {
        self.ranges.iter().any(|&(lo, hi, st)| v >= lo && v <= hi && (v - lo) % st.max(1) == 0)
    }

    /// The assembly of this class at value `v` (the caller checks `contains`).
    pub fn instantiate(&self, v: i64) -> String {
        let mut by_line: HashMap<usize, Vec<&Hole>> = HashMap::new();
        for h in &self.holes {
            by_line.entry(h.line).or_default().push(h);
        }
        let mut out = String::with_capacity(self.lines.iter().map(|l| l.len() + 1).sum());
        for (li, line) in self.lines.iter().enumerate() {
            match by_line.get(&li) {
                None => out.push_str(line),
                Some(hs) => {
                    let lits = literals(line);
                    let mut at = 0;
                    for (k, (a, b, _, hexw)) in lits.iter().enumerate() {
                        out.push_str(&line[at..*a]);
                        match hs.iter().find(|h| h.ordinal == k) {
                            Some(h) => {
                                let val = h.a * v as i128 + h.b;
                                match hexw {
                                    Some(w) => {
                                        let (sign, mag) = if val < 0 { ("-", -val) } else { ("", val) };
                                        out.push_str(&format!("{sign}0x{mag:0w$x}", w = *w));
                                    }
                                    None => out.push_str(&val.to_string()),
                                }
                            }
                            None => out.push_str(&line[*a..*b]),
                        }
                        at = *b;
                    }
                    out.push_str(&line[at..]);
                }
            }
            out.push('\n');
        }
        out
    }
}

fn parse_val(s: &str) -> Option<i64> {
    let t = s.trim();
    match t.strip_prefix("0x") {
        Some(h) => i64::from_str_radix(h, 16).ok(),
        None => t.parse::<i64>().ok(),
    }
}

/// Builds the classes of one (kernel, fixed bindings, parameter) from the full-compile assembly of every value (`texts`, sorted by value). Every returned class is verified: each of its values reproduces
/// its full-compile text exactly when instantiated.
pub fn build_classes(texts: &[(i64, String)]) -> Vec<Class> {
    build_classes_report(texts).0
}

/// `build_classes` and the sizes of the skeleton groups (largest first).
pub fn build_classes_report(texts: &[(i64, String)]) -> (Vec<Class>, Vec<usize>, Vec<String>) {
    // group by skeleton (the text with every integer literal masked)
    let mut groups: Vec<(Vec<String>, Vec<usize>)> = Vec::new();
    let mut index: HashMap<Vec<String>, usize> = HashMap::new();
    for (i, (_, t)) in texts.iter().enumerate() {
        let sk: Vec<String> = t.lines().map(mask).collect();
        match index.get(&sk) {
            Some(&g) => groups[g].1.push(i),
            None => {
                index.insert(sk.clone(), groups.len());
                groups.push((sk, vec![i]));
            }
        }
    }
    let mut sizes: Vec<usize> = groups.iter().map(|g| g.1.len()).collect();
    sizes.sort_by(|a, b| b.cmp(a));
    let mut classes = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    for (_, group) in groups {
        // cover the group with affine regimes: fit on the first two members left, keep the members that reproduce their full compile exactly (a literal such as `m ^ 1024` is affine only on the
        // values with the same bit pattern), repeat with the rest
        let mut remaining = group;
        while remaining.len() >= 2 {
            let (i0, i1) = (remaining[0], remaining[1]);
            let (v0, v1) = (texts[i0].0 as i128, texts[i1].0 as i128);
            let l0: Vec<&str> = texts[i0].1.lines().collect();
            let l1: Vec<&str> = texts[i1].1.lines().collect();
            let mut holes = Vec::new();
            for (li, (a, b)) in l0.iter().zip(&l1).enumerate() {
                let (la, lb) = (literals(a), literals(b));
                for (k, (x, y)) in la.iter().zip(&lb).enumerate() {
                    if x.2 != y.2 {
                        let d = y.2 - x.2;
                        if d % (v1 - v0) != 0 {
                            continue; // not integral: the check below rejects every member but the base
                        }
                        let slope = d / (v1 - v0);
                        holes.push(Hole { line: li, ordinal: k, a: slope, b: x.2 - slope * v0 });
                    }
                }
            }
            let mut class = Class { lines: l0.iter().map(|s| s.to_string()).collect(), holes, ranges: Vec::new() };
            let mut valid: Vec<i64> = Vec::new();
            let mut rest: Vec<usize> = Vec::new();
            for &i in &remaining {
                let v = texts[i].0;
                let got = class.instantiate_unchecked(v);
                if got == texts[i].1 {
                    valid.push(v);
                } else {
                    if notes.len() < 3 {
                        let (x, y) = got.lines().zip(texts[i].1.lines()).enumerate().find(|(_, (x, y))| x != y).map(|(k, (x, y))| (format!("line {k}: {x}"), y.to_string())).unwrap_or_default();
                        notes.push(format!("value {v} is outside this regime: template {x} | compile {y}"));
                    }
                    rest.push(i);
                }
            }
            if valid.len() < 2 {
                break;
            }
            class.ranges = compress(&valid);
            classes.push(class);
            remaining = rest;
        }
    }
    (classes, sizes, notes)
}

impl Class {
    fn instantiate_unchecked(&self, v: i64) -> String {
        self.instantiate(v)
    }
}

/// Sorted values -> arithmetic-progression ranges with the smallest positive step of the first gap (runs of the same step).
fn compress(v: &[i64]) -> Vec<(i64, i64, i64)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        if i + 1 >= v.len() {
            out.push((v[i], v[i], 1));
            break;
        }
        let step = v[i + 1] - v[i];
        let mut j = i + 1;
        while j + 1 < v.len() && v[j + 1] - v[j] == step {
            j += 1;
        }
        out.push((v[i], v[j], step));
        i = j + 1;
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------------------ storage

pub fn key(kernel: &str, fixed: &[(String, String)], param: &str, backend: &str) -> String {
    let mut f = fixed.to_vec();
    f.sort();
    crate::md5::hex(format!("{kernel}|{}|{param}|{backend}", f.iter().map(|(a, b)| format!("{a}={b}")).collect::<Vec<_>>().join(",")).as_bytes())
}

pub fn write(dir: &str, t: &Template) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let mut s = String::from("amdc-template 1\n");
    s.push_str(&format!("kernel {}\nparam {}\nbackend {}\nfixed {}\nclasses {}\n", t.kernel, t.param, t.backend, t.fixed.iter().map(|(a, b)| format!("{a}={b}")).collect::<Vec<_>>().join(","), t.classes.len()));
    for c in &t.classes {
        s.push_str(&format!("class\nranges {}\nholes {}\n", c.ranges.iter().map(|(a, b, c)| format!("{a}:{b}:{c}")).collect::<Vec<_>>().join(" "), c.holes.len()));
        for h in &c.holes {
            s.push_str(&format!("hole {} {} {} {}\n", h.line, h.ordinal, h.a, h.b));
        }
        s.push_str(&format!("lines {}\n", c.lines.len()));
        for l in &c.lines {
            s.push_str(l);
            s.push('\n');
        }
    }
    let path = format!("{dir}/{}.tpl", key(&t.kernel, &t.fixed, &t.param, &t.backend));
    let tmp = format!("{path}.{}.tmp", std::process::id());
    fs::write(&tmp, s)?;
    fs::rename(tmp, path)
}

fn parse(text: &str) -> Option<Template> {
    let mut it = text.lines();
    if it.next()? != "amdc-template 1" {
        return None;
    }
    let kv = |l: Option<&str>, k: &str| l.and_then(|x| x.strip_prefix(k)).map(|x| x.trim().to_string());
    let kernel = kv(it.next(), "kernel ")?;
    let param = kv(it.next(), "param ")?;
    let backend = kv(it.next(), "backend ")?;
    let fixed_s = it.next()?.strip_prefix("fixed")?.trim().to_string();
    let fixed: Vec<(String, String)> = fixed_s.split(',').filter(|x| !x.is_empty()).filter_map(|x| x.split_once('=').map(|(a, b)| (a.to_string(), b.to_string()))).collect();
    let n: usize = kv(it.next(), "classes ")?.parse().ok()?;
    let mut classes = Vec::new();
    for _ in 0..n {
        if it.next()? != "class" {
            return None;
        }
        let ranges: Vec<(i64, i64, i64)> = it.next()?.strip_prefix("ranges")?.split_whitespace().filter_map(|r| { let p: Vec<&str> = r.split(':').collect(); Some((p.first()?.parse().ok()?, p.get(1)?.parse().ok()?, p.get(2)?.parse().ok()?)) }).collect();
        let nh: usize = kv(it.next(), "holes ")?.parse().ok()?;
        let mut holes = Vec::new();
        for _ in 0..nh {
            let p: Vec<&str> = it.next()?.strip_prefix("hole ")?.split_whitespace().collect();
            holes.push(Hole { line: p.first()?.parse().ok()?, ordinal: p.get(1)?.parse().ok()?, a: p.get(2)?.parse().ok()?, b: p.get(3)?.parse().ok()? });
        }
        let nl: usize = kv(it.next(), "lines ")?.parse().ok()?;
        let mut lines = Vec::with_capacity(nl);
        for _ in 0..nl {
            lines.push(it.next()?.to_string());
        }
        classes.push(Class { lines, holes, ranges });
    }
    Some(Template { kernel, param, fixed, backend, classes })
}

fn cache() -> &'static Mutex<HashMap<String, Option<Arc<Template>>>> {
    static C: std::sync::OnceLock<Mutex<HashMap<String, Option<Arc<Template>>>>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn load_file(path: &str) -> Option<Arc<Template>> {
    if let Some(t) = cache().lock().unwrap().get(path) {
        return t.clone();
    }
    let t = fs::read_to_string(path).ok().and_then(|s| parse(&s)).map(Arc::new);
    cache().lock().unwrap().insert(path.to_string(), t.clone());
    t
}

/// The verified assembly of `kernel` at `spec` (`name=value` bindings) from the templates of `dir`, if some template covers it exactly: same kernel, same other bindings, same backend, the value in a verified set.
pub fn lookup(dir: &str, kernel: &str, spec: &[(String, String)], backend: &str) -> Option<String> {
    for (p, pv) in spec {
        let Some(v) = parse_val(pv) else { continue };
        let fixed: Vec<(String, String)> = spec.iter().filter(|(n, _)| n != p).cloned().collect();
        let path = format!("{dir}/{}.tpl", key(kernel, &fixed, p, backend));
        let Some(t) = load_file(&path) else { continue };
        if t.kernel != kernel || t.param != *p || t.backend != backend {
            continue;
        }
        let mut want = fixed.clone();
        want.sort();
        if t.fixed != want {
            continue;
        }
        if let Some(c) = t.classes.iter().find(|c| c.contains(v)) {
            return Some(c.instantiate(v));
        }
    }
    None
}

/// Builds the template of (`kernel`, `fixed` bindings, `param` over lo..=hi step `step`): full compile of every value of the domain on `jobs` threads, then `build_classes`. Returns the template and
/// (domain size, values that compiled).
pub fn build(kernel: &str, fixed: &[(String, String)], param: &str, lo: i64, hi: i64, step: i64, jobs: usize) -> (Template, usize, usize, Vec<usize>, Vec<String>) {
    use amdgpu_codegen::backend::OwnTail;
    use amdgpu_codegen::{compile_with_options, CompileOptions};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let values: Vec<i64> = (0..).map(|i| lo + i * step.max(1)).take_while(|v| *v <= hi).collect();
    let next = AtomicUsize::new(0);
    let texts: Mutex<Vec<(i64, String)>> = Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..jobs.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(&v) = values.get(i) else { break };
                let mut spec = fixed.to_vec();
                spec.push((param.to_string(), v.to_string()));
                let Ok((f, _)) = crate::jitc::load(kernel, &spec, None) else { continue };
                if let Ok(c) = compile_with_options(&f, &OwnTail, &CompileOptions { verify: true }) {
                    texts.lock().unwrap().push((v, c.asm));
                }
            });
        }
    });
    let mut texts = texts.into_inner().unwrap();
    texts.sort_by_key(|t| t.0);
    let mut fx = fixed.to_vec();
    fx.sort();
    let compiled = texts.len();
    let (classes, sizes, notes) = build_classes_report(&texts);
    (Template { kernel: kernel.to_string(), param: param.to_string(), fixed: fx, backend: "own".into(), classes }, values.len(), compiled, sizes, notes)
}

/// `amdc template-build FILE --template-dir DIR [--jobs N]`: lines `kernel<TAB>name=v,..<TAB>param<TAB>lo:hi:step`.
pub fn build_file(file: &str, dir: &str, jobs: usize) -> Vec<String> {
    let text = fs::read_to_string(file).unwrap_or_default();
    let mut report = Vec::new();
    for l in text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
        let f: Vec<&str> = l.split('\t').collect();
        if f.len() < 4 {
            report.push(format!("bad line: {l}"));
            continue;
        }
        let fixed = if f[1].trim().is_empty() { Ok(Vec::new()) } else { amdgpu_dsl::specialize::parse_bindings(f[1]) };
        let dom: Vec<i64> = f[3].split(':').filter_map(|x| x.trim().parse().ok()).collect();
        let (Ok(fixed), true) = (fixed, dom.len() == 3) else {
            report.push(format!("bad line: {l}"));
            continue;
        };
        let (t, n, ok, sizes, notes) = build(f[0], &fixed, f[2], dom[0], dom[1], dom[2], jobs);
        let valid: usize = t.classes.iter().map(|c| c.ranges.iter().map(|&(lo, hi, st)| ((hi - lo) / st.max(1) + 1) as usize).sum::<usize>()).sum();
        if !t.classes.is_empty() {
            let _ = write(dir, &t);
        }
        report.push(format!("{} {} over {}..{} step {}: {n} values, {ok} compiled, {} class(es), {valid} verified values ({:.0} %); skeleton groups of the domain (sizes): {:?}", f[0], f[2], dom[0], dom[1], dom[2], t.classes.len(), 100.0 * valid as f64 / n.max(1) as f64, &sizes[..sizes.len().min(8)]));
        for n in notes {
            report.push(format!("    note: {n}"));
        }
    }
    report
}
