//! Structured, localized diagnostics for agents (docs/diagnostics.md): every finding / compile failure as a record with a stable code, the
//! kernel, the statement, its weave region, the Rust source location of the builder call that created the IR node (`Function::src_locs`),
//! the values involved (with their source locations) and a suggested fix where one is known; and a compact kernel summary. JSON is written
//! by hand (no dependencies).
use amdgpu_dsl::{Function, ValueId};

/// One diagnostic.
#[derive(Clone, Debug, Default)]
pub struct Diag {
    pub code: String,
    pub severity: String,
    pub message: String,
    pub kernel: String,
    /// statement index into `Function::stmts()` (the IR node path)
    pub stmt: Option<usize>,
    pub region: Option<String>,
    /// `file:line:col` of the builder call; `src_exact` = false when it is the nearest preceding value's
    pub src: Option<String>,
    pub src_exact: bool,
    /// values named by the diagnostic (`vN` / registers mapped back to values) with their source locations
    pub values: Vec<(String, Option<String>)>,
    pub fix: Option<String>,
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn opt(s: &Option<String>) -> String {
    s.as_ref().map(|x| esc(x)).unwrap_or_else(|| "null".to_string())
}

impl Diag {
    pub fn to_json(&self) -> String {
        let vals: Vec<String> = self.values.iter().map(|(n, l)| format!("{{\"value\":{},\"src\":{}}}", esc(n), opt(l))).collect();
        format!(
            "{{\"code\":{},\"severity\":{},\"message\":{},\"kernel\":{},\"stmt\":{},\"region\":{},\"src\":{},\"src_exact\":{},\"values\":[{}],\"fix\":{}}}",
            esc(&self.code),
            esc(&self.severity),
            esc(&self.message),
            esc(&self.kernel),
            self.stmt.map(|s| s.to_string()).unwrap_or_else(|| "null".into()),
            opt(&self.region),
            opt(&self.src),
            self.src_exact,
            vals.join(","),
            opt(&self.fix)
        )
    }
}

/// A JSON array of diagnostics.
pub fn to_json(ds: &[Diag]) -> String {
    format!("[{}]", ds.iter().map(|d| d.to_json()).collect::<Vec<_>>().join(","))
}

/// The first `stmt N` in a message.
fn stmt_of(msg: &str) -> Option<usize> {
    let i = msg.find("stmt ")?;
    let rest = &msg[i + 5..];
    let n: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    n.parse().ok()
}

/// Every `vN` / `Val(N)` value reference in a message.
fn values_of(msg: &str) -> Vec<ValueId> {
    let mut out = Vec::new();
    let b = msg.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let (skip, ok) = if msg[i..].starts_with("Val(") {
            (4, true)
        } else if b[i] == b'v' && (i == 0 || !(b[i - 1] as char).is_ascii_alphanumeric()) && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            (1, true)
        } else {
            (1, false)
        };
        if ok {
            let n: String = msg[i + skip..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = n.parse::<ValueId>() {
                if !out.contains(&v) {
                    out.push(v);
                }
            }
        }
        i += skip.max(1);
    }
    out
}

/// Attach the localization (statement, region, source, values) of `d.message` against `f`.
pub fn localize(f: &Function, mut d: Diag) -> Diag {
    d.kernel = f.name.clone();
    if d.stmt.is_none() {
        d.stmt = stmt_of(&d.message);
    }
    if let Some(s) = d.stmt {
        d.region = f.region_of(s);
        if let Some((l, exact)) = f.stmt_loc(s) {
            d.src = Some(l);
            d.src_exact = exact;
        }
    }
    // the statement's operand values (where the inputs of the failing node were built)
    if let Some(s) = d.stmt {
        if let Some(st) = f.stmts().get(s) {
            for v in amdgpu_dsl::facts::stmt_value_uses(st).into_iter().take(6) {
                d.values.push((format!("v{v}"), f.value_loc(v)));
            }
        }
    }
    for v in values_of(&d.message) {
        if d.values.iter().any(|(n, _)| *n == format!("v{v}")) {
            continue;
        }
        if (v as usize) < f.value_tys.len() {
            d.values.push((format!("v{v}"), f.value_loc(v)));
        }
    }
    d
}

/// The diagnostics of a GPU-semantics report.
pub fn from_report(f: &Function, r: &crate::gpu_verify::Report) -> Vec<Diag> {
    r.findings
        .iter()
        .map(|fd| {
            let fix = fix_for(fd.code, &fd.msg);
            localize(f, Diag { code: fd.code.to_string(), severity: format!("{:?}", fd.severity).to_lowercase(), message: fd.msg.clone(), fix, ..Default::default() })
        })
        .collect()
}

fn fix_for(code: &str, msg: &str) -> Option<String> {
    let s = match code {
        "GV-BUF-WRAP" => "use an out-of-range sentinel that stays >= num_records after the instruction offset (e.g. 0x7fffff00), or fold the immediate into the register offset",
        "GV-BUF-OVER" => "size the descriptor to the declared extent (num_records <= allocation - base); for > 4 GiB tensors move the base with ptr_mad",
        "GV-GLB-PROV" => "build the descriptor from a kernarg pointer (or a ptr_mad of one with provably non-negative factors)",
        "GV-LDS-OOB" => "keep LDS addresses inside the declared lds() size (clamp or mask the index)",
        "GV-GLB-OOB" => "bound the index (min / clamp) or declare the extent the access really needs",
        "GV-REG-OCCUPANCY" => "use fewer threads per workgroup or fewer live registers",
        _ => return if msg.contains("not provably wave-uniform") { Some(UNIFORM_FIX.to_string()) } else { None },
    };
    Some(s.to_string())
}

const UNIFORM_FIX: &str = "branch on wave-uniform values (kernel arguments, workgroup ids, readfirstlane / readlane / ballot) and predicate per-lane work; wrap the condition in any_lane(..) if the wave-level 'any lane' reading is intended";

/// The diagnostics of a compile failure (the error string of `compile*` / `lower`): IR-verifier errors one per line, the register budget,
/// running out of registers (the live values at the peak mapped back to their builder calls), and anything else as COMPILE-ERROR.
pub fn from_compile_error(f: &Function, e: &str) -> Vec<Diag> {
    let mut out = Vec::new();
    if let Some(rest) = e.strip_prefix("IR verification failed:") {
        let body = rest.split("\nIR:").next().unwrap_or(rest);
        for line in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let code = if line.contains("not provably wave-uniform") {
                "IR-UNIFORM-BRANCH"
            } else if line.contains("not a power of two") {
                "IR-CVT-SCALE"
            } else {
                "IR-VERIFY"
            };
            let fix = if code == "IR-UNIFORM-BRANCH" {
                Some(UNIFORM_FIX.to_string())
            } else if code == "IR-CVT-SCALE" {
                Some("pass a power-of-two scale (the hardware uses only the exponent, E8M0)".to_string())
            } else {
                None
            };
            out.push(localize(f, Diag { code: code.into(), severity: "error".into(), message: line.to_string(), fix, ..Default::default() }));
        }
        return out;
    }
    let (code, fix) = if e.contains("register budget") || e.contains("for the launch") || e.contains("launch budget") {
        ("COMPILE-REG-BUDGET", Some("use fewer threads per workgroup or fewer live registers (see the live values listed)".to_string()))
    } else if e.contains("out of VGPRs") || e.contains("out of AGPRs") || e.contains("out of SGPRs") {
        ("RA-OUT-OF-REGISTERS", Some("shorten the live ranges of the listed values (reload / recompute instead of keeping them), or split the kernel".to_string()))
    } else {
        ("COMPILE-ERROR", None)
    };
    let mut d = localize(f, Diag { code: code.into(), severity: "error".into(), message: e.lines().take(8).collect::<Vec<_>>().join("\n"), fix, ..Default::default() });
    // registers named in the message (`rN`) -> the values they were lowered from -> their builder calls
    let origins = crate::lower::last_vreg_origins();
    let b = e.as_bytes();
    let mut seen = Vec::new();
    for i in 0..b.len() {
        if b[i] == b'r' && (i == 0 || !(b[i - 1] as char).is_ascii_alphanumeric()) {
            let n: String = e[i + 1..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(r) = n.parse::<u32>() {
                if let Some((_, v)) = origins.iter().find(|(x, _)| *x == r) {
                    if !seen.contains(v) && seen.len() < 64 {
                        seen.push(*v);
                        d.values.push((format!("r{r}=v{v}"), f.value_loc(*v)));
                    }
                }
            }
        }
    }
    out.push(d);
    out
}

/// `amdc summary --json`: resources, per-region instruction mix, waits / nops, and the certificate counts of a kernel.
pub fn summary_json(f: &Function) -> Result<String, String> {
    let (m, a) = crate::compile_mir_alloc(f)?;
    let agpr = crate::regalloc::agpr_count(&m, &a);
    let mut per_region: std::collections::BTreeMap<String, [usize; 7]> = Default::default();
    let name_of = |id: u16| -> String { m.reg_hints.regions.iter().find(|(_, i)| *i == id).map(|(n, _)| n.clone()).unwrap_or_else(|| if id == 0 { "(kernel)".into() } else { format!("region{id}") }) };
    let (mut waits, mut nops, mut nop_states) = (0usize, 0usize, 0usize);
    for b in &m.blocks {
        for i in &b.insts {
            let at = crate::op_attrs::attrs(i.op);
            let k = if at.mfma.is_some() {
                0
            } else if at.mem.map_or(false, |x| matches!(x.space, crate::sched::effects::Space::Global)) {
                1
            } else if at.mem.is_some() {
                2
            } else if at.smem {
                3
            } else if matches!(i.op, crate::mir::MOp::SWaitVm | crate::mir::MOp::SWaitLgkm | crate::mir::MOp::SNop | crate::mir::MOp::SBarrier) {
                4
            } else if format!("{:?}", i.op).starts_with('S') {
                5
            } else {
                6
            };
            per_region.entry(name_of(i.region)).or_insert([0; 7])[k] += 1;
            match i.op {
                crate::mir::MOp::SWaitVm | crate::mir::MOp::SWaitLgkm => waits += 1,
                crate::mir::MOp::SNop => {
                    nops += 1;
                    nop_states += i.imm as usize + 1;
                }
                _ => {}
            }
        }
    }
    let regions: Vec<String> = per_region
        .iter()
        .map(|(n, c)| format!("{{\"name\":{},\"mfma\":{},\"vmem\":{},\"lds\":{},\"smem\":{},\"sync\":{},\"salu\":{},\"valu\":{}}}", esc(n), c[0], c[1], c[2], c[3], c[4], c[5], c[6]))
        .collect();
    let cert = match crate::gpu_verify::analyse(f) {
        Ok(r) => {
            let l = r.ledger();
            let c = &f.contract;
            let contracts = c.extents.len() + c.ka_ranges.len() + c.grid_defs.len() + c.ka_defs.len() + c.ka_sets.len() + c.sym_ranges.len();
            format!(
                "{{\"errors\":{},\"warnings\":{},\"trusted_assumes\":{},\"trusted_contracts\":{}}}",
                r.errors().len(),
                r.findings.iter().filter(|x| x.severity == crate::gpu_verify::Severity::Warn).count(),
                l.total(),
                contracts
            )
        }
        Err(e) => format!("{{\"error\":{}}}", esc(&e.lines().next().unwrap_or(""))),
    };
    let insts: usize = m.blocks.iter().map(|b| b.insts.len()).sum();
    Ok(format!(
        "{{\"kernel\":{},\"workgroup_size\":{},\"vgpr\":{},\"agpr\":{},\"sgpr\":{},\"lds_bytes\":{},\"blocks\":{},\"instructions\":{},\"waits\":{},\"nops\":{},\"nop_wait_states\":{},\"regions\":[{}],\"certificate\":{}}}",
        esc(&f.name),
        f.workgroup_size,
        a.max_v,
        agpr,
        a.max_s,
        m.lds_bytes,
        m.blocks.len(),
        insts,
        waits,
        nops,
        nop_states,
        regions.join(","),
        cert
    ))
}
