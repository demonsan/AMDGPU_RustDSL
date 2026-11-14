//! The back-end-added instruction table: per reachable kernel, the final instructions the back end added on its own, by class
//! (`exact::synthetic_kind`: copies, constant materializations, waits, s_nops, readfirstlanes, implicit M0 writes). The metric of the
//! default-path lowering work: a change that removes added instructions shows here, kernel by kernel.
//!
//! Format: a header line, then `name<TAB>insts<TAB>copy<TAB>loopcopy<TAB>s2v<TAB>const<TAB>wait<TAB>nop<TAB>nop_states<TAB>rfl<TAB>m0<TAB>other`
//! per kernel (or `name<TAB>ERROR`), sorted by name; `#` lines are comments. Golden: tests/golden/added_table.tsv.

use amdgpu_codegen::mir::MOp;

pub const COLUMNS: &[&str] = &["insts", "copy", "loopcopy", "s2v", "const", "wait", "nop", "nop_states", "rfl", "m0", "other"];

/// The counts of one kernel (COLUMNS order).
pub fn count(f: &amdgpu_dsl::Function) -> Result<Vec<usize>, String> {
    let (m, _a) = amdgpu_codegen::compile_mir_alloc(f)?;
    let stmts = amdgpu_codegen::lower::last_stmts();
    let mut c = vec![0usize; COLUMNS.len()];
    for b in &m.blocks {
        for i in &b.insts {
            c[0] += 1;
            let Some(k) = amdgpu_codegen::exact::synthetic_kind(&m, &stmts, i) else { continue };
            let col = match k {
                "register copy" => 1,
                "loop-variable copy" => 2,
                "SGPR -> VGPR copy" => 3,
                "constant materialization" => 4,
                "wait (added after lowering)" | "automatic wait" => 5,
                "s_nop" | "s_nop (hazard pass)" => 6,
                "readfirstlane" => 8,
                "implicit M0 write (+ s_nop) of an LDS-DMA" => 9,
                _ => 10,
            };
            c[col] += 1;
            if col == 6 && i.op == MOp::SNop {
                c[7] += i.imm.max(0) as usize + 1;
            }
        }
    }
    Ok(c)
}

/// (name, row text) for every manifest kernel, compiled on `jobs` threads; refused when an `AMDC_*` variable is set.
pub fn compute(jobs: usize) -> Result<Vec<(String, String)>, String> {
    let ov = crate::manifest::env_overrides();
    if !ov.is_empty() {
        return Err(format!("the table needs a scrubbed environment; unset: {}", ov.join(" ")));
    }
    let names: Vec<String> = amdgpu_kernels::reachable::manifest_names().into_iter().collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::with_capacity(names.len()));
    std::thread::scope(|s| {
        for _ in 0..jobs.max(1) {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(k) = names.get(i) else { break };
                let v = match amdgpu_kernels::kernel_by_name(k) {
                    None => "MISSING".to_string(),
                    Some(f) => match count(&f) {
                        Ok(c) => c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\t"),
                        Err(_) => "ERROR".to_string(),
                    },
                };
                out.lock().unwrap().push((k.clone(), v));
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort();
    Ok(v)
}

pub fn render(entries: &[(String, String)]) -> String {
    let mut s = String::from("# back-end-added instructions by class (amdc added-table --out tests/golden/added_table.tsv; see tests/golden/README.md)\n");
    s.push_str(&format!("# name\t{}\n", COLUMNS.join("\t")));
    for (k, v) in entries {
        s.push_str(&format!("{k}\t{v}\n"));
    }
    s
}

pub fn parse(text: &str) -> Vec<(String, String)> {
    text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).filter_map(|l| l.split_once('\t').map(|(a, b)| (a.trim().to_string(), b.trim().to_string()))).collect()
}

/// The per-kernel differences, with the per-column deltas, and the column totals before -> after.
pub fn diff(golden: &[(String, String)], now: &[(String, String)]) -> (Vec<String>, String) {
    use std::collections::BTreeMap;
    let g: BTreeMap<&str, &str> = golden.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    let nums = |s: &str| -> Option<Vec<i64>> { s.split('\t').map(|x| x.parse().ok()).collect() };
    let mut d = Vec::new();
    let (mut tg, mut tn) = (vec![0i64; COLUMNS.len()], vec![0i64; COLUMNS.len()]);
    for (k, v) in now {
        let w = g.get(k.as_str()).copied();
        if let (Some(a), Some(b)) = (w.and_then(nums), nums(v)) {
            for (j, x) in a.iter().enumerate().take(COLUMNS.len()) {
                tg[j] += x;
            }
            for (j, x) in b.iter().enumerate().take(COLUMNS.len()) {
                tn[j] += x;
            }
            if a != b {
                let ch: Vec<String> = COLUMNS.iter().enumerate().filter(|(j, _)| a.get(*j) != b.get(*j)).map(|(j, c)| format!("{c} {:+}", b.get(j).unwrap_or(&0) - a.get(j).unwrap_or(&0))).collect();
                d.push(format!("changed  {k}: {}", ch.join(", ")));
            }
        } else if w != Some(v.as_str()) {
            d.push(format!("changed  {k}: {} -> {v}", w.unwrap_or("(absent)")));
        }
    }
    for (k, _) in golden {
        if !now.iter().any(|(n, _)| n == k) {
            d.push(format!("removed  {k}"));
        }
    }
    let tot = COLUMNS.iter().enumerate().map(|(j, c)| format!("{c} {} -> {}", tg[j], tn[j])).collect::<Vec<_>>().join(", ");
    (d, tot)
}
