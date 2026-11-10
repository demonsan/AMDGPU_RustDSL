//! JIT metadata (`amdc jit-info`, docs/jit.md): one JSON object per (kernel, specialization) for the Python JIT front end, which calls the
//! `amdc` CLI as a subprocess (no FFI). Schema 1:
//!
//! ```text
//! {"schema":1, "kernel", "symbol", "arch":"gfx950", "compiler":{"version","git"}, "compiler_hash", "backend", "specialize":{name:value}, "derived":{name:value},
//!  "block":[x,1,1], "lds_bytes", "vgpr", "agpr", "sgpr", "scratch_bytes", "kernarg_size", "kernarg_align",
//!  "args":[{"name","offset","size","kind":"ptr|i32|u32|f32|pad","specializable","specialized":null|value}],
//!  "contract":{"grid":[{"axis","ext":E}], "ka_is":[{"offset","ext":E}], "ka_range":[{"offset","lo","hi"|null}], "ka_set":[{"offset","values"}],
//!              "tensors":[{"arg","alloc_bytes":E,"logical_bytes":E|null}], "syms":[{"name","lo","hi"|null}]}}
//! E := {"c":int} | {"ka":offset} | {"grid":axis} | {"sym":name} | {"add":[E,E]} | {"sub":[E,E]} | {"mul":[E,E]} | {"ceil_div":[E,k]} | {"floor_div":[E,k]}
//! ```

use amdgpu_dsl::specialize::{arg_kind, specializable, SpecValue};
use amdgpu_dsl::{Ext, Function};
use std::collections::BTreeSet;

/// Compiler identity: Cargo version + `git describe` at build time (build.rs; "unknown" without git).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT: &str = env!("AMDC_GIT_DESCRIBE");

fn esc(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// The expression tree of a contract expression.
pub fn ext_json(e: &Ext) -> String {
    match e {
        Ext::Const(c) => format!("{{\"c\":{c}}}"),
        Ext::Ka(o) => format!("{{\"ka\":{o}}}"),
        Ext::Grid(a) => format!("{{\"grid\":{a}}}"),
        Ext::Sym(n) => format!("{{\"sym\":\"{}\"}}", esc(n)),
        Ext::Add(a, b) => format!("{{\"add\":[{},{}]}}", ext_json(a), ext_json(b)),
        Ext::Sub(a, b) => format!("{{\"sub\":[{},{}]}}", ext_json(a), ext_json(b)),
        Ext::Mul(a, b) => format!("{{\"mul\":[{},{}]}}", ext_json(a), ext_json(b)),
        Ext::CeilDiv(a, k) => format!("{{\"ceil_div\":[{},{k}]}}", ext_json(a)),
        Ext::FloorDiv(a, k) => format!("{{\"floor_div\":[{},{k}]}}", ext_json(a)),
    }
}

fn syms_of(e: &Ext, out: &mut BTreeSet<String>) {
    match e {
        Ext::Sym(n) => {
            out.insert(n.clone());
        }
        Ext::Add(a, b) | Ext::Sub(a, b) | Ext::Mul(a, b) => {
            syms_of(a, out);
            syms_of(b, out);
        }
        Ext::CeilDiv(a, _) | Ext::FloorDiv(a, _) => syms_of(a, out),
        _ => {}
    }
}

/// FNV-1a 64 of a text, 16 hex digits.
pub fn fnv_hex(s: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// FNV-1a 64 over the parts.
pub fn compiler_hash(backend: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in [VERSION, "|", GIT, "|", backend] {
        for b in part.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

/// Kernel-descriptor numbers read back from the emitted assembly (what actually runs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resources {
    pub lds_bytes: u64,
    pub vgpr: u64,
    pub agpr: u64,
    pub sgpr: u64,
    pub scratch_bytes: u64,
}

pub fn resources(asm: &str) -> Resources {
    let get = |k: &str| -> Option<u64> {
        asm.lines().map(|l| l.trim()).find_map(|l| l.strip_prefix(k).and_then(|r| r.trim().parse().ok()))
    };
    let next_v = get(".amdhsa_next_free_vgpr").unwrap_or(0);
    let accum = get(".amdhsa_accum_offset").unwrap_or(next_v);
    Resources {
        lds_bytes: get(".amdhsa_group_segment_fixed_size").unwrap_or(0),
        vgpr: accum.min(next_v),
        agpr: next_v.saturating_sub(accum),
        sgpr: get(".amdhsa_next_free_sgpr").unwrap_or(0),
        scratch_bytes: get(".amdhsa_private_segment_fixed_size").unwrap_or(0),
    }
}

/// The JSON document. `f` is the (possibly specialized) kernel, `spec` the bound values by argument name.
/// `derived`: kernargs the bindings fix through the contract's `ka_is` relations (also constants in the specialized kernel).
pub fn info_json(f: &Function, spec: &[(String, u32, SpecValue)], derived: &[(u32, i64)], backend: &str, res: Resources) -> String {
    info_json_src(f, spec, derived, backend, res, "registry")
}

/// As `info_json`, with the kernel's source: `registry` or `ir:<16-hex FNV-1a of the irv1 text>` (part of the JIT cache key).
pub fn info_json_src(f: &Function, spec: &[(String, u32, SpecValue)], derived: &[(u32, i64)], backend: &str, res: Resources, source: &str) -> String {
    let mut args: Vec<&amdgpu_dsl::Arg> = f.args.iter().collect();
    args.sort_by_key(|a| a.kernarg_offset);
    let mut ja: Vec<String> = Vec::new();
    let mut pos = 0u32;
    let pad = |off: u32, size: u32| format!("{{\"name\":\"_pad{off}\",\"offset\":{off},\"size\":{size},\"kind\":\"pad\",\"specializable\":false,\"specialized\":null}}");
    for a in &args {
        if a.kernarg_offset > pos {
            ja.push(pad(pos, a.kernarg_offset - pos));
        }
        let sv = spec
            .iter()
            .find(|(n, _, _)| *n == a.name)
            .map(|(_, _, v)| v.to_json())
            .or_else(|| derived.iter().find(|(o, _)| *o == a.kernarg_offset).map(|(_, v)| v.to_string()))
            .unwrap_or_else(|| "null".to_string());
        ja.push(format!(
            "{{\"name\":\"{}\",\"offset\":{},\"size\":{},\"kind\":\"{}\",\"specializable\":{},\"specialized\":{}}}",
            esc(&a.name),
            a.kernarg_offset,
            a.size,
            arg_kind(a.ty),
            specializable(a.ty),
            sv
        ));
        pos = pos.max(a.kernarg_offset + a.size);
    }
    if f.kernarg_size > pos {
        ja.push(pad(pos, f.kernarg_size - pos));
    }
    let c = &f.contract;
    let grid: Vec<String> = c.grid_defs.iter().map(|(ax, e)| format!("{{\"axis\":{ax},\"ext\":{}}}", ext_json(e))).collect();
    let ka_is: Vec<String> = c.ka_defs.iter().map(|(o, e)| format!("{{\"offset\":{o},\"ext\":{}}}", ext_json(e))).collect();
    let hi = |h: i64| if h == i64::MAX { "null".to_string() } else { h.to_string() };
    let ka_range: Vec<String> = c.ka_ranges.iter().map(|(o, lo, h)| format!("{{\"offset\":{o},\"lo\":{lo},\"hi\":{}}}", hi(*h))).collect();
    let ka_set: Vec<String> = c.ka_sets.iter().map(|(o, vs)| format!("{{\"offset\":{o},\"values\":[{}]}}", vs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))).collect();
    let tensors: Vec<String> = c
        .extents
        .iter()
        .map(|e| format!("{{\"arg\":\"{}\",\"alloc_bytes\":{},\"logical_bytes\":{}}}", esc(&e.arg), ext_json(&e.alloc), e.logical.as_ref().map(ext_json).unwrap_or_else(|| "null".to_string())))
        .collect();
    let mut names: BTreeSet<String> = BTreeSet::new();
    for e in c.grid_defs.iter().map(|x| &x.1).chain(c.ka_defs.iter().map(|x| &x.1)).chain(c.extents.iter().map(|x| &x.alloc)).chain(c.extents.iter().filter_map(|x| x.logical.as_ref())) {
        syms_of(e, &mut names);
    }
    for (n, _, _) in &c.sym_ranges {
        names.insert(n.clone());
    }
    let syms: Vec<String> = names
        .iter()
        .map(|n| {
            let r = c.sym_ranges.iter().find(|(x, _, _)| x == n);
            let (lo, h) = r.map(|x| (x.1, x.2)).unwrap_or((1, i64::MAX));
            format!("{{\"name\":\"{}\",\"lo\":{lo},\"hi\":{}}}", esc(n), hi(h))
        })
        .collect();
    let specd: Vec<String> = spec.iter().map(|(n, _, v)| format!("\"{}\":{}", esc(n), v.to_json())).collect();
    let name_of = |o: u32| f.args.iter().find(|a| a.kernarg_offset == o).map(|a| a.name.clone()).unwrap_or_else(|| format!("ka{o}"));
    let derd: Vec<String> = derived.iter().map(|(o, v)| format!("\"{}\":{v}", esc(&name_of(*o)))).collect();
    format!(
        "{{\"schema\":1,\"source\":\"{}\",\"kernel\":\"{k}\",\"symbol\":\"{k}\",\"arch\":\"{}\",\"compiler\":{{\"version\":\"{}\",\"git\":\"{}\"}},\"compiler_hash\":\"{}\",\"backend\":\"{}\",\"specialize\":{{{}}},\"derived\":{{{}}},\
\"block\":[{},1,1],\"lds_bytes\":{},\"vgpr\":{},\"agpr\":{},\"sgpr\":{},\"scratch_bytes\":{},\"kernarg_size\":{},\"kernarg_align\":8,\"args\":[{}],\
\"contract\":{{\"grid\":[{}],\"ka_is\":[{}],\"ka_range\":[{}],\"ka_set\":[{}],\"tensors\":[{}],\"syms\":[{}]}}}}",
        esc(source),
        amdgpu_codegen::target::T.name,
        esc(VERSION),
        esc(GIT),
        compiler_hash(backend),
        esc(backend),
        specd.join(","),
        derd.join(","),
        f.workgroup_size,
        res.lds_bytes,
        res.vgpr,
        res.agpr,
        res.sgpr,
        res.scratch_bytes,
        f.kernarg_size,
        ja.join(","),
        grid.join(","),
        ka_is.join(","),
        ka_range.join(","),
        ka_set.join(","),
        tensors.join(","),
        syms.join(","),
        k = esc(&f.name)
    )
}
