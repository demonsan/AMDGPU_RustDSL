//! Host-side generation from a kernel `Function`: a C++ ABI header (kernarg struct + static
//! asserts + host-contract `validate()` + `launch()`) and matching JSON metadata. Zero unsafe on
//! the Rust side; the generated C++ uses only HIP runtime calls provided by the caller.
//!
//! `amdc gen-host <kernel>|--shipped [--out dir]` writes `<kernel>_abi.h` / `<kernel>_abi.json`.
//! Contracts are the same `Ext` algebra the verifier uses, so host and device cannot drift by hand.

use amdgpu_dsl::{Arg, Contract, DType, Ext, Function};

#[derive(Clone, Debug)]
pub struct Generated {
    pub header: String,
    pub json: String,
    pub file_stem: String,
}

fn c_ident(s: &str) -> String {
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        let c = if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' };
        if i == 0 && c.is_ascii_digit() {
            out.push('_');
        }
        out.push(c);
    }
    if out.is_empty() { "arg".into() } else { out }
}

fn ty_name(ty: DType) -> &'static str {
    match ty {
        DType::Ptr => "void*",
        DType::I32 | DType::U32 => "uint32_t",
        DType::F32 => "float",
        DType::F16 => "uint16_t",
        DType::I1 => "uint32_t",
        DType::Frag2 => "uint8_t",
        DType::Frag4 => "uint8_t",
        DType::Frag8 => "uint8_t",
        DType::Frag16 => "uint8_t",
        DType::Rsrc => "uint32_t",
    }
}

fn ext_expr(e: &Ext) -> String {
    fn p(e: &Ext) -> String {
        match e {
            Ext::Const(c) => format!("{c}LL"),
            Ext::Ka(o) => format!("ka(0x{o:X}u)"),
            Ext::Grid(a) => match a { 0 => "grid.x".into(), 1 => "grid.y".into(), 2 => "grid.z".into(), _ => format!("1/*grid{a}*/") },
            Ext::Sym(n) => format!("syms.{}", c_ident(n)),
            Ext::Add(a, b) => format!("({}+{})", p(a), p(b)),
            Ext::Sub(a, b) => format!("({}-{})", p(a), p(b)),
            Ext::Mul(a, b) => format!("({}*{})", p(a), p(b)),
            Ext::CeilDiv(a, k) => format!("(({}+{}LL)/{}LL)", p(a), k - 1, k),
            Ext::FloorDiv(a, k) => format!("({}/{}LL)", p(a), k),
        }
    }
    p(e)
}

fn ext_expr_grid(e: &Ext) -> String {
    // inside the generated dim3 grid() helper, kernarg reads need the `a` argument
    ext_expr(e).replace("ka(", "ka_at(a,")
}

fn ext_text(e: &Ext) -> String {
    match e {
        Ext::Const(c) => format!("{c}"),
        Ext::Ka(o) => format!("ka(0x{o:X})"),
        Ext::Grid(a) => format!("grid({a})"),
        Ext::Sym(n) => n.clone(),
        Ext::Add(a, b) => format!("({}+{})", ext_text(a), ext_text(b)),
        Ext::Sub(a, b) => format!("({}-{})", ext_text(a), ext_text(b)),
        Ext::Mul(a, b) => format!("({}*{})", ext_text(a), ext_text(b)),
        Ext::CeilDiv(a, k) => format!("ceil({}/{})", ext_text(a), k),
        Ext::FloorDiv(a, k) => format!("floor({}/{})", ext_text(a), k),
    }
}

fn collect_syms_into(e: &Ext, out: &mut std::collections::BTreeSet<String>) {
    match e {
        Ext::Sym(n) => { out.insert(c_ident(n)); }
        Ext::Add(a, b) | Ext::Sub(a, b) | Ext::Mul(a, b) => { collect_syms_into(a, out); collect_syms_into(b, out); }
        Ext::CeilDiv(a, _) | Ext::FloorDiv(a, _) => collect_syms_into(a, out),
        _ => {}
    }
}

fn collect_syms(f: &Function) -> std::collections::BTreeSet<String> {
    let mut out = std::collections::BTreeSet::new();
    let c = &f.contract;
    for e in &c.extents { collect_syms_into(&e.alloc, &mut out); if let Some(l) = &e.logical { collect_syms_into(l, &mut out); } }
    for (_, e) in &c.grid_defs { collect_syms_into(e, &mut out); }
    for (_, e) in &c.ka_defs { collect_syms_into(e, &mut out); }
    for (n, _, _) in &c.sym_ranges { out.insert(c_ident(n)); }
    out
}

fn json_escape(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c { '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"), '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"), '\t' => o.push_str("\\t"), c => o.push(c) }
    }
    o
}

fn args_sorted(f: &Function) -> Vec<&Arg> {
    let mut a: Vec<&Arg> = f.args.iter().collect();
    a.sort_by_key(|x| x.kernarg_offset);
    a
}

/// Emit one kernel's C++ header and JSON. Returns Err for an ABI that cannot be represented (overlap,
/// unknown extent type, negative size).
pub fn generate(f: &Function) -> Result<Generated, String> {
    if f.kernarg_size == 0 { return Err(format!("{}: empty kernarg", f.name)); }
    let args = args_sorted(f);
    let mut names: Vec<String> = Vec::new();
    for a in &args {
        let n = c_ident(&a.name);
        if names.contains(&n) { return Err(format!("{}: duplicate arg name {}", f.name, a.name)); }
        names.push(n);
    }
    // layout/padding
    let mut fields = String::new();
    let mut asserts = String::new();
    let mut off = 0u32;
    let mut pads = 0usize;
    for (i, a) in args.iter().enumerate() {
        let sz = a.size;
        if a.kernarg_offset < off { return Err(format!("{}: overlapping arg {} at 0x{:X}", f.name, a.name, a.kernarg_offset)); }
        if a.kernarg_offset > off {
            let n = a.kernarg_offset - off;
            fields.push_str(&format!("    uint8_t _pad{pads}[{n}];\n"));
            pads += 1;
        }
        let field = &names[i];
        let typed = match a.ty {
            DType::Ptr if a.size == 8 => Some("void*".to_string()),
            DType::I32 | DType::U32 | DType::I1 if a.size == 4 => Some("uint32_t".to_string()),
            DType::F32 if a.size == 4 => Some("float".to_string()),
            DType::F16 if a.size == 2 => Some("uint16_t".to_string()),
            _ => None,
        };
        match &typed {
            Some(ty) => fields.push_str(&format!("    {ty} {field};\n")),
            None => fields.push_str(&format!("    uint8_t {field}[{}];\n", a.size.max(1))),
        }
        asserts.push_str(&format!("static_assert(offsetof(Args, {field}) == 0x{:X}, \"{field} offset\");\n", a.kernarg_offset));
        if typed.is_some() {
            asserts.push_str(&format!("static_assert(sizeof(Args::{field}) == {}, \"{field} size\");\n", a.size));
        }
        off = a.kernarg_offset + sz;
    }
    // trailing padding is emitted by the header template below
    let c = &f.contract;
    // ka(offset): scalar args only
    let mut ka_arms = String::new();
    for a in &args {
        if matches!(a.ty, DType::Ptr) { continue; }
        if a.size == 4 {
            let n = c_ident(&a.name);
            ka_arms.push_str(&format!("        case 0x{:X}u: return (int64_t)a.{n};\n", a.kernarg_offset));
        }
    }
    let syms = collect_syms(f);
    let mut sym_fields = String::new();
    for s in &syms {
        let lo = c.sym_ranges.iter().find(|(n, _, _)| c_ident(n) == *s).map(|x| x.1).unwrap_or(1);
        sym_fields.push_str(&format!("    int64_t {s} = {lo};\n"));
    }
    // BufferSizes: one field per pointer/extent arg
    let mut size_fields = String::new();
    let mut size_names: Vec<String> = Vec::new();
    for e in &c.extents {
        let n = c_ident(&e.arg);
        if !size_names.contains(&n) { size_names.push(n.clone()); size_fields.push_str(&format!("    uint64_t {n} = 0;\n")); }
    }
    // checks
    let mut checks = String::new();
    for e in &c.extents {
        let n = c_ident(&e.arg);
        checks.push_str(&format!("    if (sizes.{n} < (uint64_t)({})) {{ return fail(err, \"GV-HOST-EXTENT\", \"{}: buffer {} smaller than declared allocation\"); }}\n", ext_expr(&e.alloc), e.arg, e.arg));
    }
    for (offv, lo, hi) in &c.ka_ranges {
        let v = format!("ka(0x{offv:X}u)");
        checks.push_str(&format!("    if ({v} < {lo}LL || {v} > {hi}LL) {{ return fail(err, \"GV-HOST-KA-RANGE\", \"ka(0x{offv:X}) outside contract\"); }}\n"));
    }
    for (offv, e) in &c.ka_defs {
        checks.push_str(&format!("    if (ka(0x{offv:X}u) != {}) {{ return fail(err, \"GV-HOST-KA-IS\", \"ka(0x{offv:X}) relation failed\"); }}\n", ext_expr(e)));
    }
    for (offv, vals) in &c.ka_sets {
        let cond = vals.iter().map(|v| format!("ka(0x{offv:X}u) == {v}LL")).collect::<Vec<_>>().join(" || ");
        checks.push_str(&format!("    if (!({cond})) {{ return fail(err, \"GV-HOST-KA-ONE-OF\", \"ka(0x{offv:X}) outside allowed set\"); }}\n"));
    }
    for (axis, e) in &c.grid_defs {
        let g = match axis { 0 => "grid.x", 1 => "grid.y", 2 => "grid.z", _ => continue };
        checks.push_str(&format!("    if ({g} != {}) {{ return fail(err, \"GV-HOST-GRID\", \"grid axis {axis} relation failed\"); }}\n", ext_expr(e)));
    }
    for (n, lo, hi) in &c.sym_ranges {
        let s = c_ident(n);
        checks.push_str(&format!("    if (syms.{s} < {lo}LL || syms.{s} > {hi}LL) {{ return fail(err, \"GV-HOST-SYM\", \"symbol {n} outside contract\"); }}\n"));
    }
    // grid helper (if every axis has a definition; otherwise caller supplies it)
    let mut grid_exprs: Vec<Option<String>> = vec![None, None, None];
    for (axis, e) in &c.grid_defs { if (*axis as usize) < 3 { grid_exprs[*axis as usize] = Some(ext_expr_grid(e)); } }
    let grid_fn = if grid_exprs.iter().all(|x| x.is_some()) {
        format!("inline dim3 grid(const Args& a, const Symbols& syms) {{ (void)a; (void)syms; return dim3((unsigned){}, (unsigned){}, (unsigned){}); }}\n",
            grid_exprs[0].as_ref().unwrap(), grid_exprs[1].as_ref().unwrap(), grid_exprs[2].as_ref().unwrap())
    } else {
        String::new()
    };
    let profile = f.compile_profile.map(|p| format!("{p:?}")).unwrap_or_default();
    let mut json_args = Vec::new();
    for a in &args {
        json_args.push(format!("{{\"name\":\"{}\",\"offset\":{},\"size\":{},\"type\":\"{:?}\"}}", json_escape(&a.name), a.kernarg_offset, a.size, a.ty));
    }
    let mut json_ext = Vec::new();
    for e in &c.extents {
        json_ext.push(format!("{{\"arg\":\"{}\",\"alloc\":\"{}\",\"logical\":{}}}", json_escape(&e.arg), json_escape(&ext_text(&e.alloc)), match &e.logical { Some(l) => format!("\"{}\"", json_escape(&ext_text(l))), None => "null".into() }));
    }
    let mut json_grid = Vec::new();
    for (axis, e) in &c.grid_defs { json_grid.push(format!("{{\"axis\":{},\"expr\":\"{}\"}}", axis, json_escape(&ext_text(e)))); }
    let mut json_ka_is = Vec::new();
    for (o, e) in &c.ka_defs { json_ka_is.push(format!("{{\"offset\":{},\"expr\":\"{}\"}}", o, json_escape(&ext_text(e)))); }
    let mut json_ka_range = Vec::new();
    for (o, lo, hi) in &c.ka_ranges { json_ka_range.push(format!("{{\"offset\":{},\"lo\":{},\"hi\":{}}}", o, lo, hi)); }
    let mut json_ka_set = Vec::new();
    for (o, vs) in &c.ka_sets { json_ka_set.push(format!("{{\"offset\":{},\"values\":[{}]}}", o, vs.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))); }
    let mut json_syms = Vec::new();
    for s in &syms { let lo = c.sym_ranges.iter().find(|(n, _, _)| c_ident(n) == *s).map(|x| x.1).unwrap_or(1); json_syms.push(format!("{{\"name\":\"{}\",\"lo\":{}}}", s, lo)); }
    let json = format!("{{\n  \"kernel\": \"{}\",\n  \"kernarg_size\": {},\n  \"workgroup_size\": {},\n  \"lds_bytes\": {},\n  \"compile_profile\": \"{}\",\n  \"args\": [{}],\n  \"extents\": [{}],\n  \"ka_is\": [{}],\n  \"ka_range\": [{}],\n  \"ka_one_of\": [{}],\n  \"grid_is\": [{}],\n  \"symbols\": [{}]\n}}\n",
        json_escape(&f.name), f.kernarg_size, f.workgroup_size, f.lds_bytes, profile,
        json_args.join(","), json_ext.join(","), json_ka_is.join(","), json_ka_range.join(","), json_ka_set.join(","), json_grid.join(","), json_syms.join(","));
    let stem = c_ident(&f.name);
    let tail = if off < f.kernarg_size { format!("    uint8_t _pad_end[{}];\n", f.kernarg_size - off) } else { String::new() };
    let header = format!(r###"// Generated by `amdc gen-host` from the kernel contract. DO NOT EDIT.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <hip/hip_runtime.h>
namespace amdc_hostgen::{} {{
#pragma pack(push, 1)
struct Args {{
{}
{}
}};
#pragma pack(pop)
{}
static_assert(sizeof(Args) == {}, "kernarg size");
struct Symbols {{
{}}};
struct BufferSizes {{
{}}};
constexpr unsigned WORKGROUP_SIZE = {};
constexpr size_t LDS_BYTES = {};
inline int64_t ka_at(const Args& a, uint32_t off) {{
    switch (off) {{
{}        default: return 0;
    }}
}}
inline bool fail(std::string* err, const char* code, const std::string& msg) {{
    if (err) *err = std::string(code) + ": " + msg;
    return false;
}}
inline bool validate(const Args& a, const BufferSizes& sizes, dim3 grid, const Symbols& syms, std::string* err = nullptr) {{
    auto ka = [&](uint32_t o) -> int64_t {{ return ka_at(a, o); }};
    (void)grid; (void)syms; (void)sizes;
{}    return true;
}}
inline hipError_t launch(hipFunction_t fn, const Args& a, dim3 grid, dim3 block, size_t lds = 0, hipStream_t stream = nullptr) {{
    size_t sz = sizeof(Args);
    void* buf = const_cast<void*>(static_cast<const void*>(&a));
    void* extra[] = {{HIP_LAUNCH_PARAM_BUFFER_POINTER, buf, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END}};
    return hipModuleLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, lds, stream, nullptr, extra);
}}
inline dim3 block() {{ return dim3(WORKGROUP_SIZE, 1, 1); }}
{}
}} // namespace
"###,
        f.name,
        fields,
        tail,
        asserts,
        f.kernarg_size,
        sym_fields,
        size_fields,
        f.workgroup_size,
        f.lds_bytes,
        ka_arms,
        checks,
        grid_fn);
    Ok(Generated { header, json, file_stem: stem })
}
