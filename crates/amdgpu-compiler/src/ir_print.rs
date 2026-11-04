//! Readable printing and structural equality for the DSL IR and MIR.
//!
//! The DSL is currently a flat statement list with paired scope markers (item B will make the
//! regions nested). `print_dsl` first reconstructs a small scope tree (`Node`) and then prints it
//! recursively; only `parse_stmt_scope` needs changing when the IR becomes nested. Structural
//! equality is defined on an alpha-canonical text form: ValueIds/VarIds are renamed by first
//! occurrence, so two builders that allocate different SSA ids but construct the same program are
//! equal and hash equal. Function names are intentionally not part of the structural identity
//! (the expr-layer probes and their manual twins have different names).
//!
//! MIR printing takes an optional register allocation: `print_mir_raw` shows virtual registers and
//! classes; `print_mir_alloc` shows physical `vN`/`sN` plus the original class.

use amdgpu_dsl::{Contract, Ext, Function, Op, Opnd, Stmt, VarId, ValueId};
use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{Hash, Hasher};

use crate::mir::{Block, MFunction, MInst, Src, Term, VReg};
use crate::regalloc::Alloc;

// ---------------------------------------------------------------------------------------------
// alpha canonicalisation
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Canon {
    vals: HashMap<u32, u32>,
    vars: HashMap<VarId, u32>,
    next_v: u32,
    next_r: u32,
}

impl Canon {
    fn val(&mut self, v: u32) -> String {
        let n = *self.vals.entry(v).or_insert_with(|| { let n = self.next_v; self.next_v += 1; n });
        format!("%v{n}")
    }
    fn var(&mut self, v: VarId) -> String {
        let n = *self.vars.entry(v).or_insert_with(|| { let n = self.next_r; self.next_r += 1; n });
        format!("%r{n}")
    }
    /// Replace Rust `Debug` spelling of ids with the canonical names. This keeps the canonical
    /// form exhaustive for all 100+ Op/Stmt variants without a per-variant match that item B would
    /// have to update.
    fn canon_text(&mut self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let b = s.as_bytes();
        let mut i = 0;
        while i < b.len() {
            for (pat, is_var) in [("ValueId(", false), ("VarId(", true)] {
                if s[i..].starts_with(pat) {
                    let start = i + pat.len();
                    if let Some(close) = s[start..].find(')') {
                        if let Ok(n) = s[start..start + close].parse::<u32>() {
                            let name = if is_var {
                                self.var(VarId(n))
                            } else {
                                self.val(n)
                            };
                            out.push_str(&name);
                            i = start + close + 1;
                            break;
                        }
                    }
                }
            }
            if i < b.len() {
                out.push(b[i] as char);
                i += 1;
            }
        }
        out
    }
}

fn canon_function(f: &Function, include_contract: bool) -> String {
    let mut c = Canon::default();
    let mut s = String::new();
    // Signature/ABI/contract/tiles are structural, but the function *name* is not.
    let _ = writeln!(s, "kernarg={} wg={} lds={}", f.kernarg_size, f.workgroup_size, f.lds_bytes);
    let _ = writeln!(s, "args={:?}", f.args);
    if include_contract {
        let _ = writeln!(s, "contract={:?}", f.contract);
    }
    let _ = writeln!(s, "value_tys={:?}", f.value_tys);
    let _ = writeln!(s, "var_tys={:?}", f.var_tys);
    let _ = writeln!(s, "profile={:?} lower={:?} sched={:?}", f.compile_profile, f.lower_opts, f.sched_opts);
    let _ = writeln!(s, "allow_var_rereads={}", f.allow_var_rereads);
    let _ = writeln!(s, "agpr={}", c.canon_text(&format!("{:?}", f.agpr)));
    for st in &f.stmts {
        let line = c.canon_text(&format!("{st:?}"));
        let _ = writeln!(s, "{line}");
    }
    s
}

/// Structural equality over the DSL IR (alpha-equivalent ValueIds/VarIds; function name ignored).
pub fn dsl_structural_eq(a: &Function, b: &Function) -> bool {
    canon_function(a, true) == canon_function(b, true)
}

/// Body/ABI structural equality ignoring the declared contract metadata. Useful when comparing a
/// builder refactor before the same annotations are added to both versions (e.g. the expr-layer
/// probes vs their older manual twins).
pub fn dsl_body_structural_eq(a: &Function, b: &Function) -> bool {
    canon_function(a, false) == canon_function(b, false)
}

/// Structural hash consistent with [`dsl_structural_eq`].
pub fn dsl_structural_hash(f: &Function) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    canon_function(f, true).hash(&mut h);
    h.finish()
}

/// Structural hash of the body/ABI without contract metadata.
pub fn dsl_body_structural_hash(f: &Function) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    canon_function(f, false).hash(&mut h);
    h.finish()
}

/// Canonical alpha-renamed text form (useful for golden tests / diagnostics).
pub fn canonical_dsl(f: &Function) -> String {
    canon_function(f, true)
}

/// Canonical form without contract metadata.
pub fn canonical_dsl_body(f: &Function) -> String {
    canon_function(f, false)
}

// ---------------------------------------------------------------------------------------------
// DSL scope tree + pretty printer
// ---------------------------------------------------------------------------------------------

enum Node<'a> {
    Leaf(&'a Stmt),
    Loop { begin: &'a Stmt, body: Vec<Node<'a>>, end: Option<&'a Stmt> },
    If { begin: &'a Stmt, then_body: Vec<Node<'a>>, else_body: Vec<Node<'a>>, end: Option<&'a Stmt> },
    Tile { begin: &'a Stmt, body: Vec<Node<'a>>, end: Option<&'a Stmt> },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stop { Loop, If, Tile, Top }

fn is_tile_marker(st: &Stmt) -> bool {
    matches!(st, Stmt::TileProduce { .. } | Stmt::TileConsume { .. })
}

fn parse_scope(stmts: &[Stmt], mut i: usize, stop: Stop) -> (Vec<Node<'_>>, usize) {
    let mut nodes = Vec::new();
    while i < stmts.len() {
        match (&stmts[i], stop) {
            (Stmt::EndLoop, Stop::Loop) | (Stmt::EndIf, Stop::If) | (Stmt::TileEnd, Stop::Tile) => break,
            (Stmt::Else, Stop::If) => break,
            (Stmt::BeginLoop { .. }, _) => {
                let begin = &stmts[i];
                let (body, j) = parse_scope(stmts, i + 1, Stop::Loop);
                let end = if j < stmts.len() && matches!(stmts[j], Stmt::EndLoop) { Some(&stmts[j]) } else { None };
                nodes.push(Node::Loop { begin, body, end });
                i = if end.is_some() { j + 1 } else { j };
            }
            (Stmt::BeginIf { .. }, _) => {
                let begin = &stmts[i];
                let (then_body, j) = parse_scope(stmts, i + 1, Stop::If);
                let mut else_body = Vec::new();
                let mut k = j;
                let mut saw_else = false;
                if k < stmts.len() && matches!(stmts[k], Stmt::Else) {
                    saw_else = true;
                    let (e, l) = parse_scope(stmts, k + 1, Stop::If);
                    else_body = e;
                    k = l;
                }
                let end = if k < stmts.len() && matches!(stmts[k], Stmt::EndIf) { Some(&stmts[k]) } else { None };
                let _ = saw_else;
                nodes.push(Node::If { begin, then_body, else_body, end });
                i = if end.is_some() { k + 1 } else { k };
            }
            (s, _) if is_tile_marker(s) => {
                let begin = &stmts[i];
                let (body, j) = parse_scope(stmts, i + 1, Stop::Tile);
                let end = if j < stmts.len() && matches!(stmts[j], Stmt::TileEnd) { Some(&stmts[j]) } else { None };
                nodes.push(Node::Tile { begin, body, end });
                i = if end.is_some() { j + 1 } else { j };
            }
            (s, _) => {
                nodes.push(Node::Leaf(s));
                i += 1;
            }
        }
    }
    (nodes, i)
}

fn ty_of_value(f: &Function, v: ValueId) -> &'static str {
    f.value_ty(v).map(|t| t.name()).unwrap_or("?")
}

fn fmt_opnd_real(o: &Opnd) -> String {
    match o {
        Opnd::Val(v) => format!("%v{v}"),
        Opnd::Var(v) => format!("%r{}", v.0),
        Opnd::I32(x) => x.to_string(),
        Opnd::F32(x) => format!("{x:?}f"),
    }
}

/// Debug spelling with real (not alpha-canonical) names, used by the human-readable printer.
fn fmt_real_debug(s: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        let mut hit = false;
        for pat in ["ValueId(", "VarId("] {
            if s[i..].starts_with(pat) {
                let start = i + pat.len();
                if let Some(close) = s[start..].find(')') {
                    if let Ok(n) = s[start..start + close].parse::<u32>() {
                        out.push_str(if pat == "VarId(" { "%r" } else { "%v" });
                        out.push_str(&n.to_string());
                        i = start + close + 1;
                        hit = true;
                        break;
                    }
                }
            }
        }
        if !hit {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

fn fmt_op(op: &Op) -> String {
    fmt_real_debug(&format!("{op:?}"))
}

fn print_contract(c: &Contract, out: &mut String) {
    let _ = writeln!(out, "  contract:");
    for t in &c.tiles {
        let _ = writeln!(out, "    tile {}{} base={} slot={} slots={}{}", t.name,
            if t.private { " private" } else if t.global { " global" } else { "" }, t.base, t.slot_size, t.slots, "");
    }
    if !c.tile_aliases.is_empty() { let _ = writeln!(out, "    tile_aliases={:?}", c.tile_aliases); }
    if let Some(t) = &c.tile_trust { let _ = writeln!(out, "    tile_trust={t:?}"); }
    for e in &c.extents {
        let _ = writeln!(out, "    arg_tensor {} alloc={:?}{}", e.arg, e.alloc,
            e.logical.as_ref().map(|l| format!(" logical={l:?}")).unwrap_or_default());
    }
    if !c.ka_ranges.is_empty() { let _ = writeln!(out, "    ka_ranges={:?}", c.ka_ranges); }
    if !c.grid_defs.is_empty() { let _ = writeln!(out, "    grid_defs={:?}", c.grid_defs.iter().map(|(a,e)| (a, e)).collect::<Vec<_>>()); }
    for (o, e) in &c.ka_defs { let _ = writeln!(out, "    ka_is 0x{o:X} = {e:?}"); }
    for (o, vs) in &c.ka_sets { let _ = writeln!(out, "    ka_one_of 0x{o:X} = {vs:?}"); }
    if !c.sym_ranges.is_empty() { let _ = writeln!(out, "    sym_ranges={:?}", c.sym_ranges); }
    if !c.buf_regions.is_empty() { let _ = writeln!(out, "    buf_regions={:?}", c.buf_regions); }
    if !c.last_arrivers.is_empty() { let _ = writeln!(out, "    last_arrivers={:?}", c.last_arrivers); }
    if !c.buf_trusts.is_empty() { let _ = writeln!(out, "    buf_trusts={:?}", c.buf_trusts); }
}

fn print_node(n: &Node<'_>, depth: usize, f: &Function, out: &mut String) {
    let pad = "  ".repeat(depth);
    match n {
        Node::Leaf(st) => {
            match st {
                Stmt::Let { dst, op } => {
                    let _ = writeln!(out, "{pad}%v{}: {} = {}", dst, ty_of_value(f, *dst), fmt_op(op));
                }
                Stmt::SetVar { var, value } => {
                    let _ = writeln!(out, "{pad}%r{} <- {}", var.0, fmt_opnd_real(value));
                }
                Stmt::Assume { target, lo, hi, trusted } => {
                    let _ = writeln!(out, "{pad}assume{}{} in [{:?}, {:?}]", if trusted.is_some() { "_trusted" } else { "" }, fmt_opnd_real(target), lo, hi);
                }
                s => { let _ = writeln!(out, "{pad}{}", fmt_real_debug(&format!("{s:?}"))); }
            }
        }
        Node::Loop { begin, body, end } => {
            let _ = writeln!(out, "{pad}{}", fmt_real_debug(&format!("{begin:?}")).replace("BeginLoop { ", "loop ").trim_end_matches('}').to_string());
            for x in body { print_node(x, depth + 1, f, out); }
            if let Some(e) = end { let _ = writeln!(out, "{pad}{}", fmt_real_debug(&format!("{e:?}"))); }
        }
        Node::If { begin, then_body, else_body, end } => {
            let _ = writeln!(out, "{pad}{}", fmt_real_debug(&format!("{begin:?}")));
            for x in then_body { print_node(x, depth + 1, f, out); }
            if !else_body.is_empty() {
                let _ = writeln!(out, "{pad}else");
                for x in else_body { print_node(x, depth + 1, f, out); }
            }
            if let Some(e) = end { let _ = writeln!(out, "{pad}{}", fmt_real_debug(&format!("{e:?}"))); }
        }
        Node::Tile { begin, body, end } => {
            let _ = writeln!(out, "{pad}{}", fmt_real_debug(&format!("{begin:?}")));
            for x in body { print_node(x, depth + 1, f, out); }
            if let Some(e) = end { let _ = writeln!(out, "{pad}{}", fmt_real_debug(&format!("{e:?}"))); }
        }
    }
}

/// Human-readable DSL IR. Scope indentation comes from a reconstructed tree, so item B's nested
/// regions only require replacing [`parse_scope`].
pub fn print_dsl(f: &Function) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "dsl @{}", f.name);
    let _ = writeln!(out, "  kernarg_size={} workgroup={} lds={}", f.kernarg_size, f.workgroup_size, f.lds_bytes);
    let _ = writeln!(out, "  profile={:?} lower_opts={:?} sched_opts={:?}", f.compile_profile, f.lower_opts, f.sched_opts);
    for a in &f.args {
        let _ = writeln!(out, "  arg {}: {} @0x{:X} +{}", a.name, a.ty.name(), a.kernarg_offset, a.size);
    }
    print_contract(&f.contract, &mut out);
    let (nodes, _) = parse_scope(&f.stmts, 0, Stop::Top);
    let _ = writeln!(out, "  body:");
    for n in &nodes { print_node(n, 2, f, &mut out); }
    out
}

// ---------------------------------------------------------------------------------------------
// MIR printer
// ---------------------------------------------------------------------------------------------

fn reg_name(r: VReg, m: &MFunction, alloc: Option<&Alloc>) -> String {
    let cls = m.vregs.get(r as usize).copied().unwrap_or(crate::mir::RClass::V1);
    match alloc {
        Some(a) => {
            let p = a.phys.get(r as usize).copied().unwrap_or(r);
            if *a.scalar.get(r as usize).unwrap_or(&false) { format!("s{p}:{cls:?}") } else { format!("v{p}:{cls:?}") }
        }
        None => format!("v{r}:{cls:?}"),
    }
}

fn fmt_src(s: Src, m: &MFunction, alloc: Option<&Alloc>) -> String {
    match s {
        Src::Reg(r) => reg_name(r, m, alloc),
        Src::Imm(v) => v.to_string(),
    }
}

fn fmt_inst(i: &MInst, m: &MFunction, alloc: Option<&Alloc>) -> String {
    let mut s = String::new();
    if let Some(d) = i.dst { s.push_str(&format!("{} = ", reg_name(d, m, alloc))); }
    s.push_str(&format!("{:?}", i.op));
    if !i.srcs.is_empty() {
        s.push_str(" (");
        s.push_str(&i.srcs.iter().map(|x| fmt_src(*x, m, alloc)).collect::<Vec<_>>().join(", "));
        s.push(')');
    }
    if i.imm != 0 { s.push_str(&format!(" imm={}", i.imm)); }
    if i.auto { s.push_str(" [auto]"); }
    if i.pin { s.push_str(" [pin]"); }
    s
}

fn fmt_term(t: &Term) -> String {
    match t {
        Term::Goto(l) => format!("goto b{l}"),
        Term::CondGoto { on, target, fall } => format!("if {on:?} goto b{target} else b{fall}"),
        Term::Endpgm => "endpgm".into(),
    }
}

fn print_block(b: &Block, m: &MFunction, alloc: Option<&Alloc>, out: &mut String) {
    let _ = writeln!(out, "  b{}:{}", b.label, b.pred.map(|p| format!(" pred={p:?}")).unwrap_or_default());
    for i in &b.insts { let _ = writeln!(out, "    {}", fmt_inst(i, m, alloc)); }
    let _ = writeln!(out, "    {}", fmt_term(&b.term));
}

/// MIR with virtual registers (pre-RA, i.e. `lower::lower`) or physical registers if `alloc` is
/// supplied (post-RA).
pub fn print_mir(m: &MFunction, alloc: Option<&Alloc>) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "mir @{}", m.name);
    let _ = writeln!(out, "  kernarg_size={} workgroup={} lds={} blocks={} vregs={}", m.kernarg_size, m.workgroup_size, m.lds_bytes, m.blocks.len(), m.vregs.len());
    for a in &m.args { let _ = writeln!(out, "  arg {}: {} @0x{:X} +{}", a.name, a.dsl_ty.name(), a.offset, a.size); }
    for b in &m.blocks { print_block(b, m, alloc, &mut out); }
    out
}

/// Convenience: lower + allocate and print the post-RA MIR.
pub fn print_mir_ra(f: &Function) -> Result<String, String> {
    let (m, a) = crate::compile_mir_alloc(f)?;
    Ok(print_mir(&m, Some(&a)))
}
