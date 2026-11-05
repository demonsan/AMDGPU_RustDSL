//! Readable printing of the MIR (`amdc print-ir --stage mir|ra`): `print_mir` shows virtual registers and classes, or physical `vN`/`sN`
//! plus the original class when an allocation is given. The DSL printer and the structural equality live in `amdgpu_dsl::print`
//! (re-exported here).

pub use amdgpu_dsl::print::*;

use amdgpu_dsl::Function;
use std::fmt::Write;

use crate::mir::{Block, MFunction, MInst, Src, Term, VReg};
use crate::regalloc::Alloc;

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
