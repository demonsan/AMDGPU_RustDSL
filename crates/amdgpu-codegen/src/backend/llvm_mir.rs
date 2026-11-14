//! Export of the final allocated `MFunction` as post-RA LLVM MIR (text), ready for
//! `llc -start-before=<tail start>`.
//!
//! Instruction order, register assignment and block layout are ours and are passed through
//! unchanged; LLVM's tail passes only insert (waits, nops, delay_alu, clauses) and never reorder.
//! Own `s_nop` padding is dropped (LLVM recomputes hazards). Explicit and auto-inserted
//! `s_waitcnt` are kept; SIInsertWaitcnts merges them with what it computes.

use std::collections::{BTreeSet, HashMap};

use super::llvm20::{self, RenderCtx, Table};
use crate::mir::*;
use crate::regalloc::Alloc;
use amdgpu_dsl::DType;

/// Physical register slot used by the liveness computation (per dword).
type Slot = u32;
const SGPR_BASE: Slot = 10_000;
const AGPR_BASE: Slot = 30_000;
const VCC: Slot = 20_000;
const SCC: Slot = 20_001;

fn slot_name(s: Slot) -> String {
    match s {
        VCC => "$vcc".to_string(),
        SCC => "$scc".to_string(),
        s if s >= AGPR_BASE => format!("$agpr{}", s - AGPR_BASE),
        s if s >= SGPR_BASE => format!("$sgpr{}", s - SGPR_BASE),
        s => format!("$vgpr{s}"),
    }
}

/// Physical dword slots named by a MIR register token (`$vgpr4_vgpr5`, `$vcc`, ...).
fn slots_of(name: &str) -> Vec<Slot> {
    let name = name.trim_start_matches('$');
    let name = name.split('.').next().unwrap_or(name);
    match name {
        "vcc" | "vcc_lo" | "vcc_hi" => return vec![VCC],
        "scc" => return vec![SCC],
        _ => {}
    }
    let mut out = Vec::new();
    for part in name.split('_') {
        if let Some(n) = part.strip_prefix("vgpr") {
            if let Ok(n) = n.parse::<u32>() {
                out.push(n);
            }
        } else if let Some(n) = part.strip_prefix("sgpr") {
            if let Ok(n) = n.parse::<u32>() {
                out.push(SGPR_BASE + n);
            }
        } else if let Some(n) = part.strip_prefix("agpr") {
            if let Ok(n) = n.parse::<u32>() {
                out.push(AGPR_BASE + n);
            }
        }
    }
    out
}

/// (defs, uses) of one rendered MIR instruction line.
fn defs_uses(line: &str) -> (Vec<Slot>, Vec<Slot>) {
    let line = line.split(" :: ").next().unwrap_or(line);
    let (lhs, rhs) = match line.split_once(" = ") {
        Some((l, r)) => (l, r),
        None => ("", line),
    };
    let mut defs = Vec::new();
    let mut uses = Vec::new();
    for tok in lhs.split(',') {
        if let Some(i) = tok.find('$') {
            let end = tok[i..].find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$')).map(|e| e + i).unwrap_or(tok.len());
            defs.extend(slots_of(&tok[i..end]));
        }
    }
    let mut rest = rhs;
    while let Some(i) = rest.find('$') {
        let before = &rest[..i];
        let end = rest[i..].find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '$')).map(|e| e + i).unwrap_or(rest.len());
        let slots = slots_of(&rest[i..end]);
        // `implicit-def dead $scc` is a def too (reading it as a use made SCC live into loops that build buffer descriptors)
        if before.ends_with("implicit-def ") || before.ends_with("implicit-def dead ") {
            defs.extend(slots);
        } else {
            uses.extend(slots);
        }
        rest = &rest[end..];
    }
    (defs, uses)
}

fn ir_type(t: DType) -> &'static str {
    match t {
        DType::Ptr => "ptr addrspace(1)",
        DType::F32 => "float",
        _ => "i32",
    }
}

/// IR parameter list reproducing the DSL's kernarg layout. LLVM lays kernel parameters out
/// naturally aligned and packed, so gaps in the DSL layout (the vendor ABI has 16-byte
/// spaced scalars and an opaque tail) are filled with dummy integer parameters. Returns the
/// parameter list and the end offset.
fn kernarg_ir(m: &MFunction) -> Result<(Vec<String>, u32), String> {
    let mut params = Vec::new();
    let mut cursor = 0u32;
    let mut pad_id = 0;
    let mut pad = |cursor: &mut u32, to: u32, params: &mut Vec<String>| {
        while *cursor < to {
            let gap = to - *cursor;
            let ty = if *cursor % 8 == 0 && gap >= 8 {
                ("i64", 8)
            } else if *cursor % 4 == 0 && gap >= 4 {
                ("i32", 4)
            } else {
                ("i8", 1)
            };
            params.push(format!("{} %pad{pad_id}", ty.0));
            pad_id += 1;
            *cursor += ty.1;
        }
    };
    // (in offset order: a kernel may declare its arguments in another order; the IR names keep the DSL index)
    let mut order: Vec<(usize, &crate::mir::MArg)> = m.args.iter().enumerate().collect();
    order.sort_by_key(|(_, a)| a.offset);
    for (i, a) in order {
        if a.offset < cursor {
            return Err(format!("kernarg '{}' at offset {} overlaps the previous argument", a.name, a.offset));
        }
        pad(&mut cursor, a.offset, &mut params);
        let natural = match a.dsl_ty {
            DType::Ptr => a.size == 8 && a.offset % 8 == 0,
            _ => a.size == 4 && a.offset % 4 == 0,
        };
        if natural {
            params.push(format!("{} %a{i}", ir_type(a.dsl_ty)));
            cursor += a.size;
        } else {
            let to = a.offset + a.size;
            pad(&mut cursor, to, &mut params);
        }
    }
    // The own emitter declares the segment rounded up to 8 bytes; keep the LLVM descriptor identical.
    pad(&mut cursor, ((m.kernarg_size + 7) / 8) * 8, &mut params);
    Ok((params, cursor))
}
pub struct Exported {
    pub mir: String,
}

pub fn export(m: &MFunction, a: &Alloc) -> Result<Exported, String> {
    export_with(m, a, false)
}

/// markers: begin every block with a unique s_setprio <block> (oracle tooling only).
pub fn export_with(m: &MFunction, a: &Alloc, markers: bool) -> Result<Exported, String> {
    export_opts(m, a, markers, false)
}

/// drop_auto_waits: omit the s_waitcnt lowering inserted by itself (explicit DSL waits are kept)
/// and leave wait computation to SIInsertWaitcnts.
pub fn export_opts(m: &MFunction, a: &Alloc, markers: bool, drop_auto_waits: bool) -> Result<Exported, String> {
    let table = Table::load();
    let label_idx: HashMap<u32, usize> = m.blocks.iter().enumerate().map(|(i, b)| (b.label, i)).collect();
    let nblocks = m.blocks.len();

    // 1. Render instructions + terminator per block.
    let mut lines: Vec<Vec<String>> = Vec::new();
    let mut succs: Vec<Vec<usize>> = Vec::new();
    for (bi, b) in m.blocks.iter().enumerate() {
        let cx = RenderCtx { m, a, target: None };
        let mut out = Vec::new();
        if markers {
            let mk = MInst::new(MOp::SNop).imm(bi as i64);
            out.push(llvm20::render_text(&table.rows["X_Marker"].text[0], &mk, &cx)?);
            for _ in 0..2 {
                out.push(llvm20::render_text(&table.rows["X_Sep"].text[0], &mk, &cx)?);
            }
        }
        for inst in &b.insts {
            if drop_auto_waits && inst.auto && matches!(inst.op, MOp::SWaitVm | MOp::SWaitLgkm) {
                continue;
            }
            out.extend(llvm20::render_inst(&table, inst, &cx).map_err(|e| format!("block {bi}: {e}"))?);
        }
        if markers {
            out.push(llvm20::render_text(&table.rows["X_Drain"].text[0], &MInst::new(MOp::SNop), &cx)?);
        }
        let next = bi + 1;
        let mut s = Vec::new();
        match &b.term {
            Term::Goto(t) => {
                let ti = *label_idx.get(t).ok_or("unknown label")?;
                if ti != next {
                    out.push(llvm20::render_term(&table, "T_Goto", Some(format!("%bb.{ti}")), m, a)?);
                }
                s.push(ti);
            }
            Term::CondGoto { on, target, fall } => {
                let ti = *label_idx.get(target).ok_or("unknown label")?;
                let fi = *label_idx.get(fall).ok_or("unknown label")?;
                let key = match on {
                    Pred::Scc0 => "T_Scc0",
                    Pred::Scc1 => "T_Scc1",
                    Pred::Vccnz => "T_Vccnz",
                    Pred::Vccz => "T_Vccz",
                };
                out.push(llvm20::render_term(&table, key, Some(format!("%bb.{ti}")), m, a)?);
                if fi != next {
                    out.push(llvm20::render_term(&table, "T_Goto", Some(format!("%bb.{fi}")), m, a)?);
                }
                s.push(ti);
                if fi != ti {
                    s.push(fi);
                }
            }
            Term::Endpgm => out.push(llvm20::render_term(&table, "T_Endpgm", None, m, a)?),
        }
        lines.push(out);
        succs.push(s);
    }

    // 2. Physical-register liveness (backward dataflow) for per-block liveins.
    let mut genset: Vec<BTreeSet<Slot>> = vec![BTreeSet::new(); nblocks];
    let mut kill: Vec<BTreeSet<Slot>> = vec![BTreeSet::new(); nblocks];
    for bi in 0..nblocks {
        for l in &lines[bi] {
            let (d, u) = defs_uses(l);
            for x in u {
                if !kill[bi].contains(&x) {
                    genset[bi].insert(x);
                }
            }
            kill[bi].extend(d);
        }
    }
    let mut live_in: Vec<BTreeSet<Slot>> = genset.clone();
    loop {
        let mut changed = false;
        for bi in (0..nblocks).rev() {
            let mut out: BTreeSet<Slot> = BTreeSet::new();
            for &s in &succs[bi] {
                out.extend(live_in[s].iter().copied());
            }
            let mut new_in = genset[bi].clone();
            for x in out {
                if !kill[bi].contains(&x) {
                    new_in.insert(x);
                }
            }
            if new_in != live_in[bi] {
                live_in[bi] = new_in;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // 2b. kill / dead flags from the same liveness (lets LLVM's peepholes and the verifier see exact liveness).
    {
        for bi in 0..nblocks {
            let mut live: BTreeSet<Slot> = BTreeSet::new();
            for &s in &succs[bi] {
                live.extend(live_in[s].iter().copied());
            }
            annotate_flags(&mut lines[bi], live);
        }
    }

    // 3. Header from the per-version template.
    let (irargs, kernarg) = kernarg_ir(m)?;
    let kernarg = ((kernarg + 7) / 8) * 8;
    let mut mir = llvm20::current().header_src
        .replace("@NAME@", &m.name)
        .replace("@IRARGS@", &irargs.join(", "))
        .replace("@FLATWG@", &m.workgroup_size.to_string())
        .replace("@KERNARG@", &kernarg.to_string())
        .replace("@LDS@", &m.lds_bytes.to_string());
    // 4. Body.
    if !mir.ends_with('\n') {
        mir.push('\n');
    }
    for bi in 0..nblocks {
        mir.push_str(&format!("  bb.{bi}:\n"));
        if !succs[bi].is_empty() {
            let s: Vec<String> = succs[bi].iter().map(|x| format!("%bb.{x}")).collect();
            mir.push_str(&format!("    successors: {}\n", s.join(", ")));
        }
        if !live_in[bi].is_empty() {
            mir.push_str(&format!("    liveins: {}\n", livein_text(&live_in[bi])));
        }
        mir.push_str("  \n");
        for l in &lines[bi] {
            mir.push_str("    ");
            mir.push_str(l);
            mir.push('\n');
        }
        mir.push_str("  \n");
    }
    mir.push_str("...\n");
    Ok(Exported { mir })
}

/// Register slots named by an operand text (`$vgpr4_vgpr5`, `$vgpr1(tied-def 0)`, `$vcc`).
fn operand_slots(text: &str) -> Vec<Slot> {
    let t = text.split('(').next().unwrap_or(text);
    if !t.starts_with('$') {
        return Vec::new();
    }
    slots_of(t)
}

/// Add `killed` to the last use of a register and `dead` to defs nobody reads, walking the block backwards
/// from its live-out set. Whole-operand granularity only: a tuple is `killed`/`dead` when every dword is.
fn annotate_flags(lines: &mut [String], mut live: BTreeSet<Slot>) {
    use super::mir_roundtrip::parse_inst_line;
    for line in lines.iter_mut().rev() {
        let Some(mut inst) = parse_inst_line(line) else { continue };
        if inst.opcode == "BUNDLE" || inst.raw.is_some() {
            continue;
        }
        // operands: (is_def, index into defs or ops)
        let mut defs: Vec<(bool, usize, Vec<Slot>)> = Vec::new();
        let mut uses: Vec<(usize, Vec<Slot>)> = Vec::new();
        for (i, o) in inst.defs.iter().enumerate() {
            defs.push((true, i, operand_slots(&o.text)));
        }
        for (i, o) in inst.ops.iter().enumerate() {
            let sl = operand_slots(&o.text);
            if o.flags.iter().any(|f| f == "implicit-def" || f == "def") {
                defs.push((false, i, sl));
            } else {
                uses.push((i, sl));
            }
        }
        let defined: BTreeSet<Slot> = defs.iter().flat_map(|d| d.2.iter().copied()).collect();
        for (is_def_list, i, sl) in &defs {
            if sl.is_empty() || sl.iter().any(|s| live.contains(s)) {
                continue;
            }
            let o = if *is_def_list { &mut inst.defs[*i] } else { &mut inst.ops[*i] };
            if !o.flags.iter().any(|f| f == "dead") {
                o.flags.push("dead".to_string());
            }
        }
        // uses: only the last operand touching a slot may carry the kill
        let mut seen: BTreeSet<Slot> = BTreeSet::new();
        for (i, sl) in uses.iter().rev() {
            if sl.is_empty() {
                continue;
            }
            let free = sl.iter().all(|s| !seen.contains(s));
            let dies = sl.iter().all(|s| !live.contains(s) || defined.contains(s));
            seen.extend(sl.iter().copied());
            if free && dies && !inst.ops[*i].flags.iter().any(|f| f == "killed") {
                inst.ops[*i].flags.push("killed".to_string());
            }
        }
        // liveness before this instruction
        for s in &defined {
            live.remove(s);
        }
        for (_, sl) in &uses {
            live.extend(sl.iter().copied());
        }
        *line = inst.print().join("\n");
    }
}

/// Conformance helper (`--level=reflag`): recompute kill/dead flags for the register slots we track over the body
/// lines of one function of a hand-written MIR test, exactly as the exporter does for our kernels. Blocks containing
/// bundles are left alone. Returns the rewritten lines.
pub fn reflag_body(body: &[String]) -> Vec<String> {
    use super::mir_roundtrip::parse_inst_line;
    let mut out: Vec<String> = body.to_vec();
    // block boundaries
    let mut starts: Vec<usize> = Vec::new();
    for (i, l) in body.iter().enumerate() {
        if l.trim_start().starts_with("bb.") && l.trim_end().ends_with(':') {
            starts.push(i);
        }
    }
    if starts.is_empty() {
        return out;
    }
    let nb = starts.len();
    let end_of = |b: usize| if b + 1 < nb { starts[b + 1] } else { body.len() };
    let is_inst = |l: &str| parse_inst_line(l).is_some();
    let mut inst_idx: Vec<Vec<usize>> = vec![Vec::new(); nb];
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); nb];
    let mut has_bundle = false;
    let id_of = |l: &str| -> Option<usize> {
        let t = l.trim_start().strip_prefix("bb.")?;
        t.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
    };
    let ids: Vec<usize> = starts.iter().map(|&s| id_of(&body[s]).unwrap_or(0)).collect();
    let pos_of = |id: usize| ids.iter().position(|&x| x == id);
    for b in 0..nb {
        let mut explicit: Option<Vec<usize>> = None;
        let mut last_op = String::new();
        for i in starts[b] + 1..end_of(b) {
            let l = &body[i];
            if l.trim_start().starts_with("successors:") {
                let v: Vec<usize> = l.split("%bb.").skip(1).filter_map(|x| x.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()).collect();
                explicit = Some(v);
                continue;
            }
            if l.trim_start().starts_with("BUNDLE") {
                has_bundle = true;
            }
            if is_inst(l) {
                inst_idx[b].push(i);
                last_op = parse_inst_line(l).map(|x| x.opcode).unwrap_or_default();
            }
        }
        let mut s: Vec<usize> = match explicit {
            Some(v) => v.into_iter().filter_map(pos_of).collect(),
            None => {
                let mut v: Vec<usize> = Vec::new();
                for &i in &inst_idx[b] {
                    for t in body[i].split("%bb.").skip(1) {
                        if let Some(n) = t.split(|c: char| !c.is_ascii_digit()).next().and_then(|x| x.parse::<usize>().ok()) {
                            if let Some(p) = pos_of(n) {
                                v.push(p);
                            }
                        }
                    }
                }
                if b + 1 < nb && !matches!(last_op.as_str(), "S_BRANCH" | "S_ENDPGM" | "SI_RETURN_TO_EPILOG" | "S_SETPC_B64_return") {
                    v.push(b + 1);
                }
                v
            }
        };
        s.sort();
        s.dedup();
        succs[b] = s;
    }
    if has_bundle {
        return out;
    }
    // per-block gen/kill over our slots
    let mut gen_: Vec<BTreeSet<Slot>> = vec![BTreeSet::new(); nb];
    let mut kill: Vec<BTreeSet<Slot>> = vec![BTreeSet::new(); nb];
    for b in 0..nb {
        for &i in &inst_idx[b] {
            let (d, u) = defs_uses(&body[i]);
            for x in u {
                if !kill[b].contains(&x) {
                    gen_[b].insert(x);
                }
            }
            kill[b].extend(d);
        }
    }
    let mut live_in = gen_.clone();
    loop {
        let mut changed = false;
        for b in (0..nb).rev() {
            let mut new_in = gen_[b].clone();
            for &s in &succs[b] {
                for x in &live_in[s] {
                    if !kill[b].contains(x) {
                        new_in.insert(*x);
                    }
                }
            }
            if new_in != live_in[b] {
                live_in[b] = new_in;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for b in 0..nb {
        let mut live: BTreeSet<Slot> = BTreeSet::new();
        for &s in &succs[b] {
            live.extend(live_in[s].iter().copied());
        }
        let mut lines: Vec<String> = inst_idx[b].iter().map(|&i| body[i].clone()).collect();
        annotate_flags(&mut lines, live);
        for (k, &i) in inst_idx[b].iter().enumerate() {
            out[i] = lines[k].clone();
        }
    }
    out
}

fn livein_text(set: &BTreeSet<Slot>) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut it = set.iter().copied().peekable();
    while let Some(s) = it.next() {
        // Pair aligned adjacent SGPRs the way the ABI names them ($sgpr0_sgpr1).
        if s >= SGPR_BASE && s < VCC && s < AGPR_BASE && (s - SGPR_BASE) % 2 == 0 && it.peek() == Some(&(s + 1)) {
            out.push(format!("$sgpr{}_sgpr{}", s - SGPR_BASE, s + 1 - SGPR_BASE));
            it.next();
        } else {
            out.push(slot_name(s));
        }
    }
    out.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defs_uses_parse() {
        let (d, u) = defs_uses("$vgpr2, $vcc = V_ADD_CO_U32_e64 $vgpr4, 5, 0, implicit $exec");
        assert_eq!(d, vec![2, VCC]);
        assert_eq!(u, vec![4]);
        let (d, u) = defs_uses("$vgpr1 = V_CVT_PK_FP8_F32_e64 0, $vgpr2, 0, $vgpr3, $vgpr1(tied-def 0), 0, implicit $mode, implicit $exec");
        assert_eq!(d, vec![1]);
        assert_eq!(u, vec![2, 3, 1]);
        let (d, u) = defs_uses("V_CMP_LT_I32_e32 $vgpr2, $vgpr3, implicit-def $vcc, implicit $exec");
        assert_eq!(d, vec![VCC]);
        assert_eq!(u, vec![2, 3]);
        let (d, u) = defs_uses("$vgpr0_vgpr1_vgpr2_vgpr3 = DS_READ_B128_gfx9 $vgpr5, 16, 0, implicit $exec :: (load (s128), addrspace 3)");
        assert_eq!(d, vec![0, 1, 2, 3]);
        assert_eq!(u, vec![5]);
    }
}