//! Generic MIR text importer/printer for conformance testing (LLVM's own lit tests).
//!
//! `roundtrip_text` rewrites every `body: |` section of an `.mir` (lit test) file: instruction
//! lines are parsed into a structured form (defs, instruction flags, opcode, operands with
//! register flags, opaque suffix such as memory operands / debug locations, BUNDLE children)
//! and printed back from that structure. Everything that is not an instruction (block headers,
//! `successors:`, `liveins:`, comments, IR sections, RUN/CHECK lines) is passed through verbatim.
//! Lines the parser does not understand are kept verbatim and counted as `opaque`.
//!
//! `table_report` checks our opcode table against the instructions found in such files: for each
//! instruction whose opcode has a single-line table row it matches the row template against the
//! instruction (register flags and instruction flags ignored) and classifies the result.

use std::collections::{BTreeMap, HashMap};

use super::llvm20::Table;

const REG_FLAGS: &[&str] = &[
    "implicit", "implicit-def", "def", "killed", "dead", "undef", "internal", "renamable", "early-clobber", "debug-use",
];
const INST_FLAGS: &[&str] = &[
    "nofpexcept", "nuw", "nsw", "exact", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc", "nneg", "disjoint",
    "samesign", "frame-setup", "frame-destroy", "unpredictable", "noconvergent", "nusw", "inbounds",
];
const SUFFIX_MARKS: &[&str] = &[
    " :: ", ", debug-location", ", pre-instr-symbol", ", post-instr-symbol", ", heap-alloc-marker", ", pcsections",
    ", cfi-type", ", mmra", ", debug-instr-number",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operand {
    pub flags: Vec<String>,
    pub text: String,
}

impl Operand {
    fn print(&self) -> String {
        if self.flags.is_empty() {
            self.text.clone()
        } else {
            format!("{} {}", self.flags.join(" "), self.text)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Inst {
    pub indent: String,
    pub defs: Vec<Operand>,
    pub iflags: Vec<String>,
    pub opcode: String,
    pub ops: Vec<Operand>,
    pub suffix: String,
    pub children: Vec<Inst>,
    pub raw: Option<String>,
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub files: usize,
    pub bodies: usize,
    pub insts: usize,
    pub parsed: usize,
    pub opaque: usize,
    pub bundles: usize,
    pub changed_lines: usize,
}

impl Stats {
    pub fn add(&mut self, o: &Stats) {
        self.files += o.files;
        self.bodies += o.bodies;
        self.insts += o.insts;
        self.parsed += o.parsed;
        self.opaque += o.opaque;
        self.bundles += o.bundles;
        self.changed_lines += o.changed_lines;
    }
}

/// Split at top-level occurrences of `sep` (brackets, quotes respected).
fn split_top<'a>(s: &'a str, sep: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let b = s.as_bytes();
    let (mut depth, mut q, mut start, mut i) = (0i32, 0u8, 0usize, 0usize);
    while i < b.len() {
        let c = b[i];
        if q != 0 {
            if c == b'\\' {
                i += 1;
            } else if c == q {
                q = 0;
            }
        } else if c == b'"' || c == b'\'' {
            q = c;
        } else if matches!(c, b'(' | b'[' | b'{' | b'<') {
            depth += 1;
        } else if matches!(c, b')' | b']' | b'}' | b'>') {
            depth -= 1;
        } else if depth == 0 && s[i..].starts_with(sep) {
            out.push(&s[start..i]);
            i += sep.len();
            start = i;
            continue;
        }
        i += 1;
    }
    out.push(&s[start..]);
    out
}

fn parse_operand(t: &str) -> Operand {
    let t = t.trim();
    let words: Vec<&str> = t.split(' ').collect();
    let mut k = 0;
    while k + 1 < words.len() && REG_FLAGS.contains(&words[k]) {
        k += 1;
    }
    let rest = words[k..].join(" ");
    if k > 0 && !(rest.starts_with('$') || rest.starts_with('%')) {
        return Operand { flags: vec![], text: t.to_string() };
    }
    Operand { flags: words[..k].iter().map(|s| s.to_string()).collect(), text: rest }
}

pub fn parse_inst_line(line: &str) -> Option<Inst> {
    let indent: String = line.chars().take_while(|c| *c == ' ').collect();
    let body = line[indent.len()..].trim_end();
    if body.is_empty() || body.starts_with(';') || body.starts_with("bb.") || body.starts_with("liveins:")
        || body.starts_with("successors:") || body.starts_with('}') || body.starts_with("INLINEASM")
    {
        return None;
    }
    // Cut the verbatim suffix.
    let mut cut = body.len();
    for m in SUFFIX_MARKS {
        if let Some(p) = body.find(m) {
            cut = cut.min(p);
        }
    }
    let (main, suffix) = body.split_at(cut);
    // defs = rest
    let parts = split_top(main, " = ");
    let (lhs, rhs) = match parts.len() {
        1 => ("", parts[0]),
        2 => (parts[0], parts[1]),
        _ => return None,
    };
    let defs: Vec<Operand> = if lhs.is_empty() { vec![] } else { split_top(lhs, ", ").into_iter().map(parse_operand).collect() };
    let mut words: Vec<&str> = rhs.splitn(usize::MAX, ' ').collect();
    // leading instruction flags, then the opcode
    let mut iflags = Vec::new();
    while !words.is_empty() && INST_FLAGS.contains(&words[0]) {
        iflags.push(words.remove(0).to_string());
    }
    if words.is_empty() {
        return None;
    }
    let opcode = words[0].to_string();
    if !opcode.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.') || !opcode.chars().next().unwrap().is_ascii_uppercase() {
        return None;
    }
    let opstr = words[1..].join(" ");
    let ops: Vec<Operand> = if opstr.trim().is_empty() { vec![] } else { split_top(&opstr, ", ").into_iter().map(parse_operand).collect() };
    // Reject lines whose operand text would not survive (stray braces etc.).
    Some(Inst { indent, defs, iflags, opcode, ops, suffix: suffix.to_string(), children: vec![], raw: None })
}

impl Inst {
    pub fn print(&self) -> Vec<String> {
        if let Some(r) = &self.raw {
            return vec![r.clone()];
        }
        let mut s = self.indent.clone();
        if !self.defs.is_empty() {
            s.push_str(&self.defs.iter().map(|o| o.print()).collect::<Vec<_>>().join(", "));
            s.push_str(" = ");
        }
        for f in &self.iflags {
            s.push_str(f);
            s.push(' ');
        }
        s.push_str(&self.opcode);
        if !self.ops.is_empty() {
            s.push(' ');
            s.push_str(&self.ops.iter().map(|o| o.print()).collect::<Vec<_>>().join(", "));
        }
        s.push_str(&self.suffix);
        if self.opcode == "BUNDLE" && !self.children.is_empty() {
            s.push_str(" {");
            let mut out = vec![s];
            for c in &self.children {
                out.extend(c.print());
            }
            out.push(format!("{}}}", self.indent));
            return out;
        }
        vec![s]
    }
}

/// Parse lines[i..] as instruction (+ bundle children). Returns (inst, next index).
fn parse_at(lines: &[&str], i: usize, st: &mut Stats) -> (Inst, usize) {
    let line = lines[i];
    let is_bundle = line.trim_start().starts_with("BUNDLE") && line.trim_end().ends_with('{');
    if is_bundle {
        st.bundles += 1;
        let header = line.trim_end().trim_end_matches('{').trim_end();
        let mut inst = parse_inst_line(header).unwrap_or_else(|| Inst {
            indent: String::new(), defs: vec![], iflags: vec![], opcode: String::new(), ops: vec![], suffix: String::new(), children: vec![], raw: Some(header.to_string()),
        });
        let mut j = i + 1;
        let mut children = Vec::new();
        while j < lines.len() && lines[j].trim() != "}" {
            let (c, nj) = parse_at(lines, j, st);
            children.push(c);
            j = nj;
        }
        // closing brace line
        if inst.raw.is_some() {
            // keep header+children+brace verbatim
            let mut all = vec![format!("{header} {{")];
            for c in &children {
                all.extend(c.print());
            }
            all.push(lines.get(j).map(|s| s.to_string()).unwrap_or_default());
            return (Inst { raw: Some(all.join("\n")), ..inst }, j + 1);
        }
        inst.children = children;
        return (inst, j + 1);
    }
    st.insts += 1;
    match parse_inst_line(line) {
        Some(inst) => {
            st.parsed += 1;
            (inst, i + 1)
        }
        None => {
            let t = line.trim_start();
            let looks_inst = !(t.is_empty() || t.starts_with(';') || t.starts_with("bb.") || t.starts_with("liveins:") || t.starts_with("successors:"));
            if looks_inst {
                st.opaque += 1;
            } else {
                st.insts -= 1;
            }
            (Inst { indent: String::new(), defs: vec![], iflags: vec![], opcode: String::new(), ops: vec![], suffix: String::new(), children: vec![], raw: Some(line.to_string()) }, i + 1)
        }
    }
}

/// How much information the printed MIR keeps.
/// * `Identity`: everything (structure round trip only).
/// * `Flags`: drop register flags our exporter never writes (`killed dead renamable`; `undef` and `internal` are kept: they state intent / bundle membership, which our exporter never produces and which the verifier needs).
/// * `All`: additionally drop instruction flags, memory operands and debug locations, i.e. the
///   information level of MIR as our exporter writes it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    Identity,
    Flags,
    /// `Flags` followed by our own kill/dead annotation (`llvm_mir::reflag_body`).
    Reflag,
    All,
}

fn degrade(i: &mut Inst, lv: Level) {
    let lv = if lv == Level::Reflag { Level::Flags } else { lv };
    if lv == Level::Identity || i.raw.is_some() {
        return;
    }
    let strip = |o: &mut Operand| o.flags.retain(|f| !matches!(f.as_str(), "killed" | "dead" | "renamable"));
    i.defs.iter_mut().for_each(strip);
    i.ops.iter_mut().for_each(strip);
    if lv == Level::All {
        i.iflags.clear();
        i.suffix.clear();
    }
    for c in i.children.iter_mut() {
        degrade(c, lv);
    }
}

/// Rewrite the bodies of a whole test file.
pub fn roundtrip_text(text: &str) -> (String, Stats) {
    roundtrip_text_level(text, Level::Identity)
}

pub fn roundtrip_text_level(text: &str, lv: Level) -> (String, Stats) {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut out: Vec<String> = Vec::new();
    let mut st = Stats { files: 1, ..Default::default() };
    let mut i = 0;
    while i < lines.len() {
        let l = lines[i];
        out.push(l.to_string());
        i += 1;
        let t = l.trim_end();
        if t.starts_with("body:") && t.trim_start_matches("body:").trim() == "|" {
            st.bodies += 1;
            let start = i;
            while i < lines.len() && (lines[i].is_empty() || lines[i].starts_with(' ')) {
                i += 1;
            }
            let body = &lines[start..i];
            let body_start = out.len();
            let mut j = 0;
            while j < body.len() {
                let (mut inst, nj) = parse_at(body, j, &mut st);
                let orig = body[j..nj].join("\n");
                degrade(&mut inst, lv);
                let printed = inst.print().join("\n");
                if printed != orig {
                    st.changed_lines += 1;
                }
                out.push(printed);
                j = nj;
            }
            if lv == Level::Reflag {
                let blk: Vec<String> = out[body_start..].iter().flat_map(|s| s.split('\n').map(|x| x.to_string()).collect::<Vec<_>>()).collect();
                out.truncate(body_start);
                out.extend(super::llvm_mir::reflag_body(&blk));
            }
        }
    }
    (out.join("\n"), st)
}

// ---------------------------------------------------------------------------------------------
// Table report

#[derive(Default)]
pub struct TableReport {
    pub total: usize,
    pub matched: usize,
    pub shape_mismatch: usize,
    pub implicit_differs: usize,
    pub implicit_examples: BTreeMap<String, String>,
    pub unknown_opcode: usize,
    pub unknown_by_opcode: BTreeMap<String, usize>,
    pub mismatch_examples: BTreeMap<String, String>,
    pub matched_by_opcode: BTreeMap<String, usize>,
}

fn strip_reg_flags(s: &str) -> String {
    // drop register flags that are irrelevant for the shape: "killed $vgpr0" -> "$vgpr0"
    let mut out = Vec::new();
    for w in s.split(' ') {
        if matches!(w, "killed" | "dead" | "undef" | "renamable" | "internal" | "early-clobber" | "debug-use") {
            continue;
        }
        out.push(w);
    }
    out.join(" ")
}

/// Does `text` (an instruction without defs prefix flags) match a template with {tokens}?
fn matches_template(tpl: &str, text: &str) -> bool {
    // Literal segments between tokens must appear in order; a token consumes up to the next literal.
    let mut segs: Vec<(bool, String)> = Vec::new(); // (is_token, text)
    let mut rest = tpl;
    while let Some(i) = rest.find('{') {
        if i > 0 {
            segs.push((false, rest[..i].to_string()));
        }
        let j = rest[i..].find('}').map(|x| x + i).unwrap_or(rest.len() - 1);
        segs.push((true, rest[i + 1..j].to_string()));
        rest = &rest[j + 1..];
    }
    if !rest.is_empty() {
        segs.push((false, rest.to_string()));
    }
    let mut pos = 0usize;
    let mut k = 0;
    while k < segs.len() {
        let (is_tok, s) = &segs[k];
        if !*is_tok {
            if !text[pos..].starts_with(s.as_str()) {
                return false;
            }
            pos += s.len();
        } else {
            // token value: register / immediate / %bb: runs to next literal (or end)
            let end = match segs.get(k + 1) {
                Some((false, lit)) => match text[pos..].find(lit.as_str()) {
                    Some(p) => pos + p,
                    None => return false,
                },
                _ => text.len(),
            };
            let v = &text[pos..end];
            if v.is_empty() || v.contains(", ") {
                return false;
            }
            pos = end;
        }
        k += 1;
    }
    pos == text.len()
}

/// Template normalisation for the report: drop memory operands and `(tied-def N)` annotations
/// (tests omit them) and turn literal numeric operands (modifier/cpol/cbsz values) into wildcards.
fn normalise_template(x: &str) -> String {
    let mut s = x.split(" :: ").next().unwrap_or(x).to_string();
    while let Some(i) = s.find("(tied-def ") {
        let j = s[i..].find(')').map(|k| k + i).unwrap_or(s.len() - 1);
        s.replace_range(i..=j, "");
    }
    let (pre, rest) = match s.find(" = ") {
        Some(p) => (s[..p + 3].to_string(), s[p + 3..].to_string()),
        None => (String::new(), s.clone()),
    };
    // rest = "OPCODE op, op, ..."
    let (op, ops) = match rest.find(' ') {
        Some(p) => (rest[..p].to_string(), rest[p + 1..].to_string()),
        None => (rest.clone(), String::new()),
    };
    let ops: Vec<String> = ops
        .split(", ")
        .filter(|z| !z.is_empty())
        .map(|z| if z.parse::<i64>().is_ok() { "{n}".to_string() } else { z.to_string() })
        .collect();
    if ops.is_empty() {
        format!("{pre}{op}")
    } else {
        format!("{pre}{op} {}", ops.join(", "))
    }
}

fn opcode_of_row(line: &str) -> Option<String> {
    let rhs = line.split(" = ").last()?;
    let mut w = rhs.split(' ').filter(|x| !INST_FLAGS.contains(x));
    let o = w.next()?;
    if o.chars().next()?.is_ascii_uppercase() {
        Some(o.to_string())
    } else {
        None
    }
}

pub fn table_report(files: &[(String, String)], t: &Table) -> TableReport {
    // opcode -> single-line templates
    let mut by_op: HashMap<String, Vec<String>> = HashMap::new();
    for row in t.rows.values() {
        if row.text.len() != 1 {
            continue;
        }
        if let Some(o) = opcode_of_row(&row.text[0]) {
            by_op.entry(o).or_default().push(row.text[0].clone());
        }
    }
    let mut rep = TableReport::default();
    for (_name, text) in files {
        let mut in_body = false;
        for l in text.split('\n') {
            if l.trim_end().starts_with("body:") && l.trim_start_matches("body:").trim() == "|" {
                in_body = true;
                continue;
            }
            if in_body && !(l.is_empty() || l.starts_with(' ')) {
                in_body = false;
            }
            if !in_body {
                continue;
            }
            let Some(inst) = parse_inst_line(l) else { continue };
            rep.total += 1;
            let Some(tpls) = by_op.get(&inst.opcode) else {
                rep.unknown_opcode += 1;
                *rep.unknown_by_opcode.entry(inst.opcode.clone()).or_default() += 1;
                continue;
            };
            // explicit part (defs + non-implicit operands) and the sorted implicit set
            let split = |ops: &[Operand]| -> (Vec<String>, Vec<String>) {
                let (mut ex, mut im) = (Vec::new(), Vec::new());
                for o in ops {
                    let p = strip_reg_flags(&o.print());
                    if o.flags.iter().any(|f| f.starts_with("implicit")) {
                        im.push(p);
                    } else {
                        ex.push(p);
                    }
                }
                im.sort();
                (ex, im)
            };
            let defs_ex: Vec<String> = inst.defs.iter().filter(|o| !o.flags.iter().any(|f| f.starts_with("implicit"))).map(|o| strip_reg_flags(&o.print())).collect();
            let defs_im: Vec<String> = inst.defs.iter().filter(|o| o.flags.iter().any(|f| f.starts_with("implicit"))).map(|o| strip_reg_flags(&o.print())).collect();
            let (ex, mut im) = split(&inst.ops);
            im.extend(defs_im);
            im.sort();
            let mut s = String::new();
            if !defs_ex.is_empty() {
                s.push_str(&defs_ex.join(", "));
                s.push_str(" = ");
            }
            s.push_str(&inst.opcode);
            if !ex.is_empty() {
                s.push(' ');
                s.push_str(&ex.join(", "));
            }
            let mut best = 0; // 0 = no match, 1 = explicit operands match but implicit set differs, 2 = exact
            for tp in tpls {
                // template: strip instruction flags, split implicit operands off
                let mut x = tp.clone();
                for f in INST_FLAGS {
                    x = x.replace(&format!("{f} "), "");
                }
                x = normalise_template(&x);
                let (head, tail) = match x.find(", implicit") {
                    Some(p) => (x[..p].to_string(), x[p + 2..].to_string()),
                    None => (x.clone(), String::new()),
                };
                // "implicit-def $vcc" inside the explicit part of the template (V_ADD_CO ... implicit-def $vcc): handled by the tail
                if matches_template(&head, &s) {
                    let mut tim: Vec<String> = tail.split(", ").filter(|z| !z.is_empty()).map(|z| z.to_string()).collect();
                    tim.sort();
                    let mut iim: Vec<String> = im.clone();
                    // instruction implicit operands print as "implicit $exec"; compare as strings
                    iim.iter_mut().for_each(|z| *z = z.trim().to_string());
                    best = best.max(if tim == iim { 2 } else { 1 });
                }
            }
            match best {
                2 => {
                    rep.matched += 1;
                    *rep.matched_by_opcode.entry(inst.opcode.clone()).or_default() += 1;
                }
                1 => {
                    rep.implicit_differs += 1;
                    rep.implicit_examples.entry(inst.opcode.clone()).or_insert_with(|| format!("{} | implicit: {}", s, im.join(", ")));
                }
                _ => {
                    rep.shape_mismatch += 1;
                    rep.mismatch_examples.entry(inst.opcode.clone()).or_insert_with(|| s.clone());
                }
            }
        }
    }
    rep
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_is_identity_on_typical_lines() {
        let src = "name: x\nbody:             |\n  bb.0:\n    liveins: $vgpr0\n  \n    renamable $vgpr1 = nofpexcept V_ADD_F32_e32 1065353216, killed $vgpr1, implicit $mode, implicit $exec\n    $sgpr0_sgpr1 = S_LOAD_DWORDX2_IMM killed renamable $sgpr4_sgpr5, 0, 0 :: (dereferenceable invariant load (s64), align 16, addrspace 4)\n    S_ENDPGM 0\n\n...\n";
        let (out, st) = roundtrip_text(src);
        assert_eq!(out, src);
        assert_eq!(st.opaque, 0);
        assert_eq!(st.parsed, 3);
    }

    #[test]
    fn bundle_roundtrip() {
        let src = "body:             |\n  bb.0:\n    BUNDLE implicit-def $vgpr0, implicit $vgpr1 {\n      $vgpr0 = V_MOV_B32_e32 $vgpr1, implicit $exec\n    }\n";
        let (out, _) = roundtrip_text(src);
        assert_eq!(out, src);
    }

    #[test]
    fn template_matching() {
        assert!(matches_template("{d} = V_ADD_U32_e32 {s0}, {s1}, implicit $exec", "$vgpr1 = V_ADD_U32_e32 $vgpr2, 5, implicit $exec"));
        assert!(!matches_template("{d} = V_ADD_U32_e32 {s0}, {s1}, implicit $exec", "$vgpr1 = V_ADD_U32_e32 $vgpr2, $vgpr3"));
    }
}