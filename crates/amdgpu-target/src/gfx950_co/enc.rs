//! In-process gfx950 instruction encoder: one line of the compiler's own assembly text -> machine code bytes.
//!
//! The compiler prints a small, regular assembly dialect (one instruction per line, explicit operand forms); this module encodes exactly that dialect, byte for byte as `llvm-mc -mcpu=gfx950` does
//! (the oracle is `tests/co_oracle.rs` and `amdc co-check`). The opcode and format of every mnemonic are in `tables.rs` (generated from the oracle by `tools/gen_enc_tables.py`); the field layouts per
//! format are below. Unknown mnemonics or operand shapes are errors (the caller falls back to the external assembler).

use super::tables::OPS;

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// register range: kind 'v' | 's' | 'a', first, count; neg / abs source modifiers
    Reg { k: char, n: u32, cnt: u32, neg: bool, abs: bool },
    /// vcc, vcc_lo, ... exec, m0, off, scc ...
    Sp(String),
    Int(i64),
    Flt(f32),
    Label(String),
    /// anything else (named counters such as `lgkmcnt(0)`)
    Raw(String),
}

struct Inst {
    mn: String,
    ops: Vec<Op>,
    /// key:value modifiers and flag words
    mods: Vec<(String, String)>,
    raw: Vec<String>,
}

fn split_top(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0i32;
    for c in s.chars() {
        match c {
            '[' | '(' => {
                depth += 1;
                cur.push(c)
            }
            ']' | ')' => {
                depth -= 1;
                cur.push(c)
            }
            ' ' | '\t' if depth == 0 => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn parse_reg(t: &str) -> Option<Op> {
    let (neg, t) = match t.strip_prefix('-') {
        Some(r) if !r.is_empty() && (r.starts_with(['v', 's', 'a', '|'])) => (true, r),
        _ => (false, t),
    };
    let (abs, t) = if t.starts_with('|') && t.ends_with('|') && t.len() > 2 { (true, &t[1..t.len() - 1]) } else { (false, t) };
    let k = t.chars().next()?;
    if !matches!(k, 'v' | 's' | 'a') {
        return None;
    }
    let rest = &t[1..];
    if let Some(r) = rest.strip_prefix('[') {
        let r = r.strip_suffix(']')?;
        let (a, b) = r.split_once(':')?;
        let (a, b): (u32, u32) = (a.parse().ok()?, b.parse().ok()?);
        return Some(Op::Reg { k, n: a, cnt: b - a + 1, neg, abs });
    }
    let n: u32 = rest.parse().ok()?;
    Some(Op::Reg { k, n, cnt: 1, neg, abs })
}

fn parse_num(t: &str) -> Option<i64> {
    let (neg, u) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t),
    };
    let v = if let Some(h) = u.strip_prefix("0x").or_else(|| u.strip_prefix("0X")) { i64::from_str_radix(h, 16).ok()? } else { u.parse::<i64>().ok()? };
    Some(if neg { -v } else { v })
}

fn parse_operand(t: &str) -> Op {
    if let Some(r) = parse_reg(t) {
        return r;
    }
    if matches!(t, "vcc" | "vcc_lo" | "vcc_hi" | "m0" | "exec" | "exec_lo" | "exec_hi" | "off" | "scc" | "null") {
        return Op::Sp(t.to_string());
    }
    if let Some(v) = parse_num(t) {
        return Op::Int(v);
    }
    if t.contains('.') || t.contains('e') && t.chars().next().map_or(false, |c| c.is_ascii_digit() || c == '-') {
        if let Ok(f) = t.parse::<f32>() {
            return Op::Flt(f);
        }
    }
    if t.contains('(') {
        return Op::Raw(t.to_string());
    }
    Op::Label(t.to_string())
}

fn parse_inst(line: &str) -> Result<Inst, String> {
    let toks = split_top(line);
    let mn = toks.first().ok_or("empty line")?.clone();
    let mut ops = Vec::new();
    let mut mods = Vec::new();
    let mut raw = Vec::new();
    for t in &toks[1..] {
        let t = t.trim_end_matches(',');
        if t.is_empty() {
            continue;
        }
        if let Some((k, v)) = t.split_once(':') {
            if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !k.chars().next().unwrap().is_ascii_digit() {
                mods.push((k.to_string(), v.to_string()));
                continue;
            }
        }
        if matches!(t, "offen" | "idxen" | "lds" | "nt" | "sc0" | "sc1" | "glc" | "slc" | "gds" | "clamp" | "tfe" | "dlc" | "bound_ctrl" | "row_mirror" | "row_half_mirror") {
            mods.push((t.to_string(), String::new()));
            continue;
        }
        if t.starts_with("vmcnt(") || t.starts_with("lgkmcnt(") || t.starts_with("expcnt(") || t == "&" {
            raw.push(t.to_string());
            continue;
        }
        ops.push(parse_operand(t));
    }
    Ok(Inst { mn, ops, mods, raw })
}

fn md<'a>(i: &'a Inst, k: &str) -> Option<&'a str> {
    i.mods.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
}
fn has(i: &Inst, k: &str) -> bool {
    i.mods.iter().any(|(a, _)| a == k)
}

/// The 32-bit literal inline-constant codes (isInlinableLiteral32): float bit patterns.
fn inline_fp32(bits: u32) -> Option<u16> {
    Some(match bits {
        0x3f000000 => 240,
        0xbf000000 => 241,
        0x3f800000 => 242,
        0xbf800000 => 243,
        0x40000000 => 244,
        0xc0000000 => 245,
        0x40800000 => 246,
        0xc0800000 => 247,
        0x3e22f983 => 248,
        _ => return None,
    })
}

/// Source operand: (9-bit code, optional literal dword). `scalar_only`: SOP-style 8-bit sources.
fn src(op: &Op) -> Result<(u16, Option<u32>), String> {
    Ok(match op {
        Op::Reg { k: 'v', n, .. } | Op::Reg { k: 'a', n, .. } => (256 + *n as u16, None),
        Op::Reg { k: 's', n, .. } => (*n as u16, None),
        Op::Sp(s) => (
            match s.as_str() {
                "vcc" | "vcc_lo" => 106,
                "vcc_hi" => 107,
                "m0" => 124,
                "exec" | "exec_lo" => 126,
                "exec_hi" => 127,
                "scc" => 253,
                _ => return Err(format!("special operand {s} as a source")),
            },
            None,
        ),
        Op::Int(v) => {
            if (0..=64).contains(v) {
                (128 + *v as u16, None)
            } else if (-16..0).contains(v) {
                (192 + (-*v) as u16, None)
            } else {
                let bits = *v as u32;
                if let Some(c) = inline_fp32(bits) {
                    (c, None)
                } else {
                    (255, Some(bits))
                }
            }
        }
        Op::Flt(f) => {
            let c = match *f {
                x if x == 0.0 => 128,
                x if x == 0.5 => 240,
                x if x == -0.5 => 241,
                x if x == 1.0 => 242,
                x if x == -1.0 => 243,
                x if x == 2.0 => 244,
                x if x == -2.0 => 245,
                x if x == 4.0 => 246,
                x if x == -4.0 => 247,
                x if (x - 0.15915494).abs() < 1e-9 => 248,
                _ => return Ok((255, Some(f.to_bits()))),
            };
            (c, None)
        }
        other => return Err(format!("operand {other:?} as a source")),
    })
}

fn reg_n(op: &Op) -> Result<u32, String> {
    match op {
        Op::Reg { n, .. } => Ok(*n),
        Op::Sp(s) if s == "vcc" || s == "vcc_lo" => Ok(106),
        Op::Sp(s) if s == "m0" => Ok(124),
        Op::Sp(s) if s == "exec" || s == "exec_lo" => Ok(126),
        o => Err(format!("{o:?} where a register is expected")),
    }
}

fn is_agpr(op: &Op) -> bool {
    matches!(op, Op::Reg { k: 'a', .. })
}

fn d32(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn lookup(mn: &str) -> Option<(&'static str, u32)> {
    OPS.binary_search_by(|e| e.0.cmp(mn)).ok().map(|i| (OPS[i].1, OPS[i].2 as u32))
}

/// Encodes one instruction line. `pc` is the byte offset of the instruction and `label` resolves a label to its byte offset (branches only).
pub fn encode_line(line: &str, pc: usize, label: &dyn Fn(&str) -> Option<usize>) -> Result<Vec<u8>, String> {
    let i = parse_inst(line)?;
    let (fmt, op) = lookup(&i.mn).ok_or_else(|| format!("unknown mnemonic {}", i.mn))?;
    match fmt {
        "SOP2" | "SOP1" | "SOPC" => enc_sop(&i, fmt, op),
        "SOPP" => enc_sopp(&i, op, pc, label),
        "SMEM" => enc_smem(&i, op),
        "VOP1" | "VOP2" | "VOPC" => enc_vop12c(&i, fmt, op),
        "VOP3" => enc_vop3(&i, op),
        "DS" => enc_ds(&i, op),
        "MUBUF" => enc_mubuf(&i, op),
        "GLOBAL" => enc_global(&i, op),
        f => Err(format!("format {f} ({}) is not encoded in-process", i.mn)),
    }
}

fn enc_sop(i: &Inst, fmt: &str, op: u32) -> Result<Vec<u8>, String> {
    let mut lit = None;
    let mut s = |o: &Op| -> Result<u32, String> {
        let (c, l) = src(o)?;
        if c > 255 {
            return Err("vector source in a scalar instruction".into());
        }
        if l.is_some() {
            lit = l;
        }
        Ok(c as u32)
    };
    let d0 = match fmt {
        "SOP2" => {
            if i.ops.len() != 3 {
                return Err(format!("{}: expected 3 operands", i.mn));
            }
            let sd = reg_n(&i.ops[0])?;
            let a = s(&i.ops[1])?;
            let b = s(&i.ops[2])?;
            0x8000_0000 | op << 23 | sd << 16 | b << 8 | a
        }
        "SOP1" => {
            if i.ops.len() != 2 {
                return Err(format!("{}: expected 2 operands", i.mn));
            }
            let sd = reg_n(&i.ops[0])?;
            let a = s(&i.ops[1])?;
            0xBE80_0000 | sd << 16 | op << 8 | a
        }
        _ => {
            if i.ops.len() != 2 {
                return Err(format!("{}: expected 2 operands", i.mn));
            }
            let a = s(&i.ops[0])?;
            let b = s(&i.ops[1])?;
            0xBF00_0000 | op << 16 | b << 8 | a
        }
    };
    let mut v = d32(d0);
    if let Some(l) = lit {
        v.extend(d32(l));
    }
    Ok(v)
}

fn count_of(r: &str) -> Result<u32, String> {
    let inner = r.split_once('(').and_then(|(_, b)| b.strip_suffix(')')).ok_or("bad counter")?;
    inner.trim().parse::<u32>().map_err(|e| format!("{e}"))
}

fn enc_sopp(i: &Inst, op: u32, pc: usize, label: &dyn Fn(&str) -> Option<usize>) -> Result<Vec<u8>, String> {
    let simm: u32 = if i.mn == "s_waitcnt" {
        let (mut vm, mut ex, mut lg) = (63u32, 7u32, 15u32);
        for r in &i.raw {
            let c = count_of(r)?;
            if r.starts_with("vmcnt") {
                vm = c;
            } else if r.starts_with("expcnt") {
                ex = c;
            } else if r.starts_with("lgkmcnt") {
                lg = c;
            }
        }
        (vm & 15) | ex << 4 | lg << 8 | (vm >> 4) << 14
    } else if let Some(Op::Label(l)) = i.ops.first() {
        let t = label(l).ok_or_else(|| format!("unknown label {l}"))?;
        let d = (t as i64 - (pc as i64 + 4)) / 4;
        (d as i16) as u16 as u32
    } else if let Some(Op::Int(v)) = i.ops.first() {
        *v as u32 & 0xffff
    } else {
        0
    };
    Ok(d32(0xBF80_0000 | op << 16 | simm))
}

fn enc_smem(i: &Inst, op: u32) -> Result<Vec<u8>, String> {
    let sdata = reg_n(&i.ops[0])?;
    let mut d0 = 0xC000_0000 | op << 18 | sdata << 6;
    let d1;
    if i.mn == "s_memtime" {
        d1 = 0;
    } else {
        let base = reg_n(&i.ops[1])?;
        d0 |= base >> 1;
        match i.ops.get(2) {
            Some(Op::Int(v)) => {
                d0 |= 1 << 17;
                d1 = *v as u32 & 0x1f_ffff;
            }
            Some(Op::Reg { k: 's', n, .. }) => d1 = *n << 25 >> 25,
            _ => return Err("smem offset operand".into()),
        }
    }
    if has(i, "glc") {
        d0 |= 1 << 16;
    }
    let mut v = d32(d0);
    v.extend(d32(d1));
    Ok(v)
}

fn dpp_ctrl(s: &str) -> Result<u32, String> {
    Ok(if let Some(r) = s.strip_prefix('[') {
        let v: Vec<u32> = r.trim_end_matches(']').split(',').map(|x| x.trim().parse().unwrap_or(0)).collect();
        v[0] | v[1] << 2 | v[2] << 4 | v[3] << 6
    } else {
        return Err(format!("dpp control {s}"));
    })
}

fn enc_vop12c(i: &Inst, fmt: &str, op: u32) -> Result<Vec<u8>, String> {
    let dpp = i.mn.ends_with("_dpp");
    let sdwa = i.mn.ends_with("_sdwa");
    // operands without the implicit vcc tokens
    let mut o: Vec<&Op> = i.ops.iter().collect();
    let is_vcc = |x: &Op| matches!(x, Op::Sp(s) if s == "vcc" || s == "vcc_lo");
    let (dst, s0, s1): (Option<&Op>, &Op, Option<&Op>);
    match fmt {
        "VOP2" => {
            if o.len() == 4 && is_vcc(o[3]) {
                o.pop();
            }
            if o.len() != 3 {
                return Err(format!("{}: expected 3 operands", i.mn));
            }
            dst = Some(o[0]);
            s0 = o[1];
            s1 = Some(o[2]);
        }
        "VOPC" => {
            if o.len() == 3 && is_vcc(o[0]) {
                o.remove(0);
            }
            if o.len() != 2 {
                return Err(format!("{}: expected 2 operands after vcc", i.mn));
            }
            dst = None;
            s0 = o[0];
            s1 = Some(o[1]);
        }
        _ => {
            if o.len() != 2 {
                return Err(format!("{}: expected 2 operands", i.mn));
            }
            dst = Some(o[0]);
            s0 = o[1];
            s1 = None;
        }
    }
    let (mut c0, mut lit) = src(s0)?;
    if dpp {
        c0 = 0xFA;
        lit = None;
    }
    if sdwa {
        c0 = 0xF9;
        lit = None;
    }
    let vsrc1 = match s1 {
        Some(Op::Reg { k: 'v', n, .. }) => *n,
        Some(o) => return Err(format!("{}: second source {o:?} must be a VGPR in the 32-bit form", i.mn)),
        None => 0,
    };
    let vd = match dst {
        Some(d) => reg_n(d)?,
        None => 0,
    };
    let d0 = match fmt {
        "VOP2" => op << 25 | vd << 17 | vsrc1 << 9 | c0 as u32,
        "VOPC" => 0x7C00_0000 | op << 17 | vsrc1 << 9 | c0 as u32,
        _ => 0x7E00_0000 | vd << 17 | op << 9 | c0 as u32,
    };
    let mut v = d32(d0);
    if dpp {
        let ctrl = if let Some(q) = md(i, "quad_perm") {
            dpp_ctrl(q)?
        } else if let Some(r) = md(i, "row_bcast") {
            match r {
                "15" => 0x142,
                "31" => 0x143,
                _ => return Err("row_bcast".into()),
            }
        } else if has(i, "row_mirror") {
            0x140
        } else if has(i, "row_half_mirror") {
            0x141
        } else if let Some(x) = md(i, "row_shl") {
            0x100 + x.parse::<u32>().map_err(|e| e.to_string())?
        } else if let Some(x) = md(i, "row_shr") {
            0x110 + x.parse::<u32>().map_err(|e| e.to_string())?
        } else if let Some(x) = md(i, "row_ror") {
            0x120 + x.parse::<u32>().map_err(|e| e.to_string())?
        } else {
            return Err("dpp control".into());
        };
        let rm = md(i, "row_mask").map_or(Ok(0xf), |x| parse_num(x).ok_or("row_mask")).map_err(|e| e.to_string())? as u32;
        let bm = md(i, "bank_mask").map_or(Ok(0xf), |x| parse_num(x).ok_or("bank_mask")).map_err(|e| e.to_string())? as u32;
        let s0n = reg_n(s0)?;
        let bc = if has(i, "bound_ctrl") { 1 << 19 } else { 0 };
        v.extend(d32(s0n | ctrl << 8 | bc | bm << 24 | rm << 28));
    } else if sdwa {
        let sel = |s: Option<&str>, d: u32| -> u32 {
            match s {
                Some("BYTE_0") => 0,
                Some("BYTE_1") => 1,
                Some("BYTE_2") => 2,
                Some("BYTE_3") => 3,
                Some("WORD_0") => 4,
                Some("WORD_1") => 5,
                Some("DWORD") => 6,
                _ => d,
            }
        };
        let s0n = reg_n(s0)?;
        let mut w = s0n | sel(md(i, "dst_sel"), 6) << 8 | sel(md(i, "src0_sel"), 6) << 16;
        if s1.is_some() {
            w |= sel(md(i, "src1_sel"), 6) << 24;
        }
        v.extend(d32(w));
    } else if let Some(l) = lit {
        v.extend(d32(l));
    }
    Ok(v)
}

const SDST_AFTER: &[&str] = &["v_add_co_u32", "v_addc_co_u32", "v_sub_co_u32", "v_subb_co_u32", "v_subrev_co_u32", "v_subbrev_co_u32", "v_mad_u64_u32", "v_mad_i64_i32"];

fn sel_bits(s: &str) -> Vec<u32> {
    s.trim_matches(|c| c == '[' || c == ']').split(',').map(|x| x.trim().parse().unwrap_or(0)).collect()
}

fn enc_vop3(i: &Inst, op: u32) -> Result<Vec<u8>, String> {
    let mn = i.mn.as_str();
    let mai = mn.starts_with("v_mfma") || mn.starts_with("v_smfmac");
    let scale = mn.starts_with("v_mfma_scale");
    if scale {
        return enc_mfma_scale(i, op);
    }
    let p = mn.starts_with("v_pk_") || mn.starts_with("v_accvgpr");
    let ops = &i.ops;
    let mut d0: u32 = 0xD000_0000 | op << 16;
    let mut d1: u32 = 0;
    let (vd, srcs): (u32, &[Op]);
    let mut sdst: Option<u32> = None;
    if mai {
        if ops.len() != 4 {
            return Err(format!("{mn}: expected 4 operands"));
        }
        vd = reg_n(&ops[0])?;
        srcs = &ops[1..];
        if is_agpr(&ops[0]) {
            d0 |= 1 << 15;
        }
        if is_agpr(&ops[1]) {
            d1 |= 1 << 27;
        }
        if is_agpr(&ops[2]) {
            d1 |= 1 << 28;
        }
        let cbsz = md(i, "cbsz").map_or(0, |x| x.parse::<u32>().unwrap_or(0));
        let abid = md(i, "abid").map_or(0, |x| x.parse::<u32>().unwrap_or(0));
        let blgp = md(i, "blgp").map_or(0, |x| x.parse::<u32>().unwrap_or(0));
        d0 |= cbsz << 8 | abid << 11;
        d1 |= blgp << 29;
    } else if mn.starts_with("v_cmp") {
        vd = reg_n(&ops[0])?;
        srcs = &ops[1..];
    } else if SDST_AFTER.contains(&mn) {
        vd = reg_n(&ops[0])?;
        sdst = Some(reg_n(&ops[1])?);
        srcs = &ops[2..];
    } else if mn == "v_readlane_b32" {
        vd = reg_n(&ops[0])?;
        srcs = &ops[1..];
    } else {
        vd = reg_n(&ops[0])?;
        srcs = &ops[1..];
    }
    if srcs.len() > 3 {
        return Err(format!("{mn}: too many sources"));
    }
    d0 |= vd;
    if let Some(s) = sdst {
        d0 |= s << 8;
    }
    let mut shifts = [0u32, 9, 18];
    let mut lit_seen = false;
    for (k, s) in srcs.iter().enumerate() {
        let (c, l) = src(s)?;
        if l.is_some() {
            lit_seen = true;
        }
        d1 |= (c as u32) << shifts[k];
        if let Op::Reg { neg, abs, .. } = s {
            if *neg {
                d1 |= 1 << (29 + k);
            }
            if *abs {
                d0 |= 1 << (8 + k);
            }
        }
    }
    shifts[0] = 0;
    if lit_seen {
        return Err(format!("{mn}: literal in a VOP3 instruction"));
    }
    if has(i, "clamp") {
        d0 |= 1 << 15;
    }
    let nsrc = srcs.len();
    if let Some(s) = md(i, "op_sel") {
        for (k, b) in sel_bits(s).iter().enumerate() {
            if *b != 0 {
                d0 |= 1 << if k < nsrc { 11 + k } else { 14 };
            }
        }
    }
    if p {
        // op_sel_hi defaults to all ones: bits 27, 28 (dword 1) and 14 (dword 0)
        let hi = md(i, "op_sel_hi").map_or(vec![1, 1, 1], sel_bits);
        let h = |k: usize| hi.get(k).copied().unwrap_or(1);
        d1 |= (h(0) & 1) << 27 | (h(1) & 1) << 28;
        d0 |= (h(2) & 1) << 14;
        if let Some(s) = md(i, "neg_lo") {
            for (k, b) in sel_bits(s).iter().enumerate() {
                d1 |= (*b & 1) << (29 + k);
            }
        }
        if let Some(s) = md(i, "neg_hi") {
            for (k, b) in sel_bits(s).iter().enumerate() {
                d0 |= (*b & 1) << (8 + k);
            }
        }
    }
    let mut v = d32(d0);
    v.extend(d32(d1));
    Ok(v)
}

fn enc_mfma_scale(_i: &Inst, _op: u32) -> Result<Vec<u8>, String> {
    Err("v_mfma_scale: not yet".into())
}

fn enc_ds(i: &Inst, op: u32) -> Result<Vec<u8>, String> {
    let mn = i.mn.as_str();
    let ops = &i.ops;
    let reads = mn.contains("read") || mn.contains("_rtn") || mn.contains("permute") || mn.contains("swizzle");
    let (mut vdst, mut addr, mut d0v, mut d1v) = (0u32, 0u32, 0u32, 0u32);
    let mut acc = false;
    let mut k = 0usize;
    let mut take = |k: &mut usize| -> Result<&Op, String> {
        let o = ops.get(*k).ok_or_else(|| format!("{mn}: missing operand"))?;
        *k += 1;
        Ok(o)
    };
    if reads {
        let o = take(&mut k)?;
        vdst = reg_n(o)?;
        acc |= is_agpr(o);
    }
    let o = take(&mut k)?;
    addr = reg_n(o)?;
    if let Some(o) = ops.get(k) {
        d0v = reg_n(o)?;
        acc |= is_agpr(o);
        k += 1;
    }
    if let Some(o) = ops.get(k) {
        d1v = reg_n(o)?;
        k += 1;
    }
    let mut off16: u32 = 0;
    let (mut o0, mut o1) = (0u32, 0u32);
    let mut two = false;
    if let Some(v) = md(i, "offset") {
        if let Some(sw) = v.strip_prefix("swizzle(QUAD_PERM,") {
            let p: Vec<u32> = sw.trim_end_matches(')').split(',').map(|x| x.parse().unwrap_or(0)).collect();
            off16 = 0x8000 | p[0] | p[1] << 2 | p[2] << 4 | p[3] << 6;
        } else {
            off16 = parse_num(v).ok_or("ds offset")? as u32 & 0xffff;
        }
    }
    if let Some(v) = md(i, "offset0") {
        o0 = parse_num(v).unwrap_or(0) as u32 & 0xff;
        two = true;
    }
    if let Some(v) = md(i, "offset1") {
        o1 = parse_num(v).unwrap_or(0) as u32 & 0xff;
        two = true;
    }
    if two {
        off16 = o0 | o1 << 8;
    }
    let mut d0 = 0xD800_0000 | (op & 0x7f) << 18 | off16;
    if acc || op & 0x80 != 0 {
        d0 |= 1 << 25;
    }
    if has(i, "gds") {
        d0 |= 1 << 17;
    }
    let d1 = addr | d0v << 8 | d1v << 16 | vdst << 24;
    let mut v = d32(d0);
    v.extend(d32(d1));
    Ok(v)
}

fn enc_mubuf(i: &Inst, op: u32) -> Result<Vec<u8>, String> {
    let mn = i.mn.as_str();
    let ops = &i.ops;
    let mut d0 = 0xE000_0000 | op << 18;
    let off = md(i, "offset").map_or(0, |v| parse_num(v).unwrap_or(0) as u32 & 0xfff);
    d0 |= off;
    if has(i, "offen") {
        d0 |= 1 << 12;
    }
    if has(i, "idxen") {
        d0 |= 1 << 13;
    }
    if has(i, "sc0") || has(i, "glc") {
        d0 |= 1 << 14;
    }
    if has(i, "sc1") {
        d0 |= 1 << 15;
    }
    if has(i, "lds") {
        d0 |= 1 << 16;
    }
    if has(i, "nt") || has(i, "slc") {
        d0 |= 1 << 17;
    }
    if mn == "buffer_wbl2" || mn == "buffer_inv" {
        let mut v = d32(d0);
        v.extend(d32(0));
        return Ok(v);
    }
    // [vdata,] [vaddr,] srsrc, soffset
    let lds = has(i, "lds");
    let mut k = 0usize;
    let mut vdata = 0u32;
    let mut acc = false;
    if !lds {
        let o = ops.get(k).ok_or("buffer: vdata")?;
        vdata = reg_n(o)?;
        acc = is_agpr(o);
        k += 1;
    }
    let mut vaddr = 0u32;
    if has(i, "offen") || has(i, "idxen") {
        vaddr = reg_n(ops.get(k).ok_or("buffer: vaddr")?)?;
        k += 1;
    }
    let srsrc = reg_n(ops.get(k).ok_or("buffer: srsrc")?)?;
    k += 1;
    let (so, _) = src(ops.get(k).ok_or("buffer: soffset")?)?;
    let mut d1 = vaddr | vdata << 8 | (srsrc >> 2) << 16 | (so as u32) << 24;
    if acc {
        d1 |= 1 << 23;
    }
    let mut v = d32(d0);
    v.extend(d32(d1));
    Ok(v)
}

fn enc_global(i: &Inst, op: u32) -> Result<Vec<u8>, String> {
    let mn = i.mn.as_str();
    let ops = &i.ops;
    let store = mn.contains("store");
    let atomic = mn.contains("atomic");
    let lds = mn.contains("_lds");
    let mut d0 = 0xDC00_8000 | op << 18;
    let off = md(i, "offset").map_or(0, |v| parse_num(v).unwrap_or(0) as u32 & 0x1fff);
    d0 |= off;
    if has(i, "sc0") || has(i, "glc") {
        d0 |= 1 << 16;
    }
    if has(i, "nt") || has(i, "slc") {
        d0 |= 1 << 17;
    }
    if has(i, "sc1") {
        d0 |= 1 << 25;
    }
    let (mut vaddr, mut vdata, mut vdst) = (0u32, 0u32, 0u32);
    let mut acc = false;
    let saddr_op;
    if lds {
        vaddr = reg_n(&ops[0])?;
        saddr_op = &ops[1];
    } else if store {
        vaddr = reg_n(&ops[0])?;
        vdata = reg_n(&ops[1])?;
        acc = is_agpr(&ops[1]);
        saddr_op = &ops[2];
    } else if atomic {
        // [vdst(return),] vaddr, vdata, saddr
        if ops.len() == 4 {
            vdst = reg_n(&ops[0])?;
            vaddr = reg_n(&ops[1])?;
            vdata = reg_n(&ops[2])?;
            saddr_op = &ops[3];
        } else {
            vaddr = reg_n(&ops[0])?;
            vdata = reg_n(&ops[1])?;
            saddr_op = &ops[2];
        }
    } else {
        vdst = reg_n(&ops[0])?;
        acc = is_agpr(&ops[0]);
        vaddr = reg_n(&ops[1])?;
        saddr_op = &ops[2];
    }
    let saddr = match saddr_op {
        Op::Sp(s) if s == "off" => 0x7f,
        o => reg_n(o)?,
    };
    let mut d1 = vaddr | vdata << 8 | saddr << 16 | vdst << 24;
    if acc {
        d1 |= 1 << 23;
    }
    let mut v = d32(d0);
    v.extend(d32(d1));
    Ok(v)
}
