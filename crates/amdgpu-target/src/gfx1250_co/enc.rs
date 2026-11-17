//! In-process gfx1250 instruction encoder (docs/gfx1250-plan.md G1): the gfx1250 emitter's assembly text -> instruction bytes. Formats are
//! the gfx12 encodings of the public LLVM AMDGPU backend; opcodes come from `tables` (generated from the pinned llvm-mc). Every encoding is
//! checked byte for byte against that llvm-mc (crates/amdgpu-kernels/tests/gfx1250_encoder.rs).

use super::tables::OPS;

#[derive(Debug, Clone, PartialEq)]
enum Op {
    /// first register, count
    S(u32, u32),
    V(u32, u32),
    /// a special scalar operand by its 8-bit encoding (vcc_lo 106, ttmp 108.., null 124, m0 125, exec_lo 126)
    Special(u32),
    Imm(i64),
    Label(String),
}

struct Inst {
    mn: String,
    ops: Vec<(Op, bool)>, // (operand, negated)
    mods: Vec<(String, String)>,
}

fn parse_reg(t: &str) -> Option<Op> {
    let t = t.trim();
    let named = |n: &str| -> Option<u32> {
        Some(match n {
            "vcc_lo" | "vcc" => 106,
            "vcc_hi" => 107,
            "null" => 124,
            "m0" => 125,
            "exec_lo" | "exec" => 126,
            "exec_hi" => 127,
            _ => {
                let k = n.strip_prefix("ttmp")?.parse::<u32>().ok()?;
                if k > 15 {
                    return None;
                }
                108 + k
            }
        })
    };
    if let Some(n) = named(t) {
        return Some(Op::Special(n));
    }
    let (kind, rest) = (t.chars().next()?, &t[1..]);
    if kind != 's' && kind != 'v' {
        return None;
    }
    let (a, n) = if let Some(r) = rest.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
        let (lo, hi) = r.split_once(':')?;
        let (lo, hi) = (lo.parse::<u32>().ok()?, hi.parse::<u32>().ok()?);
        (lo, hi.checked_sub(lo)? + 1)
    } else {
        (rest.parse::<u32>().ok()?, 1)
    };
    Some(if kind == 's' { Op::S(a, n) } else { Op::V(a, n) })
}

fn parse_num(t: &str) -> Option<i64> {
    let t = t.trim();
    let (neg, b) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t),
    };
    if let Some(h) = b.strip_prefix("0x") {
        let v = i64::from_str_radix(h, 16).ok()?;
        return Some(if neg { -v } else { v });
    }
    if let Ok(v) = b.parse::<i64>() {
        return Some(if neg { -v } else { v });
    }
    // a float operand: its f32 bit pattern
    let f = b.parse::<f32>().ok()?;
    Some((if neg { -f } else { f }).to_bits() as i64)
}

fn parse_inst(line: &str) -> Result<Inst, String> {
    let line = line.split("//").next().unwrap_or("").split(';').next().unwrap_or("").trim();
    let (mn, rest) = match line.find(char::is_whitespace) {
        Some(i) => (&line[..i], line[i..].trim()),
        None => (line, ""),
    };
    let mut ops = Vec::new();
    let mut mods = Vec::new();
    // operands are comma separated; trailing modifiers (`off`, `offset:N`, `th:X`, `scope:Y`, `nv`) are blank separated after the last one
    let mut parts: Vec<String> = rest.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if let Some(last) = parts.pop() {
        let mut toks = last.split_whitespace();
        if let Some(first) = toks.next() {
            parts.push(first.to_string());
        }
        for m in toks {
            let (k, v) = m.split_once(':').unwrap_or((m, ""));
            mods.push((k.to_string(), v.to_string()));
        }
    }
    for p in parts {
        if p == "off" {
            ops.push((Op::Special(124), false));
            continue;
        }
        let (neg, body) = match p.strip_prefix('-') {
            Some(b) if parse_reg(b).is_some() => (true, b),
            _ => (false, p.as_str()),
        };
        let o = if let Some(r) = parse_reg(body) {
            r
        } else if let Some(n) = parse_num(body) {
            Op::Imm(n)
        } else if body.starts_with("BB") || body.starts_with(".L") {
            Op::Label(body.to_string())
        } else {
            return Err(format!("`{line}`: operand `{p}`"));
        };
        ops.push((o, neg));
    }
    Ok(Inst { mn: mn.to_string(), ops, mods })
}

fn lookup(mn: &str) -> Option<(&'static str, u32)> {
    OPS.iter().find(|(m, _, _)| *m == mn).map(|(_, f, o)| (*f, *o as u32))
}

fn d32(v: u32) -> [u8; 4] {
    v.to_le_bytes()
}

/// The 32-bit inline constants (integers -16..64 and the float constants by their bit pattern).
fn inline32(v: i64) -> Option<u32> {
    if (0..=64).contains(&v) {
        return Some(128 + v as u32);
    }
    if (-16..=-1).contains(&v) {
        return Some((192 - v) as u32);
    }
    let bits = v as u32;
    Some(match bits {
        0x3f00_0000 => 240,
        0xbf00_0000 => 241,
        0x3f80_0000 => 242,
        0xbf80_0000 => 243,
        0x4000_0000 => 244,
        0xc000_0000 => 245,
        0x4080_0000 => 246,
        0xc080_0000 => 247,
        0x3e22_f983 => 248,
        _ => return None,
    })
}

/// A source operand: (9-bit VOP encoding, the literal dword it needs).
fn src9(o: &Op) -> Result<(u32, Option<u32>), String> {
    Ok(match o {
        Op::V(r, _) => (256 + r, None),
        Op::S(r, _) => (*r, None),
        Op::Special(c) => (*c, None),
        Op::Imm(v) => match inline32(*v) {
            Some(c) => (c, None),
            None => (255, Some(*v as u32)),
        },
        Op::Label(l) => return Err(format!("label {l} as a source")),
    })
}

fn sdst(o: &Op) -> Result<u32, String> {
    match o {
        Op::S(r, _) => Ok(*r),
        Op::Special(c) => Ok(*c),
        x => Err(format!("{x:?} is not a scalar destination")),
    }
}

fn vreg(o: &Op) -> Result<u32, String> {
    match o {
        Op::V(r, _) => Ok(*r),
        x => Err(format!("{x:?} is not a VGPR")),
    }
}

fn op_at<'a>(i: &'a Inst, k: usize) -> Result<&'a Op, String> {
    i.ops.get(k).map(|x| &x.0).ok_or_else(|| format!("{}: operand {k} missing", i.mn))
}

fn modv<'a>(i: &'a Inst, k: &str) -> Option<&'a str> {
    i.mods.iter().find(|(m, _)| m == k).map(|(_, v)| v.as_str())
}

/// Encodes one instruction at byte offset `pc` (`label` resolves branch targets to byte offsets).
pub fn encode_line(line: &str, pc: usize, label: &dyn Fn(&str) -> Option<usize>) -> Result<Vec<u8>, String> {
    let t = line.trim();
    if let Some(words) = t.strip_prefix(".long") {
        let mut v = Vec::new();
        for w in words.split("//").next().unwrap_or("").split(',') {
            let n = parse_num(w).ok_or_else(|| format!("`{t}`: word `{w}`"))?;
            v.extend_from_slice(&d32(n as u32));
        }
        return Ok(v);
    }
    // s_setreg_imm32_b32 hwreg(NAME, offset, size), imm (SOPK with a literal); only the registers the emitter writes
    if let Some(rest) = t.strip_prefix("s_setreg_imm32_b32") {
        let (h, imm) = rest.trim().rsplit_once(',').ok_or_else(|| format!("`{t}`"))?;
        let h = h.trim().strip_prefix("hwreg(").and_then(|x| x.strip_suffix(')')).ok_or_else(|| format!("`{t}`: hwreg"))?;
        let f: Vec<&str> = h.split(',').map(|x| x.trim()).collect();
        let id = match f.first() {
            Some(&"HW_REG_WAVE_MODE") => 1u32,
            x => return Err(format!("`{t}`: hardware register {x:?}")),
        };
        let off = f.get(1).and_then(|x| x.parse::<u32>().ok()).unwrap_or(0);
        let size = f.get(2).and_then(|x| x.parse::<u32>().ok()).unwrap_or(32);
        let simm = id | off << 6 | (size - 1) << 11;
        let (_, op) = lookup("s_setreg_imm32_b32").ok_or("s_setreg_imm32_b32 not in the tables")?;
        let mut v = d32(0xB000_0000 | op << 23 | simm).to_vec();
        v.extend_from_slice(&d32(parse_num(imm).ok_or_else(|| format!("`{t}`: value"))? as u32));
        return Ok(v);
    }
    let i = parse_inst(t)?;
    let base = i.mn.strip_suffix("_e32").or_else(|| i.mn.strip_suffix("_e64")).unwrap_or(&i.mn);
    let mut e64 = i.mn.ends_with("_e64");
    let (fmt, op) = lookup(base).ok_or_else(|| format!("`{t}`: mnemonic not in the gfx1250 encoder tables"))?;
    // no explicit suffix: the e32 form when the operands allow it (VOP2 / VOPC: the second source a VGPR, no negation), else VOP3
    if !e64 && !i.mn.ends_with("_e32") && matches!(fmt, "VOP2" | "VOPC") {
        let srcs: Vec<&(Op, bool)> = i.ops.iter().filter(|o| o.0 != Op::Special(106)).collect();
        let k = if fmt == "VOP2" { 2 } else { 1 };
        e64 = srcs.iter().any(|o| o.1) || !matches!(srcs.get(k).map(|o| &o.0), Some(Op::V(..)));
    }
    let mut out = Vec::new();
    match fmt {
        "SOP1" => {
            let (d, s) = if i.ops.len() == 1 { (0, op_at(&i, 0)?) } else { (sdst(op_at(&i, 0)?)?, op_at(&i, 1)?) };
            let (s0, lit) = src9(s)?;
            out.extend_from_slice(&d32(0xBE80_0000 | d << 16 | op << 8 | (s0 & 0xff)));
            if let Some(l) = lit {
                out.extend_from_slice(&d32(l));
            }
        }
        "SOP2" => {
            let d = sdst(op_at(&i, 0)?)?;
            let (s0, l0) = src9(op_at(&i, 1)?)?;
            let (s1, l1) = src9(op_at(&i, 2)?)?;
            out.extend_from_slice(&d32(0x8000_0000 | op << 23 | d << 16 | (s1 & 0xff) << 8 | (s0 & 0xff)));
            if let Some(l) = l0.or(l1) {
                out.extend_from_slice(&d32(l));
            }
        }
        "SOPC" => {
            let (s0, l0) = src9(op_at(&i, 0)?)?;
            let (s1, l1) = src9(op_at(&i, 1)?)?;
            out.extend_from_slice(&d32(0xBF00_0000 | op << 16 | (s1 & 0xff) << 8 | (s0 & 0xff)));
            if let Some(l) = l0.or(l1) {
                out.extend_from_slice(&d32(l));
            }
        }
        "SOPP" => {
            let simm = match i.ops.first().map(|x| &x.0) {
                None => 0,
                Some(Op::Imm(v)) => (*v as u32) & 0xffff,
                Some(Op::Label(l)) => {
                    let tgt = label(l).ok_or_else(|| format!("`{t}`: unknown label {l}"))?;
                    let d = (tgt as i64 - (pc as i64 + 4)) / 4;
                    (d as u32) & 0xffff
                }
                Some(x) => return Err(format!("`{t}`: {x:?}")),
            };
            out.extend_from_slice(&d32(0xBF80_0000 | op << 16 | simm));
        }
        "SMEM" => {
            let d = sdst(op_at(&i, 0)?)?;
            let b = match op_at(&i, 1)? {
                Op::S(r, 2) => *r,
                x => return Err(format!("`{t}`: base {x:?}")),
            };
            let off = match op_at(&i, 2)? {
                Op::Imm(v) => *v,
                x => return Err(format!("`{t}`: offset {x:?}")),
            };
            out.extend_from_slice(&d32(0xF400_0000 | op << 13 | d << 6 | (b >> 1)));
            out.extend_from_slice(&d32(0x7C << 25 | (off as u32 & 0x00ff_ffff)));
        }
        "VOP1" if !e64 => {
            if base == "v_nop" {
                out.extend_from_slice(&d32(0x7E00_0000));
            } else {
                let d = match op_at(&i, 0)? {
                    Op::V(r, _) => *r,
                    Op::S(r, _) => *r, // v_readfirstlane_b32 sD
                    x => return Err(format!("`{t}`: {x:?}")),
                };
                let (s0, lit) = src9(op_at(&i, 1)?)?;
                out.extend_from_slice(&d32(0x7E00_0000 | d << 17 | op << 9 | s0));
                if let Some(l) = lit {
                    out.extend_from_slice(&d32(l));
                }
            }
        }
        "VOP2" if !e64 => {
            // carry forms: vD, vcc_lo, src0, vsrc1, vcc_lo; cndmask: vD, src0, vsrc1, vcc_lo
            let srcs: Vec<&Op> = i.ops.iter().map(|x| &x.0).filter(|o| **o != Op::Special(106)).collect();
            let d = vreg(srcs.first().ok_or("vdst")?)?;
            let (s0, lit) = src9(srcs.get(1).ok_or("src0")?)?;
            let s1 = vreg(srcs.get(2).ok_or("vsrc1")?)?;
            if i.ops.iter().any(|x| x.1) {
                return Err(format!("`{t}`: a negated source needs the e64 form"));
            }
            out.extend_from_slice(&d32(op << 25 | d << 17 | s1 << 9 | s0));
            if let Some(l) = lit {
                out.extend_from_slice(&d32(l));
            }
        }
        "VOPC" if !e64 => {
            // vcc_lo, src0, vsrc1
            let (s0, lit) = src9(op_at(&i, 1)?)?;
            let s1 = vreg(op_at(&i, 2)?)?;
            out.extend_from_slice(&d32(0x7C00_0000 | op << 17 | s1 << 9 | s0));
            if let Some(l) = lit {
                out.extend_from_slice(&d32(l));
            }
        }
        "VOP1" | "VOP2" | "VOPC" | "VOP3" => {
            // the VOP3 forms: VOP1 at 384 + op, VOP2 at 256 + op, VOPC at op; VOP3b (carry out / in) keeps the SGPR destination in [14:8]
            let op3 = match fmt {
                "VOP1" => 384 + op,
                "VOP2" => 256 + op,
                _ => op,
            };
            let vop3b = matches!(base, "v_add_co_u32" | "v_sub_co_u32" | "v_add_co_ci_u32" | "v_sub_co_ci_u32");
            let (dst, srcs): (u32, Vec<(Op, bool)>) = if fmt == "VOPC" {
                (sdst(op_at(&i, 0)?)?, i.ops[1..].to_vec())
            } else if vop3b {
                // vD, sdst, srcs...
                (vreg(op_at(&i, 0)?)?, i.ops[2..].to_vec())
            } else {
                (match op_at(&i, 0)? {
                    Op::V(r, _) => *r,
                    Op::S(r, _) => *r,
                    x => return Err(format!("`{t}`: {x:?}")),
                }, i.ops[1..].to_vec())
            };
            let sd = if vop3b { sdst(op_at(&i, 1)?)? } else { 0 };
            // (an unused source field holds the inline constant 0)
            let mut enc = [128u32; 3];
            let mut neg = 0u32;
            let mut lit = None;
            for (k, (o, n)) in srcs.iter().enumerate().take(3) {
                let (c, l) = src9(o)?;
                enc[k] = c;
                if l.is_some() {
                    if lit.is_some() && lit != l {
                        return Err(format!("`{t}`: two different literals"));
                    }
                    lit = l;
                }
                if *n {
                    neg |= 1 << k;
                }
            }
            let clamp = i.mods.iter().any(|(m, _)| m == "clamp") as u32;
            out.extend_from_slice(&d32(0xD400_0000 | op3 << 16 | clamp << 15 | sd << 8 | dst));
            out.extend_from_slice(&d32(neg << 29 | enc[2] << 18 | enc[1] << 9 | enc[0]));
            if let Some(l) = lit {
                out.extend_from_slice(&d32(l));
            }
        }
        "VGLOBAL" => {
            // loads: vD, vaddr, saddr|off; stores: vaddr, vdata, saddr|off; prefetch: vaddr, saddr|off
            let store = base.starts_with("global_store");
            let (data, addr, saddr) = if base.starts_with("global_prefetch") {
                (0, op_at(&i, 0)?, op_at(&i, 1)?)
            } else if store {
                (vreg(op_at(&i, 1)?)?, op_at(&i, 0)?, op_at(&i, 2)?)
            } else {
                (vreg(op_at(&i, 0)?)?, op_at(&i, 1)?, op_at(&i, 2)?)
            };
            const SC: [&str; 4] = ["", "SCOPE_SE", "SCOPE_DEV", "SCOPE_SYS"];
            let scope = modv(&i, "scope").map(|v| SC.iter().position(|x| *x == v).unwrap_or(0) as u32).unwrap_or(0);
            let va = vreg(addr)?;
            let sa = match saddr {
                Op::Special(124) => 0x7C,
                Op::S(r, 2) => *r,
                x => return Err(format!("`{t}`: saddr {x:?}")),
            };
            let off = modv(&i, "offset").map(|v| parse_num(v).ok_or(format!("`{t}`: offset"))).transpose()?.unwrap_or(0);
            out.extend_from_slice(&d32(0xEE00_0000 | op << 14 | sa));
            out.extend_from_slice(&d32(if store { data << 23 } else { data } | scope << 18));
            out.extend_from_slice(&d32(((off as u32) & 0x00ff_ffff) << 8 | va));
        }
        "VDS" => {
            let store = base.starts_with("ds_store");
            let (vdst, addr, data0) = if store { (0, vreg(op_at(&i, 0)?)?, vreg(op_at(&i, 1)?)?) } else { (vreg(op_at(&i, 0)?)?, vreg(op_at(&i, 1)?)?, 0) };
            let off = modv(&i, "offset").map(|v| parse_num(v).ok_or(format!("`{t}`: offset"))).transpose()?.unwrap_or(0) as u32;
            out.extend_from_slice(&d32(0xD800_0000 | op << 18 | (off & 0xffff)));
            out.extend_from_slice(&d32(vdst << 24 | data0 << 8 | addr));
        }
        "VIMAGE" => {
            // tensor ops: g0 (4 SGPRs), g1 (8)[, g2, g3]; DMASK = 1, DIM = 1, VADDR4 = NULL (the node assembler's spelling)
            let g = |k: usize| -> Result<u32, String> {
                match i.ops.get(k).map(|x| &x.0) {
                    None => Ok(0x7C),
                    Some(Op::S(r, _)) => Ok(*r),
                    Some(Op::Special(124)) => Ok(0x7C),
                    Some(x) => Err(format!("`{t}`: group {x:?}")),
                }
            };
            const TH: [&str; 8] = ["", "NT", "HT", "BYPASS", "LU", "NT_RT", "RT_NT", "NT_HT"];
            const SC: [&str; 4] = ["", "SCOPE_SE", "SCOPE_DEV", "SCOPE_SYS"];
            let th = modv(&i, "th").map(|v| {
                let v = v.trim_start_matches("TH_LOAD_").trim_start_matches("TH_STORE_");
                TH.iter().position(|x| *x == v).unwrap_or(0) as u32
            });
            let scope = modv(&i, "scope").map(|v| SC.iter().position(|x| *x == v).unwrap_or(0) as u32);
            let nv = i.mods.iter().any(|(m, _)| m == "nv") as u32;
            out.extend_from_slice(&d32(0xD000_0000 | 1 << 22 | op << 14 | nv << 6 | 1));
            out.extend_from_slice(&d32(0x7C << 24 | th.unwrap_or(0) << 20 | scope.unwrap_or(0) << 18));
            out.extend_from_slice(&d32(g(0)? | g(1)? << 8 | g(2)? << 16 | g(3)? << 24));
        }
        "VOP3P" | "VOP3PX2" => {
            // WMMA: vD, A, B, C (C a VGPR tuple or the inline 0); op_sel_hi = 3 (the default). VOP3PX2 (the scaled forms) is first the scale
            // prefix (VOP3P op 0x35: scale A, scale B, the inline 0; matrix_b_scale:MATRIX_SCALE_ROW1 sets its op_sel_hi bit 0), then the
            // matrix instruction with op_sel_hi[2] (d0 bit 14) set
            let d = vreg(op_at(&i, 0)?)?;
            let mut enc = [0u32; 3];
            for k in 0..3 {
                let (c, l) = src9(op_at(&i, k + 1)?)?;
                if l.is_some() {
                    return Err(format!("`{t}`: a literal in a matrix operand"));
                }
                enc[k] = c;
            }
            if fmt == "VOP3PX2" {
                let (sa, _) = src9(op_at(&i, 4)?)?;
                let (sb, _) = src9(op_at(&i, 5)?)?;
                let row1 = modv(&i, "matrix_b_scale") == Some("MATRIX_SCALE_ROW1");
                out.extend_from_slice(&d32(0xCC00_0000 | 0x35 << 16));
                out.extend_from_slice(&d32((row1 as u32) << 27 | 128 << 18 | sb << 9 | sa));
            }
            let px2 = (fmt == "VOP3PX2") as u32;
            out.extend_from_slice(&d32(0xCC00_0000 | op << 16 | px2 << 14 | d));
            out.extend_from_slice(&d32(3 << 27 | enc[2] << 18 | enc[1] << 9 | enc[0]));
        }
        f => return Err(format!("`{t}`: format {f} not encoded")),
    }
    Ok(out)
}

/// The instruction bytes of a function's assembly text (`label:` lines, instructions, `.long`; other directives are skipped). Two passes:
/// sizes, then the branch offsets.
pub fn encode_text(asm: &str) -> Result<Vec<u8>, String> {
    let lines: Vec<&str> = asm.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with("//")).collect();
    let is_inst = |l: &str| !(l.ends_with(':') || (l.starts_with('.') && !l.starts_with(".long")));
    let mut labels = std::collections::HashMap::new();
    let mut pc = 0usize;
    for l in &lines {
        if l.ends_with(':') {
            labels.insert(l.trim_end_matches(':').to_string(), pc);
        } else if is_inst(l) {
            pc += encode_line(l, pc, &|_| Some(0))?.len();
        }
    }
    let mut out = Vec::new();
    for l in &lines {
        if is_inst(l) {
            let pc = out.len();
            out.extend(encode_line(l, pc, &|n| labels.get(n).copied())?);
        }
    }
    Ok(out)
}
