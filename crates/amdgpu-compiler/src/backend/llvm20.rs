//! LLVM 20 (AMD ROCm 7.0.2) dialect: the opcode/operand table for post-RA MIR and the few
//! version-specific encodings. Everything that has to change when the pinned LLVM changes lives
//! in this file plus `llvm20.tbl` and `llvm20_header.txt`.

use std::collections::HashMap;

use crate::mir::*;
use crate::regalloc::Alloc;

pub const NAME: &str = "llvm20";
/// `llc --version` first line this dialect was written and validated against.
pub const LLC_VERSION_PREFIX: &str = "AMD LLVM version 20.";
pub const TABLE_SRC: &str = include_str!("llvm20.tbl");
pub const HEADER_SRC: &str = include_str!("llvm20_header.txt");
/// Pass the tail starts at; first one that exists in the pipeline is used.
pub const START_PASS_PREFERENCE: &[&str] = &["gcn-create-vopd", "si-memory-legalizer"];
pub const LLC_ARGS: &[&str] = &["-mtriple=amdgcn-amd-amdhsa", "-mcpu=gfx950", "-O2"];

/// Operand class named in a table signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SigClass {
    Reg(RClass),
    Imm,
}

#[derive(Clone, Debug)]
pub struct Row {
    pub key: String,
    pub dst: Option<RClass>,
    pub srcs: Vec<SigClass>,
    pub text: Vec<String>,
}

pub struct Table {
    pub rows: HashMap<String, Row>,
}

fn parse_class(s: &str) -> Result<SigClass, String> {
    Ok(match s {
        "V1" => SigClass::Reg(RClass::V1),
        "V2" => SigClass::Reg(RClass::V2),
        "V4" => SigClass::Reg(RClass::V4),
        "V8" => SigClass::Reg(RClass::V8),
        "V16" => SigClass::Reg(RClass::V16),
        "S1" => SigClass::Reg(RClass::S1),
        "S2" => SigClass::Reg(RClass::S2),
        "I" => SigClass::Imm,
        _ => return Err(format!("unknown signature class '{s}'")),
    })
}

impl Table {
    pub fn parse(src: &str) -> Result<Table, String> {
        let mut rows = HashMap::new();
        for (ln, line) in src.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.splitn(3, '|').collect();
            if parts.len() != 3 {
                return Err(format!("llvm20.tbl:{}: expected 'Key | sig | text'", ln + 1));
            }
            let key = parts[0].trim().to_string();
            let sig: Vec<&str> = parts[1].split("<-").collect();
            if sig.len() != 2 {
                return Err(format!("llvm20.tbl:{}: signature needs '<-'", ln + 1));
            }
            let dst = match sig[0].trim() {
                "" => None,
                s => match parse_class(s).map_err(|e| format!("llvm20.tbl:{}: {e}", ln + 1))? {
                    SigClass::Reg(c) => Some(c),
                    SigClass::Imm => return Err(format!("llvm20.tbl:{}: dst cannot be I", ln + 1)),
                },
            };
            let mut srcs = Vec::new();
            for s in sig[1].split_whitespace() {
                srcs.push(parse_class(s).map_err(|e| format!("llvm20.tbl:{}: {e}", ln + 1))?);
            }
            let text: Vec<String> = parts[2].split(" ;; ").map(|s| s.trim().to_string()).collect();
            if rows.insert(key.clone(), Row { key, dst, srcs, text }).is_some() {
                return Err(format!("llvm20.tbl:{}: duplicate key", ln + 1));
            }
        }
        Ok(Table { rows })
    }

    pub fn load() -> Table {
        Table::parse(TABLE_SRC).expect("llvm20.tbl is malformed")
    }

    /// Most specific row for an instruction: `Op:kinds` then `Op`.
    pub fn lookup(&self, op: &str, kinds: &str) -> Option<&Row> {
        self.rows.get(&format!("{op}:{kinds}")).or_else(|| self.rows.get(op))
    }
}

pub fn kinds_of(inst: &MInst) -> String {
    if inst.srcs.is_empty() {
        return "-".to_string();
    }
    inst.srcs
        .iter()
        .map(|s| match s {
            Src::Reg(_) => 'r',
            Src::Imm(_) => 'i',
        })
        .collect()
}

/// gfx9 S_WAITCNT immediate: vmcnt [3:0]+[15:14], expcnt [6:4], lgkmcnt [11:8].
pub fn waitcnt_imm(vm: Option<u32>, lgkm: Option<u32>) -> u32 {
    let vm = vm.unwrap_or(63).min(63);
    let lgkm = lgkm.unwrap_or(15).min(15);
    (vm & 0xf) | ((vm >> 4) << 14) | (7 << 4) | (lgkm << 8)
}

fn reg_name(class: RClass, phys: u32) -> String {
    let (prefix, n) = if class.is_vector() { ("vgpr", class.dwords()) } else { ("sgpr", class.dwords()) };
    let parts: Vec<String> = (0..n).map(|k| format!("{prefix}{}", phys + k)).collect();
    format!("${}", parts.join("_"))
}

fn dword_name(class: RClass, phys: u32, k: u32) -> String {
    let prefix = if class.is_vector() { "vgpr" } else { "sgpr" };
    format!("${prefix}{}", phys + k)
}

pub struct RenderCtx<'a> {
    pub m: &'a MFunction,
    pub a: &'a Alloc,
    pub target: Option<String>,
}

impl RenderCtx<'_> {
    fn phys(&self, r: VReg) -> u32 {
        self.a.phys[r as usize]
    }
    fn class(&self, r: VReg) -> RClass {
        self.m.vreg_class(r)
    }
}

fn src_reg(inst: &MInst, n: usize) -> Result<VReg, String> {
    match inst.srcs.get(n) {
        Some(Src::Reg(r)) => Ok(*r),
        _ => Err(format!("{:?}: source {n} is not a register", inst.op)),
    }
}

fn eval_token(tok: &str, inst: &MInst, cx: &RenderCtx) -> Result<String, String> {
    let (base, dword) = match tok.split_once('.') {
        Some((b, k)) => (b, Some(k.parse::<u32>().map_err(|_| format!("bad token {{{tok}}}"))?)),
        None => (tok, None),
    };
    let dst = || inst.dst.ok_or_else(|| format!("{:?}: token {{{tok}}} needs a destination", inst.op));
    Ok(match base {
        "d" => {
            let d = dst()?;
            match dword {
                Some(k) => dword_name(cx.class(d), cx.phys(d), k),
                None => reg_name(cx.class(d), cx.phys(d)),
            }
        }
        "imm" => format!("{}", inst.imm),
        "imm32" => format!("{}", inst.imm as i32),
        "i0" => match inst.srcs.first() {
            Some(Src::Imm(i)) => format!("{}", *i as i32),
            Some(Src::Reg(_)) => return Err(format!("{:?}: {{i0}} on a register source", inst.op)),
            None => format!("{}", inst.imm as i32),
        },
        "dsub" => {
            let d = dst()?;
            dword_name(RClass::V1, cx.phys(d), inst.imm as u32)
        }
        "s0sub" => {
            let r = src_reg(inst, 0)?;
            dword_name(RClass::V1, cx.phys(r), inst.imm as u32)
        }
        "pv" => format!("$vgpr{}", inst.imm),
        "ps" => format!("$sgpr{}", inst.imm),
        "vmcnt" => format!("{}", waitcnt_imm(Some(inst.imm as u32), None)),
        "lgkmcnt" => format!("{}", waitcnt_imm(None, Some(inst.imm as u32))),
        "t" => cx.target.clone().ok_or("{t} outside a terminator")?,
        s if s.starts_with('s') && s[1..].chars().all(|c| c.is_ascii_digit()) && s.len() > 1 => {
            let n: usize = s[1..].parse().unwrap();
            match inst.srcs.get(n) {
                Some(Src::Reg(r)) => match dword {
                    Some(k) => dword_name(cx.class(*r), cx.phys(*r), k),
                    None => reg_name(cx.class(*r), cx.phys(*r)),
                },
                Some(Src::Imm(i)) => format!("{}", *i as i32),
                None => return Err(format!("{:?}: missing source {n}", inst.op)),
            }
        }
        _ => return Err(format!("unknown token {{{tok}}}")),
    })
}

pub fn render_text(tpl: &str, inst: &MInst, cx: &RenderCtx) -> Result<String, String> {
    let mut out = String::new();
    let mut rest = tpl;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let j = rest[i..].find('}').ok_or("unterminated token")? + i;
        out.push_str(&eval_token(&rest[i + 1..j], inst, cx)?);
        rest = &rest[j + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// MIR lines for one instruction (empty for dropped ones such as `s_nop`).
pub fn render_inst(t: &Table, inst: &MInst, cx: &RenderCtx) -> Result<Vec<String>, String> {
    let name = format!("{:?}", inst.op);
    let row = t
        .lookup(&name, &kinds_of(inst))
        .ok_or_else(|| format!("llvm20.tbl has no row for {name} (kinds '{}')", kinds_of(inst)))?;
    if row.text.len() == 1 && row.text[0] == "DROP" {
        return Ok(Vec::new());
    }
    row.text.iter().map(|tpl| render_text(tpl, inst, cx)).collect()
}

pub fn render_term(t: &Table, key: &str, target: Option<String>, m: &MFunction, a: &Alloc) -> Result<String, String> {
    let row = t.rows.get(key).ok_or_else(|| format!("llvm20.tbl has no row {key}"))?;
    let cx = RenderCtx { m, a, target };
    render_text(&row.text[0], &MInst::new(MOp::SNop), &cx)
}