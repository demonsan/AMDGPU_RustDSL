//! LLVM 20 (AMD ROCm 7.0.2) dialect: the opcode/operand table for post-RA MIR and the few
//! version-specific encodings. Everything that has to change when the pinned LLVM changes lives
//! in this file plus `llvm20.tbl` and `llvm20_header.txt`.

use std::collections::HashMap;

use crate::mir::*;
use crate::regalloc::Alloc;

pub const NAME: &str = "llvm20";

/// One LLVM version's data: selected by the probed llc --version (or AMDC_DIALECT).
#[derive(Debug)]
pub struct Dialect {
    pub name: &'static str,
    pub version_prefix: &'static str,
    pub table_src: &'static str,
    pub header_src: &'static str,
}

pub const DIALECTS: &[Dialect] = &[
    Dialect { name: "llvm20", version_prefix: "AMD LLVM version 20.", table_src: include_str!("llvm20.tbl"), header_src: include_str!("llvm20_header.txt") },
    Dialect { name: "llvm24", version_prefix: "AMD LLVM version 24.", table_src: include_str!("llvm24.tbl"), header_src: include_str!("llvm24_header.txt") },
];

/// The dialect for the pinned LLVM: $AMDC_DIALECT, else matched against llc --version, else the primary pin (llvm24).
pub fn current() -> &'static Dialect {
    static CUR: std::sync::OnceLock<&'static Dialect> = std::sync::OnceLock::new();
    CUR.get_or_init(|| {
        if let Ok(n) = std::env::var("AMDC_DIALECT") {
            if let Some(d) = DIALECTS.iter().find(|d| d.name == n) {
                return d;
            }
        }
        if let Ok(v) = super::probe::llc_version(&super::LlvmTools::locate()) {
            if let Some(d) = DIALECTS.iter().find(|d| v.starts_with(d.version_prefix)) {
                return d;
            }
        }
        &DIALECTS[DIALECTS.len() - 1]
    })
}
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
        "S4" => SigClass::Reg(RClass::S4),
        "A1" => SigClass::Reg(RClass::A1),
        "A2" => SigClass::Reg(RClass::A2),
        "A4" => SigClass::Reg(RClass::A4),
        "A8" => SigClass::Reg(RClass::A8),
        "A16" => SigClass::Reg(RClass::A16),
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
        Table::parse(current().table_src).expect("llvm20.tbl is malformed")
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
    let (prefix, n) = if class.is_agpr() { ("agpr", class.dwords()) } else if class.is_vector() { ("vgpr", class.dwords()) } else { ("sgpr", class.dwords()) };
    let phys = if class.is_agpr() { phys - AGPR_TAG } else { phys };
    let parts: Vec<String> = (0..n).map(|k| format!("{prefix}{}", phys + k)).collect();
    format!("${}", parts.join("_"))
}

fn dword_name(class: RClass, phys: u32, k: u32) -> String {
    let prefix = if class.is_agpr() { "agpr" } else if class.is_vector() { "vgpr" } else { "sgpr" };
    let phys = if class.is_agpr() { phys - AGPR_TAG } else { phys };
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
        // two-address DS offsets (imm = offset0 | offset1 << 8)
        "off0" => format!("{}", inst.imm & 0xff),
        "off1" => format!("{}", (inst.imm >> 8) & 0xff),
        // the first three dwords of a 4-dword tuple (96-bit accesses use V4 registers)
        "d3" => {
            let d = dst()?;
            (0..3).map(|k| dword_name(cx.class(d), cx.phys(d), k)).collect::<Vec<_>>().join("_").replacen("_$", "_", 2)
        }
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
        // scaled MFMA byte selects (inst.imm: bits 0..1 scale A byte, bits 4..5 scale B byte): MIR op_sel immediates are 4 * byte (llc dump: A=3,B=2 -> 12, 8)
        "opselA" => format!("{}", 4 * (inst.imm & 3)),
        "opselB" => format!("{}", 4 * ((inst.imm >> 4) & 3)),
        "pv" => format!("$vgpr{}", inst.imm),
        "ps" => format!("$sgpr{}", inst.imm),
        "vmcnt" => format!("{}", waitcnt_imm(Some(inst.imm as u32), None)),
        "lgkmcnt" => format!("{}", waitcnt_imm(None, Some(inst.imm as u32))),
        "t" => cx.target.clone().ok_or("{t} outside a terminator")?,
        s if s.starts_with('s') && s.ends_with("t3") && s[1..s.len() - 2].chars().all(|c| c.is_ascii_digit()) && s.len() > 3 => {
            let n: usize = s[1..s.len() - 2].parse().unwrap();
            let r = src_reg(inst, n)?;
            (0..3).map(|k| dword_name(cx.class(r), cx.phys(r), k)).collect::<Vec<_>>().join("_").replacen("_$", "_", 2)
        }
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
    // Op@a rows: the accumulator-register (acc_cd) form of an MFMA, chosen when the destination is an AGPR; other ops use their normal row
    // (a ds_read/global_load into an AGPR tuple is the same LLVM opcode).
    let kinds = kinds_of(inst);
    let a_form = inst.dst.map_or(false, |d| cx.class(d).is_agpr());
    let row = (if a_form { t.lookup(&format!("{name}@a"), &kinds) } else { None })
        .or_else(|| t.lookup(&name, &kinds))
        .ok_or_else(|| format!("llvm20.tbl has no row for {name} (kinds '{kinds}')"))?;
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