//! The DSL IR text (`irv1`, docs/ir-text.md): a complete, parseable printing of a [`Function`] and its parser, so a kernel can be dumped
//! (`amdc print-ir --stage dsl`), edited and compiled again (`amdc emit --from-ir`) without touching Rust.
//!
//! Format: a token stream (whitespace and newlines are insignificant inside values) of keyword items. Every field VALUE is spelled exactly
//! as Rust's `Debug` prints it (so new IR fields / variants print automatically); the decoders are generated (`ir_codec.rs`,
//! `tools/gen_ir_codec.py`) with an exhaustive guard. Structure is explicit (`loop .. { .. }`, `if .. { .. } else { .. }`,
//! `region .. { .. }`, `stage .. { .. }`); indentation is cosmetic. Value / variable ids are kept as they are (no renumbering), so
//! `parse(print(f))` is field-for-field equal to `f` and compiles to the same code.
//!
//! ```text
//! irv1
//! dsl "name"
//! kernarg_size 36  workgroup 64  lds 0  value_count 18
//! profile None  lower_opts []  sched_opts None
//! var_tys [I32]
//! arg Arg { name: "a", ty: Ptr, kernarg_offset: 0, size: 8 }
//! agpr AgprMark { .. }           reg_plan PinV { .. }          region_decl RegionDecl { .. }
//! contract Contract { .. }
//! values { %v7: I32 }            (types of values not defined by a `Let`)
//! body {
//!   %v0: Ptr = KernargPtr { offset: 0 }
//!   loop vars [VarId(0)] init [I32(0)] {
//!     StoreF32 { base: Val(9), byte_off: 0, value: Val(15) }
//!   }
//!   if Val(3) { .. } else { .. }
//!   region TileProduce tile "K" slot I32(0) version I32(0) { .. }
//!   stage "name" StageContract { .. } { .. }
//! }
//! ```

use crate::*;
use std::fmt::Write;

// ---------------------------------------------------------------------------------------------
// generic Debug-value tree
// ---------------------------------------------------------------------------------------------

/// A parsed `Debug` value.
#[derive(Clone, Debug, PartialEq)]
pub enum IrVal {
    /// a bare identifier: unit variant / unit struct / `None` / `true` / `false`
    Ident(String),
    /// `Name(a, b)` or an anonymous tuple `(a, b)` (name `None`)
    Tuple(Option<String>, Vec<IrVal>),
    /// `Name { f: v, .. }`
    Struct(String, Vec<(String, IrVal)>),
    List(Vec<IrVal>),
    /// a number, kept as text (decoded per target type)
    Num(String),
    Str(String),
}

/// A decode / parse error (the parser adds the line).
#[derive(Clone, Debug, PartialEq)]
pub struct IrErr(pub String);

impl std::fmt::Display for IrErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn short(v: &IrVal) -> String {
    let s = format!("{v:?}");
    if s.len() > 80 {
        format!("{}…", &s[..s.char_indices().nth(80).map(|x| x.0).unwrap_or(s.len())])
    } else {
        s
    }
}

impl IrVal {
    pub fn err(&self, m: String) -> IrErr {
        IrErr(format!("{m} (at {})", short(self)))
    }
    /// The constructor name of a variant / struct value.
    pub fn name(&self) -> Result<&str, IrErr> {
        match self {
            IrVal::Ident(n) | IrVal::Struct(n, _) => Ok(n),
            IrVal::Tuple(Some(n), _) => Ok(n),
            _ => Err(self.err("expected a named value".into())),
        }
    }
    pub fn expect_unit(&self, n: &str) -> Result<(), IrErr> {
        match self {
            IrVal::Ident(x) if x == n => Ok(()),
            _ => Err(self.err(format!("expected `{n}`"))),
        }
    }
    pub fn expect_struct(&self, n: &str) -> Result<(), IrErr> {
        match self {
            IrVal::Struct(x, _) if x == n => Ok(()),
            // a struct with no fields prints as a bare name
            IrVal::Ident(x) if x == n => Ok(()),
            _ => Err(self.err(format!("expected `{n} {{ .. }}`"))),
        }
    }
    pub fn expect_tuple(&self, n: &str, k: usize) -> Result<(), IrErr> {
        match self {
            IrVal::Tuple(Some(x), items) if x == n && items.len() == k => Ok(()),
            IrVal::Tuple(Some(x), items) if x == n => Err(self.err(format!("`{n}` takes {k} value(s), {} given", items.len()))),
            _ => Err(self.err(format!("expected `{n}(..)`"))),
        }
    }
    pub fn field(&self, f: &str) -> Result<&IrVal, IrErr> {
        match self {
            IrVal::Struct(_, fs) => fs.iter().find(|(n, _)| n == f).map(|x| &x.1).ok_or_else(|| self.err(format!("missing field `{f}`"))),
            _ => Err(self.err(format!("missing field `{f}`"))),
        }
    }
    pub fn item(&self, k: usize) -> Result<&IrVal, IrErr> {
        match self {
            IrVal::Tuple(_, items) => items.get(k).ok_or_else(|| self.err(format!("missing value {k}"))),
            _ => Err(self.err("expected a tuple".into())),
        }
    }
}

/// Decoding from the `Debug` spelling.
pub trait FromIr: Sized {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr>;
}

/// Visiting every ValueId / VarId (alpha renaming).
pub trait IrIds {
    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId));
}

macro_rules! num_from_ir {
    ($($t:ty),*) => {$(
        impl FromIr for $t {
            fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
                match v {
                    IrVal::Num(s) => s.parse::<$t>().map_err(|_| v.err(format!("not a {}", stringify!($t)))),
                    _ => Err(v.err(format!("expected a {}", stringify!($t)))),
                }
            }
        }
        impl IrIds for $t {
            fn ids(&mut self, _: &mut dyn FnMut(&mut ValueId), _: &mut dyn FnMut(&mut VarId)) {}
        }
    )*};
}
num_from_ir!(u8, u16, u32, u64, i32, i64, usize);

impl FromIr for f32 {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v {
            IrVal::Num(s) => s.parse::<f32>().map_err(|_| v.err("not an f32".into())),
            IrVal::Ident(s) if s == "inf" => Ok(f32::INFINITY),
            IrVal::Ident(s) if s == "NaN" => Ok(f32::NAN),
            _ => Err(v.err("expected an f32".into())),
        }
    }
}
impl IrIds for f32 {
    fn ids(&mut self, _: &mut dyn FnMut(&mut ValueId), _: &mut dyn FnMut(&mut VarId)) {}
}
impl FromIr for bool {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v {
            IrVal::Ident(s) if s == "true" => Ok(true),
            IrVal::Ident(s) if s == "false" => Ok(false),
            _ => Err(v.err("expected a bool".into())),
        }
    }
}
impl IrIds for bool {
    fn ids(&mut self, _: &mut dyn FnMut(&mut ValueId), _: &mut dyn FnMut(&mut VarId)) {}
}
impl FromIr for String {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v {
            IrVal::Str(s) => Ok(s.clone()),
            _ => Err(v.err("expected a string".into())),
        }
    }
}
impl IrIds for String {
    fn ids(&mut self, _: &mut dyn FnMut(&mut ValueId), _: &mut dyn FnMut(&mut VarId)) {}
}
impl<T: FromIr> FromIr for Option<T> {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v {
            IrVal::Ident(s) if s == "None" => Ok(None),
            IrVal::Tuple(Some(s), items) if s == "Some" && items.len() == 1 => Ok(Some(T::from_ir(&items[0])?)),
            _ => Err(v.err("expected `None` or `Some(..)`".into())),
        }
    }
}
impl<T: IrIds> IrIds for Option<T> {
    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId)) {
        if let Some(x) = self {
            x.ids(fv, fr);
        }
    }
}
impl<T: FromIr> FromIr for Vec<T> {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v {
            IrVal::List(items) => items.iter().map(T::from_ir).collect(),
            _ => Err(v.err("expected a list `[..]`".into())),
        }
    }
}
impl<T: IrIds> IrIds for Vec<T> {
    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId)) {
        for x in self.iter_mut() {
            x.ids(fv, fr);
        }
    }
}
impl<T: FromIr, const N: usize> FromIr for [T; N] {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        let xs: Vec<T> = Vec::from_ir(v)?;
        let n = xs.len();
        xs.try_into().map_err(|_| v.err(format!("expected {N} values, {n} given")))
    }
}
impl<T: IrIds, const N: usize> IrIds for [T; N] {
    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId)) {
        for x in self.iter_mut() {
            x.ids(fv, fr);
        }
    }
}
impl<T: FromIr> FromIr for Box<T> {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        Ok(Box::new(T::from_ir(v)?))
    }
}
impl<T: IrIds> IrIds for Box<T> {
    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId)) {
        (**self).ids(fv, fr);
    }
}
impl<A: FromIr, B: FromIr> FromIr for (A, B) {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v {
            IrVal::Tuple(None, items) if items.len() == 2 => Ok((A::from_ir(&items[0])?, B::from_ir(&items[1])?)),
            _ => Err(v.err("expected a pair `(a, b)`".into())),
        }
    }
}
impl<A: IrIds, B: IrIds> IrIds for (A, B) {
    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId)) {
        self.0.ids(fv, fr);
        self.1.ids(fv, fr);
    }
}
impl<A: FromIr, B: FromIr, C: FromIr> FromIr for (A, B, C) {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v {
            IrVal::Tuple(None, items) if items.len() == 3 => Ok((A::from_ir(&items[0])?, B::from_ir(&items[1])?, C::from_ir(&items[2])?)),
            _ => Err(v.err("expected a triple `(a, b, c)`".into())),
        }
    }
}
impl<A: IrIds, B: IrIds, C: IrIds> IrIds for (A, B, C) {
    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId)) {
        self.0.ids(fv, fr);
        self.1.ids(fv, fr);
        self.2.ids(fv, fr);
    }
}

// safe-asm blocks: decoded through their public constructor; the compiler re-verifies every block before lowering
impl FromIr for amdgpu_safeasm::AClass {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        use amdgpu_safeasm::AClass as A;
        Ok(match v.name()? {
            "V1" => A::V1,
            "V2" => A::V2,
            "V4" => A::V4,
            "V8" => A::V8,
            "V16" => A::V16,
            "S1" => A::S1,
            "S2" => A::S2,
            o => return Err(v.err(format!("unknown AClass `{o}`"))),
        })
    }
}
impl FromIr for amdgpu_safeasm::ASrc {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        match v.name()? {
            "Reg" => Ok(amdgpu_safeasm::ASrc::Reg(u32::from_ir(v.item(0)?)?)),
            "Imm" => Ok(amdgpu_safeasm::ASrc::Imm(i64::from_ir(v.item(0)?)?)),
            o => Err(v.err(format!("unknown ASrc `{o}`"))),
        }
    }
}
impl FromIr for amdgpu_safeasm::AsmPort {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        v.expect_struct("AsmPort")?;
        Ok(amdgpu_safeasm::AsmPort { reg: FromIr::from_ir(v.field("reg")?)?, class: FromIr::from_ir(v.field("class")?)?, mutable: FromIr::from_ir(v.field("mutable")?)? })
    }
}
impl FromIr for amdgpu_safeasm::AInst {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        v.expect_struct("AInst")?;
        let op: String = FromIr::from_ir(v.field("op")?)?;
        // the op name is a `&'static str` in the block: interned for the life of the process (one leak per distinct parsed name)
        let op: &'static str = Box::leak(op.into_boxed_str());
        Ok(amdgpu_safeasm::AInst { op, dst: FromIr::from_ir(v.field("dst")?)?, srcs: FromIr::from_ir(v.field("srcs")?)?, imm: FromIr::from_ir(v.field("imm")?)? })
    }
}
impl FromIr for amdgpu_safeasm::AsmBlock {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        v.expect_struct("AsmBlock")?;
        Ok(amdgpu_safeasm::AsmBlock::from_parts(
            FromIr::from_ir(v.field("name")?)?,
            FromIr::from_ir(v.field("regs")?)?,
            FromIr::from_ir(v.field("alias")?)?,
            FromIr::from_ir(v.field("insts")?)?,
            FromIr::from_ir(v.field("inputs")?)?,
            FromIr::from_ir(v.field("outputs")?)?,
            FromIr::from_ir(v.field("exit_in_flight")?)?,
            FromIr::from_ir(v.field("loop_body")?)?,
            FromIr::from_ir(v.field("pinned")?)?,
        ))
    }
}
impl FromIr for std::sync::Arc<amdgpu_safeasm::AsmBlock> {
    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {
        Ok(std::sync::Arc::new(amdgpu_safeasm::AsmBlock::from_ir(v)?))
    }
}
impl IrIds for std::sync::Arc<amdgpu_safeasm::AsmBlock> {
    fn ids(&mut self, _: &mut dyn FnMut(&mut ValueId), _: &mut dyn FnMut(&mut VarId)) {}
}

// ---------------------------------------------------------------------------------------------
// lexer + value parser
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Ident(String),
    Num(String),
    Str(String),
    /// `%v12`
    ValName(u32),
    P(char),
}

struct Lexer {
    toks: Vec<(Tok, usize)>,
    pos: usize,
}

fn lex(src: &str) -> Result<Vec<(Tok, usize)>, IrErr> {
    let b: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    let mut line = 1usize;
    while i < b.len() {
        let c = b[i];
        if c == '\n' {
            line += 1;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '#' {
            // comment to end of line
            while i < b.len() && b[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '"' {
            let mut s = String::new();
            i += 1;
            loop {
                if i >= b.len() {
                    return Err(IrErr(format!("line {line}: unterminated string")));
                }
                let d = b[i];
                if d == '"' {
                    i += 1;
                    break;
                }
                if d == '\\' {
                    i += 1;
                    let e = *b.get(i).ok_or_else(|| IrErr(format!("line {line}: bad escape")))?;
                    match e {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '0' => s.push('\0'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        '\'' => s.push('\''),
                        'u' => {
                            // \u{XXXX}
                            let close = (i..b.len()).find(|&k| b[k] == '}').ok_or_else(|| IrErr(format!("line {line}: bad \\u escape")))?;
                            let hex: String = b[i + 2..close].iter().collect();
                            let cp = u32::from_str_radix(&hex, 16).map_err(|_| IrErr(format!("line {line}: bad \\u escape")))?;
                            s.push(char::from_u32(cp).ok_or_else(|| IrErr(format!("line {line}: bad \\u escape")))?);
                            i = close;
                        }
                        other => return Err(IrErr(format!("line {line}: unknown escape \\{other}"))),
                    }
                    i += 1;
                    continue;
                }
                if d == '\n' {
                    line += 1;
                }
                s.push(d);
                i += 1;
            }
            out.push((Tok::Str(s), line));
            continue;
        }
        if c == '%' && b.get(i + 1) == Some(&'v') {
            let mut j = i + 2;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            let n: String = b[i + 2..j].iter().collect();
            let id = n.parse::<u32>().map_err(|_| IrErr(format!("line {line}: bad value name")))?;
            out.push((Tok::ValName(id), line));
            i = j;
            continue;
        }
        let num_start = c.is_ascii_digit() || (c == '-' && b.get(i + 1).map_or(false, |d| d.is_ascii_digit() || *d == 'i'));
        if num_start {
            let mut j = i + 1;
            if c == '-' && b.get(i + 1) == Some(&'i') {
                // -inf
                j = i + 4;
                out.push((Tok::Num("-inf".into()), line));
                i = j;
                continue;
            }
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == '.' || b[j] == '_' || ((b[j] == '-' || b[j] == '+') && (b[j - 1] == 'e' || b[j - 1] == 'E') && !(b[i..j].iter().any(|x| *x == 'x')))) {
                j += 1;
            }
            let s: String = b[i..j].iter().collect();
            out.push((Tok::Num(s), line));
            i = j;
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let mut j = i + 1;
            while j < b.len() && (b[j].is_alphanumeric() || b[j] == '_') {
                j += 1;
            }
            let s: String = b[i..j].iter().collect();
            out.push((Tok::Ident(s), line));
            i = j;
            continue;
        }
        if "{}()[],:=".contains(c) {
            out.push((Tok::P(c), line));
            i += 1;
            continue;
        }
        return Err(IrErr(format!("line {line}: unexpected character `{c}`")));
    }
    Ok(out)
}

impl Lexer {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos).map(|x| &x.0)
    }
    fn peek2(&self) -> Option<&Tok> {
        self.toks.get(self.pos + 1).map(|x| &x.0)
    }
    fn line(&self) -> usize {
        self.toks.get(self.pos).or(self.toks.last()).map(|x| x.1).unwrap_or(0)
    }
    fn e(&self, m: impl std::fmt::Display) -> IrErr {
        IrErr(format!("line {}: {m}", self.line()))
    }
    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).map(|x| x.0.clone());
        self.pos += 1;
        t
    }
    fn expect_p(&mut self, c: char) -> Result<(), IrErr> {
        match self.next() {
            Some(Tok::P(x)) if x == c => Ok(()),
            other => {
                self.pos -= 1;
                Err(self.e(format!("expected `{c}`, found {}", other.map(|t| format!("{t:?}")).unwrap_or("end of input".into()))))
            }
        }
    }
    fn ident(&mut self) -> Result<String, IrErr> {
        match self.next() {
            Some(Tok::Ident(s)) => Ok(s),
            other => {
                self.pos -= 1;
                Err(self.e(format!("expected a name, found {other:?}")))
            }
        }
    }
    fn keyword(&mut self, k: &str) -> Result<(), IrErr> {
        let s = self.ident()?;
        if s == k {
            Ok(())
        } else {
            self.pos -= 1;
            Err(self.e(format!("expected `{k}`, found `{s}`")))
        }
    }
    /// One `Debug` value.
    fn value(&mut self) -> Result<IrVal, IrErr> {
        match self.next() {
            Some(Tok::Num(s)) => Ok(IrVal::Num(s)),
            Some(Tok::Str(s)) => Ok(IrVal::Str(s)),
            Some(Tok::P('[')) => {
                let items = self.items(']')?;
                Ok(IrVal::List(items))
            }
            Some(Tok::P('(')) => {
                let items = self.items(')')?;
                Ok(IrVal::Tuple(None, items))
            }
            Some(Tok::Ident(n)) => match self.peek() {
                Some(Tok::P('(')) => {
                    self.pos += 1;
                    let items = self.items(')')?;
                    Ok(IrVal::Tuple(Some(n), items))
                }
                // `Name { field: ..` (a struct value) - but not `if Val(3) {` (a block): a struct brace is followed by `ident :` or `}`
                Some(Tok::P('{')) if matches!((self.toks.get(self.pos + 1).map(|x| &x.0), self.toks.get(self.pos + 2).map(|x| &x.0)), (Some(Tok::Ident(_)), Some(Tok::P(':'))) | (Some(Tok::P('}')), _)) && n.chars().next().map_or(false, |c| c.is_uppercase()) => {
                    self.pos += 1;
                    let mut fs = Vec::new();
                    loop {
                        if let Some(Tok::P('}')) = self.peek() {
                            self.pos += 1;
                            break;
                        }
                        let f = self.ident()?;
                        self.expect_p(':')?;
                        let v = self.value()?;
                        fs.push((f, v));
                        match self.next() {
                            Some(Tok::P(',')) => {}
                            Some(Tok::P('}')) => break,
                            other => return Err(self.e(format!("expected `,` or `}}` in `{n}`, found {other:?}"))),
                        }
                    }
                    Ok(IrVal::Struct(n, fs))
                }
                _ => Ok(IrVal::Ident(n)),
            },
            other => Err(self.e(format!("expected a value, found {}", other.map(|t| format!("{t:?}")).unwrap_or("end of input".into())))),
        }
    }
    fn items(&mut self, close: char) -> Result<Vec<IrVal>, IrErr> {
        let mut v = Vec::new();
        loop {
            if self.peek() == Some(&Tok::P(close)) {
                self.pos += 1;
                return Ok(v);
            }
            v.push(self.value()?);
            match self.next() {
                Some(Tok::P(',')) => {}
                Some(Tok::P(c)) if c == close => return Ok(v),
                other => return Err(self.e(format!("expected `,` or `{close}`, found {other:?}"))),
            }
        }
    }
    fn typed<T: FromIr>(&mut self) -> Result<T, IrErr> {
        let line = self.line();
        let v = self.value()?;
        T::from_ir(&v).map_err(|e| IrErr(format!("line {line}: {}", e.0)))
    }
}

/// Decode one `Debug`-spelled value of type `T` (e.g. an `Op`, a `Contract`, a safe-asm `AsmBlock`).
pub fn parse_value<T: FromIr>(src: &str) -> Result<T, IrErr> {
    let mut lx = Lexer { toks: lex(src)?, pos: 0 };
    let v: T = lx.typed()?;
    if lx.peek().is_some() {
        return Err(lx.e("trailing text after the value"));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------------------------
// printer
// ---------------------------------------------------------------------------------------------

/// The `irv1` text of `f` (complete: `parse(&print(f))` reproduces `f` exactly).
pub fn print(f: &Function) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "irv1");
    let _ = writeln!(s, "dsl {:?}", f.name);
    let _ = writeln!(s, "kernarg_size {}  workgroup {}  lds {}  value_count {}", f.kernarg_size, f.workgroup_size, f.lds_bytes, f.value_tys.len());
    let _ = writeln!(s, "profile {:?}  lower_opts {:?}  sched_opts {:?}", f.compile_profile, f.lower_opts, f.sched_opts);
    if f.exact {
        let _ = writeln!(s, "exact");
    }
    if let Some(m) = f.fp_mode {
        let _ = writeln!(s, "fp_mode {m:?}");
    }
    let _ = writeln!(s, "var_tys {:?}", f.var_tys);
    for a in &f.args {
        let _ = writeln!(s, "arg {a:?}");
    }
    for a in &f.agpr {
        let _ = writeln!(s, "agpr {a:?}");
    }
    for r in &f.reg_plan {
        let _ = writeln!(s, "reg_plan {r:?}");
    }
    for r in &f.regions {
        let _ = writeln!(s, "region_decl {r:?}");
    }
    // the kernel plan (docs/two-weavers.md), one item per line; absent for kernels without a plan
    if let Some(p) = &f.plan {
        if let Some(c) = &p.cta {
            let _ = writeln!(s, "plan_cta {c:?}");
        }
        for x in &p.roles {
            let _ = writeln!(s, "plan_role {x:?}");
        }
        for x in &p.rings {
            let _ = writeln!(s, "plan_ring {x:?}");
        }
        for x in &p.phases {
            let _ = writeln!(s, "plan_phase {x:?}");
        }
        for x in &p.units {
            let _ = writeln!(s, "plan_unit {x:?}");
        }
        for x in &p.deps {
            let _ = writeln!(s, "plan_dep {x:?}");
        }
        for x in &p.handoffs {
            let _ = writeln!(s, "plan_handoff {x:?}");
        }
        for x in &p.sync {
            let _ = writeln!(s, "plan_sync {x:?}");
        }
        for x in &p.reg_constraints {
            let _ = writeln!(s, "plan_reg {x:?}");
        }
        let _ = writeln!(s, "plan_hints {:?}", p.hints);
    }
    let _ = writeln!(s, "contract {:#?}", f.contract);
    // types of values that no `Let` defines
    let mut defined = vec![false; f.value_tys.len()];
    for st in f.stmts() {
        if let Stmt::Let { dst, .. } = st {
            if let Some(d) = defined.get_mut(*dst as usize) {
                *d = true;
            }
        }
    }
    let _ = write!(s, "values {{");
    for (i, d) in defined.iter().enumerate() {
        if !d {
            let _ = write!(s, " %v{i}: {:?}", f.value_tys[i]);
        }
    }
    let _ = writeln!(s, " }}");
    let _ = writeln!(s, "body {{");
    print_nodes(f, f.body(), 1, &mut s);
    let _ = writeln!(s, "}}");
    s
}

fn print_nodes(f: &Function, nodes: &[Node], depth: usize, s: &mut String) {
    let pad = "  ".repeat(depth);
    for n in nodes {
        match n {
            Node::S(Stmt::Let { dst, op }) => {
                let ty = f.value_tys.get(*dst as usize).copied().unwrap_or(DType::I32);
                let _ = writeln!(s, "{pad}%v{dst}: {ty:?} = {op:?}");
            }
            Node::S(st) => {
                let _ = writeln!(s, "{pad}{st:?}");
            }
            Node::Loop { vars, init, body } => {
                let _ = writeln!(s, "{pad}loop vars {vars:?} init {init:?} {{");
                print_nodes(f, body, depth + 1, s);
                let _ = writeln!(s, "{pad}}}");
            }
            Node::If { cond, then, els } => {
                let _ = writeln!(s, "{pad}if {cond:?} {{");
                print_nodes(f, then, depth + 1, s);
                match els {
                    Some(e) => {
                        let _ = writeln!(s, "{pad}}} else {{");
                        print_nodes(f, e, depth + 1, s);
                        let _ = writeln!(s, "{pad}}}");
                    }
                    None => {
                        let _ = writeln!(s, "{pad}}}");
                    }
                }
            }
            Node::Region { kind, tile, slot, version, body } => {
                let _ = writeln!(s, "{pad}region {kind:?} tile {tile:?} slot {slot:?} version {version:?} {{");
                print_nodes(f, body, depth + 1, s);
                let _ = writeln!(s, "{pad}}}");
            }
            Node::Stage { name, contract, body } => {
                let _ = writeln!(s, "{pad}stage {name:?} {contract:?} {{");
                print_nodes(f, body, depth + 1, s);
                let _ = writeln!(s, "{pad}}}");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// parser
// ---------------------------------------------------------------------------------------------

/// Parse `irv1` text into a [`Function`], then validate it (`check`): types of `Let`s, single definitions, the IR verifier.
pub fn parse(src: &str) -> Result<Function, IrErr> {
    let f = parse_unchecked(src)?;
    let errs = check(&f);
    if errs.is_empty() {
        Ok(f)
    } else {
        Err(IrErr(errs.join("\n")))
    }
}

/// Parse without the semantic checks (syntax and decoding only).
pub fn parse_unchecked(src: &str) -> Result<Function, IrErr> {
    let mut lx = Lexer { toks: lex(src)?, pos: 0 };
    lx.keyword("irv1").map_err(|_| IrErr("line 1: not an irv1 file (the first token must be `irv1`)".into()))?;
    lx.keyword("dsl")?;
    let name: String = lx.typed()?;
    let mut kernarg_size = 0u32;
    let mut workgroup = 0u32;
    let mut lds = 0u32;
    let mut value_count: Option<usize> = None;
    let mut profile: Option<CompileProfile> = None;
    let mut lower_opts: Vec<String> = Vec::new();
    let mut sched_opts: Option<String> = None;
    let mut var_tys: Vec<DType> = Vec::new();
    let mut args: Vec<Arg> = Vec::new();
    let mut agpr: Vec<AgprMark> = Vec::new();
    let mut reg_plan: Vec<RegMark> = Vec::new();
    let mut regions: Vec<RegionDecl> = Vec::new();
    let mut plan: Option<crate::plan::KernelPlan> = None;
    let mut contract = Contract::default();
    let mut extra_tys: Vec<(u32, DType)> = Vec::new();
    let mut exact = false;
    let mut fp_mode: Option<(u8, u8, bool)> = None;
    loop {
        let k = lx.ident()?;
        match k.as_str() {
            "kernarg_size" => kernarg_size = lx.typed()?,
            "workgroup" => workgroup = lx.typed()?,
            "lds" => lds = lx.typed()?,
            "value_count" => value_count = Some(lx.typed()?),
            "profile" => profile = lx.typed()?,
            "lower_opts" => lower_opts = lx.typed()?,
            "sched_opts" => sched_opts = lx.typed()?,
            "var_tys" => var_tys = lx.typed()?,
            "exact" => exact = true,
            "fp_mode" => fp_mode = Some(lx.typed()?),
            "arg" => args.push(lx.typed()?),
            "agpr" => agpr.push(lx.typed()?),
            "reg_plan" => reg_plan.push(lx.typed()?),
            "region_decl" => regions.push(lx.typed()?),
            "plan_cta" => plan.get_or_insert_with(Default::default).cta = Some(lx.typed()?),
            "plan_role" => plan.get_or_insert_with(Default::default).roles.push(lx.typed()?),
            "plan_ring" => plan.get_or_insert_with(Default::default).rings.push(lx.typed()?),
            "plan_reg" => plan.get_or_insert_with(Default::default).reg_constraints.push(lx.typed()?),
            "plan_phase" => plan.get_or_insert_with(Default::default).phases.push(lx.typed()?),
            "plan_unit" => plan.get_or_insert_with(Default::default).units.push(lx.typed()?),
            "plan_dep" => plan.get_or_insert_with(Default::default).deps.push(lx.typed()?),
            "plan_handoff" => plan.get_or_insert_with(Default::default).handoffs.push(lx.typed()?),
            "plan_sync" => plan.get_or_insert_with(Default::default).sync.push(lx.typed()?),
            "plan_hints" => plan.get_or_insert_with(Default::default).hints = lx.typed()?,
            "contract" => contract = lx.typed()?,
            "values" => {
                lx.expect_p('{')?;
                loop {
                    match lx.next() {
                        Some(Tok::P('}')) => break,
                        Some(Tok::ValName(id)) => {
                            lx.expect_p(':')?;
                            extra_tys.push((id, lx.typed()?));
                        }
                        other => return Err(lx.e(format!("expected `%vN: Type` or `}}` in `values`, found {other:?}"))),
                    }
                }
            }
            "body" => break,
            other => return Err(lx.e(format!("unknown header item `{other}`"))),
        }
    }
    lx.expect_p('{')?;
    let mut tys: Vec<Option<DType>> = Vec::new();
    let body = parse_block(&mut lx, &mut tys)?;
    if lx.peek().is_some() {
        return Err(lx.e("text after the end of `body { .. }`"));
    }
    for (id, t) in extra_tys {
        let i = id as usize;
        if tys.len() <= i {
            tys.resize(i + 1, None);
        }
        if tys[i].is_some() {
            return Err(IrErr(format!("%v{id}: type given in `values` but the value is defined by a Let")));
        }
        tys[i] = Some(t);
    }
    let n = value_count.unwrap_or(tys.len()).max(tys.len());
    tys.resize(n, None);
    let mut value_tys = Vec::with_capacity(n);
    for (i, t) in tys.into_iter().enumerate() {
        value_tys.push(t.ok_or_else(|| IrErr(format!("%v{i}: no type (not defined by a Let and not listed in `values`)")))?);
    }
    let mut f = Function::from_parts(name, args, kernarg_size, workgroup, lds, contract, body, var_tys, value_tys, agpr, profile, lower_opts, sched_opts, reg_plan, regions);
    f.plan = plan;
    f.exact = exact;
    f.fp_mode = fp_mode;
    Ok(f)
}

fn parse_block(lx: &mut Lexer, tys: &mut Vec<Option<DType>>) -> Result<Vec<Node>, IrErr> {
    let mut out = Vec::new();
    loop {
        match lx.peek().cloned() {
            None => return Err(lx.e("missing `}`")),
            Some(Tok::P('}')) => {
                lx.pos += 1;
                return Ok(out);
            }
            Some(Tok::ValName(id)) => {
                lx.pos += 1;
                lx.expect_p(':')?;
                let ty: DType = lx.typed()?;
                lx.expect_p('=')?;
                let op: Op = lx.typed()?;
                let i = id as usize;
                if tys.len() <= i {
                    tys.resize(i + 1, None);
                }
                if tys[i].is_some() {
                    return Err(lx.e(format!("%v{id} is defined twice")));
                }
                tys[i] = Some(ty);
                out.push(Node::S(Stmt::Let { dst: id, op }));
            }
            Some(Tok::Ident(k)) if k == "loop" && lx.peek2() == Some(&Tok::Ident("vars".into())) => {
                lx.pos += 2;
                let vars: Vec<VarId> = lx.typed()?;
                lx.keyword("init")?;
                let init: Vec<Opnd> = lx.typed()?;
                lx.expect_p('{')?;
                let body = parse_block(lx, tys)?;
                out.push(Node::Loop { vars, init, body });
            }
            Some(Tok::Ident(k)) if k == "if" => {
                lx.pos += 1;
                let cond: Opnd = lx.typed()?;
                lx.expect_p('{')?;
                let then = parse_block(lx, tys)?;
                let els = if lx.peek() == Some(&Tok::Ident("else".into())) {
                    lx.pos += 1;
                    lx.expect_p('{')?;
                    Some(parse_block(lx, tys)?)
                } else {
                    None
                };
                out.push(Node::If { cond, then, els });
            }
            Some(Tok::Ident(k)) if k == "region" && matches!(lx.peek2(), Some(Tok::Ident(_))) => {
                lx.pos += 1;
                let kind: RegionKind = lx.typed()?;
                lx.keyword("tile")?;
                let tile: String = lx.typed()?;
                lx.keyword("slot")?;
                let slot: Opnd = lx.typed()?;
                lx.keyword("version")?;
                let version: Opnd = lx.typed()?;
                lx.expect_p('{')?;
                let body = parse_block(lx, tys)?;
                out.push(Node::Region { kind, tile, slot, version, body });
            }
            Some(Tok::Ident(k)) if k == "stage" && matches!(lx.peek2(), Some(Tok::Str(_))) => {
                lx.pos += 1;
                let name: String = lx.typed()?;
                let contract: Box<StageContract> = lx.typed()?;
                lx.expect_p('{')?;
                let body = parse_block(lx, tys)?;
                out.push(Node::Stage { name, contract, body });
            }
            Some(_) => {
                let line = lx.line();
                let st: Stmt = lx.typed()?;
                if matches!(
                    st,
                    Stmt::Let { .. }
                        | Stmt::BeginLoop { .. }
                        | Stmt::EndLoop
                        | Stmt::BeginIf { .. }
                        | Stmt::Else
                        | Stmt::EndIf
                        | Stmt::TileProduce { .. }
                        | Stmt::TileConsume { .. }
                        | Stmt::TileEnd
                        | Stmt::StageBegin { .. }
                        | Stmt::StageEnd
                ) {
                    return Err(IrErr(format!("line {line}: structure statements are written as `%vN: T = op`, `loop vars .. {{ }}`, `if .. {{ }}`, `region .. {{ }}`, `stage .. {{ }}`")));
                }
                out.push(Node::S(st));
            }
        }
    }
}

/// Semantic checks of a parsed function: every `Let`'s declared type is its op's result type, ids are in range, and the IR verifier
/// (`verify`: scoping, operand types, loop variables, region marks) passes.
pub fn check(f: &Function) -> Vec<String> {
    let mut errs = Vec::new();
    for st in f.stmts() {
        if let Stmt::Let { dst, op } = st {
            let want = op.result_ty();
            let got = f.value_tys.get(*dst as usize).copied();
            if got != Some(want) {
                errs.push(format!("%v{dst}: declared type {got:?} but `{}` produces {want:?}", short_op(op)));
            }
        }
    }
    let nv = f.value_tys.len() as u32;
    let nr = f.var_tys.len() as u32;
    let (mut bad_v, mut bad_r): (Vec<u32>, Vec<u32>) = (Vec::new(), Vec::new());
    for st in f.stmts() {
        let mut st = st.clone();
        st.ids(
            &mut |v| {
                if *v >= nv {
                    bad_v.push(*v);
                }
            },
            &mut |r| {
                if r.0 >= nr {
                    bad_r.push(r.0);
                }
            },
        );
    }
    errs.extend(bad_v.iter().map(|v| format!("%v{v} is out of range (value_count {nv})")));
    errs.extend(bad_r.iter().map(|r| format!("VarId({r}) has no type in var_tys")));
    if errs.is_empty() {
        errs.extend(verify(f));
    }
    errs
}

fn short_op(op: &Op) -> String {
    let s = format!("{op:?}");
    s.split(|c| c == '(' || c == ' ').next().unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------------------------
// alpha canonical form
// ---------------------------------------------------------------------------------------------

/// `f` with ValueIds / VarIds renumbered by first occurrence in program order (defs and uses), types permuted accordingly: two functions that
/// differ only in id allocation become identical. Annotations (AGPR marks, register plan) are renamed with the same maps.
pub fn alpha_canonical(f: &Function) -> Function {
    use std::collections::HashMap;
    let mut vmap: HashMap<ValueId, ValueId> = HashMap::new();
    let mut rmap: HashMap<u32, u32> = HashMap::new();
    let mut g = f.clone();
    {
        let mut fv = |v: &mut ValueId| {
            let n = vmap.len() as u32;
            *v = *vmap.entry(*v).or_insert(n);
        };
        let mut fr = |r: &mut VarId| {
            let n = rmap.len() as u32;
            r.0 = *rmap.entry(r.0).or_insert(n);
        };
        g.edit_stmts(|stmts| {
            for st in stmts.iter_mut() {
                st.ids(&mut fv, &mut fr);
            }
        });
        for a in g.agpr.iter_mut() {
            a.ids(&mut fv, &mut fr);
        }
        for r in g.reg_plan.iter_mut() {
            r.ids(&mut fv, &mut fr);
        }
    }
    // ids never mentioned keep their relative order after the mentioned ones
    let mut vt = vec![None; f.value_tys.len().max(vmap.len())];
    let mut next = vmap.len() as u32;
    for i in 0..f.value_tys.len() as u32 {
        let n = *vmap.entry(i).or_insert_with(|| {
            let x = next;
            next += 1;
            x
        });
        if let Some(slot) = vt.get_mut(n as usize) {
            *slot = Some(f.value_tys[i as usize]);
        }
    }
    g.value_tys = vt.into_iter().map(|t| t.unwrap_or(DType::I32)).collect();
    let mut rt = vec![DType::I32; f.var_tys.len().max(rmap.len())];
    let mut next = rmap.len() as u32;
    for i in 0..f.var_tys.len() as u32 {
        let n = *rmap.entry(i).or_insert_with(|| {
            let x = next;
            next += 1;
            x
        });
        if let Some(slot) = rt.get_mut(n as usize) {
            *slot = f.var_tys[i as usize];
        }
    }
    g.var_tys = rt;
    g
}
