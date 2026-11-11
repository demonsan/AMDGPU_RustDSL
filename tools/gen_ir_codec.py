#!/usr/bin/env python3
"""Generates crates/amdgpu-dsl/src/ir_codec.rs (docs/ir-text.md): for every IR data type reachable from `Function`
  * `impl FromIr for T` - decoding of the Rust `Debug` spelling (the irv1 text uses `Debug` for every field value),
  * `impl IrIds for T` - visiting every ValueId / VarId inside (alpha renaming for structural equality),
  * an exhaustive `match` per enum (no wildcard), so adding / removing a variant or a field breaks the build until this file is regenerated.

usage: python3 tools/gen_ir_codec.py            (writes the file)
       python3 tools/gen_ir_codec.py --check    (exit 1 when the checked-in file is stale)
"""
import re, sys, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRCS = {
    "crate": os.path.join(ROOT, "crates/amdgpu-dsl/src/lib.rs"),
    "plan": os.path.join(ROOT, "crates/amdgpu-dsl/src/plan.rs"),
}
OUT = os.path.join(ROOT, "crates/amdgpu-dsl/src/ir_codec.rs")
# the IR types (data reachable from Function); FnBuilder / expr types are not data of a Function
TYPES = ["VarId", "DType", "Opnd", "Op", "Ext", "ArgExtent", "TileDecl", "BufRegion", "LastArriver", "WgMap", "WgWinner", "Contract", "StageVal", "StageScalar", "StageOutstanding", "StageResources", "StageContract",
         "Stmt", "Arg", "AgprTarget", "AgprMark", "RegRef", "RegMark", "RegionKind", "RegionDecl", "CompileProfile", "UnOp", "BinOp", "TriOp", "CmpPred", "CmpTy", "MemTy", "D16Ty", "MmaShape", "MmaTy", "MmaKind", "MmaScale", "CachePolicy", "CvtOp", "DppCtrl", "Dpp", "DppAluOp", "AtomicOp",
         # the kernel plan (plan.rs, docs/two-weavers.md)
         "KernelPlan", "CtaShape", "RoleDecl", "Cadence", "RingOwner", "RingDecl", "CrossCtaProtocol", "MemOrder", "RegConstraint", "Phase", "PhaseKind", "OrderSpec", "Budget", "M0Use", "UnitClass", "UnitDecl", "DepKind", "Dep", "RegFileKind", "Via", "HandOff", "SyncRule", "Hints"]


def strip_comments(s):
    s = re.sub(r"//[^\n]*", "", s)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"#\[[^\]]*\]", "", s)
    return s


def match_close(s, i):
    """index of the bracket closing the one at s[i]"""
    pairs = {"{": "}", "(": ")", "[": "]", "<": ">"}
    stack = [s[i]]
    j = i + 1
    while j < len(s):
        c = s[j]
        if c in "{([<":
            stack.append(c)
        elif c in "})]>":
            if c == "-" :
                pass
            if pairs[stack[-1]] == c:
                stack.pop()
                if not stack:
                    return j
        elif c == "-" and s[j + 1:j + 2] == ">":
            j += 2
            continue
        j += 1
    raise ValueError("unbalanced")


def split_top(s):
    out, depth, cur = [], 0, ""
    for c in s:
        if c in "{([<":
            depth += 1
        elif c in "})]>":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur)
    return [x.strip() for x in out if x.strip()]


def parse_fields(body):
    """struct-like body -> [(name, type)]"""
    res = []
    for f in split_top(body):
        f = re.sub(r"^pub(\([^)]*\))?\s+", "", f.strip())
        name, ty = f.split(":", 1)
        res.append((name.strip(), ty.strip()))
    return res


def parse_types(src):
    s = strip_comments(src)
    defs = {}
    for m in re.finditer(r"\bpub (enum|struct) (\w+)\s*([({;])", s):
        kind, name, br = m.group(1), m.group(2), m.group(3)
        if name not in TYPES:
            continue
        if br == ";":
            defs[name] = ("unit_struct", None)
            continue
        i = m.end() - 1
        j = match_close(s, i)
        body = s[i + 1:j]
        if kind == "struct":
            if br == "{":
                defs[name] = ("struct", parse_fields(body))
            else:
                defs[name] = ("tuple_struct", [re.sub(r"^pub\s+", "", x) for x in split_top(body)])
        else:
            variants = []
            k = 0
            b = body
            while k < len(b):
                mm = re.match(r"\s*(\w+)\s*", b[k:])
                if not mm:
                    k += 1
                    continue
                vname = mm.group(1)
                k += mm.end()
                if k < len(b) and b[k] == "{":
                    e = match_close(b, k)
                    variants.append((vname, "struct", parse_fields(b[k + 1:e])))
                    k = e + 1
                elif k < len(b) and b[k] == "(":
                    e = match_close(b, k)
                    variants.append((vname, "tuple", split_top(b[k + 1:e])))
                    k = e + 1
                else:
                    variants.append((vname, "unit", None))
                mm = re.match(r"\s*(=\s*[^,]+)?\s*,?", b[k:])
                k += mm.end() if mm else 1
            defs[name] = ("enum", variants)
    return defs


def ids_stmt(expr, ty):
    """code visiting the ids inside `expr` (a place expression of type `ty`)"""
    t = ty.replace(" ", "")
    if t == "ValueId":
        return f"fv(&mut {expr});"
    if t == "VarId":
        return f"fr(&mut {expr});"
    if t in ("Vec<ValueId>",) or re.fullmatch(r"\[ValueId;\d+\]", t):
        return f"for x in {expr}.iter_mut() {{ fv(x); }}"
    if t == "Option<ValueId>":
        return f"if let Some(x) = {expr}.as_mut() {{ fv(x); }}"
    if t in ("Vec<VarId>",) or re.fullmatch(r"\[VarId;\d+\]", t):
        return f"for x in {expr}.iter_mut() {{ fr(x); }}"
    if t == "Option<VarId>":
        return f"if let Some(x) = {expr}.as_mut() {{ fr(x); }}"
    if "ValueId" in t:
        raise SystemExit(f"unsupported ValueId container {ty}")
    return f"IrIds::ids(&mut {expr}, fv, fr);"


def gen(defs):
    L = []
    w = L.append
    w("// GENERATED by tools/gen_ir_codec.py - do not edit. Regenerate after changing an IR type (the exhaustive matches below fail to compile otherwise).")
    w("#![allow(clippy::all, unused_variables, non_snake_case)]")
    w("use crate::ir_text::{FromIr, IrErr, IrIds, IrVal};")
    w("use crate::plan::*;")
    w("use crate::*;")
    w("")
    for name in TYPES:
        kind, data = defs[name]
        # FromIr
        w(f"impl FromIr for {name} {{")
        w("    fn from_ir(v: &IrVal) -> Result<Self, IrErr> {")
        if kind == "struct":
            w(f"        v.expect_struct(\"{name}\")?;")
            w(f"        Ok({name} {{")
            for f, t in data:
                w(f"            {f}: FromIr::from_ir(v.field(\"{f}\")?)?,")
            w("        })")
        elif kind == "tuple_struct":
            w(f"        v.expect_tuple(\"{name}\", {len(data)})?;")
            w(f"        Ok({name}(" + ", ".join(f"FromIr::from_ir(v.item({k})?)?" for k in range(len(data))) + "))")
        elif kind == "unit_struct":
            w(f"        v.expect_unit(\"{name}\")?;")
            w(f"        Ok({name})")
        else:
            w("        match v.name()? {")
            for vn, vk, vd in data:
                if vk == "unit":
                    w(f"            \"{vn}\" => {{ v.expect_unit(\"{vn}\")?; Ok({name}::{vn}) }}")
                elif vk == "tuple":
                    w(f"            \"{vn}\" => {{ v.expect_tuple(\"{vn}\", {len(vd)})?; Ok({name}::{vn}(" + ", ".join(f"FromIr::from_ir(v.item({k})?)?" for k in range(len(vd))) + ")) }")
                else:
                    w(f"            \"{vn}\" => {{ v.expect_struct(\"{vn}\")?; Ok({name}::{vn} {{ " + ", ".join(f"{f}: FromIr::from_ir(v.field(\"{f}\")?)?" for f, _ in vd) + " }) }")
            w(f"            other => Err(v.err(format!(\"unknown {name} variant `{{other}}`\"))),")
            w("        }")
        w("    }")
        w("}")
        # IrIds
        w(f"impl IrIds for {name} {{")
        w("    fn ids(&mut self, fv: &mut dyn FnMut(&mut ValueId), fr: &mut dyn FnMut(&mut VarId)) {")
        if name == "VarId":
            w("        fr(self);")
        elif kind == "struct":
            for f, t in data:
                w("        " + ids_stmt(f"self.{f}", t))
        elif kind == "tuple_struct":
            for k, t in enumerate(data):
                w("        " + ids_stmt(f"self.{k}", t))
        elif kind == "unit_struct":
            pass
        else:
            w("        match self {")
            for vn, vk, vd in data:
                if vk == "unit":
                    w(f"            {name}::{vn} => {{}}")
                elif vk == "tuple":
                    binds = [f"x{k}" for k in range(len(vd))]
                    body = " ".join(ids_stmt(f"(*{b})", t) for b, t in zip(binds, vd))
                    w(f"            {name}::{vn}(" + ", ".join(binds) + f") => {{ {body} }}")
                else:
                    binds = [f for f, _ in vd]
                    body = " ".join(ids_stmt(f"(*{f})", t) for f, t in vd)
                    w(f"            {name}::{vn} {{ " + ", ".join(binds) + f" }} => {{ {body} }}")
            w("        }")
        w("    }")
        w("}")
        w("")
    return "\n".join(L) + "\n"


def main():
    src = open(SRCS["crate"], encoding="utf-8").read() + "\n" + open(SRCS["plan"], encoding="utf-8").read()
    defs = parse_types(src)
    missing = [t for t in TYPES if t not in defs]
    if missing:
        raise SystemExit(f"types not found: {missing}")
    if "--types" in sys.argv:
        ts = set()
        for k, dd in defs.values():
            if k == "struct":
                ts.update(t for _, t in dd)
            elif k == "tuple_struct":
                ts.update(dd)
            elif k == "enum":
                for vn, vk, vd in dd:
                    if vk == "struct":
                        ts.update(t for _, t in vd)
                    elif vk == "tuple":
                        ts.update(vd)
        print(sorted(ts))
        return
    code = gen(defs)
    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read().replace("\r\n", "\n") if os.path.exists(OUT) else ""
        if cur != code:
            print("ir_codec.rs is stale: run python3 tools/gen_ir_codec.py")
            sys.exit(1)
        print("ir_codec.rs up to date")
        return
    open(OUT, "w", encoding="utf-8", newline="\n").write(code)
    print(f"wrote {OUT}: {len(TYPES)} types")


main()
