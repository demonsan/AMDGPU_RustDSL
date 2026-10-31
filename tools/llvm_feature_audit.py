#!/usr/bin/env python3
"""LLVM AMDGPU CodeGen test audit: which of LLVM's own AMDGPU tests could be expressed with our Rust kernel DSL and compiled by our pipeline?

Usage:  python3 tools/llvm_feature_audit.py --tests <llvm>/test/CodeGen/AMDGPU --repo . --out tools/llvm_feature_audit.json

Per test (every *.ll and *.mir under the directory) the tool extracts
  (a) kind        : codegen (llc) | mir-pass (llc -run-pass/-stop-after/...) | pass (opt) | assembler (llvm-mc) | binary-tool | other
  (b) target      : per RUN line mcpu/triple -> class gfx950 | gfx94x | gfx9x | gfx10 | gfx11 | gfx12 | pre-gfx9 | r600 | unknown
  (c) opcodes     : mnemonics of the CHECK lines whose prefix belongs to a RUN line (asm), MIR opcodes for .mir tests, normalised (operands and
                    encoding suffixes stripped); split by target class
  (d) features    : IR / ABI features (calls, alloca/stack, atomics, intrinsics, debug info, exceptions, varargs, calling conventions, attributes, kernarg,
                    workgroup id, LDS, barriers/waitcnt, spills, subtarget -mattr, address spaces, types)
and classifies each test against OUR opcode universe (derived from crates/amdgpu-compiler/src/emit.rs and backend/llvm{20,24}.tbl):
  COVERED       gfx950-class codegen test, every required opcode exists in our compiler, no unsupported IR feature
  PARTIAL       gfx950-class codegen test, no hard-unsupported feature, but opcodes / features are missing (listed)
  OUT-OF-SCOPE  pass-level or non-codegen test, non-gfx950-class subtarget, or an IR feature the kernel DSL does not model (calls, stack, exceptions, debug info, ...)
See docs/llvm-feature-audit.md for the methodology and the honest meaning of these words.
"""
import argparse, collections, json, os, re, sys

# --------------------------------------------------------------------------------------------------------------------------------------------
# normalisation
MN_PREFIXES = ("v_", "s_", "buffer_", "global_", "ds_", "flat_", "scratch_", "image_", "tbuffer_", "exp")
ASM_SUFFIXES = ("_e32", "_e64", "_sdwa", "_dpp", "_vop3", "_vop3p", "_nsa", "_vi", "_si", "_ci", "_gfx9", "_gfx90a", "_gfx940", "_gfx10", "_gfx11", "_gfx12")
MIR_ADDR_SUFFIXES = ("_offen", "_offset", "_idxen", "_bothen", "_addr64", "_saddr", "_lds_offen", "_lds_offset", "_lds_idxen", "_lds_bothen", "_vbuffer")


def norm_mn(s, mir=False):
    s = s.lower()
    changed = True
    while changed:
        changed = False
        for suf in ASM_SUFFIXES + (MIR_ADDR_SUFFIXES if mir else ()):
            if s.endswith(suf) and len(s) > len(suf) + 2:
                s = s[: -len(suf)]
                changed = True
    return s


def is_mnemonic(tok):
    return tok.startswith(MN_PREFIXES) and re.fullmatch(r"[a-z][a-z0-9_]*", tok) is not None


MIR_OPC = re.compile(r"\b((?:V|S|BUFFER|GLOBAL|DS|FLAT|SCRATCH|IMAGE|TBUFFER|SI|G)_[A-Z0-9_a-z]+)\b")
MIR_EXCLUDE = ("VGPR_", "SGPR_", "AGPR_", "SReg_", "VReg_", "VS_", "SI_RETURN_TO_EPILOG_")  # register classes, not opcodes

# --------------------------------------------------------------------------------------------------------------------------------------------
# targets
R600 = {"r600", "rv710", "rv730", "rv770", "redwood", "cypress", "cayman", "barts", "juniper", "cedar", "sumo", "sumo2", "turks", "caicos", "evergreen", "rs880", "palm", "aruba", "hemlock"}
PRE9 = {"tahiti", "pitcairn", "verde", "oland", "hainan", "bonaire", "kabini", "kaveri", "hawaii", "mullins", "tonga", "iceland", "carrizo", "fiji", "stoney", "polaris10", "polaris11", "polaris12", "bristol", "generic", "generic-hsa", "amdgcn"}


def cls_of_cpu(cpu):
    c = (cpu or "").lower()
    if not c:
        return "pre-gfx9"  # amdgcn without -mcpu = the generic SI target
    if c in R600:
        return "r600"
    if c == "gfx950":
        return "gfx950"
    if c in ("gfx940", "gfx941", "gfx942", "gfx943"):
        return "gfx94x"
    m = re.match(r"gfx(\d+)", c)
    if m:
        n = m.group(1)
        if n.startswith("12"):
            return "gfx12"
        if n.startswith("11"):
            return "gfx11"
        if n.startswith("10"):
            return "gfx10"
        if n.startswith("9"):
            return "gfx9x"
        return "pre-gfx9"  # gfx6xx/gfx7xx/gfx8xx
    if c in PRE9:
        return "pre-gfx9"
    return "unknown"


GFX950_CLASS = ("gfx950", "gfx94x")


def effective_cpu(mcpu, triple):
    """-mcpu if given, else the cpu encoded in a versioned triple (amdgpu9.42 = gfx942, amdgpu10.30 = gfx1030, amdgpu6.00 = gfx600, amdgpu9.0a = gfx90a)."""
    if mcpu:
        return mcpu
    m = re.match(r"amdgpu(\d+)\.([0-9a-f]+)", (triple or "").lower())
    if m:
        return "gfx" + m.group(1) + m.group(2)
    return ""


def class_of_run(mcpu, triple):
    tr = triple or ""
    if tr.startswith("r600"):
        return "r600"
    return cls_of_cpu(effective_cpu(mcpu, triple))

# --------------------------------------------------------------------------------------------------------------------------------------------
# RUN lines
RUN_RE = re.compile(r"^\s*[;#]\s*RUN:\s?(.*)$")
DIRECTIVE_RE = re.compile(r"^\s*[;#]\s*(REQUIRES|UNSUPPORTED|XFAIL):\s*(.*)$")
CHECK_RE = re.compile(r"^\s*[;#]\s*([A-Za-z0-9_]+?)(-NEXT|-DAG|-LABEL|-SAME|-NOT|-EMPTY|-COUNT-\d+)?:\s*(.*)$")
PIPELINE_TOOLS = {"llc": "llc", "opt": "opt", "llvm-mc": "llvm-mc", "llvm-objdump": "binary-tool", "llvm-readobj": "binary-tool", "llvm-readelf": "binary-tool",
                  "llvm-dis": "other", "llvm-as": "other", "filecheck": "FileCheck", "ld.lld": "binary-tool", "lld": "binary-tool", "clang": "clang", "llvm-link": "other"}
MIR_PASS_FLAGS = ("-run-pass", "-stop-after", "-stop-before", "-start-before", "-start-after", "-passes=", "-print-after", "-mir-")


def logical_runs(lines):
    runs, cur = [], None
    for ln in lines:
        m = RUN_RE.match(ln)
        if not m:
            cur = cur if cur is None else cur
            continue
        body = m.group(1).rstrip()
        cont = body.endswith("\\")
        if cont:
            body = body[:-1]
        if cur is None:
            cur = body
        else:
            cur += " " + body
        if not cont:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    return runs


def parse_run(run):
    """-> dict(kind, mcpu, triple, mattr, prefixes, globalisel)"""
    segs = [s.strip() for s in re.split(r"\s\|\s|\|(?=\s)", run)]
    out = {"tools": [], "kind": "other", "mcpu": None, "triple": None, "mattr": [], "prefixes": [], "globalisel": False, "o0": False}
    kinds = []
    for seg in segs:
        toks = seg.split()
        while toks and toks[0] in ("not", "env"):
            toks = toks[1:]
        if not toks:
            continue
        tool = os.path.basename(toks[0]).lower()
        tool = PIPELINE_TOOLS.get(tool, "other")
        out["tools"].append(tool)
        if tool == "llc":
            kinds.append("mir-pass" if any(f in seg for f in MIR_PASS_FLAGS) else "codegen")
            for key, rx in (("mcpu", r"-mcpu=([^\s]+)"), ("triple", r"-mtriple=([^\s]+)|-march=([^\s]+)")):
                mm = re.search(rx, seg)
                if mm:
                    out[key] = next(g for g in mm.groups() if g)
            for mm in re.finditer(r"-mattr=([^\s]+)", seg):
                out["mattr"] += [a for a in mm.group(1).strip("\"'").split(",") if a]
            if "-global-isel" in seg or "-new-reg-bank-select" in seg:
                out["globalisel"] = True
            if "-O0" in seg:
                out["o0"] = True
        elif tool == "opt":
            kinds.append("pass")
            mm = re.search(r"-mcpu=([^\s]+)", seg)
            if mm and not out["mcpu"]:
                out["mcpu"] = mm.group(1)
            mm = re.search(r"-mtriple=([^\s]+)", seg)
            if mm and not out["triple"]:
                out["triple"] = mm.group(1)
        elif tool == "llvm-mc":
            kinds.append("assembler")
            mm = re.search(r"-mcpu=([^\s]+)", seg)
            if mm and not out["mcpu"]:
                out["mcpu"] = mm.group(1)
            mm = re.search(r"-triple[= ]([^\s]+)|-mtriple=([^\s]+)", seg)
            if mm and not out["triple"]:
                out["triple"] = next(g for g in mm.groups() if g)
            for mm in re.finditer(r"-mattr=([^\s]+)", seg):
                out["mattr"] += [a for a in mm.group(1).strip("\"'").split(",") if a]
        elif tool == "binary-tool":
            kinds.append("binary-tool")
        elif tool == "FileCheck":
            pf = []
            for mm in re.finditer(r"--?check-prefix(?:es)?[= ]([^\s]+)", seg):
                pf += mm.group(1).strip("\"'").split(",")
            out["prefixes"] += pf if pf else ["CHECK"]
        elif tool == "clang":
            kinds.append("other")
    for k in ("codegen", "mir-pass", "pass", "assembler", "binary-tool"):
        if k in kinds:
            out["kind"] = k
            break
    return out


def triple_os(triple):
    t = (triple or "").lower()
    if "amdhsa" in t:
        return "hsa"
    if "mesa" in t:
        return "mesa3d"
    if "amdpal" in t:
        return "pal"
    if t.startswith("r600"):
        return "r600"
    return "none"

# --------------------------------------------------------------------------------------------------------------------------------------------
# IR features
INTR_RE = re.compile(r"@(llvm\.[A-Za-z0-9_.]+)")
CALL_RE = re.compile(r"\b(?:call|invoke|callbr)\b[^\n@%]*?(?:@([\w.$\"-]+)|(%[\w.$\"-]+))")
CC_RE = re.compile(r"\bdefine\s+((?:[\w]+\s+)*?)(amdgpu_kernel|amdgpu_ps|amdgpu_vs|amdgpu_gs|amdgpu_hs|amdgpu_es|amdgpu_cs|amdgpu_ls|amdgpu_gfx|amdgpu_cs_chain_preserve|amdgpu_cs_chain|fastcc|coldcc|ccc|spir_func|spir_kernel|cc\s+\d+|webkit_jscc|anyregcc|preserve_\w+)\b")
ATTR_RE = re.compile(r"\"([a-z][a-z0-9_.-]+)\"(?:=\"([^\"]*)\")?")


def ir_features(text):
    f = set()
    intr = collections.OrderedDict()
    code = "\n".join(re.sub(r";.*$", "", l) for l in text.splitlines() if not l.lstrip().startswith(";"))
    for m in INTR_RE.finditer(code):
        intr[m.group(1)] = 1
    for m in CALL_RE.finditer(code):
        callee, ind = m.group(1), m.group(2)
        if ind and not callee:
            f.add("indirect_call")
        elif callee and not callee.startswith("llvm."):
            f.add("call_nonintrinsic")
    if re.search(r"\binvoke\b|\blandingpad\b|\bpersonality\b|\bresume\b|\bcleanuppad\b", code):
        f.add("exceptions")
    if re.search(r"\balloca\b", code):
        f.add("alloca")
        if re.search(r"\balloca\s+[^,\n]+,\s*(?:i\d+\s+)?%", code):
            f.add("dynamic_alloca")
    if "llvm.stacksave" in code or "llvm.stackrestore" in code or "llvm.amdgcn.dynamic.alloca" in code:
        f.add("stacksave")
    for kw, tag in ((r"\batomicrmw\b", "atomicrmw"), (r"\bcmpxchg\b", "cmpxchg"), (r"\bfence\b", "fence"), (r"\bload\s+atomic\b", "atomic_load"), (r"\bstore\s+atomic\b", "atomic_store")):
        if re.search(kw, code):
            f.add(tag)
    if re.search(r"!dbg\b|\bDICompileUnit\b|llvm\.dbg\.", text):
        f.add("debug_info")
    if re.search(r"\.\.\.\)|\bva_arg\b|llvm\.va_", code):
        f.add("varargs")
    ccs = set()
    for m in CC_RE.finditer(code):
        ccs.add(m.group(2).replace(" ", ""))
    defs = re.findall(r"^\s*define\b[^\n]*", code, re.M)
    if any("amdgpu_kernel" not in d for d in defs):
        f.add("non_kernel_function")
    for d in defs:
        if "amdgpu_kernel" in d:
            mm = re.search(r"@[\w.$\"-]+\((.*)\)", d)
            if mm and mm.group(1).strip():
                f.add("kernarg")
    attrs = set()
    for m in re.finditer(r"^attributes\s+#\d+\s*=\s*\{([^}]*)\}", code, re.M):
        for a in ATTR_RE.finditer(m.group(1)):
            attrs.add(a.group(1))
    spaces = sorted(set(re.findall(r"addrspace\((\d+)\)", code)), key=int)
    if "3" in spaces:
        f.add("lds")
    if "5" in spaces or "alloca" in f:
        f.add("private_memory")
    if "3" in spaces or "4" in spaces:
        pass
    names = list(intr)
    if any("workgroup.id" in n for n in names):
        f.add("workgroup_id")
    if any("workitem.id" in n for n in names):
        f.add("workitem_id")
    if any("dispatch.ptr" in n or "implicitarg" in n or "kernarg.segment" in n or "queue.ptr" in n or "dispatch.id" in n for n in names):
        f.add("abi_pointers")
    if any(n.endswith("s.barrier") or ".barrier" in n for n in names):
        f.add("barrier")
    if any("s.waitcnt" in n or "s.sleep" in n or "s.setprio" in n for n in names):
        f.add("waitcnt_or_sched_intrinsic")
    types = set()
    for tag, rx in (("f64", r"\bdouble\b"), ("f16", r"\bhalf\b"), ("bf16", r"\bbfloat\b"), ("i128", r"\bi128\b"), ("i64", r"\bi64\b"), ("i16", r"\bi16\b"), ("i8", r"\bi8\b"), ("vector", r"<\d+ x ")):
        if re.search(rx, code):
            types.add(tag)
    if re.search(r"\bi(?:256|512|1024)\b", code):
        types.add("wide_int")
    return sorted(f), names, sorted(ccs), sorted(attrs), spaces, sorted(types)


def mir_features(text):
    f = set()
    if re.search(r"\bSI_CALL\b|\bS_SWAPPC_B64\b|\bSI_TCRETURN|\bSI_CALL_ISEL\b|\bADJCALLSTACK", text):
        f.add("call_nonintrinsic")
    if re.search(r"hasCalls:\s*true", text):
        f.add("call_nonintrinsic")
    if re.search(r"^stack:\s*$|^\s+- \{ id: \d+, name:.*type: (?:default|spill-slot)", text, re.M):
        f.add("stack_objects")
    if re.search(r"isEntryFunction:\s*true", text):
        f.add("kernel_entry")
    elif re.search(r"machineFunctionInfo:", text):
        f.add("non_kernel_function")
    if re.search(r"\bG_[A-Z0-9_]+\b", text):
        f.add("generic_mir")
    if re.search(r"\bDBG_VALUE\b|debug-location|!DILocation", text):
        f.add("debug_info")
    if re.search(r"\bATOMIC|_ATOMIC_|\bGLOBAL_ATOMIC|\bDS_.*_RTN|\bBUFFER_ATOMIC", text):
        f.add("atomics")
    return sorted(f)

# --------------------------------------------------------------------------------------------------------------------------------------------
def analyse(path, rel):
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()
    is_mir = path.endswith(".mir")
    rec = {"path": rel, "ext": "mir" if is_mir else "ll", "dir": os.path.dirname(rel) or ".", "lines": len(lines)}
    runs = [parse_run(r) for r in logical_runs(lines)]
    rec["n_run"] = len(runs)
    kinds = collections.OrderedDict((r["kind"], 1) for r in runs)
    order = ("codegen", "mir-pass", "pass", "assembler", "binary-tool", "other")
    rec["kinds"] = [k for k in order if k in kinds]
    rec["kind"] = rec["kinds"][0] if rec["kinds"] else "other"
    rec["globalisel"] = any(r["globalisel"] for r in runs)
    mcpus, classes, triples, mattrs = [], [], [], []
    for r in runs:
        if r["kind"] in ("codegen", "mir-pass", "assembler", "pass"):
            c = effective_cpu(r["mcpu"], r["triple"])
            tr = (r["triple"] or "")
            cl = class_of_run(r["mcpu"], r["triple"])
            mcpus.append(c or "(default amdgcn = SI generic)")
            classes.append(cl)
            triples.append(tr)
            mattrs += r["mattr"]
    # target triple inside the file (when the RUN lines do not give one)
    mm = re.search(r"^target triple\s*=\s*\"([^\"]+)\"", text, re.M)
    if mm:
        triples.append(mm.group(1))
    rec["mcpus"] = sorted(set(mcpus))
    rec["target_classes"] = sorted(set(classes))
    rec["triples"] = sorted(set(t for t in triples if t))
    rec["os"] = sorted(set(triple_os(t) for t in triples if t)) or ["none"]
    rec["mattr"] = sorted(set(mattrs))
    # directives
    dirs = {"REQUIRES": [], "UNSUPPORTED": [], "XFAIL": []}
    for ln in lines:
        m = DIRECTIVE_RE.match(ln)
        if m:
            dirs[m.group(1)].append(m.group(2).strip())
    rec["requires"], rec["unsupported"], rec["xfail"] = dirs["REQUIRES"], dirs["UNSUPPORTED"], dirs["XFAIL"]
    # prefix -> class map (a CHECK line belongs to the classes of the RUN lines that use its prefix)
    pf_class = collections.defaultdict(set)
    for r in runs:
        if r["kind"] not in ("codegen", "mir-pass"):
            continue
        cl = class_of_run(r["mcpu"], r["triple"])
        for p in r["prefixes"]:
            pf_class[p].add(cl)
    asm_by_class = collections.defaultdict(set)
    mir_by_class = collections.defaultdict(set)
    spill = False
    for ln in lines:
        m = CHECK_RE.match(ln)
        if not m:
            continue
        pfx, mod, rest = m.group(1), m.group(2) or "", m.group(3)
        if pfx not in pf_class or mod == "-NOT":
            continue
        classes_here = pf_class[pfx]
        if re.search(r"[Ss]pill|ScratchSize|private_segment_fixed_size", rest):
            spill = True
        tok = re.match(r"([a-z][a-z0-9_]*)\b", rest.lstrip())
        if tok and is_mnemonic(tok.group(1)):
            for c in classes_here:
                asm_by_class[c].add(norm_mn(tok.group(1)))
        # MIR syntax inside CHECK lines
        for mm in MIR_OPC.finditer(rest):
            op = mm.group(1)
            if op.startswith(MIR_EXCLUDE) or not re.search(r"[A-Z]", op):
                continue
            if op.upper() == op or re.match(r"^[A-Z0-9_]+_[eE]\d+$", op) or re.match(r"^[A-Z0-9_]+(_[a-z0-9]+)*$", op):
                for c in classes_here:
                    mir_by_class[c].add(norm_mn(op, mir=True))
    if is_mir:
        # the MIR input itself defines the instructions the test needs
        body = "\n".join(l for l in lines if not l.lstrip().startswith(("#", ";")))
        ops = set()
        for mm in MIR_OPC.finditer(body):
            op = mm.group(1)
            if op.startswith(MIR_EXCLUDE) or op.startswith("G_") or op.startswith("SI_"):
                continue
            if op != op.upper() and not re.match(r"^[A-Z0-9_]+_[eE]\d+$", op):
                continue
            ops.add(norm_mn(op, mir=True))
        allc = sorted(set(classes)) or ["unknown"]
        for c in allc:
            mir_by_class[c] |= ops
    rec["opcodes_by_class"] = {c: sorted(asm_by_class[c] | mir_by_class[c]) for c in sorted(set(asm_by_class) | set(mir_by_class))}
    union = set()
    for v in rec["opcodes_by_class"].values():
        union |= set(v)
    rec["opcodes"] = sorted(union)
    if is_mir:
        feats = mir_features(text)
        rec["intrinsics"], rec["calling_conventions"], rec["attributes"], rec["address_spaces"], rec["types"] = [], [], [], [], []
    else:
        feats, intr, ccs, attrs, spaces, types = ir_features(text)
        rec["intrinsics"] = intr
        rec["calling_conventions"] = ccs
        rec["attributes"] = attrs
        rec["address_spaces"] = spaces
        rec["types"] = types
    if spill:
        feats = sorted(set(feats) | {"spill_or_scratch_checked"})
    if any(o.startswith("scratch_") for o in rec["opcodes"]):
        feats = sorted(set(feats) | {"scratch_instructions"})
    rec["features"] = feats
    rec["amdgcn_intrinsics"] = [i for i in rec["intrinsics"] if i.startswith("llvm.amdgcn.")]
    return rec

# --------------------------------------------------------------------------------------------------------------------------------------------
# our opcode universe
LIT_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
TOK_RE = re.compile(r"\b((?:v|s|buffer|global|ds|flat|scratch)_[a-z0-9_]*[a-z0-9])\b")


def our_universe(repo):
    emit = os.path.join(repo, "crates", "amdgpu-compiler", "src", "emit.rs")
    src = open(emit, errors="replace").read()
    arms = list(re.finditer(r"((?:MOp::\w+\s*\|\s*)*MOp::\w+)\s*(?:if\b[^=]*?)?=>", src))
    by_mop = {}
    for i, m in enumerate(arms):
        end = arms[i + 1].start() if i + 1 < len(arms) else len(src)
        block = src[m.end():end]
        toks = set()
        for lit in LIT_RE.findall(block):
            for t in TOK_RE.findall(lit):
                toks.add(norm_mn(t))
        names = re.findall(r"MOp::(\w+)", m.group(1))
        for n in names:
            by_mop.setdefault(n, set()).update(toks)
    # table-derived MIR opcode names (both dialects)
    tbl_ops = {}
    for tbl in ("llvm20.tbl", "llvm24.tbl"):
        p = os.path.join(repo, "crates", "amdgpu-compiler", "src", "backend", tbl)
        rows = set()
        for ln in open(p, errors="replace"):
            if ln.startswith("#") or "|" not in ln:
                continue
            key = ln.split("|")[0].strip().split(":")[0]
            for mm in re.finditer(r"\b((?:V|S|BUFFER|GLOBAL|DS|FLAT|SCRATCH)_[A-Z0-9_a-z]+)\b", ln):
                op = mm.group(1)
                if op.startswith(MIR_EXCLUDE):
                    continue
                tbl_ops.setdefault(key, set()).add(norm_mn(op, mir=True))
            rows.add(key)
        tbl_ops["__rows_" + tbl] = rows
    mnemonics = set()
    for v in by_mop.values():
        mnemonics |= v
    tbl_mn = set()
    for k, v in tbl_ops.items():
        if not k.startswith("__rows_"):
            tbl_mn |= v
    # control-flow / terminator mnemonics are emitted by render_term / terminators, not in match arms
    for extra in ("s_branch", "s_cbranch_vccnz", "s_cbranch_vccz", "s_cbranch_scc0", "s_cbranch_scc1", "s_endpgm", "s_nop", "s_barrier", "s_waitcnt", "s_setprio"):
        mnemonics.add(extra)
    mops = sorted(set(by_mop) | set(k for k in tbl_ops if not k.startswith("__rows_")))
    return {
        "n_mops": len(mops),
        "mop_to_mnemonics": {k: sorted(v) for k, v in sorted(by_mop.items())},
        "emit_mnemonics": sorted(mnemonics),
        "table_mir_opcodes_normalised": sorted(tbl_mn),
        "rows_llvm20": len(tbl_ops.get("__rows_llvm20.tbl", ())),
        "rows_llvm24": len(tbl_ops.get("__rows_llvm24.tbl", ())),
        "universe": sorted(mnemonics | tbl_mn),
    }

# Intrinsics the kernel DSL models (directly or through a DSL op): informational; they do not change the opcode-level status
MODELLED_INTRINSIC_PREFIXES = (
    "llvm.amdgcn.workitem.id.x", "llvm.amdgcn.workgroup.id.", "llvm.amdgcn.s.barrier", "llvm.amdgcn.readfirstlane",
    "llvm.amdgcn.mfma.f32.32x32x8f16", "llvm.amdgcn.mfma.f32.32x32x64.f8f6f4", "llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4", "llvm.amdgcn.smfmac.f32.16x16x128",
    "llvm.amdgcn.mfma.f32.16x16x16f16", "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8", "llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8",
    "llvm.amdgcn.permlane32.swap", "llvm.amdgcn.ds.read.tr8.b64", "llvm.amdgcn.global.load.lds", "llvm.amdgcn.raw.buffer.load", "llvm.amdgcn.raw.buffer.store",
    "llvm.amdgcn.raw.ptr.buffer.load", "llvm.amdgcn.raw.ptr.buffer.store", "llvm.amdgcn.s.waitcnt", "llvm.amdgcn.cvt.pknorm.u16", "llvm.amdgcn.cvt.scalef32.pk.fp8.bf16",
    "llvm.amdgcn.exp2", "llvm.amdgcn.log", "llvm.amdgcn.rcp", "llvm.amdgcn.perm", "llvm.amdgcn.s.setprio",
)
HARD_OOS_FEATURES = ("call_nonintrinsic", "indirect_call", "exceptions", "alloca", "dynamic_alloca", "stacksave", "debug_info", "varargs", "non_kernel_function", "stack_objects", "private_memory")
PARTIAL_FEATURES = ("atomicrmw", "cmpxchg", "fence", "atomic_load", "atomic_store", "atomics", "f64", "wide_int", "scratch_instructions", "spill_or_scratch_checked")


def classify(rec, universe):
    uni = set(universe)
    reasons = []
    if rec["kind"] != "codegen":
        kind_reason = {"mir-pass": "MIR pass test (llc -run-pass/-stop-after): tests an LLVM machine pass, not instruction support", "pass": "IR pass test (opt)",
                       "assembler": "assembler/disassembler test (llvm-mc)", "binary-tool": "object/binary tool test", "other": "no codegen RUN line"}[rec["kind"]]
        # a MIR file that is lowered to asm by llc without a pass flag is codegen; anything else here is out of scope
        rec["status"], rec["reasons"] = "OUT-OF-SCOPE", [kind_reason]
        rec["missing_opcodes"] = sorted(set(rec["opcodes"]) - uni)
        return rec
    in_class = [c for c in rec["target_classes"] if c in GFX950_CLASS]
    ops = set()
    for c in (in_class or rec["target_classes"]):
        ops |= set(rec["opcodes_by_class"].get(c, ()))
    missing = sorted(ops - uni)
    rec["missing_opcodes"] = missing
    rec["opcodes_gfx950_class"] = sorted(ops) if in_class else []
    if not in_class:
        rec["status"] = "OUT-OF-SCOPE"
        rec["reasons"] = ["subtarget not gfx950-class: " + ",".join(rec["target_classes"] or ["unknown"])]
        return rec
    hard = [f for f in rec["features"] if f in HARD_OOS_FEATURES]
    if hard:
        rec["status"] = "OUT-OF-SCOPE"
        rec["reasons"] = ["IR feature not modelled by the kernel DSL: " + ",".join(hard)]
        return rec
    partial = [f for f in rec["features"] if f in PARTIAL_FEATURES]
    unmodelled = [i for i in rec["amdgcn_intrinsics"] if not i.startswith(MODELLED_INTRINSIC_PREFIXES)]
    rec["intrinsics_unmodelled"] = unmodelled
    if missing or partial:
        rec["status"] = "PARTIAL"
        if missing:
            reasons.append("missing opcodes: " + ",".join(missing))
        if partial:
            reasons.append("features only partly modelled: " + ",".join(partial))
        rec["reasons"] = reasons
        return rec
    rec["status"] = "COVERED"
    rec["reasons"] = (["no opcode checks in the gfx950-class RUN lines (vacuous)"] if not ops else []) + (["unmodelled amdgcn intrinsics (informational): " + ",".join(unmodelled[:6])] if unmodelled else [])
    rec["vacuous"] = not ops
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out", default="tools/llvm_feature_audit.json")
    ap.add_argument("--summary-out", default="")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    uni = our_universe(a.repo)
    recs = []
    for root, _, files in sorted(os.walk(a.tests)):
        for fn in sorted(files):
            if fn.endswith((".ll", ".mir")):
                p = os.path.join(root, fn)
                recs.append(classify(analyse(p, os.path.relpath(p, a.tests).replace(os.sep, "/")), uni["universe"]))
    summary = summarise(recs, uni, a.top)
    out = {"tests_dir": a.tests, "universe": uni, "summary": summary, "tests": recs}
    with open(a.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    sp = a.summary_out or os.path.splitext(a.out)[0] + "_summary.json"
    with open(sp, "w") as fh:
        json.dump({"universe": {k: v for k, v in uni.items() if k in ("n_mops", "rows_llvm20", "rows_llvm24")} | {"n_universe_mnemonics": len(uni["universe"])}, "summary": summary}, fh, indent=1)
    print(json.dumps(summary, indent=1)[:6000])


def summarise(recs, uni, top):
    C = collections.Counter
    s = {"n_tests": len(recs), "by_ext": C(r["ext"] for r in recs), "by_dir": C(r["dir"] for r in recs), "by_kind": C(r["kind"] for r in recs),
         "by_status": C(r["status"] for r in recs), "globalisel_tests": sum(1 for r in recs if r["globalisel"]),
         "tests_with_requires": sum(1 for r in recs if r["requires"]), "tests_with_unsupported": sum(1 for r in recs if r["unsupported"]), "tests_with_xfail": sum(1 for r in recs if r["xfail"])}
    tc = C()
    for r in recs:
        for c in (r["target_classes"] or ["unknown"]):
            tc[c] += 1
    s["by_target_class(any_run)"] = tc
    primary = C()
    for r in recs:
        cl = r["target_classes"]
        gfx = [c for c in cl if c in GFX950_CLASS]
        primary["gfx950-class" if gfx else ("+".join(cl) if len(cl) == 1 else ("mixed:" + "+".join(cl) if cl else "unknown"))] += 1
    s["by_target_primary"] = primary
    s["by_mcpu"] = C(m for r in recs for m in r["mcpus"]).most_common(60)
    s["by_status_x_kind"] = {k: C(r["status"] for r in recs if r["kind"] == k) for k in sorted(set(r["kind"] for r in recs))}
    s["by_status_x_ext"] = {k: C(r["status"] for r in recs if r["ext"] == k) for k in ("ll", "mir")}
    s["oos_reasons"] = C((r["reasons"][0].split(":")[0] if r["reasons"] else "") for r in recs if r["status"] == "OUT-OF-SCOPE").most_common(20)
    s["gfx950_class_codegen"] = C(r["status"] for r in recs if r["kind"] == "codegen" and any(c in GFX950_CLASS for c in r["target_classes"]))
    s["gfx950_only_codegen"] = C(r["status"] for r in recs if r["kind"] == "codegen" and "gfx950" in r["target_classes"])
    # features
    s["features"] = C(f for r in recs for f in r["features"]).most_common(60)
    s["calling_conventions"] = C(c for r in recs for c in r["calling_conventions"]).most_common(20)
    s["attributes"] = C(a for r in recs for a in r["attributes"]).most_common(40)
    s["address_spaces"] = C(a for r in recs for a in r["address_spaces"]).most_common(20)
    s["types"] = C(t for r in recs for t in r["types"]).most_common(20)
    s["mattr"] = C(a for r in recs for a in r["mattr"]).most_common(40)
    s["intrinsics_amdgcn_top"] = C(i for r in recs for i in r["amdgcn_intrinsics"]).most_common(40)
    s["n_distinct_amdgcn_intrinsics"] = len(set(i for r in recs for i in r["amdgcn_intrinsics"]))
    # opcodes
    uni_set = set(uni["universe"])
    allops = C(o for r in recs for o in r["opcodes"])
    s["n_distinct_opcodes_in_tests"] = len(allops)
    s["top_opcodes_in_tests"] = allops.most_common(top)
    def missing_rank(filter_fn, key="opcodes"):
        miss = collections.defaultdict(list)
        for r in recs:
            if not filter_fn(r):
                continue
            for o in r[key]:
                if o not in uni_set:
                    miss[o].append(r["path"])
        rank = sorted(miss.items(), key=lambda kv: -len(kv[1]))[:top]
        return [{"opcode": o, "tests": len(v), "examples": v[:5]} for o, v in rank]
    s["top_missing_opcodes_gfx950_class"] = missing_rank(lambda r: r["kind"] == "codegen" and any(c in GFX950_CLASS for c in r["target_classes"]), "opcodes_gfx950_class")
    s["top_missing_opcodes_all_codegen"] = missing_rank(lambda r: r["kind"] == "codegen")
    # opcode-level coverage ignoring target/features (how much of the ISA text of the tests exists in our compiler at all)
    cod = [r for r in recs if r["kind"] == "codegen" and r["opcodes"]]
    full = sum(1 for r in cod if all(o in uni_set for o in r["opcodes"]))
    s["codegen_tests_with_opcode_checks"] = len(cod)
    s["codegen_tests_all_opcodes_in_our_universe(any_target)"] = full
    s["opcodes_in_universe_that_tests_use"] = sorted(o for o in uni_set if o in allops)
    s["opcodes_in_tests_that_we_have"] = len([o for o in allops if o in uni_set])
    return s


if __name__ == "__main__":
    main()