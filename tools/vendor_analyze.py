#!/usr/bin/env python3
"""Structural analysis of an AMDGPU (gfx950) kernel disassembly for docs/vendor-spec.md.

Input: `llvm-objdump -d --mcpu=gfx950 --no-show-raw-insn <code-object>` output.
(Produce it with tools/vendor_disasm.sh, which also handles the .s/.co plumbing.)

Commands:
  summary                      instruction count, pipe classes, opcode histogram
  labels                       named-label regions (only present on the aiter vendor kernel)
  loops [--top N]              backward-branch loops with pipe-class and MFMA breakdown
  phases <label> [--seq]       split a label loop at s_barrier into phases with notable ops
  region <label> [<end-label>] all opcodes/sequence of one label region
  sync                         every s_barrier / s_setprio with its enclosing label
  branches                     all branches with target and direction (control skeleton)
  mfma                         all MFMA/SMFMAC instructions + counts
  lds                          all LDS instructions
  staging                      all VMEM instructions and M0 setups

This is the tool behind the phase/opcode/softmax/LDS/staging sections of docs/vendor-spec.md.
"""
import re
import sys
from collections import Counter, OrderedDict

LABEL_RE = re.compile(r"^([0-9a-fA-F]{6,16}) <(.+)>:")
INS_RE = re.compile(r"^\s+([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*//\s*([0-9a-fA-F]{6,16}):")

CONTROL = {
    "s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccnz", "s_cbranch_vccz",
    "s_cbranch_execz", "s_endpgm", "s_barrier", "s_setprio", "s_nop", "s_waitcnt", "s_sleep",
}


class Ins:
    __slots__ = ("addr", "mnem", "ops", "line")

    def __init__(self, addr, mnem, ops, line):
        self.addr, self.mnem, self.ops, self.line = addr, mnem, ops, line


def parse(path):
    ins, labels, label_at = [], OrderedDict(), {}
    with open(path, "r", errors="replace") as f:
        for ln, line in enumerate(f):
            m = LABEL_RE.match(line)
            if m:
                addr, name = int(m.group(1), 16), m.group(2)
                labels[name] = addr
                label_at[addr] = name
                continue
            m = INS_RE.match(line)
            if m:
                ins.append(Ins(int(m.group(3), 16), m.group(1), m.group(2).strip(), ln))
    ins.sort(key=lambda i: i.addr)
    return ins, labels, label_at


def cls(mnem):
    if mnem.startswith(("v_mfma", "v_smfmac")):
        return "MFMA"
    if mnem.startswith("ds_"):
        return "LDS"
    if mnem.startswith(("global_", "buffer_", "scratch_", "flat_")):
        return "VMEM"
    if mnem.startswith(("s_load", "s_buffer_load")):
        return "SMEM"
    if mnem.startswith("v_"):
        return "VALU"
    if mnem.startswith("s_"):
        return "CTRL" if mnem in CONTROL else "SALU"
    return "OTHER"


def hist(iregs):
    return Counter(i.mnem for i in iregs)


def fmt_hist(h, n=40):
    return "\n".join(f"{k:5d} {m}" for m, k in h.most_common(n))


def region(ins, a, b):
    return [i for i in ins if a <= i.addr < b]


def region_by_label(ins, labels, name, end_name=None):
    a = labels[name]
    if end_name:
        b = labels[end_name]
    else:
        nxt = [v for v in labels.values() if v > a]
        b = min(nxt) if nxt else max(i.addr for i in ins) + 1
    return region(ins, a, b)


def loop_of(ins, label_name, labels):
    """Region from a label to the backward branch that targets it again."""
    start = labels[label_name]
    idx0 = next(k for k, i in enumerate(ins) if i.addr == start)
    end = None
    for k in range(idx0 + 1, len(ins)):
        i = ins[k]
        if i.mnem.startswith(("s_branch", "s_cbranch")):
            tgt = i.ops.split()[0].strip() if i.ops else ""
            if tgt == label_name:
                end = k
    if end is None:
        nxt = [v for v in labels.values() if v > start]
        return region(ins, start, min(nxt) if nxt else ins[-1].addr + 1)
    return ins[idx0:end + 1]


def enclosing(labels, addr):
    name = None
    for n, a in sorted(labels.items(), key=lambda kv: kv[1]):
        if a <= addr:
            name = n
        else:
            break
    return name


def rle(iregs, limit=500):
    out, prev, n = [], None, 0
    for i in iregs:
        if i.mnem == prev:
            n += 1
            continue
        if prev is not None:
            out.append(prev + (f" x{n}" if n > 1 else ""))
        prev, n = i.mnem, 1
    if prev is not None:
        out.append(prev + (f" x{n}" if n > 1 else ""))
    return out[:limit]


def summary(ins, labels, label_at):
    print(f"instructions: {len(ins)}  labels: {len(labels)}")
    print("pipe classes:")
    for k, v in Counter(cls(i.mnem) for i in ins).most_common():
        print(f"  {k:6s} {v}")
    print("opcode histogram (top 50):")
    for m, k in hist(ins).most_common(50):
        print(f"  {k:5d} {m}")


def labels_cmd(ins, labels, label_at):
    items = sorted(labels.items(), key=lambda kv: kv[1])
    for k, (name, addr) in enumerate(items):
        end = items[k + 1][1] if k + 1 < len(items) else ins[-1].addr + 8
        r = region(ins, addr, end)
        cc = Counter(cls(i.mnem) for i in r)
        print(f"{addr:04x}  {end - addr:6d}B  {len(r):5d} ins  {name}")
        print("           " + " ".join(f"{a}={b}" for a, b in cc.most_common()))


def loops_cmd(ins, labels, label_at, top=20):
    backs = []
    for k, i in enumerate(ins):
        if i.mnem.startswith(("s_branch", "s_cbranch")):
            tgt = i.ops.split()[0].strip() if i.ops else ""
            if tgt in labels and labels[tgt] <= i.addr:
                idx0 = next(j for j, x in enumerate(ins) if x.addr == labels[tgt])
                if idx0 <= k:
                    backs.append((idx0, k, tgt))
    backs.sort(key=lambda t: t[1] - t[0], reverse=True)
    print(f"backward branches (loops): {len(backs)}")
    for idx0, k, tgt in backs[:top]:
        r = ins[idx0:k + 1]
        cc = Counter(cls(i.mnem) for i in r)
        mfma = Counter(i.mnem for i in r if cls(i.mnem) == "MFMA")
        print(f"  {labels[tgt]:04x}-{ins[k].addr:04x} ({len(r):4d} ins) -> {label_at.get(labels[tgt], hex(labels[tgt]))}  "
              + " ".join(f"{a}={b}" for a, b in cc.most_common())
              + (f"  MFMA[{', '.join(f'{m}x{c}' for m, c in mfma.items())}]" if mfma else ""))


def phases_cmd(ins, labels, label_at, name, seq=False):
    r = loop_of(ins, name, labels)
    print(f"loop {name}: {len(r)} instructions, {r[0].addr:04x}-{r[-1].addr:04x}")
    phases, cur = [], []
    for i in r:
        cur.append(i)
        if i.mnem == "s_barrier":
            phases.append(cur)
            cur = []
    if cur:
        phases.append(cur)
    print(f"barrier phases: {len(phases)}")
    for p, ph in enumerate(phases):
        if not ph:
            continue
        cc = Counter(cls(i.mnem) for i in ph)
        mf = Counter(i.mnem for i in ph if cls(i.mnem) == "MFMA")
        waits = [i for i in ph if i.mnem == "s_waitcnt"]
        prios = [i for i in ph if i.mnem == "s_setprio"]
        print(f"  phase {p}: {len(ph):4d} ins {ph[0].addr:04x}-{ph[-1].addr:04x} "
              + " ".join(f"{a}={b}" for a, b in cc.most_common()))
        if mf:
            print("      MFMA: " + ", ".join(f"{m}x{c}" for m, c in mf.items()))
        if prios:
            print("      setprio at: " + ", ".join(hex(i.addr) for i in prios))
        if waits:
            print("      waitcnt: " + ", ".join(i.ops.replace(" ", "") for i in waits))
        if seq:
            print("      seq: " + " | ".join(rle(ph)))


def region_cmd(ins, labels, label_at, name, end_name=None):
    r = region_by_label(ins, labels, name, end_name)
    print(f"region {name}: {len(r)} instructions")
    print("pipe classes: " + " ".join(f"{a}={b}" for a, b in Counter(cls(i.mnem) for i in r).most_common()))
    print("opcode histogram:")
    for line in fmt_hist(hist(r), 60).splitlines():
        print("    " + line)
    print("sequence: " + " | ".join(rle(r)))


def dump_cmd(ins, pred, title):
    print(f"=== {title}")
    for i in ins:
        if pred(i):
            print(f"{i.addr:04x}  {i.mnem:45s} {i.ops}")


def sync_cmd(ins, labels):
    print("=== s_barrier / s_setprio")
    for i in ins:
        if i.mnem in ("s_barrier", "s_setprio"):
            print(f"{i.addr:04x} {i.mnem:12s} {i.ops:6s} in {enclosing(labels, i.addr)}")
    waits = Counter(i.ops.replace(" ", "") for i in ins if i.mnem == "s_waitcnt")
    print("=== s_waitcnt histogram")
    for w, k in waits.most_common():
        print(f"  {k:5d} {w}")


def branches_cmd(ins, labels):
    print("=== branches (source -> target, direction, enclosing label)")
    for i in ins:
        if i.mnem.startswith(("s_branch", "s_cbranch")):
            tgt = i.ops.split()[0] if i.ops else ""
            ta = labels.get(tgt)
            direction = "BACK" if ta is not None and ta <= i.addr else "fwd "
            print(f"{i.addr:04x} {i.mnem:18s} -> {tgt:42s} {direction}  in {enclosing(labels, i.addr)}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    path = sys.argv[1]
    cmd = sys.argv[2] if len(sys.argv) > 2 else "summary"
    ins, labels, label_at = parse(path)
    if cmd == "summary":
        summary(ins, labels, label_at)
    elif cmd == "labels":
        labels_cmd(ins, labels, label_at)
    elif cmd == "loops":
        top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 20
        loops_cmd(ins, labels, label_at, top)
    elif cmd == "phases":
        phases_cmd(ins, labels, label_at, sys.argv[3], "--seq" in sys.argv)
    elif cmd == "region":
        region_cmd(ins, labels, label_at, sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif cmd == "sync":
        sync_cmd(ins, labels)
    elif cmd == "branches":
        branches_cmd(ins, labels)
    elif cmd == "mfma":
        dump_cmd(ins, lambda i: cls(i.mnem) == "MFMA", "MFMA / SMFMAC instructions")
        print("counts: " + ", ".join(f"{m}x{k}" for m, k in hist([i for i in ins if cls(i.mnem) == "MFMA"]).most_common()))
    elif cmd == "lds":
        dump_cmd(ins, lambda i: cls(i.mnem) == "LDS", "LDS instructions")
    elif cmd == "staging":
        dump_cmd(ins, lambda i: cls(i.mnem) == "VMEM", "VMEM instructions")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
