#!/usr/bin/env python3
"""VALU-writes-VGPR -> v_readfirstlane-reads-it hazard probe (gfx950; docs/exact-mode.md section 5.2). LLVM (and our valu_writer_delay)
asks 2 wait states; the vendor FMHA uses 1 (`v_mov v184, s28 ; v_mov v185, 1.0 ; v_readfirstlane_b32 s28, v184`).

Each test writes a STALE value into vX long before, then a VALU writer puts the FRESH value into vX, N wait states (s_nop), then
`v_readfirstlane_b32 sY, vX`, and every lane stores sY. A stale read shows the stale value.

usage: THREADS=512 readfirstlane_hazard_probe.py gen <template.s> <out.s>  -> prints "<kernel> <bytes>"
       THREADS=512 readfirstlane_hazard_probe.py check <out.bin>
"""
import os, re, struct, sys

THREADS = int(os.environ.get("THREADS", "64"))
VREGS = ["v5", "v20", "v40", "v63"]
SREGS = ["s10", "s20", "s40"]
STALE = 0xDEAD
WRITERS = {
    # fresh value = 0x1000 + test index (an SGPR source, a literal, a VALU add)
    "mov_from_sgpr": lambda v, t: [f"s_mov_b32 s9, {0x1000 + t}", "s_nop 7", f"v_mov_b32_e32 {v}, s9"],
    "mov_literal": lambda v, t: [f"v_mov_b32_e32 {v}, {0x1000 + t}"],
    "add": lambda v, t: [f"v_mov_b32_e32 v8, {0x1000 + t - 1}", "s_nop 7", f"v_add_u32_e32 {v}, 1, v8"],
}
TESTS = []
for w in WRITERS:
    for i, v in enumerate(VREGS):
        sreg = SREGS[i % len(SREGS)]
        for n in [0, 1, 2]:
            TESTS.append((f"{w} {v}->{sreg} nops={n}", w, v, sreg, n))

def body():
    out = ["s_load_dwordx2 s[4:5], s[0:1], 8", "v_lshlrev_b32 v7, 2, v0", "s_waitcnt lgkmcnt(0)"]
    for t, (name, w, v, sreg, n) in enumerate(TESTS):
        out.append(f"; test {t}: {name}")
        out.append(f"v_mov_b32_e32 {v}, {STALE}")
        out.append("s_nop 7")
        out.append("s_nop 7")
        out += WRITERS[w](v, t)
        if n:
            out.append(f"s_nop {n - 1}")
        out.append(f"v_readfirstlane_b32 {sreg}, {v}")
        out.append("s_nop 7")
        out.append(f"v_mov_b32_e32 v3, {sreg}")
        out.append(f"v_add_u32 v6, {t * THREADS * 4}, v7")
        out.append("global_store_dword v6, v3, s[4:5]")
    out += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return out

def gen(template, dst):
    s = open(template).read()
    m = re.search(r"^(\w+):\n", s, re.M)
    name = m.group(1)
    start = m.end()
    end = s.index("s_endpgm", start) + len("s_endpgm")
    s = s[:start] + "".join(f"    {l}\n" for l in body()) + s[end + 1:]
    s = re.sub(r"\.amdhsa_next_free_vgpr \d+", ".amdhsa_next_free_vgpr 64", s)
    s = re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 48", s)
    s = re.sub(r"\.max_flat_workgroup_size:\s+\d+", ".max_flat_workgroup_size: 1024", s)
    open(dst, "w").write(s)
    print(name, len(TESTS) * THREADS * 4)

def check(path):
    data = open(path, "rb").read()
    bad_any = False
    for t, (name, w, v, sreg, n) in enumerate(TESTS):
        vals = struct.unpack_from(f"<{THREADS}I", data, t * THREADS * 4)
        bad = sum(1 for x in vals if x != 0x1000 + t)
        stale = sum(1 for x in vals if x == STALE)
        print(f"{'FAIL' if bad else 'ok  '} {name:34s} wrong lanes {bad:3d} (stale {stale})")
        bad_any |= bad > 0
    return 1 if bad_any else 0

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3])
    else:
        sys.exit(check(sys.argv[2]))
