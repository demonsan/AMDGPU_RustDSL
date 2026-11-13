#!/usr/bin/env python3
"""VALU-writes-lane-mask -> v_cndmask-reads-it hazard probe (gfx950): a VOP3 compare writing an SGPR pair (or a VOP2 compare writing VCC)
read as the lane mask of v_cndmask after N wait states (docs/exact-mode.md section 5.2; LLVM asks 2 for a VALU SGPR write read by a VALU).

Builds a raw-asm kernel(in, out) from the descriptor of an emitted kernel (argv[1], any small kernel of ours) with a body of tests; each test
leaves one dword per lane in v3 and stores it to out[test * 64 + lane]. A test sets VCC to a known per-lane pattern P (v_cmp), then a VCC
writer W with the opposite pattern, then a VCC reader R after N wait states (s_nop). A reader that sees P instead of W read a stale VCC.

usage: vcc_hazard_probe.py gen <template.s> <out.s>   -> prints the test list
       vcc_hazard_probe.py check <out.bin>
"""
import os, re, struct, sys

# 64 = one wave; 512 = 8 waves (two per SIMD) running the same VALU stream (contention)
THREADS = int(os.environ.get("THREADS", "64"))

# (name, writer, reader, nops, expected per lane: function of lane)
#   P (stale) = odd lanes 1;  W (fresh) = even lanes 1 (writers below produce W)
PAIRS = ["s[6:7]", "s[20:21]", "s[40:41]", "s[94:95]"]
WRITERS = {f"cmp_e64_{p}": f"v_cmp_eq_u32_e64 {p}, 0, v4" for p in PAIRS}
WRITERS["cmp_e32_vcc"] = "v_cmp_eq_u32_e32 vcc, 0, v4"
STALE = {f"cmp_e64_{p}": f"v_cmp_eq_u32_e64 {p}, 1, v4" for p in PAIRS}
STALE["cmp_e32_vcc"] = "v_cmp_eq_u32_e32 vcc, 1, v4"
READERS = {
    "cndmask_e64": {**{f"cmp_e64_{p}": f"v_cndmask_b32_e64 v3, 0, 1, {p}" for p in PAIRS}, "cmp_e32_vcc": "v_cndmask_b32_e64 v3, 0, 1, vcc"},
    "cndmask_e32": {**{f"cmp_e64_{p}": None for p in PAIRS}, "cmp_e32_vcc": "v_cndmask_b32_e32 v3, 0, v9, vcc"},
}
TESTS = []
for w in WRITERS:
    for r in READERS:
        if READERS[r][w] is None:
            continue
        for n in [0, 1, 2]:
            TESTS.append((f"{w}->{r} nops={n}", w, r, n))

def body():
    out = []
    out.append("s_load_dwordx2 s[4:5], s[0:1], 8")
    out.append("v_and_b32 v4, 1, v0")                       # lane & 1
    out.append("v_mov_b32 v9, 1")
    out.append("v_lshlrev_b32 v7, 2, v0")                   # lane * 4
    out.append("s_waitcnt lgkmcnt(0)")
    for t, (name, w, r, n) in enumerate(TESTS):
        out.append(f"; test {t}: {name}")
        out.append(STALE[w])                                # stale pattern P: odd lanes
        out.append("s_nop 7")
        out.append(WRITERS[w])
        if n:
            out.append(f"s_nop {n - 1}")
        out.append(READERS[r][w])
        out.append("s_nop 7")
        out.append(f"v_add_u32 v8, {t * THREADS * 4}, v7")
        out.append("global_store_dword v8, v3, s[4:5]")
    out.append("s_waitcnt vmcnt(0)")
    out.append("s_endpgm")
    return out

def gen(template, dst):
    s = open(template).read()
    m = re.search(r"^(\w+):\n", s, re.M)
    name = m.group(1)
    start = m.end()
    end = s.index("s_endpgm", start) + len("s_endpgm")
    s = s[:start] + "".join(f"    {l}\n" if not l.startswith(";") else f"    {l}\n" for l in body()) + s[end + 1:]
    s = re.sub(r"\.amdhsa_next_free_vgpr \d+", ".amdhsa_next_free_vgpr 16", s)
    s = re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 96", s)
    s = re.sub(r"\.max_flat_workgroup_size:\s+\d+", ".max_flat_workgroup_size: 1024", s)
    open(dst, "w").write(s)
    print(name, len(TESTS) * THREADS * 4)

def check(path):
    data = open(path, "rb").read()
    bad_any = False
    for t, (name, w, r, n) in enumerate(TESTS):
        v = struct.unpack_from(f"<{THREADS}I", data, t * THREADS * 4)
        exp = [1 if tid % 2 == 0 else 0 for tid in range(THREADS)]
        bad = sum(1 for a, b in zip(v, exp) if a != b)
        stale = sum(1 for tid, a in enumerate(v) if a == (tid % 2))
        print(f"{'FAIL' if bad else 'ok  '} {name:36s} wrong lanes {bad:3d} (stale-pattern lanes {stale})")
        bad_any |= bad > 0
    return 1 if bad_any else 0

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3])
    else:
        sys.exit(check(sys.argv[2]))
