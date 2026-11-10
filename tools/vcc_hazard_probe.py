#!/usr/bin/env python3
"""VALU-writes-VCC -> VALU-reads-VCC hazard probe (gfx950; LLVM GCNHazardRecognizer hasVDecCoExecHazard: 2 wait states).

Builds a raw-asm kernel(in, out) from the descriptor of an emitted kernel (argv[1], any small kernel of ours) with a body of tests; each test
leaves one dword per lane in v3 and stores it to out[test * 64 + lane]. A test sets VCC to a known per-lane pattern P (v_cmp), then a VCC
writer W with the opposite pattern, then a VCC reader R after N wait states (s_nop). A reader that sees P instead of W read a stale VCC.

usage: vcc_hazard_probe.py gen <template.s> <out.s>   -> prints the test list
       vcc_hazard_probe.py check <out.bin>
"""
import re, struct, sys

# (name, writer, reader, nops, expected per lane: function of lane)
#   P (stale) = odd lanes 1;  W (fresh) = even lanes 1 (writers below produce W)
WRITERS = {
    # v_add_co with carry exactly in the even lanes: v5 = 0xffffffff in even lanes, 0 in odd lanes; + 1
    "add_co": "v_add_co_u32 v6, vcc, v5, 1",
    # v_cmp: even lanes (v4 = lane & 1 == 0)
    "cmp": "v_cmp_eq_u32 vcc, 0, v4",
    # v_addc with carry-in 0 producing a carry in the even lanes: v5 + 1 + 0 (vcc must be 0 before: set by a cmp that is false everywhere)
}
READERS = {
    # carry-in read: v3 = 0 + 0 + vcc
    "addc": "v_addc_co_u32 v3, vcc, 0, 0, vcc",
    # lane-mask read: v3 = vcc ? 1 : 0
    "cndmask": "v_cndmask_b32 v3, 0, 1, vcc",
}
TESTS = []
for w in WRITERS:
    for r in READERS:
        for n in [0, 1, 2]:
            TESTS.append((f"{w}->{r} nops={n}", w, r, n))

def body():
    out = []
    out.append("s_load_dwordx2 s[4:5], s[0:1], 8")
    out.append("v_and_b32 v4, 1, v0")                       # lane & 1
    out.append("v_cmp_eq_u32 vcc, 0, v4")
    out.append("v_cndmask_b32 v5, 0, -1, vcc")              # v5 = 0xffffffff in even lanes
    out.append("v_lshlrev_b32 v7, 2, v0")                   # lane * 4
    out.append("s_waitcnt lgkmcnt(0)")
    for t, (name, w, r, n) in enumerate(TESTS):
        out.append(f"; test {t}: {name}")
        out.append("v_cmp_eq_u32 vcc, 1, v4")               # stale pattern P: odd lanes
        out.append("s_nop 7")
        out.append(WRITERS[w])
        if n:
            out.append(f"s_nop {n - 1}")
        out.append(READERS[r])
        out.append("s_nop 7")
        out.append(f"v_add_u32 v8, {t * 256}, v7")
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
    s = re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 16", s)
    open(dst, "w").write(s)
    print(name, len(TESTS) * 64 * 4)

def check(path):
    data = open(path, "rb").read()
    bad_any = False
    for t, (name, w, r, n) in enumerate(TESTS):
        v = struct.unpack_from("<64I", data, t * 256)
        exp = [1 if lane % 2 == 0 else 0 for lane in range(64)]
        bad = sum(1 for a, b in zip(v, exp) if a != b)
        stale = sum(1 for lane, a in enumerate(v) if a == (lane % 2))
        print(f"{'FAIL' if bad else 'ok  '} {name:28s} wrong lanes {bad:2d} (stale-pattern lanes {stale})")
        bad_any |= bad > 0
    return 1 if bad_any else 0

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3])
    else:
        sys.exit(check(sys.argv[2]))
