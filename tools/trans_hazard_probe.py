#!/usr/bin/env python3
"""Transcendental result -> VALU reader hazard probe (gfx950; docs/exact-mode.md section 5.2). Our table (`trans-result`) asks 4 wait
states before a VALU reads a v_exp / v_rcp / v_log / v_sqrt result; the vendor FMHA reads it after 1 (`v_exp_f32 v188, v188 ; s_nop 0 ;
v_mov_b32 v189, v188`) or 2.

Each test computes vX = f(vX) in place (the vendor's form), N wait states, then a reader copies vX into v3 (v_mov / v_cndmask_e64 with an
all-zero mask / v_add_f32 0 + vX), and every lane stores v3. A stale read shows the input instead of f(input).

usage: THREADS=512 trans_hazard_probe.py gen <template.s> <out.s>  -> prints "<kernel> <bytes>"
       THREADS=512 trans_hazard_probe.py check <out.bin>
"""
import os, re, struct, sys

THREADS = int(os.environ.get("THREADS", "64"))
F = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
# (op, input, exact result)
TRANS = {"v_exp_f32": (3.0, 8.0), "v_rcp_f32": (4.0, 0.25), "v_log_f32": (8.0, 3.0), "v_sqrt_f32": (16.0, 4.0)}
READERS = {
    "mov": lambda v: ["v_mov_b32_e32 v3, " + v],
    "cndmask_e64": lambda v: ["v_cndmask_b32_e64 v3, " + v + ", 0, s[6:7]"],
    "add_f32": lambda v: ["v_add_f32_e32 v3, 0, " + v],
}
VREGS = ["v20", "v41", "v63"]
TESTS = []
for op in TRANS:
    for r in READERS:
        for i, v in enumerate(VREGS):
            for n in [0, 1, 2, 3]:
                TESTS.append((f"{op}->{r} {v} nops={n}", op, r, v, n))

def body():
    out = ["s_load_dwordx2 s[4:5], s[0:1], 8", "v_lshlrev_b32 v7, 2, v0", "s_mov_b64 s[6:7], 0", "s_waitcnt lgkmcnt(0)"]
    for t, (name, op, r, v, n) in enumerate(TESTS):
        x, _ = TRANS[op]
        out.append(f"; test {t}: {name}")
        out.append(f"v_mov_b32_e32 {v}, 0x{F(x):08x}")
        out.append("s_nop 7")
        out.append(f"{op}_e32 {v}, {v}")
        if n:
            out.append(f"s_nop {n - 1}")
        out += READERS[r](v)
        out.append("s_nop 7")
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
    s = re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 16", s)
    s = re.sub(r"\.max_flat_workgroup_size:\s+\d+", ".max_flat_workgroup_size: 1024", s)
    open(dst, "w").write(s)
    print(name, len(TESTS) * THREADS * 4)

def check(path):
    data = open(path, "rb").read()
    bad_any = False
    for t, (name, op, r, v, n) in enumerate(TESTS):
        x, y = TRANS[op]
        vals = struct.unpack_from(f"<{THREADS}I", data, t * THREADS * 4)
        bad = sum(1 for z in vals if z != F(y))
        stale = sum(1 for z in vals if z == F(x))
        print(f"{'FAIL' if bad else 'ok  '} {name:36s} wrong lanes {bad:3d} (stale {stale})")
        bad_any |= bad > 0
    return 1 if bad_any else 0

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3])
    else:
        sys.exit(check(sys.argv[2]))
