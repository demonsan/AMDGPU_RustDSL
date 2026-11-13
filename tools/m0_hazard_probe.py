#!/usr/bin/env python3
"""SALU M0 write -> LDS-DMA (`buffer_load_dword ... lds`) hazard probe (gfx950; docs/exact-mode.md section 5.2). Our table (`m0-pad`)
asks 1 wait state between an SALU write of M0 and the LDS-DMA that reads it; the vendor FMHA issues `s_add_u32 m0, sX, k` immediately
followed by `buffer_load_dwordx4 ... lds`.

Each test zeroes two LDS regions A and B of the wave, sets M0 = A long before, then writes M0 = B with `s_add_u32 m0, s_base, B` (the
vendor's form), N wait states, one LDS-DMA load of 64 dwords, waits, and reads region B back. A stale M0 puts the data into A (B stays 0).
Every wave uses its own regions (512 threads: 8 waves, all SIMDs).

usage: THREADS=512 m0_hazard_probe.py gen <template.s> <out.s>  -> prints "<kernel> <bytes>"
       THREADS=512 m0_hazard_probe.py inputs <in.bin>
       THREADS=512 m0_hazard_probe.py check <out.bin>
"""
import os, re, struct, sys

THREADS = int(os.environ.get("THREADS", "64"))
WIDTHS = {"dword": 1, "dwordx4": 4}
TESTS = []
for w in WIDTHS:
    for n in [0, 1, 2]:
        for rep in range(4):
            TESTS.append((f"buffer_load_{w} lds nops={n} #{rep}", w, n))
IN_DWORDS = 64 * 4 * len(TESTS)

def inp(i):
    return (i * 7 + 1) & 0xFFFFFFFF

def body():
    o = ["s_load_dwordx2 s[8:9], s[0:1], 0", "s_load_dwordx2 s[4:5], s[0:1], 8",
         "v_lshrrev_b32 v10, 6, v0", "v_readfirstlane_b32 s12, v10",             # wave id
         "v_and_b32 v11, 63, v0", "v_lshlrev_b32 v12, 2, v11",                 # lane * 4
         "s_lshl_b32 s13, s12, 13",                                            # wave base: 8 KiB per wave
         "v_lshlrev_b32 v7, 2, v0",
         "s_waitcnt lgkmcnt(0)",
         "s_and_b32 s9, s9, 0xffff", f"s_mov_b32 s10, {IN_DWORDS * 4}", "s_mov_b32 s11, 0x20000",
         "v_mov_b32 v13, 0"]
    for t, (name, w, n) in enumerate(TESTS):
        d = WIDTHS[w]
        o.append(f"; test {t}: {name}")
        # zero region A (wave base) and B (wave base + 4096) of this wave: 64 lanes x d dwords each
        o.append("v_add_u32 v14, s13, v12")
        for k in range(d):
            o.append(f"ds_write_b32 v14, v13 offset:{k * 256}")
            o.append(f"ds_write_b32 v14, v13 offset:{4096 + k * 256}")
        o.append("s_waitcnt lgkmcnt(0)")
        o.append("s_mov_b32 m0, s13")                    # stale M0 = A
        o += ["s_nop 7"] * 3
        o.append("s_add_u32 m0, s13, 4096")              # fresh M0 = B (the vendor's form)
        if n:
            o.append(f"s_nop {n - 1}")
        # source: dwords [t*256 + lane*d ..] (voffset = lane*4*d + t*1024)
        o.append(f"v_lshlrev_b32 v15, {2 + (d == 4) * 2}, v11")
        o.append(f"v_add_u32 v15, {t * 1024}, v15")
        o.append(f"buffer_load_{w} v15, s[8:11], 0 offen lds")
        o.append("s_waitcnt vmcnt(0)")
        # read back the first dword each lane wrote into B: B + lane*4*d
        o.append(f"v_lshlrev_b32 v16, {2 + (d == 4) * 2}, v11")
        o.append("v_add_u32 v16, s13, v16")
        o.append("ds_read_b32 v3, v16 offset:4096")
        o.append("s_waitcnt lgkmcnt(0)")
        o.append(f"v_add_u32 v6, {t * THREADS * 4}, v7")
        o.append("global_store_dword v6, v3, s[4:5]")
    o += ["s_waitcnt vmcnt(0)", "s_endpgm"]
    return o

def gen(template, dst):
    s = open(template).read()
    m = re.search(r"^(\w+):\n", s, re.M)
    name = m.group(1)
    start = m.end()
    end = s.index("s_endpgm", start) + len("s_endpgm")
    s = s[:start] + "".join(f"    {l}\n" for l in body()) + s[end + 1:]
    s = re.sub(r"\.amdhsa_next_free_vgpr \d+", ".amdhsa_next_free_vgpr 32", s)
    s = re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 24", s)
    s = re.sub(r"\.amdhsa_group_segment_fixed_size \d+", ".amdhsa_group_segment_fixed_size 65536", s)
    s = re.sub(r"\.group_segment_fixed_size:\s+\d+", ".group_segment_fixed_size: 65536", s)
    s = re.sub(r"\.max_flat_workgroup_size:\s+\d+", ".max_flat_workgroup_size: 1024", s)
    open(dst, "w").write(s)
    print(name, len(TESTS) * THREADS * 4)

def inputs(path):
    open(path, "wb").write(struct.pack(f"<{IN_DWORDS}I", *[inp(i) for i in range(IN_DWORDS)]))

def check(path):
    data = open(path, "rb").read()
    bad_any = False
    for t, (name, w, n) in enumerate(TESTS):
        d = WIDTHS[w]
        vals = struct.unpack_from(f"<{THREADS}I", data, t * THREADS * 4)
        exp = [inp(t * 256 + (tid % 64) * d) for tid in range(THREADS)]
        bad = sum(1 for a, b in zip(vals, exp) if a != b)
        zero = sum(1 for a in vals if a == 0)
        print(f"{'FAIL' if bad else 'ok  '} {name:34s} wrong lanes {bad:3d} (still zero {zero})")
        bad_any |= bad > 0
    return 1 if bad_any else 0

if __name__ == "__main__":
    {"gen": lambda: gen(sys.argv[2], sys.argv[3]), "inputs": lambda: inputs(sys.argv[2]), "check": lambda: sys.exit(check(sys.argv[2]))}[sys.argv[1]]()
