#!/usr/bin/env python3
"""LDS read timing-reliance probe (gfx950), for exact-mode `TrustWait` evidence (docs/exact-mode.md section 5.2).

The pattern under test: two ds_read_b128 into X are issued, an `s_waitcnt lgkmcnt(2)` deliberately leaves them in flight, then GAP
instructions, two NEWER ds_read_b128 (Y), and a consumer of X with no wait (the lgkm counter would require lgkmcnt(2) there). All 8 waves of
the workgroup (512 threads) run it in a loop with extra LDS reads in flight (contention). Each thread sums everything it read; the host
compares with the exact sums. A stale X read changes the sum.

The GAP=0 variant is the sensitivity control: the consumer issues right behind the Y reads, which should catch a stale read if the hardware
can produce one at all (a probe that cannot fail proves nothing).

usage: timing_probe.py gen <template.s> <out.s> <gap> <iters>   -> prints "<kernel name> <out bytes>"
       timing_probe.py check <out.bin> <iters>
"""
import re, struct, sys

THREADS = 512
MUL, ADD = 0x9E3779B1, 0x1234567
M32 = (1 << 32) - 1

def body(gap, iters):
    # gap < 0: the SAFE control (the counter-guaranteed wait before the consumer, |gap| instructions of gap)
    safe = gap < 0
    gap = abs(gap)
    o = []
    o.append("s_load_dwordx2 s[4:5], s[0:1], 8")
    o.append(f"s_mov_b32 s6, 0x{MUL:08x}")
    # LDS fill: dword i = tid + 512 j (j < 16) holds i * MUL + ADD (32 KiB)
    for j in range(16):
        o.append(f"v_add_u32_e32 v2, {512 * j}, v0")
        o.append("v_mul_lo_u32 v3, v2, s6")
        o.append(f"v_add_u32_e32 v3, 0x{ADD:08x}, v3")
        o.append("v_lshlrev_b32_e32 v2, 2, v2")
        o.append("ds_write_b32 v2, v3")
    o.append("s_waitcnt lgkmcnt(0)")
    o.append("s_barrier")
    o.append("v_mov_b32_e32 v30, 0")      # sum
    o.append("s_mov_b32 s7, 0")           # iteration
    o.append("BB_loop:")
    o.append("v_add_u32_e32 v4, s7, v0")
    o.append("v_lshlrev_b32_e32 v4, 6, v4")
    o.append("v_and_b32_e32 v4, 0x3fc0, v4")       # X base (16 KiB window, 64-byte aligned)
    o.append("v_xor_b32_e32 v5, 0x4000, v4")       # extra reads
    o.append("v_xor_b32_e32 v6, 0x2000, v4")       # Y base
    for k in range(4):
        o.append(f"ds_read_b128 v[{40 + 4 * k}:{43 + 4 * k}], v5 offset:{16 * k}")
    o.append("ds_read_b128 v[10:13], v4")
    o.append("ds_read_b128 v[14:17], v4 offset:16")
    o.append("s_waitcnt lgkmcnt(2)")               # the extra reads are complete, X is still in flight
    # GAP instructions (consuming the extra reads, like the vendor's ALU work between)
    for k in range(gap):
        o.append(f"v_add_u32_e32 v30, v30, v{40 + (k % 16)}")
    for k in range(gap, 16):
        o.append(f"v_add_u32_e32 v31, v31, v{40 + k}")     # the rest of the extras, AFTER the consumer (keeps the sums equal)
    o.append("ds_read_b128 v[18:21], v6")
    o.append("ds_read_b128 v[22:25], v6 offset:16")
    if safe:
        o.append("s_waitcnt lgkmcnt(2)")
    for k in range(8):
        o.append(f"v_add_u32_e32 v30, v30, v{10 + k}")      # X consumed with NO wait (the timing reliance)
    o.append("s_waitcnt lgkmcnt(0)")
    for k in range(8):
        o.append(f"v_add_u32_e32 v30, v30, v{18 + k}")
    o.append("s_add_u32 s7, s7, 1")
    o.append(f"s_cmp_lt_u32 s7, {iters}")
    o.append("s_cbranch_scc1 BB_loop")
    o.append("v_add_u32_e32 v30, v30, v31")
    o.append("v_lshlrev_b32_e32 v7, 2, v0")
    o.append("s_waitcnt lgkmcnt(0)")
    o.append("global_store_dword v7, v30, s[4:5]")
    o.append("s_waitcnt vmcnt(0)")
    o.append("s_endpgm")
    return o

def gen(template, dst, gap, iters):
    s = open(template).read()
    m = re.search(r"^(\w+):\n", s, re.M)
    name = m.group(1)
    start = m.end()
    end = s.index("s_endpgm", start) + len("s_endpgm")
    pre = ["v_mov_b32_e32 v31, 0"]
    s = s[:start] + "".join(f"    {l}\n" if not l.endswith(":") else f"{l}\n" for l in pre + body(gap, iters)) + s[end + 1:]
    s = re.sub(r"\.amdhsa_next_free_vgpr \d+", ".amdhsa_next_free_vgpr 64", s)
    s = re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 16", s)
    s = re.sub(r"\.amdhsa_group_segment_fixed_size \d+", ".amdhsa_group_segment_fixed_size 65536", s)
    s = re.sub(r"\.max_flat_workgroup_size:\s+\d+", ".max_flat_workgroup_size: 1024", s)
    s = re.sub(r"\.group_segment_fixed_size:\s+\d+", ".group_segment_fixed_size: 65536", s)
    open(dst, "w").write(s)
    print(name, THREADS * 4)

def lds(i):
    return (i * MUL + ADD) & M32

def expected(tid, iters):
    acc = 0
    for it in range(iters):
        x = (((tid + it) << 6) & 0x3FC0)
        for base in (x ^ 0x4000, x, x ^ 0x2000):
            for d in range(8 if base != (x ^ 0x4000) else 16):
                acc = (acc + lds(base // 4 + d)) & M32
    return acc

def check(path, iters):
    v = struct.unpack_from(f"<{THREADS}I", open(path, "rb").read())
    bad = [t for t in range(THREADS) if v[t] != expected(t, iters)]
    print(f"timing probe: {THREADS - len(bad)}/{THREADS} threads exact" + (f"; first wrong: {bad[:6]}" if bad else ""))
    return 1 if bad else 0

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    else:
        sys.exit(check(sys.argv[2], int(sys.argv[3])))
