#!/usr/bin/env python3
"""MFMA result latency probe (gfx950): how many wait states a VALU / VMEM reader needs after a 16x16x128 f8f6f4 MFMA, per operand format.

Raw-asm kernel(in, out) on our descriptor (template = any small emitted kernel). Per test: the D registers hold a sentinel, A and B hold 1.0 in
the format, C = 0; after N `s_nop` wait states a reader copies D[0]; out[test * 64 + lane] = the copy. 16x16x128 with all-ones inputs gives
128.0 in every element (0x43000000); the sentinel means the reader was too early.

usage: mfma_lat_probe.py gen <template.s> <out.s>  |  check <out.bin>
"""
import re, struct, sys

def pack(bits_per, code, nwords):
    v = 0
    for i in range(nwords * 32 // bits_per):
        v |= code << (i * bits_per)
    return [(v >> (32 * k)) & 0xFFFFFFFF for k in range(nwords)]

# (format name, cbsz/blgp code, bits, 1.0 code, operand dwords)
FORMATS = [("fp8", 0, 8, 0x38, 8), ("fp6e2m3", 2, 6, 0x08, 6), ("fp4", 4, 4, 0x2, 4)]
FORMS = [("plain", "v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[16:{a}], v[24:{b}], 0 cbsz:{c} blgp:{c}"),
         ("scale", "v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[16:{a}], v[24:{b}], 0, v40, v40 cbsz:{c} blgp:{c}")]
READERS = [("valu", "v_mov_b32 v3, v8"), ("vmem", None)]
NOPS = list(range(0, 14))
TESTS = [(f, fm, rd, n) for f in FORMATS for fm in FORMS for rd in READERS for n in NOPS]

def body():
    o = ["s_load_dwordx2 s[4:5], s[0:1], 8", "v_lshlrev_b32 v7, 2, v0", "v_mov_b32 v40, 127", "s_waitcnt lgkmcnt(0)"]
    for t, ((fname, code, bits, one, nw), (form, mn), (rname, rd), n) in enumerate(TESTS):
        o.append(f"; test {t}: {fname} {form} {rname} nops={n}")
        for k, w in enumerate(pack(bits, one, nw)):
            o.append(f"v_mov_b32 v{16 + k}, {w:#x}")
            o.append(f"v_mov_b32 v{24 + k}, {w:#x}")
        for k in range(4):
            o.append(f"v_mov_b32 v{8 + k}, 0x5eed5eed")
        o.append("v_add_u32 v9, 0, v9")       # keep the sentinel moves retired
        o.append("s_nop 7")
        o.append(mn.format(a=16 + nw - 1, b=24 + nw - 1, c=code))
        left = n
        while left > 0:
            k = min(left, 8)
            o.append(f"s_nop {k - 1}")
            left -= k
        o.append(f"v_add_u32 v41, {t * 256}, v7")
        if rd is None:
            o.append("global_store_dword v41, v8, s[4:5]")
        else:
            o.append(rd)
            o.append("s_nop 7")
            o.append("global_store_dword v41, v3, s[4:5]")
        o.append("s_waitcnt vmcnt(0)")
        o.append("s_nop 7")
        o.append("s_nop 7")
    o.append("s_endpgm")
    return o

def gen(template, dst):
    s = open(template).read()
    m = re.search(r"^(\w+):\n", s, re.M)
    start = m.end()
    end = s.index("s_endpgm", start) + len("s_endpgm")
    s = s[:start] + "".join(f"    {l}\n" for l in body()) + s[end + 1:]
    s = re.sub(r"\.amdhsa_next_free_vgpr \d+", ".amdhsa_next_free_vgpr 48", s)
    s = re.sub(r"\.amdhsa_accum_offset \d+", ".amdhsa_accum_offset 48", s)
    s = re.sub(r"\.amdhsa_next_free_sgpr \d+", ".amdhsa_next_free_sgpr 16", s)
    open(dst, "w").write(s)
    print(m.group(1), len(TESTS) * 256)

def check(path):
    d = open(path, "rb").read()
    first = {}
    for t, ((fname, *_), (form, _), (rname, _), n) in enumerate(TESTS):
        v = struct.unpack_from("<64I", d, t * 256)
        ok = all(x == 0x43000000 for x in v)
        stale = sum(1 for x in v if x == 0x5eed5eed)
        key = (fname, form, rname)
        if ok and key not in first:
            first[key] = n
        if not ok and key in first:
            print(f"NONMONOTONE {key} nops={n} stale={stale} values={hex(v[0])}")
    for k in sorted(set((f[0], fm[0], r[0]) for f in FORMATS for fm in FORMS for r in READERS)):
        print(f"{k[0]:8s} {k[1]:6s} {k[2]:5s}: first correct after {first.get(k, 'never')} wait states")
    return 0

if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen(sys.argv[2], sys.argv[3])
    else:
        sys.exit(check(sys.argv[2]))
