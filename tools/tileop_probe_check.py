#!/usr/bin/env python3
"""Inputs and CPU references of the tile-op probes (docs/op-coverage-tileops.md §6.5; kernels in crates/amdgpu-kernels/src/probes/tileops.rs).

usage: tileop_probe_check.py gen   <probe> <input.bin>          -> writes the input, prints the output size in bytes
       tileop_probe_check.py check <probe> <input.bin> <output.bin>  -> exit 0 when every checked value matches
"""
import math, random, struct, sys

LANES = 64

def f32(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

def u32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]

def to_f32(x):
    """round a Python float to f32 (RNE), keeping inf / nan"""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)

def s32(u):
    u &= 0xFFFFFFFF
    return u - (1 << 32) if u & 0x80000000 else u

def h2f(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]

def f2h(x):
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return 0x7C00 if x > 0 else 0xFC00

# ---------------------------------------------------------------- probe_valu_p1

UN = ["FloorF32", "CeilF32", "TruncF32", "RndneF32", "FractF32", "RsqF32", "SqrtF32", "SinF32", "CosF32", "CvtI32F32", "CvtU32F32",
      "CvtF32I32", "CvtF32U32", "CvtF32Ubyte0", "CvtF32Ubyte1", "CvtF32Ubyte2", "CvtF32Ubyte3"]
BIN = ["MinF32", "MaxI32", "MaxU32", "MinU32", "SubU32", "MulU32U24", "AshrI32", "MulHiU32", "MulHiI32", "OrB32", "PkAddF16", "PkMulF16",
       "PkMaxF16", "PkMinF16"]
TRI = ["Min3F32", "Med3F32", "Min3I32", "Max3I32", "Med3I32", "Min3U32", "Max3U32", "Med3U32", "MadU32U24", "LshlAddU32", "Add3U32", "BfeU32",
       "BfeI32", "BfiB32", "PkFmaF16"]
CMPS = [("Lt", "F32"), ("Le", "F32"), ("Gt", "F32"), ("Ge", "F32"), ("Eq", "F32"), ("Ne", "F32"), ("Lg", "F32"), ("Uno", "F32"), ("Ord", "F32"),
        ("Lt", "I32"), ("Le", "I32"), ("Gt", "I32"), ("Ge", "I32"), ("Eq", "I32"), ("Ne", "I32"),
        ("Lt", "U32"), ("Le", "U32"), ("Gt", "U32"), ("Ge", "U32"), ("Eq", "U32"), ("Ne", "U32"), ("Ne", "F32")]

SPECIAL_F = [0.0, -0.0, 0.5, -0.5, 1.5, 2.5, -2.5, 3.0, 1e9, -1e9, 1e-30, math.inf, -math.inf, math.nan, 7.999, -0.25]
SPECIAL_I = [0, 1, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFF, 0x1000000, 5, 31, 32, 0x12345678, 0xF0F0F0F0, 2, 0x8000, 0xFFFF0000, 7]

def gen_valu_p1():
    rnd = random.Random(1234)
    planes = []
    for p in range(3):
        v = []
        for l in range(LANES):
            if l < 16:
                x = SPECIAL_F[(l + 5 * p) % 16]
            else:
                x = rnd.uniform(-8, 8)
            v.append(u32(to_f32(x)))
        planes.append(v)
    for p in range(3):
        v = []
        for l in range(LANES):
            if l < 16:
                x = SPECIAL_I[(l + 3 * p) % 16]
            elif l < 32:
                x = rnd.randrange(0, 40)  # small (shift amounts, field widths)
            else:
                x = rnd.getrandbits(32)
            v.append(x)
        planes.append(v)
    return planes

def fmin(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return min(a, b)

def fmax(a, b):
    if math.isnan(a):
        return b
    if math.isnan(b):
        return a
    return max(a, b)

def un_ref(op, a, ia):
    if op == "FloorF32": return ("f", math.floor(a) if math.isfinite(a) else a)
    if op == "CeilF32": return ("f", math.ceil(a) if math.isfinite(a) else a)
    if op == "TruncF32": return ("f", math.trunc(a) if math.isfinite(a) else a)
    if op == "RndneF32": return ("f", float(round(a)) if math.isfinite(a) else a)
    if op == "FractF32":
        if not math.isfinite(a): return ("f", math.nan)
        return ("f", min(a - math.floor(a), f32(0x3F7FFFFF)))
    if op == "RsqF32":
        if math.isnan(a) or a < 0: return ("f", math.nan)
        if a == 0: return ("f", math.copysign(math.inf, a))
        if a == math.inf: return ("f", 0.0)
        return ("rel", 1 / math.sqrt(a), 2 ** -21)
    if op == "SqrtF32":
        if math.isnan(a) or a < 0: return ("f", math.nan)
        if a == math.inf: return ("f", a)
        return ("rel", math.sqrt(a), 2 ** -21)
    if op in ("SinF32", "CosF32"):
        if not math.isfinite(a) or abs(a) > 256: return ("any",)
        return ("abs", (math.sin if op == "SinF32" else math.cos)(2 * math.pi * a), 2e-6)
    if op == "CvtI32F32":
        if math.isnan(a): return ("i", 0)
        return ("i", max(-2 ** 31, min(2 ** 31 - 1, math.trunc(a) if math.isfinite(a) else (2 ** 31 if a > 0 else -2 ** 32))) & 0xFFFFFFFF)
    if op == "CvtU32F32":
        if math.isnan(a): return ("i", 0)
        return ("i", max(0, min(2 ** 32 - 1, math.trunc(a) if math.isfinite(a) else (2 ** 33 if a > 0 else 0))))
    if op == "CvtF32I32": return ("f", to_f32(float(s32(ia))))
    if op == "CvtF32U32": return ("f", to_f32(float(ia)))
    k = int(op[-1])
    return ("f", float((ia >> (8 * k)) & 255))

def pk(op, x, y, z=None):
    r = 0
    for h in range(2):
        a, b = h2f(x >> (16 * h)), h2f(y >> (16 * h))
        if op == "PkAddF16": v = a + b
        elif op == "PkMulF16": v = a * b
        elif op == "PkMaxF16": v = fmax(a, b)
        elif op == "PkMinF16": v = fmin(a, b)
        else: v = a * b + h2f(z >> (16 * h)) if not (math.isinf(a * b) and math.isinf(h2f(z >> (16 * h))) and (a * b) != h2f(z >> (16 * h))) else math.nan
        r |= f2h(v) << (16 * h)
    return r

def bin_ref(op, a, b, ia, ib):
    sa, sb = s32(ia), s32(ib)
    if op == "MinF32": return ("f", fmin(a, b))
    if op == "MaxI32": return ("i", max(sa, sb) & 0xFFFFFFFF)
    if op == "MaxU32": return ("i", max(ia, ib))
    if op == "MinU32": return ("i", min(ia, ib))
    if op == "SubU32": return ("i", (ia - ib) & 0xFFFFFFFF)
    if op == "MulU32U24": return ("i", ((ia & 0xFFFFFF) * (ib & 0xFFFFFF)) & 0xFFFFFFFF)
    if op == "AshrI32": return ("i", (sa >> (ib & 31)) & 0xFFFFFFFF)
    if op == "MulHiU32": return ("i", (ia * ib) >> 32)
    if op == "MulHiI32": return ("i", ((sa * sb) >> 32) & 0xFFFFFFFF)
    if op == "OrB32": return ("i", ia | ib)
    return ("h", pk(op, ia, ib))

def tri_ref(op, a, b, c, ia, ib, ic):
    sa, sb, sc = s32(ia), s32(ib), s32(ic)
    if op in ("Min3F32", "Med3F32"):
        if any(math.isnan(x) for x in (a, b, c)): return ("any",)
        return ("f", min(a, b, c) if op == "Min3F32" else sorted([a, b, c])[1])
    if op == "Min3I32": return ("i", min(sa, sb, sc) & 0xFFFFFFFF)
    if op == "Max3I32": return ("i", max(sa, sb, sc) & 0xFFFFFFFF)
    if op == "Med3I32": return ("i", sorted([sa, sb, sc])[1] & 0xFFFFFFFF)
    if op == "Min3U32": return ("i", min(ia, ib, ic))
    if op == "Max3U32": return ("i", max(ia, ib, ic))
    if op == "Med3U32": return ("i", sorted([ia, ib, ic])[1])
    if op == "MadU32U24": return ("i", ((ia & 0xFFFFFF) * (ib & 0xFFFFFF) + ic) & 0xFFFFFFFF)
    if op == "LshlAddU32": return ("i", (((ia << (ib & 31)) & 0xFFFFFFFF) + ic) & 0xFFFFFFFF)
    if op == "Add3U32": return ("i", (ia + ib + ic) & 0xFFFFFFFF)
    if op in ("BfeU32", "BfeI32"):
        off, w = ib & 31, ic & 31
        if w == 0: return ("i", 0)
        v = (ia >> off) & ((1 << w) - 1)
        if op == "BfeI32" and off + w <= 32 and v & (1 << (w - 1)):
            v -= 1 << w
        if op == "BfeI32" and off + w > 32:
            return ("any",)
        return ("i", v & 0xFFFFFFFF)
    if op == "BfiB32": return ("i", (ia & ib) | (~ia & ic & 0xFFFFFFFF))
    return ("h", pk(op, ia, ib, ic))

def cmp_ref(p, ty, a, b, ia, ib):
    if ty == "F32":
        nan = math.isnan(a) or math.isnan(b)
        r = {"Lt": not nan and a < b, "Le": not nan and a <= b, "Gt": not nan and a > b, "Ge": not nan and a >= b, "Eq": not nan and a == b,
             "Ne": nan or a != b, "Lg": not nan and a != b, "Uno": nan, "Ord": not nan}[p]
    else:
        x, y = (s32(ia), s32(ib)) if ty == "I32" else (ia, ib)
        r = {"Lt": x < y, "Le": x <= y, "Gt": x > y, "Ge": x >= y, "Eq": x == y, "Ne": x != y}[p]
    return ("i", 1 if r else 0)

def match(exp, got):
    kind = exp[0]
    if kind == "any":
        return True
    if kind == "i":
        return got == exp[1]
    g = f32(got)
    if kind == "h":
        ok = True
        for h in range(2):
            e, x = h2f(exp[1] >> (16 * h)), h2f(got >> (16 * h))
            ok &= (math.isnan(e) and math.isnan(x)) or e == x
        return ok
    if kind == "f":
        e = to_f32(exp[1]) if math.isfinite(exp[1]) else exp[1]
        return (math.isnan(e) and math.isnan(g)) or e == g
    if kind == "rel":
        return math.isfinite(g) and abs(g - exp[1]) <= exp[2] * abs(exp[1])
    if kind == "abs":
        return abs(g - exp[1]) <= exp[2]
    raise ValueError(kind)

def check_valu_p1(planes, out):
    fa, fb, fc = [[f32(x) for x in planes[p]] for p in range(3)]
    ia, ib, ic = planes[3], planes[4], planes[5]
    k = 0
    bad = 0
    total = 0
    def put(name, refs):
        nonlocal k, bad, total
        fails = []
        for l in range(LANES):
            got = out[k * LANES + l]
            total += 1
            if not match(refs[l], got):
                fails.append((l, refs[l], hex(got)))
        if fails:
            bad += len(fails)
            print(f"FAIL {name}: {len(fails)} lanes, e.g. lane {fails[0][0]} want {fails[0][1]} got {fails[0][2]}")
        k += 1
    for op in UN:
        put(op, [un_ref(op, fa[l], ia[l]) for l in range(LANES)])
    for op in BIN:
        put(op, [bin_ref(op, fa[l], fb[l], ia[l], ib[l]) for l in range(LANES)])
    for op in TRI:
        put(op, [tri_ref(op, fa[l], fb[l], fc[l], ia[l], ib[l], ic[l]) for l in range(LANES)])
    for p, ty in CMPS:
        put(f"Cmp{p}{ty}", [cmp_ref(p, ty, fa[l], fb[l], ia[l], ib[l]) for l in range(LANES)])
    print(f"probe_valu_p1: {k} ops x {LANES} lanes, {total - bad} / {total} values match")
    return bad == 0


# ---------------------------------------------------------------- probe_mem_p1

MEM_PLANES = 64

def gen_mem_p1():
    data = bytes(((i * 37 + 11) ^ (i >> 5)) & 255 for i in range(4096))
    return [list(struct.unpack("<1024I", data))]

def mem_ref(inb):
    """expected planes (64 x 64 dwords) and the two 64 x 32-byte store regions"""
    def rd(addr, n):
        return int.from_bytes(inb[addr:addr + n], "little")
    def sx(v, bits):
        return (v - (1 << bits)) & 0xFFFFFFFF if v & (1 << (bits - 1)) else v
    merge = 0  # gfx950 zeroes the unused half of a d16 load
    def d16(v, bits, signed, hi):
        h = (sx(v, bits) if signed else v) & 0xFFFF
        return ((merge & 0xFFFF) | (h << 16)) if hi else ((merge & 0xFFFF0000) | h)
    planes = [[0] * LANES for _ in range(MEM_PLANES)]
    lds = bytearray(4096)
    for l in range(LANES):
        lds[16 * l:16 * l + 16] = inb[16 * l:16 * l + 16]
    for l in range(LANES):
        b = 16 * l
        vals = [rd(b + 1, 1), sx(rd(b + 3, 1), 8), rd(b + 2, 2), sx(rd(b + 6, 2), 16), rd(b + 4, 4)]
        vals += [rd(b + 8, 4), rd(b + 12, 4)]
        vals += [rd(b, 4), rd(b + 4, 4), rd(b + 8, 4)]
        vals += [rd(b + 4 * i, 4) for i in range(4)]
        vals += [d16(rd(b + 1, 1), 8, False, False), d16(rd(b + 1, 1), 8, False, True), d16(rd(b + 3, 1), 8, True, False), d16(rd(b + 3, 1), 8, True, True),
                 d16(rd(b + 2, 2), 16, False, False), d16(rd(b + 2, 2), 16, False, True)]
        # buffer: same bytes
        vals += [rd(b + 1, 1), sx(rd(b + 3, 1), 8), rd(b + 2, 2), sx(rd(b + 6, 2), 16), rd(b, 4), rd(b + 4, 4), rd(b + 8, 4)]
        # LDS loads (region 0 holds the input)
        L = lambda a, n: int.from_bytes(lds[a:a + n], "little")
        vals += [L(b + 1, 1), sx(L(b + 3, 1), 8), L(b + 2, 2), sx(L(b + 6, 2), 16), L(b, 4), L(b + 4, 4), L(b + 8, 4)]
        vals += [d16(L(b + 1, 1), 8, False, False), d16(L(b + 1, 1), 8, False, True), d16(L(b + 3, 1), 8, True, False), d16(L(b + 3, 1), 8, True, True),
                 d16(L(b + 2, 2), 16, False, False), d16(L(b + 2, 2), 16, False, True)]
        vals += [L(b + 4, 4), L(b + 12, 4)]
        vals += [L(b, 4), L(b + 4, 4), L(b + 8, 4), L(b + 12, 4)]
        vals += [L(4 * l, 4), L(4 * l + 256, 4)]
        vals += [L(8 * l, 4), L(8 * l + 4, 4), L(8 * l + 512, 4), L(8 * l + 516, 4)]
        x, y = rd(b, 4), rd(b + 4, 4)
        r1 = bytearray(16); r1[0] = x & 255; r1[1] = (x >> 16) & 255; r1[2:4] = (y & 0xFFFF).to_bytes(2, "little"); r1[4:6] = (y >> 16).to_bytes(2, "little")
        r2 = bytearray(inb[b:b + 12]) + bytearray(4)
        r3 = bytearray(16); r3[0:4] = x.to_bytes(4, "little"); r3[8:12] = y.to_bytes(4, "little")
        for r in (r1, r2, r3):
            vals += list(struct.unpack("<4I", bytes(r)))
        assert len(vals) == MEM_PLANES, len(vals)
        for k, v in enumerate(vals):
            planes[k][l] = v & 0xFFFFFFFF
    region = bytearray()
    for l in range(LANES):
        b = 16 * l
        x, y = rd(b, 4), rd(b + 4, 4)
        r = bytearray([0xA5] * 32)
        r[0] = x & 255; r[1] = (x >> 16) & 255; r[2:4] = (y & 0xFFFF).to_bytes(2, "little"); r[4:6] = (y >> 16).to_bytes(2, "little")
        r[16:28] = inb[b:b + 12]
        region += r
    return planes, bytes(region)

def check_mem_p1(planes_in, out):
    inb = struct.pack("<1024I", *planes_in[0])
    planes, region = mem_ref(inb)
    names = (["gU8", "gI8", "gU16", "gI16", "gB32", "gB64.0", "gB64.1", "gB96.0", "gB96.1", "gB96.2", "gB128.0", "gB128.1", "gB128.2", "gB128.3",
              "gD16U8", "gD16U8Hi", "gD16I8", "gD16I8Hi", "gD16B16", "gD16B16Hi", "bU8", "bI8", "bU16", "bI16", "bB96.0", "bB96.1", "bB96.2",
              "lU8", "lI8", "lU16", "lI16", "lB96.0", "lB96.1", "lB96.2", "lD16U8", "lD16U8Hi", "lD16I8", "lD16I8Hi", "lD16B16", "lD16B16Hi",
              "lRead2B32.0", "lRead2B32.1", "lRead2B64.0", "lRead2B64.1", "lRead2B64.2", "lRead2B64.3", "lRead2St64B32.0", "lRead2St64B32.1",
              "lRead2St64B64.0", "lRead2St64B64.1", "lRead2St64B64.2", "lRead2St64B64.3"]
             + [f"ldsStores.r{r}.{i}" for r in range(1, 4) for i in range(4)])
    bad = 0
    for k in range(MEM_PLANES):
        fails = [l for l in range(LANES) if out[k * LANES + l] != planes[k][l]]
        if fails:
            bad += len(fails)
            l = fails[0]
            print(f"FAIL {names[k]}: {len(fails)} lanes, e.g. lane {l} want {planes[k][l]:#x} got {out[k * LANES + l]:#x}")
    ob = struct.pack(f"<{len(out)}I", *out)
    base = MEM_PLANES * LANES * 4
    for name, off in (("global stores", base), ("buffer stores", base + LANES * 32)):
        got = ob[off:off + LANES * 32]
        if got != region:
            i = next(i for i in range(len(region)) if got[i] != region[i])
            bad += 1
            print(f"FAIL {name}: first difference at lane {i // 32} byte {i % 32}: want {region[i]:#x} got {got[i]:#x}")
    print(f"probe_mem_p1: {MEM_PLANES} planes x {LANES} lanes + 2 store regions, {'all match' if bad == 0 else f'{bad} mismatches'}")
    return bad == 0

# ---------------------------------------------------------------- probe_mfma_p1
# kinds: (name, M, K, element type, acc dwords); layouts (gfx950, hypothesis checked by the probe):
#   A: lane l holds row l % M and k = (16 bytes / elem) * (l / M) + 0..; B: column l % M, same k; D (16x16): lane l column l % 16,
#   rows 4 (l / 16) + 0..3; D (32x32): lane l column l % 32, value r: row 8 (r / 4) + 4 (l / 32) + r % 4.
MFMA_KINDS = [("bf16_16x16x32", 16, 32, "bf16", 4), ("bf16_32x32x16", 32, 16, "bf16", 16), ("f16_16x16x32", 16, 32, "f16", 4),
              ("f16_32x32x16", 32, 16, "f16", 16), ("i8_16x16x64", 16, 64, "i8", 4), ("i8_32x32x32", 32, 32, "i8", 16)]

def enc(ty, v):
    if ty == "bf16": return struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16
    if ty == "f16": return struct.unpack("<H", struct.pack("<e", float(v)))[0]
    return v & 0xFF

def gen_mfma_p1():
    rnd = random.Random(77)
    words = []
    for name, M, K, ty, n in MFMA_KINDS:
        per = 8 if ty != "i8" else 16
        bits = 16 if ty != "i8" else 8
        for _ in range(2):  # A then B: 64 lanes x 16 bytes
            for l in range(LANES):
                vals = [rnd.randint(-4, 4) for _ in range(per)]
                x = 0
                for i, v in enumerate(vals):
                    x |= enc(ty, v) << (bits * i)
                words += [(x >> (32 * j)) & 0xFFFFFFFF for j in range(4)]
    return [words]

def dec_frag(ty, words4):
    x = sum(w << (32 * j) for j, w in enumerate(words4))
    if ty == "i8":
        return [s8 if (s8 := (x >> (8 * i)) & 255) < 128 else s8 - 256 for i in range(16)]
    out = []
    for i in range(8):
        h = (x >> (16 * i)) & 0xFFFF
        out.append(struct.unpack("<f", struct.pack("<I", h << 16))[0] if ty == "bf16" else h2f(h))
    return out

def check_mfma_p1(planes, out):
    words = planes[0]
    pos = 0
    obase = 0
    ok = True
    for name, M, K, ty, n in MFMA_KINDS:
        per = 8 if ty != "i8" else 16
        A = [[0] * K for _ in range(M)]
        B = [[0] * M for _ in range(K)]
        for which in range(2):
            for l in range(LANES):
                vals = dec_frag(ty, words[pos + 4 * l: pos + 4 * l + 4])
                for i, v in enumerate(vals):
                    kk = per * (l // M) + i
                    if which == 0:
                        A[l % M][kk] = v
                    else:
                        B[kk][l % M] = v
            pos += 4 * LANES
        D = [[sum(A[r][k] * B[k][c] for k in range(K)) for c in range(M)] for r in range(M)]
        def at(l, r):
            if n == 4:
                return D[4 * (l // 16) + r][l % 16]
            return D[8 * (r // 4) + 4 * (l // 32) + r % 4][l % 32]
        for j, mult in enumerate((1, 2, 2)):
            bad = 0
            first = None
            for l in range(LANES):
                for r in range(n):
                    got = out[obase + (j * LANES + l) * n + r]
                    want = at(l, r) * mult
                    gv = s32(got) if ty == "i8" else f32(got)
                    if gv != want:
                        bad += 1
                        first = first or (l, r, want, gv)
            form = ["zero-C", "C", "var"][j]
            if bad:
                ok = False
                print(f"FAIL {name} {form}: {bad} values, e.g. lane {first[0]} value {first[1]} want {first[2]} got {first[3]}")
            else:
                print(f"ok   {name} {form}")
        if name == "bf16_16x16x32":
            slice_ref = [[at(l, r) for r in range(4)] for l in range(LANES)]
        obase += 3 * LANES * n
    got = [[out[obase + 4 * l + r] for r in range(4)] for l in range(LANES)]
    if all(f32(got[l][r]) == slice_ref[l][r] for l in range(LANES) for r in range(4)):
        print("ok   var_slice4 views (bf16_16x16x32 from a Frag8 loop variable)")
    else:
        ok = False
        print("FAIL var_slice4 views")
    return ok

# ---------------------------------------------------------------- probe_f8f6f4_p1
# formats: (bits, exponent bits, mantissa bits, bias); 32 elements per lane packed contiguously (element v at bit v * bits).
FMT = {"fp8": (8, 4, 3, 7), "bf8": (8, 5, 2, 15), "fp6": (6, 2, 3, 1), "bf6": (6, 3, 2, 3), "fp4": (4, 2, 1, 1)}
F8F6F4_KINDS = [(16, a, b) for a, b in [("fp4", "fp4"), ("fp8", "fp4"), ("fp4", "fp8"), ("fp8", "fp8"), ("bf8", "bf8"), ("fp6", "fp6"), ("bf6", "bf6"), ("bf8", "fp6"), ("bf6", "fp8"), ("fp4", "bf8")]] + \
               [(32, a, b) for a, b in [("fp4", "fp4"), ("fp8", "fp4"), ("fp6", "bf8")]]
VALS = [0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 6.0, -6.0]

def mf_decode(fmt, code):
    bits, eb, mb, bias = FMT[fmt]
    s = -1.0 if code >> (bits - 1) & 1 else 1.0
    e = (code >> mb) & ((1 << eb) - 1)
    m = code & ((1 << mb) - 1)
    if e == 0:
        return s * m * 2.0 ** (1 - bias - mb)
    return s * (1 + m / (1 << mb)) * 2.0 ** (e - bias)

def mf_encode(fmt, v):
    bits = FMT[fmt][0]
    for code in range(1 << bits):
        if fmt == "fp8" and code & 0x7F == 0x7F:
            continue
        if mf_decode(fmt, code) == v and not (v == 0 and code):
            return code
    raise ValueError((fmt, v))

def gen_f8f6f4_p1():
    rnd = random.Random(99)
    words = []
    for M, fa, fb in F8F6F4_KINDS:
        for fmt in (fa, fb):
            bits = FMT[fmt][0]
            for l in range(LANES):
                x = 0
                for v in range(32):
                    x |= mf_encode(fmt, rnd.choice(VALS)) << (bits * v)
                words += [(x >> (32 * j)) & 0xFFFFFFFF for j in range(8)]
            words += [0] * (512 - 8 * LANES)  # pad each operand block to 2048 bytes
    return [words]

def kmap(c, G, g, v):
    return c * g + (v % c) + (v // c) * c * G

def check_f8f6f4_p1(planes, out):
    words = planes[0]
    pos = obase = 0
    ok = True
    for M, fa, fb in F8F6F4_KINDS:
        K = 128 if M == 16 else 64
        G = 64 // M
        elems = []
        for fmt in (fa, fb):
            bits = FMT[fmt][0]
            per_lane = []
            for l in range(LANES):
                x = sum(w << (32 * j) for j, w in enumerate(words[pos + 8 * l: pos + 8 * l + 8]))
                per_lane.append([mf_decode(fmt, (x >> (bits * v)) & ((1 << bits) - 1)) for v in range(32)])
            elems.append(per_lane)
            pos += 512
        n = 4 if M == 16 else 16
        def at(D, l, r):
            if n == 4:
                return D[4 * (l // 16) + r][l % 16]
            return D[8 * (r // 4) + 4 * (l // 32) + r % 4][l % 32]
        found = None
        for ca in (4, 8, 16, 32):
            for cb in (4, 8, 16, 32):
                A = [[0.0] * K for _ in range(M)]
                B = [[0.0] * M for _ in range(K)]
                for l in range(LANES):
                    for v in range(32):
                        A[l % M][kmap(ca, G, l // M, v)] = elems[0][l][v]
                        B[kmap(cb, G, l // M, v)][l % M] = elems[1][l][v]
                D = [[sum(A[r][k] * B[k][c] for k in range(K)) for c in range(M)] for r in range(M)]
                good = all(f32(out[obase + l * n + r]) == at(D, l, r) and f32(out[obase + (LANES + l) * n + r]) == 2 * at(D, l, r)
                           for l in range(LANES) for r in range(n))
                if good:
                    found = (ca, cb)
                    break
            if found:
                break
        name = f"{M}x{M}x{K} {fa} x {fb}"
        if found:
            print(f"ok   {name}: A k-chunk {found[0]}, B k-chunk {found[1]} (k = c (l / {M}) + v % c + (v / c) c {G})")
        else:
            ok = False
            print(f"FAIL {name}: no layout hypothesis matches (got lane 0: {[f32(out[obase + r]) for r in range(min(n, 4))]})")
        obase += 2 * LANES * n
    return ok

# ---------------------------------------------------------------- probe_cpol_p1: out == in (4 copy paths with cache-policy modifiers)
def gen_cpol_p1():
    return [[(i * 2654435761 + 12345) & 0xFFFFFFFF for i in range(1024)]]

def check_cpol_p1(planes, out):
    bad = [i for i in range(1024) if out[i] != planes[0][i]]
    for path in range(4):
        n = len([i for i in bad if path * 256 <= i < (path + 1) * 256])
        print(("FAIL" if n else "ok  ") + f" copy path {path}: {256 - n} / 256 dwords")
    return not bad

# ---------------------------------------------------------------- probe_cvt_p1
CVT_CASES = [("PkF32Fp8", [0], 0), ("PkF32Fp8", [0], 1), ("PkF32Bf8", [0], 1), ("F32Fp8", [0], 2), ("F32Bf8", [0], 3), ("PkBf8F32", [2, 3, 8], 0),
             ("PkBf8F32", [2, 3, 8], 1), ("ScalePkBf16Fp8", [0, 7], 1), ("ScalePkF16Fp8", [0, 7], 0), ("ScalePkBf16Bf8", [0, 7], 1),
             ("ScalePkF16Bf8", [0, 7], 0), ("ScalePkF32Fp8", [0, 7], 1), ("ScalePkF32Bf8", [0, 7], 0), ("ScalePkF32Fp4", [1, 7], 2),
             ("ScalePkBf16Fp4", [1, 7], 3), ("ScalePkF16Fp4", [1, 7], 1), ("ScalePkFp8F32", [2, 3, 7, 8], 1), ("ScalePkBf8F32", [2, 3, 7, 8], 0),
             ("ScalePkFp4F32", [2, 3, 7, 8], 2), ("ScalePkF32Fp8", [0, 9], 0), ("ScalePkBf16Fp4", [1, 10], 1), ("ScalePkFp8F32", [2, 3, 10, 8], 0),
             ("PkU8F32", [4, 8], 2), ("PkI16I32", [5, 6], 0), ("PkU16U32", [5, 6], 0)]
TWO = {"PkF32Fp8", "PkF32Bf8", "ScalePkF32Fp8", "ScalePkF32Bf8", "ScalePkF32Fp4"}

def gen_cvt_p1():
    rnd = random.Random(5)
    ok_byte = [b for b in range(256) if (b & 0x7F) != 0x7F and ((b >> 2) & 31) <= 25]
    S2 = [2 * v for v in VALS]
    planes = [[], [], [], [], [], [], [], [], [], [], []]
    for l in range(LANES):
        planes[0].append(sum(rnd.choice(ok_byte) << (8 * i) for i in range(4)))
        planes[1].append(rnd.getrandbits(32))
        planes[2].append(u32(rnd.choice(S2)))
        planes[3].append(u32(rnd.choice(S2)))
        c = [-10.0, -0.5, 0.0, 0.5, 1.5, 2.5, 127.5, 254.5, 255.0, 255.6, 300.0, 3.49][l % 12] if l < 24 else rnd.uniform(-5, 260)
        planes[4].append(u32(to_f32(c)))
        planes[5].append(rnd.choice([0, 1, 0xFFFFFFFF, 0x7FFF, 0x8000, 0xFFFF7FFF, 0x10000, 0x7FFFFFFF, 0x80000000]) if l < 32 else rnd.getrandbits(32) >> rnd.randrange(32))
        planes[6].append(rnd.getrandbits(32) >> rnd.randrange(32))
        planes[7].append(u32(2.0))
        planes[8].append(0x5A5A5A5A)
        planes[9].append(u32(1.63))
        planes[10].append(u32(3.0))
    return planes

def enc_rne(fmt, v):
    """nearest representable (ties to even code), saturating to the largest finite"""
    bits = FMT[fmt][0]
    best = None
    for code in range(1 << bits):
        if fmt == "fp8" and code & 0x7F == 0x7F:
            continue
        if fmt == "bf8" and (code >> 2) & 31 == 31:
            continue
        d = mf_decode(fmt, code)
        key = (abs(d - v), code & 1)
        if best is None or key < best[0]:
            best = (key, code)
    return best[1]

def bf16(v):
    return u32(v) >> 16

def h16(v):
    return f2h(v)

def e8m0(s):
    """the hardware uses only the exponent of an f32 scale (measured)"""
    return 2.0 ** math.floor(math.log2(abs(s)))

def cvt_ref(op, sel, x, hyp):
    """expected dword(s); hyp: 'div' (quant: value / scale, dequant: value * scale) or 'mul' (quant: value * scale)"""
    if op in ("PkF32Fp8", "PkF32Bf8", "F32Fp8", "F32Bf8"):
        fmt = "fp8" if "Fp8" in op else "bf8"
        src = x[0]
        if op.startswith("Pk"):
            return [u32(mf_decode(fmt, (src >> (16 * sel + 8 * i)) & 255)) for i in range(2)]
        return [u32(mf_decode(fmt, (src >> (8 * sel)) & 255))]
    if op == "PkBf8F32":
        w = enc_rne("bf8", f32(x[0])) | (enc_rne("bf8", f32(x[1])) << 8)
        return [(x[2] & ~(0xFFFF << (16 * sel)) & 0xFFFFFFFF) | (w << (16 * sel))]
    if op.startswith("ScalePk") and op.endswith(("Fp8", "Bf8", "Fp4")) and not op.endswith("F32"):
        src_fmt = op[-3:].lower()
        out = op[7:-3]
        s = e8m0(f32(x[1]))
        if src_fmt == "fp4":
            vals = [mf_decode("fp4", (x[0] >> (8 * sel + 4 * i)) & 15) * s for i in range(2)]
        else:
            vals = [mf_decode(src_fmt, (x[0] >> (16 * sel + 8 * i)) & 255) * s for i in range(2)]
        if out == "F32":
            return [u32(v) for v in vals]
        f = bf16 if out == "Bf16" else h16
        return [f(vals[0]) | (f(vals[1]) << 16)]
    if op in ("ScalePkFp8F32", "ScalePkBf8F32", "ScalePkFp4F32"):
        s = e8m0(f32(x[2]))
        q = (lambda v: v / s) if hyp == "div" else (lambda v: v * s)
        a, b = q(f32(x[0])), q(f32(x[1]))
        if op == "ScalePkFp4F32":
            byte = enc_rne("fp4", a) | (enc_rne("fp4", b) << 4)
            return [(x[3] & ~(0xFF << (8 * sel)) & 0xFFFFFFFF) | (byte << (8 * sel))]
        fmt = "fp8" if "Fp8" in op else "bf8"
        w = enc_rne(fmt, a) | (enc_rne(fmt, b) << 8)
        return [(x[3] & ~(0xFFFF << (16 * sel)) & 0xFFFFFFFF) | (w << (16 * sel))]
    if op == "PkU8F32":
        v = f32(x[0])
        r = round(v) if hyp == "div" else math.floor(v + 0.5)
        r = max(0, min(255, int(r)))
        return [(x[1] & ~(0xFF << (8 * sel)) & 0xFFFFFFFF) | (r << (8 * sel))]
    if op == "PkI16I32":
        c = lambda v: max(-32768, min(32767, s32(v))) & 0xFFFF
        return [c(x[0]) | (c(x[1]) << 16)]
    if op == "PkU16U32":
        c = lambda v: min(65535, v)
        return [c(x[0]) | (c(x[1]) << 16)]
    raise ValueError(op)

def check_cvt_p1(planes, out):
    k = 0
    ok = True
    for op, srcs, sel in CVT_CASES:
        n = 2 if op in TWO else 1
        res = {}
        for hyp in ("div", "mul"):
            bad = []
            for l in range(LANES):
                want = cvt_ref(op, sel, [planes[p][l] for p in srcs], hyp)
                got = [out[(k + i) * LANES + l] for i in range(n)]
                same = all((math.isnan(f32(w)) and math.isnan(f32(g))) or w == g for w, g in zip(want, got))
                if not same:
                    bad.append((l, [hex(w) for w in want], [hex(g) for g in got]))
            res[hyp] = bad
        best = min(res, key=lambda h: len(res[h]))
        bad = res[best]
        tag = f"{op} sel {sel}" + (f" ({'value / scale' if best == 'div' else 'value * scale'})" if op in ("ScalePkFp8F32", "ScalePkBf8F32", "ScalePkFp4F32") else "") + (f" ({'round-half-even' if best == 'div' else 'round-half-up'})" if op == "PkU8F32" else "")
        if bad:
            ok = False
            print(f"FAIL {tag}: {len(bad)} lanes, e.g. lane {bad[0][0]} want {bad[0][1]} got {bad[0][2]}")
        else:
            print(f"ok   {tag}")
        k += n
    return ok

# ---------------------------------------------------------------- probe_dpp_p1
DPP_CASES = [(("qp", (3, 2, 1, 0)), 15, 15, False, None), (("shl", 1), 15, 15, False, None), (("shr", 3), 15, 15, False, None), (("ror", 5), 15, 15, False, None),
             (("wshl",), 15, 15, False, None), (("wrol",), 15, 15, False, None), (("wshr",), 15, 15, False, None), (("wror",), 15, 15, False, None),
             (("mirror",), 15, 15, False, None), (("hmirror",), 15, 15, False, None), (("bc15",), 15, 15, False, None), (("bc31",), 15, 15, False, None),
             (("shr", 1), 15, 15, True, None), (("bc15",), 0xA, 15, False, None), (("shr", 2), 15, 0x5, False, None),
             (("shr", 1), 15, 15, False, "AddU32"), (("bc31",), 0xC, 15, False, "MaxI32"), (("shr", 4), 15, 15, True, "AddF32"),
             (("qp", (1, 0, 3, 2)), 15, 15, False, "MinU32"), (("hmirror",), 15, 15, False, "XorB32")]

def gen_dpp_p1():
    rnd = random.Random(11)
    return [[rnd.getrandbits(32) for _ in range(LANES)], [rnd.getrandbits(32) for _ in range(LANES)],
            [u32(float(rnd.randint(-50, 50))) for _ in range(LANES)], [u32(float(rnd.randint(-50, 50))) for _ in range(LANES)]]

def dpp_src(ctrl, i):
    """source lane of lane i, or None (invalid)"""
    r, j = i // 16 * 16, i % 16
    k = ctrl[0]
    if k == "qp": return (i & ~3) + ctrl[1][i & 3]
    if k == "shl": return i + ctrl[1] if j + ctrl[1] < 16 else None
    if k == "shr": return i - ctrl[1] if j >= ctrl[1] else None
    if k == "ror": return r + (j - ctrl[1]) % 16
    if k == "wshl": return i + 1 if i < 63 else None
    if k == "wshr": return i - 1 if i > 0 else None
    if k == "wrol": return (i + 1) % 64
    if k == "wror": return (i - 1) % 64
    if k == "mirror": return r + 15 - j
    if k == "hmirror": return i // 8 * 8 + 7 - i % 8
    # broadcasts: the rows they do not target read their own lane (measured), not an invalid source
    if k == "bc15": return r - 1 if r >= 16 else i
    if k == "bc31": return 31 if i >= 32 else i
    raise ValueError(k)

def check_dpp_p1(planes, out):
    x, y, fx, fy = planes
    ok = True
    for k, (ctrl, row, bank, bz, op) in enumerate(DPP_CASES):
        bad = []
        for i in range(LANES):
            old = (0xDEAD0000 + i) & 0xFFFFFFFF
            written = (row >> (i // 16)) & 1 and (bank >> ((i % 16) // 4)) & 1
            s = dpp_src(ctrl, i)
            if s is None and not bz:
                written = False
            if not written:
                want = old
            else:
                if op == "AddF32":
                    a = f32(fx[s]) if s is not None else 0.0
                    want = u32(a + f32(fy[i]))
                else:
                    a = x[s] if s is not None else 0
                    b = y[i]
                    if op is None: want = a
                    elif op == "AddU32": want = (a + b) & 0xFFFFFFFF
                    elif op == "MaxI32": want = max(s32(a), s32(b)) & 0xFFFFFFFF
                    elif op == "MinU32": want = min(a, b)
                    elif op == "XorB32": want = a ^ b
            got = out[k * LANES + i]
            if got != want:
                bad.append((i, hex(want), hex(got)))
        name = f"{op or 'mov'} {ctrl} row_mask {row:#x} bank_mask {bank:#x}{' bound_ctrl' if bz else ''}"
        if bad:
            ok = False
            print(f"FAIL {name}: {len(bad)} lanes, e.g. {bad[:3]}")
        else:
            print(f"ok   {name}")
    k = len(DPP_CASES)
    na = [x[i] if (i // 16) % 2 == 0 else y[i - 16] for i in range(LANES)]
    nb = [x[i + 16] if (i // 16) % 2 == 0 else y[i] for i in range(LANES)]
    for j, want in enumerate((na, nb)):
        got = out[(k + j) * LANES:(k + j + 1) * LANES]
        if got != want:
            ok = False
            print(f"FAIL permlane16_swap {'ab'[j]}: lane 0..3 got {[hex(v) for v in got[:20:4]]} want {[hex(v) for v in want[:20:4]]}")
        else:
            print(f"ok   permlane16_swap {'ab'[j]}")
    return ok

# ---------------------------------------------------------------- probe_tr_b16_p1 (layout discovery: prints the map; checked against TR_B16_MAP)
def gen_tr_b16_p1():
    return [[(2 * i) | ((2 * i + 1) << 16) for i in range(1024)]]

def tr_b16_formula(l, e, extra):
    """(source lane whose address is used, element) of value e of lane l: see check output; None until established"""
    return None

def check_tr_b16_p1(planes, out):
    ok = True
    for j in range(2):
        for l in range(LANES):
            w = out[j * 128 + 2 * l: j * 128 + 2 * l + 2]
            vals = [(w[0] & 0xFFFF), w[0] >> 16, w[1] & 0xFFFF, w[1] >> 16]
            want = [256 * j + 4 * (16 * (l // 16) + (l % 16) // 4 + 4 * e) + l % 4 for e in range(4)]
            if vals != want:
                ok = False
    print("tr_b16 map " + ("matches lane d <- element d % 4 of lanes 16 (d / 16) + (d % 16) / 4 + 4 e" if ok else "MISMATCH"))
    if ok and "-v" not in sys.argv:
        return True
    for j in range(2):
        print(f"read {j}: lane: [values / 4 (source lane), values % 4 (element)]")
        for l in range(LANES):
            w = out[j * 128 + 2 * l: j * 128 + 2 * l + 2]
            vals = [(w[0] & 0xFFFF), w[0] >> 16, w[1] & 0xFFFF, w[1] >> 16]
            base = 256 * j
            print(f"  {l:2d}: {[(v - base) // 4 for v in vals]} {[(v - base) % 4 for v in vals]}")
    return ok

# ---------------------------------------------------------------- probe_tr_b16_mfma: D = T^T x B with A from ds_read_b64_tr_b16
def tr_mfma_data():
    rnd = random.Random(21)
    T = [[rnd.randint(-4, 4) for m in range(16)] for k in range(32)]
    B = [[rnd.randint(-4, 4) for n in range(16)] for k in range(32)]
    return T, B

def gen_tr_b16_mfma():
    T, B = tr_mfma_data()
    words = []
    flat = [enc("bf16", T[k][m]) for k in range(32) for m in range(16)]
    words += [flat[2 * i] | (flat[2 * i + 1] << 16) for i in range(256)]
    for l in range(LANES):
        vals = [enc("bf16", B[8 * (l // 16) + v][l % 16]) for v in range(8)]
        words += [vals[2 * i] | (vals[2 * i + 1] << 16) for i in range(4)]
    return [words]

def check_tr_b16_mfma(planes, out):
    T, B = tr_mfma_data()
    D = [[sum(T[k][m] * B[k][n] for k in range(32)) for n in range(16)] for m in range(16)]
    bad = sum(1 for l in range(LANES) for r in range(4) if f32(out[4 * l + r]) != D[4 * (l // 16) + r][l % 16])
    print("tr_b16 -> v_mfma_f32_16x16x32_bf16 A operand (T^T): " + ("exact" if not bad else f"{bad} wrong values"))
    return bad == 0

# ---------------------------------------------------------------- probe_atomic_p1
def gen_atomic_p1():
    rnd = random.Random(31)
    return [[rnd.getrandbits(32) for _ in range(LANES)], [u32(float(rnd.randint(-20, 20))) for _ in range(LANES)]]

def check_atomic_p1(planes, out):
    x, y = planes
    P = lambda k: out[k * LANES:(k + 1) * LANES]
    ctr = [0] * 64
    for l in range(LANES):
        ctr[l % 4] += 1
        ctr[4 + l % 4] += l
    ctr[8] = max(x[l] for l in range(0, 64, 2)); ctr[9] = max(x[l] for l in range(1, 64, 2))
    ctr[10] = min(s32(x[l]) for l in range(0, 64, 2)) & 0xFFFFFFFF; ctr[11] = min(s32(x[l]) for l in range(1, 64, 2)) & 0xFFFFFFFF
    ctr[12] = 64 % 16
    ctr[13] = u32(float(sum(f32(v) for v in y)))
    ok = True
    def report(name, cond):
        nonlocal ok
        ok &= cond
        print(("ok   " if cond else "FAIL ") + name)
    for space, ret_add, ret_inc, fin in (("lds", P(0), P(2), P(3)), ("global", P(4), P(5), out[6 * LANES:7 * LANES])):
        report(f"{space} final counters (add, add no-ret, max_u32, min_i32, inc, add_f32)", list(fin) == ctr)
        if list(fin) != ctr:
            print("   got ", [hex(v) for v in fin[:14]]); print("   want", [hex(v) for v in ctr[:14]])
        report(f"{space} add returns: each counter returns 0..15 once", all(sorted(ret_add[l] for l in range(64) if l % 4 == c) == list(range(16)) for c in range(4)))
        report(f"{space} inc returns (limit 15): every 0..15 four times", sorted(ret_inc) == sorted(list(range(16)) * 4))
    mins = P(1)
    report("lds min_i32 returns: old values (<= 0x7fffffff, final the min)", all(s32(v) >= s32(ctr[10 + l % 2]) for l, v in enumerate(mins)))
    return ok

# ---------------------------------------------------------------- probe_lane_p1
def gen_lane_p1():
    rnd = random.Random(41)
    v = [rnd.getrandbits(32) for _ in range(LANES)]
    v[9] = 0
    return [v]

def check_lane_p1(planes, out):
    x = planes[0]
    lo = sum(1 << l for l in range(32) if x[l] & 1)
    hi = sum(1 << (l - 32) for l in range(32, 64) if x[l] & 1)
    full = lo | (hi << 32)
    ffbl = lambda v: (v & -v).bit_length() - 1 if v else 0xFFFFFFFF
    ffbh = lambda v: 32 - v.bit_length() if v else 0xFFFFFFFF
    perm = [None] * 64
    for l in range(64):
        perm[(l + 5) % 64] = x[l]
    want = [
        [x[5]] * 64, [x[63]] * 64, [x[7] if l == 3 else l for l in range(64)], [lo] * 64, [hi] * 64, list(range(64)),
        [bin(full & ((1 << l) - 1)).count("1") for l in range(64)], [bin(v).count("1") for v in x], [ffbl(v) for v in x], [ffbh(v) for v in x],
        [x[(7 * l) % 64] for l in range(64)], perm, [x[l ^ 1] for l in range(64)], [x[(l & ~3) + (3 - l % 4)] for l in range(64)]]
    names = ["read_lane 5", "read_lane 63", "write_lane 3", "ballot lo", "ballot hi", "mbcnt(-1,-1)", "mbcnt(ballot)", "bcnt", "ffbl", "ffbh",
             "bpermute", "permute", "swizzle xor 1", "swizzle quad_perm"]
    ok = True
    for k, (n, w) in enumerate(zip(names, want)):
        got = out[k * LANES:(k + 1) * LANES]
        good = list(got) == [v & 0xFFFFFFFF for v in w]
        ok &= good
        print(("ok   " if good else "FAIL ") + n + ("" if good else f": lanes 0..3 got {[hex(v) for v in got[:4]]} want {[hex(v & 0xFFFFFFFF) for v in w[:4]]}"))
    return ok

PROBES = {
    "probe_valu_p1": (gen_valu_p1, check_valu_p1, (len(UN) + len(BIN) + len(TRI) + len(CMPS)) * LANES * 4),
    "probe_mem_p1": (gen_mem_p1, check_mem_p1, MEM_PLANES * LANES * 4 + 2 * LANES * 32),
    "probe_f8f6f4_p1": (gen_f8f6f4_p1, check_f8f6f4_p1, sum(2 * LANES * 4 * (4 if k[0] == 16 else 16) for k in F8F6F4_KINDS)),
    "probe_cvt_p1": (gen_cvt_p1, check_cvt_p1, sum(2 if c[0] in TWO else 1 for c in CVT_CASES) * LANES * 4),
    "probe_dpp_p1": (gen_dpp_p1, check_dpp_p1, (len(DPP_CASES) + 2) * LANES * 4),
    "probe_tr_b16_p1": (gen_tr_b16_p1, check_tr_b16_p1, 1024),
    "probe_tr_b16_mfma": (gen_tr_b16_mfma, check_tr_b16_mfma, 1024),
    "probe_atomic_p1": (gen_atomic_p1, check_atomic_p1, 7 * LANES * 4),
    "probe_lane_p1": (gen_lane_p1, check_lane_p1, 14 * LANES * 4),
    "probe_cpol_p1": (gen_cpol_p1, check_cpol_p1, 4096),
    "probe_mfma_p1": (gen_mfma_p1, check_mfma_p1, sum(3 * LANES * 4 * k[4] for k in MFMA_KINDS) + LANES * 16),
}

def main():
    mode, probe, inp = sys.argv[1], sys.argv[2], sys.argv[3]
    gen, chk, out_bytes = PROBES[probe]
    planes = gen()
    if mode == "gen":
        with open(inp, "wb") as f:
            for p in planes:
                f.write(struct.pack(f"<{len(p)}I", *p))
        print(out_bytes)
        return
    with open(sys.argv[4], "rb") as f:
        data = f.read()
    out = list(struct.unpack(f"<{len(data) // 4}I", data))
    sys.exit(0 if chk(planes, out) else 1)

main()
