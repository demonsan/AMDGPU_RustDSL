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

PROBES = {
    "probe_valu_p1": (gen_valu_p1, check_valu_p1, (len(UN) + len(BIN) + len(TRI) + len(CMPS)) * LANES * 4),
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
