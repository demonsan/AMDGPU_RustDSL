#!/usr/bin/env python3
"""GPU-node check of the amdgpu_jit ops (torch-free, stdlib only): each op on random data against a Python reference that follows the
C++ host reference of its kernel family (tests/gpu/*_host.cpp: formulas and tolerances). Small shapes (the references are pure Python).

    HIP_VISIBLE_DEVICES=0 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/ops_check.py [name-substring ...]
"""
import math
import os
import random
import struct
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amdgpu_jit as aj  # noqa: E402

R = random.Random(7)


# ------------------------------------------------------------------------------------------------------------- bf16 / fp8 / fp4 helpers
def bf16_bits(x):
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16(h):
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def rbf(x):
    """round to bf16"""
    return bf16(bf16_bits(x))


def e4m3(c):
    s, e, m = c >> 7, (c >> 3) & 15, c & 7
    v = math.ldexp(m / 8.0, -6) if e == 0 else math.ldexp(1.0 + m / 8.0, e - 7)
    return -v if s else v


def e2m1(c):
    v = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][c & 7]
    return -v if c & 8 else v


def up_bf16(vals, shape):
    return aj.DeviceBuffer(shape, "bfloat16", data=struct.pack(f"<{len(vals)}H", *[bf16_bits(v) for v in vals]))


def rand_bf16(n, sd=1.0):
    return [rbf(R.gauss(0, sd)) for _ in range(n)]


def down_bf16(buf, n):
    return [bf16(h) for h in struct.unpack(f"<{n}H", buf.download(2 * n))]


def down_f32(buf, n):
    return list(struct.unpack(f"<{n}f", buf.download(4 * n)))


def down_u8(buf, n):
    return list(buf.download(n))


class Res:
    def __init__(self, name):
        self.name, self.bad, self.cnt, self.note = name, 0, 0, ""

    def cmp(self, got, ref, tol):
        self.cnt += 1
        if not abs(got - ref) <= tol:
            if self.bad < 3:
                print(f"    bad: got {got} ref {ref} tol {tol}")
            self.bad += 1


def rms_ref(x, g, N, eps):
    ss = sum(v * v for v in x)
    inv = 1.0 / math.sqrt(ss / N + eps)
    return [x[k] * inv * g[k] for k in range(N)]


# ------------------------------------------------------------------------------------------------------------- checks
def check_select_f32():
    """The compiler keeps the f32 select semantics (docs/peephole-audit.md): select(x > y, x, y) with a NaN y returns the NaN, with
    +0 / -0 the compare's choice - a `v_max_f32` fold returns x / its own zero (the case the old fold got wrong)."""
    r = Res("select_f32 (NaN / signed zero)")
    nan, inf = float("nan"), float("inf")
    pairs = [(1.0, nan), (nan, 1.0), (0.0, -0.0), (-0.0, 0.0), (2.0, 1.0), (1.0, 2.0), (-inf, nan), (nan, nan)]
    xs = [pairs[i % len(pairs)][0] for i in range(64)]
    ys = [pairs[i % len(pairs)][1] for i in range(64)]
    din = aj.DeviceBuffer((128,), "float32", data=struct.pack("<128f", *(xs + ys)))
    dout = aj.DeviceBuffer((192,), "float32")
    aj.launch("probe_select_f32", args={"in": din, "out": dout}, grid=(1, 1, 1))
    got = down_f32(dout, 192)
    bits = lambda v: struct.pack("<f", v)
    for i in range(64):
        x, y = xs[i], ys[i]
        for k, ref in enumerate([x if x > y else y, x if x < y else y]):
            r.cnt += 1
            g = got[64 * k + i]
            same = (math.isnan(g) and math.isnan(ref)) or bits(g) == bits(ref)
            if not same:
                if r.bad < 3:
                    print(f"    bad: plane {k} x={x} y={y}: got {g} ref {ref}")
                r.bad += 1
    return r


def check_rmsnorm(M=32, N=512, eps=1e-6):
    r = Res(f"rmsnorm M={M} N={N}")
    x, g = rand_bf16(M * N), rand_bf16(N)
    dx, dg, do = up_bf16(x, (M, N)), up_bf16(g, (N,)), aj.DeviceBuffer((M, N), "bfloat16")
    aj.rmsnorm(dx, dg, do, eps=eps)
    out = down_bf16(do, M * N)
    for m in range(M):
        y = rms_ref(x[m * N:(m + 1) * N], g, N, eps)
        for k in range(N):
            r.cmp(out[m * N + k], y[k], abs(y[k]) / 64 + 1e-3)
    return r


def check_add_rmsnorm(M=32, N=512, eps=1e-6, quant=False):
    r = Res(f"add_rmsnorm{'_quant_fp8' if quant else ''} M={M} N={N}")
    x, res, g = rand_bf16(M * N), rand_bf16(M * N), rand_bf16(N)
    dx, dr, dg = up_bf16(x, (M, N)), up_bf16(res, (M, N)), up_bf16(g, (N,))
    dro = aj.DeviceBuffer((M, N), "bfloat16")
    if quant:
        do, ds = aj.DeviceBuffer((M, N), "uint8"), aj.DeviceBuffer((M,), "float32")
        aj.add_rmsnorm_quant_fp8(dx, dr, dg, do, dro, ds, eps=eps)
        q, sc = down_u8(do, M * N), down_f32(ds, M)
    else:
        do = aj.DeviceBuffer((M, N), "bfloat16")
        aj.add_rmsnorm(dx, dr, dg, do, dro, eps=eps)
        out = down_bf16(do, M * N)
    ro = down_bf16(dro, M * N)
    for m in range(M):
        s = [rbf(x[m * N + k] + res[m * N + k]) for k in range(N)]
        for k in range(N):
            r.cmp(ro[m * N + k], s[k], 0)
        y = rms_ref(s, g, N, eps)
        if quant:
            sref = max(max(abs(v) for v in y), 1e-12) / 448
            r.cmp(sc[m], sref, sref * 1e-3)
            for k in range(N):
                r.cmp(e4m3(q[m * N + k]) * sc[m], y[k], abs(y[k]) / 16 * 1.05 + sref * 2 ** -9 + abs(y[k]) / 64)
        else:
            for k in range(N):
                r.cmp(out[m * N + k], y[k], abs(y[k]) / 64 + 1e-3)
    return r


def check_rmsnorm_quant_fp8(M=32, N=512, eps=1e-6):
    r = Res(f"rmsnorm_quant_fp8 M={M} N={N}")
    x, g = rand_bf16(M * N), rand_bf16(N)
    dx, dg, dq, ds = up_bf16(x, (M, N)), up_bf16(g, (N,)), aj.DeviceBuffer((M, N), "uint8"), aj.DeviceBuffer((M,), "float32")
    aj.rmsnorm_quant_fp8(dx, dg, dq, ds, eps=eps)
    q, sc = down_u8(dq, M * N), down_f32(ds, M)
    for m in range(M):
        y = rms_ref(x[m * N:(m + 1) * N], g, N, eps)
        sref = max(max(abs(v) for v in y), 1e-12) / 448
        r.cmp(sc[m], sref, sref * 1e-3)
        for k in range(N):
            r.cmp(e4m3(q[m * N + k]) * sc[m], y[k], abs(y[k]) / 16 * 1.02 + sref * 2 ** -9)
    return r


def check_rmsnorm_quant_mxfp4(M=32, N=512, eps=1e-6):
    r = Res(f"rmsnorm_quant_mxfp4 M={M} N={N}")
    x, g = rand_bf16(M * N), rand_bf16(N)
    dx, dg = up_bf16(x, (M, N)), up_bf16(g, (N,))
    dq, ds = aj.DeviceBuffer((M, N // 2), "uint8"), aj.DeviceBuffer((M, N // 32), "uint8")
    aj.rmsnorm_quant_mxfp4(dx, dg, dq, ds, eps=eps)
    q, sc = down_u8(dq, M * N // 2), down_u8(ds, M * N // 32)
    for m in range(M):
        y = rms_ref(x[m * N:(m + 1) * N], g, N, eps)
        for k in range(N):
            byte = q[(m * N + k) // 2]
            c = byte & 15 if k % 2 == 0 else byte >> 4
            # the scales are in the shuffled a4w4 layout (tests/gpu/mxnorm_host.cpp): group G of row m at
            # (((m / 16) TP + G / 16) 64 + 16 (G % 4) + m % 16) 4 + (G % 16) / 4, TP = N / 512
            G, TP = k // 32, N // 512
            w = G % 16
            s = 2.0 ** (max(sc[(((m // 16) * TP + G // 16) * 64 + 16 * (w % 4) + m % 16) * 4 + w // 4], 1) - 127)
            # e2m1 steps are at most 2 units of the block scale (between 4 and 6); saturation at 6 s
            r.cmp(e2m1(c) * s, max(-6 * s, min(6 * s, y[k])), 1.01 * s + 1e-30)
    return r


def check_quant_token(M=32, N=512, dtype="fp8"):
    r = Res(f"quant_per_token {dtype} M={M} N={N}")
    x = rand_bf16(M * N)
    dx, dq, ds = up_bf16(x, (M, N)), aj.DeviceBuffer((M, N), "uint8"), aj.DeviceBuffer((M,), "float32")
    aj.quant_per_token(dx, dq, ds, dtype=dtype)
    q, sc = down_u8(dq, M * N), down_f32(ds, M)
    qmax = 448.0 if dtype == "fp8" else 127.0
    for m in range(M):
        row = x[m * N:(m + 1) * N]
        sref = max(max(abs(v) for v in row), 1e-12) / qmax
        r.cmp(sc[m], sref, sref * 1e-4)
        for k in range(N):
            d = (e4m3(q[m * N + k]) if dtype == "fp8" else (q[m * N + k] - 256 if q[m * N + k] > 127 else q[m * N + k])) * sc[m]
            tol = sref * 0.5 * 1.01 + 1e-6 if dtype == "int8" else abs(row[k]) / 16 * 1.02 + sref * 2 ** -9
            r.cmp(d, row[k], tol)
    return r


def check_quant_group(M=32, N=512):
    r = Res(f"quant_per_group128 M={M} N={N}")
    x = rand_bf16(M * N)
    dx, dq, ds = up_bf16(x, (M, N)), aj.DeviceBuffer((M, N), "uint8"), aj.DeviceBuffer((M, N // 128), "float32")
    aj.quant_per_group128(dx, dq, ds)
    q, sc = down_u8(dq, M * N), down_f32(ds, M * N // 128)
    for m in range(M):
        for gi in range(N // 128):
            grp = x[m * N + gi * 128:m * N + gi * 128 + 128]
            sref = max(max(abs(v) for v in grp), 1e-12) / 448
            s = sc[m * (N // 128) + gi]
            r.cmp(s, sref, sref * 1e-4)
            for k in range(128):
                r.cmp(e4m3(q[m * N + gi * 128 + k]) * s, grp[k], abs(grp[k]) / 16 * 1.02 + sref * 2 ** -9)
    return r


def act(a, kind, g=None):
    if kind == "swiglu_oai":
        gt, lin = min(a, 7.0), max(-7.0, min(g, 7.0))
        return gt / (1 + math.exp(-1.702 * gt)) * (lin + 1.0)
    if kind == "silu":
        return a / (1 + math.exp(-a))
    if kind == "gelu":
        return 0.5 * a * (1 + math.erf(a / math.sqrt(2)))
    return 0.5 * a * (1 + math.tanh(0.7978845608028654 * (a + 0.044715 * a ** 3)))


def check_act_mul(M=16, D=512, kind="silu"):
    r = Res(f"act_mul {kind} M={M} D={D}")
    x = rand_bf16(M * 2 * D)
    dx, do = up_bf16(x, (M, 2 * D)), aj.DeviceBuffer((M, D), "bfloat16")
    aj.act_mul(dx, do, act=kind)
    out = down_bf16(do, M * D)
    for m in range(M):
        for k in range(D):
            a_, g_ = x[m * 2 * D + k], x[m * 2 * D + D + k]
            ref = act(a_, kind, g_) if kind == "swiglu_oai" else act(a_, kind) * g_
            r.cmp(out[m * D + k], ref, abs(ref) / 128 * 1.05 + 1e-6)
    return r


def check_quant_static(n=8192, scale=0.01):
    r = Res(f"quant_static_fp8 n={n}")
    x = rand_bf16(n)
    dx, do = up_bf16(x, (n,)), aj.DeviceBuffer((n,), "uint8")
    aj.quant_static_fp8(dx, do, scale)
    q = down_u8(do, n)
    for i in range(n):
        y = max(-448.0, min(448.0, x[i] / scale))
        r.cmp(e4m3(q[i]), y, abs(y) / 16 * 1.02 + 2 ** -9)
    return r


def check_silu_mul_quant(M=16, D=512):
    r = Res(f"silu_mul_quant_fp8 M={M} D={D}")
    x = rand_bf16(M * 2 * D)
    dx, dq, ds = up_bf16(x, (M, 2 * D)), aj.DeviceBuffer((M, D), "uint8"), aj.DeviceBuffer((M,), "float32")
    aj.silu_mul_quant_fp8(dx, dq, ds)
    q, sc = down_u8(dq, M * D), down_f32(ds, M)
    for m in range(M):
        y = [act(x[m * 2 * D + k], "silu") * x[m * 2 * D + D + k] for k in range(D)]
        sref = max(max(abs(v) for v in y), 1e-12) / 448
        r.cmp(sc[m], sref, sref * 1e-2)
        for k in range(D):
            r.cmp(e4m3(q[m * D + k]) * sc[m], y[k], abs(y[k]) / 16 * 1.05 + sref * 2 ** -9 + abs(y[k]) / 64)
    return r



def up_i32(vals, shape, i64=False):
    fmt = "q" if i64 else "i"
    return aj.DeviceBuffer(shape, "int64" if i64 else "int32", itemsize=8 if i64 else 4, data=struct.pack(f"<{len(vals)}{fmt}", *vals))


def cos_sin_table(P, R):
    cs = [0.0] * (P * R)
    for p in range(P):
        for i in range(R // 2):
            ang = p * 10000.0 ** (-2.0 * i / R)
            cs[p * R + i], cs[p * R + R // 2 + i] = rbf(math.cos(ang)), rbf(math.sin(ang))
    return cs


def rope_ref(xv, cs, pos, P, D, neox):
    if pos >= P:
        return [0.0] * D
    C = lambda j: cs[pos * D + j]
    S = lambda j: cs[pos * D + D // 2 + j]
    out = []
    for i in range(D):
        if neox:
            j = i % (D // 2)
            a, c = xv[j], xv[D // 2 + j]
            out.append(a * C(j) - c * S(j) if i < D // 2 else c * C(j) + a * S(j))
        else:
            j = i // 2
            a, c = xv[2 * j], xv[2 * j + 1]
            out.append(a * C(j) - c * S(j) if i % 2 == 0 else c * C(j) + a * S(j))
    return out


def check_rope(T=16, H=8, D=128, P=64, style="neox", norm=False, eps=1e-6):
    r = Res(f"{'qknorm_' if norm else ''}rope {style} T={T} H={H} D={D}")
    x, cs, g = rand_bf16(T * H * D), cos_sin_table(P, D), rand_bf16(D)
    pos = [R.randrange(P) for _ in range(T)]
    pos[-1] = P + 5
    dx, do, dp, dc, dg = up_bf16(x, (T, H, D)), aj.DeviceBuffer((T, H, D), "bfloat16"), up_i32(pos, (T,)), up_bf16(cs, (P, D)), up_bf16(g, (D,))
    if norm:
        aj.qknorm_rope(dx, do, dp, dc, dg, eps=eps, style=style)
    else:
        aj.rope(dx, do, dp, dc, style=style)
    out = down_bf16(do, T * H * D)
    for tt in range(T):
        for h in range(H):
            xv = x[(tt * H + h) * D:(tt * H + h + 1) * D]
            if norm:
                xv = rms_ref(xv, g, D, eps)
            y = rope_ref(xv, cs, pos[tt], P, D, style == "neox")
            for i in range(D):
                r.cmp(out[(tt * H + h) * D + i], y[i], abs(y[i]) / 128 * 1.05 + 2e-3 + (abs(y[i]) / 64 if norm else 0))
    return r


def check_qknorm_rope_kvcache(T=16, H=8, D=128, P=64, S=64, fp8=False, i64=False, eps=1e-6):
    r = Res(f"qknorm_rope_kvcache {'fp8' if fp8 else 'bf16'}{' i64' if i64 else ''} T={T} H={H} D={D} S={S}")
    k, v, cs, g = rand_bf16(T * H * D), rand_bf16(T * H * D), cos_sin_table(P, D), rand_bf16(D)
    pos = [R.randrange(P) for _ in range(T)]
    perm = list(range(S))
    R.shuffle(perm)
    slots = [(-1 if t % 9 == 4 else perm[t]) for t in range(T)]
    dk, dv, do = up_bf16(k, (T, H, D)), up_bf16(v, (T, H, D)), aj.DeviceBuffer((T, H, D), "bfloat16")
    dp, dc, dg, ds = up_i32(pos, (T,)), up_bf16(cs, (P, D)), up_bf16(g, (D,)), up_i32(slots, (T,), i64)
    esz = 1 if fp8 else 2
    kc, vc = aj.DeviceBuffer((S, H, D), "uint8" if fp8 else "bfloat16", itemsize=esz), aj.DeviceBuffer((S, H, D), "uint8" if fp8 else "bfloat16", itemsize=esz)
    kc.fill(0x55)
    vc.fill(0x55)
    ksc, vsc = 0.05, 0.02
    aj.qknorm_rope_kvcache(dk, do, dp, dc, dg, dv, kc, vc, ds, eps=eps, k_scale=ksc if fp8 else None, v_scale=vsc if fp8 else None)
    kb, vb = kc.download(), vc.download()
    for tt in range(T):
        if slots[tt] < 0:
            continue
        for h in range(H):
            y = rope_ref(rms_ref(k[(tt * H + h) * D:(tt * H + h + 1) * D], g, D, eps), cs, pos[tt], P, D, True)
            for i in range(D):
                idx = (slots[tt] * H + h) * D + i
                yv = v[(tt * H + h) * D + i]
                if fp8:
                    gk, gv = e4m3(kb[idx]) * ksc, e4m3(vb[idx]) * vsc
                    sk, sv = max(-448 * ksc, min(448 * ksc, y[i])), max(-448 * vsc, min(448 * vsc, yv))
                    r.cmp(gk, sk, abs(sk) / 16 * 1.05 + ksc * 2 ** -9 + abs(sk) / 64)
                    r.cmp(gv, sv, abs(sv) / 16 * 1.05 + vsc * 2 ** -9)
                else:
                    gk = bf16(struct.unpack_from("<H", kb, idx * 2)[0])
                    gv = bf16(struct.unpack_from("<H", vb, idx * 2)[0])
                    r.cmp(gk, y[i], abs(y[i]) / 128 * 1.05 + 2e-3 + abs(y[i]) / 64)
                    r.cmp(gv, yv, 0)
    return r


def check_reshape_cache(T=24, H=8, D=128, S=64, layout="flash", BS=16):
    r = Res(f"reshape_cache {layout} T={T} H={H} D={D} S={S}")
    k, v = rand_bf16(T * H * D), rand_bf16(T * H * D)
    perm = list(range(S))
    R.shuffle(perm)
    slots = [(-1 if t % 7 == 3 else perm[t]) for t in range(T)]
    dk, dv, ds = up_bf16(k, (T, H, D)), up_bf16(v, (T, H, D)), up_i32(slots, (T,))
    if layout == "flash":
        kc, vc = aj.DeviceBuffer((S, H, D), "bfloat16"), aj.DeviceBuffer((S, H, D), "bfloat16")
    else:
        NB = S // BS
        kc, vc = aj.DeviceBuffer((NB, H, D // 8, BS, 8), "bfloat16"), aj.DeviceBuffer((NB, H, D, BS), "bfloat16")
    kc.fill(0x55)
    vc.fill(0x55)
    aj.reshape_cache(dk, dv, kc, vc, ds, layout=layout)
    kb, vb = kc.download(), vc.download()
    for tt in range(T):
        s = slots[tt]
        if s < 0:
            continue
        blk, off = s // BS, s % BS
        for h in range(H):
            for d in range(D):
                if layout == "flash":
                    ki = vi = (s * H + h) * D + d
                else:
                    ki = (((blk * H + h) * (D // 8) + d // 8) * BS + off) * 8 + d % 8
                    vi = ((blk * H + h) * D + d) * BS + off
                r.cmp(bf16(struct.unpack_from("<H", kb, ki * 2)[0]), k[(tt * H + h) * D + d], 0)
                r.cmp(bf16(struct.unpack_from("<H", vb, vi * 2)[0]), v[(tt * H + h) * D + d], 0)
    return r


def check_topk(T=32, E=128, K=8, renorm=False):
    r = Res(f"topk_softmax{'_renorm' if renorm else ''} T={T} E={E} K={K}")
    gt = rand_bf16(T * E, 2.0)
    dg, dw, di = up_bf16(gt, (T, E)), aj.DeviceBuffer((T, K), "float32"), aj.DeviceBuffer((T, K), "int32")
    aj.topk_softmax(dg, dw, di, renorm=renorm)
    w, ix = down_f32(dw, T * K), list(struct.unpack(f"<{T * K}i", di.download()))
    for tt in range(T):
        row = gt[tt * E:(tt + 1) * E]
        mx = max(row)
        p = [math.exp(x - mx) for x in row]
        s = sum(p)
        p = [x / s for x in p]
        order = sorted(range(E), key=lambda e: (-p[e], e))[:K]
        tot = sum(p[e] for e in order)
        for j in range(K):
            r.cmp(ix[tt * K + j], order[j], 0)
            wref = p[order[j]] / tot if renorm else p[order[j]]
            r.cmp(w[tt * K + j], wref, 1e-5 + 2e-5 * wref)
    return r


def check_moe_sorting(M=40, E=64, K=2, BM=16):
    r = Res(f"moe_sorting M={M} E={E} K={K} BM={BM}")
    ids = [R.randrange(E) for _ in range(M * K)]
    for i in range(0, M * K, 5):
        ids[i] = 0  # skew
    w = [R.random() for _ in range(M * K)]
    maxp, nt, nnv, nts = aj.moe_sorting_sizes(M, E, K, BM)
    di, dw = up_i32(ids, (M, K)), aj.DeviceBuffer((M, K), "float32", data=struct.pack(f"<{M * K}f", *w))
    dsid, dsw = aj.DeviceBuffer((maxp,), "uint32"), aj.DeviceBuffer((maxp,), "float32")
    deid, dnv, dts = aj.DeviceBuffer((nt,), "int32"), aj.DeviceBuffer((nnv,), "int32"), aj.DeviceBuffer((nts,), "int32")
    aj.moe_sorting(di, dw, dsid, dsw, deid, dnv, dts, num_experts=E, block_m=BM)
    sid = struct.unpack(f"<{maxp}I", dsid.download())
    sw = struct.unpack(f"<{maxp}f", dsw.download())
    eid, nv, ts = struct.unpack(f"<{nt}i", deid.download()), struct.unpack("<2i", dnv.download()), struct.unpack(f"<{nts}i", dts.download())
    cnt = [ids.count(e) for e in range(E)]
    off = [0]
    for e in range(E):
        off.append(off[-1] + -(-cnt[e] // BM) * BM)
    total = off[E]
    r.cmp(nv[0], total, 0)
    r.cmp(nv[1], M, 0)
    for e in range(E + 1):
        r.cmp(ts[e], off[e] // BM, 0)
    for j in range(maxp // BM):  # (the allocation is ceil(maxp / BM) tiles; the kernel writes the floor, as tests/gpu/moe_sorting_host.cpp checks)
        ref = E
        if j * BM < total:
            ref = max(e for e in range(E) if off[e] <= j * BM and off[e + 1] > j * BM)
        r.cmp(eid[j], ref, 0)
    sentinel = (K << 24) | M
    for e in range(E):
        got = sorted((sid[q], struct.pack("<f", sw[q])) for q in range(off[e], off[e] + cnt[e]))
        ref = sorted((((i % K) << 24) | (i // K), struct.pack("<f", w[i])) for i in range(M * K) if ids[i] == e)
        r.cmp(0 if got == ref else 1, 0, 0)
        for q in range(off[e] + cnt[e], off[e + 1]):
            r.cmp(0 if (sid[q] == sentinel and sw[q] == 0.0) else 1, 0, 0)
    return r


def check_moe_reduce(T=8, K=8, D=512, weighted=True):
    r = Res(f"moe_{'reduce' if weighted else 'combine'} T={T} K={K} D={D}")
    x = rand_bf16(T * K * D)
    w = [R.random() for _ in range(T * K)]
    dx, do = up_bf16(x, (T, K, D)), aj.DeviceBuffer((T, D), "bfloat16")
    dw = aj.DeviceBuffer((T, K), "float32", data=struct.pack(f"<{T * K}f", *w)) if weighted else None
    aj.moe_reduce(dx, do, dw)
    out = down_bf16(do, T * D)
    for tt in range(T):
        for d in range(D):
            terms = [x[(tt * K + k) * D + d] * (w[tt * K + k] if weighted else 1.0) for k in range(K)]
            ref, mag = sum(terms), sum(abs(a) for a in terms)
            r.cmp(out[tt * D + d], ref, abs(ref) / 128 * 1.02 + 1e-6 * mag + 1e-30)
    return r


# ------------------------------------------------------------------------------------------------------------- compute ops
def fp8_codes(n):
    """random e4m3 codes with exponents 4..10 (no NaN), as tests/gpu/vendor_gemm_host.cpp"""
    return [((R.randrange(2) << 7) | ((4 + R.randrange(7)) << 3) | R.randrange(8)) for _ in range(n)]


def up_bytes(b, shape, dtype="uint8"):
    return aj.DeviceBuffer(shape, dtype, itemsize=1, nbytes=len(b), data=bytes(b))


def up_f32(vals, shape):
    return aj.DeviceBuffer(shape, "float32", data=struct.pack(f"<{len(vals)}f", *vals))


def check_gemm(kind="bf16", M=256, N=256, K=256, tile="auto", rows=4, run=None):
    """``run(da, db, do, dsa, dsb, K)`` (default: ``aj.gemm``) launches the GEMM and returns (kernel, S): graph_check.py runs the L1 plan."""
    r = Res(f"gemm {kind} M={M} N={N} K={K} tile={tile}")
    sa = sb = None
    if kind == "bf16":
        af, bf = [rbf(R.uniform(-1, 1)) for _ in range(M * K)], [rbf(R.uniform(-1, 1)) for _ in range(N * K)]
        da = up_bf16(af, (M, K))
        db = up_bytes(aj.shuffle_weight(struct.pack(f"<{N * K}H", *[bf16_bits(v) for v in bf]), N, K, 2), (N, K * 2))
    elif kind in ("fp8_rowcol", "fp8_blockscale"):
        ac, bc = fp8_codes(M * K), fp8_codes(N * K)
        af, bf = [e4m3(c) for c in ac], [e4m3(c) for c in bc]
        da, db = up_bytes(ac, (M, K)), up_bytes(aj.shuffle_weight(bytes(bc), N, K, 1), (N, K))
        if kind == "fp8_rowcol":
            sa, sb = [1e-3 * (1 + R.randrange(100) / 100) for _ in range(M)], [1e-3 * (1 + R.randrange(100) / 100) for _ in range(N)]
        else:
            sa = [2.0 ** -R.randrange(4) * (1 + R.randrange(8) / 8) for _ in range(M * (K // 128))]
            sb = [2.0 ** -R.randrange(4) * (1 + R.randrange(8) / 8) for _ in range(N // 128 * (K // 128))]
    else:  # mxfp4
        ab, bb = bytes(R.randrange(256) for _ in range(M * K // 2)), bytes(R.randrange(256) for _ in range(N * K // 2))
        af = [e2m1((ab[i // 2] >> (4 * (i % 2))) & 15) for i in range(M * K)]
        bf = [e2m1((bb[i // 2] >> (4 * (i % 2))) & 15) for i in range(N * K)]
        sa8, sb8 = bytes(120 + R.randrange(10) for _ in range(M * K // 32)), bytes(120 + R.randrange(10) for _ in range(N * K // 32))
        da, db = up_bytes(ab, (M, K // 2)), up_bytes(aj.shuffle_weight(bb, N, K // 2, 1), (N, K // 2))
        dsa, dsb = up_bytes(aj.shuffle_mx_scales(sa8, M, K), (M, K // 32)), up_bytes(aj.shuffle_mx_scales(sb8, N, K), (N, K // 32))
    if kind in ("fp8_rowcol", "fp8_blockscale"):
        dsa, dsb = up_f32(sa, (len(sa),)), up_f32(sb, (len(sb),))
    elif kind == "bf16":
        dsa = dsb = None
    do = aj.DeviceBuffer((M, N), "bfloat16")
    if run is None:
        name, S = aj.gemm(da, db, do, kind=kind, a_scale=dsa, b_scale=dsb, K=K, tile=tile)
    else:
        name, S = run(da, db, do, dsa, dsb, K)
    r.name += f" -> {name} S={S}"
    out = down_bf16(do, M * N)
    for ri in range(rows):
        m = ri * (M - 1) // (rows - 1)
        for n in range(N):
            ref = mag = 0.0
            if kind == "fp8_blockscale":
                for kb in range(K // 128):
                    p_ = sum(af[m * K + k] * bf[n * K + k] for k in range(kb * 128, kb * 128 + 128))
                    s = sa[m * (K // 128) + kb] * sb[(n // 128) * (K // 128) + kb]
                    ref += p_ * s
                    mag += abs(p_ * s)
            elif kind == "mxfp4":
                for g in range(K // 32):
                    p_ = sum(af[m * K + k] * bf[n * K + k] for k in range(g * 32, g * 32 + 32))
                    s = 2.0 ** (sa8[m * (K // 32) + g] - 127) * 2.0 ** (sb8[n * (K // 32) + g] - 127)
                    ref += p_ * s
                    mag += abs(p_ * s)
            else:
                terms = [af[m * K + k] * bf[n * K + k] for k in range(K)]
                ref, mag = sum(terms), sum(abs(x) for x in terms)
                if kind == "fp8_rowcol":
                    ref, mag = ref * sa[m] * sb[n], mag * sa[m] * sb[n]
            # fp32 accumulation over K terms (+ one rounding per scaled block). The f8f6f4 MFMA sums a 128-term block with an internal precision of
            # about 2^-10 of the block's term magnitude (visible under cancellation; also with unit scales and a single K block): the 2^-8 mag term
            tol = 1e-2 * abs(ref) + (4.0 if kind in ("fp8_blockscale", "mxfp4") else 2.0) * K * 2 ** -24 * mag + 1e-6
            if kind == "fp8_blockscale":
                tol += 2 ** -8 * mag
            r.cmp(out[m * N + n], ref, tol)
    return r


def attn_ref_row(q, k, v, b, h, qi, SQ, SK, H, HK, D, causal, scale):
    hk = h // (H // HK)
    qv = q[((b * SQ + qi) * H + h) * D:((b * SQ + qi) * H + h + 1) * D]
    kmax = min(SK - 1, qi + SK - SQ) if causal else SK - 1
    s = []
    for kk in range(kmax + 1):
        base = ((b * SK + kk) * HK + hk) * D
        s.append(sum(qv[d] * k[base + d] for d in range(D)) * scale)
    m = max(s)
    p = [math.exp(x - m) for x in s]
    l_ = sum(p)
    o = [0.0] * D
    for kk, w in enumerate(p):
        base = ((b * SK + kk) * HK + hk) * D
        for d in range(D):
            o[d] += w * v[base + d]
    return [x / l_ for x in o], m + math.log(l_)


def check_prefill(B=1, H=4, HK=2, SQ=384, SK=1024, causal=False, rows=6):
    D = 128
    r = Res(f"prefill_bf16 B={B} H={H} HK={HK} SQ={SQ} SK={SK} causal={int(causal)}")
    q, k, v = rand_bf16(B * SQ * H * D), rand_bf16(B * SK * HK * D), rand_bf16(B * SK * HK * D)
    dq, dk, dv = up_bf16(q, (B, SQ, H, D)), up_bf16(k, (B, SK, HK, D)), up_bf16(v, (B, SK, HK, D))
    do, dl = aj.DeviceBuffer((B, SQ, H, D), "bfloat16"), aj.DeviceBuffer((B, H, SQ), "float32")
    name, S = aj.prefill_bf16(dq, dk, dv, do, causal=causal, lse=dl)
    r.name += f" -> {name} S={S}"
    out, lse = down_bf16(do, B * SQ * H * D), down_f32(dl, B * H * SQ)
    scale = 1.0 / math.sqrt(D)
    for c in range(rows):
        b, h = R.randrange(B), (0 if c == 0 else H - 1 if c == 1 else R.randrange(H))
        qi = 0 if c == 0 else SQ - 1 if c == 1 else R.randrange(SQ)
        o, l_ = attn_ref_row(q, k, v, b, h, qi, SQ, SK, H, HK, D, causal, scale)
        for d in range(D):
            r.cmp(out[((b * SQ + qi) * H + h) * D + d], o[d], 1e-2 * abs(o[d]) + 1e-2)
        r.cmp(lse[(b * H + h) * SQ + qi], l_, 1e-3 * abs(l_) + 1e-3)
    return r


def check_moe(M=16, K=2, E=8, D=1024, I=256, fp8=False, tokens=3):
    r = Res(f"fused_moe {'fp8' if fp8 else 'bf16'} M={M} K={K} E={E} D={D} I={I}")
    ids = []
    for _ in range(M):
        ids += R.sample(range(E), K)
    wts = [R.random() for _ in range(M * K)]
    if fp8:
        xc, w1c, w2c = fp8_codes(M * D), fp8_codes(E * 2 * I * D), fp8_codes(E * D * I)
        xs = [0.01 * (1 + R.random()) for _ in range(M)]
        w1s = [0.002 * (1 + R.random()) for _ in range(E * 2 * I)]
        w2s = [0.002 * (1 + R.random()) for _ in range(E * D)]
        xf = [e4m3(c) * xs[i // D] for i, c in enumerate(xc)]
        w1f = [e4m3(c) * w1s[i // D] for i, c in enumerate(w1c)]
        w2f = [e4m3(c) * w2s[i // I] for i, c in enumerate(w2c)]
        dx = up_bytes(xc, (M, D))
        dw1 = up_bytes(aj.shuffle_weight(aj.moe_w1_interleave(bytes(w1c), E, I, D, 1), E * 2 * I, D, 1), (E * 2 * I, D))
        dw2 = up_bytes(aj.shuffle_weight(bytes(w2c), E * D, I, 1), (E * D, I))
        dxs = up_f32(xs, (M,))
        dw1s = aj.DeviceBuffer((E * 2 * I,), "float32", data=aj.moe_w1_interleave(struct.pack(f"<{E * 2 * I}f", *w1s), E, I, 1, 4))
        dw2s = up_f32(w2s, (E * D,))
    else:
        xf = rand_bf16(M * D)
        w1f = [rbf(R.gauss(0, 0.05)) for _ in range(E * 2 * I * D)]
        w2f = [rbf(R.gauss(0, 0.05)) for _ in range(E * D * I)]
        dx = up_bf16(xf, (M, D))
        dw1 = up_bytes(aj.shuffle_weight(aj.moe_w1_interleave(struct.pack(f"<{len(w1f)}H", *[bf16_bits(x) for x in w1f]), E, I, D, 2), E * 2 * I, D, 2), (E * 2 * I, D * 2))
        dw2 = up_bytes(aj.shuffle_weight(struct.pack(f"<{len(w2f)}H", *[bf16_bits(x) for x in w2f]), E * D, I, 2), (E * D, I * 2))
        dxs = dw1s = dw2s = None
    di, dwt = up_i32(ids, (M, K)), up_f32(wts, (M, K))
    do = aj.DeviceBuffer((M, D), "bfloat16")
    bm = aj.fused_moe(dx, dw1, dw2, di, dwt, do, dtype="fp8" if fp8 else "bf16", x_scale=dxs, w1_scale=dw1s, w2_scale=dw2s, num_experts=E,
                      inter=I)
    r.name += f" block_m={bm}"
    out = down_bf16(do, M * D)
    for tt in R.sample(range(M), tokens):
        xv = xf[tt * D:(tt + 1) * D]
        acc, mag = [0.0] * D, [0.0] * D
        for j in range(K):
            e, w = ids[tt * K + j], wts[tt * K + j]
            y = []
            for i in range(I):
                g = sum(xv[d] * w1f[(e * 2 * I + i) * D + d] for d in range(D))
                u = sum(xv[d] * w1f[(e * 2 * I + I + i) * D + d] for d in range(D))
                y.append(rbf(g / (1 + math.exp(-g)) * u))
            for d in range(D):
                s = sum(y[i] * w2f[(e * D + d) * I + i] for i in range(I))
                acc[d] += w * s
                mag[d] += abs(w * s) + w * sum(abs(y[i] * w2f[(e * D + d) * I + i]) for i in range(0, I, 8)) * 8 / I
        for d in range(D):
            tol = (0.04 if fp8 else 1e-2) * abs(acc[d]) + (0.03 if fp8 else 2e-3) * mag[d] + 1e-4
            r.cmp(out[tt * D + d], acc[d], tol)
    return r


# ------------------------------------------------------------------------------------------------------------- remaining families
def check_mla(B=2, H=16, CTX=256, P=16, heads=4):
    r = Res(f"mla_decode B={B} H={H} CTX={CTX} P={P}")
    maxp = -(-CTX // P)
    npages = B * maxp
    lens = [R.randint(CTX // 2, CTX) for _ in range(B)]
    pages = list(range(npages))
    R.shuffle(pages)
    bt = [pages[b * maxp:(b + 1) * maxp] for b in range(B)]
    q, kv = rand_bf16(B * H * 576), rand_bf16(npages * P * 576)
    dq, dkv = up_bf16(q, (B, H, 576)), up_bf16(kv, (npages, P, 576))
    dbt, dcl = up_i32([x for row in bt for x in row], (B, maxp)), up_i32(lens, (B,))
    do = aj.DeviceBuffer((B, H, 512), "bfloat16")
    name, S = aj.mla_decode(dq, dkv, dbt, dcl, do)
    r.name += f" -> {name} S={S}"
    out = down_bf16(do, B * H * 512)
    sc = 1 / math.sqrt(576)
    for b in range(B):
        for h in R.sample(range(H), heads):
            qv = q[(b * H + h) * 576:(b * H + h + 1) * 576]
            rows = [((bt[b][key // P] * P + key % P) * 576) for key in range(lens[b])]
            s = [sum(qv[d] * kv[base + d] for d in range(576)) * sc for base in rows]
            m = max(s)
            pp = [math.exp(x - m) for x in s]
            l_ = sum(pp)
            for d in range(512):
                ref = sum(w * kv[base + d] for w, base in zip(pp, rows)) / l_
                r.cmp(out[(b * H + h) * 512 + d], ref, 1e-2 * abs(ref) + 2e-2)
    return r


def sample_ref(row, inv_t, u, top_k, top_p):
    V = len(row)
    xs = [v * inv_t for v in row]
    mx = max(xs)
    p = [math.exp(x - mx) for x in xs]
    keep = [True] * V
    if top_k > 0:
        kth = sorted(xs, reverse=True)[min(top_k, V) - 1]
        keep = [x >= kth for x in xs]
    if top_p < 1.0:
        sk = sum(p[i] for i in range(V) if keep[i])
        kv = sorted((xs[i] for i in range(V) if keep[i]), reverse=True)
        cum, thr = 0.0, kv[-1]
        for x in kv:
            cum += math.exp(x - mx)
            if cum >= top_p * sk:
                thr = x
                break
        keep = [keep[i] and xs[i] >= thr for i in range(V)]
    sk2 = sum(p[i] for i in range(V) if keep[i])
    target, acc = min(u, 0.99999994) * sk2, 0.0
    for i in range(V):
        if keep[i]:
            acc += p[i]
            if acc >= target:
                return i, p, keep, sk2
    return V - 1, p, keep, sk2


def check_sampling(B=8, V=1024, mode="sample", top_k=0, top_p=1.0, temp=0.8):
    r = Res(f"{mode} B={B} V={V}" + (f" top_k={top_k} top_p={top_p}" if mode == "sample" else ""))
    lg = [rbf(R.gauss(0, 3)) for _ in range(B * V)]
    for b in range(B):
        lg[b * V + (b * 7919) % V] = 30.0
        if b % 3 == 0:
            lg[b * V + (b * 104729) % V] = 30.0
    dl, do = up_bf16(lg, (B, V)), aj.DeviceBuffer((B,), "int32")
    if mode == "argmax":
        aj.argmax(dl, do)
    else:
        us = [R.random() for _ in range(B)]
        aj.sample(dl, up_f32(us, (B,)), do, temperature=temp, top_k=top_k, top_p=top_p)
    got = struct.unpack(f"<{B}i", do.download())
    for b in range(B):
        row = lg[b * V:(b + 1) * V]
        if mode == "argmax":
            r.cmp(got[b], max(range(V), key=lambda i: (row[i], -i)), 0)
            continue
        ref, p, keep, sk2 = sample_ref(row, 1 / temp, us[b], top_k, top_p)
        if got[b] != ref and 0 <= got[b] < V:
            lo, hi = min(got[b], ref), max(got[b], ref)
            between = sum(p[i] for i in range(lo + 1, hi + 1) if keep[i])
            r.cmp(0 if between / sk2 < 2e-4 else 1, 0, 0)  # the kernel's float cumulative sums: neighbours at the boundary
        else:
            r.cmp(got[b], ref, 0)
    return r


def check_topk_per_row(mode="decode", B=4, stride=1024, K=128):
    r = Res(f"topk_per_row_{mode} R={B} stride={stride} K={K}")
    lg = [R.gauss(0, 1) for _ in range(B * stride)]
    dl, di = up_f32(lg, (B, stride)), aj.DeviceBuffer((B, K), "int32")
    if mode == "decode":
        lens = [R.randint(K // 2, stride) for _ in range(B)]
        aj.topk_per_row_decode(dl, up_i32(lens, (B,)), di)
        spans = [(0, L) for L in lens]
    else:
        spans = []
        for _ in range(B):
            s0 = R.randrange(stride // 2)
            spans.append((s0, R.randint(s0 + 1, stride)))
        aj.topk_per_row_prefill(dl, up_i32([s for s, _ in spans], (B,)), up_i32([e for _, e in spans], (B,)), di)
    ix = struct.unpack(f"<{B * K}i", di.download())
    for b in range(B):
        s0, e0 = spans[b]
        L = e0 - s0
        row = ix[b * K:(b + 1) * K]
        n = min(K, L)
        sel = row[:n]
        r.cmp(len(set(sel)), n, 0)
        r.cmp(sum(1 for c in sel if not (s0 <= c < e0)), 0, 0)
        want = sorted((lg[b * stride + c] for c in range(s0, e0)), reverse=True)[:n]
        r.cmp(0 if sorted((lg[b * stride + c] for c in sel if s0 <= c < e0), reverse=True) == want else 1, 0, 0)
        r.cmp(sum(1 for c in row[n:] if c != -1), 0, 0)
    return r


def silu(x):
    return x / (1 + math.exp(-x))


def check_conv1d_fwd(dim=256, W=4, lens=(5, 13, 2), slots=4):
    B, T = len(lens), sum(lens)
    r = Res(f"causal_conv1d_fwd B={B} dim={dim} W={W}")
    x, wt, bias = rand_bf16(T * dim), rand_bf16(dim * W, 0.5), rand_bf16(dim, 0.5)
    st = rand_bf16(slots * dim * (W - 1))
    cu = [0]
    for L in lens:
        cu.append(cu[-1] + L)
    cidx = [2, 0, -1][:B]
    hinit = [1, 0, 1][:B]
    dx, dw, db = up_bf16(x, (T, dim)), up_bf16(wt, (dim, W)), up_bf16(bias, (dim,))
    dst = up_bf16(st, (slots, dim, W - 1))
    do = aj.DeviceBuffer((T, dim), "bfloat16")
    do.fill(0)
    aj.causal_conv1d_fwd(dx, dw, db, up_i32(cu, (B + 1,)), up_i32(cidx, (B,)), up_bytes(hinit, (B,)), dst, do, max_seqlen=max(lens))
    out, st2 = down_bf16(do, T * dim), down_bf16(dst, slots * dim * (W - 1))
    for b in range(B):
        if cidx[b] < 0:
            continue
        for f in range(dim):
            init = [st[(cidx[b] * dim + f) * (W - 1) + i] if hinit[b] else 0.0 for i in range(W - 1)]
            seq = init + [x[(cu[b] + t) * dim + f] for t in range(lens[b])]
            for t in range(lens[b]):
                acc = bias[f] + sum(wt[f * W + i] * seq[t + i] for i in range(W))
                ref = silu(acc)
                r.cmp(out[(cu[b] + t) * dim + f], ref, abs(ref) / 64 + 1e-2)
            for i in range(W - 1):
                r.cmp(st2[(cidx[b] * dim + f) * (W - 1) + i], seq[len(seq) - (W - 1) + i], 0)
    return r


def check_conv1d_update(B=4, dim=256, W=4, SL=3, slots=6):
    r = Res(f"causal_conv1d_update B={B} dim={dim} W={W} SL={SL}")
    x, wt, bias, st = rand_bf16(B * dim), rand_bf16(dim * W, 0.5), rand_bf16(dim, 0.5), rand_bf16(slots * dim * SL)
    idx = [3, -1, 0, 5][:B]
    dx, dw, db, dst = up_bf16(x, (B, dim)), up_bf16(wt, (dim, W)), up_bf16(bias, (dim,)), up_bf16(st, (slots, dim, SL))
    do = aj.DeviceBuffer((B, dim), "bfloat16")
    do.fill(0)
    aj.causal_conv1d_update(dx, dw, db, dst, do, state_indices=up_i32(idx, (B,)))
    out, st2 = down_bf16(do, B * dim), down_bf16(dst, slots * dim * SL)
    for b in range(B):
        s = idx[b]
        if s < 0:
            continue
        for c in range(dim):
            row = st[(s * dim + c) * SL:(s * dim + c + 1) * SL]
            xv = row[SL - (W - 1):] + [x[b * dim + c]]
            ref = silu(bias[c] + sum(wt[c * W + j] * xv[j] for j in range(W)))
            r.cmp(out[b * dim + c], ref, abs(ref) / 64 + 1e-2)
            new = row[1:] + [x[b * dim + c]]
            for l in range(SL):
                r.cmp(st2[(s * dim + c) * SL + l], new[l], 0)
    return r


def check_gemm_w4(kind="a16w4", M=64, N=128, K=256, rows=4):
    r = Res(f"gemm_w4 {kind} M={M} N={N} K={K}")
    bb = bytes(R.randrange(256) for _ in range(N * K // 2))
    bf = [e2m1((bb[i // 2] >> (4 * (i % 2))) & 15) for i in range(N * K)]
    sb8 = bytes(124 + R.randrange(6) for _ in range(N * K // 32))
    if kind == "a16w4":
        af = [rbf((R.randrange(17) - 8) / 8) for _ in range(M * K)]
        da = up_bf16(af, (M, K))
        db = up_bytes(aj.shuffle_weight(aj.a16w4_permute(bb, N, K), N, K // 2, 1), (N, K // 2))
        dsb, dsa, sa8 = up_bytes(sb8, (N, K // 32)), None, None
    else:
        ac = fp8_codes(M * K)
        af = [e4m3(c) for c in ac]
        sa8 = bytes(124 + R.randrange(6) for _ in range(M * K // 32))
        da = up_bytes(ac, (M, K))
        db = up_bytes(aj.shuffle_weight(bb, N, K // 2, 1), (N, K // 2))
        dsa, dsb = up_bytes(aj.shuffle_mx_scales(sa8, M, K), (M, K // 32)), up_bytes(aj.shuffle_mx_scales(sb8, N, K), (N, K // 32))
    do = aj.DeviceBuffer((M, N), "bfloat16")
    aj.gemm_w4(da, db, do, b_scale=dsb, a_scale=dsa, kind=kind, K=K)
    out = down_bf16(do, M * N)
    for ri in range(rows):
        m = ri * (M - 1) // (rows - 1)
        for n in range(N):
            ref = mag = 0.0
            for g in range(K // 32):
                p_ = sum(af[m * K + k] * bf[n * K + k] for k in range(g * 32, g * 32 + 32))
                s = 2.0 ** (sb8[n * (K // 32) + g] - 127) * (2.0 ** (sa8[m * (K // 32) + g] - 127) if sa8 else 1.0)
                ref += p_ * s
                mag += abs(p_ * s)
            r.cmp(out[m * N + n], ref, 1e-2 * abs(ref) + 2 ** -8 * mag + 1e-6)
    return r


def check_attn_bwd(B=1, H=2, HK=1, SQ=256, SK=256, causal=False, nrows=3):
    D = 128
    r = Res(f"attention_bwd B={B} H={H} HK={HK} SQ={SQ} SK={SK} causal={int(causal)}")
    G = H // HK
    q, k, v, dov = rand_bf16(B * SQ * H * D), rand_bf16(B * SK * HK * D), rand_bf16(B * SK * HK * D), rand_bf16(B * SQ * H * D)
    sc = 1 / math.sqrt(D)
    qi = lambda b, s, h: ((b * SQ + s) * H + h) * D
    ki = lambda b, s, h: ((b * SK + s) * HK + h) * D
    # CPU forward: P rows, O (bf16) and LSE
    P_, O, L = {}, [0.0] * (B * SQ * H * D), [0.0] * (B * H * SQ)
    for b in range(B):
        for h in range(H):
            for s in range(SQ):
                kmax = min(SK - 1, s + SK - SQ) if causal else SK - 1
                sc_ = [sum(q[qi(b, s, h) + d] * k[ki(b, kk, h // G) + d] for d in range(D)) * sc for kk in range(kmax + 1)]
                m = max(sc_)
                e = [math.exp(x - m) for x in sc_]
                l_ = sum(e)
                L[(b * H + h) * SQ + s] = m + math.log(l_)
                p = [x / l_ for x in e]
                P_[(b, h, s)] = p
                for d in range(D):
                    O[qi(b, s, h) + d] = rbf(sum(p[kk] * v[ki(b, kk, h // G) + d] for kk in range(len(p))))
    dq_, dk_, dv_ = up_bf16(q, (B, SQ, H, D)), up_bf16(k, (B, SK, HK, D)), up_bf16(v, (B, SK, HK, D))
    do_, dO_ = up_bf16(O, (B, SQ, H, D)), up_bf16(dov, (B, SQ, H, D))
    dl = up_f32(L, (B, H, SQ))
    gq, gk, gv = aj.DeviceBuffer((B, SQ, H, D), "bfloat16"), aj.DeviceBuffer((B, SK, H, D), "bfloat16"), aj.DeviceBuffer((B, SK, H, D), "bfloat16")
    name = aj.attention_bwd_bf16(dq_, dk_, dv_, do_, dO_, dl, gq, gk, gv, causal=causal)
    r.name += f" -> {name}"
    GQ, GK, GV = down_bf16(gq, B * SQ * H * D), down_bf16(gk, B * SK * H * D), down_bf16(gv, B * SK * H * D)
    dot = {}

    def dS(b, h, s):
        if (b, h, s) not in dot:
            dot[(b, h, s)] = sum(dov[qi(b, s, h) + d] * O[qi(b, s, h) + d] for d in range(D))
        p = P_[(b, h, s)]
        dp = [sum(dov[qi(b, s, h) + d] * v[ki(b, kk, h // G) + d] for d in range(D)) for kk in range(len(p))]
        return [p[kk] * (dp[kk] - dot[(b, h, s)]) for kk in range(len(p))]
    for _ in range(nrows):
        b, h, s = R.randrange(B), R.randrange(H), R.randrange(SQ)
        ds = dS(b, h, s)
        for d in range(D):
            ref = sc * sum(ds[kk] * k[ki(b, kk, h // G) + d] for kk in range(len(ds)))
            r.cmp(GQ[qi(b, s, h) + d], ref, 2e-2 * abs(ref) + 2e-2)
    for _ in range(nrows):
        b, h, kk = R.randrange(B), R.randrange(H), R.randrange(SK)
        rows = [s for s in range(SQ) if len(P_[(b, h, s)]) > kk]
        dss = {s: dS(b, h, s)[kk] for s in rows}
        for d in range(D):
            refk = sc * sum(dss[s] * q[qi(b, s, h) + d] for s in rows)
            refv = sum(P_[(b, h, s)][kk] * dov[qi(b, s, h) + d] for s in rows)
            o_ = ((b * SK + kk) * H + h) * D + d
            r.cmp(GK[o_], refk, 2e-2 * abs(refk) + 2e-2)
            r.cmp(GV[o_], refv, 2e-2 * abs(refv) + 2e-2)
    return r

CHECKS = [
    ("rmsnorm", check_rmsnorm),
    ("add_rmsnorm", check_add_rmsnorm),
    ("add_rmsnorm_quant_fp8", lambda: check_add_rmsnorm(quant=True)),
    ("rmsnorm_quant_fp8", check_rmsnorm_quant_fp8),
    ("rmsnorm_quant_mxfp4", check_rmsnorm_quant_mxfp4),
    ("quant_per_token_fp8", check_quant_token),
    ("quant_per_token_int8", lambda: check_quant_token(dtype="int8")),
    ("quant_per_group128", check_quant_group),
    ("act_mul_silu", check_act_mul),
    ("act_mul_gelu", lambda: check_act_mul(kind="gelu")),
    ("act_mul_gelu_tanh", lambda: check_act_mul(kind="gelu_tanh")),
    ("quant_static_fp8", check_quant_static),
    ("act_mul_swiglu_oai", lambda: check_act_mul(kind="swiglu_oai")),
    ("silu_mul_quant_fp8", check_silu_mul_quant),
    ("rope_neox", check_rope),
    ("rope_gptj", lambda: check_rope(style="gptj")),
    ("qknorm_rope", lambda: check_rope(norm=True)),
    ("qknorm_rope_kvcache_bf16", check_qknorm_rope_kvcache),
    ("qknorm_rope_kvcache_fp8_i64", lambda: check_qknorm_rope_kvcache(fp8=True, i64=True)),
    ("reshape_cache_flash", check_reshape_cache),
    ("reshape_cache_blocked", lambda: check_reshape_cache(layout="blocked")),
    ("topk_softmax", check_topk),
    ("topk_softmax_renorm", lambda: check_topk(E=256, renorm=True)),
    ("moe_sorting", check_moe_sorting),
    ("moe_reduce", check_moe_reduce),
    ("moe_combine", lambda: check_moe_reduce(weighted=False)),
    ("gemm_bf16", check_gemm),
    ("gemm_bf16_256", lambda: check_gemm(tile=256)),
    ("gemm_fp8_rowcol", lambda: check_gemm("fp8_rowcol")),
    ("gemm_fp8_blockscale", lambda: check_gemm("fp8_blockscale", M=4096, N=4096, K=256, rows=2)),
    ("gemm_fp8_blockscale_splitk", lambda: check_gemm("fp8_blockscale", M=256, N=1024, K=512)),
    ("gemm_mxfp4", lambda: check_gemm("mxfp4", K=512)),
    ("prefill_bf16", check_prefill),
    ("prefill_bf16_causal", lambda: check_prefill(causal=True)),
    ("fused_moe_bf16", check_moe),
    ("fused_moe_fp8", lambda: check_moe(fp8=True)),
    ("mla_decode", check_mla),
    ("select_f32", check_select_f32),
    ("argmax", lambda: check_sampling(mode="argmax")),
    ("sample", check_sampling),
    ("sample_topkp", lambda: check_sampling(top_k=50, top_p=0.9)),
    ("topk_per_row_decode", check_topk_per_row),
    ("topk_per_row_prefill", lambda: check_topk_per_row(mode="prefill")),
    ("causal_conv1d_fwd", check_conv1d_fwd),
    ("causal_conv1d_update", check_conv1d_update),
    ("gemm_a16w4", check_gemm_w4),
    ("gemm_a8w4", lambda: check_gemm_w4("a8w4", M=128, N=128, K=512)),
    ("attention_bwd", check_attn_bwd),
    ("attention_bwd_causal", lambda: check_attn_bwd(causal=True)),
]


def main(filters):
    ok = True
    for name, fn in CHECKS:
        if filters and not any(f in name for f in filters):
            continue
        try:
            r = fn()
            aj.hip.synchronize()
            good = r.bad == 0
            print(f"{'ok  ' if good else 'FAIL'} {r.name}: {r.cnt} values, bad={r.bad}")
        except Exception as e:  # noqa: BLE001 - report every op
            good = False
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}")
            if os.environ.get("TRACE"):
                traceback.print_exc()
        ok &= good
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
