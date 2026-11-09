#!/usr/bin/env python3
"""GPU-node check of the amdgpu_jit ops (torch-free, stdlib only): each op on random data against a Python reference that follows the
C++ host reference of its kernel family (tests/gpu/*_host.cpp: formulas and tolerances). Small shapes (the references are pure Python).

    HIP_VISIBLE_DEVICES=6 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/ops_check.py [name-substring ...]
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
            s = 2.0 ** (sc[m * (N // 32) + k // 32] - 127)
            # e2m1 steps are at most 2 units of the block scale (between 4 and 6); saturation at 6 s
            r.cmp(e2m1(c) * s, max(-6 * s, min(6 * s, y[k])), 1.0 * s + 1e-6)
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


def act(a, kind):
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
            ref = act(x[m * 2 * D + k], kind) * x[m * 2 * D + D + k]
            r.cmp(out[m * D + k], ref, abs(ref) / 128 * 1.05 + 1e-6)
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
    ("silu_mul_quant_fp8", check_silu_mul_quant),
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
