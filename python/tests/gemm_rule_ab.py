#!/usr/bin/env python3
"""Same-run A/B of the Python ``gemm()`` launch rule: the former Python rule (256 x 256 when M, N % 256 == 0 and the grid has >= CUs / 4 CTAs,
else ``_128``; blockscale: 128-row tiles with the dispatch split) against the one rule ``dispatch::gemm_kernels`` (``amdc dispatch gemm``),
over the GEMM family shape grid. Interleaved rounds, hip events, median of the rounds (docs/python-jit.md).

    HIP_VISIBLE_DEVICES=0 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/gemm_rule_ab.py [--rounds 5] [--iters 50]
"""
import argparse
import os
import statistics
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amdgpu_jit as aj  # noqa: E402
from amdgpu_jit import hip  # noqa: E402

SHAPES = [(4096, 4096, 4096), (2048, 4096, 4096), (1024, 4096, 8192), (512, 7168, 4096), (256, 7168, 4096), (128, 4096, 8192), (64, 8192, 8192)]
KINDS = ["bf16", "fp8_rowcol", "fp8_blockscale"]
BASE = {"bf16": "gemm_a16w16_bf16", "fp8_rowcol": "gemm_a8w8_rowcol_fp8", "fp8_blockscale": "gemm_a8w8_blockscale"}


def legacy(kind, M, N, K, cus=256):
    """the former Python rule: (kernel, splits, reduce)"""
    base = BASE[kind]
    if kind in ("bf16", "fp8_rowcol"):
        big = M % 256 == 0 and N % 256 == 0 and (M // 256) * (N // 256) * 4 >= cus
        return (base if big else base + "_128"), 1, None
    S = int(aj.dispatch("gemm_blockscale", M, N, K, "--cus", cus)["splits"])
    if S > 1 and N % 1024:
        S = 1
    return (base + "_splitk" if S > 1 else base), S, (f"gemm_splitk_reduce_s{S}" if S > 1 else None)


def time_calls(fn, iters, warm=10):
    for _ in range(warm):
        fn()
    hip.synchronize()
    e0, e1 = hip.Event(), hip.Event()
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_ms(e1) / iters * 1e3


def bufs(kind, M, N, K):
    a = aj.DeviceBuffer((M, K), "uint8", itemsize=1, data=os.urandom(M * K))
    b = aj.DeviceBuffer((N, K), "uint8", itemsize=1, data=os.urandom(N * K))
    if kind == "bf16":
        # finite bf16 values in [1, 2): exponent 127, random mantissa
        pa = struct.pack("<1024H", *[0x3F80 | (i * 37 & 0x7F) for i in range(1024)])
        pb = struct.pack("<1024H", *[0x3F80 | (i * 53 & 0x7F) for i in range(1024)])
        a = aj.DeviceBuffer((M, K), "bfloat16", data=pa * (M * K // 1024))
        b = aj.DeviceBuffer((N, K), "bfloat16", data=pb * (N * K // 1024))
    sa = sb = None
    if kind == "fp8_rowcol":
        sa, sb = (aj.DeviceBuffer((n,), "float32", data=struct.pack("<f", 1e-3) * n) for n in (M, N))
    elif kind == "fp8_blockscale":
        sa = aj.DeviceBuffer((M * (K // 128),), "float32", data=struct.pack("<f", 1.0) * (M * (K // 128)))
        sb = aj.DeviceBuffer((N // 128 * (K // 128),), "float32", data=struct.pack("<f", 1.0) * (N // 128 * (K // 128)))
    out = aj.DeviceBuffer((M, N), "bfloat16")
    return a, b, sa, sb, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--cus", type=int, default=256)
    args = ap.parse_args()
    print("| kind | M x N x K | former rule | us | dispatch::gemm_kernels | us | new / old |")
    print("|---|---|---|---:|---|---:|---:|")
    for kind in KINDS:
        for M, N, K in SHAPES:
            a, b, sa, sb, out = bufs(kind, M, N, K)
            old = legacy(kind, M, N, K, args.cus)
            d = aj.dispatch("gemm", kind, M, N, K, "--cus", args.cus)
            new = (d["kernel"], int(d["splits"]), d["reduce"])
            runs = {old: [], new: []}

            def call(c):
                return lambda: aj.gemm(a, b, out, kind=kind, a_scale=sa, b_scale=sb, M=M, N=N, K=K, kernel=c[0], splits=c[1], reduce=c[2],
                                       check=False)

            bad = {}
            for _ in range(args.rounds):
                for c in ([old, new] if old != new else [old]):
                    if c in bad:
                        continue
                    try:
                        runs[c].append(time_calls(call(c), args.iters))
                    except Exception as e:  # noqa: BLE001 - the former rule cannot launch some shapes
                        bad[c] = type(e).__name__
            fmt = lambda c: c[0] + (f" S{c[1]}" if c[1] > 1 else "")
            med = lambda c: statistics.median(runs[c]) if runs[c] else None
            to, tn = med(old), (med(new) if old != new else med(old))
            ts = lambda x, c: f"{x:.1f}" if x is not None else "fails (" + bad.get(c, "-") + ")"
            ratio = f"{tn / to:.2f}" if to and tn else "-"
            print(f"| {kind} | {M} x {N} x {K} | {fmt(old)} | {ts(to, old)} | {fmt(new)} | {ts(tn, new)} | {ratio} |", flush=True)


if __name__ == "__main__":
    main()
