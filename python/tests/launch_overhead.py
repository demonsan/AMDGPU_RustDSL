#!/usr/bin/env python3
"""Host time per launch of the Python ops (GPU node; stdlib only): paged_decode_bf16 (fused and last-arriver), rmsnorm (generic launch),
gemm bf16; first call excluded, N calls back to back without synchronisation (the launch is asynchronous), with and without the contract check.

    HIP_VISIBLE_DEVICES=6 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/launch_overhead.py
"""
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amdgpu_jit as aj  # noqa: E402


def per_call(fn, n=2000):
    fn()
    aj.hip.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    aj.hip.synchronize()
    return (t1 - t0) / n * 1e6


def decode_case(B, H, HK, CTX, P=16):
    maxp = -(-CTX // P)
    np_ = B * maxp
    q = aj.DeviceBuffer((B, H, 128), "bfloat16")
    kc = aj.DeviceBuffer((np_, P, HK, 128), "bfloat16")
    vc = aj.DeviceBuffer((np_, P, HK, 128), "bfloat16")
    bt = aj.DeviceBuffer((B, maxp), "int32", data=struct.pack(f"<{B * maxp}i", *range(B * maxp)))
    cl = aj.DeviceBuffer((B,), "int32", data=struct.pack(f"<{B}i", *([CTX] * B)))
    out = aj.DeviceBuffer((B, H, 128), "bfloat16")
    for b in (q, kc, vc, out):
        b.fill(0)
    return q, kc, vc, bt, cl, out


def main():
    rows = []
    for B, label in ((64, "fused"), (4, "last-arriver")):
        q, kc, vc, bt, cl, out = decode_case(B, 64, 8, 2048)
        name, S = aj.paged_decode_bf16(q, kc, vc, bt, cl, out, max_context=2048)
        for chk in (True, False):
            us = per_call(lambda: aj.paged_decode_bf16(q, kc, vc, bt, cl, out, max_context=2048, check=chk))
            rows.append((f"paged_decode_bf16 B{B} ({label}: {name} S={S})", chk, us))
    x = aj.DeviceBuffer((64, 4096), "bfloat16")
    g = aj.DeviceBuffer((4096,), "bfloat16")
    o = aj.DeviceBuffer((64, 4096), "bfloat16")
    x.fill(0); g.fill(0)
    for chk in (True, False):
        rows.append(("rmsnorm 64x4096 (generic launch)", chk, per_call(lambda: aj.rmsnorm(x, g, o, check=chk))))
    a = aj.DeviceBuffer((256, 256), "bfloat16")
    w = aj.DeviceBuffer((256, 512), "uint8", itemsize=1, nbytes=256 * 512)
    c = aj.DeviceBuffer((256, 256), "bfloat16")
    a.fill(0); w.fill(0)
    for chk in (True, False):
        rows.append(("gemm bf16 256^3 (generic launch)", chk, per_call(lambda: aj.gemm(a, w, c, kind="bf16", K=256, check=chk), 500)))
    print(f"| op | check | host us / call |\n|---|---|--:|")
    for n, chk, us in rows:
        print(f"| {n} | {'on' if chk else 'off'} | {us:.1f} |")


if __name__ == "__main__":
    main()
