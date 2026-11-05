#!/usr/bin/env python3
"""GPU-node checks of amdgpu_jit (stdlib only; docs/python-jit.md):

  1. bit identity: the JIT ops vs the C++ harnesses (tests/gpu/gemm_ref_host.cpp, tests/gpu/ref_host.cpp) on the SAME inputs (the harness dumps
     its device inputs with INPUT_DUMP / REF_INPUT_DUMP and its output with OURS_DUMP / REF_DUMP2), specialized and generic kernels;
  2. cache behaviour: cold compile, disk hit in a fresh process, in-process hit (latency of the first call each);
  3. steady-state launch overhead (host time per call, with and without the contract check);
  4. specialized vs generic performance (hip events, interleaved rounds).

Usage (repo root, GPU node):  HIP_VISIBLE_DEVICES=5 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/node_check.py [--quick] [--skip-perf]
Env: REF_CO (vendor FMHA .co, default the session copy), ROCM_PATH (HIP headers/libs for building the harnesses).
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
import amdgpu_jit as aj  # noqa: E402
from amdgpu_jit import hip  # noqa: E402

ROC = os.environ.get("ROCM_PATH", "/shared/apps/ubuntu/opt/rocm-7.0.2")
VENDOR_FMHA = os.environ.get("REF_CO", "/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co")
BUILD = os.path.join(REPO, "build", "pyjit")
RESULTS = []


def log(*a):
    print(*a, flush=True)


def result(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def harness(src, out):
    exe = os.path.join(BUILD, out)
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(src):
        subprocess.run(["g++", "-O2", "-std=c++17", "-D__HIP_PLATFORM_AMD__", src, "-o", exe, f"-I{ROC}/include", "-Itests/gpu", "-I.",
                        f"-L{ROC}/lib", "-lamdhip64", f"-Wl,-rpath,{ROC}/lib"], check=True, cwd=REPO)
    return exe


def read(p):
    with open(p, "rb") as f:
        return f.read()


def dev(data, shape, itemsize=1, nbytes=None):
    b = aj.DeviceBuffer(shape, "uint8" if itemsize == 1 else "bfloat16", itemsize=itemsize, nbytes=nbytes or len(data))
    b.upload(data)
    return b


# ------------------------------------------------------------------------------------------------------------------------------- GEMM
def gemm_identity(M, N, K):
    exe = harness("tests/gpu/gemm_ref_host.cpp", "gemm_ref_host")
    generic = aj.get_kernel("gemm_mxfp8_bs_v4", None)
    pre = os.path.join(BUILD, f"g{M}x{N}x{K}")
    env = dict(os.environ, GEMM_M=str(M), GEMM_N=str(N), GEMM_K=str(K), OURS_CO=os.path.join(generic.path, "kernel.co"),
               OURS_KNAME=generic.info.symbol, OURS_TILE_M="256", OURS_TILE_N="256", REF_ROWS="4", REF_ITERS="0", INPUT_DUMP=pre, OURS_DUMP=pre + ".out")
    env.pop("REF_CO", None)
    r = subprocess.run([exe], env=env, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0 or "GEMM_OK" not in r.stdout:
        result(f"gemm {M}x{N}x{K} harness", False, (r.stdout + r.stderr)[-800:])
        return
    a, b, sa, sb, want = (read(pre + e) for e in (".a", ".b", ".sa", ".sb", ".out"))
    da, db = dev(a, (M, K)), dev(b, (N, K))
    dsa, dsb = dev(sa, (K // 128, M)), dev(sb, (N // 128, K // 128))
    for spec in (True, False):
        out = aj.DeviceBuffer((M, N), "bfloat16", itemsize=2)
        out.fill(0xFF)
        aj.gemm_mxfp8_blockscale(da, db, dsa, dsb, out, specialize=spec)
        hip.synchronize()
        got = out.download()
        result(f"gemm {M}x{N}x{K} {'specialized' if spec else 'generic'} == gemm_ref_host", got == want,
               f"{sum(x != y for x, y in zip(got[::2], want[::2]))} differing bf16 words" if got != want else "bit-identical")


# ------------------------------------------------------------------------------------------------------------------------------- FMHA
def fmha_identity(B, H, S, SK, gqa=1):
    exe = harness("tests/gpu/ref_host.cpp", "ref_host")
    HK = H // gqa
    ks = {n: aj.get_kernel(n, None) for n in (aj.ops.FMHA_VS128, aj.ops.FMHA_T128, aj.ops.FMHA_KVSPLIT, aj.ops.FMHA_COMBINE)}
    co = {n: os.path.join(k.path, "kernel.co") for n, k in ks.items()}
    pre = os.path.join(BUILD, f"f{B}_{H}_{S}_{SK}_{gqa}")
    env = dict(os.environ, REF_B=str(B), REF_H=str(H), REF_S=str(S), REF_SK=str(SK), REF_HK=str(HK), REF_ROWS="4", REF_ITERS="0",
               REF_CO2=co[aj.ops.FMHA_VS128], REF_KNAME2=aj.ops.FMHA_VS128, REF_CO2_SMALL=co[aj.ops.FMHA_T128], REF_KNAME2_SMALL=aj.ops.FMHA_T128,
               REF_CO2_SPLIT=co[aj.ops.FMHA_KVSPLIT], REF_KNAME2_SPLIT=aj.ops.FMHA_KVSPLIT, REF_CO2_COMBINE=co[aj.ops.FMHA_COMBINE],
               REF_KNAME2_COMBINE=aj.ops.FMHA_COMBINE, REF_COMBINE_ELEMS="8", REF_SPLIT="0", REF_INPUT_DUMP=pre, REF_DUMP2=pre + ".out")
    r = subprocess.run([exe, VENDOR_FMHA], env=env, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0 or "REF_KERNEL_OK" not in r.stdout:
        result(f"fmha B{B}H{H}S{S}SK{SK} harness", False, (r.stdout + r.stderr)[-800:])
        return
    hpath = next((ln for ln in r.stdout.splitlines() if ln.startswith("[grid]")), "")
    q, k, v, qs, ksc, want = (read(pre + e) for e in (".q", ".k", ".v", ".qs", ".ks", ".out"))
    dq, dk, dv = dev(q, (B, S, H, 128)), dev(k, (B, SK, HK, 128)), dev(v, (B, SK, HK, 128))
    dqs, dks = dev(qs, (B, S, H, 4)), dev(ksc, (B, SK, HK, 4))
    dvs = aj.DeviceBuffer((1,), "float32", itemsize=4, nbytes=4096)
    dvs.fill(0)
    dvs.upload(b"\x00\x00\x80\x3f")
    for spec in (True, False):
        out = aj.DeviceBuffer((B, S, H, 128), "bfloat16", itemsize=2)
        out.fill(0)
        path, splits = aj.fmha_fwd_mxfp8(dq, dk, dv, dqs, dks, dvs, out, specialize=spec)
        hip.synchronize()
        got = out.download()
        same_path = (path == "kvsplit" and "KV-split" in hpath) or (path == "t128" and "v3_t128" in hpath) or (path == "vs128" and "v3_vs128" in hpath)
        result(f"fmha B{B}H{H}S{S}SK{SK} gqa{gqa} {'specialized' if spec else 'generic'} ({path}{splits or ''}) == ref_host", got == want and same_path,
               ("bit-identical" if got == want else f"{sum(x != y for x, y in zip(got[::2], want[::2]))} differing bf16 words") + ("" if same_path else f"; path differs: {hpath}"))


# ------------------------------------------------------------------------------------------------------------------------------- cache
CHILD = r"""
import sys, time, json, os
sys.path.insert(0, os.path.join(sys.argv[1], "python"))
import amdgpu_jit as aj
t0 = time.perf_counter(); k = aj.get_kernel("gemm_mxfp8_bs_v4", {"m": 1024, "n": 1024, "k": 2048}); t1 = time.perf_counter()
k2 = aj.get_kernel("gemm_mxfp8_bs_v4", {"m": 1024, "n": 1024, "k": 2048}); t2 = time.perf_counter()
print(json.dumps({"first": t1 - t0, "second": t2 - t1, "stats": aj.stats}))
"""


def cache_behaviour():
    d = tempfile.mkdtemp(prefix="ajcache.")
    env = dict(os.environ, AMDGPU_JIT_CACHE_DIR=d)
    runs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", CHILD, REPO], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            result("cache", False, r.stderr[-800:])
            return
        runs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    cold, warm = runs
    ok = cold["stats"]["compiles"] == 1 and warm["stats"]["compiles"] == 0 and warm["stats"]["disk_hits"] == 1 and cold["stats"]["memory_hits"] == 1
    entries = [e for e in os.listdir(d) if not e.startswith(".")]
    result("cache miss -> disk hit -> memory hit", ok and len(entries) == 1,
           f"cold first call {cold['first']:.2f} s (compile), new process first call {warm['first'] * 1e3:.1f} ms (disk), in-process {cold['second'] * 1e6:.0f} us; "
           f"entries {len(entries)}")
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------------------------------------- overhead / perf
def gemm_buffers(M, N, K):
    import random

    rnd = random.Random(1)
    a = dev(bytes(rnd.choice((0x38, 0x30, 0xB8, 0x40)) for _ in range(M * K)), (M, K))
    b = dev(bytes(rnd.choice((0x38, 0x30, 0xB8, 0x40)) for _ in range(N * K)), (N, K))
    sa, sb = dev(bytes([127]) * (K // 128 * M), (K // 128, M)), dev(bytes([127]) * (N // 128 * K // 128), (N // 128, K // 128))
    out = aj.DeviceBuffer((M, N), "bfloat16", itemsize=2)
    return a, b, sa, sb, out


def launch_overhead():
    bufs = gemm_buffers(256, 256, 128)
    for check in (True, False):
        aj.gemm_mxfp8_blockscale(*bufs, check=check)
        hip.synchronize()
        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            aj.gemm_mxfp8_blockscale(*bufs, check=check)
        t1 = time.perf_counter()
        hip.synchronize()
        t2 = time.perf_counter()
        result(f"launch overhead (check={check})", True, f"{(t1 - t0) / n * 1e6:.1f} us host time per call (issue only), {(t2 - t0) / n * 1e6:.1f} us per call incl. drain")


def time_calls(fn, iters=50, warm=20):
    for _ in range(warm):
        fn()
    hip.synchronize()
    e0, e1 = hip.Event(), hip.Event()
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    e1.synchronize()
    return e0.elapsed_ms(e1) / iters


def perf_spec_vs_generic(quick):
    rounds = 3 if quick else 7
    M, N, K = 4096, 4096, 8192
    g = gemm_buffers(M, N, K)
    res = {True: [], False: []}
    for _ in range(rounds):
        for spec in (True, False):
            ms = time_calls(lambda: aj.gemm_mxfp8_blockscale(*g, specialize=spec))
            res[spec].append(2.0 * M * N * K / (ms * 1e-3) / 1e12)
    s, gn = statistics.median(res[True]), statistics.median(res[False])
    result(f"gemm {M}x{N}x{K} specialized vs generic", True, f"{s:.1f} vs {gn:.1f} TFLOP/s ({(s / gn - 1) * 100:+.1f}%), median of {rounds}")
    for (B, H, S) in ((4, 8, 2048), (1, 8, 8192), (1, 8, 2048)):
        HK, SK = H, S
        q = dev(bytes([0x38]) * (B * S * H * 128), (B, S, H, 128))
        k = dev(bytes([0x30]) * (B * SK * HK * 128), (B, SK, HK, 128))
        v = dev(bytes([0x38]) * (B * SK * HK * 128), (B, SK, HK, 128))
        qs = aj.DeviceBuffer((B, S, H, 4), nbytes=B * S * H * 4 + 512 * H * 4)
        qs.fill(127)
        ks = aj.DeviceBuffer((B, SK, HK, 4), nbytes=B * SK * HK * 4 + 640 * H * 4)
        ks.fill(127)
        vs = aj.DeviceBuffer((1,), "float32", itemsize=4, nbytes=4096)
        vs.upload(b"\x00\x00\x80\x3f")
        out = aj.DeviceBuffer((B, S, H, 128), "bfloat16", itemsize=2)
        res = {True: [], False: []}
        path = aj.select_path(B, H, S, SK)
        for _ in range(rounds):
            for spec in (True, False):
                ms = time_calls(lambda: aj.fmha_fwd_mxfp8(q, k, v, qs, ks, vs, out, specialize=spec))
                res[spec].append(4.0 * B * H * S * SK * 128 / (ms * 1e-3) / 1e12)
        s, gn = statistics.median(res[True]), statistics.median(res[False])
        result(f"fmha B{B}H{H}S{S} ({path[0]}{path[1] or ''}) specialized vs generic", True, f"{s:.1f} vs {gn:.1f} TFLOP/s ({(s / gn - 1) * 100:+.1f}%), median of {rounds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-perf", action="store_true")
    args = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    gemm_identity(1024, 1024, 2048)
    if not args.quick:
        gemm_identity(1280, 6144, 7168)
    for shp in ((4, 8, 2048, 2048, 1), (2, 8, 2048, 2048, 1), (1, 8, 2048, 2048, 1), (1, 2, 200, 1337, 2), (1, 4, 77, 1, 1)):
        fmha_identity(shp[0], shp[1], shp[2], shp[3], shp[4])
    cache_behaviour()
    launch_overhead()
    if not args.skip_perf:
        perf_spec_vs_generic(args.quick)
    fails = [r for r in RESULTS if not r[1]]
    log(f"\nnode_check: {len(RESULTS) - len(fails)}/{len(RESULTS)} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
