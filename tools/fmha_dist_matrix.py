#!/usr/bin/env python3
"""Ours vs vendor FMHA over input distributions (REF_DATA=lcg|uniform|normal of tests/gpu/ref_host.cpp): accuracy (bad / mean_err vs CPU, both
kernels) and same-run interleaved timing with the GPU clock and power sampled (amd-smi) while each kernel runs alone (REF_TIME_ONLY).

Usage (repo root, GPU node; kernels in build/dm/*.co, see --build):
  HIP_VISIBLE_DEVICES=5 python3 tools/fmha_dist_matrix.py --build [--reps 5] [--modes lcg,uniform,normal] [--shapes "4 8 2048 2048 auto" ...]
Shapes: "B H S SK path" with path auto | vs (forced _vs128, 256 rows/block).
"""
import argparse
import json
import os
import statistics
import subprocess
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROC = os.environ.get("ROCM_PATH", "/shared/apps/ubuntu/opt/rocm-7.0.2")
CL = os.path.join(os.environ.get("ASM_ROC", "/shared/apps/ubuntu/opt/rocm-10.1.0"), "llvm", "bin", "clang")
AMDC = os.environ.get("AMDC", "/tmp/target-kfmha/debug/amdc")
VENDOR = os.environ.get("REF_CO", "/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co")
B_DIR = os.path.join(REPO, "build", "dm")
P = "fmha_fwd_hd128_mxfp8_v3"
KN = {"vs": P + "_vs128", "t128": P + "_t128", "split": P + "_kvsplit", "comb": P + "_kvsplit_combine8h"}
SHAPES = ["4 8 2048 2048 auto", "1 8 8192 8192 auto", "8 8 1024 1024 auto", "8 8 8192 8192 auto", "1 8 2048 2048 vs", "1 8 2048 2048 auto", "2 8 2048 2048 auto"]
GPU = int(os.environ.get("HIP_VISIBLE_DEVICES", "5").split(",")[0])
SMI_GPU = int(os.environ.get("SMI_GPU", str(GPU - 1 if GPU > 0 else 0)))  # amd-smi index of the HIP device (node 53: HIP 5 = amd-smi 4)


def build():
    os.makedirs(B_DIR, exist_ok=True)
    for k in KN.values():
        s = os.path.join(B_DIR, k + ".s")
        subprocess.run([AMDC, "emit", k, "--backend", "own", "--out", s], check=True, capture_output=True)
        subprocess.run([CL, "-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx950", "-mcode-object-version=6", "-o", s[:-2] + ".co", s], check=True)
    subprocess.run(["g++", "-O2", "-std=c++17", "-D__HIP_PLATFORM_AMD__", "tests/gpu/ref_host.cpp", "-o", os.path.join(B_DIR, "ref_host"), f"-I{ROC}/include",
                    "-Itests/gpu", "-I.", f"-L{ROC}/lib", "-lamdhip64", f"-Wl,-rpath,{ROC}/lib"], check=True, cwd=REPO, stderr=subprocess.DEVNULL)


def env_for(b, h, s, sk, path, mode, extra):
    e = dict(os.environ, REF_B=str(b), REF_H=str(h), REF_S=str(s), REF_SK=str(sk), REF_HK=str(h), REF_DATA=mode, REF_COMBINE_ELEMS="8")
    co = lambda k: os.path.join(B_DIR, KN[k] + ".co")
    if path == "vs":
        e.update(REF_CO2=co("vs"), REF_KNAME2=KN["vs"], REF_ROWS_PER_BLOCK="256")
    else:
        e.update(REF_CO2=co("vs"), REF_KNAME2=KN["vs"], REF_CO2_SMALL=co("t128"), REF_KNAME2_SMALL=KN["t128"], REF_CO2_SPLIT=co("split"),
                 REF_KNAME2_SPLIT=KN["split"], REF_CO2_COMBINE=co("comb"), REF_KNAME2_COMBINE=KN["comb"], REF_SPLIT="0")
    e.update(extra)
    return e


def smi_sample():
    out = subprocess.run([os.path.join(ROC, "bin", "amd-smi"), "metric", "-g", str(SMI_GPU), "-c", "-p", "--json"], capture_output=True, text=True).stdout
    d = json.loads(out)
    d = d["gpu_data"][0] if isinstance(d, dict) and "gpu_data" in d else d[0]
    c = d["clock"]
    cl = [c[k]["clk"]["value"] for k in c if k.startswith("gfx_") and isinstance(c[k].get("clk"), dict) and isinstance(c[k]["clk"].get("value"), (int, float))]
    return float(d["power"]["socket_power"]["value"]), (sum(cl) / len(cl) if cl else 0.0)


def run(args_env):
    """Runs ref_host while sampling amd-smi; returns (stdout, [(t, W, MHz)], t_end)."""
    samples, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                w, mhz = smi_sample()
                samples.append((time.time(), w, mhz))
            except Exception:
                time.sleep(0.05)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    r = subprocess.run([os.path.join(B_DIR, "ref_host"), VENDOR], env=args_env, capture_output=True, text=True, cwd=REPO)
    t_end = time.time()
    stop.set()
    th.join()
    return r.stdout + r.stderr, samples, t_end


def tf_of(out, who):
    lines = out.splitlines()
    if who == "ours":
        # a KV-split launch prints the main kernel alone first and then "timing ours total:" (main + combine): take the total
        tot = [ln for ln in lines if ln.startswith("timing ours total:")]
        if tot:
            return float(tot[-1].split("TFLOP/s")[0].split(",")[-1])
    for ln in lines:
        if who == "vendor" and ln.startswith("timing vendor"):
            return float(ln.split("TFLOP/s")[0].split(",")[-1])
        if who == "ours" and ln.startswith("timing ours "):
            return float(ln.split("TFLOP/s")[0].split(",")[-1])
    return 0.0


def acc_of(out, label):
    for ln in out.splitlines():
        if " ".join(ln.split()).startswith(" ".join(label.split())):
            f = dict(x.split("=") for x in ln.split(":", 1)[1].split() if "=" in x)
            return int(f["bad"]), float(f["mean_err"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--modes", default="lcg,uniform,normal")
    ap.add_argument("--shapes", nargs="*", default=SHAPES)
    ap.add_argument("--window", type=float, default=2.0, help="seconds of timing per kernel and rep")
    a = ap.parse_args()
    if a.build:
        build()
    rows = []
    for shp in a.shapes:
        b, h, s, sk, path = shp.split()
        b, h, s, sk = int(b), int(h), int(s), int(sk)
        flops = 4.0 * b * h * s * sk * 128
        iters = max(50, int(a.window / (flops / 1.5e15)))
        for mode in a.modes.split(","):
            out, _, _ = run(env_for(b, h, s, sk, path, mode, {"REF_ROWS": "16", "REF_ITERS": "0"}))
            va, oa = acc_of(out, "vendor kernel vs CPU"), acc_of(out, "our kernel vs CPU")
            dpath = next((ln.split("ours (")[1].split(")")[0] for ln in out.splitlines() if ln.startswith("[grid]") and "ours (" in ln), "?")
            res = {"vendor": [], "ours": []}
            for _ in range(a.reps):
                for who in ("vendor", "ours"):
                    o, smp, t_end = run(env_for(b, h, s, sk, path, mode, {"REF_ROWS": "1", "REF_WARMUP": "100", "REF_ITERS": str(iters), "REF_TIME_ONLY": who}))
                    win = [x for x in smp if t_end - 0.85 * a.window - 0.3 <= x[0] <= t_end - 0.3]
                    res[who].append((tf_of(o, who), statistics.mean(x[1] for x in win) if win else 0, statistics.mean(x[2] for x in win) if win else 0))
            med = {w: tuple(statistics.median(x[i] for x in res[w]) for i in range(3)) for w in res}
            row = dict(shape=f"B{b}H{h}S{s}" + (" percu" if path == "vs" else ""), mode=mode, path=dpath, ours=med["ours"], vendor=med["vendor"],
                       ratio=med["ours"][0] / med["vendor"][0] if med["vendor"][0] else 0, acc_ours=oa, acc_vendor=va)
            rows.append(row)
            print(json.dumps(row), flush=True)
    print("\n| shape | data | path | ours TF/s | vendor TF/s | ratio | ours MHz / W | vendor MHz / W | ours bad / mean | vendor bad / mean |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        o, v = r["ours"], r["vendor"]
        ao, av = r["acc_ours"] or (-1, 0), r["acc_vendor"] or (-1, 0)
        print(f"| {r['shape']} | {r['mode']} | {r['path']} | {o[0]:.1f} | {v[0]:.1f} | {r['ratio']:.3f} | {o[2]:.0f} / {o[1]:.0f} | {v[2]:.0f} / {v[1]:.0f} | "
              f"{ao[0]} / {ao[1]:.2e} | {av[0]} / {av[1]:.2e} |")


if __name__ == "__main__":
    main()
