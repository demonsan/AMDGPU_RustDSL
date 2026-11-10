#!/usr/bin/env python3
"""GPU-node check of the L1 graph layer (docs/l1-graph.md §7.3; torch-free, stdlib only).

For each graph, the fused plan (``amdgpu_jit.graph.plan``) and the unfused plan (``fuse=False``, every op through its own kernel) are run on
the same random data. Both must agree with the Python reference of the graph (the formulas and tolerances of ops_check.py) and with each
other within the quantization step. This checks that each fusion rule's pattern computes what its fused kernel computes. The graphs whose
ops have no standalone kernel (add, qknorm) are checked fused against the reference only.

    HIP_VISIBLE_DEVICES=6 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/graph_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import amdgpu_jit as aj  # noqa: E402
from ops_check import Res, act, down_bf16, down_f32, down_u8, e4m3, rand_bf16, rbf, rms_ref, up_bf16  # noqa: E402

G = aj.graph


def g_norm_quant(M, N):
    return {"tensors": {"x": {"dtype": "bf16", "shape": ["M", N]}, "g": {"dtype": "bf16", "shape": [N]}, "y": {"dtype": "bf16", "shape": ["M", N]},
                        "q": {"dtype": "fp8", "shape": ["M", N]}, "s": {"dtype": "f32", "shape": ["M"]}},
            "dims": {"M": M},
            "ops": [{"id": "norm", "op": "rmsnorm", "in": ["x", "g"], "out": ["y"], "attrs": {"eps": "1e-6"}},
                    {"id": "quant", "op": "quant_token", "in": ["y"], "out": ["q", "s"], "attrs": {"to": "fp8"}}],
            "outputs": ["q", "s"]}


def g_act_quant(M, D):
    return {"tensors": {"x": {"dtype": "bf16", "shape": ["M", 2 * D]}, "a": {"dtype": "bf16", "shape": ["M", D]},
                        "q": {"dtype": "fp8", "shape": ["M", D]}, "s": {"dtype": "f32", "shape": ["M"]}},
            "dims": {"M": M},
            "ops": [{"id": "act", "op": "act_mul", "in": ["x"], "out": ["a"], "attrs": {"act": "silu"}},
                    {"id": "quant", "op": "quant_token", "in": ["a"], "out": ["q", "s"], "attrs": {"to": "fp8"}}],
            "outputs": ["q", "s"]}


def g_add_norm_quant(M, N):
    return {"tensors": {"x": {"dtype": "bf16", "shape": ["M", N]}, "r": {"dtype": "bf16", "shape": ["M", N]}, "g": {"dtype": "bf16", "shape": [N]},
                        "h": {"dtype": "bf16", "shape": ["M", N]}, "y": {"dtype": "bf16", "shape": ["M", N]},
                        "q": {"dtype": "fp8", "shape": ["M", N]}, "s": {"dtype": "f32", "shape": ["M"]}},
            "dims": {"M": M},
            "ops": [{"id": "res", "op": "add", "in": ["x", "r"], "out": ["h"]},
                    {"id": "norm", "op": "rmsnorm", "in": ["h", "g"], "out": ["y"], "attrs": {"eps": "1e-6"}},
                    {"id": "quant", "op": "quant_token", "in": ["y"], "out": ["q", "s"], "attrs": {"to": "fp8"}}],
            "outputs": ["h", "q", "s"]}


def run_quant(graph, inputs, M, N, fuse):
    """Plans and runs a graph whose outputs are q (fp8 [M, N]) and s (f32 [M]) (+ h bf16 [M, N] when present); returns (plan, q*s, h)."""
    p = G.plan(graph, fuse=fuse)
    names = set(graph["tensors"])
    t = dict(inputs)
    for n in names - set(t):
        dt = graph["tensors"][n]["dtype"]
        shape = tuple(M if d == "M" else d for d in graph["tensors"][n]["shape"])
        t[n] = aj.DeviceBuffer(shape, {"bf16": "bfloat16", "fp8": "uint8", "f32": "float32"}[dt])
    G.run(p, t)
    aj.hip.synchronize()
    q, s = down_u8(t["q"], M * N), down_f32(t["s"], M)
    deq = [e4m3(q[i]) * s[i // N] for i in range(M * N)]
    h = down_bf16(t["h"], M * N) if "h" in t else None
    return p, deq, h, s


def check_pair(name, graph, inputs, ref_rows, M, N, want_fused, want_unfused):
    """Fused vs unfused vs the reference. ``ref_rows(m)`` = the reference row y (before quantization)."""
    r = Res(f"graph {name} M={M} N={N}")
    pf, df, _, sf = run_quant(graph, inputs, M, N, True)
    pu, du, _, su = run_quant(graph, inputs, M, N, False)
    kf, ku = G.kernels(pf), G.kernels(pu)
    if kf != want_fused or ku != want_unfused:
        print(f"    plans: fused {kf} (want {want_fused}), unfused {ku} (want {want_unfused})")
        r.bad += 1
    for m in range(M):
        y = ref_rows(m)
        sref = max(max(abs(v) for v in y), 1e-12) / 448
        r.cmp(sf[m], sref, sref * 1e-2)
        r.cmp(su[m], sref, sref * 1e-2)
        for k in range(N):
            tol = abs(y[k]) / 16 * 1.05 + sref * 2 ** -9 + abs(y[k]) / 64
            r.cmp(df[m * N + k], y[k], tol)
            r.cmp(du[m * N + k], y[k], tol)
            # fused and unfused differ only by the bf16 rounding of the intermediate (one fp8 step at most)
            r.cmp(df[m * N + k], du[m * N + k], abs(y[k]) / 8 + sref * 2 ** -8)
    r.note = f"{len(pf['launches'])} vs {len(pu['launches'])} launches"
    return r


def check_norm_quant(M=32, N=512):
    x, g = rand_bf16(M * N), rand_bf16(N)
    ins = {"x": up_bf16(x, (M, N)), "g": up_bf16(g, (N,))}
    return check_pair("rmsnorm+quant_token.fp8", g_norm_quant(M, N), ins, lambda m: rms_ref(x[m * N:(m + 1) * N], g, N, 1e-6), M, N,
                      [f"rmsnorm_quant_fp8_n{N}"], [f"rmsnorm_bf16_n{N}", f"quant_token_bf16_fp8_n{N}"])


def check_act_quant(M=16, D=512):
    x = rand_bf16(M * 2 * D)
    ins = {"x": up_bf16(x, (M, 2 * D))}
    return check_pair("act_mul.silu+quant_token.fp8", g_act_quant(M, D), ins,
                      lambda m: [act(x[m * 2 * D + k], "silu") * x[m * 2 * D + D + k] for k in range(D)], M, D,
                      [f"silu_mul_quant_fp8_d{D}"], [f"silu_mul_bf16_d{D}", f"quant_token_bf16_fp8_n{D}"])


def check_add_norm_quant(M=32, N=512):
    """Fused only (``add`` has no standalone kernel; the unfused plan must report it)."""
    r = Res(f"graph add+rmsnorm+quant_token.fp8 M={M} N={N}")
    x, res, g = rand_bf16(M * N), rand_bf16(M * N), rand_bf16(N)
    graph = g_add_norm_quant(M, N)
    ins = {"x": up_bf16(x, (M, N)), "r": up_bf16(res, (M, N)), "g": up_bf16(g, (N,))}
    p, deq, h, sc = run_quant(graph, ins, M, N, True)
    if G.kernels(p) != [f"add_rmsnorm_fp8_n{N}"]:
        print(f"    plan {G.kernels(p)}")
        r.bad += 1
    pu = G.plan(graph, fuse=False)
    if not any(e["op"] == "res" for e in pu["errors"]):
        print(f"    unfused plan should report the standalone add: {pu['errors']}")
        r.bad += 1
    for m in range(M):
        s = [rbf(x[m * N + k] + res[m * N + k]) for k in range(N)]
        for k in range(N):
            r.cmp(h[m * N + k], s[k], 0)
        y = rms_ref(s, g, N, 1e-6)
        sref = max(max(abs(v) for v in y), 1e-12) / 448
        r.cmp(sc[m], sref, sref * 1e-2)
        for k in range(N):
            r.cmp(deq[m * N + k], y[k], abs(y[k]) / 16 * 1.05 + sref * 2 ** -9 + abs(y[k]) / 64)
    return r


def gemm_graph(kind, M, N, K):
    adt = {"bf16": "bf16", "fp8_rowcol": "fp8", "fp8_blockscale": "fp8"}[kind]
    ts = {"a": {"dtype": adt, "shape": [M, K]}, "b": {"dtype": adt, "shape": [N, K]}, "c": {"dtype": "bf16", "shape": [M, N]}}
    ins = ["a", "b"]
    if kind != "bf16":
        ts["sa"] = {"dtype": "f32", "shape": [M] if kind == "fp8_rowcol" else [M, K // 128]}
        ts["sb"] = {"dtype": "f32", "shape": [N] if kind == "fp8_rowcol" else [N // 128, K // 128]}
        ins += ["sa", "sb"]
    return {"tensors": ts, "ops": [{"id": "mm", "op": "gemm", "in": ins, "out": ["c"], "attrs": {"kind": kind}}], "outputs": ["c"]}


def check_gemm_plan(kind, M, N, K, want):
    """The GEMM through its L1 plan (``dispatch::gemm_kernels``: tile, split-K, reduce; plan-owned split-K workspace) against the
    reference of ops_check.check_gemm."""
    import ops_check
    plans = []

    def run(da, db, do, dsa, dsb, k):
        p = G.plan(gemm_graph(kind, M, N, K))
        plans.append(G.kernels(p))
        t = {"a": da, "b": db, "c": do}
        if dsa is not None:
            t.update(sa=dsa, sb=dsb)
        G.run(p, t)
        aj.hip.synchronize()
        l = p["launches"][0]
        return l["kernels"][0], l["params"]["splits"]

    r = ops_check.check_gemm(kind, M, N, K, run=run)
    r.name = "graph " + r.name
    if plans and plans[0] != want:
        print(f"    plan {plans[0]} (want {want})")
        r.bad += 1
    return r


CHECKS = [("norm_quant", check_norm_quant), ("act_quant", check_act_quant), ("add_norm_quant", check_add_norm_quant),
          ("gemm_bf16_64_splitk", lambda: check_gemm_plan("bf16", 64, 1024, 512, ["gemm_a16w16_bf16_64_splitk", "gemm_splitk_reduce_s4"])),
          ("gemm_fp8_rowcol_64_splitk", lambda: check_gemm_plan("fp8_rowcol", 64, 1024, 512, ["gemm_a8w8_rowcol_fp8_64_splitk", "gemm_splitk_reduce_s4_rc"])),
          ("gemm_fp8_blockscale_64_splitk", lambda: check_gemm_plan("fp8_blockscale", 64, 1024, 512, ["gemm_a8w8_blockscale_64_splitk", "gemm_splitk_reduce_s4"])),
          ("gemm_bf16_64", lambda: check_gemm_plan("bf16", 256, 256, 256, ["gemm_a16w16_bf16_64"]))]


def main(filters):
    ok = True
    for name, fn in CHECKS:
        if filters and not any(f in name for f in filters):
            continue
        try:
            r = fn()
            good = r.bad == 0
            print(f"{'ok  ' if good else 'FAIL'} {r.name}: {r.cnt} values, bad={r.bad} {r.note}")
        except Exception as e:  # noqa: BLE001
            good = False
            print(f"FAIL {name}: {type(e).__name__}: {str(e)[:400]}")
        ok &= good
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
