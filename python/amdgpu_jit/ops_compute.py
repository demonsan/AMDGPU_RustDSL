"""Compute-bound ops (docs/op-coverage-design.md §7): the GEMM family (bf16, fp8 row/col scales, fp8 blockscale with the split-K dispatch,
MXFP4), the bf16 attention prefill (the dispatch rule picks the kernel and the key splits) and the fused MoE (routing sort, stage 1, the
re-quantization, stage 2, the top-k sum). Thin wrappers over ``_generic.launch``; the dispatch rules are asked from amdc (``amdc dispatch``,
the single rule in crates/amdgpu-kernels/src/dispatch.rs). Host-side layout helpers (``shuffle_weight``, ``shuffle_mx_scales``,
``moe_w1_interleave``) prepare weights from bytes for torch-free callers."""
import json
import math
import subprocess

from . import _hip
from ._compile import amdc_path, get_kernel
from ._generic import contract_grid, infer_syms, launch
from ._kernarg import pack_fast, scalars_to_ka
from ._tensor import RawPtr, as_ref

_disp = {}


def dispatch(*args):
    """``amdc dispatch <args>`` as a dict (cached)."""
    key = tuple(str(a) for a in args)
    if key not in _disp:
        out = subprocess.run([amdc_path(), "dispatch", *key], capture_output=True, text=True, check=True).stdout
        _disp[key] = json.loads(out)
    return _disp[key]


_ws = {}


def _workspace(key, nbytes):
    buf = _ws.get(key)
    if buf is None or buf[1] < nbytes:
        if buf is not None:
            from ._generic import invalidate_launch_memos
            invalidate_launch_memos()
            _hip.free(buf[0])
        buf = _ws[key] = (_hip.malloc(nbytes), nbytes)
    return buf[0]


def _Raw(ptr, nbytes):
    """a workspace pointer with its size (a tensor for ``launch``)"""
    return RawPtr(ptr, (nbytes,), 1, nbytes)


# ----------------------------------------------------------------------------------------------------------------------------- layouts
def shuffle_weight(data: bytes, rows: int, k: int, es: int) -> bytes:
    """aiter ``shuffle_weight(layout=(16, 16))`` of a row-major [rows, k] array of ``es``-byte elements (the B operand of the GEMMs and the
    MoE weights): 16-row x 16-byte blocks. rows % 16 == 0, k * es % 32 == 0."""
    kp, bk = 16 // es, 32
    out = bytearray(len(data))
    for n in range(rows):
        n0, ni = divmod(n, 16)
        for kk in range(k):
            k0, rem = divmod(kk, bk)
            d = n0 * 16 * k + k0 * 16 * bk + (rem // kp) * 16 * kp + ni * kp + rem % kp
            s = (n * k + kk) * es
            out[d * es:(d + 1) * es] = data[s:s + es]
    return bytes(out)


def shuffle_mx_scales(data: bytes, rows: int, k: int) -> bytes:
    """The E8M0 1x32 scales of ``gemm_a4w4_mxfp4`` ([rows, k/32] bytes, row-major) in its shuffled order: [rows/16][k/512][64 lanes][4]."""
    kg, kt = k // 32, k // 256
    out = bytearray(len(data))
    for rb in range(rows // 16):
        for p in range(kt // 2):
            for lane in range(64):
                r, g = rb * 16 + lane % 16, lane // 16
                d = ((rb * (kt // 2) + p) * 64 + lane) * 4
                for h in range(2):
                    t = 2 * p + h
                    out[d + 2 * h] = data[r * kg + 8 * t + g]
                    out[d + 2 * h + 1] = data[r * kg + 8 * t + 4 + g]
    return bytes(out)


def moe_w1_interleave(data: bytes, experts: int, inter: int, d: int, es: int) -> bytes:
    """W1 [E, 2I, D] (gate rows, then up rows) -> the stage-1 order: gate / up interleaved in 16-row blocks (row 32 q + r = gate 16 q + r,
    32 q + 16 + r = up 16 q + r); apply ``shuffle_weight`` afterwards. Row scales (fp8): ``moe_w1_interleave(scales, E, I, 1, 4)``."""
    row = d * es
    out = bytearray(len(data))
    for e in range(experts):
        for r in range(2 * inter):
            q, rr = divmod(r, 32)
            src = 16 * q + rr if rr < 16 else inter + 16 * q + rr - 16
            o, s = (e * 2 * inter + r) * row, (e * 2 * inter + src) * row
            out[o:o + row] = data[s:s + row]
    return bytes(out)


# ----------------------------------------------------------------------------------------------------------------------------- GEMM
_GEMM = {"bf16": ("gemm_a16w16_bf16", 2, 64), "fp8_rowcol": ("gemm_a8w8_rowcol_fp8", 1, 128), "fp8_blockscale": ("gemm_a8w8_blockscale", 1, 128),
         "mxfp4": ("gemm_a4w4_mxfp4", 0, 256)}


def gemm(a, b, out, *, kind="bf16", a_scale=None, b_scale=None, M=None, N=None, K=None, splits="auto", tile="auto", cus=256, stream=None,
         check=True):
    """``out[M, N] (bf16) = A[M, K] B[N, K]^T`` with B preshuffled (``shuffle_weight``, 16 x 16). kind:
      "bf16"           A, B bf16
      "fp8_rowcol"     A, B fp8 e4m3; a_scale f32 [M], b_scale f32 [N] (per row / column)
      "fp8_blockscale" A, B fp8; a_scale f32 [M, K/128] (row-major), b_scale f32 [N/128, K/128]; split-K by the dispatch rule (``splits``)
      "mxfp4"          A [M, K/2], B [N, K/2] packed e2m1; scales E8M0 [rows, K/32] in ``shuffle_mx_scales`` order
    Tiles: 256 x 256 CTAs (blockscale: 128) when M, N are multiples of 256 and the grid has at least ``cus`` / 4 CTAs, else the ``_128``
    kernels (bf16 / fp8_rowcol). Shapes default to the tensors'. Returns (kernel, splits)."""
    ra, rb, ro = as_ref(a), as_ref(b), as_ref(out)
    M = M or ro.shape[0]
    N = N or ro.shape[1]
    base, es, kt_el = _GEMM[kind]
    K = K or (ra.shape[1] * 2 if kind == "mxfp4" else ra.shape[1])
    name = base
    if kind in ("bf16", "fp8_rowcol"):
        big = M % 256 == 0 and N % 256 == 0 and (M // 256) * (N // 256) * 4 >= cus
        if tile == 128 or (tile == "auto" and not big):
            name = base + "_128"
    S = 1
    if kind == "fp8_blockscale":
        S = dispatch("gemm_blockscale", M, N, K, "--cus", cus)["splits"] if splits == "auto" else int(splits)
        if S > 1 and N % 1024:
            S = 1  # the reduce kernel covers N in 1024-column blocks
    kt = K // kt_el
    vals = {"ptr_a": ra, "ptr_b": rb, "ptr_c": ro, "m": M, "n": N, "k": K, "batch": 1, "stride_a": K, "stride_b": K, "stride_c": N,
            "stride_a_batch": 0, "stride_b_batch": 0, "stride_c_batch": 0, "ptr_sfa": a_scale if a_scale is not None else 0,
            "ptr_sfb": b_scale if b_scale is not None else 0, "stride_sfa": M, "stride_sfb": kt, "stride_sfa_batch": 0, "stride_sfb_batch": 0}
    if S == 1:
        launch(name, vals, stream=stream, check=check)
        return name, 1
    # split-K: fp32 partials [S][MP][N] in a workspace, then gemm_splitk_reduce_s<S>
    MP = -(-M // 128) * 128
    ws = _Raw(_workspace("gemm_splitk", S * MP * N * 4), S * MP * N * 4)
    vals.update(ptr_c=ws, batch=S, stride_sfa_batch=kt // S, stride_sfb_batch=MP)
    launch(name + "_splitk", vals, stream=stream, check=check)
    launch(f"gemm_splitk_reduce_s{S}", ptrs=[ws, ro, 0, 0], scalars=[M, N, S, MP], stream=stream, check=check)
    return name + "_splitk", S


# ----------------------------------------------------------------------------------------------------------------------------- attention
def prefill_bf16(q, k, v, out, *, causal=False, lse=None, scale=None, cus=256, stream=None, check=True):
    """bf16 attention forward, head dim 128: q, out [B, SQ, H, 128], k, v [B, SK, HK, 128] (BSHD, contiguous; H = G HK). Causal is bottom-right
    aligned (key k visible to query q iff k <= q + SK - SQ). ``lse`` (optional) f32 [B, H, SQ] (natural log). The kernel and its key splits
    come from the dispatch rule (``amdc dispatch prefill_bf16``). Returns (kernel, splits)."""
    rq, rk = as_ref(q), as_ref(k)
    B, SQ, H, D = rq.shape
    SK, HK = rk.shape[1], rk.shape[2]
    if D != 128:
        raise ValueError("prefill_bf16: head dim 128")
    plan = dispatch("prefill_bf16", B, H, SQ, SK, int(bool(causal)), "--cus", cus)
    name, S = plan["kernel"], int(plan["splits"])
    qs, ks = H * D * 2, HK * D * 2
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    want_lse = lse is not None
    if lse is None:  # the contract sizes the LSE buffer even when it is not requested
        lse = _Raw(_workspace("prefill_lse", B * H * SQ * 4), B * H * SQ * 4)
    vals = {"ptr_o": out, "ptr_q": rq, "ptr_k": rk, "ptr_v": v, "ptr_lse": lse, "scalar": sc, "s_seq_len": SQ,
            "s_Seqs": qs, "s_Ts": 256 * qs, "s_Hs": D * 2, "s_Bs": SQ * qs, "s_gqa": H // HK, "s_k_Seqs": ks, "s_k_Hs": D * 2, "s_k_Bs": SK * ks,
            "s_opt": 3 if causal and (H % 8 != 0 or SQ > 16384) else 5, "s_lse": int(want_lse), "s_kv_seq_len": SK, "s_qk_head_dim": D,
            "s_v_head_dim": D, "s_q_head_num": H, "s_v_Seqs": ks, "s_v_Hs": D * 2, "s_v_Bs": SK * ks, "s_o_Seqs": qs, "s_o_Hs": D * 2,
            "s_o_Bs": SQ * qs, "s_lse_Hs": SQ * 4}
    kern = get_kernel(name)
    if S > 1:
        po, pl = B * H * SQ * S * D * 4, B * H * SQ * S * 4
        wpo, wpl = _Raw(_workspace("prefill_po", po), po), _Raw(_workspace("prefill_plse", pl), pl)
        vals.update(lg_s=S.bit_length() - 1, S=S, ptr_po=wpo, ptr_plse=wpl)
    vals = {n: x for n, x in vals.items() if n in kern.info.by_name}
    # grid: the q tiles and heads from the contract, z = B S
    _, scal = pack_fast(kern.info, {n: (x if isinstance(x, (int, float)) else 0) for n, x in vals.items()})
    ka = scalars_to_ka(kern.info, scal)
    syms = {"BB": B} if "BB" in [s["name"] for s in kern.info.contract.get("syms", [])] else {}
    syms = infer_syms(kern.info, ka, None, syms)
    g = list(contract_grid(kern.info, ka, syms))
    g[2] = B * S
    launch(name, vals, grid=tuple(g), syms=syms, stream=stream, check=check)
    if S > 1:
        launch("attn_combine_bf16_hd128_q16", ptrs=[out, vals["ptr_po"], vals["ptr_plse"], lse], scalars=[H, SQ, S, int(want_lse)], grid=((SQ + 15) // 16, H, B), stream=stream, check=check)
    return name, S


# ----------------------------------------------------------------------------------------------------------------------------- fused MoE
def fused_moe(x, w1, w2, topk_ids, topk_weights, out, *, dtype="bf16", act="silu", x_scale=None, w1_scale=None, w2_scale=None,
              num_experts=None, inter=None, block_m="auto", stream=None, check=True):
    """Fused MoE (aiter 2-stage flow): ``out[t] = sum_j w[t, j] W2[e_j] (act(W1g[e_j] x[t]) * W1u[e_j] x[t])`` for the top-k experts e_j.
    x [M, D] (bf16, or fp8 e4m3 with x_scale f32 [M]); w1 = W1 [E, 2I, D] gate / up rows put through ``moe_w1_interleave`` then
    ``shuffle_weight``; w2 = W2 [E, D, I] through ``shuffle_weight``; fp8: per-row weight scales w1_scale f32 [E, 2I] (interleaved like
    W1), w2_scale [E, D]. topk_ids int32 / topk_weights f32 [M, K]; out [M, D] bf16. The routing sort, stage 1 (act * up, bf16 [M, K, I]),
    the per-token fp8 re-quantization (fp8), stage 2 (weighted, f32 [M, K, D]) and the top-k sum run as kernels on the stream; the
    intermediates live in a workspace. block_m from the dispatch rule. E >= 8, D % 1024 == 0, I % 128 == 0, M K % 4 == 0 (fp8). Returns block_m."""
    rx, ri = as_ref(x), as_ref(topk_ids)
    M, D = rx.shape
    K = ri.shape[1]
    E = num_experts
    I = inter
    if E is None or I is None:
        raise ValueError("fused_moe: num_experts and inter (I) are required (the weights are preshuffled bytes)")
    fp8 = dtype == "fp8"
    BM = dispatch("moe", M, K, E)["block_m"] if block_m == "auto" else int(block_m)
    bms = "_bm64" if BM == 64 else ""
    # routing sort
    maxp = M * K + E * BM - K
    ntile = -(-maxp // BM)
    sid, sw = _Raw(_workspace("moe_sid", maxp * 4), maxp * 4), _Raw(_workspace("moe_sw", maxp * 4), maxp * 4)
    seid, nv, ts = _Raw(_workspace("moe_seid", ntile * 4), ntile * 4), _Raw(_workspace("moe_nv", 8), 8), _Raw(_workspace("moe_ts", (E + 1) * 4), (E + 1) * 4)
    launch(f"moe_sorting_e{E}_k{K}_b{BM}", ptrs=[ri, topk_weights, sid, sw, seid, nv, ts], scalars=[M], stream=stream, check=check)
    MT = maxp // BM
    es = 1 if fp8 else 2
    dt = "fp8" if fp8 else "bf16"
    y1 = _Raw(_workspace("moe_y1", M * K * I * 2), M * K * I * 2)
    route = {"ptr_sorted_ids": sid, "ptr_sorted_expert_ids": seid, "ptr_num_valid_ids": nv, "ptr_sorted_weights": sw, "topk": K, "E": E,
             "M_tokens": M}

    def gemm_vals(A, B, C, m, n, k, sa, sb, nout):
        return dict(route, ptr_a=A, ptr_b=B, ptr_c=C, m=m, n=n, k=k, batch=1, stride_a=k, stride_b=k, stride_c=nout, stride_a_batch=0,
                    stride_b_batch=0, stride_c_batch=0, ptr_sfa=sa if sa is not None else 0, ptr_sfb=sb if sb is not None else 0,
                    stride_sfa=m, stride_sfb=k * es // 128, stride_sfa_batch=0, stride_sfb_batch=0, n_out=nout)

    launch(f"gemm_moe1_{dt}_{act}{bms}", gemm_vals(rx, w1, y1, M, 2 * I, D, x_scale, w1_scale, I), grid=(I // 128, MT, 1),
           syms={"MT": MT}, stream=stream, check=check)
    a2, s2 = y1, None
    if fp8:
        a2 = _Raw(_workspace("moe_y1q", M * K * I), M * K * I)
        s2 = _Raw(_workspace("moe_y1s", M * K * 4), M * K * 4)
        launch(f"quant_token_bf16_fp8_n{I}", ptrs=[y1, a2, s2], scalars=[M * K], stream=stream, check=check)
    t = _Raw(_workspace("moe_t", M * K * D * 4), M * K * D * 4)
    launch(f"gemm_moe2_{dt}{bms}", gemm_vals(a2, w2, t, M * K, D, I, s2, w2_scale, D), grid=(D // 128, MT, 1), syms={"MT": MT},
           stream=stream, check=check)
    launch(f"moe_sum_bf16_k{K}", ptrs=[t, out], scalars=[M, D], stream=stream, check=check)
    return BM
