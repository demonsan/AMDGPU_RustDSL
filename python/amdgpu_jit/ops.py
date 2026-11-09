"""High-level ops: the MXFP8 blockscale GEMM, the MXFP8 FMHA forward (head dim 128) and the bf16 paged-KV decode, with the shipped dispatch.

Both ops take device tensors (torch tensors, ``__dlpack__`` objects, ``DeviceBuffer`` or ``RawPtr``), validate shapes / element sizes /
alignment, specialize the kernel for the call's shape by default (``specialize=False``: the generic kernel), check the launch against the
kernel's verified host contract and launch on the given stream (default: torch's current stream when torch tensors are passed, else 0).
"""
import math

from . import _hip
from ._compile import get_kernel
from ._tensor import as_ref, current_stream, expect

GEMM_KERNEL = "gemm_mxfp8_bs_v4"
FMHA_VS128 = "fmha_fwd_hd128_mxfp8_v3_vs128"
FMHA_T128 = "fmha_fwd_hd128_mxfp8_v3_t128"
FMHA_KVSPLIT = "fmha_fwd_hd128_mxfp8_v3_kvsplit"
FMHA_COMBINE = "fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"
D = 128


# ----------------------------------------------------------------------------------------------------------------------------- GEMM
def gemm_mxfp8_blockscale(a, b, sa, sb, out, *, specialize=True, stream=None, check=True):
    """``out[m, n] (bf16) = sum_k a[m, k] 2^(sa[k/128][m] - 127) * b[n, k] 2^(sb[n/128][k/128] - 127)``.

    a   [M, K]           fp8 e4m3 (1-byte elements), row-major
    b   [N, K]           fp8 e4m3, preshuffled with aiter ``shuffle_weight(layout=(16, 16))``
    sa  [K/128, M]       E8M0 bytes (the column-major view of the 1x128 activation scales)
    sb  [N/128, K/128]   E8M0 bytes
    out [M, N]           bf16 (2-byte elements)
    M, N multiples of 256, K a multiple of 128, K <= 8192. Returns the kernel used.
    """
    ra, rb, rsa, rsb, ro = (as_ref(x) for x in (a, b, sa, sb, out))
    M, K = ra.shape
    N = rb.shape[0]
    if M % 256 or N % 256 or K % 128:
        raise ValueError(f"gemm_mxfp8_blockscale: M, N must be multiples of 256 and K of 128 (got {M}x{N}x{K})")
    KB = K // 128
    expect(ra, "a", (M, K), 1)
    expect(rb, "b", (N, K), 1)
    expect(rsa, "sa", (KB, M), 1, align=4)
    expect(rsb, "sb", (N // 128, KB), 1, align=4)
    expect(ro, "out", (M, N), 2)
    spec = {"m": M, "n": N, "k": K} if specialize else None
    kern = get_kernel(GEMM_KERNEL, spec)
    vals = {"ptr_a": ra.ptr, "ptr_b": rb.ptr, "ptr_c": ro.ptr, "m": M, "n": N, "k": K, "batch": 1, "stride_a": K, "stride_b": K, "stride_c": N,
            "stride_a_batch": M * K, "stride_b_batch": N * K, "stride_c_batch": M * N, "ptr_sfa": rsa.ptr, "ptr_sfb": rsb.ptr, "stride_sfa": M,
            "stride_sfb": KB, "stride_sfa_batch": M * KB, "stride_sfb_batch": (N // 128) * KB}
    grid = (N // 256, M // 256, 1)
    syms = {"MT": M // 256, "NT": N // 256, "KT": KB}
    sizes = {"ptr_a": ra.cap, "ptr_b": rb.cap, "ptr_c": ro.cap, "ptr_sfa": rsa.cap, "ptr_sfb": rsb.cap}
    kern.launch(_present(kern, vals), grid, syms, sizes, current_stream([ra, rb, rsa, rsb, ro], stream), check)
    return kern


def _present(kern, vals):
    """Drops values of arguments the kernel does not declare (none for the shipped kernels; keeps ops robust to ABI-tail differences)."""
    return {k: v for k, v in vals.items() if k in kern.info.by_name}


# ----------------------------------------------------------------------------------------------------------------------------- FMHA
def select_path(B, H, S, SK, *, cus=256, partial_bytes=2):
    """The shipped dispatch (tests/gpu/ref_host.cpp, fitted on realistic data): KV split when ntiles >= 16 and N256 <= 64 (4 splits) or
    ntiles >= 32 and N256 <= 128 (2 splits), with N256 = ceil(S/256) H B and ntiles = ceil(SK/128); otherwise ``t128`` when N256 <= 64, else
    ``vs128``. Returns (path, splits)."""
    blocks256 = math.ceil(S / 256) * H * B
    ntiles = (SK + 127) // 128
    s = 4 if (ntiles >= 16 and blocks256 <= 64) else 2 if (ntiles >= 32 and blocks256 <= 128) else 0
    if s:
        tps = (ntiles + s - 1) // s
        if (s - 1) * tps < ntiles:
            return "kvsplit", s
    return ("t128" if blocks256 <= 64 else "vs128"), 0

_workspace = {}


def _ws(key, nbytes):
    """Persistent device workspace (KV-split partials), grown on demand."""
    buf = _workspace.get(key)
    if buf is None or buf[1] < nbytes:
        if buf is not None:
            _hip.free(buf[0])
        buf = _workspace[key] = (_hip.malloc(nbytes), nbytes)
    return buf[0]


def fmha_fwd_mxfp8(q, k, v, q_scale, k_scale, v_scale, out, *, path="auto", splits=None, specialize=True, stream=None, check=True):
    """MXFP8 attention forward, head dim 128 (``out = softmax(q k^T) v`` in base 2 with the E8M0 1x32 scales; the vendor kernel ABI).

    q        [B, S, H, 128]   fp8 e4m3          k, v [B, SK, HK, 128] fp8 e4m3 (H = gqa HK, gqa a power of two <= 16)
    q_scale  [B, S, H, 4]     E8M0 bytes        k_scale [B, SK, HK, 4] E8M0 bytes; both need the zeroed slack tail the kernel contract
                                                 states (512 H 4 / 640 H 4 bytes behind the tensor: allocate it and pass a view)
    v_scale  f32 V descale (a buffer of at least 4096 bytes)
    out      [B, S, H, 128]   bf16
    path: "auto" (the shipped dispatch, ``select_path``) | "vs128" | "t128" | "kvsplit" (``splits`` in {2, 4, 8}). Returns (path, splits).
    """
    rq, rk, rv, rqs, rks, rvs, ro = (as_ref(x) for x in (q, k, v, q_scale, k_scale, v_scale, out))
    B, S, H, d = rq.shape
    _, SK, HK, _ = rk.shape
    if d != D or rk.shape[3] != D or H % HK or (H // HK) & (H // HK - 1) or H // HK > 16:
        raise ValueError(f"fmha_fwd_mxfp8: head dim must be 128 and H a power-of-two multiple (<= 16) of HK (q {rq.shape}, k {rk.shape})")
    gqa = H // HK
    expect(rq, "q", (B, S, H, D), 1)
    expect(rk, "k", (B, SK, HK, D), 1)
    expect(rv, "v", (B, SK, HK, D), 1)
    expect(rqs, "q_scale", (B, S, H, 4), 1, align=4)
    expect(rks, "k_scale", (B, SK, HK, 4), 1, align=4)
    expect(ro, "out", (B, S, H, D), 2)
    if rvs.cap < 4:
        raise ValueError("v_scale: needs at least one f32")
    if path == "auto":
        path, splits = select_path(B, H, S, SK)
    elif path == "kvsplit":
        splits = splits or 4
    st = current_stream([rq, rk, rv, rqs, rks, rvs, ro], stream)
    base = {"ptr_o": ro.ptr, "ptr_q": rq.ptr, "ptr_k": rk.ptr, "ptr_v": rv.ptr, "s_seq_len": S, "s_Seqs": H * D, "s_Hs": D, "s_Bs": S * H * D,
            "s_gqa": gqa, "s_k_Seqs": HK * D, "s_k_Hs": D, "s_k_Bs": SK * HK * D, "s_kv_seq_len": SK, "s_q_head_num": H, "s_v_Seqs": HK * D,
            "s_v_Hs": D, "s_v_Bs": SK * HK * D, "s_o_Seqs": H * D * 2, "s_o_Hs": D * 2, "s_o_Bs": S * H * D * 2, "ptr_q_descale": rqs.ptr,
            "ptr_k_descale": rks.ptr, "ptr_v_descale": rvs.ptr, "s_descale_q_Bs": S * H * 4, "s_descale_q_Hs": 4, "s_descale_k_Bs": SK * HK * 4,
            "s_descale_k_Hs": 4, "abi_tail": 0}
    sizes = {"ptr_o": ro.cap, "ptr_q": rq.cap, "ptr_k": rk.cap, "ptr_v": rv.cap, "ptr_q_descale": rqs.cap, "ptr_k_descale": rks.cap, "ptr_v_descale": rvs.cap}
    syms = {"HK": HK, "B": B}
    shape_spec = {"s_seq_len": S, "s_kv_seq_len": SK, "s_q_head_num": H, "s_gqa": gqa}
    if path in ("vs128", "t128"):
        rows = 256 if path == "vs128" else 128
        kern = get_kernel(FMHA_VS128 if path == "vs128" else FMHA_T128, shape_spec if specialize else None)
        kern.launch(_present(kern, base), (math.ceil(S / rows), H, B), syms, sizes, st, check)
        return path, 0
    if path != "kvsplit" or splits not in (2, 4, 8):
        raise ValueError(f"fmha_fwd_mxfp8: unknown path {path!r} / splits {splits}")
    rows_total = B * S * H
    ohat = _ws(("ohat", st), splits * rows_total * D * 4)
    lse = _ws(("lse", st), splits * rows_total * 4)
    kern = get_kernel(FMHA_KVSPLIT, dict(shape_spec, kvs_splits=splits) if specialize else None)
    vals = dict(base, ptr_ohat=ohat, ptr_lse=lse, kvs_splits=splits, kvs_rows_total=rows_total)
    sizes2 = dict(sizes, ptr_ohat=splits * rows_total * D * 4, ptr_lse=splits * rows_total * 4)
    kern.launch(_present(kern, vals), (math.ceil(S / 256), H, B * splits), syms, sizes2, st, check)
    comb = get_kernel(FMHA_COMBINE, {"rows_total": rows_total, "splits": splits} if specialize else None)
    cvals = {"ptr_ohat": ohat, "ptr_lse": lse, "ptr_out": ro.ptr, "rows_total": rows_total, "d": D, "splits": splits}
    cgrid = max(1, math.ceil(rows_total * D / (256 * 8)))
    comb.launch(_present(comb, cvals), (cgrid, 1, 1), {}, {"ptr_ohat": sizes2["ptr_ohat"], "ptr_lse": sizes2["ptr_lse"], "ptr_out": ro.cap}, st, check)
    return path, splits


def decode16_plan(b, hk, ctx, page=16):
    """The paged-decode (bf16, hd128, GQA <= 16) launch plan - the single rule ``amdgpu_kernels::dispatch::decode16_plan``, asked from amdc
    (``amdc dispatch decode16``): returns ``(kernel_name, form, splits)`` with form "fused" (S = 1, no combine kernel) or "lastc" (the
    last-arriving CTA merges the S partials, no combine kernel)."""
    import json
    import subprocess
    from ._compile import amdc_path
    out = subprocess.run([amdc_path(), "dispatch", "decode16", str(int(b)), str(int(hk)), str(int(ctx)), "--page", str(int(page))],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    return j["kernel"], j["form"], int(j["splits"])


# ----------------------------------------------------------------------------------------------------------------------------- paged decode
def _plan_cached(b, hk, ctx, page):
    key = (int(b), int(hk), int(ctx), int(page))
    hit = _plans.get(key)
    if hit is None:
        hit = _plans[key] = decode16_plan(*key)
    return hit


_plans = {}


def _ws_zeroed(key, nbytes):
    """Persistent workspace that is zero when first allocated (the last-arriver counters: the kernel resets them after each use)."""
    buf = _workspace.get(key)
    if buf is None or buf[1] < nbytes:
        if buf is not None:
            _hip.free(buf[0])
        ptr = _hip.malloc(nbytes)
        _hip.memset(ptr, 0, nbytes)
        buf = _workspace[key] = (ptr, nbytes)
    return buf[0]


def paged_decode_bf16(q, k_cache, v_cache, block_table, context_lens, out, *, scale=None, max_context=None, stream=None, check=True):
    """Paged-KV decode attention (one new token per sequence), bf16, head dim 128, GQA group G = H / HK <= 16:
    ``out[b, h] = softmax(scale q[b, h] . K[b, :len_b, h / G]) V[b, :len_b, h / G]``.

    q             [B, H, 128]               bf16
    k_cache       [num_pages, P, HK, 128]   bf16 (NHD pages, P = page size, a power of two >= 16)
    v_cache       [num_pages, P, HK, 128]   bf16
    block_table   [B, max_pages]            int32 (page of each P-token block of a sequence)
    context_lens  [B]                       int32 (tokens per sequence, <= max_pages P)
    out           [B, H, 128]               bf16
    scale defaults to 1 / sqrt(128); ``max_context`` (the longest sequence, default max_pages P) only steers the dispatch. The kernel and
    its split count come from the single dispatch rule (``decode16_plan``): the fused form for B HK >= 256, else the last-arriver form
    (``_c``; partials and the ticket counters live in a persistent workspace). Returns (kernel_name, splits).
    """
    rq, rk, rv, rbt, rcl, ro = (as_ref(x) for x in (q, k_cache, v_cache, block_table, context_lens, out))
    B, H, Dq = rq.shape
    NP, P, HK, Dk = rk.shape
    if Dq != D or Dk != D:
        raise ValueError(f"paged_decode_bf16: head dim must be {D} (q {Dq}, cache {Dk})")
    if H % HK or H // HK > 16:
        raise ValueError(f"paged_decode_bf16: H ({H}) must be a multiple of HK ({HK}) with G = H / HK <= 16")
    if P < 16 or P & (P - 1):
        raise ValueError(f"paged_decode_bf16: page size {P} must be a power of two >= 16")
    MP = rbt.shape[1]
    expect(rq, "q", (B, H, D), 2)
    expect(rk, "k_cache", (NP, P, HK, D), 2)
    expect(rv, "v_cache", (NP, P, HK, D), 2)
    expect(rbt, "block_table", (B, MP), 4, align=4)
    expect(rcl, "context_lens", (B,), 4, align=4)
    expect(ro, "out", (B, H, D), 2)
    ctx = int(max_context) if max_context else MP * P
    name, form, S = _plan_cached(B, HK, ctx, P)
    kern = get_kernel(name)
    st = current_stream([rq, rk, rv, rbt, rcl, ro], stream)
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    vals = {"ptr_q": rq.ptr, "ptr_k": rk.ptr, "ptr_v": rv.ptr, "ptr_bt": rbt.ptr, "ptr_cl": rcl.ptr, "scale": sc, "H": H, "HK": HK,
            "max_pages": MP, "num_pages": NP, "S": S}
    sizes = {"ptr_q": rq.cap, "ptr_k": rk.cap, "ptr_v": rv.cap, "ptr_bt": rbt.cap, "ptr_cl": rcl.cap}
    lse_bytes = B * H * S * 16
    vals["ptr_plse"] = _ws("pa_lse", lse_bytes)
    sizes["ptr_plse"] = lse_bytes
    if form == "fused":
        vals["ptr_po"] = ro.ptr
        sizes["ptr_po"] = ro.cap
    else:
        po_bytes = B * H * S * 4 * 4 * D
        cnt_bytes = B * HK * 4
        vals["ptr_po"] = _ws("pa_partials", po_bytes)
        sizes["ptr_po"] = po_bytes
        vals["ptr_counter"] = _ws_zeroed("pa_counters", cnt_bytes)
        sizes["ptr_counter"] = cnt_bytes
        vals["ptr_out"] = ro.ptr
        sizes["ptr_out"] = ro.cap
    kern.launch(_present(kern, vals), (HK, B, S), {"G": H // HK, "B": B}, sizes, st, check)
    return name, S
