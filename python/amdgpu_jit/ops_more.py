"""The remaining families: MLA decode, sampling (argmax / temperature / top-k + top-p), per-row top-K, causal_conv1d (prefill fwd and the
decode update), the A16W4 / A8W4 GEMMs and the attention backward. Thin wrappers over ``_generic.launch``; dispatch rules via ``amdc
dispatch`` (``ops_compute.dispatch``)."""
import math

from ._generic import launch
from ._tensor import as_ref
from .ops_compute import _Raw, _workspace, dispatch


# ----------------------------------------------------------------------------------------------------------------------------- MLA decode
def mla_decode(q, kv_cache, block_table, context_lens, out, *, scale=None, stream=None, check=True):
    """MLA decode (DeepSeek, absorbed): q [B, H, 576] bf16 (512 latent + 64 rope dims), kv_cache [num_pages, P, 576] bf16 (one latent KV
    head; V = the first 512 dims), block_table [B, max_pages] int32, context_lens [B] int32, out [B, H, 512] bf16. The form (per-wave or
    head-parallel) and the splits come from ``mla_config``; partials in a workspace, then ``attn_combine_bf16_hd512``. Returns (kernel, S)."""
    rq, rk, rbt = as_ref(q), as_ref(kv_cache), as_ref(block_table)
    B, H, Dq = rq.shape
    NP, P = rk.shape[0], rk.shape[1]
    if Dq != 576 or rk.shape[2] != 576:
        raise ValueError("mla_decode: q / kv_cache rows of 576 (512 + 64)")
    plan = dispatch("mla", B, H)
    hp, S = bool(plan["hp"]), int(plan["splits"])
    name = f"mla_decode_bf16_{'hp_' if hp else ''}p{P}"
    parts = S if hp else 4 * S
    po, pl = B * H * parts * 512 * 4, B * H * parts * 4
    wpo, wpl = _Raw(_workspace("mla_po", po), po), _Raw(_workspace("mla_plse", pl), pl)
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(576)
    launch(name, {"ptr_po": wpo, "ptr_q": rq, "ptr_kv": rk, "ptr_bt": rbt, "ptr_cl": context_lens, "scale": sc, "H": H,
                  "max_pages": rbt.shape[1], "num_pages": NP, "S": S, "ptr_plse": wpl}, syms={"B": B}, stream=stream, check=check)
    launch("attn_combine_bf16_hd512", {"ptr_o": out, "ptr_po": wpo, "ptr_plse": wpl, "H": H, "NS": parts}, syms={"B": B}, stream=stream,
           check=check)
    return name, S


# ----------------------------------------------------------------------------------------------------------------------------- sampling
def argmax(logits, out, *, stream=None, check=True):
    """Greedy: out[b] (int32) = argmax_v logits[b, v] (bf16 [B, V]; ties: the lowest index)."""
    rl = as_ref(logits)
    B, V = rl.shape
    return launch(f"argmax_bf16_v{V}", ptrs=[rl, out], scalars=[B], stream=stream, check=check)


def sample(logits, u, out, *, temperature=1.0, top_k=0, top_p=1.0, stream=None, check=True):
    """Temperature sampling with the caller's uniform numbers u [B] f32 in [0, 1): p = softmax(logits / T) (bf16 [B, V]), optionally
    restricted to the top_k largest (0 = off) and then the top_p nucleus (1 = off; values kept as sets: ties at the threshold stay);
    out[b] (int32) = the smallest index whose cumulative probability reaches u[b] of the kept mass."""
    rl = as_ref(logits)
    B, V = rl.shape
    inv_t = 1.0 / float(temperature)
    if top_k > 0 or top_p < 1.0:
        return launch(f"sample_topkp_bf16_v{V}", ptrs=[rl, u, out], scalars=[inv_t, float(top_p), int(top_k), B], stream=stream, check=check)
    return launch(f"sample_bf16_v{V}", ptrs=[rl, u, out], scalars=[inv_t, B], stream=stream, check=check)


def topk_per_row_decode(logits, seq_lens, indices, *, next_n=1, stream=None, check=True):
    """Sparse-attention indexer, decode: logits f32 [B next_n, stride]; row b next_n + j has seq_lens[b] - next_n + j + 1 valid columns;
    indices int32 [B next_n, K] = the columns of the K largest (unordered; -1 past a short row)."""
    rl, ri = as_ref(logits), as_ref(indices)
    R, stride = rl.shape
    K = ri.shape[1]
    B = R // next_n
    return launch(f"topk_per_row_decode_f32_k{K}", ptrs=[rl, seq_lens, ri], scalars=[stride, next_n, R], grid=(B, next_n, 1),
                  stream=stream, check=check)


def topk_per_row_prefill(logits, row_starts, row_ends, indices, *, stream=None, check=True):
    """Prefill: row r's valid columns are [row_starts[r], row_ends[r]); indices int32 [R, K] absolute columns (-1 past a short row)."""
    rl, ri = as_ref(logits), as_ref(indices)
    R, stride = rl.shape
    K = ri.shape[1]
    return launch(f"topk_per_row_prefill_f32_k{K}", ptrs=[rl, row_starts, row_ends, ri], scalars=[stride, R], stream=stream, check=check)


# ----------------------------------------------------------------------------------------------------------------------------- causal conv1d
def causal_conv1d_fwd(x, weight, bias, cu_seqlens, cache_indices, has_initial, conv_states, out, *, max_seqlen, silu=True, stream=None,
                      check=True):
    """Prefill / varlen causal conv1d (channel-last): x, out [T, dim] bf16, weight [dim, W], bias [dim], cu_seqlens int32 [B + 1],
    cache_indices int32 [B] (negative = skip), has_initial uint8 [B], conv_states bf16 [slots, dim, W - 1] (read as the initial state, then
    the last W - 1 inputs written back). ``out = bias + sum_i weight[:, i] x[t - (W - 1) + i]`` (silu optional)."""
    rx, rw, rs = as_ref(x), as_ref(weight), as_ref(conv_states)
    T, dim = rx.shape
    W = rw.shape[1]
    B = as_ref(cache_indices).shape[0]
    name = f"causal_conv1d_fwd_bf16_w{W}{'_silu' if silu else ''}"
    return launch(name, ptrs=[rx, rw, bias, cu_seqlens, cache_indices, has_initial, rs, out], scalars=[dim, T, rs.shape[0]],
                  grid=(-(-dim // 2048), B, -(-int(max_seqlen) // 8)), stream=stream, check=check)


def causal_conv1d_update(x, weight, bias, state, out, *, state_indices=None, silu=True, stream=None, check=True):
    """Decode update, one new token per sequence: x, out [B, dim] bf16, weight [dim, W], bias [dim], state bf16 [slots, dim, SL] (shifted left
    with x appended); ``state_indices`` int32 [B] (continuous batching: the state slot per sequence, negative = padding) or None (slot = b)."""
    rx, rw, rs = as_ref(x), as_ref(weight), as_ref(state)
    B, dim = rx.shape
    W, SL = rw.shape[1], rs.shape[2]
    name = f"causal_conv1d_update_bf16_w{W}_s{SL}{'_idx' if state_indices is not None else ''}{'_silu' if silu else ''}"
    ptrs = [rx, rw, bias, rs, out] + ([state_indices] if state_indices is not None else [])
    return launch(name, ptrs=ptrs, scalars=[dim, rs.shape[0]], grid=(-(-dim // 256), B, 1), stream=stream, check=check)


# ----------------------------------------------------------------------------------------------------------------------------- W4 GEMMs
def a16w4_permute(data: bytes, n: int, k: int) -> bytes:
    """The K order of ``gemm_a16w4_mxfp4``'s B inside every 256-K tile (packed fp4 [n, k/2], low nibble first): physical position p of a tile
    holds logical K 128 (p/128) + 16 ((p/64)%2) + 8 ((p/32)%2) + 32 ((p/8)%4) + p%8. Apply before ``shuffle_weight(.., n, k/2, 1)``."""
    out = bytearray(len(data))
    for r in range(n):
        for kk in range(k):
            t, p = divmod(kk, 256)
            src = r * k + 256 * t + 128 * (p // 128) + 16 * ((p // 64) % 2) + 8 * ((p // 32) % 2) + 32 * ((p // 8) % 4) + p % 8
            code = (data[src // 2] >> (4 * (src & 1))) & 15
            dst = r * k + kk
            out[dst // 2] |= code << (4 * (dst & 1))
    return bytes(out)


def gemm_w4(a, b, out, *, b_scale, a_scale=None, kind="a16w4", K=None, stream=None, check=True):
    """MXFP4-weight GEMMs, ``out[M, N] bf16 = A B^T``: B [N, K/2] packed e2m1, preshuffled (``shuffle_weight``; a16w4: ``a16w4_permute``
    first), E8M0 1x32 scales.
      "a16w4"  A bf16 [M, K] exact; b_scale [N, K/32] unshuffled; M % 64, N % 128, K % 256 == 0  (``gemm_a16w4_mxfp4``)
      "a8w4"   A fp8 e4m3 [M, K] with a_scale E8M0 [M, K/32]; both scales in ``shuffle_mx_scales`` order; M, N % 128 (``gemm_a8w4_mxfp_128``)"""
    ra, ro = as_ref(a), as_ref(out)
    M, N = ro.shape
    K = K or ra.shape[1]
    if kind == "a16w4":
        ones = M * (K // 32)
        sa = a_scale if a_scale is not None else _Raw(_ones127(ones), ones)
        name = "gemm_a16w4_mxfp4"
    elif kind == "a8w4":
        sa = a_scale
        name = "gemm_a8w4_mxfp_128"
    else:
        raise ValueError("gemm_w4: kind 'a16w4' or 'a8w4'")
    kt = K // 256
    vals = {"ptr_a": ra, "ptr_b": b, "ptr_c": ro, "m": M, "n": N, "k": K, "batch": 1, "stride_a": K, "stride_b": K, "stride_c": N,
            "stride_a_batch": 0, "stride_b_batch": 0, "stride_c_batch": 0, "ptr_sfa": sa, "ptr_sfb": b_scale, "stride_sfa": M,
            "stride_sfb": kt, "stride_sfa_batch": 0, "stride_sfb_batch": 0}
    launch(name, vals, stream=stream, check=check)
    return name


def _ones127(n):
    """a persistent buffer of E8M0 ones (127) for the unused A scales of a16w4"""
    from . import _hip
    key = ("e8m0_ones", n)
    from .ops_compute import _ws
    if key not in _ws:
        ptr = _hip.malloc(n)
        _hip.memset(ptr, 127, n)
        _ws[key] = (ptr, n)
    return _ws[key][0]


# ----------------------------------------------------------------------------------------------------------------------------- attention bwd
def attention_bwd_bf16(q, k, v, o, do, lse, dq, dk, dv, *, causal=False, scale=None, cus=256, stream=None, check=True):
    """Attention backward, head dim 128, bf16 BSHD: q, o, do, dq [B, SQ, H, 128], k, v [B, SK, HK, 128], lse f32 [B, H, SQ] (natural log, from
    the forward), dk, dv [B, SK, H, 128] PER QUERY HEAD (sum the G = H / HK heads of a group for GQA). Causal bottom-right aligned. Runs
    ``fmha_bwd_dot`` (Dot = rowsum(dO O), workspace), then ``fmha_bwd_dq`` / ``fmha_bwd_dkdv``; causal: the kernels of the dispatch rule
    (``bwd_causal_kernels``: the causal kernels (longest-first launch order built in) ``*_causal``, grid (H, B, tiles) from their contracts). Returns the dkdv kernel."""
    rq, rk = as_ref(q), as_ref(k)
    B, SQ, H, D = rq.shape
    SK, HK = rk.shape[1], rk.shape[2]
    if D != 128:
        raise ValueError("attention_bwd_bf16: head dim 128")
    dot_bytes = B * H * SQ * 4
    dot = _Raw(_workspace("bwd_dot", dot_bytes), dot_bytes)
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    vals = {"q": rq, "k": rk, "v": v, "do": do, "lse": lse, "dot": dot, "dq": dq, "dk": dk, "dv": dv, "scale": sc, "SQ": SQ, "SK": SK, "H": H,
            "HK": HK, "o": o}
    syms = {"B": B}
    if causal:
        plan = dispatch("bwd_causal")
        dqk, dkdv = plan["dq"], plan["dkdv"]
    else:
        dqk, dkdv = "fmha_bwd_dq_hd128_bf16", "fmha_bwd_dkdv_hd128_bf16"
    launch("fmha_bwd_dot_hd128_bf16", vals, syms=syms, stream=stream, check=check)
    launch(dqk, vals, syms=syms, stream=stream, check=check)
    launch(dkdv, vals, syms=syms, stream=stream, check=check)
    return dkdv
