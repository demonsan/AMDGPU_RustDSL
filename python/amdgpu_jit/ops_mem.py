"""Memory-bound ops (docs/op-coverage-membound.md): thin wrappers over ``_generic.launch`` - pick the kernel for the shape, check shapes /
element sizes, pass the tensors. Row counts must satisfy the kernel's verified contract (e.g. a multiple of the rows a CTA covers); the
contract check names the violated relation. All tensors contiguous, 16-byte aligned (scales: 4)."""
from ._generic import launch
from ._tensor import as_ref, expect


def _rows(ref, name, ncols, itemsize):
    if len(ref.shape) != 2 or ref.shape[1] != ncols:
        raise ValueError(f"{name}: shape {tuple(ref.shape)}, expected [M, {ncols}]")
    expect(ref, name, ref.shape, itemsize)
    return ref.shape[0]


def _mn(x):
    r = as_ref(x)
    if len(r.shape) != 2:
        raise ValueError(f"expected a 2-D [M, N] tensor, got shape {tuple(r.shape)}")
    return r, r.shape[0], r.shape[1]


# ----------------------------------------------------------------------------------------------------------------------------- norms
def rmsnorm(x, gamma, out, *, eps=1e-6, stream=None, check=True):
    """``out = x / sqrt(mean(x^2) + eps) * gamma`` per row; x, out [M, N] bf16, gamma [N] bf16."""
    rx, M, N = _mn(x)
    _rows(as_ref(out), "out", N, 2)
    return launch(f"rmsnorm_bf16_n{N}", ptrs=[rx, gamma, out], scalars=[M, eps], stream=stream, check=check)


def add_rmsnorm(x, res, gamma, out, res_out, *, eps=1e-6, stream=None, check=True):
    """``res_out = x + res`` (bf16), ``out = rmsnorm(res_out) * gamma``; all [M, N] bf16, gamma [N]."""
    rx, M, N = _mn(x)
    for t, n in ((res, "res"), (out, "out"), (res_out, "res_out")):
        _rows(as_ref(t), n, N, 2)
    return launch(f"add_rmsnorm_bf16_n{N}", ptrs=[rx, res, gamma, out, res_out], scalars=[M, eps], stream=stream, check=check)


def add_rmsnorm_quant_fp8(x, res, gamma, out, res_out, scale, *, eps=1e-6, stream=None, check=True):
    """``res_out = x + res``, ``y = rmsnorm(res_out) * gamma``, out = per-token fp8 e4m3 of y with ``scale[m] = max|y| / 448`` (f32 [M])."""
    rx, M, N = _mn(x)
    _rows(as_ref(out), "out", N, 1)
    return launch(f"add_rmsnorm_fp8_n{N}", ptrs=[rx, res, gamma, out, res_out, scale], scalars=[M, eps], stream=stream, check=check)


def rmsnorm_quant_fp8(x, gamma, q, scale, *, eps=1e-6, stream=None, check=True):
    """``y = rmsnorm(x) * gamma``; q [M, N] fp8 e4m3 per token, scale [M] f32 = max|y| / 448."""
    rx, M, N = _mn(x)
    _rows(as_ref(q), "q", N, 1)
    return launch(f"rmsnorm_quant_fp8_n{N}", ptrs=[rx, gamma, q, scale], scalars=[M, eps], stream=stream, check=check)


def rmsnorm_quant_mxfp4(x, gamma, q, scale, *, eps=1e-6, stream=None, check=True):
    """``y = rmsnorm(x) * gamma`` quantized to MXFP4: q [M, N/2] packed e2m1 (low nibble first), scale [M, N/32] E8M0 bytes in the
    shuffled a4w4 GEMM layout (group G of row m at (((m/16) (N/512) + G/16) 64 + 16 (G%4) + m%16) 4 + (G%16)/4); M a multiple of 16."""
    rx, M, N = _mn(x)
    _rows(as_ref(q), "q", N // 2, 1)
    _rows(as_ref(scale), "scale", N // 32, 1)
    return launch(f"rmsnorm_quant_mxfp4_bf16_k{N}", ptrs=[rx, gamma, q, scale], scalars=[M, eps], stream=stream, check=check)


# ----------------------------------------------------------------------------------------------------------------------------- quantization
def quant_per_token(x, q, scale, *, dtype="fp8", stream=None, check=True):
    """Dynamic per-token quantization: ``scale[m] = max|x[m]| / (448 | 127)`` (f32 [M]), ``q = round(x / scale)`` as fp8 e4m3 or int8."""
    rx, M, N = _mn(x)
    _rows(as_ref(q), "q", N, 1)
    if dtype not in ("fp8", "int8"):
        raise ValueError("quant_per_token: dtype must be 'fp8' or 'int8'")
    return launch(f"quant_token_bf16_{dtype}_n{N}", ptrs=[rx, q, scale], scalars=[M], stream=stream, check=check)


def quant_per_group128(x, q, scale, *, stream=None, check=True):
    """Dynamic per-group (1 x 128) fp8 e4m3 quantization: scale [M, N/128] f32 = max|group| / 448."""
    rx, M, N = _mn(x)
    _rows(as_ref(q), "q", N, 1)
    _rows(as_ref(scale), "scale", N // 128, 4)
    return launch(f"quant_group128_bf16_fp8_n{N}", ptrs=[rx, q, scale], scalars=[M], stream=stream, check=check)


def quant_static_fp8(x, out, scale, *, stream=None, check=True):
    """Static per-tensor fp8 e4m3: ``out = x / scale`` (scale a Python float); x bf16, out fp8, any shape (flattened)."""
    rx = as_ref(x)
    n = rx.nbytes // 2
    return launch("quant_bf16_fp8_static", ptrs=[rx, out], scalars=[float(scale), n], stream=stream, check=check)


# ----------------------------------------------------------------------------------------------------------------------------- activations
_ACT = {"silu": "silu_mul", "gelu": "gelu_mul", "gelu_tanh": "gelu_tanh_mul", "swiglu_oai": "swiglu_oai_mul"}


def act_mul(x, out, *, act="silu", stream=None, check=True):
    """Gated activation ``out = act(x[:, :D]) * x[:, D:]``; x [M, 2D] bf16, out [M, D] bf16. act: silu | gelu | gelu_tanh | swiglu_oai
    (gpt-oss: ``min(a, 7) sigmoid(1.702 min(a, 7)) (clamp(g, -7, 7) + 1)``)."""
    rx, M, D2 = _mn(x)
    D = D2 // 2
    _rows(as_ref(out), "out", D, 2)
    if act not in _ACT:
        raise ValueError(f"act_mul: act must be one of {sorted(_ACT)}")
    return launch(f"{_ACT[act]}_bf16_d{D}", ptrs=[rx, out], scalars=[M], stream=stream, check=check)


def silu_mul_quant_fp8(x, q, scale, *, stream=None, check=True):
    """``y = silu(x[:, :D]) * x[:, D:]`` quantized per token to fp8 e4m3: q [M, D], scale [M] f32 = max|y| / 448."""
    rx, M, D2 = _mn(x)
    D = D2 // 2
    _rows(as_ref(q), "q", D, 1)
    return launch(f"silu_mul_quant_fp8_d{D}", ptrs=[rx, q, scale], scalars=[M], stream=stream, check=check)


# ----------------------------------------------------------------------------------------------------------------------------- rope / kv cache
def _thd(x, name):
    r = as_ref(x)
    if len(r.shape) != 3:
        raise ValueError(f"{name}: expected [T, H, D], got shape {tuple(r.shape)}")
    expect(r, name, r.shape, 2)
    return r, r.shape[0], r.shape[1], r.shape[2]


def _style(style):
    if style not in ("neox", "gptj"):
        raise ValueError("style must be 'neox' or 'gptj'")
    return style


def rope(x, out, positions, cos_sin, *, style="neox", stream=None, check=True):
    """Rotary embedding: x, out [T, H, D] bf16, positions [T] int32, cos_sin [P, D] bf16 (cos | sin halves of the rotary dims). neox: pairs
    (i, i + D/2); gptj: pairs (2i, 2i + 1). Positions >= P give a zero rotary part."""
    rx, T, H, D = _thd(x, "x")
    P = as_ref(cos_sin).shape[0]
    return launch(f"rope_{_style(style)}_bf16_h{H}_d{D}", ptrs=[rx, out, positions, cos_sin], scalars=[T, P], stream=stream, check=check)


def qknorm_rope(x, out, positions, cos_sin, gamma, *, eps=1e-6, style="neox", stream=None, check=True):
    """Per-head RMSNorm (gamma [D] bf16) then RoPE: x, out [T, H, D] bf16."""
    rx, T, H, D = _thd(x, "x")
    P = as_ref(cos_sin).shape[0]
    return launch(f"qknorm_rope_{_style(style)}_bf16_h{H}_d{D}", ptrs=[rx, out, positions, cos_sin, gamma], scalars=[T, P, eps], stream=stream,
                  check=check)


def qknorm_rope_kvcache(k, out, positions, cos_sin, gamma, v, key_cache, value_cache, slot_mapping, *, eps=1e-6, style="neox",
                        k_scale=None, v_scale=None, stream=None, check=True):
    """Fused per-head RMSNorm + RoPE of k [T, H, D] (also written to ``out``) and the KV-cache write of the rotated k and of v at
    ``slot_mapping[t]`` (int32 or int64 [T]; negative / out of range = skipped) of the flash-layout caches [S, H, D]: bf16 caches, or fp8 e4m3
    caches with static ``k_scale`` / ``v_scale`` (stored value = x / scale)."""
    rk, T, H, D = _thd(k, "k")
    rc = as_ref(key_cache)
    S = rc.shape[0]
    fp8 = rc.itemsize == 1
    i64 = as_ref(slot_mapping).itemsize == 8
    name = f"qknorm_rope_kvcache_{_style(style)}_{'fp8' if fp8 else 'bf16'}_h{H}_d{D}{'_i64' if i64 else ''}"
    args = {"x": rk, "out": out, "pos": positions, "cos_sin": cos_sin, "t": T, "p": as_ref(cos_sin).shape[0], "gamma": gamma, "eps": eps,
            "v": v, "key_cache": rc, "value_cache": value_cache, "slot_mapping": slot_mapping, "s": S}
    if fp8:
        if k_scale is None or v_scale is None:
            raise ValueError("qknorm_rope_kvcache: fp8 caches need k_scale and v_scale")
        args.update(k_scale=float(k_scale), v_scale=float(v_scale))
    return launch(name, args, stream=stream, check=check)


def reshape_cache(k, v, key_cache, value_cache, slot_mapping, *, layout="flash", k_scale=None, v_scale=None, k_scales=None, v_scales=None,
                  stream=None, check=True):
    """KV-cache write of k, v [T, H, D] bf16 at ``slot_mapping[t]`` (negative = padding token).
    layout "flash":   caches [S, H, D] (bf16, or fp8 e4m3 with static ``k_scale`` / ``v_scale``)
    layout "blocked": key_cache [NB, H, D/x, BS, x], value_cache [NB, H, D, BS] (bf16: x = 8; fp8: x = 16 with per-token dynamic scales
                      ``k_scales`` / ``v_scales`` f32 [H, NB BS] written by the kernel); slot_mapping int32 or int64 (fp8)."""
    rk, T, H, D = _thd(k, "k")
    rc = as_ref(key_cache)
    fp8 = rc.itemsize == 1
    sm = as_ref(slot_mapping)
    if layout == "flash":
        S = rc.shape[0]
        name = f"reshape_cache_flash_{'fp8' if fp8 else 'bf16'}_h{H}_d{D}"
        args = {"k": rk, "v": v, "key_cache": rc, "value_cache": value_cache, "slot_mapping": sm, "t": T, "s": S}
        if fp8:
            args.update(k_scale=float(k_scale), v_scale=float(v_scale))
    elif layout == "blocked":
        NB, BS = rc.shape[0], rc.shape[3]
        name = f"reshape_cache_blocked_{'fp8' if fp8 else 'bf16'}_h{H}_d{D}_b{BS}{'_i64' if sm.itemsize == 8 else ''}"
        args = {"k": rk, "v": v, "key_cache": rc, "value_cache": value_cache, "slot_mapping": sm, "t": T, "s": NB * BS}
        if fp8:
            args.update(k_scales=k_scales, v_scales=v_scales)
    else:
        raise ValueError("reshape_cache: layout must be 'flash' or 'blocked'")
    return launch(name, args, stream=stream, check=check)


# ----------------------------------------------------------------------------------------------------------------------------- MoE routing
def topk_softmax(gating, weights, indices, *, renorm=False, stream=None, check=True):
    """Expert routing: softmax over the E experts of each row of gating [T, E] bf16, then the top-k probabilities (weights [T, K] f32,
    renormalized to sum 1 with ``renorm``) and ids (indices [T, K] int32; ties: lowest id)."""
    rg, T, E = _mn(gating)
    K = as_ref(weights).shape[1]
    return launch(f"topk_softmax{'_renorm' if renorm else ''}_bf16_e{E}_k{K}", ptrs=[rg, weights, indices], scalars=[T], stream=stream,
                  check=check)


def moe_sorting_sizes(M, E, K, block_m):
    """Element counts of the moe_sorting outputs: (sorted_token_ids / sorted_weights, sorted_expert_ids, num_valid_ids, tile_start)."""
    maxp = M * K + E * block_m - K
    return maxp, -(-maxp // block_m), 2, E + 1


def moe_sorting(topk_ids, topk_weights, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, tile_start, *, num_experts,
                block_m, stream=None, check=True):
    """aiter-style MoE alignment: the (token, slot) pairs of topk_ids [M, K] int32 grouped by expert, each expert padded to a multiple of
    block_m. sorted_token_ids u32 ``(slot << 24) | token`` (padding ``(K << 24) | M``), sorted_weights f32, sorted_expert_ids int32 per
    block_m tile (E past the valid tiles), num_valid_ids int32 [total, M], tile_start int32 [E + 1]. Sizes: ``moe_sorting_sizes``."""
    ri = as_ref(topk_ids)
    M, K = ri.shape
    return launch(f"moe_sorting_e{num_experts}_k{K}_b{block_m}", ptrs=[ri, topk_weights, sorted_token_ids, sorted_weights,
                  sorted_expert_ids, num_valid_ids, tile_start], scalars=[M], stream=stream, check=check)


def moe_reduce(x, out, weights=None, *, stream=None, check=True):
    """MoE combine: ``out[t] = sum_k w[t, k] x[t, k]`` (``weights`` f32 [T, K]) or the plain sum (weights None); x [T, K, D] bf16, out [T, D]."""
    rx = as_ref(x)
    T, K, D = rx.shape
    if weights is None:
        return launch(f"moe_combine_bf16_k{K}_d{D}", ptrs=[rx, out], scalars=[T], stream=stream, check=check)
    return launch(f"moe_reduce_bf16_k{K}_d{D}", ptrs=[rx, weights, out], scalars=[T], stream=stream, check=check)
