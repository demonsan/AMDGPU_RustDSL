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
    """``y = rmsnorm(x) * gamma`` quantized to MXFP4: q [M, N/2] packed e2m1 (low nibble first), scale [M, N/32] E8M0 bytes."""
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
