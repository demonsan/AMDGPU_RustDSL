"""amdgpu_jit: a DeepGEMM-style JIT entry for the AMDGPU Rust DSL kernels (pure Python stdlib + ctypes on libamdhip64; docs/python-jit.md).

    import amdgpu_jit as aj
    aj.gemm_mxfp8_blockscale(a, b, sa, sb, out)                       # compiles (cached) gemm_mxfp8_bs_v4 specialized for M/N/K, launches
    aj.fmha_fwd_mxfp8(q, k, v, q_scale, k_scale, v_scale, out)       # shipped path selection (KV-split + combine / _t128 / _vs128)

Lower level: ``get_kernel(name, specialize)`` returns a loaded ``Kernel`` (``info`` = the ``amdc jit-info`` JSON, ``launch(values, grid, ...)``).
"""
from ._compile import CompileError, Kernel, cache_dir, clear_memory_cache, get_kernel, stats
from ._kernarg import ContractError, KernelInfo
from ._tensor import DeviceBuffer, RawPtr, TensorRef, as_ref
from .ops import decode16_plan, fmha_fwd_mxfp8, gemm_mxfp8_blockscale, paged_decode_bf16, select_path
from ._generic import launch
from .ops_more import (a16w4_permute, argmax, attention_bwd_bf16, causal_conv1d_fwd, causal_conv1d_update, gemm_w4, mla_decode, sample,
                       topk_per_row_decode, topk_per_row_prefill)
from .ops_compute import dispatch, fused_moe, gemm, moe_w1_interleave, prefill_bf16, shuffle_mx_scales, shuffle_weight
from .ops_mem import (act_mul, add_rmsnorm, add_rmsnorm_quant_fp8, quant_per_group128, quant_per_token, quant_static_fp8, rmsnorm,
                      rmsnorm_quant_fp8, rmsnorm_quant_mxfp4, silu_mul_quant_fp8, rope, qknorm_rope, qknorm_rope_kvcache, reshape_cache,
                      topk_softmax, moe_sorting, moe_sorting_sizes, moe_reduce)
from . import _hip as hip

__all__ = ["launch", "mla_decode", "argmax", "sample", "topk_per_row_decode", "topk_per_row_prefill", "causal_conv1d_fwd",
           "causal_conv1d_update", "gemm_w4", "a16w4_permute", "attention_bwd_bf16", "dispatch", "gemm", "prefill_bf16", "fused_moe", "shuffle_weight", "shuffle_mx_scales", "moe_w1_interleave", "rmsnorm", "add_rmsnorm", "add_rmsnorm_quant_fp8", "rmsnorm_quant_fp8", "rmsnorm_quant_mxfp4", "quant_per_token",
           "quant_per_group128", "quant_static_fp8", "act_mul", "silu_mul_quant_fp8", "rope", "qknorm_rope", "qknorm_rope_kvcache",
           "reshape_cache", "topk_softmax", "moe_sorting", "moe_sorting_sizes", "moe_reduce", "gemm_mxfp8_blockscale", "fmha_fwd_mxfp8", "select_path", "decode16_plan", "paged_decode_bf16", "get_kernel", "Kernel", "KernelInfo", "ContractError", "CompileError",
           "DeviceBuffer", "RawPtr", "TensorRef", "as_ref", "cache_dir", "clear_memory_cache", "stats", "hip"]
__version__ = "0.1.0"
