"""amdgpu_jit: a DeepGEMM-style JIT entry for the AMDGPU Rust DSL kernels (pure Python stdlib + ctypes on libamdhip64; docs/python-jit.md).

    import amdgpu_jit as aj
    aj.gemm_mxfp8_blockscale(a, b, sa, sb, out)                       # compiles (cached) gemm_mxfp8_bs_v4 specialized for M/N/K, launches
    aj.fmha_fwd_mxfp8(q, k, v, q_scale, k_scale, v_scale, out)       # shipped path selection (KV-split + combine / _t128 / _vs128)

Lower level: ``get_kernel(name, specialize)`` returns a loaded ``Kernel`` (``info`` = the ``amdc jit-info`` JSON, ``launch(values, grid, ...)``).
"""
from ._compile import CompileError, Kernel, cache_dir, clear_memory_cache, get_kernel, stats
from ._kernarg import ContractError, KernelInfo
from ._tensor import DeviceBuffer, RawPtr, TensorRef, as_ref
from .ops import fmha_fwd_mxfp8, gemm_mxfp8_blockscale, select_path
from . import _hip as hip

__all__ = ["gemm_mxfp8_blockscale", "fmha_fwd_mxfp8", "select_path", "get_kernel", "Kernel", "KernelInfo", "ContractError", "CompileError",
           "DeviceBuffer", "RawPtr", "TensorRef", "as_ref", "cache_dir", "clear_memory_cache", "stats", "hip"]
__version__ = "0.1.0"
