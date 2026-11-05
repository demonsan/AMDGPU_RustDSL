# Python JIT entry (`python/amdgpu_jit`)

A DeepGEMM-style JIT entry for the shipped kernels: Python calls the `amdc` compiler as a subprocess, assembles the result with ROCm clang,
caches the code object on disk, loads it with the HIP runtime through `ctypes` and launches it. Pure Python standard library — no torch,
numpy or Rust FFI is required (torch tensors are accepted when torch is installed). Shape specialization is on by default.

```python
import sys; sys.path.insert(0, "python")
import amdgpu_jit as aj

aj.gemm_mxfp8_blockscale(a, b, sa, sb, out)                        # gemm_mxfp8_bs_v4, specialized for (M, N, K)
aj.fmha_fwd_mxfp8(q, k, v, q_scale, k_scale, v_scale, out)        # the shipped path selection: KV-split + combine / _t128 / _vs128
```

## Pipeline

```
op(args) ──► shapes / dtypes / alignment checks ──► get_kernel(name, specialize)
                                                      │ in-process module cache hit? ─► Kernel
                                                      │ disk cache hit? ─► load kernel.co + info.json ─► Kernel
                                                      └ miss: amdc emit --specialize … ─► kernel.s ─► clang ─► kernel.co
                                                              amdc jit-info --specialize … ─► info.json      (atomic publish)
Kernel.launch(values, grid, syms, sizes) ──► pack kernarg (info.json args) ──► validate the host contract ──► hipModuleLaunchKernel
```

| file | role |
|---|---|
| `python/amdgpu_jit/_hip.py` | `ctypes` binding of `libamdhip64.so`: `hipModuleLoadData` / `hipModuleGetFunction` / `hipModuleLaunchKernel` (kernarg as one buffer, `HIP_LAUNCH_PARAM_BUFFER_POINTER`), memory, events |
| `python/amdgpu_jit/_compile.py` | toolchain discovery, cache key, `amdc emit` + `amdc jit-info` + clang, disk / in-process caches, `Kernel` |
| `python/amdgpu_jit/_kernarg.py` | `jit-info` parsing, kernarg packing, contract evaluation (`ContractError`) |
| `python/amdgpu_jit/_tensor.py` | argument adapters: torch tensors, `__dlpack__` objects, `DeviceBuffer` (hipMalloc), `RawPtr` (an integer pointer + metadata); stream choice |
| `python/amdgpu_jit/ops.py` | `gemm_mxfp8_blockscale`, `fmha_fwd_mxfp8`, `select_path` |
| `python/tests/test_unit.py` | host-only unit tests (no GPU) |
| `python/tests/node_check.py` | GPU-node checks (bit identity vs the C++ harnesses, cache, launch overhead, specialized vs generic) |

## Configuration (environment)

| variable | default | meaning |
|---|---|---|
| `AMDGPU_JIT_AMDC` | `<repo>/target/{release,debug}/amdc`, `$AMD_CARGO_TARGET/{release,debug}/amdc` | the compiler binary |
| `AMDGPU_JIT_CLANG` | `$ASM_ROC/llvm/bin/clang`, `$ROCM_PATH/llvm/bin/clang`, ROCm installs | the assembler |
| `AMDGPU_JIT_ARCH` | `gfx950` | `-mcpu` and cache key |
| `AMDGPU_JIT_BACKEND` | `own` | `amdc --backend` (`own` or `llvm-tail`) |
| `AMDGPU_JIT_CACHE_DIR` | `~/.cache/amdgpu_jit` | disk cache root |
| `AMDGPU_JIT_CACHE` | `1` | `0` disables the disk cache |
| `AMDGPU_JIT_HIP_LIB` | `$ROCM_PATH/lib/libamdhip64.so`, ROCm installs | the HIP runtime |

## Cache

The key is a SHA-256 over: cache format version, kernel name, the specialization (`name=value`, sorted), backend, arch, the content hash
of the `amdc` binary, and `clang --version`. One directory per key holds `kernel.co`, `info.json`, `kernel.s` and `key.json`. A miss compiles
into a temporary directory under the cache root and renames it into place (atomic: concurrent processes compiling the same key are safe; the
loser discards its copy). Inside a process, loaded kernels are kept by key (`aj.stats` counts memory hits, disk hits, compiles and compile
time). Rebuilding `amdc` changes its hash and therefore every key.

## Specialization

`amdc emit <k> --specialize name=value,…` binds kernarg scalars (names from the kernel's argument list) to constants: the ABI does not change
(the argument is still passed and must equal the constant — `pack` enforces it), the kernel is re-verified with the bound values
(`gpu_verify --strict`). Defaults:

| op | specialized arguments |
|---|---|
| `gemm_mxfp8_blockscale` | `m`, `n`, `k` |
| `fmha_fwd_mxfp8` (`_vs128` / `_t128`) | `s_seq_len`, `s_kv_seq_len`, `s_q_head_num`, `s_gqa` |
| `fmha_fwd_mxfp8` (KV-split) | the above + `kvs_splits`; the combine kernel: `rows_total`, `splits` |

`specialize=False` uses the generic kernels (one compile per kernel for all shapes).

## `amdc jit-info` (schema 1)

One JSON object: `kernel`, `symbol`, `arch`, `compiler` / `compiler_hash`, `backend`, `specialize`, `block`, `lds_bytes`, `vgpr`, `agpr`,
`sgpr`, `scratch_bytes`, `kernarg_size`, `kernarg_align`, `args` (sorted by offset, every byte covered; `kind` ∈ `ptr`, `i32`, `u32`, `f32`,
`pad`; `specialized` = the bound value or null) and `contract`:

- `grid[]` `{axis, ext}` — launch grid equalities; `ka_is[]` `{offset, ext}` — kernarg equalities; `ka_range[]` `{offset, lo, hi|null}`;
  `ka_set[]` `{offset, values}`; `tensors[]` `{arg, alloc_bytes, logical_bytes|null}` — minimum addressable bytes per pointer;
  `syms[]` `{name, lo, hi|null}` — free symbols the host supplies (e.g. `MT`, `NT`, `KT`, `HK`, `B`).
- `ext` expression trees: `{"c": int}`, `{"ka": offset}` (the u32 kernarg scalar), `{"grid": axis}`, `{"sym": name}`,
  `{"add" | "sub" | "mul": [E, E]}`, `{"ceil_div" | "floor_div": [E, k]}` (k a positive constant).

`validate` evaluates every item for the concrete launch; a launch that passes is inside what the GPU-semantics verifier proved the kernel
for (docs/gpu-semantics-verifier.md). `check=False` skips it (the packing checks stay).

## The ops

**`gemm_mxfp8_blockscale(a, b, sa, sb, out, *, specialize=True, stream=None, check=True)`** — `a [M, K]` fp8, `b [N, K]` fp8 preshuffled
(aiter `shuffle_weight(layout=(16, 16))`), `sa [K/128, M]` and `sb [N/128, K/128]` E8M0 bytes, `out [M, N]` bf16; M, N multiples of 256,
K a multiple of 128 up to 8192. Grid `(N/256, M/256)`.

**`fmha_fwd_mxfp8(q, k, v, q_scale, k_scale, v_scale, out, *, path="auto", splits=None, specialize=True, stream=None, check=True)`** —
`q [B, S, H, 128]`, `k`, `v [B, SK, HK, 128]` fp8 (H = gqa · HK, gqa a power of two ≤ 16), `q_scale [B, S, H, 4]`, `k_scale [B, SK, HK, 4]`
E8M0 bytes, `v_scale` an f32 in a buffer of at least 4096 bytes, `out [B, S, H, 128]` bf16. The descale tensors need the zeroed slack tail the
contract states (`512·H·4` bytes behind `q_scale`, `640·H·4` behind `k_scale`): allocate the larger buffer and pass a view — the addressable
size (torch storage, `DeviceBuffer(nbytes=…)`) is what the contract check uses. `path="auto"` is the shipped dispatch of
`tests/gpu/ref_host.cpp` (`select_path`): the best KV split s ∈ {2, 4} passing the guard and the cost model, else `_t128` when
`2·ceil(S/256)·H·B ≤ 256`, else `_vs128`; the KV-split partials live in a persistent per-stream workspace. Returns `(path, splits)`.

Arguments: torch tensors (any 1-byte dtype for fp8 / E8M0, any 2-byte dtype for bf16), objects with `__dlpack__` (GPU device), `DeviceBuffer`,
or `RawPtr(ptr, shape, itemsize)`. All must be contiguous and 16-byte aligned (scales: 4). Stream: the explicit `stream` (an integer handle or
an object with `cuda_stream`), else torch's current stream when torch tensors are passed, else the null stream.

## Checks on the GPU node

```bash
python3 -m unittest discover -s python/tests -p 'test_unit.py'                 # host only
HIP_VISIBLE_DEVICES=5 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/node_check.py
```

`node_check.py` replays the inputs the C++ harnesses generate (`INPUT_DUMP` in `tests/gpu/gemm_ref_host.cpp`, `REF_INPUT_DUMP` in
`tests/gpu/ref_host.cpp`) and requires the JIT outputs to be bit-identical to the harness outputs (`OURS_DUMP` / `REF_DUMP2`) for the
specialized and the generic kernels, the FMHA path to match the harness's dispatch, the cache to go miss → disk hit → memory hit, and reports
the first-call latencies, the host time per launch and specialized-vs-generic performance.
