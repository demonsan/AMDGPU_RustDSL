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
`tests/gpu/ref_host.cpp` (`select_path`, fitted on realistic data): a KV split with s = 4 when ntiles ≥ 16 and N256 ≤ 64, s = 2 when ntiles ≥ 32 and
N256 ≤ 128 (N256 = `ceil(S/256)·H·B`, ntiles = `ceil(SK/128)`), else `_t128` when N256 ≤ 64, else `_vs128`; the KV-split partials live in a persistent per-stream workspace. Returns `(path, splits)`.

**`paged_decode_bf16(q, k_cache, v_cache, block_table, context_lens, out, *, scale=None, max_context=None, stream=None, check=True)`** —
paged-KV decode (one new token per sequence), bf16, head dim 128: `q [B, H, 128]`, `k_cache` / `v_cache [num_pages, P, HK, 128]` (NHD pages,
P a power of two ≥ 16), `block_table [B, max_pages]` int32, `context_lens [B]` int32, `out [B, H, 128]`; G = H / HK ≤ 16; `scale` defaults
to 1/√128. The kernel and split count come from the single dispatch rule (`decode16_plan`, also `amdgpu_jit.decode16_plan(b, hk, ctx)`):
the fused form when B·HK ≥ 256, else the last-arriver `_c` form with S splits (partials and the zero-initialised ticket counters live in a
persistent workspace; the kernel resets the counters). `max_context` only steers that choice (default `max_pages·P`). Returns
`(kernel, splits)`. `python3 python/tests/decode_check.py` is the torch-free example and node check (stdlib only: random bf16 data, shuffled
block tables, pa_host's CPU reference): both forms bad = 0.

Arguments: torch tensors (any 1-byte dtype for fp8 / E8M0, any 2-byte dtype for bf16), objects with `__dlpack__` (GPU device), `DeviceBuffer`,
or `RawPtr(ptr, shape, itemsize)`. All must be contiguous and 16-byte aligned (scales: 4). Stream: the explicit `stream` (an integer handle or
an object with `cuda_stream`), else torch's current stream when torch tensors are passed, else the null stream.

## Generic launch and the memory-bound ops

**`launch(kernel, args=None, *, ptrs=None, scalars=None, grid=None, syms=None, spec=None, stream=None, check=True)`** calls any registered
kernel from its `amdc jit-info`: arguments by name (`args`) or positionally (`ptrs` = tensors for the pointer arguments in offset order,
`scalars` = numbers for the scalar arguments in offset order). The contract's free symbols come from the scalar relations (`ka_is`, e.g.
`m = 4·MT`), the grid from the contract's grid expressions; the launch is checked against the verified contract (allocation sizes
included), so a row count that is not a multiple of a CTA's rows is reported as the violated relation. The ops below are thin wrappers
(`python/amdgpu_jit/ops_mem.py`) that pick the kernel for the shape:

| op | kernel | tensors |
|---|---|---|
| `rmsnorm(x, gamma, out, eps)` | `rmsnorm_bf16_n<N>` | x, out [M, N] bf16; gamma [N] |
| `add_rmsnorm(x, res, gamma, out, res_out, eps)` | `add_rmsnorm_bf16_n<N>` | res_out = x + res |
| `add_rmsnorm_quant_fp8(x, res, gamma, out, res_out, scale, eps)` | `add_rmsnorm_fp8_n<N>` | out [M, N] fp8, scale [M] f32 |
| `rmsnorm_quant_fp8(x, gamma, q, scale, eps)` | `rmsnorm_quant_fp8_n<N>` | per-token fp8 |
| `rmsnorm_quant_mxfp4(x, gamma, q, scale, eps)` | `rmsnorm_quant_mxfp4_bf16_k<N>` | q [M, N/2] e2m1, scale E8M0 in the shuffled a4w4 layout; M % 16 = 0 |
| `quant_per_token(x, q, scale, dtype="fp8"\|"int8")` | `quant_token_bf16_<dtype>_n<N>` | scale [M] f32 = max / (448 \| 127) |
| `quant_per_group128(x, q, scale)` | `quant_group128_bf16_fp8_n<N>` | scale [M, N/128] f32 |
| `quant_static_fp8(x, out, scale)` | `quant_bf16_fp8_static` | per-tensor scale (float) |
| `act_mul(x, out, act="silu"\|"gelu"\|"gelu_tanh"\|"swiglu_oai")` | `<act>_mul_bf16_d<D>` | x [M, 2D] → out [M, D] |
| `silu_mul_quant_fp8(x, q, scale)` | `silu_mul_quant_fp8_d<D>` | q [M, D] fp8, scale [M] f32 |
| `rope(x, out, positions, cos_sin, style="neox"\|"gptj")` | `rope_<style>_bf16_h<H>_d<D>` | x [T, H, D]; cos_sin [P, D] bf16 (cos \| sin) |
| `qknorm_rope(x, out, positions, cos_sin, gamma, eps, style)` | `qknorm_rope_<style>_bf16_h<H>_d<D>` | per-head RMSNorm, then RoPE |
| `qknorm_rope_kvcache(k, out, positions, cos_sin, gamma, v, key_cache, value_cache, slot_mapping, eps, style, k_scale, v_scale)` | `qknorm_rope_kvcache_<style>_<bf16\|fp8>_h<H>_d<D>[_i64]` | flash caches [S, H, D]; fp8 with static scales; i32 / i64 slots |
| `reshape_cache(k, v, key_cache, value_cache, slot_mapping, layout="flash"\|"blocked", …)` | `reshape_cache_<layout>_<bf16\|fp8>_h<H>_d<D>[_b<BS>][_i64]` | blocked: K [NB, H, D/x, BS, x], V [NB, H, D, BS]; fp8 blocked with per-token scales |
| `topk_softmax(gating, weights, indices, renorm=False)` | `topk_softmax[_renorm]_bf16_e<E>_k<K>` | gating [T, E] bf16 → weights f32, indices i32 [T, K] |
| `moe_sorting(topk_ids, topk_weights, sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, tile_start, num_experts, block_m)` | `moe_sorting_e<E>_k<K>_b<BM>` | sizes: `moe_sorting_sizes(M, E, K, BM)` |
| `moe_reduce(x, out, weights=None)` | `moe_reduce_bf16_k<K>_d<D>` / `moe_combine_bf16_k<K>_d<D>` | x [T, K, D] → out [T, D] |
| `gemm(a, b, out, kind="bf16"\|"fp8_rowcol"\|"fp8_blockscale"\|"mxfp4", a_scale, b_scale, splits="auto", tile="auto")` | `gemm_a16w16_bf16[_128]`, `gemm_a8w8_rowcol_fp8[_128]`, `gemm_a8w8_blockscale[_splitk]` + `gemm_splitk_reduce_s<S>`, `gemm_a4w4_mxfp4` | B preshuffled (`shuffle_weight`); blockscale split-K from `blockscale_split_k`; mxfp4 scales via `shuffle_mx_scales` |
| `prefill_bf16(q, k, v, out, causal, lse=None, scale=None)` | `bf16_prefill_kernel` choice: `fmha_fwd_hd128_bf16_v3[_causal][_ks]` / `fmha_fwd_hd128_bf16_causal` (+ `attn_combine_bf16_hd128_q16`) | q, out [B, SQ, H, 128], k, v [B, SK, HK, 128]; lse f32 [B, H, SQ] |
| `fused_moe(x, w1, w2, topk_ids, topk_weights, out, dtype="bf16"\|"fp8", act, x_scale, w1_scale, w2_scale, num_experts, inter)` | `moe_sorting_*`, `gemm_moe1_<dt>_<act>[_bm64]`, `quant_token_bf16_fp8_n<I>` (fp8), `gemm_moe2_<dt>[_bm64]`, `moe_sum_bf16_k<K>` | W1 via `moe_w1_interleave` + `shuffle_weight`; block_m from `moe_block_m`; E ≥ 8, D % 1024 = 0 |

| `mla_decode(q, kv_cache, block_table, context_lens, out, scale)` | `mla_decode_bf16_[hp_]p<P>` + `attn_combine_bf16_hd512` | q [B, H, 576], kv [pages, P, 576], out [B, H, 512]; form / splits from `mla_config` |
| `argmax(logits, out)` / `sample(logits, u, out, temperature, top_k=0, top_p=1.0)` | `argmax_bf16_v<V>`, `sample[_topkp]_bf16_v<V>` | logits bf16 [B, V]; u f32 [B]; out i32 [B] |
| `topk_per_row_decode(logits, seq_lens, indices, next_n)` / `topk_per_row_prefill(logits, row_starts, row_ends, indices)` | `topk_per_row_{decode,prefill}_f32_k<K>` | logits f32 [R, stride] → indices i32 [R, K] |
| `causal_conv1d_fwd(x, weight, bias, cu_seqlens, cache_indices, has_initial, conv_states, out, max_seqlen, silu)` | `causal_conv1d_fwd_bf16_w<W>[_silu]` | varlen prefill, x [T, dim]; states [slots, dim, W-1] |
| `causal_conv1d_update(x, weight, bias, state, out, state_indices=None, silu)` | `causal_conv1d_update_bf16_w<W>_s<SL>[_idx][_silu]` | decode, x [B, dim]; state [slots, dim, SL] |
| `gemm_w4(a, b, out, b_scale, a_scale, kind="a16w4"\|"a8w4")` | `gemm_a16w4_mxfp4`, `gemm_a8w4_mxfp_128` | a16w4: B through `a16w4_permute` + `shuffle_weight` |
| `attention_bwd_bf16(q, k, v, o, do, lse, dq, dk, dv, causal, scale)` | `fmha_bwd_dot` + `fmha_bwd_dq` / `fmha_bwd_dkdv` (causal: `*_causal` from `bwd_causal_kernels`) | dk, dv per query head [B, SK, H, 128] |

Every other registered kernel is reachable through `launch(name, …)` directly (arguments by the names `amdc jit-info <name>` lists).

The dispatch rules are asked from amdc: `amdgpu_jit.dispatch("prefill_bf16", B, H, SQ, SK, causal)`, `dispatch("gemm_blockscale", M, N, K)`,
`dispatch("moe", M, topk, E)`, `dispatch("decode16", B, HK, ctx)`, `dispatch("mla", B, H)`, `dispatch("bwd_causal")` (= `amdc dispatch …`, the functions of `crates/amdgpu-kernels/src/dispatch.rs`).
The fp8 GEMM checks allow for the f8f6f4 MFMA's internal precision: a 128-term block is summed to about 2^-10 of its term magnitude
(visible under cancellation — also with unit scales and a single K block, so it is the instruction, not the kernel).

`python3 python/tests/ops_check.py [name …]` (GPU node, stdlib only; GEMM / prefill / MoE checks sample rows and tokens) runs every op on random data against a Python reference that follows the
C++ host reference of its family (formulas and tolerances of `tests/gpu/{rows,quant,eltwise,rmsnorm_quant,mxnorm,rope,qknorm_rope_kv,cache,cache_blocked,topk,moe_sorting,moe_reduce}_host.cpp`).

## Launch overhead

Every op keeps a *prepared launch* per call signature (the tensors by identity, data pointer and shape, the scalars, the grid): the kernarg
segment is packed and checked against the contract once, later calls with the same signature are one `hipModuleLaunchKernel`
(`_hip.Prepared`). A reallocated workspace invalidates them. Host time per call (`python3 python/tests/launch_overhead.py`, MI355X,
contract check on / off):

| op | before | after |
|---|--:|--:|
| `paged_decode_bf16` B64 (fused) | 26.6 / 19.2 µs | 5.8 / 5.5 µs |
| `paged_decode_bf16` B4 (last-arriver S4) | 23.6 / 20.2 µs | 5.5 / 5.5 µs |
| `rmsnorm` 64x4096 (generic `launch`) | 29.2 / 22.2 µs | 6.5 / 6.2 µs |
| `gemm` bf16 256^3 | 57.4 / 40.2 µs | 9.3 / 9.3 µs |

## Checks on the GPU node

`HIP_VISIBLE_DEVICES=6 bash python/tests/run_all.sh [--full] [filters]` runs all of them (unit tests, `ops_check.py`, `decode_check.py`;
`--full` adds `node_check.py`) and prints a table of every op check with an OVERALL line.


```bash
python3 -m unittest discover -s python/tests -p 'test_unit.py'                 # host only
HIP_VISIBLE_DEVICES=5 AMDGPU_JIT_AMDC=/path/to/amdc python3 python/tests/node_check.py
```

`node_check.py` replays the inputs the C++ harnesses generate (`INPUT_DUMP` in `tests/gpu/gemm_ref_host.cpp`, `REF_INPUT_DUMP` in
`tests/gpu/ref_host.cpp`) and requires the JIT outputs to be bit-identical to the harness outputs (`OURS_DUMP` / `REF_DUMP2`) for the
specialized and the generic kernels, the FMHA path to match the harness's dispatch, the cache to go miss → disk hit → memory hit, and reports
the first-call latencies, the host time per launch and specialized-vs-generic performance.

## Measured (MI355X GPU 5, debug `amdc`, `python3 python/tests/node_check.py`, 22/22 PASS)

| check | result |
|---|---|
| bit identity vs the C++ harnesses, specialized and generic | GEMM 1024×1024×2048 and 1280×6144×7168; FMHA B4H8S2048 (`_vs128`), B2H8S2048 (`_t128`), B1H8S2048 and B1H2S200SK1337 gqa2 (KV-split 4 + combine), B1H4S77SK1 (`_t128`): all bit-identical, the same path as the harness dispatch |
| cache | cold first call 1.2–1.4 s (amdc emit + jit-info in parallel + clang); first call in a new process 13.5 ms (disk hit, after the 180 ms HIP runtime initialisation); in-process hit 26–45 µs |
| host time per op call | GEMM: 20 µs with the contract check (memoised), 19 µs without; `Kernel.launch_raw` (prepacked kernarg) 5.6 µs |
| launch path adds no GPU time | GEMM 4096×4096×8192, same kernel and inputs: JIT 3207 vs `gemm_ref_host` 3210 TFLOP/s |
| specialized vs generic | GEMM 4096×4096×8192 +0.2 %; FMHA B4H8S2048 −1.9 %, B1H8S8192 −1.1 %, B1H8S2048 KV-split −4.1 % (specialization does not help these kernels; `specialize=False` keeps one compile per kernel) |

Note on input data: the GEMM runs about 13 % slower on uniformly random fp8 bytes (`os.urandom`) than on the harnesses' LCG data
(3210 → 2790 TFLOP/s, same kernel, same launch path) - a power / clock effect of the data, not of the launch.