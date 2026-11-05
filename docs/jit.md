# JIT entry: shape specialization and metadata (`amdc emit --specialize`, `amdc jit-info`)

The Python JIT front end (DeepGEMM style: compile on first use, cache by key, launch) calls the `amdc` CLI as a subprocess — there is
no FFI. This document is the CLI / JSON contract between the two. Compiler side: `crates/amdgpu-dsl/src/specialize.rs`
(binding, contract closure, folding), `crates/amdc/src/jit.rs` (JSON), `crates/amdc/src/main.rs` (flags); tests `crates/amdc/tests/jit.rs`.

## 1. `--specialize name=value[,name=value...]`

Accepted by `amdc emit`, `amdc jit-info` and `amdc gpu-verify` (repeatable; `--specialize=a=1` also works).

* Keys are the kernel's **kernarg argument names** (`Function.args[].name`, listed by `amdc jit-info <k>`): e.g. GEMM `m`, `n`, `k`;
  FMHA `s_seq_len`, `s_kv_seq_len`, `s_q_head_num`, `s_gqa`. Scalars only (`i32`/`u32` integers, `f32` decimals).
* **ABI and symbol are unchanged**: the host still passes every argument (with the same values); the kernel just treats the bound ones as
  constants. The cache key (kernel, specialization, `compiler_hash`) separates the variants.
* **Closure over the contract**: the bindings are propagated through the host contract's relations. A relation `ka(x) == expr` whose right side
  becomes constant fixes `x` too (a *derived* binding, e.g. GEMM `stride_a = KT * 128` once `k` is bound); a relation `ka == c * SYM` with a
  bound left side fixes the symbol (`m = 4096` gives `MT = 16`) — this is where divisibility is checked. Derived kernargs are folded as well.
* **Rejected with exit code 2** (one-line message on stderr): unknown name, pointer argument, duplicate name, non-numeric value, a value outside
  a declared `ka_range` or not in a `ka_set`, a violated relation, a value that is not a multiple the contract requires, an implied symbol
  outside its `sym_range`.
* **What changes in the kernel**: the kernarg loads become constants; the contract gets the exact facts (`ka_range(off, v, v)`, symbol ranges
  `[v, v]`, constant grid / extents); integer arithmetic of constants is folded and an `if` on a compare of constants keeps only the branch that
  runs (dead tails, dead clamps and the unused loop copy disappear); the lowering then folds the rest (immediates, scalar compares).
* **The specialized kernel must verify**: `emit` / `jit-info` run `gpu_verify` on it and print its strict certificate on stderr
  (`specialized <k>: certificate: …`); any unproven property → exit 1, nothing is emitted.
* Without `--specialize` nothing changes (the asm md5 of every kernel is identical).

## 2. `amdc jit-info <kernel> [--specialize …] [--backend own|llvm-tail|auto]`

Prints one JSON object on stdout (deterministic; runs the full compile, about the cost of `emit`). Schema 1:

| key | meaning |
|---|---|
| `schema` | 1 |
| `kernel`, `symbol` | kernel name = ELF symbol (unchanged by specialization) |
| `arch` | `gfx950` |
| `compiler` | `{"version": Cargo version, "git": git describe / commit at build time}` |
| `compiler_hash` | 16 hex digits, FNV-1a over version, git and backend: part of the cache key |
| `backend` | the backend actually used (`own` / `llvm-tail`; `auto` resolves per kernel) |
| `specialize` | the bindings given `{name: value}`; `derived` the kernargs they fix through the contract |
| `block` | `[threads, 1, 1]` |
| `lds_bytes`, `vgpr`, `agpr`, `sgpr`, `scratch_bytes` | read back from the emitted kernel descriptor (`vgpr` = architectural VGPRs = accum offset) |
| `kernarg_size`, `kernarg_align` | segment size; alignment 8 |
| `args` | `[{name, offset, size, kind: ptr/i32/u32/f32/pad, specializable, specialized: null or value}]`, sorted by offset, gaps covered by `pad` entries named `_pad<offset>` (every byte covered once) |
| `contract.grid` | `[{axis, ext: E}]`: launch grid (workgroups) per axis |
| `contract.ka_is` | `[{offset, ext: E}]`: the scalar at `offset` must equal E |
| `contract.ka_range` | `[{offset, lo, hi}]` (`hi` = null: unbounded) |
| `contract.ka_set` | `[{offset, values}]` |
| `contract.tensors` | `[{arg, alloc_bytes: E, logical_bytes: E or null}]`: required allocation of each pointer argument |
| `contract.syms` | `[{name, lo, hi}]`: free symbols (e.g. FMHA batch `B`, GEMM tile counts `MT/NT/KT`) the host supplies when evaluating E |

Expression tree E — exactly the operators of `Ext`:
`{"c": int}` | `{"ka": offset}` (u32 kernarg scalar) | `{"grid": axis}` | `{"sym": name}` | `{"add": [E, E]}` | `{"sub": [E, E]}` |
`{"mul": [E, E]}` | `{"ceil_div": [E, k]}` | `{"floor_div": [E, k]}` (k a positive integer constant).

The Python side evaluates these to validate a launch (grid, kernarg relations, tensor sizes) before calling the kernel.

## 3. Measurements (GPU 6, same run)

See the report in [`current-status.md`](current-status.md) §4 (instruction counts and A/B of the specialized vs generic kernels).
