# L1 graph layer: op graphs, fusion rules and launch plans

Design note for the L1 content of [`refactor-4layer-plan.md`](refactor-4layer-plan.md) in the `amdgpu-graph` crate. Today the crate holds:
- the stage-level `opgraph` (the op graph *inside* a fused kernel family, input of the skeleton generator);
- the dispatch rules (`dispatch.rs`).

This note adds the **model-level** layer above them:
- a small op graph of framework ops (rmsnorm, quant, rope, MoE, attention, ...);
- **fusion rules** that map an op subgraph to an existing fused kernel and its dispatch rule;
- a **launch plan** that `amdc graph plan <json>` prints and the Python ops can execute.

It introduces no new kernels. Every rule names a kernel that the registry builds today.

## 1. Why

The fused kernels exist. Choosing them is spread over three places:
- the Python wrappers (`python/amdgpu_jit/ops_mem.py`, `ops_compute.py`) build the names by hand (`f"rmsnorm_quant_fp8_n{N}"`, ...);
- the multi-kernel flows (`fused_moe`, `prefill_bf16` + combine) hard-code their launch sequence and workspaces;
- `reachable::python_op_names` repeats the same names to keep them under the md5 manifest.

A fusion decision ("rmsnorm followed by per-token fp8 quant becomes `rmsnorm_quant_fp8_n{N}`") is an L1 fact. It should live in one place and be visible as a rule, testable without a GPU, and shared by every host.

## 2. The op graph (input)

The input is JSON: plain data, not an IR. It uses the same hand-written JSON reader as `opgraph::json`, so the crate gains no dependency.

```json
{ "tensors": { "x":   {"dtype": "bf16", "shape": ["M", 4096]},
               "g":   {"dtype": "bf16", "shape": [4096]},
               "y":   {"dtype": "bf16", "shape": ["M", 4096]},
               "q":   {"dtype": "fp8",  "shape": ["M", 4096]},
               "s":   {"dtype": "f32",  "shape": ["M"]} },
  "dims":    { "M": 256 },
  "ops": [ {"id": "n1", "op": "rmsnorm",     "in": ["x", "g"], "out": ["y"], "attrs": {"eps": 1e-6}},
           {"id": "q1", "op": "quant_token", "in": ["y"],      "out": ["q", "s"], "attrs": {"to": "fp8"}} ],
  "outputs": ["q", "s"],
  "target": {"cus": 256} }
```

- **Ops.** The op vocabulary is closed (§3). Each op declares its inputs, outputs and attributes; arity and dtypes are checked per op kind.
- **Dims.** Shapes may use symbols. `dims` binds them for one plan; dispatch rules need concrete sizes (M, B, SQ, ...). Python plans once per shape key and caches the result, as `ops_compute.dispatch` does today.
- **Outputs.** `outputs` lists the tensors visible after the graph. A tensor that is neither an output nor read by a later op is a candidate for staying on chip.

## 3. Op vocabulary (first set)

These are the ops of today's Python surface, named by what they compute rather than by kernel.

| Op | Inputs → outputs | Attributes |
|---|---|---|
| `add` | a, b → c | — |
| `rmsnorm` | x, gamma → y | `eps` |
| `quant_token` | x → q, scale | `to`: fp8 / int8 |
| `quant_group` | x → q, scale | `to`: fp8, `group`: 128 |
| `quant_mx` | x → q, scale | `to`: mxfp4, `layout`: a4w4_shuffled |
| `quant_static` | x, scale → q | `to`: fp8 |
| `act_mul` | x → y | `act`: silu / gelu_tanh (gate and up halves of x) |
| `rope` | x, positions, cos_sin → y | `style`: neox / gptj |
| `qknorm` | x, gamma → y | `eps` (per head) |
| `kvcache_write` | k, v, slot_mapping, key_cache, value_cache → (caches) | `layout`: flash / blocked, `block`, `fp8` |
| `topk_softmax` | gating → weights, ids | `k`, `renorm` |
| `moe_experts` | x, w1, w2, ids, weights → y | `act`, `dtype`, `experts`, `inter` (the whole two-stage MoE; see §5.3) |
| `attention` | q, k, v → o (, lse) | `causal`, `scale`, `layout`: bshd |
| `paged_attention` | q, k_cache, v_cache, block_table, context_lens → o | `page` |
| `mla_attention` | q, kv_cache, block_table, context_lens → o | `page` |
| `argmax`, `sample`, `topk_per_row`, `moe_reduce`, `causal_conv1d`, ... | as the Python ops | as the Python ops |

Every op also has a **single-op rule**, so any graph built from the vocabulary has a plan; fusion is an optimisation over that baseline.

## 4. Fusion rules

A rule is Rust data plus two functions, kept in `amdgpu-graph/src/fusion/rules.rs`:

```rust
pub struct Rule {
    pub name: &'static str,             // "rmsnorm+quant_token.fp8"
    pub pattern: Pattern,               // a small DAG of op kinds anchored at its last op
    pub exposes: &'static [&'static str],  // interior tensors the kernel still writes (e.g. add_rmsnorm's residual)
    pub accept: fn(&Match, &Graph) -> Result<(), String>,  // shape / dtype / attr constraints (the family's supported set)
    pub plan: fn(&Match, &Graph, &Target) -> Vec<Launch>,  // kernel name(s) + dispatch parameters + bindings + workspaces
}
```

**Matching.** Rules are tried **largest pattern first**: maximal munch over a topological walk of the graph. A match covers its ops when:
1. the op kinds and edges match the pattern;
2. every interior tensor is read only inside the match and is not a graph output, unless the rule `exposes` it (the kernel writes it anyway, like `res_out` of `add_rmsnorm`);
3. `accept` passes, i.e. the shapes, dtypes and attributes lie in the family's supported set (e.g. N ∈ the built row widths, head dim 128).

Ops that no multi-op rule covers fall back to their single-op rule. A rejected candidate is reported with its reason (§6), so "why did this not fuse" has an answer.

**Kernel names and dispatch.** `plan` builds the names exactly as the Python wrappers do today, using the same format strings. Where a launch parameter depends on the shape, it calls the existing rule in `dispatch.rs`:
- `moe_block_m`;
- `bf16_prefill_kernel` (key splits);
- `decode16_plan` / `decode16_kernel`;
- `mla_config`;
- `blockscale_config`.

`amdc dispatch` keeps working: it and `graph plan` call the same functions.

**Ownership of the supported sets.** `accept` mirrors the parse constraints of the family's `by_name` (`amdgpu-graph` does not depend on `amdgpu-kernels`). The kernels crate checks the mirror in a test (§7): every name a rule produces over the canonical shape grid must build.

## 5. The first rule set (today's hand-written fused kernels)

### 5.1 Single-kernel fusions

| Rule | Pattern | Kernel | Exposed interior | Constraints |
|---|---|---|---|---|
| `rmsnorm+quant_token.fp8` | rmsnorm → quant_token(fp8) | `rmsnorm_quant_fp8_n{N}` | — | bf16 x, N as built |
| `rmsnorm+quant_mx.fp4` | rmsnorm → quant_mx(mxfp4) | `rmsnorm_quant_mxfp4_bf16_k{N}` | — | M % 16 == 0, a4w4 shuffled scales |
| `add+rmsnorm` | add → rmsnorm | `add_rmsnorm_bf16_n{N}` | the sum (`res_out`) | |
| `add+rmsnorm+quant_token.fp8` | add → rmsnorm → quant_token(fp8) | `add_rmsnorm_fp8_n{N}` | the sum | |
| `act_mul.silu+quant_token.fp8` | act_mul(silu) → quant_token(fp8) | `silu_mul_quant_fp8_d{D}` | — | |
| `qknorm+rope` | qknorm → rope | `qknorm_rope_{style}_bf16_h{H}_d{D}` | — | D = 128 |
| `qknorm+rope+kvcache` | qknorm → rope → kvcache_write (+ v) | `qknorm_rope_kvcache_{style}_{bf16,fp8}_h{H}_d{D}[_i64]` | the roped k only if it is an output | fp8 cache ⇒ the `_fp8` form; i64 slot mapping ⇒ `_i64` |

The single-op rules cover the rest:
- `rmsnorm_bf16_n{N}`, `quant_token_bf16_{fp8,int8}_n{N}`, `quant_group128_bf16_fp8_n{N}`, `quant_bf16_fp8_static`;
- `{silu,gelu_tanh}_mul_bf16_d{D}`, `rope_{style}_bf16_h{H}_d{D}`, `reshape_cache_*`, `topk_softmax*`;
- sampling, `topk_per_row_*`, `causal_conv1d_*`.

### 5.2 Attention: split → combine (one op, a multi-launch plan)

`attention(causal, bshd, D = 128, bf16)` plans through `dispatch::bf16_prefill_kernel(B, H, SQ, SK, causal, cus)`:
- **S = 1:** one launch of the chosen kernel.
- **S > 1:** two launches:
  1. the `_ks` kernel, writing the partials into the workspaces `po` (B·H·SQ·S·D·4 bytes) and `plse` (B·H·SQ·S·4 bytes);
  2. `attn_combine_bf16_hd128_q16` (the `templates::split` combine).

  The plan carries the workspaces and S.

Other attention ops follow the same pattern with their existing rules:
- `fp8pt` / hd192: `fp8pt_prefill_kernel`, `hd192_prefill_kernel`;
- paged decode: `decode16_plan`. It gives either the fused form (one launch) or the last-arriver form (one launch plus a zeroed counter workspace); the separate-combine form is `attn_combine_bf16_hd128`;
- MLA: `mla_config`, with `attn_combine_bf16_hd512` when split.

This is the T4 split / merge choice of l2-templates §3.4, now made at L1.

### 5.3 MoE: sorting → stage 1 → requant → stage 2 → sum

The pattern is `topk_softmax → moe_experts`, or `moe_experts` alone when the router output is given. It plans the `fused_moe` sequence:
1. `moe_sorting_e{E}_k{K}_b{BM}`;
2. `gemm_moe1_{dt}_{act}[_bm64]`;
3. `quant_token_bf16_fp8_n{I}`, only when `dtype = fp8`;
4. `gemm_moe2_{dt}[_bm64]`;
5. `moe_sum_bf16_k{K}`.

`BM = dispatch::moe_block_m(M, K, E)`. The workspaces (sorted ids / weights / expert ids, `num_valid_ids`, tile starts, y1, y1q / y1s, t) come with their byte sizes from the same formulas as `fused_moe`.

The rule is the place where a future fusion would go. If sorting is folded into stage 1, or the sum into stage 2, only `plan` changes; hosts are unaffected.

## 6. Plan output (`amdc graph plan <json> [--cus N]`)

```json
{ "launches": [
    { "rule": "rmsnorm+quant_token.fp8", "covers": ["n1", "q1"], "kernels": ["rmsnorm_quant_fp8_n4096"], "py": "rmsnorm_quant_fp8",
      "bind": {"x": "x", "gamma": "g", "q": "q", "scale": "s"}, "params": {"eps": "1e-6"}, "workspaces": [] } ],
  "rejected": [ {"rule": "...", "at": "n1", "why": "interior tensor `y` is also read by op q2"} ],
  "errors": [] }
```

- **`bind`** maps the kernel's argument roles to graph tensors. The roles are the parameter names of the Python wrapper named in `py`, so the Python side executes a plan by calling its existing, tested wrappers.
- **Multi-launch plans** (§5.2, §5.3) use `py` = `prefill_bf16` / `fused_moe` at first, i.e. the whole flow, which already allocates its workspaces. Splitting them into per-launch steps with plan-owned workspaces is a later step, once the wrappers accept explicit workspaces.
- **`rejected`** lists the candidates that failed condition 2 or 3 of §4, with the reason. **`errors`** lists the ops no rule covers (no standalone kernel, or the shapes are outside every kernel); `amdc graph plan` then exits 1.
- **Floats.** Float attributes (`eps`, `scale`, `k_scale`) are JSON strings: the reader has integers only. They are passed through to the wrapper unchanged.
- **Output order.** The printed plan is deterministic (topological, rules in table order), so it can be checked as a golden file.

## 7. Tests and gates

1. **Names build** (kernels crate, extends `tests/dispatch_compile.rs`): for every rule, `plan` over a canonical shape grid (the `python_op_names` sizes) yields names that `kernel_by_name` builds.

   `reachable::python_op_names` then becomes *derived* from the rules instead of hand-listed. The md5 manifest keeps the same set: a manifest diff here means a rule changed a name.
2. **Golden plans** (`tests/golden/graph_plans/*.json` + the expected output): one graph per rule, plus negative cases (an interior tensor with a second consumer, an unsupported N, an fp8 cache without scales).
3. **Numerics** (Python, on GPU, existing harness): executing a fused plan and the unfused single-op plan of the same graph must agree within the families' current tolerances. This checks that each rule's pattern really computes the fused kernel's function, i.e. the rule is semantically right, not only its name.
4. **No codegen change.** The layer only chooses names, so `md5-manifest --check` must stay identical.

## 8. Placement and steps

- **Placement.** `amdgpu-graph/src/fusion/{mod.rs (graph, match), vocab.rs (op kinds + checks), rules.rs (§5), plan.rs (output)}`. The stage-level `opgraph` and `dispatch` stay as they are; `fusion` calls `dispatch`.
- **Step 1** (next commit): vocabulary, matcher, and the §5.1 rules plus their single-op fallbacks. Also the §5.2 prefill split → combine and the §5.3 MoE plans with `py` set to the whole-flow wrappers. Plus `amdc graph plan`, golden plans and the names-build test. No Python change.
- **Step 2:** Python `amdgpu_jit.graph.plan(graph)` / `run(plan, tensors)` over the existing wrappers; the numerics test of §7.3.
- **Step 3:** derive `python_op_names` from the rules. Wrappers take explicit workspaces, so multi-launch plans become per-launch.
- **Later:** rules for new fusions are where new fused kernels plug in. The rule lands with the kernel and its gate; hosts pick it up with no change.

## 9. Open points

- **Attribute equivalence.** The fused kernels fix some attributes that the unfused ops take freely, e.g. `add_rmsnorm_fp8`'s scale semantics (dynamic per token, max|y| / 448). `accept` must reject a graph whose `quant_token` asks for anything else (static scale, other max). The vocabulary needs those attributes explicit.
- **Symbolic M.** The plan is per concrete shape. If plan latency matters on the Python side, the rule output can mark the M-dependent parameters (block_m, splits) as "dispatch at launch" instead of re-planning per M.
- **Graph source.** For now graphs are hand-written JSON or built by Python helpers. Importing from a framework trace (torch.fx, vLLM model code) is out of scope for this layer.

## 10. Status

- **Step 1** (`l1-graph-step1`):
  - `amdgpu-graph::fusion` with the §5 rules and the single-op rules;
  - `amdc graph plan <json | -> [--cus N] [--no-fuse]`;
  - `tests/golden/graph_plans` (12 graphs + 5 negatives) checked by `amdgpu-graph/tests/fusion.rs`;
  - `amdgpu-kernels/tests/graph_plan_names.rs` (every planned kernel is registered).
- **Step 2:**
  - `amdgpu_jit.graph.plan(graph, cus=None, fuse=True)` / `run(plan, tensors)` / `kernels(plan)`. `run` calls the wrappers named by `py`, with the plan's parameters filtered by each wrapper's signature.
  - `python/tests/graph_check.py` (in `run_all.sh`) runs the fused and unfused (`--no-fuse`) plans of rmsnorm → quant fp8 and silu_mul → quant fp8 on the same data, plus add → rmsnorm → quant fp8 fused only. It checks both against the reference and against each other. Passes on gfx950 (1 launch vs 2 per graph).
- **Found while writing the goldens:** an MLP with intermediate 14336 (Llama 8B) had no fused or single quant kernel (N ≤ 8192). **Closed:**
  - sched's long-row quant extends `quant_token` fp8 / int8, `silu_mul_quant`, `situv2` and `group128` to N ≤ 32768;
  - the rules accept that range (`quant_n`); the norm kernels stay at ≤ 8192;
  - golden `llama_mlp_14336` plans `silu_mul_quant_fp8_d14336`;
  - `graph_check.py` `act_quant_long` (M 4, D 14336): fused vs unfused vs reference OK.
- **Step 3** (`l1-graph-step3`):
  - **Derived names.** `reachable::python_op_names` is derived from the rules over `amdgpu_graph::fusion::canonical::graphs()`, about 40 small graphs at the former hand-listed shapes. The md5 manifest is identical, so the derived set equals the old list.
  - **Plan-owned workspaces.** Every multi-kernel launch lists its `steps` (kernel, reads, writes, through the named workspaces). `graph.run` allocates the plan's workspaces and the wrappers use them (`ops_compute._ws_override`, consulted first by `_workspace`). Kernarg packing stays in the wrappers.
- **Vocabulary extended:**
  - `gemm`: bf16 | fp8_rowcol | i8_rowcol | fp8_blockscale | mxfp4. Tile, split-K and reduce come from the new `dispatch::gemm_kernels`, built on the family rules. That function now also owns the "reduce covers N in 1024-column blocks" guard, which the Python wrapper used to apply.
  - `gemm_w4`, `conv` (implicit GEMM, CTA from `conv_cta`; no Python wrapper, so `py` is empty and `run` refuses it), `attention_bwd`, `moe_reduce`, `argmax`, `sample`, `topk_per_row` (decode / prefill), `causal_conv1d` (fwd / update).
  - Twelve more golden plans.
- **GEMM numerics through the plan** (`graph_check.py`): the `_64_splitk` + `gemm_splitk_reduce_s4[_rc]` paths (bf16, fp8 row/col, fp8 blockscale; M 64, N 1024, K 512) and the `_64` form match the `ops_check` reference.
- **Python `gemm` (direct calls).** It takes `kernel` / `splits` / `reduce` from the plan. Its default tile choice (256 or `_128` only) is unchanged, so direct calls still differ from `dispatch::gemm_kernels`. Switching the default is a behaviour change (performance) left for a gated step.
- **One rule per Python entry point** (`one-rule-attn`). Every Python entry point that picks a kernel now asks the dispatch rule its plan uses (`amdc dispatch …`):

  | Python entry point | Rule (`amdc dispatch …`) |
  |---|---|
  | `gemm` | `gemm_kernels` (`gemm`) |
  | `fmha_fwd_mxfp8` / `select_path` | `mxfp8_prefill_path` (`mxfp8`). Formerly a Python rule; ported unchanged and checked by `amdgpu-graph/tests/dispatch_rules.rs` |
  | `prefill_bf16` | `bf16_prefill_kernel` |
  | `paged_decode_bf16` | `decode16_plan` |
  | `mla_decode` | `mla_config` |
  | `fused_moe` | `moe_block_m` |
  | `attention_bwd_bf16` | `bwd_causal_kernels` |

  - **Kept in the wrapper:** `prefill_bf16`'s `s_opt` kernarg is the vendor ABI's option word. Our kernels declare it but do not read it, so it is not a kernel choice and stays where it is.
  - **Graph `attention` op** also covers MXFP8 (fp8 q / k / v + scales → `fmha_fwd_mxfp8` with the path's kernels and the KV-split partials as plan workspaces).
  - **Checks:** `node_check.py` 17/17 (bit-identical FMHA vs the C++ harness through the new path rule); Python unit tests OK.
