# Family bodies as L2 templates: decoupling plan for M7

This is a design note for the crate split ([`crate-split.md`](crate-split.md) M7) and the L2 template catalogue ([`l2-templates.md`](l2-templates.md) §6).
It covers how the three family-body modules split out in `thin-families-ko` can become `amdgpu-stage` templates:
- `crates/amdgpu-kernels/src/fmha/attn3.rs`
- `crates/amdgpu-kernels/src/fmha/block3.rs`
- `crates/amdgpu-kernels/src/gemm/body.rs`

The modules must not depend on the kernel-crate family types (`FmhaIo`, `GemmConfig`). The note proposes no code change. Every step it proposes
is a move or a type narrowing, and must leave `amdc md5-manifest --check` identical.

## 1. What the bodies depend on today

Counts are field accesses in the module (Main after `thin-families-ko`).

| Module | Family type used | Accesses | Nature of the dependency |
|---|---|---|---|
| `fmha/attn3.rs` (`KvRing3`, `AttnBody3`) | `FmhaBlock` (`blk.*`) and `FmhaIo` (`io.*`) | `sq` 12, `pt` 10, `sel` 4, `lane` 4, `q` 3, `kv_len` 3, `kv_tile` 3, `g4` 3, `out_scale` / `v` / `k` / `row0` / `ksc` 1 each | per-block **runtime state** (lane, row origin, key range, Q fragments, scale selectors, per-tensor scales) and the **operand tensors** (K / V / K-scale views for the DMA plans) |
| `fmha/block3.rs` (`declare_io`, `block3*`, `store_out3*`, `store_partial3`) | `FmhaIo` (`io.*`) | `s_len` 13, `kvs` 4, `o` 4, `gqa` / `qsc` / `q` 2 each, `vsc` / `heads` / `kv_len` / `ptr_vd` / `ksc` 1 each | `declare_io` *is* the ABI: kernarg offsets and the derived host contract. `block3_at` mixes generic launch maths (key range, split decode, lane set-up) with ABI reads. The literal kernarg offsets in the file are `0x050` (scale), `0x060` (SQ), `0x0B0` (GQA), `0x110` (SK), `0x140` (H) and `0x280` / `0x284` (KV split), plus `splits_off` |
| `gemm/body.rs` (`Body`, `Panel`, `SfaJob`, `Mx32`, `PipeBody for Body`) | `GemmConfig` (`cfg.*`), `Scales`, `Elem`, `PANEL_KB`, `ConvRt`, `Dma` | `cfg.elem` 11, `wave_m` 4, `wave_n` 3, `scales` 3, `mid_sync_mmas` 2, `pre_sync_mmas` 1; `Scales::*` 25 | a handful of **numbers and two format enums** from the configuration; the conv runtime (`ConvRt`, row-source maths); the DMA plan (`gemm/dma.rs`, layout-generic) |

None of the three uses a per-shape decision. Every family-type use is either data or an ABI read.

## 2. Principle: move data down, keep the ABI up

Plan §3 and l2-templates §0 already draw the line:
- **ABI and contract declarations**, kernarg offsets and per-shape choices stay in `amdgpu-kernels`.
- **Protocol and state** move to `amdgpu-stage`.

So each family type splits into:

| Today | Moves to `amdgpu-stage` (L2) | Stays in `amdgpu-kernels` |
|---|---|---|
| `FmhaBlock` | all of it, as `attention::BlockState`. It is pure per-block runtime state (lane, ids, row origin, key range, Q frags, scale selectors, `PtScale`); no field is an ABI fact | — |
| `FmhaIo` | a trait `attention::Operands`, implemented by `FmhaIo` | the struct, `declare_io` (the ABI + derived contract) and the kernarg offsets |
| `KvsArgs` (KV-split extension) | the data part (`splits`, `rows_total`, partial pointers) as `split::KvSplitArgs` | `splits_off` and the offsets (ABI) |
| `GemmConfig` | `gemm::BodyParams { elem, scales, wave_m, wave_n, pre_sync_mmas, mid_sync_mmas, panel_kb }` plus the format enums `Elem`, `Scales` (operand formats are L2 / L3 data, not shape choices) | `GemmConfig` and its builders (`v4()`, `a4w4()`, ...); `GemmConfig::body_params()` |
| `ConvRt` | — (it stays: conv address maths is per-op) | `ConvRt`; the body sees it through the row-source trait below |
| `gemm/dma.rs` (`Dma`, `dma_plan*`) | L3: `amdgpu-tile::dma` next to `partition_dma` / `issue_rows_shared`. The plans are pure functions of layouts + `Ctx` | — |

### 2.1 Attention: `Operands` trait

```rust
// amdgpu-stage::attention
pub trait Operands {
    fn q(&self) -> &Tensor;          // + qsc() for the MXFP8 forms
    fn k(&self) -> &Tensor;
    fn v(&self) -> &Tensor;
    fn o(&self) -> &Tensor;
    fn ksc(&self) -> &Tensor;
    fn vsc(&self) -> Option<&Tensor>;
    fn s_len(&self) -> I32;          // query length (I32 expression)
    fn kv_len(&self) -> I32;
    fn heads(&self) -> ValueId;
    fn gqa(&self) -> ValueId;
    fn kv_split(&self) -> Option<&KvSplitArgs>;
}
```

- `KvRing3::new_x`, `AttnBody3`, `store_out3*` and `store_partial3` take `&dyn Operands`, or a generic `O: Operands` where monomorphisation is wanted.
- Either form emits the same IR: the trait only changes how Rust reaches a field, not which DSL ops are built, so the md5s are unchanged by construction.
- `FmhaIo` implements the trait in `fmha/io.rs` with one-line getters.

### 2.2 Attention: launch maths without literal offsets

`block3_at` reads kernarg offsets directly: H at `0x140`, SQ at `0x060`, SK at `0x110`, the split arguments at `0x280` / `0x284` or `splits_off`. The generic part takes them as data:

```rust
pub struct LaunchAbi { pub h_ka: u32, pub sq_ka: u32, pub sk_ka: u32, pub gqa_ka: u32, pub splits_ka: Option<u32> }   // filled by the kernel crate
pub fn block_state<O: Operands>(b: &mut FnBuilder, io: &O, abi: &LaunchAbi, rows_lg: i32, qtile: ValueId, head: ValueId, pt: Option<PtMode>) -> BlockState;
```

The grid map and split decode already live in `templates::grid` (`xcd_tile_head` takes `h_ka` / `sq_ka` today). `block3_x_mode` becomes a two-line kernel-side wrapper:
1. `grid::xcd_tile_head(..)` or the plain ids;
2. `attention::block_state(..)`.

`declare_io` stays in `amdgpu-kernels` unchanged.

### 2.3 GEMM: `BodyParams` and a row source

```rust
// amdgpu-stage::gemm
pub struct BodyParams { pub elem: Elem, pub scales: Scales, pub wave_m: i64, pub wave_n: i64,
                        pub pre_sync_mmas: usize, pub mid_sync_mmas: usize, pub panel_kb: i64 }
pub trait RowSource { fn row_src(&self, b: &mut FnBuilder, r: I32) -> ValueId; }   // MoE gather and conv implicit-GEMM rows
```

- `Body` holds `params: BodyParams` instead of `cfg: Cfg`; `self.cfg.elem` becomes `self.params.elem` (20 sites, mechanical).
- `conv: Option<ConvRt>` becomes `rows: Option<&dyn RowSource>`. `dma_plan_gather` already takes the row source as `&dyn Fn(&mut FnBuilder, I32) -> ValueId`, so the trait only names it.
- `PANEL_KB` becomes `params.panel_kb`. The skeleton slots (`GemmSlots`, the per-shape assembly) stay in `gemm/mod.rs` and build `BodyParams` from `GemmConfig`.

## 3. The generated skeletons

- `fmha/skeleton_{vs128,kvsplit,t128}.rs` and `gemm/skeleton_v4.rs` are generated and use `super::*`. They name `FmhaIo` / `FmhaBlock` / `KvRing3` / `Body` through the family module.
- After the split they keep compiling if each family module re-exports the moved items (`pub use amdgpu_stage::attention::{BlockState as FmhaBlock, KvRing3, AttnBody3, ..}`). That also covers M7's "only the `use` lines change" rule.
- The glob `use amdgpu_tile::*` (`fmha/mod.rs`, inherited by `attn3.rs` / `block3.rs` via `use super::*`) must be replaced by explicit imports in each new file before the move. Otherwise the stage-side files would depend on a kernel-crate glob.

## 4. Order of work (each step md5-gated)

1. **Narrow the types in place** (kernel crate, no file moves), one commit per family:
   - attention: introduce `Operands` + `LaunchAbi` and make `attn3` / `block3` use them;
   - GEMM: introduce `BodyParams` + `RowSource` and make `gemm/body.rs` use them.
   - Gate: `md5-manifest --check` 0 differences, `cargo test`, shipped / strict verify.
2. **Explicit imports** in `attn3.rs`, `block3.rs`, `gemm/{body,dma,mma}.rs` (removes the glob dependency). Gate: md5.
3. **Move** (sched, M7):
   - `FmhaBlock`, `KvRing3`, `AttnBody3`, the launch maths, the epilogues, `Operands` / `LaunchAbi` → `amdgpu-stage::attention`;
   - `Body` + `BodyParams` + `Elem` / `Scales` → `amdgpu-stage::gemm`;
   - `gemm/dma.rs` → `amdgpu-tile::dma` (L3).
   - The kernel crate keeps `FmhaIo` / `declare_io` / `GemmConfig` / the skeleton slots / the registry and adds re-exports. Gate: md5.
4. Only then any behaviour change (T6 common scale API, `PairRing` as a `Cadence`), each with its own A/B and md5 re-baseline.

Steps 1–2 are kernel-opt work once M5 / M6 have landed. Step 3 is sched's M7.

## 5. Open points

- **`dyn` vs generics.** `PipeBody` is already used as `&mut dyn PipeBody`; `&dyn Operands` keeps the same style and avoids generic parameters on `AttnBody3<'a>`. Generics are only needed if a hot getter shows up in build time (not expected: the bodies are built once per kernel).
- **Where `Elem` / `Scales` live.** They describe operand formats (fp8 / fp4 / i8 / bf16, Block128 / Mx128 / Mx32 / RowCol). L2 is the minimum; if an L3 atom table needs them (`amdgpu-isa` already distinguishes the MFMA formats), they belong one layer lower. Decide in step 1.
- **`PtScale` / `PtMode` (per-tensor FP8).** They are block state (per-tensor descales folded into the Q / K scale), so they move with `BlockState`. The ABI of the descale pointers stays in `fp8pt.rs`'s `declare_io_pt`.
- **Verifier facts.** `block3_at`'s trusted ranges reference kernarg symbols through `Ext::ka(offset)`. With `LaunchAbi` the offsets become data, so the facts are built from `abi.*`. This is the same pattern `templates::grid::xcd_tile_head` already uses (`h_ka`, `sq_ka`).

## 6. Status: steps 1 and 2 done (kernel-opt, `narrow-types-ko`)

Both steps are md5-neutral: `amdc md5-manifest --check` reports 234 kernels, 0 differences.

- **Attention.**
  - `Operands` and `LaunchAbi` are defined in `fmha/block3.rs`; `FmhaIo` implements `Operands` in `fmha/io.rs`.
  - `KvRing3::{new, new_x, rebuilt}`, `block3*`, `store_out3*` and `store_partial3` take `&dyn Operands`.
  - The literal kernarg offsets of the block launch now come from `io.abi()`: H `0x140`, SQ `0x060`, SK `0x110`, softmax scale `0x050`.
  - The final `LaunchAbi` has `scale_ka` where §2.2 sketched `gqa_ka`. GQA is reached through `Operands::gqa()`, so no GQA offset is needed.
  - `declare_io` (the ABI) moved from `block3.rs` to `fmha/mod.rs`, so `block3.rs` holds no ABI declaration.
- **GEMM.**
  - `Body` holds `params: BodyParams` (built by `GemmConfig::body_params`) instead of the configuration.
  - The convolution row source is the trait `body::ConvSource` (`tile_voffsets`, `zero`), implemented by `ConvRt` in `gemm/mod.rs`. It is named `ConvSource` rather than §2.3's `RowSource` to avoid a clash with `amdgpu_tile::rows::RowSource`.
  - `PANEL_KB` appears in `body.rs` only in a comment, so `BodyParams` has no `panel_kb`.
- **Explicit imports.** `fmha/{attn3, block3}.rs` and `gemm/{abi, body, dma, epilogue, mma}.rs` have no `use super::*` and no `amdgpu_tile::*` glob. Imports name the source crates directly.
- **Family items still imported through `super::`** (they move with the body or stay as configuration, see §2):
  - fmha: `io::FmhaBlock`, `block3::Operands`, `attn3::attention_orientation`;
  - gemm: `Elem`, `Scales`, `Cfg`, `MoeCx`, `MoeStage`, `ConvCfg`, `EPI_PITCH`, `PANEL_KB`, `dma::Dma`, `mma::*`.
