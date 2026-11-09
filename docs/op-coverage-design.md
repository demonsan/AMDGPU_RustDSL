# 基础算子覆盖：迁移设计（对照 aiter kernel zoo，gfx950）

**状态**：设计稿（kernel-opt 主笔；§5 访存模板由 sched 写，§6 tile-op 覆盖矩阵由 backend-llvm-tail 写）。分支 `opcov-design`，基于 Main `f25294dc`。

**目标**：对照 MI_Kernel_Agent 的 aiter kernel zoo，把通信以外的基础算子用 rust_dsl 实现。先做类 TIRx 的基础算子覆盖，重点放在 tile / pipeline / 访存这些决定性能的设计上。

**输入**：
- `MI_Kernel_Agent/knowledge/aiter-kernel-zoo`（README、operators.md、groups.md、categories/*.md；清单固定在 aiter `475cf0f6`）；
- aiter 源码 `../aiter`（本地检出 HEAD `6b5ef5db`，比清单新，含清单未列的 gfx950 Opus 文件）；
- 本仓库的 stage 库（[`fusion-stage-library-design.md`](fusion-stage-library-design.md)）、算子图（[`opgraph.md`](opgraph.md)）、weave2（[`weave-capture.md`](weave-capture.md)）。

**证据标注**：
- `[src]`：读了源码或 CSV；
- `[推导]`：按 traits 公式算出；
- `[待核]`：未确认；
- `[实测]`：在 MI355X 上测过（注明 GPU / 协议）。

清单只基于读取，没有编译或运行 aiter。

| § | 内容 |
|---|---|
| 1 | 算子清单（语义变体） |
| 2 | 映射到原语与 pipeline 模板 |
| 3 | 计算密集 pipeline 的迁移设计（gfx950 tile 配置、LDS / 寄存器预算、wave 角色、环深） |
| 4 | 访存设计规则 |
| 5 | 访存受限算子的 pipeline 模板 → [`op-coverage-memtemplates.md`](op-coverage-memtemplates.md)（sched） |
| 6 | TIRx 式 tile-op 覆盖矩阵 → [`op-coverage-tileops.md`](op-coverage-tileops.md)（backend-llvm-tail） |
| 7 | 批次与验收 |

---

## 1. 算子清单（通信除外）

按 zoo 的类别列出语义变体。分类轴：dtype / 量化方案 / 布局 / 融合。tile 事实见 §3。

### 1.1 GEMM / BMM

约定：`Y[M,N] = A[M,K]·B[N,K]ᵀ`，A、B 都是 K 连续（TN），累加 fp32。

| 变体 | A / B / C | 量化 | 布局 | 融合 |
|---|---|---|---|---|
| A16W16 | bf16 / bf16 / bf16、fp32 | — | 普通 B；BMM batch 在前；ASM 有 B preshuffle | bias `[N]` / `[batch,N]`；split-K（fp32 工作区 + 归约 kernel）；gated FFN；输出量化 |
| A8W8 无缩放 | fp8 / fp8 / fp32 | — | 普通 | — |
| A8W8 per-token × per-channel | int8 / fp8 → bf16 / fp16 | 行 scale × 列 scale（fp32） | 普通或 B preshuffle | bias、split-K |
| A8W8 blockscale（DeepSeek-V3） | fp8 / fp8 → bf16 | A 1×128，B 128×128（fp32 scale `[M,K/128]`、`[N/128,K/128]`） | 普通 / B preshuffle / AB preshuffle | — |
| MXFP8 BMM | e4m3 / e4m3 → bf16 / fp32 | E8M0，group_k 32（MX）或 128（DSv4） | batch 在前，B 可 preshuffle | split-K（两段或同 XCD 计数器融合）、M 外层持久 |
| MXFP4（A4W4） | e2m1 / e2m1 → bf16 | E8M0 1×32 | 普通 / B preshuffle | split-K、A 预量化融合 |
| MXFP6 与混合 MX（f6f6 / f4f6 / f6f4） | → bf16 | E8M0 1×32 | packed pad_k | A 量化融合 |
| A16 / A8 × FP4 权重 | bf16 / fp8 × e2m1 | 权重 MXFP4 | — | — |
| grouped GEMM（非 MoE） | bf16 | 可选 scale | `group_layout` | — |
| 融合 norm + quant + GEMM、BMM + RoPE + KV-cache 写 | — | — | — | prologue / epilogue 链 |

**形状区间**：
- decode / 窄（M ≤ 128）：split-K、单 wave 无 LDS、B 直读寄存器；
- prefill / 大 M：256×256 tile、8 wave。

### 1.2 卷积

| 变体 | 说明 |
|---|---|
| conv3d / 2d 隐式 GEMM（bf16） | M = N·D·H·W，N = Cout，K = C/g·kT·kH·kW；grouped、bias、split-K、NDHWC / NCDHW、im2col 或 strip / patch gather |
| conv2d 其他形式（Triton） | 1×1（即 GEMM）、3×3 direct、NCHWc、NHWC、Winograd F(4,3) |
| causal depthwise conv1d（Mamba） | fwd / 单 token update / split_qkv 融合；访存受限 |

### 1.3 Attention、KV cache、RoPE

**方向**：
- fwd；
- bwd，三个 kernel：`odo` 行和、`dqdkdv` 主体、`dq_convert`（fp32 dq 原子累加 → bf16）。

**阶段**：dense prefill、varlen、paged KV 上的 batch prefill、chunked / extend、decode、MTP decode（q 1–64）。

**头拓扑**：
- MHA / GQA / MQA：`nhead_q` / `nhead_k`；
- MLA：QK 576 = 512 latent + 64 rope，V 512 = K 的 nope 部分；heads 打包进 M（qh16–128）；prefill 192 / 128。

**mask**：
- 无 / causal（top-left、bottom-right）；
- sliding window、sinks、additive bias（CK 路径）；
- block-sparse：工作表 + tile mask；
- DSA top-k 稀疏 MLA：indexer = fp8 / mxfp4 的 MQA logits + top-k。

**head dim**：64 / 128 / 192 / 256；MLA 576 / 512。

**dtype / 量化**：
- bf16 / fp16；
- fp8 per-tensor / per-head（输出 bf16）；
- MXFP8（我们的出厂 kernel）；
- v4 的格式组合：BF16-QK + FP8-V、INT8-QK + FP8-V、MXFP6、MXFP4、F8F6 等，scale 有 f32 per-tensor / head / token / channel 与 E8M0 1×32 两类；
- MLA：a16w16 / a16w8 / a8w8，以及 MXFP8 MLA（fp8 nope + E8M0 + bf16 rope）。

**paged KV**：
- 页大小：16（vLLM 布局）、256 / 1024（per-block fp8）；
- 寻址：`block_tables`，或 `kv_indptr + page_indices + last_page_lens`（ragged / 持久 / MLA）；
- KV 量化：per-token fp8 / int8，或 per-block fp8。

**split-KV 与合并**：
- PA 按 256 token 分区（`exp_sums` / `max_logits` / `tmp_out`）+ reduce；
- MLA 由 metadata work list 驱动（`o_accum` / `lse_accum`）+ reduce；
- 我们的 KV-split + `combine8h`。

**其他特性**：
- LSE 输出（fp32）；
- dropout / ALiBi / softcap：只在 CK 与 PA rocm / v1 路径出现，gfx950 ASM / Opus / FlyDSL 中未见 `[待核]`。

**KV-cache 布局与写入**：
- vLLM K `[blocks, kvh, hd/x, bs, x]`（x = 16 B / elem），V `[blocks, kvh, hd, bs]` 或 ASM shuffle `[blocks, kvh, bs/x, hd, x]`；
- flash NHD；
- per-token scale `[blocks, kvh, bs]`；
- 写入算子：
  - `reshape_and_cache`（含 per-token / block 量化，以及重分桶到 ASM-PA 的 16 token 块）；
  - `concat_and_cache_mla`（latent ‖ rope，seg 形式）；
  - indexer K 量化 + cache；
  - `copy_blocks` / `swap_blocks`。

**RoPE**：
- NeoX（半段旋转）/ GPT-J（交错）；部分旋转（nope_first）；
- 频率来源：uncached freqs、cached cos/sin、+ positions、+ offsets；
- 布局：sbhd / thd / 2d；1 通道或 q+k 2 通道；fwd / bwd；
- M-RoPE 3D；
- 融合：qk-norm + rope + cache 写 + 量化（shuffle 布局），MLA rope + concat + cache。

### 1.4 MoE

**路由**：
- topk-softmax：logits `[T,E]` bf16 / f16 / f32 → weights fp32、ids int32；可 renorm，可带 shared experts；
- sigmoid / sqrt-softplus 打分；
- DeepSeek-V3 分组 top-k：sigmoid + bias，先选 topk_group 组再组内 top-k，输出用 bias 前的权重。

**排序 / 对齐**：
- 输出：
  - `sorted_token_ids`：按 expert 分组并补齐到 block_m，低 24 位 token、高 8 位 top-k 槽；
  - `sorted_weights`；
  - `sorted_expert_ids`：每个 M tile 一个；
  - padded 计数。
- 选项：EP 局部 expert 掩码；跳过空 expert；顺带清零 stage-2 的 `moe_buf`。
- 补齐后长度 = T·k + E·(block_m − 1)。

**stage 1**：
- `act(X[tok]·Wgu[e]ᵀ)`，gate × up；
- 激活：silu / gelu / swiglu（含 clamp）/ relu2；
- 可选 `doweight_stage1`；
- block_m ∈ {32, 64, 128}，按 CU 利用率选，默认 32。

**stage 2**：
- `out[tok] += w·(Y1·Wd[e]ᵀ)`；
- 累加方式：直接累加到 `moe_buf`（原子与否 `[待核]`），或按 route 写出后单独 reduce（Opus a8w4）。

**量化组合**：a16w16、a8w8（per-tensor / per-token，fp8 / int8）、fp8 1×128 blockscale、a16w4 / a8w4 / a4w4（MXFP4 权重）、w-int4、smoothquant int8。

**其他**：gather / 量化 scatter / gather-reduce；1 段融合 fmoe 与 2 段。

### 1.5 Normalization / 量化 / 激活

**Normalization**：
- RMSNorm（+ residual add 并写回、Gemma 的 w + 1、+ fp8 / int8 per-token / per-group / mxfp4 量化）；
- gated RMSNorm（y·silu(z)）；
- qk-rmsnorm + group quant；
- LayerNorm；
- GroupNorm（Welford 两段）；
- adaLN。

**量化**：
- static / dynamic per-tensor；
- dynamic per-token（可带 scale 上界）；
- per-group 1×128 / 1×32；
- MXFP4：e2m1 两个一字节 + E8M0 / 32；
- MXFP8；
- fp8 128×128 blockwise；
- int8 smoothquant；
- scale 的 shuffle / preshuffle 布局；transpose；dequant + gather。

**激活**：
- `act_and_mul(x[T,2d])`，act 取 silu / gelu / gelu_tanh / swiglu / relu2，可带 bias；
- scaled → fp8；
- `act_and_mul_quant`（per-group fp8 / mxfp4）；
- 一元 gelu_fast 等。

### 1.6 采样 / top-k / 其他

**采样 / top-k**：
- greedy / random 采样：softmax(logits·T) / 指数噪声 → argmax；
- top-p、top-k + top-p、top-k renorm：拒绝轮次 + pivot，11 位 radix select；
- per-row top-k（含 2048 的长行 top-k，用于稀疏注意力 indexer）。

**其他**：
- cache ops；
- 二元 / 一元逐元素（带广播）；
- softmax、cross-entropy；
- split-K 归约；
- 窄 GEMM（`wvSplitK`）；
- `dcp_topk_merge`；
- gather_kv_b_proj；
- 旋转 + 量化（dsv4_rotate_quant、inverse_rope_group_quant）；
- embedding：在已查的 6 个类别中未见；attention / rope 类别未查。

**scan / sort**：没有独立 kernel（检索 `csrc` 的 cumsum / radix_sort / bitonic）。scan 与 sort 都内嵌在其他算子里：
- DPP wave cumsum：MoE 排序与分组 top-k；
- hipcub BlockScan：采样；
- bitonic：top-k。

---

## 2. 映射：原语 + pipeline 模板

原语定义见 [`fusion-stage-library-design.md`](fusion-stage-library-design.md) §1（M 搬运 / A MMA 主循环 / E 逐元素 / R 归约 / O 在线归约 / Q 量化 / S 选择 / P 扫描）。

**计算密集的 pipeline 模板**（§3）：

| 模板 | 用途 |
|---|---|
| `GemmMS` | 多级直达 LDS |
| `GemmRS` | 寄存器暂存、4 wave AGPR（我们的 v4） |
| `AttnPP` | 两组 ping-pong（我们的 v3） |
| `Decode` | paged / split-K |
| `Grouped` | MoE / grouped GEMM |
| `ImplicitGemm` | 卷积 |

**访存受限的模板**（§5，T1–T7）：

| 模板 | 用途 |
|---|---|
| T1 | 流式逐元素 |
| T2 | 行归约 |
| T3 | 分段 / 列归约 |
| T4 | softmax / norm 的一趟 / 两趟 |
| T5 | 前缀扫描 |
| T6 | 排序 / top-k |
| T7 | gather / scatter / cache 写 |

| 算子族 | stage 组合 | 模板 | 已有 |
|---|---|---|---|
| GEMM A16W16 / A8W8 per-token / MXFP8 / MXFP4 | `M(A,B) → A → Epilogue(E*,[R],[Q],M)` | `GemmMS`（大 M）/ `GemmRS`（MXFP8 blockscale）/ 窄 M：`Decode` 式 split-K | `gemm_mxfp8_bs_v4`（`GemmRS`）、`mlp_fp8_ws_*`（权重驻留 LDS） |
| GEMM blockscale 1×128 / 128×128 | 同上，A 带每 128-K scale 的 `E` 修正（scale 在累加器上） | `GemmRS` | v4 的 scale 路径（32 元素 E8M0）需要扩成 fp32 per-128 |
| grouped GEMM、MoE stage 1 / 2 | `M`（带索引：sorted ids）→ `A` →（s1）`E`(act·mul)[→`Q`] /（s2）`E`(×w) → `M`（scatter 或原子 `R`） | `Grouped` | MLP stage（`gemm1_act`、epilogue 链） |
| MoE 路由 / 排序 | `R`/`S`(top-k) + `E`(softmax / sigmoid) → `S`(计数) + `P`(块偏移) → `M`(scatter) | T6 + T5 | 无 |
| FMHA fwd（各 dtype、mask、GQA） | 见 [`fmha-case-study.md`](fmha-case-study.md) | `AttnPP` | v3 三条出厂路径（MXFP8，hd128） |
| FMHA bwd | `A`×4 + `R`(dO·O) + `E` + dq 原子 `R` | `AttnPP` 的变体（KV 驻留） | 无 |
| paged decode / MLA decode | `M`（页表 gather）→ `A` → `O` → `A` → 部分结果 + `R`(combine) | `Decode` | KV-split + combine8h（非分页） |
| conv 隐式 GEMM | `M`(im2col 地址) → `A` → `Epilogue` | `ImplicitGemm`（= `GemmMS` + 地址生成） | 无 |
| norm / 量化 / 激活 / RoPE / cache 写 | `M → [E] → R → E → [Q] → M` | T1 / T2 / T3 / T7 | `rmsnorm_quant_fp8_n<N>` |
| softmax / cross-entropy | `R`(max, sum) + `E` | T4 | 无 |
| top-k / 采样 | `R` + `S` [+ `P`(CDF)] | T6 / T5 | 无 |

---

## 3. 计算密集 pipeline 的迁移设计（gfx950）

### 3.0 机器约束（gfx950，每 CU）

- **寄存器**：4 个 SIMD，每 SIMD 512 个 VGPR + AGPR（统一分配，粒度 8）。分配按 kernel 统一，所以同一 kernel 里不可能有"生产者 wave 少寄存器"（[`fmha-wave-org.md`](fmha-wave-org.md) §1）。
- **LDS**：160 KiB / CU。
- **MFMA**：
  - scaled f8f6f4：32×32×64（64 周期）、16×16×128（32 周期）；
  - bf16 / f16 双速率：32×32×16、16×16×32；
  - smfmac。
- **XCD**：8 个，每个有自己的 L2，workgroup 按 linear id mod 8 轮转分派（§4.6）。
- **流带宽**：copy 5.7–6.2 TB/s；每 CU 需要 ≥ 64 KB 在途（[`op-coverage-memtemplates.md`](op-coverage-memtemplates.md) §5.1 `[实测]`）。

### 3.1 `GemmMS`：多级直达 LDS（A16W16、A8W8 per-token、MXFP4 的大 M）

**参照**：Opus a8w8 / a16w16 gfx950 `[src]`：
- 512 线程（8 wave，2×4），tile 256×256×128（fp8）或 256×256×64（bf16）；
- A、B 经 `buffer_load … lds` 直达 LDS，2 个 K 缓冲（tic / toc）× 2 个 M / N 半区；
- LDS 132 KiB `[推导]`；
- 两组 wave 错开半相位（ping-pong）+ `s_setprio`。

**我们的形态**：
- tile 256×256×K_t，8 wave，每 wave 64×128（MFMA 16×16×128 f8 或 16×16×32 bf16）；
- 环深 2（K_t = 128 fp8 → 每级 A + B = 64 KiB；bf16 K_t = 64 时同为 64 KiB）+ 每行 32 B padding 或 swizzle（§4.4），LDS ≈ 132 KiB；
- 累加器每 wave 64×128 / 64 lane = 128 fp32 dword，可放 AGPR。

**节奏与 stage**：
- 节奏：`Pipeline::TwoPhaseOffset`（两组 4 wave，与 FMHA `_vs128` 相同的骨架），拷贝 stage 在 V 相位、MMA 在 M 相位；
- stage：`move.ab_stage`（已有）+ `mma.tile`（扩成 bf16 / fp8 两种 atom）。

**寄存器**：
- A / B 片段：每个 K 子块 8 + 8 dword；
- 累加器 128；
- 地址 SGPR 化（XCD 顺序与 tile 原点都放 SGPR）；
- 预算 ≤ 256 VGPR + 128 AGPR。

**窄 M 变体**（M ≤ 64）：
- 1–4 wave、B 直读寄存器（MXFP8 wave1 式：无 LDS，约 4 级寄存器环）；
- split-K 用同 XCD 计数器融合归约，或 fp32 工作区 + T3 归约。

### 3.2 `GemmRS`：寄存器暂存、4 wave AGPR（我们的 v4；MXFP8 / blockscale）

**已出厂**：`gemm_mxfp8_bs_v4`：
- 256×256×128，4 wave，`v_mfma_scale_f32_16x16x128_f8f6f4`（swap_ab）；
- A / B 片段与部分 C 在 AGPR，两 tile 一轮、静态环槽；
- 0.988–0.996× 厂商。

**迁移规则**：
- blockscale 1×128 / 128×128（fp32 scale）不能用 MFMA 的 E8M0 字节缩放：每 128 K 做一次 `A` 子循环，累加器乘 `sa·sb` 后累加，即"两级累加器"（`E` 在累加器上）。
- 寄存器另需一份 64×128 / wave 的临时累加器。若超出预算，改为 `GemmMS` 的 8 wave 形态（每 wave tile 减半）。
- MXFP4：同一 atom（f8f6f4 的 A / B 格式选 fp4，见 §6.4 P1-d），K_t = 256。

### 3.3 `AttnPP`：两组 ping-pong 注意力（我们的 v3）

**已出厂**：
- 256 行 / 512 线程；
- `TwoPhaseOffset`：组 A 发 K、组 B 发 V，2 槽环；
- QK 与 PV 都是 32×32×64 MXFP8；
- 在线 softmax：两字参考 + 懒回滚；
- XCD 顺序。

**扩展轴**：

| 轴 | 改动 | LDS / 寄存器 |
|---|---|---|
| bf16 / fp16 | QK、PV 改 bf16 32×32×16（4 倍 MFMA 条数）；P 转 bf16（`cvt_pk_bf16`）；K / V 每 tile 字节 ×2 → kv tile 64（Opus `gqa_d128` 的选择） | K 环 2 × 64·128·2 = 32 KiB，V 同；≈ 100 KiB |
| fp8 per-tensor | 与 MXFP8 同 atom，scale 改常量 | 不变 |
| hd 64 / 192 / 256 | hd 192：q 行 128、kv tile 128（aiter v3）；hd 256：kv tile 64、O 累加器 8 个 32×32 | O 累加器 ∝ hd，hd 256 需要 AGPR |
| causal / 滑窗 | tile 级跳过 + 对角 tile 用 `KeyMask`；按 q tile 反序分派以平衡负载 | — |
| GQA / MQA | 已有 `gqa_kv_head` | — |
| LSE | 已有（KV-split 的 `lse`） | — |
| varlen | `cu_seqlens` 间接，`row0` / `kv_len` 来自表 | — |

**反向**：
- KV 驻留：每 block 一段 K / V 在 LDS，按 q tile 循环；
- `A`×4：S = QKᵀ，dP = dO·Vᵀ，dV += Pᵀ·dO，dK += dSᵀ·Q；
- dQ 用 fp32 原子累加（需要 §6.4 P2 的全局原子），后接 convert；
- 预处理 `odo` = T2 行归约。

### 3.4 `Decode`：paged / split-K 注意力与 MLA

**形态**：q 行少（GQA 组 8–16，或 MTP 16–64），KV 长 → 并行度来自 split-KV（我们的 KV-split）或持久 work list（`_ps`）。

**tile 与环**：
- heads × q 打包进 M（MLA：16 行 / wave × 8 wave）；
- kv tile 32–64 token；
- K 环 4 级、预取距离 2（Opus MLA `[src]`）；
- V 直接从 K-nope 的 LDS 用 `ds_read_b64_tr_b8` 转置读（MLA，V = K 的前 512 维）。

**页表**：`M` 的地址生成带两级索引：
- 页 → 物理块：SGPR 里的块号，每 tile 一次 `s_load`；
- 页内偏移。

页大小 16 时一个 kv tile 跨多页：每页一条描述符（MUBUF `soffset`）。

**合并**：已有 `combine8h`（在线 max + 重标定），扩展到 work list 驱动的不等长分段。

### 3.5 `Grouped`：MoE 与 grouped GEMM

**流水**：
1. 路由（T6）：topk-softmax / 分组 top-k；E ≤ 256 时一行由一组 lane 处理，DPP / permlane，无 LDS。
2. 排序 + 对齐（T5 + scatter）：E ≤ 64 且 T·k 小时，单 block 计数 + 扫描；否则三段：计数 / 扫描 / scatter。
3. stage 1：
   - 每个 M tile（block_m 32 / 64 / 128）一个 expert；A 行按 `sorted_token_ids` gather（`M` 带索引：每行一条 buffer 读，行号在 SGPR / VGPR）；
   - B = 该 expert 的 W 切片；
   - 模板 = `GemmMS`，block_m 32 时用 4 wave、N tile 128–256；
   - epilogue：`E`(act·mul)，可选 `Q`；写到排序后的连续行。
4. stage 2：同上，epilogue 乘路由权重；按 token scatter-add（原子）或 route 输出 + T3 归约。

**负载**：
- tile 数随 expert 负载变化 → 持久 grid：每 CU 1 block，循环取 tile；
- XCD 顺序：同一 expert 的 tile 放到同一 XCD，权重在 L2 中复用。

### 3.6 `ImplicitGemm`：卷积

- `GemmMS`，A 的地址生成换成 im2col（每个 K 子块对应 `(c, kt, kh, kw)`，每 lane 一个 (n, d, h, w)）；越界由 buffer 描述符 + 显式 padding 判断（`E` 掩码）处理。
- 参照 FlyDSL conv3d：
  - 128×128×32、8 wave、4 级 LDS（64 KiB）、`buffer_load_lds` 16 B；
  - 小通道 N tile 有 48 / 96 / 192（VAE）。

---

## 4. 访存设计规则

| # | 规则 | 依据 |
|---|---|---|
| 4.1 向量宽度 | 每 lane 每条 16 B（b128）：bf16 8 个、fp8 16 个、fp32 4 个、fp4 32 个；尾部用 buffer 描述符越界读 0，不单独写尾 kernel。b32 / b64 / b128 在流式拷贝里实测无差别，但 b128 指令数最少（发射受限的 kernel 里重要） | sched §5.1 `[实测]`；我们的 DMA 全用 b128 |
| 4.2 buffer 描述符 / 越界 | 张量一律走 MUBUF：`num_records` = 字节数，越界读 0、写丢弃，可以免去尾部 clamp；描述符在 SGPR，每 kernel 一次；`soffset` 放波统一的 tile 原点（SGPR），`voffset` 放每 lane 不变的偏移；`gpu_verify` 认可这种"硬件守卫" | FMHA `_vs128`、GEMM v4、rmsnorm |
| 4.3 `global_load_lds` / `buffer_load … lds` | 目的地址 = M0 + imm + 16·lane，所以从目的端反推源地址（`partition_dma`）；swizzle 搬到源地址；每 tile 一次 `readfirstlane` 得 M0；一条指令 1 KiB（64 lane × 16 B） | [`fmha-dsl-tour.md`](fmha-dsl-tour.md) §4c |
| 4.4 LDS swizzle vs padding | `ds_read_b128` 行读、`ds_read_b64_tr_b8` 转置读：用 XOR swizzle（`Swizzle<3,4,3>` / `fmha128`），因为 DMA 写入按 lane 连续，padding 不可行；寄存器写 LDS（`ds_write`）的 epilogue 转置：swizzle 或每行 +16 B padding 均可，按 bank 冲突计数选 | `rowsum` / epilogue 的证明与计数器 |
| 4.5 持久 vs 一 tile 一 block | 流式算子：一 tile 一 block、一次 b128 优于 grid-stride（实测）；计算密集：tile 数 ≤ 2 × CU 时一 tile 一 block，负载不均（MoE、varlen、causal）时持久 + work list | sched §5.4 `[实测]`；FMHA |
| 4.6 XCD 顺序 | 共享输入的 block（同一 head 的 q tile、同一 expert 的 M tile、同一 B 列块的 GEMM tile）放到同一 XCD：线性 id `L` → `(L mod 8)·N/8 + L/8`；同一 XCD 上 32 个 CU 同时读同一段时，改为每块 16 个逻辑块轮转，避开 L2 tag 热点 | [`fmha-wave-org.md`](fmha-wave-org.md) `[实测]`：L2 命中 20 → 77 %，+3.5…8.3 % |
| 4.7 L2 复用 | GEMM 的 tile 光栅：M 外 N 内的组，组宽使 B 的列块留在 L2（v4 已有 16×16 grid 的 XCD 光栅；L2 命中 80 % vs 厂商 71 % `[实测]`） | [`fmha-wave-org.md`](fmha-wave-org.md) §3 |
| 4.8 nontemporal | 流式写 `nt` 实测无收益，不作默认；保留为 epilogue 的参数（多 kernel 链时防止污染 L2），按 A/B 采纳 | sched §5.10 |
| 4.9 在途字节 | 访存受限：每 CU ≥ 64 KB 在途；计算密集：DMA 至少提前一个相位（v3 / v4 的环深 2），K/V 拷贝不在关键路径的 vmcnt 上 | sched 扫描；[`fmha-wave-org.md`](fmha-wave-org.md) §2 |
| 4.10 scale 布局 | E8M0 scale 与数据一起按 MFMA 操作数的 lane 布局 preshuffle（一个 dword 4 个 scale → `op_sel` 选字节）；fp32 per-128 scale 每 K 子块一次 SGPR / VGPR 读 | v3 K-scale 字节选择 |
| 4.11 索引访问 | gather / scatter 的行号放 VGPR，每行一条 `buffer_load` / `store`（`voffset` = 行号 × 行字节）；页表 / expert id 等波统一的放 SGPR（`s_load` / `readfirstlane`） | §3.4 / §3.5 |

---

## 5. 访存受限算子的 pipeline 模板

见 [`op-coverage-memtemplates.md`](op-coverage-memtemplates.md)（sched）：

| 节 | 内容 |
|---|---|
| §5.1 | 机器事实与 roofline 目标（实测流带宽，目标 = 同读写比流带宽的 90 %） |
| §5.3–§5.9 | T1–T7 |
| §5.10 | 调度含义 |
| §5.11 | 缺口 |
| §5.12 | 批次 2 的顺序 |

---

## 6. TIRx 式 tile-op 覆盖矩阵

见 [`op-coverage-tileops.md`](op-coverage-tileops.md)（backend-llvm-tail）：

| 节 | 内容 |
|---|---|
| §6.1 | TIRx 模型与我们的对应 |
| §6.2 | 指令族矩阵（global / buffer、LDS、跨 lane、VALU、转换、矩阵、标量 / 同步）× MIR / DSL / atom / safe-asm / HW |
| §6.3 | `tile::apply` / `tile::ewise` 在我们栈上的形态 |
| §6.4 | 缺口优先级 P1-a…P3 |
| §6.5 | 每条新指令的验证闭环 |

---

## 7. 批次与验收

### 7.1 验收（每个算子）

1. **正确性**：主机驱动（`tests/gpu/*_host.cpp`）对 CPU 参考（fp64 / fp32）。
   - 量化输出按半个 ulp 判定；bf16 按相对误差。
   - 形状覆盖：对齐、非对齐尾部、最小（1 行 / 1 tile）、大。
   - 量化 scale 单独比对。
2. **严格验证**：`gpu-verify --strict` 0 错误。证书记入 `current-status`；新的 `assume_trusted` 写明理由。
3. **性能**：
   - 访存受限：≥ 90 % × 同读写比的实测流带宽（sched §5.1）；
   - 计算密集：
     - 有可运行的厂商 kernel 时，同轮交替 A/B（≥ 5 轮，真实数据）；
     - 没有时，对 roofline：MFMA 峰值 × 利用率目标 ≥ 0.8，或同族我们已出厂 kernel 的效率。
4. **出厂条件**：md5 记录、无环境变量开关、文档登记（本文算子清单的状态列）。

### 7.2 节点上可运行的厂商对照（不需要 torch）

节点没有 torch / aiter Python 环境，可用的对照有三类：

- **aiter 预编译汇编 kernel**（`hsa/gfx950/*/*.co`）。用 HIP module API 加载，按 aiter C++ 包装（`csrc/py_itfs_cu/asm_*.cu`）的 kernarg 结构写主机驱动。我们已这样跑过 `fmha_fwd` mxfp8 与 4-wave GEMM。可直接对照的族：

  | 族 | `.co` 数 |
  |---|--:|
  | `bf16gemm` | 24 |
  | `fp8gemm_blockscale` | 6 |
  | `i8gemm` | 8 |
  | `f4gemm` / `f6gemm` / `f4f6gemm` / `f6f4gemm` | 35 / 40 / 30 / 31 |
  | `a16w8gemm` | 16 |
  | `fmha_v3_fwd` / `fmha_v4_fwd` | 20 / 20 |
  | `fmha_v3_bwd` | 126 |
  | `pa` | 87 |
  | `mla` / `mla_v4` | 89 / 2 |
  | `fmoe` / `fmoe_2stages` | 884 / 182 |
  | `topksoftmax` | 24 |
  | `reduce` | 4 |

  每族第一次接入时，从 CSV 与包装源码确定 kernarg，用一个已知形状对拍后再用于性能对照（先例：[`fmha-vs-vendor-asm-diff.md`](fmha-vs-vendor-asm-diff.md)）。
- **Opus `.cuh`**：header-only HIP，理论上可用 hipcc 加一个最小主机驱动编译。依赖面 `[待核]`，批次 0 先试一个（a16w16 gfx950）。
- **CK / Triton / FlyDSL**：需要 torch 或 Python 运行时，节点不可用。对照只能用 roofline。

### 7.3 批次

| 批次 | 内容 | 负责 | 依赖 |
|---|---|---|---|
| **0** 基础设施 | §6.4 P1-a（访存宽度族、ds_read2 / b96 / tr_b16）与 P1-b（DPP / bpermute / ballot / readlane）、P1-c 转换；`M` 的索引寻址（gather / scatter）；`R` 的作用域阶梯（wave → block）；厂商 `.co` 主机驱动框架（kernarg 从 CSV + 包装源码）；Opus 单文件编译试验 | backend + sched + kernel-opt | — |
| **1** GEMM 族 | A16W16 bf16（`GemmMS`，8 wave，256×256×64）；A8W8 per-token × per-channel（同模板，fp8 atom + epilogue 行 / 列 scale）；blockscale 1×128 / 128×128（`GemmRS` 两级累加）；MXFP4（f8f6f4 fp4 格式）；split-K（fp32 工作区 + T3 归约）与窄 M（1–4 wave B 直读）；对照 `bf16gemm` / `i8gemm` / `fp8gemm_blockscale` / `f4gemm` 的 `.co` | kernel-opt | P1-d |
| **2** 访存受限 | sched §5.12：T1 激活 / 门控 / 转换 → T2 norm / softmax / per-token 量化 → T3 per-group 量化 / transpose / 列归约 → T7 cache 写 / RoPE / MoE gather-scatter → T6 / T5 最小集（topk-softmax、MoE 对齐） | sched（kernel-opt 支持 stage 化） | 批次 0 |
| **3** attention 扩展 | bf16 / fp16 fwd（hd 64 / 128 / 192 / 256）、fp8 per-tensor、causal / 滑窗、varlen、LSE；decode：paged（页 16 / 256，block table 与 indptr 两种寻址）+ split-KV + combine；MLA decode（a8w8、a16w8）；对照 `fmha_v3_fwd`、`pa`、`mla` 的 `.co` | kernel-opt | 批次 0、1 的 bf16 atom |
| **4** MoE | 路由 + 排序（批次 2）+ stage 1 / 2（`Grouped`，a16w16 → a8w8 per-token → blockscale → a8w4 / a16w4 MXFP4）+ 持久 tile 调度 + XCD 按 expert；对照 `fmoe_2stages` | kernel-opt + sched | 1、2 |
| **5** 其余 | FMHA bwd（KV 驻留 + dq 原子）、conv 隐式 GEMM（bf16，128×128×32，4 级）、采样 / top-p（T5 / T6 的 block 级）、causal conv1d、GroupNorm / LayerNorm 两趟 | 分配待定 | P2 原子 |

**节奏**：
- 每个批次开始前，在本文 §1 的对应条目加"状态 / stage 组合 / 模板 / 对照 / 结果"列；
- 每个算子按 §7.1 验收后合入；
- 负结果同样记录，配分支与数据。

### 7.4 批次 1（GEMM 族）状态

**实现形态**：全部建在 `GemmRS`（v4 寄存器暂存结构，[`gemm/mod.rs`](../crates/amdgpu-kernels/src/gemm/mod.rs)）上，由 `GemmConfig` 的三个轴参数化：
- `elem`：
  - `Fp8`：缩放 16x16x128；
  - `Bf16`：每块每 K tile 两条 bf16 16x16x32，A / B 用同一 K 置换的 `var_slice4` 半片；
  - `Fp4`：两条 fp4×fp4 缩放 16x16x128；
  - `Fp8Bs`：零 C 的块乘积 + tile 末 `acc += t·sa·sb`。
- `scales`：`Mx128` / `RowCol` / `None` / `Mx32` / `Block128`。
- `split_k`：grid z = split，fp32 部分和直接写出，`gemm_splitk_reduce_s<S>[_rc]` 归约，行 / 列 scale 在归约里乘。

另有 CTA 256 / 128 两档。

**对照**：同进程、同数据，用 [`vendor_gemm_host.cpp`](../tests/gpu/vendor_gemm_host.cpp)。
- 容差 = bf16 舍入 + fp32 求和最坏界。
- 所有行 `bad=0`。

**数据**：MI355X GPU 5，2026-11-08，TF/s。

| kernel | 形状 | ours | 厂商 `.co` | 比 |
|---|---|--:|--:|--:|
| `gemm_a8w8_rowcol_fp8`（256） | 4096²×8192 / 8192³ | 2441 / 2562 | i8gemm 128x128：1505 / 1685 | 1.62 / 1.52 |
| `gemm_a8w8_rowcol_fp8_128` | 1024×4096×8192 / 512×7168×4096 | 1376 / 1210 | 1332 / 1154 | 1.03 / 1.05 |
| `gemm_a16w16_bf16`（256） | 4096³ / 8192³ | 1250 / 1459（行对单元后） | bf16gemm 256 bpreshuffle：1299 / 1568 | 0.96 / 0.93 |
| `gemm_a16w16_bf16_128` | 1024×4096×8192 / 512×7168×4096 | 753 / 643 | 562 / 451 | 1.34 / 1.42 |
| `gemm_a8w8_blockscale`（128） | 4096³ / 8192³ / 2048×7168² | 1168 / 1479 / 1163 | fp8gemm_blockscale 128：1125 / 1443 / 1165 | 1.04 / 1.03 / 1.00 |
| 同上 | 1024×4096×8192 / 256×7168×4096 | 865 / 512（S=2） | 1111 / 577 | 0.78 / 0.89 |
| blockscale split-K S=4 | 128×4096×8192 / 64×8192² | 317 / 317 | 203 / 202 | 1.56 / 1.57 |
| rowcol split-K S=4 | 128×4096×8192 / 64×8192² | 458 / 398 | i8：245 / 246 | 1.87 / 1.62 |
| bf16 split-K S=4 | 128×4096×8192 / 64×8192² | 293 / 265 | bf16gemm 256：74 / 74 | 3.9 / 3.6 |
| `gemm_a4w4_mxfp4`（scale 重排版，2026-11-08） | 4096³ / 8192³ / 16384³ | 3020 / 3876 / 4384 | f4gemm 256 bpreshuffle：3675 / 4720 / — | 0.82 / 0.83 / — |

**说明与待办**：
- **split-K 的取值**：
  - M ≤ 128 时 split-K 明显占优，S = 4–8；
  - M = 256 时 S = 2；
  - M ≥ 1024 时不拆。
  - 厂商对照未开 split-K（其 CSV 的 splitK 行可开，待补同条件对比）。
- **blockscale 的中等 M 差距**：1024 / 256 行 0.78 / 0.89。原因是每 CU 一个 CTA 时，块乘积的 FMA 与下一 tile 的 MMA 不能重叠。方向：块乘积延后一个 tile。
- **MXFP4**：0.6× 的结构性原因是 fp4 下 128 字节 K tile 的 MFMA 工作量只有 fp8 的一半，而每 tile 的开销不变。方向：256 字节 K tile。达到 ≥ 0.9× 之前保持"实验"标注。
- **i8**：等待 i32 → f32 转换（§6.4 P1-c）。

**批次 1 性能待办**（覆盖完成后回来做）：
- blockscale 中等 M：块乘积延后一个 tile；
- MXFP4：256 字节 K tile；
- i8；
- 与厂商同条件的 split-K 对比。

**blockscale 中等 M 的分发规则**（2026-11-08，同次运行，对照 aiter 的 BpreShuffle_128x128）：
- 规则：CTA 数 M·N/128² ≤ 约 2×CU 数时，改用 `gemm_a8w8_blockscale_splitk`，取 S = 2，再加 `gemm_splitk_reduce_s2`。规则写在 [`dispatch.rs`](../crates/amdgpu-kernels/src/dispatch.rs) 的 `blockscale_split_k` 中。
- 实测：

  | 形状 | 不切分 | S = 2 | 厂商 | 比（S = 2） |
  |---|--:|--:|--:|--:|
  | 1024×4096×8192 | 864 | 975 | 1133 | 0.86 |
  | 256×7168×4096 | 384 | 516 | 574 | 0.90 |
  | 512×7168×7168 | 761 | 840 | 997 | 0.84 |
  | 4096³ | 1159 | 不切分 | 1131 | 1.02 |

- S = 4 在所有形状上都没有赢过。
- 块缩放的 FMA 已经推迟一级（`bs_fma`）。余下的差距在每 CU 只有 1 个 CTA 时的单 tile 效率。
**MXFP4 的 scale 重排**（2026-11-08）：
- 敏感性实验：去掉逐 tile 的 1×32 E8M0 scale 读取，8192³ 从 389 µs 降到 269 µs，占 31%。原因是每个 wave 每个 tile 要发 16 条非合并的 8 字节读，各行相距 256 字节。相比之下，B 的重载占 41 µs，A 的重载占 5 µs。
- **ABI 变化**：
  - `gemm_a4w4_mxfp4`（含 `_128`）的 A / B scale 布局改为 `[rows/16][KT/2][64 lane][4 字节]`。
  - lane 16g+r 依次存放偶数 tile 的 K 组 g、4+g，再存放奇数 tile 的同两组；要求 K 是 512 的倍数。
  - host 端重排参考 [`vendor_gemm_host.cpp`](../tests/gpu/vendor_gemm_host.cpp) 中 `ours_v4_f4` 的 `shuf`。
  - 这样每个 lane 对每个块、每对 tile 只做一次合并的 dword 读，安装 scale 只需一次移位。
- **效果**：
  - 8192³：390 → 284 µs（2819 → 3876 TF/s）；
  - 4096³：59 → 45.5 µs；
  - 16384³：3188 → 2006 µs。
  - 对 aiter f4gemm 从 0.60 / 0.63× 提到 0.83 / 0.82×。
- **试过但未采纳**：寄存器 B，即 B 从 global 原地重载进 AGPR，依赖后端的 in-place reload。它只带来 +1% / 0，已从树中移除，保留为 tag `archive/mxfp4-regb`。将来若做 256 字节的 A tile，可从这个 tag 起步，因为它腾出了 B 的 ring。
**批量 GEMM（覆盖补齐）**：`gemm_a16w16_bf16_bmm`、`gemm_a8w8_rowcol_fp8_bmm`、`gemm_mxfp8_bs_v4_bmm`（`GemmConfig.batched`）。
- 网格 z = batch（kernarg 36）。各 batch 的 A、B、C 与 scale 都是稠密块，基址用 64 位 `ptr_mad` 移动；契约中每个张量乘以 BT。
- 测试用 [`vendor_gemm_host.cpp`](../tests/gpu/vendor_gemm_host.cpp) 的 `BATCH=n`：最后一个 batch 放参考数据，其余填充垃圾，检查最后一个 batch 的输出。
- 结果全部 bad=0，例如：
  - bf16 1024³ ×32：942 TF/s；
  - fp8 rowcol 256×4096×4096 ×32：1850 TF/s。
- 厂商 `opus bmm_a8w8_mxscale` 只有源码（csrc/opus_gemm），没有预编译 .co，所以没有做同次对比。
- mxfp8 v4 的 bmm 只做了 strict 校验，因为 host 里还没有 v4 mxfp8 的 family。

**per-tensor fp8 GEMM（覆盖补齐）**：`gemm_a8w8_pertensor_fp8[_128]`，即 `GemmConfig.per_tensor` 配合 RowCol。
- `ptr_sfa` / `ptr_sfb` 各指向一个 fp32，在 epilogue 中广播。
- 速度与 rowcol 相同：4096³ 为 2218 TF/s（rowcol 2225），8192³ 为 2806 TF/s；bad=0。
- aiter 没有 per-tensor fp8 的 ASM GEMM，所以没有厂商对照。

**bf16 GEMM 的 s_nop**：每个块要做两次 16x16x32 bf16 MFMA（K 的两半），第二次依赖第一次的结果。
- 原来每个调度单元只包含一行的 4 个块，依赖距离只有 4 条 MFMA，因此整个 kernel 有 120 个 s_nop。
- 改为一个单元包含两行（8 个块），依赖距离变为 8，s_nop 降到 54。
- 4096³ 从 1202 升到 1250 TF/s，8192³ 从 1432 升到 1459，2048×7168² 从 1214 升到 1269；对厂商的比值分别为 0.96、0.93、0.96。
- `gemm_a16w16_bf16` 的 md5 因此变为 8ab7cd9e。
- 调整 pre/mid sync 的 MMA 数量：差别在噪声内，未改。

**decode16 的 fused 形式（`pa_decode16_bf16_hd128_p<P>_f`；现由 `dispatch::decode16_plan` 选择，见 §7.5）**：
- 适用于 S = 1。CTA 内 4 个 wave 的归一化 partial（O 和 LSE）写到各自已不再使用的 K/V 区域（`LdsPlan::tile_over`），barrier 之后在 LDS 中合并，直接写出 bf16 O。不再需要 partial，也不再启动 combine kernel。
- 注意：overlay 的约定是每个 wave 只写自己的存储。第一版把 LSE 写进了 wave 0 的 V 区域，形成跨 wave 竞争；verifier 信任 overlay 的声明，没有发现这个问题。
- 对比（µs）：

| 形状 | fused | split S1 + combine | aiter |
|---|---|---|---|
| B64 ctx2048 | 84.9 | 95.6 | 77.6 |
| B128 ctx1024 | 84.0 | 101.6 | 75.4 |
| B256 ctx512 | 88.7 | 116.6 | 83.4 |
| B32 ctx4096 | 93.2 | 99.8 | 75.0 |

- 也就是大 batch 时达到厂商的 0.80–0.94×。
- 小 batch 仍需 S ≥ 2，例如 B16 S2 为 28.6 µs，厂商 18.4 µs；其中 combine 的 launch 占 4–10 µs。

**decode16 小 batch 的实验（均未采纳，已归档）**：
- `archive/decode16-lookahead`：把下一 tile 的块表查找放进循环状态，没有收益。
- `archive/decode16-prefetch`（`_pf` / `_fpf`）：每个 wave 两个 K/V slot（32 KB），tile t+1 的 DMA 在 tile t 计算之前发出，每 CU 1 个 CTA。
  - B16 fused 为 31.1 µs（未改前 32.7），但 split S2 不加 prefetch 仍有 28.6 µs，更快。
  - 大 batch 时因为占用率减半，反而变慢：B64 为 94.6 µs，未改前 84.9。
  - 因此 dispatch 中没有哪个形状会选它。
- sched 的 rocprofv3 计数见 batch2 分支上的 pa-decode16-profile 文档：
  - 厂商把 160 KB LDS 全部用上，K 直接加载进寄存器；
  - 我们的 LDS bank conflict 是厂商的 23–27×，SALU 是 10×；
  - 我们 SQ_WAIT_ANY 占比为 54–57%，厂商为 13–26%。
- 按 Little 定律估算，B16 要跑满 HBM，每个 wave 大约需要 21 KB 数据在途，而我们只有一个 16 KB 的 tile 在途。
- `archive/decode16-v2`：K 直接加载到 VGPR（8 次 b128），V 放进 3 个 slot 的 LDS ring（chunk 按 key % 16 做 XOR），页表项提前一个 tile 读出，循环按 3 展开。每个 wave 有 2 个 tile 在途，结果 bad=0。
  - 第一版里页表的 b64 load 不能原地写回，产生了 vmcnt(0)；改成 b128 后修复。
  - 但没有收益：B16 fused 为 32.5 µs（基线 32.7），B64 fused 为 91.9（基线 84.9），B8 S4 为 49.2（基线 47.0）。
  - 结论：加深在途数据不是瓶颈。每 tile 约 640 条指令，每个 SIMD 只有 1 个 wave，依赖延迟全部暴露，折合每 tile 约 1.6–1.9 µs。
- **削减指令数（已采纳）**：
  - 以 sched 的 ATT 测量为准：我们每个 key 133 周期、18.6 条指令，厂商为 62 周期、5.9 条；每个 tile 有 41 个 s_nop。
  - 合入了 sched 的页组查找：每个页组只做一次 wave 内统一的块表读取，并抽出 `fmha::paged` 辅助模块；同时合入 V 的 granule swizzle，使 bank conflict 降低 85–100%。
  - 本分支另做两处修改：
    - key 截断只用 min：`.max(0)` 会被降为 compare + select 加一个 s_nop。
    - key mask 只施加在需要的那个 tile 上：用 uniform 分支，分支内只有 VALU，原地修改。
  - 同次测量（µs，前为修改后，后为修改前）：
    - B16 ctx2048：S2 为 27.9（28.6），`_f` 为 29.3（32.7）
    - B64 `_f`：83.2（84.9）
    - B128 ctx1024 `_f`：81.7（84.0）
    - B8 ctx8192 S4：45.6（47.0）
    - 厂商分别为 18.4 / 77.3 / 75.2 / 42.7。
  - 测试：ctx 为 17、777、1337、2000，以及 P=64 时全部 bad=0；strict 无错误。split decode16 的 md5 由 d032e715 变为 9e212f51，`_f` 变为 a4c7902c。
  - 第二轮（已采纳）：分组路径中，每个 DMA 对应 lane 的页内行偏移和 K/V 源 swizzle 都是循环不变量，每个 tile 只需计算组的页基址，以及对越界行做一次 min 封顶（停在 k_hi−1 所在页内不超过它的行，均为真实数据，随后会被 mask）。
    - 循环体从 577 行降到 451 行，v_mul_lo 从 24 条降到 2 条。
    - B16 `_f` 为 28.7 µs（上一轮 29.3），S2 为 27.5（27.9）；B8 S4 为 45.2（45.6）。
  - `archive/decode16-f8`：每个 CTA 8 个 wave（每个 SIMD 2 个）的 fused 形式，全面变慢：B16 为 30.6 µs（`_f` 28.6），B64 为 93.4（85.3），B128 为 98.4（81.8）。说明增加每个 SIMD 的 wave 数并不能隐藏剩下的延迟。
  - 剩余差距约 1.5×（B16）。每个 tile 仍有约 450 条指令，而厂商约 190 条。需要按厂商的结构重写：K 进 AGPR 并加载 64 个 key，按块做 softmax。
  - 合入 Main 上 backend 的 lower-peep（min/max/med3 与 M0 外提）之后：B16 S2 为 27.2 µs（厂商 18.5，1.47×）。
    - 只看 decode kernel（NOCOMB）：S2 约 20.1 µs，`_k64` S2 为 19.9 µs，均接近厂商的 18.5 µs。
    - 差距主要来自 combine 的 launch（约 7 µs）。
  - `archive/decode16-lastc`（`_c`）：在 split kernel 内用 last-arriver 做合并（per-group ticket，结果经 LDS 广播，由最后一个 CTA 合并），不再启动 combine kernel。
    - 结果正确，strict 通过。
    - 但极慢：full fence 版本在 B16 S1 为 128 µs；lite 版本（sc1 读写）同样约 100 µs 量级。原因未查明，可能是 sc1 / atomic 路径或 wbl2 的成本。
  - 按照 lead 设的门槛（B16 ≤1.2×），decode 到此暂停；剩余的可做项是对 combine launch 的消除做 ATT/计数分析。

**MLA prefill（hd192/hd128）**：`fmha_fwd_hd192_hd128_bf16_v3[_causal]`，即 `fmha_fwd_bf16_v3_d(.., 192)`，使用厂商 fwd_hd192_hd128 的 ABI（qk_head_dim 192）。
- Q/K 的 head dim 为 192（128 nope + 64 rope），V/O 为 128。
- K 的 rope 部分放在第二个 ring（KR，行宽 128 字节，chunk 按 key % 8 做 XOR），由 group A 负责 staging（每个 wave 多 4 次 DMA）；QK 共 12 个 k-chunk。
- LDS 正好 160 KB（K 64 + KR 32 + V 64）。
- causal 形式不做 q tile 配对：第二个 pass 的寄存器放不下；两种形式都用 Vs128。
- 测试用 `attn_host` 的 `DV=128`，第 10 个参数为 192。全部 bad=0。
- 与 aiter `fwd_hd192_hd128_bf16[_causal]` 的同次对比（厂商用 128 行、256 线程）：

| 形式 | 形状 | ours | 厂商 | 比值 |
|---|---|---|---|---|
| 非 causal | 8192² H16 | 650 µs | 617 µs | 0.95× |
| 非 causal | 4096² H128 | 1311 µs | 1244 µs | 0.95× |
| 非 causal | 2048² B2 | 112 µs | 96 µs | 0.86× |
| 非 causal | 1024²（64 CTA） | 39.7 µs | 20.3 µs | 0.51× |
| causal | 8192² | 512 µs | 375 µs | 0.73× |
| causal | 4096² H128 | 1058 µs | 777 µs | 0.73× |

- 待办：causal 的负载均衡（配对 / 128 行 tile）以及小网格。

**hd192 的 key-split（`fmha_fwd_hd192_hd128_bf16_v3[_causal]_ks`）**：
- causal 2048² 时 KS=1 为 73 µs，不切分为 89 µs，有改善。
- 非 causal 1024² 没有改善（41 µs 对 40 µs）。
- 小网格的差距是结构性的：我们用 256 行 tile、单个 CTA 每 tile 约 3.8 µs 并带有固定开销，而厂商用 128 行、256 线程，需要另做 4 wave 的 128 行形式。

**attention sinks 与 MTP decode**：
- **sinks**：`attn_combine_bf16_sink_hd{128,64}`，ptr_sink 在 @80，为每个 head 一个 f32。
  - sink logit 只进入 softmax 的分母：M = max(maxₛ lseₛ, sink)，den += e^(sink−M)。
  - 因此任何 split-K decode（decode16 / fp8 / MTP）都能直接带上 sinks。
- **MTP**：`pa_decode16_bf16_hd128_p<P>_mtp`，qlen 在 @240，范围 1..16，要求 G·qlen ≤ 16。
  - 16 个 MFMA 列是（查询位置，head）对；查询位置 qpos 只看到 key < ctx − qlen + 1 + qpos，即新 token 之间是因果的。
  - partial 和输出按行 b·qlen + qpos 组织；combine 的网格为 (H, B·qlen)。
  - 非 MTP 的代码保持不变（md5 d032e715）。
- 测试用 `pa_host` 的 `QLEN=q` / `SINK=1`：qlen 为 1、2、3、4，以及 qlen 3 加 sinks，全部 bad=0。
- 性能：B64 ctx2048 HK16 时，qlen 1 / 2 / 4 分别为 179 / 190 / 207 µs，KV 只读一次。
- 与厂商对比，aiter 为 `pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1`，参数由 [asm_pa.cu](https://github.com/ROCm/aiter/blob/main/csrc/py_itfs_cu/asm_pa.cu) 给出：mtp = qlen−1，ptr_QTP 为 qo_indptr，在 @256。厂商结果 bad=0。
- qlen=2，ours 取最佳 S：
  - B16 ctx2048：30.4 µs，厂商 19.1 µs（0.63×）
  - B64：105 µs，厂商 80 µs（0.76×）
  - B4 ctx8192：37.9 µs，厂商 37.8 µs（1.0×）
  - B128 ctx1024：115 µs，厂商 79 µs（0.68×）
- 差距与 decode16 本身相同，属于 decode 性能待办。

**fp8 KV 的 paged decode（per-tensor）**：`pa_decode16_fp8_hd128_p<P>`，即 `pa_decode16_x(.., fp8 = true)`。
- cache 为 fp8 e4m3 [pages, P, HK, 128]，即 `reshape_cache_flash_fp8` 的布局；k_scale 在 @208，v_scale 在 @224。
- K tile 为 fp8 行（chunk 按 key % 8 做 XOR）。QK 的每个操作数读 8 字节，用 `ScalePkBf16Fp8` 转成 8 个 bf16（精确）。
- V 先把 fp8 行暂存，再转换进 bf16 V tile，供转置读使用。
- k_scale 并入 softmax scale，v_scale 并入输出。
- 测试用 [`pa_host`](../tests/gpu/pa_host.cpp) 的 `FP8KV=1`：B4、B16、B64、P256 全部 bad=0。
- 性能：B64 ctx2048 S1 为 71.6 µs（bf16 为 96 µs，快 1.34×）；B16 ctx4096 最佳 S4 为 49 µs（bf16 46 µs）。小 batch 时转换和延迟是瓶颈。
- **per-token scale（分块布局）**：`pa_decode16_fp8tok_hd128_b<BS>`，读的是 `reshape_cache_blocked_fp8` 写出的布局。
  - 布局：K 为 [NB][HK][D/16][BS][16]，V 为 [NB][HK][D][BS]，scale 为 [HK][S] 的 f32 指针，k_scales 在 @208，v_scales 在 @224。
  - K 行按块 gather，进入与 flash 形式相同的 fp8 K tile。
  - V 以 d-major 方式到达（即 V^T），转换成 bf16 V^T tile 后用普通 8 字节读。
  - 每个 key 的 scale：k 乘在 S 上，v 乘在 P 上（在行和之后）。key 的 slot 由一次块表读取加 ds_bpermute 得到。
  - 测试用 `FP8KV=2`，BS=16 和 BS=64 全部 bad=0。
  - 性能：B64 ctx2048 为 88 µs，B16 ctx4096 S4 为 59 µs。
  - 与 aiter `pa_bf16_pertokenFp8_gqa8_2tg_4w[_hp|_uhp]` 对比。参数按 asm_pa.cu 设置：K/V 为 x=16 的分块布局，与 sched 的布局相同；scale 为 [NB][HK][BS]。
    - 厂商的结果至今未与我们的 CPU 参考对上：即使 scale 全为 1，maxerr 仍为 2.1，原因待查（可能是厂商内部对 Q 做了量化，或某个布局细节不同）。
    - 因此下面只是计时，正确性未经核实：B4 ctx2000 厂商 11.6 µs，ours 25.6；B16 ctx4096 为 21.6 对 59；B64 ctx2048 为 35 对 88。

**A16W4（覆盖补齐，忠实语义）**：`gemm_a16w4_mxfp4`，即 `Elem::Bf16W4`，64×128 CTA。
- A 为 bf16，精确保留；B 为 MXFP4，在主循环中反量化为 bf16，再走 bf16 MFMA。
- 一个 K tile = 256 元素：每个 slot 放 4 份 128 字节的 A 镜像（K 64）。
- B 的 lane 片段第 d 个 dword 先移位，再用 `v_cvt_scalef32_pk_f32_fp4` 转 f32，乘组 scale 2^(e−127) 后打包成 bf16。
- host 在每个 256-K tile 内对 B 做 K 置换：物理位置 p 对应逻辑 K`(p) = 128(p/128) + 16((p/64)%2) + 8((p/32)%2) + 32((p/8)%4) + p%8`，B scale 为不重排的 [N][K/32]。
  - 片段第 d 个 dword 对应 tile 的第 d 个 K 组。
  - 置换和分组都用 host 的 `W4DBG=3` / `W4DBG=6` 探针实测得到。
- 结果 bad=0，max_rel 3.9e-3，等于 bf16 输出的舍入。
- 性能：1024³ 192 TF/s，4096³ 495，8192³ 532（反量化的 VALU 是瓶颈；bf16 GEMM 为 1250）。256×7168×2048 为 362 TF/s，而同形状 bf16 GEMM 只有 197 TF/s。
- aiter 中没有 a16w4 的 .co，无法做厂商对照。
- 另一条较快的路线是 mxquant 量化 A 再走 a8w4。它的语义不同、精度更低，精度差尚未测量。

**A8W4（覆盖补齐）**：`gemm_a8w4_mxfp_128`，即 `Elem::Fp8W4`，A 为 MXFP8，B 为 MXFP4，两者都用重排的 1×32 scale。
- 一个 K tile = 256 元素：B tile 128 字节；A 分成 lo / hi 两半，作为同一 ring slot 中的两份 A 镜像（DMA 多一轮，A 前进 2·key0 字节）。
- 每个 m 块有两个 A 寄存器片段；每个块做两次 fp8×fp4 的 scaled `16x16x128` MFMA，scale 字节取 h。
- 测试用 host 的 `ours_v4_f8f4`。结果 bad=0：1024³ 301 TF/s，4096³ 1501，8192³ 1856，256×7168×2048 675 TF/s。
- 128×128 CTA 版本的寄存器有余量；256 版本加上第二组 A 片段会超出 512 个寄存器。

**MX 量化（覆盖补齐）**：见 [`rows/mxquant.rs`](../crates/amdgpu-kernels/src/rows/mxquant.rs)，CPU 校验用 [`mxquant_host.cpp`](../tests/gpu/mxquant_host.cpp)。
- `quant_mxfp4_bf16_k<K>`：bf16 → fp4（e2m1）加 1×32 的 E8M0 scale，scale 直接写成 `gemm_a4w4_mxfp4` 的重排布局。
  - scale 规则与 aiter `dynamic_mxfp4_quant` 相同：amax 加 0x200000 后截断，取指数再减 2。
  - 4096×7168：17.9 µs，4140 GB/s，码值与 scale 全部一致。
- `quant_mxfp8_bf16_k<K>`：bf16 → fp8 e4m3 加 1×128 的 E8M0 scale，布局为 [K/128][M]（`gemm_mxfp8_bs_v4` 的 A scale）。
  - scale 规则：ceil(log2(amax / 448))，不饱和。
  - 每 16 个 lane 沿 K 处理一组，用 DPP 求 amax。
  - 4096×7168：15.7 µs，5643 GB/s，全部一致。
- fp4 的 `v_cvt_scalef32_pk_fp4_f32` 按字节选择写入的功能不可用，已报告后端；目前改为先写 byte 0 再移位。

### 7.5 批次 3（attention）状态：覆盖完成，性能待优化

**prefill**：`fmha_fwd_hd<64|128|256>_bf16[_causal][_swa][_group]`，见 [`fmha/bf16.rs`](../crates/amdgpu-kernels/src/fmha/bf16.rs)。
- **ABI**：aiter fmha_v3 的 656 字节 ABI；滑窗参数为我们的扩展，放在 656 / 672。
- **方向**：S^T = K·Q^T，O^T = V^T·P^T，用 bf16 32x32x16。每 lane 一个 query，所以在线 softmax 是逐 lane 标量，两半 lane 用 `v_permlane32_swap` 合并。
- **数据流**：
  - Q^T 常驻寄存器；
  - 64 key 的 K / V tile 用 2 槽 LDS 环，K 做 XOR swizzle；
  - V^T 用两条 `ds_read_b64_tr_b16` 读出：源 4 行块放在 key 8r + 4h，使转置后的 k 序等于 P 的打包序。
- **覆盖**：GQA、causal（右下对齐）、滑窗、varlen（group ABI）、LSE；hd256 的 O^T 放 AGPR。

**paged decode**：`pa_decode_bf16_hd128_p<P>` 加 `attn_combine_bf16_hd128`，见 [`fmha/decode.rs`](../crates/amdgpu-kernels/src/fmha/decode.rs)。
- **布局**：NHD 页，加 block table 和上下文长度。
- **分工**：一个 KV head 的 GQA 头是一个 wave 的 32 行；每个 wave 是一个 key split，经私有 32 KB LDS 流式处理，每行 key 都经 block table 翻译。
- **输出**：部分 O / l 与 LSE，再由 combine 合并。

**MLA decode**：`mla_decode_bf16_p<P>` 加 `attn_combine_bf16_hd512[_sm]`，见 [`fmha/mla.rs`](../crates/amdgpu-kernels/src/fmha/mla.rs)。
- **形状**：576 / 512，单个 latent 头，每 CTA 32 头。
- **QK**：维度按 wave 四分，S 经 LDS 归约。
- **PV**：v 维按 wave 四分。
- **其他**：key split 加 combine；`_sm` 变体读 aiter stage-1 的 [B][S][H] 布局。

**对照工具**：[`attn_host.cpp`](../tests/gpu/attn_host.cpp)、[`pa_host.cpp`](../tests/gpu/pa_host.cpp)、[`mla_host.cpp`](../tests/gpu/mla_host.cpp)。三者都在同进程、同逻辑数据上运行；PA 的 host 同时构造 aiter 的 PA 布局。

**数据**：2026-11-08，GPU 5。全部 `bad=0`（CPU 参考，含 LSE），strict 0 错误。

| 算子 | 形状 | ours | 厂商 | 比 |
|---|---|--:|--:|--:|
| prefill hd128 | B1 H8 4096² | 440 TF/s | fwd_hd128_bf16：688 | 0.64 |
| prefill hd128 causal varlen | B4 H8 ≤2000 | 87 TF/s（tr_b16 之前） | 276 | 0.32 |
| paged decode P16 | B16 H64 HK8 ctx ≤4096 | 2246 GB/s | pa_bf16_noquant_gqa8：5997 | 0.37 |
| MLA decode P1 | B32 H16 ctx ≤2048 | 1019 GB/s | mla_dec_stage1 gqa16（+我们的 combine）：2227 | 0.46 |
| MLA decode P1 | B128 H16 ctx ≤4096 | 994 GB/s | 4094 | 0.24 |

**限制与观察**：
- 厂商 bf16 fmha_v3 没有 hd64、滑窗；hd256 只有 group 模式。
- 厂商 MLA stage-1 在页大小 16 下与参考不符，只在页大小 1 下对拍通过。可能需要它的 metadata 约定 `[待核]`。

**性能进展**（2026-11-08，同次运行，逐项 A/B，只采纳实测收益）：
- **prefill `fmha_fwd_hd128_bf16_v3[_causal]`**，见 [`fmha/bf16v3.rs`](../crates/amdgpu-kernels/src/fmha/bf16v3.rs)。
  - 结构：v3 栈，即 `tile::Pipeline` 的 `TwoPhaseOffset`、两组各 4 wave、每 CTA 256 行、128 key 的 tile。
  - 采纳的改动及收益：
    - QK 改为互相独立的累加链：+21%；
    - 片段读取提前一组、地址基址外提：+43%；
    - setprio ping-pong：+3.8%；
    - v_max3：+0.8%；
    - V 做 XOR swizzle：+2%；
    - 后端的 `load_lds_tr_b16x2` / `pack4`（v_mov 从 1396 降到 478）：+6–7%；
    - causal 采用厂商的 q tile 配对：8192² 从 619 提到 955 TF/s。
  - 试过但未采纳：XCD 顺序（无增益）；调度选项 ldsink / wsink（无增益）。
  - 对 fmha_v3 的比值：

    | 形状 | 非 causal | causal |
    |---|--:|--:|
    | 8192² B2H32 | 0.90 | 0.89 |
    | 2048² B4H16 | 0.84 | 0.82 |
    | 1024² B8H16 | 0.83 | — |
    | 4096² B1H8 | 0.76 | 0.68 |

    4096² B1H8 的网格只有 64–128 个 CTA，所以比值最低。
- **bf16 prefill 第二轮**（同次运行）：
  - 非 causal 改用 `_vs128` 编译策略，加上 ldsink / wsink / ldahead32：+6.6–7.7%。
  - causal 留在 Gemm 策略（换成 Vs128 测得 −4%），但同样加上调度选项：+1–2%。
  - `_ks` 形式留在 Gemm 策略：它的 Vs128 版本在 causal S=4 下硬件出错，已报告后端。
  - 对 fmha_v3 的比值：

    | 形状 | 非 causal | causal |
    |---|--:|--:|
    | 8192² B2H32 | 0.95 | 0.90 |
    | 2048² B4H16 | 0.91 | 0.84 |
    | 1024² B8H16 | 0.88 | — |
    | 4096² B1H8 | 0.94（`_ks` S2） | 0.94（`_ks` S4） |
    | 512² B16H32 | 0.84 | — |

  - 分发规则见 [`dispatch.rs`](../crates/amdgpu-kernels/src/dispatch.rs) 的 `bf16_prefill_kernel`：
    - 极小的 causal 网格（配对后不足 64 个 CTA）仍用 v1 causal。例如 B2H8 1000×1300：v1 123 TF/s，v3 最好 94 TF/s，厂商 115 TF/s。
    - hd64 / hd256、滑窗和 varlen 仍走 v1。
- **paged decode `pa_decode16_bf16_hd128_p<P>`**，见 [`fmha/decode16.rs`](../crates/amdgpu-kernels/src/fmha/decode16.rs)。
  - 用于 G ≤ 16：每个 wave 改用 16x16x32 MFMA，消除 32 行形式下 3/4 的空算。
  - 速度从 0.51× 提到 0.62×。
  - 敏感性实验：去掉 PV 只快 6%；去掉 DMA 后时间减半。瓶颈是访存延迟没有被隐藏：每 CU 只有 4 个 wave，且每个 wave 内 DMA 与计算串行。
  - 于是加入 32 key 的 tile（每个 wave 的私有区 16 KB，每 CU 可驻留 2 个 CTA），与 64 key 版（`_k64`）并存，当时由 `decode16_config` 按 B·HK 选择 tile 大小和切分数 S（v9 起由 [`dispatch.rs`](../crates/amdgpu-kernels/src/dispatch.rs) 的 `decode16_plan` 取代，见下文 last-arriver 一节）。
  - 实测：

    | 形状（B / H / HK / ctx） | 配置 | ours | 厂商 | 比 |
    |---|---|--:|--:|--:|
    | 16 / 64 / 8 / 4096 | 32 key，S=4 | 4056 GB/s | 6018 | 0.67× |
    | 64 / 64 / 8 / 2048 | 32 key，S=1 | 4182 GB/s | 5392 | 0.78× |
    | 4 / 32 / 4 / 8192 | 64 key，S=8 | 1352 GB/s | 1303 | 1.04× |
  - block-table 查表批量化：0.37× → 0.51×。
  - **last-arriver 合并（`pa_decode16_bf16_hd128_p<P>_c`，v9）**：split 形式的 CTA 用 sc1 写穿存部分结果并 lite release，然后由一个 lane 取 (b, hk) 的 ticket（返回式 `buffer_atomic_add`，其余 lane 的偏移越界、在 L2 之前被丢弃），最后到达的 CTA acquire 后合并 S 份部分结果、直接写 bf16 输出——不需要 combine kernel。
    - 第一版（`archive/decode16-lastc`）慢约 100 µs：ticket 原子由 CTA 的全部 256 个 lane 执行（非 0 lane 加 0，仍是 L2 原子），计数器相邻，B16 时约 3.2 万个同 cache line 原子串行化。改成单 lane 后：B16 ctx 2048 S1 128 → 30.3 µs，S4 353 → 26.3 µs。fence 本身只降为等待，不是瓶颈。
    - 分发：[`dispatch.rs`](../crates/amdgpu-kernels/src/dispatch.rs) 的 `decode16_plan(B, HK, ctx)`。n = B·HK ≥ 256 时用 fused；更小的网格用 `_c`，S 取使 n·S ≥ 256 的最小 2 的幂，并限制每份至少 512 个 key（S ≤ ctx / 512）。
    - 实测（H 64，HK 8，页 16，µs，同一次运行；旧规则 = split + combine kernel，n < 64 用 `_k64`；比 = 厂商 / ours）：

      | B | ctx 1024 | ctx 2048 | ctx 8192 |
      |---|---|---|---|
      | 4 | `_c` S2 11.9（旧 18.6，厂商 8.0，0.67×） | `_c` S4 15.7（22.4，14.3，0.91×） | `_c` S8 28.8（33.0，35.7，**1.24×**） |
      | 8 | `_c` S2 12.0（25.7，8.1，0.68×） | `_c` S4 19.6（31.5，14.8，0.75×） | `_c` S4 40.5（53.0，42.8，**1.06×**） |
      | 16 | `_c` S2 15.8（24.0，11.0，0.70×） | `_c` S2 24.1（31.9，18.5，0.77×） | `_c` S2 89.8（98.6，88.3，0.98×） |
      | 32 | fused 20.4（26.7，17.0，0.83×） | fused 35.8（42.1，31.2，0.87×） | fused 166.6（175.6，149.6，0.90×） |
      | 64 | fused 33.9（不变，30.6，0.90×） | fused 83.8（不变，77.4，0.92×） | fused 303.0（不变，271.6，0.90×） |
      | 128 | fused 81.9（不变，74.6，0.91×） | fused 157.8（不变，146.5，0.93×） | fused 588.3（不变，540.8，0.92×） |

    - 网格内 `_c` S1 与 split 形式在每个点上都更慢。所有点 PA_OK；`_c` 严格验证 0 错误（2 / 19 / 0）。剩余差距在小 ctx（1024）：每个 wave 的指令下限（见 [`pa-decode16-profile.md`](pa-decode16-profile.md)）。
- **MLA decode**：`mla_decode_bf16_p<P>` 已换成 16x16 版本，见 [`fmha/mla.rs`](../crates/amdgpu-kernels/src/fmha/mla.rs)。旧的 32 行版本保留为 tag `archive/mla-decode-v1`。combine 的部分数为 NS = 4·S。
  - 结构：
    - 每 CTA 负责 16 个 head，作为 16x16x32 MFMA 的 16 列；
    - 每个 wave 处理自己的 key 段，没有跨 wave 的 S 归约和 barrier；
    - 32 key 的 tile 放在 wave 私有的 36 KB 区域；
    - 行基址由每 lane 一次块表读取加 `ds_bpermute` 得到；
    - 空的 key 段跑一个全掩码 tile。
  - 对照 aiter `mla_dec_stage1` a16w16（H16、P1、B64、ctx ≤ 4096）：旧版 178 µs，新版 80 µs，厂商 47.7 µs。比值从 0.27× 提到 0.60×。
  - H128、P16：646 → 360 µs。
  - 后续改动：
    - 把 tile 拆成两个 16 key 的半块，第 0 半块的 QK 与第 1 半块的 DMA 重叠：H16 从 79 降到 76 µs，约为厂商的 0.63×。
    - H ≥ 64 时改用 head 并行形式 `mla_decode_bf16_hp_p<P>`：每 CTA 的 4 个 wave 是 4 组各 16 个 head，共享 2 槽的 36 KB KV 环，每 64 个 head 只读一次 KV。H128 B64 ctx 4096 P16：348 → 232 µs（S=2）。
    - [`dispatch.rs`](../crates/amdgpu-kernels/src/dispatch.rs) 的 `mla_config` 负责选择形式和切分数 S。
  - 期间发现两个后端 bug：
    - `ds_bpermute` 之后缺少 lgkm 等待，tileops f841f0a2 已修复；
    - if / else 中随机 lane 的存储丢失：根因是 then 路径的 Q 预取 load 在 else 路径上结果已死但仍在途，RA 把其寄存器交给 else 的临时值，晚到的 load 覆盖之；后端已用 post-RA 的在途 load 物理 WAW 守卫（`pad_inflight_waw`）修复（见 [`current-status.md`](current-status.md) §6.1）。新版此前通过去掉该分支绕开。- **MoE stage 2**：
  - bf16 原子累加：更慢（1528 vs 672 µs），且精度更差，未采纳。
  - bf16 中间结果（`_h`）：快 26%，误差属 bf16 量级。作为可选项，默认仍用 fp32 中间结果。

**性能待办**：
- prefill：8 wave 两组 ping-pong（v3 骨架）、延迟 rescale、QK / softmax / PV 交叠；
- decode：每 wave 更大的块、私有环双缓冲、页表查找提到循环外、一 tile 多页时按页 DMA；
- MLA：S 归约改为 DPP / permlane，或换成按 head 切分。

### 7.6 批次 4（fused MoE）状态：bf16、fp8 per-token、fp8 blockscale 均已覆盖

**实现**：`gemm_moe1_<bf16|fp8|fp8bs>_<silu|gelu>[_dw]`、`gemm_moe2_<bf16|fp8|fp8bs>`、`moe_sum_bf16_k<K>`，都在 [`gemm/mod.rs`](../crates/amdgpu-kernels/src/gemm/mod.rs) 的 `MoeStage` 下。它们复用 v4 GEMM 结构，消费 aiter `moe_sorting` 的输出：
- `sorted_token_ids` 编码为 `(slot << 24) | token`，padding 为 `(topk << 24) | M`；
- `sorted_expert_ids` 按 M tile 给出；
- `num_valid_ids` 之后的 tile 直接退出。

**数据流**：
- **A 行 gather**：每 lane 的 voffset 在循环外一次算好。stage 1 按 token 取行，stage 2 按 `token·topk + slot` 取行。每个 DMA issue 有自己的 M0，不用指令偏移。
- **B**：按 expert 取基址。stage 1 的 W1 由 host 把 gate / up 以 16 行块交错后再做 `shuffle_weight(16,16)`，使同一 lane 同时持有同一输出列的 gate 和 up。
- **stage 1 epilogue**：寄存器内算 `act(gate)·up`（silu 或 tanh 形式的 gelu），可选乘路由权重，输出 bf16 `[M, topk, I]`，与 aiter stage 1 布局相同。
- **stage 2**：fp32 `[M, topk, D]` 乘路由权重，再由 `moe_sum` 沿 topk 求和。

**对照**：[`moe_host.cpp`](../tests/gpu/moe_host.cpp) 在同一进程、同一数据上运行 ours 与 aiter stage 1 ASM（W1 为 aiter 的堆叠布局）。

**数据**：2026-11-08，GPU 5。全部 `bad=0`，strict 0 错误。aiter stage 1 的输出与我们的 y1 在容差内完全一致（diffs=0）。

| 形状（fp8 per-token，silu，block_m 128） | ours stage 1 | aiter `fmoe_stage1 …128x128_pf2` | 比 |
|---|--:|--:|--:|
| M1024 D4096 I1024 E8 topk2 | 38.3 µs | 40.8 µs | 1.07 |
| M4096 D7168 I512 E32 topk8 | 345.7 µs | 395.4 µs | 1.14 |
| M256 D7168 I2048 E64 topk8 | 423.3 µs | 391.8 µs | 0.93 |

blockscale 128×128（`fp8bs`）的规则：
- A 的 scale 行随 gather 取；
- B 的 scale 按 expert 取块；
- stage 1 的 gate 与 up 各有一个 scale 字，按 n 块交替使用。

对照的是 aiter `…blockscale_g1u1_128x128_pf2`。它多做一步：输出时量化成 fp8。所以下表只比时间。

| 形状（fp8 blockscale，silu） | ours stage 1 | aiter | 比 |
|---|--:|--:|--:|
| M1024 D4096 I1024 E8 topk2 | 60.6 µs | 52.5 µs | 0.87 |
| M4096 D7168 I512 E32 topk8 | 516.3 µs | 447.4 µs | 0.87 |
| M256 D7168 I2048 E64 topk8 | 527.9 µs | 418.4 µs | 0.79 |

**待办**：
- **stage 2**：现在走 fp32 中间结果再求和，例如 4096×7168×topk8 要写 940 MB，所以只有 362 TF/s。改为 bf16 原子累加或在 CTA 内按 token 归并。aiter 没有 ASM stage 2（它用 CK），需要另找对照。
- **blockscale stage 1**：0.8–0.87×。需要与批次 1 的 blockscale 中等 M 问题一起处理，方向是块乘积延后一个 tile。
- **MXFP4 MoE（a4w4，已完成）**：`gemm_moe1_fp4_<act>[_bm64]` / `gemm_moe2_fp4[_bm64|_h]`。
  - A 和 B 都是 fp4，配 1×32 E8M0 scale，scale 不重排：A 的 scale 行随 gather 取，B 的 scale 按 expert 用 ptr_mad 定位。
  - 测试用 [`moe_fp4_host.cpp`](../tests/gpu/moe_fp4_host.cpp)：stage 1 的输出在 host 端按 aiter 规则重新量化成 MXFP4 后供 stage 2。结果全部 bad=0，maxrel 3.9e-3。
  - 例如 M2048 D2048 I1024 E8 topk2：stage 1 39 µs，874 TF/s。
  - gpt-oss 形式 `_oss`：每个 expert 带 fp32 bias（stage 1 为 [E][2I]，先 gate 后 up；stage 2 为 [E][D]）。stage 1 的激活为截断的 swiglu：gate = min(g+bg, 7)，up = clamp(u+bu, −7, 7)，y = (up+1)·gate·σ(1.702·gate)。用 host 的 `OSS=1` 测试，全部 bad=0。
  - aiter 没有 fp4 MoE 的 ASM，因此没有厂商对照。
- **a16w4 MoE（gpt-oss 路径）**：`gemm_moe{1,2}_bf16w4[_oss]_bm64`。
  - A 为 bf16，精确保留；expert 权重为 MXFP4，在主循环中反量化，与 A16W4 GEMM 相同（权重带 K 置换，scale 不重排）。
  - 每个 slot 放 4 份 A 镜像，在 gather 的 A 路径上（按 bf16 行字节）；只支持 block_m 64，以容纳 LDS。
  - 测试用 `moe_fp4_host` 的 `A16=1`，stage 2 直接吃 bf16 中间结果。全部 bad=0。
  - 性能：M512 D=I=3072 E32 topk4（gpt-oss 尺寸，K 填充到 256 的倍数）+ `_oss`：a16w4 的 stage 1 为 286 µs，a4w4 为 128 µs；M4096 时分别为 1293 µs 与 409 µs。差距来自反量化的 VALU。
  - 限制：K（即 D 与 I）必须是 256 的倍数，gpt-oss 的 2880 要填充到 3072。
  - a8w4 MoE 理论上用同一套机制即可得到（`Elem::Fp8W4` 加不重排的 Mx32），但还没有 host 测试，暂未注册。
- **block_m 64（已完成）**：`gemm_moe{1,2}_*_bm64` 是 64 行 tile。[`dispatch.rs`](../crates/amdgpu-kernels/src/dispatch.rs) 的 `moe_block_m` 在每专家平均行数 M·topk/E < 128 时选用它。
  - M256 D7168 I2048 E64 topk8，stage 1：fp8 从 420 µs 降到 381 µs（aiter 392，1.03×）；fp8bs 从 530 µs 降到 417 µs（aiter 418，1.0×）。
  - 同一形状的 stage 2 + sum：fp8 251 → 222 µs，fp8bs 262 → 213 µs。
  - M128 E8 topk2 fp8bs：53.8 → 38.0 µs。
- **blockscale 块乘积改用 `v_pk_fma_f32`**：试过，结果持平到略慢（GEMM 4096³：1152 对 1173 TF/s），未采用。
- **stage 1 → stage 2 的再量化**：已在 GPU 上完成，fp8 用 `quant_token_bf16_fp8_n<I>`，blockscale 用 `quant_group128_bf16_fp8_n<I>`（均为 sched 的 rows::quant）。host 测试计入这一步的时间，参考实现读回 GPU 量化后的码值和 scale。

**后端问题（已报告）**：一个结果未被使用的 load，其目标 VGPR 在 load 仍在途时被重新分配，load 晚到后覆盖新值（WAW）。表现为冷启动时错、热运行时对。已用"不发出死 load"绕开。

### 7.7 批次 5a（attention 反向）状态：覆盖完成，性能约为厂商的 0.37×

**实现**：[`fmha/bwd.rs`](../crates/amdgpu-kernels/src/fmha/bwd.rs)，按 FA2 拆成三个 kernel，全程不用原子操作。
- `fmha_bwd_dot_hd128_bf16`：计算 Dot = Σ dO∘O。
- `fmha_bwd_dq_hd128_bf16[_causal]`：前向方向，每 lane 一个 query。
  - 依次算 S^T、P^T（用前向给出的 LSE）、dP^T = V·dO^T、dS^T、dQ^T += K^T·dS^T。
  - K^T 用 `ds_read_b64_tr_b16` 从带 swizzle 的 K tile 读出。
- `fmha_bwd_dkdv_hd128_bf16[_causal]`：每 lane 一个 key。
  - 依次算 S = Q·K^T、dP = dO·V^T、dV^T += dO^T·P、dK^T += Q^T·dS。
  - dO^T / Q^T 用 tr_b16 读；dK / dV 的累加器放在 AGPR。
  - dK / dV 按 Q head 输出，与厂商约定相同，GQA 组需要再求和。
- causal 为右下对齐。

**对照**：[`attn_bwd_host.cpp`](../tests/gpu/attn_bwd_host.cpp) 同时运行 ours 与 aiter fmha_v3_bwd 的 odo、dqdkdv a32、dq_convert 三个 kernel。O 与 LSE 由 CPU 前向算出；CPU 参考对采样行检查 dQ / dK / dV。

**数据**：2026-11-08，GPU 5。ours 与厂商都通过（`bad=0`），ours strict 0 错误。

| 形状 | ours | aiter | 比 |
|---|--:|--:|--:|
| B1 H8 2048²，非 causal | 387.9 µs | 141.0 µs | 0.36 |
| B1 H8 HK2 2048²，causal | 416.0 µs | 162.9 µs | 0.39 |

**性能进展**（同次运行）：
- S / dP 两条累加链交错；LSE / Dot 改为按 4 对齐整读，去掉逐值选择；操作数改用 `tr_b16x2` / `pack4`。
- 效果：B1H8 2048²：391 → 237 µs（0.60×）；causal GQA：416 → 263 µs（0.62×）；小 causal：0.91×。
- 试过但未采纳：
  - 单 kernel 版本：dS^T 经 LDS，K tile 只暂存一次，dQ 用 `buffer_atomic_add_f32` 累加，另加 dq_convert。结果正确，但 246 vs 237 µs、causal 274 vs 263，更慢。代码已从树中移除，保留为 tag `archive/bwd-fused`。
  - dkdv 的 causal key tile 配对：128 个 CTA 在 256 个 CU 上被减半后反而更慢。

**编译策略（2026-11-09，已采纳）**：dq 与 dkdv 改用 `CompileProfile::Vs128` 加 `ldsink=1,wsink,ldahead32=1`。
- dkdv：B1H16 4096² 从 585 µs 降到 407–411；B1H8 2048² 从 145 降到 98。单用 tokens 只有 1065 对 1072，单用 Vs128 更慢。
- sched 的分区策略（`reorder,ldshoist` / `reorder,hzfill`）在 Gemm profile 上能把 139.5 降到 114.4，但叠加在 Vs128 + tokens 之上反而略慢（102.2 对 98.0），因此没有声明；区域 marker 保留。
- 同次运行（ours / aiter）：

| 形状 | ours | aiter | 比 |
|---|--:|--:|--:|
| B1H8 2048² | 177.2 µs | 141.9 µs | 0.80× |
| B1H8 HK2 2048² causal | 195.3 µs | 163.8 µs | 0.84× |
| B1H16 4096² | 681.1 µs | 465.0 µs | 0.68× |
| B1H16 4096² causal | 672.5 µs | 335.8 µs | 0.50× |
| B4H8 1024² causal | 123.9 µs | 116.6 µs | 0.94× |

- **causal 的最长优先发射顺序**（sched：`fmha_bwd_dq_hd128_bf16_causal_lpt` / `fmha_bwd_dkdv_hd128_bf16_causal_lpt`，由 `dispatch::bwd_causal_kernels` 选择）：
  - 网格改为 (H, B, tiles)，tile 维放在最慢的一维，最重的 tile 先发。
  - 原因：硬件按线性 id 顺序取 workgroup。tile 放在最快维时，每个 XCD 的队列里长短 tile 交替，长 tile 被拖到后面执行。ATT 显示单个 pair 的耗时与负载无关，所以差距不在单个 CTA 的速度上。
  - 同次运行，对比原顺序 / 配对 dkdv（配对已移除，保留为 tag `archive/bwd-dkdv-pair`）/ lpt，总时间（µs）：

| 形状 | 原顺序 | 配对 | lpt |
|---|--:|--:|--:|
| B1H16 4096² | 598 | 450 | 407 |
| B4H8 2048² | 323 | 249 | 228 |
| B1H8 HK2 2048² | 174 | 186 | 167 |
| B2H8 HK2 1000×1500 | 118 | 151 | 113 |

  - lpt 在所有形状上都最快。
  - 对厂商：4096² causal 为 0.81×，B4 2048² 为 0.88×，causal GQA 2048² 为 0.98×，1000×1500 为 1.18×。
- **dkdv 的 mask 改写**：每个 tile 先算出可见查询相对 q0 的区间 [dlo, qhi)，每个值只与常量比较一次再做 select。原做法是逐值做整数运算再与 select_i32 组合。
  - dkdv：2048² 从 98.4 µs 降到 93.8，2048² causal GQA 从 119.8 降到 104.1，1000×1500 causal 从 69.5 降到 59.2，4096² causal（配对）从 275 降到 236。
  - 总时间对厂商的比值：2048² 为 0.82×，causal GQA 为 0.91×，4096² causal 为 0.73×。
  - 同样的改写用在 dq 上反而让 dq 慢了 2.6×（VGPR 253，调度失常），没有采纳。
- **XCD 均衡的 snake 顺序**（sched，causal 的 dkdv key tile 与 dq query tile）：每满 8 个 tile 的一行中，奇数行反向排列，使每个 XCD 拿到的工作量相同。
  - B4H8 2048² causal：dq 从 102.3 µs 降到 96.9，dkdv 不变；其余形状持平。
- dq 的累加器固定到 AGPR：2048² 时 dq 从 71 µs 降到 65，但 4096² 时从 224 涨到 275，没有采纳。
- **dkdv 每 SIMD 2 个 wave 不可行（算术推导，未写代码）**：
  - 寄存器：2 个 wave 要求每个 wave 统一寄存器 ≤256，现为 236 VGPR + 128 AGPR = 364，需削减 108。K/V 移进 LDS 只能省 64；dK/dV 累加器的 128 个 AGPR 是必需的。
  - LDS：2 个 4-wave CTA 意味着每个 CTA ≤80 KB。但 K/V 每 wave 要 16 KB（每 CTA 64 KB），加上共享的 Q/dO ring 64 KB，共 128 KB；即使 Q/dO 只留单 slot 也有 96 KB。改成 8-wave CTA 则要 128 + 64 = 192 KB，超过 160 KB。
  - 若把 dK/dV 按维度拆给两个 wave（AGPR 减到 64），总量仍约 300。
  - 结论：在当前 tile 形状下，K/V 进 LDS 只能缩短活跃区间，无法让占用率翻倍。
- **dq 寄存器**（目标是每 SIMD 3 个 wave，需 ≤168；目前 245 VGPR，2 个 wave）：
  - 按 32-key 半块流式计算 dS：VGPR 反而升到 256，时间持平，未采纳。
  - 把 dQ 累加器固定到 AGPR：统一寄存器变为 265 > 256，掉到 1 个 wave，4096² 时变慢，未采纳。
- 当前比值：2048² 为 0.83×，causal GQA 为 0.93×，B4 2048² causal 为 0.61×（dkdv 211 µs 对厂商总时间 200 µs，仍是瓶颈）。1000×1500 causal 为 1.13×，快于厂商。

**原因与方向**：
- 余下的差距在 dkdv 的单 tile 效率：单独 139 µs，约等于厂商的总时间。
- **寄存器是瓶颈（暂停）**：
  - 现状：dkdv 用 VGPR 200 + AGPR 128 = 328，每 SIMD 只能跑 1 个 wave，ping-pong 无从重叠。P / dS 已提前打包成 bf16（性能持平，已采纳）。
  - 要降到 ≤256，需要两处重构：
    - K / V 的 B 操作数（64 VGPR）放进 LDS，会使 S / dP 的 LDS 流量加倍；
    - LSE / Dot（被提前发出的 16 条 b128 load，占 64 VGPR）要随 Q / dO 的 DMA 按 q tile 暂存进 LDS。
  - 厂商的 dqdkdv 用 512 个寄存器、每 SIMD 1 个 wave，同样做到 141 µs。所以下一步改为做单 wave 内的软件流水调度（厂商式），不再追求占用率。
- 之前计划的 v3 式处理（读取提前、8 wave ping-pong）因此搁置。
- 两个 kernel 各自重算 S / P，dP 也算了两遍。厂商是单 kernel，dQ 走 fp32 原子累加。
- causal 时 dkdv 只裁掉了前部的 q tile。
- 方向：合并为一个 kernel，dQ 用原子累加（需要后端提供 `buffer_atomic_add_f32`）；加入延迟 rescale 一类的 VALU 优化。

### 7.8 批次 5b（implicit-GEMM 卷积）状态：2d / 3d bf16 覆盖完成

**实现**：`conv{2,3}d_bf16_[t<T>]r<R>s<S>_st<st>_p<pad>_d<dil>[_c128]`，见 [`gemm/mod.rs`](../crates/amdgpu-kernels/src/gemm/mod.rs) 的 `ConvCfg`。它复用 v4 bf16 GEMM 结构。
- **GEMM 视图**：
  - M = NB·OD·OH·OW。分解 m → (nb, od, oh, ow) 用浮点倒数加一步整数校正，只在循环外算一次。
  - K = C·T·R·S，按"通道块 / 抽头"排序：每个 128 字节 K tile 是一个抽头上的 64 个通道。
- **每 tile 的计算**：先用编译期常数的乘移位分解出抽头 (dz, r, s) 和通道块，再给出 tile 偏移。每个 A issue 判断输入坐标是否越界；越界时 voffset 取越界值，硬件读回 0，padding 由此实现。
- **权重**：由 host 重排为 `[K][C/64][T][R][S][64]`，再做 `shuffle_weight(16,16)`。
- **输出**：NDHWK，即 C [M, K]；最后一个 M tile 的多余行被描述符丢弃。

**对照**：[`conv_host.cpp`](../tests/gpu/conv_host.cpp) 用 CPU 参考检查采样输出。FlyDSL 不可运行，所以性能按等效 GEMM 的 TF/s 报。

**数据**：2026-11-08，GPU 5。全部 `bad=0`，strict 0 错误。

| 形状 | TF/s |
|---|--:|
| 3×3 st1 p1，NB32 56² C64 K256 | 494 |
| 3×3，NB32 28² C256 K512 | 636 |
| 1×1，NB2 56² C256 K256 | 49 |
| 3×3 st2，NB4 57² C128 K256 | 44 |
| 3×3 d2 p2，NB2 30² C64 K256 | 20 |
| 3d 3×3×3，NB2 8×16² C64 K128（CTA 128） | 59 |

小形状 tile 数少，所以 TF/s 低。

**小形状改进**：网格不足 CU 数一半时改用 `_c128`，即 128×128 CTA，规则见 [`dispatch.rs`](../crates/amdgpu-kernels/src/dispatch.rs) 的 `conv_cta`。
- 1×1 NB2：49 → 94 TF/s；
- 3×3 s2：44 → 89 TF/s；
- 3×3 NB2 56²：69 → 135 TF/s；
- NB8 56²：255 → 408 TF/s；
- NB16 28²：345 → 567 TF/s。

conv 的 split-K 也试过：K tile 数需要能被 S 整除，而且实测更慢，所以未保留。

**待办**：
- 小 M 时用 split-K / 128 CTA；
- 抽头判断目前每 tile 每 issue 一次 VALU，可以裁掉无越界的 tile；
- C 非 64 倍数时的通道 padding；
- fp8 卷积。