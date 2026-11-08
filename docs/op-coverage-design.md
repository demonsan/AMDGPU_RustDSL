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
