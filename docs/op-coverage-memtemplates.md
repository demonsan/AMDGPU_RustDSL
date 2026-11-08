# §5 访存受限算子的 pipeline 模板

状态：设计稿（op-coverage 设计的第 5 节，由 sched 编写；批次 2 实现这里列出的模板与算子）。范围：aiter kernel zoo 中**带宽受限**的基础算子 —— 逐元素（激活 / 门控 / 量化 / 类型转换）、行归约（norm / softmax / amax）、列 / 分段归约、前缀扫描、排序 / top-k、gather / scatter（paged KV、MoE 排列、cache 写）。每个模板给出：tile 配置、wave / block 角色、HBM 饱和所需的在途字节数、LDS / 寄存器预算、需要编译器给的调度旋钮、与 stage 原语（M / R / E / Q / S / P，[`fusion-stage-library-design.md`](fusion-stage-library-design.md) §1）和 `Pipeline` / weave2 的对应、aiter 里谁用它、以及**我们的 DSL / 库缺什么**。

方法：先在 MI355X 上用一个流式拷贝 / 读 / 写基准量出"实际可达"的带宽（5.1，源码 [`tests/gpu/membw_survey.cpp`](../tests/gpu/membw_survey.cpp)），再用它给每个模板定 roofline 目标；已有的 RMSNorm + 量化内核（`rmsnorm_quant_fp8_n<N>`）是第一个实测点。

## 5.0 摘要

1. **可达带宽**：2 GiB 工作集，HBM 标称 ≈ 8 TB/s，实测拷贝（读 + 写合计）**5.7–6.2 TB/s**（每 lane 16 B × 1 次、按 tile 一块一份），只读 **6.2–6.4 TB/s**，只写 **5.9–6.0 TB/s**；256 MiB 工作集（部分落在 256 MB Infinity Cache）读 / 写到 **6.5–6.7 TB/s**。roofline 目标 = 同读写比下实测流式带宽的 90 %（见 5.1）。
2. **决定带宽的是在途字节数，不是向量宽度**：每 CU 在途读 ≥ 64 KB 才到 90 % 的可达值（16 KB → 2.4 TB/s，32 KB → 3.7，64 KB → 5.3，128 KB → 5.5）；4 B / 8 B / 16 B 每 lane 的访问只要在途量够，带宽相同（6.2–6.4 TB/s 只读）。`nontemporal` 对流式 kernel 没有可测差别（±1 %）。
3. **按 tile 一块一份（每 lane 16 B × 1）优于大展开**：U = 1 最好（6.06 TB/s），U = 8 反而 5.3；grid-stride 持久化至少要 4 块 / CU（grid ≥ 1024）才追平，且不更好。**默认模板 = 按 tile、每 lane 一次 b128、占用率优先**，不做 grid-stride。
4. **归约的延迟对带宽基本无影响**（占用高时）：一个 wave 一行的 norm 在稳态下每个 wave 每行有 ≥ 20 k 周期预算，LDS 版 `wave_reduce`（≈ 300 周期）可以接受；DPP / permlane 全程归约只在占用低（GEMM 尾部、小 M、单 wave 内多行 ILP 不足）时才值得。
5. **缺口（按优先级）**：① 缓存策略位（`glc` / `slc` / `nt`）与 sub-dword 访问；② `ds_bpermute` / `readlane` / `ballot` / `mbcnt` / `s_bcnt1`（扫描、top-k、MoE 对齐都需要）；③ 完整 DPP 行操作与 `permlane16`；④ LDS 与全局原子（add / inc / CAS / min / max）；⑤ 64 位地址乘加、整数除 / 取模；⑥ 带索引的 `M`（gather / scatter）与跨 wave 的 `R`（block 归约）、分段 `R`（group amax）、`S`、`P` 的 stage 实现（库里目前只有 `M` 的行 / DMA / 块拷贝、wave 内 `R`、fp8 `Q`）。详见 5.10。

## 5.1 机器事实与 roofline 目标

**MI355X（gfx950）**：256 CU（8 XCD × 32），每 CU 4 个 SIMD、每 SIMD ≤ 8 个 wave（≤ 32 wave / CU），VGPR（含 AGPR 的统一文件）每 SIMD 512 × 64 lane × 4 B = 128 KB，LDS 160 KB / CU，L2 每 XCD 4 MB，Infinity Cache 256 MB，HBM3E 标称 ≈ 8 TB/s。波数 / SIMD = `min(8, ⌊512 / VGPR 数（向上取整到 8）⌋)`。

**实测**（`tests/gpu/membw_survey.cpp`，GPU 4，hipEvent，15 次取中位数；"r+w" 是读写字节之和 / 时间）：

| 访问方式 | 2 GiB 工作集 | 256 MiB 工作集 |
|---|--:|--:|
| `hipMemcpy` D2D（r+w） | 4.88 TB/s | 5.18 TB/s |
| 拷贝，b32 / lane，U = 4，按 tile | 6.25 | 5.94 |
| 拷贝，b128 / lane，U = 1，按 tile | **6.06** | **6.01** |
| 拷贝，b128 / lane，U = 4，按 tile | 5.74 | 5.69 |
| 拷贝，b128 / lane，U = 8，按 tile | 5.31 | 5.23 |
| 拷贝，b128，U = 4，`nontemporal`，按 tile | 5.72 | 5.62 |
| 拷贝，grid-stride，b128，U = 4，grid = 256 / 512 / 1024 / 2048 / 4096 | 2.42 / 3.75 / 5.30 / 5.50 / 5.53 | 2.42 / 3.88 / 5.17 / 5.19 / 5.28 |
| 只读（异或归约），b128，U = 4 / 8 | 6.41 / 6.21 | 6.55 / 6.67 |
| 只读，b32 / b64 / b128，U = 8，grid-stride | 6.27 / 6.19 / 6.17 | 6.54 / 6.70 / 6.70 |
| 只写，b128，U = 4，按 tile | 5.91（`nt` 6.03） | 6.60 |

读法：
- **在途字节数是第一变量**。grid-stride 的 grid = 256 / 512 / 1024（= 1 / 2 / 4 块 × 256 线程 / CU，U = 4，b128 → 每 CU 在途读 16 / 32 / 64 KB）给出 2.4 / 3.7 / 5.3 TB/s，之后饱和在 5.5。负载下的有效 HBM 延迟约为 `64 KB × 256 / 5.3 TB/s ≈ 3 µs`。设计值：**每 CU ≥ 64 KB 在途读 → 90 %，≥ 96–128 KB → 95 %**。
- 在途量够时，向量宽度（4 / 8 / 16 B）、`nt`、持久化与否都不改变结果；大展开（U = 8）因为占用下降、每块工作量变大而略差。所以**模板的默认选择是占用率**：每 lane 一次 b128（每 wave 1 KB），每块 256 线程，让 CU 容纳尽可能多的块。
- 工作集小于 Infinity Cache（256 MB）时读 / 写能到 6.5–6.7 TB/s，roofline 目标要按实际工作集取（激活张量常常 < 256 MB）。

**roofline 目标**：`t ≥ 字节数 / BW_target`，`BW_target` 取同读写比的实测值：读 : 写 = 1 : 1 的逐元素 5.7–6.0 TB/s，读 > 写（量化 2 : 1、行归约 N : 1）6.0–6.4 TB/s；**验收 = 达到 `BW_target` 的 90 %**。已有数据点：`rmsnorm_quant_fp8_n4096`，M = 16384（x bf16 读 134 MB + q fp8 写 67 MB，读写比 2 : 1）5.3–5.5 TB/s ≈ 88–92 % —— 已在目标线上。

**每 CU 在途字节数的实现方式**：在途量 = 占用的 wave 数 × 每 wave 未完成的 `buffer_load` 字节数。例如 4 wave / SIMD（16 wave / CU，VGPR ≤ 128）→ 每 wave 4 KB = 每 lane 4 次 b128（16 个落地 VGPR）；8 wave / SIMD（VGPR ≤ 64）→ 每 wave 2 KB = 每 lane 2 次 b128。`vmcnt` 的上限是 63，在途 b128 数从不会成为限制，**限制的是落地寄存器和每 wave 的发射次序**（5.9）。

## 5.2 模板总表

| 模板 | 做什么 | stage 原语 | aiter 里谁用 | 库里现状 |
|---|---|---|---|---|
| **T1 流式逐元素** | `y = f(x [, y2, γ, …])`，含激活 / 门控 / residual add / 类型转换 / 量化的元素部分 | M → E → M（Q 的转换部分） | `activation_kernels.cu`（silu_and_mul 等）、`binary_operator.cu`、`unary_operator.cu`、`fused_mul_add` / `fused_sigmoid_mul` / `fused_clamp_act_mul`（Triton）、`quant_kernels.cu` 的 per-tensor | `prims.rs::act/affine`，G1q 的 epilogue 链；没有独立的流式 kernel |
| **T2 行归约（一行一 wave / 一行一 block / 长行两趟）** | 沿行 sum / sumsq / max / amax，结果回写到行内每个元素（norm、softmax、per-token quant） | M → R → E → (Q) → M | `rmsnorm_*`（Opus / HIP / Triton）、`layernorm`、`fused_add_rmsnorm_pad`、`rmsnorm_quant_kernels.cu`、`gated_rmsnorm_quant_kernels.cu`、`fused_qk_norm.cu`、`softmax`、`cross_entropy` | `norm/`（RMSNorm + fp8 per-token，一行一 wave，N ≤ 8192） |
| **T3 分段 / 列归约** | 行内每 G 个元素一组的 amax / sum（group quant、groupnorm）；沿列归约（bias 梯度、`moe_reduce`、列 amax） | M → R(分段 / 列) → E → Q → M | `quant_kernels.cu`（per-group）、`quant_mxfp4.cu`、`fused_*_quant`、`groupnorm.cu`、`moe_reduce`、`fast_transpose` | 无（wave 内 `R` 只做整行） |
| **T4 softmax / norm 的一趟与两趟** | 在线（一趟）统计 vs 重读（两趟）；长行（> 一行一 wave 的寄存器上限）的分块 | M → R / O → E → M | `softmax`、`topk_softmax`、`mla` / `pa` 的 combine、`cross_entropy` | `softmax.rs` 是 FMHA 的在线 softmax（O 原语），没有独立的行 softmax kernel |
| **T5 前缀扫描** | wave / block / 解耦回看的 inclusive / exclusive scan，CDF，块偏移 | P | `moe_sorting` / `moe_align_block_size`（块偏移）、sampling（CDF）、`causal_conv1d` 的状态、线性注意力分块递推 | 无 |
| **T6 排序 / top-k** | 行内 top-k（k ≤ 64 常见）、argmax、按 key 排序 / 分桶 | S | `topk_softmax_kernels(.group).cu`、`topk_gating/*.cu`、`topk_plain_kernels.cu`、`topk_per_row_kernels.cu`、`sample_kernels.cu`、`radix_topk_*.py`（FlyDSL）、`asm_topksoftmax` / `asm_topk_per_row_*`（ASM） | 无 |
| **T7 gather / scatter / cache 写** | 带索引的行搬运（paged KV、token 排列、`reshape_and_cache`、RoPE + cache、dequant_gather） | M（带索引）+ E（RoPE / 量化）+ M | `cache_kernels.cu`、`kv_cache` / `fused_kv_cache`、`fused_qk_norm_rope_cache_quant.cu`、`dsv4_dequant_gather_k.cu`、`trans_ragged_layout`、MoE gather / scatter copy、`rope/*.cu` | 无（`M` 只有仿射地址） |

## 5.3 T1 流式逐元素

**tile 配置**：每块 256 线程，每线程每次 b128（8 × bf16 / 16 × fp8），块处理的 tile = 256 × 16 B = 4 KB 输入（一份），grid = ⌈N / tile⌉ 块（**按 tile，不做 grid-stride**）；多个输入（`x`、`y`、`γ`）各自独立的 `buffer_load`，输出按输出 dtype 打包后 b128 / b64 / b32 存（fp8 输出每线程 8 B）。tail 靠 `buffer` 描述符的 `num_records`（硬件守卫：越界读 0、越界写丢弃），**不写 tail 分支**（已在 RMSNorm 内核里使用并验证，`norm/mod.rs`）。长度不是 16 B 倍数的行（例如 N % 8 ≠ 0）：退到 b64 / b32 访问，由 stage 参数 `vec` 选择，描述符范围保证正确。

**角色**：全部 wave 同构（没有生产者 / 消费者分工）；流水线是**隐式的跨块并行**（占用率换在途量），块内只有"加载 → 计算 → 存"三段。若每线程需要 ≥ 2 个 b128 才能凑够在途量（VGPR 紧时），用 U = 2（两次加载先发后算），U ≥ 8 不要用（见 5.1）。

**在途字节 / 预算**：每 wave 加载 1 KB × U；占用 ≥ 16 wave / CU（VGPR ≤ 128）→ 在途 ≥ 16 KB × U / CU，U = 4 刚到 64 KB。LDS 不需要。寄存器 = 落地（4 × U）+ f32 展开的中间值（激活要先转 f32：bf16 × 8 → 8 个 f32）。目标 BW：读写 1 : 1 → 5.5 TB/s（≥ 90 % × 6.0）。

**调度旋钮**：
1. **加载先于计算发射**：同一线程的 U 个 `buffer_load` 在任何 VALU 之前连续发射（`Hint::LatencySensitive` 的单元排在前面，现有 weave 的 `seq` + list scheduler 已能做到；见 5.9）。
2. **等待**：计算只等自己需要的那个 `vmcnt(n)`（`n` = 其后还在飞的加载数）。记分板按发射顺序推 `vmcnt`，不要在加载之间插入 `wait_vm(0)`。
3. **存放在计算后**，不需要 `wait` 存（存完成不阻塞后续，除非 kernel 末尾）；`s_endpgm` 前由硬件等。

**映射到 stage**：`M`(load，`move.row_load` 的 b128 版) → `E`(act / mul / add，`prims::act`、`affine`) → [`Q` 的转换：`fp8_pack`] → `M`(store)。一个 stage = 一个 weave 相位；元素数多时每个单元 = 一组 b128 的加载 / 计算 / 存，单元间没有依赖，`AMDC_WEAVE_AUTO` 之类的策略没有自由度可利用（带宽受限，顺序即"全部加载在前"）。

**aiter 对照**：`activation_kernels.cu` 是一块一 token，线程按 `VEC_SIZE_I`（最大 8 个元素）步进整行，`load_chunk_bytes` 取 16 / 8 / 4 B，`AUX` 模板参数设缓存策略（`RT` / `GROUP_NT`）。我们的版本去掉 `AUX`（实测无差别）和块内步进循环（改成按 tile 的 grid）。

**缺口**：缓存策略位（`nt` / `slc`，测得不影响流式性能，**低优先级**，但 `GROUP_NT` 对"只用一次的权重 / 激活"在多块复用同一 γ 的场景有用）；sub-dword 访问（u8 / u16 的 tail，当前用 buffer 硬件越界 + 对齐规避）；`v_pk_*` 的 packed f16 / bf16 数学（有 `PkAddF32` / `PkMulF32`，没有 bf16 packed）；向量化 `cvt`（fp8 / bf16 的 `CvtPkBf16F32`、`CvtPkFp8x4F32` 已有）。

## 5.4 T2 行归约

三种实现，按行长 N（元素数）和占用选：

| 变体 | 适用 | 角色 / tile | 归约 | 在途字节 |
|---|---|---|---|---|
| **一行一 wave**（已实现，`norm/`） | N × 2 B ≤ ~16 KB（N ≤ 8192），VGPR ≤ ~128 | 4 wave / 块，一块 4 行，lane `l` 持有元素 `8 (l + 64 i) + k`（每个 i 一次 b128）；M 方向不跨块 | wave 内：lane 内 → DPP quad ×2 → LDS 16 槽（`wave_reduce`） | 每 wave 一整行 = 2 × N B（N = 4096 时 8 KB），**一次发射全部加载**，占用 = VGPR 允许的 wave 数 |
| **一行一 block** | N 大（8192 < N ≤ 64 K）或要降低每线程寄存器 | 256–1024 线程，行按 b128 跨线程交错，每线程 `N / (T × 8)` 次加载 | wave 内归约 → 每 wave 把部分结果写 LDS（`4 × waves` B）→ barrier → 一个 wave 读全部并归约 → 广播（再一次 LDS 往返或 `readfirstlane`） | 每块 `2N` B 在途，块数少时靠多块 / CU 并行 |
| **长行两趟（split-row）** | N > 64 K 或 M 很小（M < 2 × 256 块）需要用满 256 CU | 一行拆成 `S` 块，各块部分和写 workspace（或全局原子），第二个 kernel / 最后到达者（`probe_last_arriver` 已验证的协议）做合并 | 块内 + 跨块 | 取决于 `S`：保证 `M × S ≥ 4 × 256` 块 |

**要点**：
- **一行一 wave 是首选**。在途量天然够（一行就是 8 KB，一个 CU 上 16 个 wave 同时各持一行 = 128 KB），归约延迟被占用率隐藏：每 CU 带宽份额 ≈ `6 TB/s / 256 = 23 GB/s ≈ 10 B / 周期`，一个 wave 的一行 8 KB 对应每 CU 约每 800 周期完成一行，16 wave 同时在飞时每个 wave 每行有 ~13 k 周期预算，而两次 `wave_reduce`（sumsq、amax）合计 ≈ 600 周期。
- **LDS 归约 vs DPP / permlane 归约**：`wave_reduce` 现在是 2 × DPP quad + 16 槽 LDS（写 → 读 4 × `ds_read_b128` → 合并，≈ 300 周期，LDS 每 wave 私有 64 B，无 barrier）。全 DPP 版（`row_shr:1,2,3`、`row_bcast:15`、`row_bcast:31` 或 `permlane32_swap` + `row_shr`）≈ 6–7 条 DPP VALU，每条要 `s_nop` 隔开数据冒险，≈ 60 周期，**但需要 DSL 里的 DPP 行操作**（现在只有 `quad_perm` 和 `permlane32_swap`）。收益只在占用低（< 8 wave / CU）、M 小或 GEMM 尾部融合 norm 时可见；带宽受限的 norm 不换。
- **两个统计量一次归约**：RMSNorm + amax 的 sumsq 和 amax 现在是两次 `wave_reduce`（`gen` 0 / 1）；若 amax 依赖 sumsq（`y = x × inv_rms × γ` 后求 amax），不能合并（已是最小依赖链）。LayerNorm 的 mean / var 用一趟 Welford（或 `Σx` 与 `Σx²` 同时归约：两个值打包成一次 `wave_reduce` 的两个槽，省一次往返）。
- **读取次数**：norm 类要两次遍历行（统计量、再归一）。一行一 wave 时行驻留寄存器（`x` 打包成 bf16 保存，f32 展开按需重算，如 `norm/mod.rs` 的做法），**HBM 只读一次**；N 超过寄存器容纳（8192 × 2 B / 64 lane = 256 B / lane = 64 VGPR，已是上限附近）就改一行一 block，或第二趟重读（L2 命中：一行 16 KB 远小于 4 MB L2，重读代价是 L2 带宽，不是 HBM）。

**LDS / 寄存器预算**：一行一 wave：LDS 每 wave 128 B（`RED` 私有区），VGPR = `2 × N / (64 × 4)`（x 打包）+ γ 打包 + 临时 ≈ N = 4096 时 88（实测 `rmsnorm_quant_fp8_n4096` 88 VGPR → 5 wave / SIMD）。一行一 block：LDS = `4 × waves` B + 广播槽，VGPR ≈ 32–64（`N / (T × 8)` 个 b128 落地）。

**调度旋钮**：(1) 全行加载一次连续发射，`wait_vm(0)`（或 `vmcnt(n)`）放在第一个统计量之前；(2) 两趟读时第二趟的 `buffer_load` 在统计量归约的 LDS 往返期间发射（预取，掩盖 L2 延迟）—— 这是 weave 的 `Hint::LatencySensitive` 单元和 `rms` 单元之间的重排，现有 `norm` 的单元划分（`load.rows` / `rms.reduce` / `quant.amax_fp8` / `store.rows`）已经是这个粒度；(3) 同一 CU 上多个 wave 的 `s_setprio`（让持有最老行的 wave 先发射）对带宽无益，不用。

**映射到 stage**：`M`(`move.row_load`) → `R`(`reduce.row_sumsq`，`wave_reduce`) → `E`(归一 × γ) → `R`(amax) → `Q`(`quant.fp8_row`) → `M`(`move.row_store`)，即现有 RMSNorm 的四个 stage；一行一 block 需要新的 **block 级 `R`**（跨 wave 的 LDS 部分和，stage 合同 `barrier_phase` / `roles` 要声明），长行两趟需要 **grid 级 `R`**（工作区 + 最后到达者，沿用 `probe_last_arriver` 的内存模型，`docs/last-arriver-memory-model.md`）。

**aiter 对照**：`rmsnorm_quant_kernels.cu`：块大小 + `thread_data_size`（每线程元素数）模板，`load_chunk_bytes` 16 / 8，`interleave` 选择按 wave 交错的行内布局，`ADD_RESIDUAL` / `FUSE_QUANT` 开关，行尾 `ooba` 越界保护；`fused_qk_norm.cu`、`gated_rmsnorm_quant_kernels.cu` 同一骨架加门控 / 两路。

**缺口**：block 级 `R` 的 stage；grid 级 `R`；LayerNorm 的 mean / var（Welford 或双槽归约）；`rsqrt` 目前是 `2^(-log2 x / 2)`（`prims::rsqrt_f32`，相对误差 ~1e-6，需要时加 `v_rsq_f32` 的 DSL 操作）；DPP 行操作。

## 5.5 T3 分段 / 列归约

**分段归约（行内每 G 个元素一组）**：group quant（G = 32 / 64 / 128 的 amax → E8M0 / fp32 scale）、groupnorm。**布局**：把一组映射到同一个 lane 的连续元素或同一个 quad / 半波，使组内归约不跨 wave。例如 G = 128 个 bf16 = 256 B = 16 lane × 16 B：每 lane 一次 b128 加载（8 个值），**16 个 lane 一组**，组内归约 = lane 内 → 4 步 `DPP`（`row_shr` / `quad_perm` 组合）→ 完成，**不需要 LDS**。G = 32：4 lane 一组 = 2 步 `quad_perm`（我们已有）。G = 64：8 lane，3 步 DPP（`quad_perm` ×2 + `row_shr:4` 一类，缺 DPP 行操作；或用 `permlane32_swap` 之外的 `ds_swizzle`）。输出：fp8 值（每线程 8 B）+ scale（每组一个：B 字节宽的 scale 张量，**布局要符合下游 GEMM 的 scale 布局**，如 E8M0 的 `[M/32][K/4]` 分块 + preshuffle，由 `Q` 的 `scale layout` 参数决定，参考 `docs/mxfp8-blockscale-*.md`）。

**列归约（沿 M 归约，如 bias 梯度、`moe_reduce` 对 top-k 个专家的加权和、列 amax）**：一块处理 `C` 列 × 全部（或一段）M 行。线程映射：列方向向量化（每线程 8 个相邻列），行方向分给多个 wave；每 wave 先在寄存器里累加自己的行（`M_w` 行，U 个加载在飞），再块内跨 wave 用 LDS 归约（`waves × C × 4 B`）。**M 很大时**每块只处理一段 M，部分结果用全局原子加（f32 原子，缺）或两阶段（第二个 kernel 合并，沿用 `splitk_epilogue` 的做法）。合并写入 = 每列一次。
- tile：`C = 64–256`（至少 128 B 的连续行段，保证合并访问），M 段 = `Mseg`，使 `M × N / (Mseg × C)` ≥ 4 × 256 块；
- 在途量：每块 `waves × U × 1 KB`，U = 4，8 wave → 32 KB / 块，≥ 2 块 / CU。
- **XCD 序**：块 → XCD 的映射是块号对 8 取模（轮转）；列归约的相邻列块读同一批行 → 不要让同一行段的块散在 8 个 XCD（每个 XCD 的 L2 各取一份）：把块号按 `(xcd, 块内序)` 重排，使同一 XCD 处理连续的列块（`grid_swizzle`，GEMM 用过的 `GRID_MODE=xcd`）。

**与 stage 的映射**：`M` → `R`(分段 / 列，参数 `axis`、`group`、`scope`) → [`E` + `Q`] → `M`。库里 `R` 只有"整行、wave 内"这一种 scope：需要加 `group` 参数（lane 内 / quad / 16 lane / 半波 / wave / block / grid 的 scope 阶梯，`fusion-stage-library-design.md` 1.1 已列）。

**aiter 对照**：`quant_kernels.cu`（per-tensor / per-token / per-group）、`quant_mxfp4.cu`、`fused_fp8_quant` / `fused_mxfp4_quant` / `fused_mxfp8_quant`（Triton）、`groupnorm.cu`、`dsv4_rotate_quant.cu`、`inverse_rope_group_quant.cu`；列归约：`moe_reduce`、`fast_transpose`。

**缺口**：DPP 行操作（`row_shr`、`row_bcast`）；块级 / 列 `R` stage；f32 全局原子加；scale 输出布局（E8M0 的分块 / preshuffle）作为 `Q` 的参数。

## 5.6 T4 softmax 与 norm 的一趟 / 两趟

- **行 softmax**（`softmax`、`cross_entropy`、`mla` / `pa` 的 combine）：`max` → `Σ exp(x − max)` → 归一。**一行一 wave、行驻寄存器**时是"一次 HBM 读 + 一次写"（三趟都在寄存器里，两次 `wave_reduce`：max、sum）。长行（N > 8 K）用**在线**一趟：每个 b128 块更新 `(m, l)`（`O` 原语 `softmax.rs` 的 `OnlineSoftmax`，量程 / 重标定逻辑可直接复用），最后一次归一重读（L2 命中）或把 `exp` 结果暂存 LDS。
- **top-k softmax 门控**（`topk_softmax_kernels.cu`：一块一行、块内 `block_reduce`，top-k 做 k 轮 argmax）：见 5.8，softmax 和选择在同一个 kernel 里，行很短（experts ≤ 256–512），**每 wave 4–8 行**以保证在途量，不要一块一行。
- **一趟 vs 两趟的判据**：行长 × 2 B ≤ 寄存器容量（~64 KB / wave，实际受占用约束为 ≤ 16 KB）→ 行驻寄存器；否则在线。
- **精度**：`exp` 用 `exp2(x × log2e)`；max 减法在 f32；`Σ` 在 f32；bf16 输出前一次舍入。

**stage 映射**：`M` → `R`(max) → `E`(exp，减 max) → `R`(sum) → `E`(× 1/sum) → `M`；在线版 `O`。**缺口**：独立的行 softmax kernel / stage（`softmax.rs` 为 FMHA 的 P 转换写，行布局不同）。

## 5.7 T5 前缀扫描

| 规模 | 做法 | 需要的指令 |
|---|---|---|
| wave 内（≤ 64 元素，MoE 每专家 token 计数、CDF） | Kogge–Stone / Hillis–Steele：`log2 64 = 6` 步，每步 `v_add` + DPP `row_shr:1,2,4,8` + `row_bcast:15,31`（gfx9 的标准写法），或 `ds_bpermute` 取 `lane − d` | DPP 行操作 / `ds_bpermute` / `v_mbcnt`（lane id）|
| block 内（≤ 1024 元素 / 线程 × V） | 每线程先对自己的 `V` 个元素顺序扫描（寄存器）→ 每 wave 末元素做 wave 扫描 → wave 总和写 LDS（`waves` 个）→ barrier → 第一个 wave 扫描 `waves` 个总和 → 每 wave 加偏移 | LDS + barrier，wave 扫描 |
| 跨 block（大数组、MoE 块偏移的全局前缀） | **解耦回看（decoupled look-back）**：每块先把自己的聚合（`aggregate`）以 `(flag = A, value)` 写全局，再往前回看前驱块的 `flag`：遇到 `P`（含前缀）就停并相加，遇到 `A` 继续往前；自己的前缀算出后以 `flag = P` 发布。flag + value 要**一次 64 位原子发布**（或先写 value 再 `fence_release` 写 flag，读者 `fence_acquire`） | 全局 64 位原子 / `release`-`acquire`（DSL 已有 `FenceRelease` / `FenceAcquire`，`AtomicAddU32`；缺 64 位 CAS / 交换） |
| 两趟 | 趟 1：每块局部扫描 + 块总和写数组；趟 2：扫描块总和（单块）；趟 3：各块加偏移。多一次读 / 写，但**不需要块间同步** | 无新指令 |

**选择**：MoE 路由对齐的扫描长度 = 专家数（≤ 512）→ **单块 / 单 wave 内**完成，不用跨块；采样 CDF 每行 vocab 长（32 K–256 K）→ 一行一块的 block 扫描（或和 top-p 的排序合并）；全局大数组（> 100 万元素）才需要解耦回看，**批次 2 不做**，先做 wave / block 级 + 两趟。

**带宽**：扫描读一次 + 写一次，目标同 T1（5.5 TB/s）；小规模（MoE 计数）是延迟受限：靠**一个 kernel 里把"计数 + 扫描 + 对齐 + 写偏移"合并**避免多次 launch。

**与 stage**：`P`（原语表里"没有实现"），合同需要 `roles`（wave 内 / 块内）和 `barrier_phase`；扫描内核的 `Q` / `E` 无需参与。

**aiter 对照**：`moe_sorting` / `moe_align_block_size`（块偏移）、`sample_kernels.cu`（CDF）、`causal_conv1d_*.cu`（状态递推，是**分块线性递推**，不是纯前缀和）、线性注意力的分块递推。

**缺口**：`ds_bpermute`、`v_mbcnt_*`（lane id / 低位 lane 计数）、DPP 行操作、64 位原子；`P` 的 stage；块内 `barrier` 之间的 LDS 协议（`roles`）。

## 5.8 T6 排序 / top-k

**规模与方案**：

| 场景 | 行长 `n`，`k` | 方案 | 角色 / 预算 |
|---|---|---|---|
| MoE 门控（`topk_softmax`、`topk_gating`） | n ≤ 256–512（experts），k ≤ 8 | **wave 内**：一行 / 一 wave 或 4–8 行 / wave 的 lane 分组；每 lane 持有 `n / 64` 个值 + 索引（打包 `(value:f32, idx:u32)` 为 64 位比较键）；k 轮"wave 求最大 + 记录 + 屏蔽"或**部分 bitonic（lane 内排序 + 跨 lane 合并 k 项）** | 无 LDS（用 DPP / permlane / `ds_bpermute`）或每 wave 1 KB；VGPR ≤ 64 |
| 采样 / 词表 top-k / top-p | n = 32 K–256 K，k ≤ 64–1024 | **block 内 bitonic**：每线程寄存器排序 `V` 个元素 → 块内跨线程 bitonic（LDS）得到全序或 top-`k` 块；多块行则每块出 top-k 再合并（`BlockTopkMerge`）；k 小时先用 **ballot 阈值筛**（`BlockTopkFilter`）剔除大多数元素 | LDS = `n_block × 8 B`（键 + 索引）；块 256–1024 线程 |
| 很大的 n（≥ 1 M）/ k 大 | n 大 | **radix 选择**：逐 8 bit（或 11 bit）一趟的直方图（LDS 原子 / wave 内 `ballot` + `popc`）+ 前缀和定位第 k 大的阈值 + 一趟收集；多块则全局直方图（全局原子） | LDS 直方图 256 × 4 B × waves；需要 LDS 原子 / `ballot` / `popc` |

**带宽**：top-k 是**读密集**（读 n 个值，写 k 个）→ 目标读带宽 6.0–6.4 TB/s 的 90 %；门控的行短（≤ 512 × 4 B = 2 KB），所以每 wave 要同时处理多行（每 lane 持有行的一段，4 行 / wave 时每 wave 在途 8 KB），**不要一块一行**（aiter 的 `topk_softmax_kernels.cu` 一块一行 + `block_reduce` ×（k + 2），在小行上被延迟支配）。

**scatter 输出**：MoE 对齐（按专家排序、写 `sorted_token_ids` / `expert_ids` / `num_tokens_post_padded`）= 计数 + 扫描（T5）+ scatter（T7）：一个 kernel 内三段，用块内 barrier 衔接。

**aiter 对照**：`topk_plain_kernels.cu`（三种实现：`BlockTopkFilter` ballot 过滤、`BlockTopkSort` bitonic 排序 / 合并、`BlockTopkMerge` 多块合并；DPP 小步长 shuffle ≤ 8 + `__ballot` / `__popcll` / `__shfl`；radix 版在 `topk_per_row_kernels.cu`）、`topk_softmax_kernels(.group).cu`、`topk_gating/topk_{softmax,sigmoid,sqrtsoftplus}.cu`、FlyDSL `radix_topk_one_block` / `topk_per_row_*`、ASM `asm_topksoftmax` / `asm_topk_per_row_{decode,prefill}`。

**缺口**（本类的主要缺口，库里完全没有 `S`）：`ballot`（`v_cmp` 结果到 SGPR 掩码）、`s_bcnt1_i32` / `popc`、`v_mbcnt`、`ds_bpermute` / `readlane` / `writelane`、完整的 `min` / `max` / `cndmask` 对（已有 `MinI32` / `MaxF32` / `Select`）、64 位键比较（打包键，缺 64 位整数操作）、LDS 原子（`ds_add_u32`）；`S` stage（k 参数、键 / 值 / 索引三元组、规模阶梯 wave / block / 多趟）。**批次 2 的最小集**：wave 内 top-k（k ≤ 8，n ≤ 512，k 轮最大 + 屏蔽，需要 `readlane` 或 DPP 全 wave 最大 + `ballot` 找最小索引的并列项）与 MoE 对齐的计数 + 扫描 + scatter；block bitonic / radix 放后。

## 5.9 T7 gather / scatter / cache 写

**带索引的行搬运**：输入一个索引（`slot_mapping[token]`、`block_table[seq][page]`、`token → expert 排列`），按索引计算源 / 目的行地址，搬一行或一段（≥ 128 B 才能合并）。

- **tile**：一块（或一 wave）处理一行（head_dim × H 个元素 = 128–1024 B 以上）；每 lane b128，`row_bytes / (64 × 16)` 次加载 / 行；一 wave 同时处理 `R` 行（`R = 4–8`）使在途量 ≥ 4 KB / wave。
- **索引加载**：每行一个索引，`s_load` / `buffer_load`（标量化：索引在 wave 内一致 → `readfirstlane` 后做**标量地址**，`soff` 用 SGPR，省 VGPR 和地址计算）；**索引加载先于数据加载发射**，数据加载依赖它，所以多行 ILP 要先把 `R` 个索引一起加载。
- **页表**：页大小 2 的幂时 `page = pos >> log2`，`off = pos & (ps − 1)`，**不需要整数除**；非 2 的幂页 → 需要整数除 / 取模的 DSL 支持（缺）。
- **KV cache 写（`reshape_and_cache`、`fused_kv_cache`、`fused_qk_norm_rope_cache_quant`）**：写入 `[block][head][d/x][page][x]` 一类的分块布局时，目的地址是 `(slot / ps, head, dim / x, slot % ps, dim % x)` 的混合基数，**每 lane 8 B（x = 8 的 fp16 / bf16 或 fp8 的 x = 16）连续写**，布局由 `Layout` 代数生成（`amdgpu-layout`），不手算地址；量化写（fp8 cache + per-token / per-head scale）= `R`(amax) + `Q` + 两个 `M`。**写带宽**：scatter 到 paged cache 是半合并写（每行 `head_dim × 2 B` = 256 B 连续），目标 ≥ 70 % × 写带宽（5.5 TB/s）。
- **MoE token 排列（gather copy / scatter copy）**：`out[i] = in[perm[i]]`，行长 = hidden（≥ 4 KB），一块一行 × 多行；整行连续，近乎拷贝带宽（目标 5.0–5.5 TB/s）。`moe_reduce`（top-k 个专家输出按权重加和）= 列归约 + gather：每输出行读 k 个源行（`perm` / `weights`），寄存器累加，**一次写**。
- **RoPE（+ cache 写）**：`E` 成对旋转，`cos` / `sin` 按位置从表里 `M`(gather)（`general_1c` / `2c`、`cached` / `uncached` / `positions` / `offsets` / `thd`、`2d`、前向 / 反向）；NeoX（半维配对）vs GPT-J（交错）只改 `(x0, x1)` 的取法（半维配对：同一行的 `d/2` 偏移，两个 b128；交错：一个 b128 内相邻元素，用 `v_pk_*` 或 lane 内操作）。部分旋转（`rotary_dim < head_dim`）= 前 `rotary_dim` 旋转，其余原样拷贝。每 token × head 一个 wave，行 = 64–128 个元素（128–256 B）→ **一个 wave 处理 `R = 4–8` 个 head 才够在途量**，Q / K 一个 kernel 完成、K 直接写 cache（融合）。

**与 stage**：`M`(带索引，参数 `addr = index_table | page_table | slot_map`) → [`E`(RoPE / 量化)] → `M`(带索引)。设计文档的 4.5 节（RoPE + KV 写）是这条链。

**调度旋钮**：索引 → 数据加载的依赖链是**唯一的串行段**（~2 × 600 周期的 L2 / HBM 延迟）；靠 `R` 行 ILP 和占用率隐藏，所以**先发索引加载、`vmcnt` 等索引、再发 `R` 个数据加载**（两级流水），而不是逐行"索引 → 数据"串行。

**缺口**：**带索引的 `M`**（设计文档的 P4，库里的 `M` 只有仿射地址）；标量地址路径（`readfirstlane` 已有，`soff` 的 SGPR 使用依赖 `Buffer*` 操作的参数，已支持）；整数除 / 取模、64 位地址乘加；sub-dword 存（fp8 / u8 的单字节 scatter：`StoreU16` 之外缺 u8）；`cos` / `sin`（RoPE 用表，所以不需要，但 `uncached` 版本 `general_*_uncached` 现算角度要 `v_sin` / `v_cos`，列为低优先级）；缓存策略位（scatter 写用 `nt` 避免污染，同样是低优先级）。

## 5.10 调度与编译器含义（汇总）

1. **`vmcnt` 流水线深度**：模板的"流水线"就是**每 wave 同时挂起的 `buffer_load` 数**。我们的记分板（`sched/waits.rs`）按发射顺序算 `vmcnt(n)`，所以**只要加载先于计算发射，等待自然变成部分等待**；要避免的是 (a) 加载之间夹 `wait_vm(0)`、(b) list scheduler 把加载下沉到消费者旁边（`Hint::LatencySensitive` 单元和 `ldshoist` 的"加载上提"对 VMEM 也要生效：`sched/hoist.rs` 现在只对 `ds_read` 上提，**对 `buffer_load` 没有上提**，批次 2 要加 `vmload` 上提或在 weave 里把加载单元排前并设为 authored 区域）。目标：每 wave 在途 ≥ 4 KB（占用 16 wave / CU）。
2. **加载次序**：同一线程的 U 个加载按"先发的先用"排列，消费者按同序，使 `vmcnt(U−1), (U−2), …, 0` 逐个放行；不要把最后发的加载放在第一个消费者里。
3. **LDS vs DPP / permlane 归约**：LDS 版 ≈ 300 周期（私有 64 B / wave，无 barrier）、DPP 版 ≈ 60 周期；带宽受限且占用 ≥ 8 wave / CU 时不换（5.4）。**DSL 要补**：DPP 的 `row_shr` / `row_bcast` / `wave_shr` / `row_mirror` 控制，`permlane16_swap`（gfx950 的 `v_permlane16_swap_b32`，现有 `permlane32_swap`）。
4. **XCD 序**：workgroup id 轮转到 8 个 XCD（`blockIdx % 8`）；带宽受限 kernel 的块之间无复用 → 不需要；有跨块复用的（γ / β 向量、cos / sin 表、列归约相邻列块、页表）→ 让复用同一数据的块落在同一 XCD（块号重排，`GRID_MODE=xcd`），γ / β 只有 `N × 2 B`（≤ 16 KB）本身驻 L2，不需要。
5. **`nontemporal` / 缓存策略位**：流式 kernel 实测无差别（5.1）；唯一可能有意义的场景是"每块用同一个小表（γ、cos / sin）+ 大流（x）"，用 `nt` 保住表在 L2。**批次 2 先不做**；DSL 加 `cache: CachePolicy { default, nt, slc, glc }` 到 `BufferLoad` / `BufferStore` 即可（汇编里是 `nt` / `sc0 sc1` 位，gfx950 的语法为 `sc0` / `sc1` / `nt`）。
6. **持久化 vs 按 tile**：按 tile（每 lane b128 × 1，一块 4 KB 输入，grid = ⌈N / tile⌉）等于或优于 grid-stride（需 ≥ 4 块 / CU 才追平）；持久化只在**块启动开销显著**（极小 tile、极大 grid > 100 万块）或需要**跨 tile 状态**（扫描的回看、在线 softmax 的行跨块）时用。
7. **越界尾部**：用 `buffer` 描述符 `num_records`（硬件守卫：读 0、写丢弃），**不生成 tail 分支**；`gpu_verify` 已按"buffer(hw-guarded)"证明（RMSNorm 的证书里有 25 个）。变长行（`cu_seqlens` / ragged）用动态 `num_records`（`buffer_desc_dyn` 已有）。
8. **占用与寄存器**：带宽受限 kernel **以占用为目标**：先定 VGPR ≤ 128（≥ 16 wave / CU）再选 U；`s_setprio` 无益；`sched` 的 `max_vgprs` 压力限制要能按"占用目标"设置（`limit=N`，`Function::sched_opts`），而不是 256。
9. **weave 层**：这些模板的 weave 相位几乎没有自由度（带宽受限，最佳顺序 = "加载 → 计算 → 存"），`AMDC_WEAVE_AUTO`（`weaver-auto.md`）对 RMSNorm 实测无差别（asm 相同 / +0.6 %）；**模板的性能由 (a) 占用率、(b) 加载先发、(c) 在途字节**决定，这三项应作为模板参数（`vec`、`unroll`、`rows_per_wave`、`vgpr_budget`）而不是靠调度器发现。

## 5.11 DSL / stage 库缺口清单（按批次 2 的实现优先级）

| 优先级 | 缺口 | 谁需要 | 备注 |
|---|---|---|---|
| P0 | **带索引的 `M`**（gather / scatter，行索引、页表、slot map），含标量地址路径 | T7、RoPE + cache、MoE 排列、dequant_gather | 设计文档 P4；先做行粒度 |
| P0 | **`R` 的 scope 阶梯**：group（16 / 32 lane）、block（LDS + barrier）；分段 amax | T2 一行一 block、T3、group quant | 用现有 `wave_reduce` 的 LDS 私有区模式扩到块 |
| P0 | 行 softmax stage（`R`(max) → `E`(exp) → `R`(sum) → `E` → `M`）；LayerNorm 的 mean / var | T4、norm 类 | 复用 `softmax.rs` 的数值部分 |
| P1 | `ballot`、`s_bcnt1`、`v_mbcnt`、`readlane` / `writelane`、`ds_bpermute` | T5、T6（门控 top-k、MoE 对齐） | 先做 wave 内 top-k（k ≤ 8） |
| P1 | DPP 行操作（`row_shr` / `row_bcast` / `wave_shr`）、`permlane16_swap` | T3（group 16 / 32 以上）、T5 wave 扫描 | |
| P1 | `S` stage（wave 内 top-k / argmax）、`P` stage（wave / block 扫描） | T5、T6 | 库里没有 |
| P1 | LDS 原子（`ds_add_u32` / `ds_inc`）、全局原子 CAS / min / max / f32 加 | radix 直方图、列归约、解耦回看 | 现在只有 `AtomicAddU32` |
| P2 | 缓存策略位（`nt` / `sc0` / `sc1`）on `Buffer*` | T1 / T7 的优化 | 流式实测无差别 |
| P2 | sub-dword 加载 / 存（u8 / u16）、64 位地址 / 键的整数运算、整数除 / 取模 | T7（非 2 的幂页）、sampling | |
| P2 | `v_rsq_f32` / `v_sqrt_f32` / `v_sin` / `v_cos` / `v_log` 的专用操作 | norm 精度、RoPE uncached、sampling | 现在用 `exp2` / `log2` 组合 |
| P2 | `vmload` 上提（`sched/hoist.rs` 对 `buffer_load`）或"加载单元排前"的 authored 区域模式 | 所有 T1 / T2 / T7 | 5.10 第 1 条 |
| P3 | 解耦回看扫描（64 位原子发布）、radix 多块选择 | 大数组扫描 / 大 n top-k | 批次 2 不做 |

## 5.12 批次 2 的实现计划（访存受限算子）

顺序（每个都带实机对拍 + `gpu_verify --strict` + 带宽验收，目标 ≥ 90 % × `BW_target`）：

1. **流式模板 T1 + 激活 / 门控 / 量化元素算子**：`silu_and_mul` / `gelu_and_mul`、`act_mul_quant`（fp8 per-token 需要 T2 的行 amax）、类型转换（bf16 ↔ f32 ↔ fp8）、residual add。模板参数 `vec`、`unroll`、`rows_per_wave`、`act`。
2. **T2 一行一 wave 的扩展**：`rmsnorm`（bf16 输出）、`layernorm`、`fused add + rmsnorm`（residual 输出）、`gated rmsnorm`、per-token fp8 / int8 量化的独立 kernel、行 softmax；然后**一行一 block**（N > 8 K）。
3. **T3 分段量化**：per-group fp8（G = 128 / 64 / 32）、E8M0 scale + 布局参数；`fast_transpose`；列归约（bias 梯度 / `moe_reduce`）。
4. **T7**：`reshape_and_cache`（fp16 / bf16 / fp8 + per-token scale）、RoPE（NeoX / GPT-J，cached / positions，Q + K 融合 + cache 写）、MoE gather / scatter copy、`dequant_gather`。
5. **T6 + T5 的最小集**：wave 内 `topk_softmax`（experts ≤ 256，k ≤ 8）、MoE 对齐（计数 + 块内扫描 + scatter）；之后 block bitonic top-k / sampling。

每一步在 op-coverage-design（kernel-opt） 的算子清单里登记：stage 组合、模板、`BW_target`、实测、证书；基准程序沿用 `tests/gpu/membw_survey.cpp` 的测法（同进程交替、hipEvent、中位数）并补每个算子的主机驱动。

## 5.13 基准与复现

```
hipcc 或 amdclang++ -x hip -O3 --offload-arch=gfx950 tests/gpu/membw_survey.cpp -o membw_survey
HIP_VISIBLE_DEVICES=4 ./membw_survey 2048     # 工作集 MiB（2048 与 256 各跑一次）
```

输出：`hipMemcpy` D2D、拷贝（宽度 / 展开 / `nt` / 按 tile vs grid-stride × grid 倍数）、只读、只写；"r+w" 为读写字节和除以时间。`rmsnorm_quant_fp8_n4096`：`tests/gpu/rmsnorm_quant_host.cpp`（`rmsnorm_quant_host <co> <kernel> 16384 4096 300`）。