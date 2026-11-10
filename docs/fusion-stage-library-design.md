# 细粒度融合的 stage 组件库：设计

状态：仅为设计（kernel-opt，2026-11-06），不改动任何出厂代码。

相关文档：
- [`fusion-layout-handoff.md`](fusion-layout-handoff.md)：`convert_layout`、orient 搜索、`StageBoundary` / `Node::Stage` / GV-STAGE、融合 MLP；
- [`fmha-dsl-tour.md`](fmha-dsl-tour.md)：FMHA 的组件分层；
- [`layout-tour.md`](layout-tour.md)。

算子清单的输入：内部知识库 `MI_Kernel_Agent/knowledge/aiter-kernel-zoo`（锁定 aiter `475cf0f6`）。
- 共 1 072 个源文件候选，分 12 类：attention / gemm / moe / normalization / quantization / communication / sampling-topk / rope-position /
  convolution / fusion-activation / cache-layout / other-kernel。
- 本文只按"语义上做了什么"分解，不对任何候选做正确性或性能断言。分类本身来自该清单的目录与文件名；未逐个读源码，只抽读了少量文件。

目标：给 agent 驱动的细粒度融合一个**小**的基础 stage 集合，加上组合 / 变体规则。
- agent 在算子图上选融合方案，库给出可验证的 kernel 骨架；
- 每个形状的流水 / 发射编排留给 kernel 自己（这是用户要求：**不进通用库，也不进 IR**）。

---

## 1. 基础 stage 原语与覆盖

### 1.1 九个原语

| # | 原语 | 做什么 | 主要参数（变体轴） | 已有实现 / 落点 |
|---|---|---|---|---|
| **M** | `Move` 搬运 | global ↔ LDS ↔ 寄存器的 tile 搬运，可带索引（gather / scatter） | 方向；源/目的布局（行主、swizzle、padding、preshuffle）；宽度（b32 / b128 / LDS-DMA / MUBUF / TDM）；地址生成（仿射、索引表、页表、im2col）；越界（clamp、buffer 硬件守卫）；环深 | `rows.rs` / `dma.rs` / `bufdma.rs` / `copy.rs` / `shared.rs` / `global.rs` |
| **A** | `MmaLoop` MMA 主循环 | `C += A·B` 沿 K 累加 | atom（dtype：bf16 / f16 / fp8 / fp4 / 混合）；缩放模式（无、per-tensor、per-row/col、32 元素块 E8M0、128×128 blockscale）；orientation（C 或 Cᵀ）；累加器存储（VGPR / AGPR pin）；K 置换 σ | `mma.rs` / `frag.rs` / `orient.rs` |
| **E** | `Map` 逐元素 | 对 fragment 逐元素算一个闭包；可多个输入（同布局） | 一元 / 二元 / 三元；激活（relu / silu / gelu / sigmoid / tanh / exp2）；门控（`act(a)·b`）；仿射（scale、bias、residual add）；成对旋转（RoPE：`(x0,x1)` 配 `cos/sin`） | `reduce.rs` 的转换词汇，零散 |
| **R** | `Reduce` 归约 | 沿行 / 列做 sum、max、sum-of-squares、amax | 轴；作用域（lane 内 → quad / DPP → 跨半波 permlane → wave → block（LDS）→ grid（原子或两趟））；是否带 key mask | `reduce.rs` / `rowsum.rs` |
| **O** | `Online` 在线归约 | 跨 K 分块保持的运行统计量（max / 参考值、和），并按统计量重标定一个**消费者累加器** | 统计量形式（exact max / 厂商两字参考 + 惰性回滚）；重标定目标（O 累加器、l）；KV-split 部分结果 | `softmax.rs`（OnlineSoftmax） |
| **Q** | `Quantize` 量化 | amax → scale → 转换 / 打包 + scale 输出 | 粒度（per-tensor / per-token / per-group(32, 128) / per-block 2D）；格式（fp8 e4m3/e5m2、mxfp8、mxfp4、int8、int4）；scale 格式（E8M0、fp32、fp16）与存储布局（含 preshuffle）；舍入 | P 转换（`reduce.rs` 的 pknorm / scalef32）；其余待建 |
| **S** | `Select` 排序 / 选择 | top-k、argmax、按 key 排序 / 分桶（MoE 路由对齐） | k；规模（wave 内 bitonic、block 内、多趟 radix / 直方图）；输出（值、索引、计数、偏移） | 无 |
| **P** | `Scan` 前缀扫描 / 递推 | 前缀和（MoE 块偏移、采样 CDF）；分块线性递推（线性注意力、gated delta rule、causal conv1d 的状态） | 运算符（+、max、(A,b) 仿射合成）；作用域（wave / block / 多块解耦回看） | 无 |
| **X** | `Exchange` 跨设备通信 | all-reduce、dispatch / combine（MoE EP） | — | **不在本库范围**（见 1.3） |

另有两个复合件。它们不是原语，但出现得太多，值得给出固定写法：
- **`Epilogue` 链**：累加器 fragment 上的 `E*`（scale / bias / act / residual）→ 可选 `R`（行 amax 等）→ 可选 `Q` → `M`（store，目标布局可为另一个 stage 的输入布局）。
- **`Prologue` 链**：`M`（load）→ 可选 `R` + `E`（例如 norm）→ 可选 `Q` → 作为 `A` 的操作数。

### 1.2 组合与变体规则

1. **算子 = 原语的有向无环组合加参数**。同一类算子的"变种"几乎都落在原语参数轴上：dtype、缩放粒度、布局、索引方式、激活闭包、是否量化输出。少数落在组合顺序上：例如 norm 在 GEMM 前（prologue）还是后（epilogue）。
2. **融合边界只发生在原语之间**。边上的张量要么留在片上（寄存器 / LDS，融合），要么经过 HBM（不融合）。片上边的代价用第 2.3 节的 boundary 代价类衡量。
3. **`O` 是唯一带"重标定消费者"语义的原语**。它必须与下游的 `A`（PV）或 `Reduce` 绑定同一个累加器；其余原语之间只有数据依赖。
4. **`S` / `P` 不是逐 fragment 的**：它们需要 block 内或跨 block 的协作。作为融合的一段时，它们自带一个 barrier 相位，这是 StageContract 的 `barrier_phase` 与 `roles`。
5. **量化 `Q` = `R`(amax) + `E`(scale 计算、转换) + 打包**。它单列出来，是因为格式和 scale 布局的组合太多。实现上由 `R` 和 `E` 拼出，接口只有一个。

### 1.3 12 类覆盖表

| 类别（清单条目数） | 典型成员（清单中的源族名） | stage 组合 | 变体主要落在 |
|---|---|---|---|
| attention（296） | mha fwd/bwd、varlen、batch prefill、paged decode（pa_decode）、MLA（含 sparse prefill）、MQA logits、split-KV、linear / gated-delta / kimi-delta attention、hstu | 前向：`M(Q)` → [`M(K)` → `A(QKᵀ)` → `O`+`E`(mask、P 转换) → `M(V)` → `A(PV)`]×K块 → `Epilogue`。split：+ `R`(combine)。paged：`M` 用页表 gather。反向：`A`×4 + `R`(dO·O) + `E`。线性 / delta 类：`A` + `P`(分块递推) | dtype / 缩放；head dim；mask（因果、滑窗、稀疏块表）；GQA/MQA 映射；页表；split 数；MLA 的 latent 维；递推形式 |
| gemm（234） | a16w16、a8w8（per-token / blockscale / bpreshuffle）、a4w4、a8w4、mxfp4/mxfp8、batched / grouped / gather_gemm、splitk、deepgemm、fused_norm_quant_gemm | `M`(A,B) → `A` → `Epilogue`；splitk：+ `R`(grid)；grouped / gather：`M` 带索引；fused_norm_quant_gemm：`Prologue`(R+E+Q) → `A` | atom dtype、缩放粒度、B 的 preshuffle 布局、split-k、batch / group 索引 |
| moe（261） | sorting / align_block_size、topk_gating、fused_moe（1 段 / 2 段）、gather / scatter_copy、moe_reduce、mega_moe、route_quant_scatter、dispatch / combine | 路由：`R`/`S`(top-k) + `E`(softmax/sigmoid 门) → `S`(按专家排序) + `P`(块偏移) → `M`(scatter，可带 `Q`) → 分组 `A`(gate/up) → `E`(silu·mul) [→ `Q`] → 分组 `A`(down) → `M`(gather) + `R`(按 top-k 权重求和)。EP 的 dispatch / combine 属于 `X` | 专家数、top-k、块大小、权重 dtype（a8w4、mxfp4）、1 段 / 2 段融合、是否 EP |
| normalization（31） | rmsnorm、layer / group norm、fused add + rmsnorm、gated rmsnorm + fp8 per-token / group quant、qk_rmsnorm | `M` → [`E`(+residual)] → `R`(sumsq / mean, var) → `E`(归一、γ、β、门控) → [`Q`] → `M` | norm 类型、residual、门控、量化粒度 |
| quantization（21） | quant（per-tensor/token/group）、quant_mxfp4、dequant_gather | `M` → `R`(amax) → `E` + `Q` → `M`(值 + scale)；dequant：`M`(gather) → `E` | 格式、粒度、scale 布局、gather |
| communication（14） | custom / quick all-reduce（含 int4） | `X`（+ `Q` 压缩、`R` 本地归约） | **范围外**：多 GPU、IPC、同步协议，与单 kernel 的 stage 合同不同。只保留 `Q` / `R` 可被其复用这一点 |
| sampling-topk（26） | topk（plain / per-row / select）、topk_softmax（含 group）、sampling | `R`(max) + `E`(softmax) + `S`(top-k)；sampling：+ `P`(CDF) + 随机数 `E` | k、规模、分组、是否带 softmax |
| rope-position（30） | rope（cached fwd/bwd、多布局）、pos_encoding | `M`(x、cos/sin 按位置 gather) → `E`(成对旋转) → `M` | 旋转方式（NeoX / GPT-J 交错）、部分旋转维、位置来源 |
| convolution（20） | conv3d implicit GEMM、causal_conv1d（fwd / update） | 隐式 GEMM：`M`(im2col 地址) → `A` → `Epilogue`。causal conv1d：`M` + `E` 的滑窗乘加，状态更新用 `P`/`M` | 维数、步长 / 膨胀、通道布局 |
| fusion-activation（33） | silu_and_mul、activation、fused_qk_norm_rope_cache_quant、fused_qk_norm_mrope_cache_quant | `E`(act·mul)；qk norm+rope+cache+quant = `R` → `E`(norm) → `E`(rope) → `Q` → `M`(cache scatter) | 激活、门控、是否带 norm / rope / quant / cache |
| cache-layout（7） | kv_cache（reshape_and_cache 等）、trans_ragged_layout | `M`(按 slot 映射 scatter，可带 `Q`) | 页布局、量化、ragged |
| other-kernel（99） | mfma_epilogues、splitk_epilogue、softmax、cross_entropy、sampling、shuffle、dsv4_rotate_quant、inverse_rope_group_quant、binary / unary operator、dcp_topk_merge 等 | 多为上面原语的组合：softmax = `R`+`E`；cross_entropy = `R`+`E`+`M`(gather label)；shuffle = `M`；`*_quant` = `E`+`Q` | — |

覆盖结论（范围：上面 12 类的源族名与抽读文件）：除 communication 外，没有发现需要第十个计算原语的条目。最依赖新原语的是：
- MoE 路由与采样，要 `S` / `P`；
- 线性 / delta 注意力，要 `P` 的分块递推。

这两组目前在本仓库没有任何实现，是实现计划里风险最大的部分（第 5 节）。

---

## 2. stage 接口

### 2.1 StageContract（在现有结构上扩展）

现有 `StageContract`（`crates/amdgpu-dsl/src/lib.rs`）的字段：`live_in` / `live_out`（`StageVal`：名字、值、元素表、位宽、存储）、`lds_freed`、`lds_reused`、`barrier_phase`、`roles`。GV-STAGE 在 region 边界检查它们。另外 `amdgpu-tile/src/contract_stage.rs` 的 `StageBoundary` 已经带了 pinned 寄存器区间和在途计数容忍度。

库化需要的补充只有四项：

| 字段 | 含义 | 由谁检查 |
|---|---|---|
| `live_in[*].layout` / `live_out[*].layout` | 每个跨边界 fragment 的 **TV 布局**（`amdgpu-layout` 的布局对象，而不是只有元素表指纹），外加存储类：VGPR / AGPR(pin) / LDS 区域名 + 布局 / global | 构建期：组合时算 boundary 代价（2.3）；GV-STAGE：元素表一致 |
| `outstanding` | 进入 / 离开时允许的 vm / lgkm 在途数（生产者最坏情况 ≤ 消费者容忍） | GV-STAGE（已在 `StageBoundary` 中，搬进 IR 合同） |
| `scalars` | 跨边界的标量状态（行统计量、scale、索引计数），按"每行一个 / 每 wave 一个 / 每 block 一个"标注归属 lane | 构建期 |
| `resources` | 本 stage 的寄存器 / LDS / barrier 需求上界（估计值，供融合方案取舍，不是证明） | agent 规划时用；RA 与 GV 仍是最终裁判 |

**stage 自身不含流水 / 发射顺序**。它只给出"做什么"和"边界上是什么"。

### 2.2 per-kernel 的部分（不进库、不进 IR）

按用户要求，以下全部由具体 kernel 手工编排，库只提供带合同的积木：
- 每个形状的流水节奏：环深、预取距离、ping-pong 或单组、K 循环展开、tile 尾部剥离；
- 发射顺序：MFMA 与 VALU / LDS 读的交错、DMA 发射点、`setprio`、计数等待的位置；
- 调度器 token（`ldsink` / `wsink` / `hzfill` / `ovl` 等）和 lowering 选项；
- wave 角色划分与占用率（每 SIMD 几个 wave、AGPR 的用法）。

FMHA 这一轮的经验（[`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md)、[`fmha-4wave-design.md`](fmha-4wave-design.md)）说明：同样的 stage 组合，编排不同，性能可以差 10–40 %。所以编排必须留在 kernel 里可改，库不应替 kernel 决定它。

现有 `Pipeline`（`amdgpu-tile/src/pipeline.rs`，几种 `Cadence`）的定位：它是**编排的辅助件**，供 kernel 选用，不是库对 stage 的要求。

### 2.3 组合：boundary 代价类与 orient

两个相邻 stage 之间，生产者的 `live_out` 与消费者期望的 `live_in` 可能布局不同。组合时按代价从低到高选最便宜的转换（与 `handoff.rs` 的 `convert_layout` 一致）：

| 代价类 | 条件 | 指令 |
|---|---|---|
| C0 `Identity / Register` | 每个 lane 已持有所需元素（允许消费者 K 置换 σ，由另一个操作数的读布局吸收） | 无，寄存器改名 |
| C1 `Quad` | 只在 lane 的 4 元组内重排 | DPP `quad_perm` + `v_cndmask` |
| C2 `CrossLane` | GF(2) 线性的跨 lane 置换 | wave 私有 LDS 往返（或 `permlane`） |
| C3 `LDS tile` | 需要块内其他 wave 的数据（行 / 列归约、转置给另一组 wave） | LDS 写 + barrier + 读 |
| C4 `HBM` | 不融合 | store + 另一个 kernel load |

- **MMA → (E/O/Q) → MMA 的交接**：先用 `orient.rs` 在两个 MMA 的 orientation × 操作数角色 × σ 上做 orient 搜索，求最低代价。FMHA 的 S→P 与融合 MLP 的 H 都是 C0。
- **stage 的 orient 自由度**要在合同中声明：MMA 的 C / Cᵀ、`A` 输入能否接受 K 置换。这样组合器才能搜索。

### 2.4 与现有模块的关系

| 现有 | 在库中的角色 |
|---|---|
| `amdgpu-layout`（布局代数、`convert`） | 合同中 TV 布局的表示与代价计算 |
| `amdgpu-tile`：`rows` / `dma` / `bufdma` / `copy` / `shared` / `global` | `M` 原语的实现件 |
| `mma` / `frag` / `orient` | `A` 原语 + orient 搜索 |
| `reduce` / `rowsum` / `softmax` | `R` / `O` / `E`（P 转换）的实现件 |
| `epilogue` | `Epilogue` 复合件（目前只有 FMHA 的 bf16 / partial 两种） |
| `contract_stage` / `handoff` | 合同与 boundary 转换 |
| `Node::Stage` + GV-STAGE | 合同挂在 region IR 上，由验证器在边界检查 |
| `pipeline` / `lane::Roles` | **编排辅助件**（per-kernel，不属于 stage） |
| 融合 MLP（`mlp_fp8_ws_{fused,g1,g2}`） | 现有唯一的多 MMA 融合样例，即 `A` → `E` → `Q` → `A`。它是"md5 不变地重写成 stage 组合"的第一个对象 |

---

## 3. 轻量算子图

**节点**：stage 实例，即原语加参数。例如 `A{atom: mxfp8_32x32x64, scale: e8m0_block32, orient: Cᵀ}`、`E{f: silu·mul}`、`Q{fmt: fp8, gran: per_token, scale: fp32}`。

**边**：张量，带以下信息：
- 形状与 dtype；
- 逻辑布局（行主 / 转置 / preshuffle / 页表）；
- 生命周期（生产者 → 消费者集合）；
- 是否允许留在片上，以及以什么作用域留：wave、block、grid。

**agent 的流程**：
1. 从框架级算子（例如 `rmsnorm → quant → gemm → silu_mul → quant → gemm`）展开成原语图（第 1.3 节的组合规则）。
2. 选**融合切分**：把图切成若干 kernel。每条被切断的边是 C4（HBM），未切断的边按 2.3 求最低代价类。
   约束是寄存器 / LDS 预算（合同的 `resources` 估计）和作用域要求：`S` / `P` 跨 block 时必须切分，或用 grid 级协作。
3. 对每个 kernel 生成**骨架**：
   - 一段 Rust 构建函数，内容是 stage 调用加合同，以及每条片上边的转换（自动选出的代价类）；
   - 留出编排位，即 TODO 注释：流水节奏、发射交错、调度 token，由人或 agent 按形状填写；
   - 骨架可以直接 strict verify（合同与协议）和跑正确性，性能靠后续编排。
4. 输出图与切分的文本描述，供审阅：每个 kernel 的 stage 列表、片上边的代价类、资源估计。

图的格式用 JSON 或 Rust 数据结构即可，不需要新的 IR。它只是骨架生成器的输入，不进入编译器。

---

## 4. 例子

### 4.1 FMHA 前向（现有 `_vs128`，用 stage 表达）

```
M(Q→VGPR) ;
loop t:  M(K_t, KD_t → LDS 环)  M(V_t → LDS 环)            -- 编排：DMA 时机、环深（per-kernel）
         A{fp8 32x32x64, e8m0 K-scale, orient=Sᵀ? → S}      -- C0 交接
         O{厂商两字参考, 惰性回滚, rescale O}  E{P 转换 pknorm+scalef32}  R{smfmac 行和}
         A{fp8, PV, O 累加于 AGPR/VGPR}
Epilogue{ E(1/l) → bf16 → M(经 LDS 转置的 store) }
```
- 各边：S → P 是 C0；P → PV 的 B 操作数是 C0，K 置换 σ 由 V 的 tr 读吸收；O → epilogue 是 C2（wave 私有 LDS 转置）。
- 编排件：`TwoPhaseOffset`、两组 ping-pong、`ldsink` / `wsink`、bias 单次 MFMA 等，全部是 per-kernel。
- KV-split 变体：`Epilogue` 改为写部分结果（O、m、l），再接一个 combine kernel `R`；这条边是 C4。

### 4.2 GEMM + scale / bias / act / quant epilogue

```
M(A,B) ; A{mxfp8 blockscale 128x128, C 于 AGPR} ;
Epilogue{ E(×s_a×s_b) E(+bias) E(silu) R(行 amax) Q(fp8 per-token, fp32 scale) M(值 + scale) }
```
- 行 amax 需要一行的所有 lane，按 C 布局决定：lane 内 + permlane 是 C2；跨 wave 是 C3。合同里 `Q` 的 scale 归属 lane 由 `R` 的作用域决定。
- 现有 `gemm_mxfp8_bs_v4` 的 epilogue 是 `E(scale)` + `M`，是这个链的前缀。

### 4.3 融合 fp8 MLP（现有 `mlp_fp8_ws_*`）

```
M(X) ; A1{fp8, Hᵀ = W1 Xᵀ} ; E{relu(α·)} ; Q{fp8 打包, 单位 scale} ; A2{fp8, Yᵀ = W2 Hᵀ, W2 预置换 σ} ; M(Y)
```
- A1 → A2 是 C0：orient 搜索的结果，σ 由预置换的 W2 吸收。
- weight-stationary / persistent / slotfill 是编排。
- 这是 stage 化的第一个验收对象（第 5 节 P1）。

### 4.4 RMSNorm + 量化（新）

```
M(x 行 → VGPR) [E(+residual) → M(residual 写回)] ; R{sumsq, 行, wave 或 block} ; E{x·rsqrt(mean+ε)·γ} ;
R{amax, 行或组} ; Q{fp8 per-token 或 group(128), fp32/E8M0 scale} ; M(值 + scale)
```
- 一行在一个 wave 内时，两个 `R` 都是 C2（DPP / permlane）；行宽 > 1 wave 时是 C3。
- 与下游 GEMM 融合（`fused_norm_quant_gemm`）时，这条链成为 `A` 的 `Prologue`，`Q` 的输出布局直接是 `A` 的操作数布局（C0 或 C3）。

### 4.5 RoPE + KV-cache 写（新）

```
M(q,k,v 行) ; M(cos/sin 按 position gather) ; E{成对旋转 (NeoX 或交错)} [; R(amax) Q(fp8 cache)] ;
M{scatter: slot_mapping → 页表位置, 布局 = cache 布局}
```
- 成对旋转需要 `(x_i, x_{i+d/2})` 或 `(x_{2i}, x_{2i+1})` 同 lane：由加载布局保证（C0），否则 C1。
- 若与 QK-norm 融合（`fused_qk_norm_rope_cache_quant`），在前面接 4.4 的 `R` + `E`。

---

## 5. 实现计划

验证总原则：
- 每一阶段的新表达，先在**已有 kernel 上 md5 不变**地复现，再引入新功能；
- 新 kernel 走 strict gpu-verify（含 GV-STAGE）+ 对 CPU 参考的精度 + `port_gate` 式的确定性 / 压力检查；
- 性能结论按同轮交替 A/B 的规范给出（真实数据）。

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** 词汇与合同 | 第 1 节原语表进文档。`StageContract` 增加 TV 布局 / 在途计数 / 标量归属 / 资源估计（2.1），GV-STAGE 增加对应检查。不改任何 kernel 输出 | 全部出厂 kernel md5 不变；新增 GV-STAGE 单元测试 |
| **P1** 现有 kernel 的 stage 化重写 | 融合 MLP → GEMM v4 epilogue → FMHA `_vs128` / `_t128` / `_kvsplit`，依次把现有构建代码改写成"stage 调用 + 合同 + per-kernel 编排"三段 | **md5 逐字节不变**（`tools/md5_kernels.sh`）；strict 证书数不变；`port_gate` / `gemm_gate` 通过 |
| **P2** `E` / `R` / `Q` 的通用件 | 激活 / 门控 / 仿射闭包库；行 / 列归约（wave、block）；量化（per-token、group、mxfp8 / mxfp4、E8M0 / fp32 scale） | 先做 RMSNorm+quant kernel（4.4）和 GEMM 的完整 epilogue 链（4.2）：精度对 CPU 参考，与 aiter 对应实现同轮比较 |
| **P3** 算子图与骨架生成 | 第 3 节的图描述、切分代价估计、骨架生成（Rust 构建函数 + 编排 TODO） | 由图重新生成 P1 的 kernel 骨架，填入原编排后 md5 不变；另由图生成 4.4 / 4.5 的骨架并通过 strict verify |
| **P4** 带索引的 `M` | gather / scatter、页表、slot mapping、im2col 地址生成 | RoPE + KV-cache 写（4.5）、paged decode 的 KV 读取原型：精度与边界（越界、空页）测试 |
| **P5** `S` / `P` | wave / block 内 top-k、按 key 分桶、前缀和；分块线性递推 | MoE 路由（topk_softmax → sorting → align）与采样的原型。风险最高：仓库里还没有这类协作原语的验证经验 |
| （范围外）`X` | 跨设备通信 | 不在本库；仅保证 `Q` / `R` 可被其复用 |

---

## 6. 关键决定（摘要）

1. **九个原语**（M / A / E / R / O / Q / S / P / X，其中 X 范围外）加两个复合件（Epilogue / Prologue）。按清单的 12 类分解时，除通信外没有发现需要更多计算原语的条目。
2. **合同描述"是什么"，编排留给 kernel**。流水节奏、发射交错、调度 token、wave 角色不进库也不进 IR；`Pipeline` 只是可选的编排辅助件。
3. **组合靠 boundary 代价类 C0–C4 与 orient 搜索**。片上交接优先找 C0：寄存器改名，K 置换由另一个操作数吸收。
4. **算子图是骨架生成器的输入，不是新 IR**。agent 在图上选切分，拿到可直接 strict verify 的骨架，再手工编排。
5. **先 md5 不变地重写现有 kernel，再扩展**。融合 MLP → GEMM → FMHA 依次 stage 化，作为接口设计的验收。

---

## 7. 三层结构：Stage / Weaver / 后端（已批准，实施中）

第 2.2 节说"编排留在 kernel"。这一节把它落成接口：stage 不再直接发射指令，而是返回**发射单元**；kernel 用 **weaver** 写编排；后端（寄存器规划、调度 pass）作为 weave 区域上的可选策略。

### 7.1 第一层：stage = 编译期按形状实例化 → 发射单元

stage 的参数（形状、dtype、slot、tile 号等）在调用点给定。实例化后得到一组 `Unit`（`crates/amdgpu-tile/src/weave.rs`），每个单元带：

| 字段 | 含义 |
|---|---|
| `emit` | 一个闭包：发射一小段自洽的 IR（一个 QK 块、一个 PV 块、一条 DMA 指令……） |
| `deps` | 必须先发射的单元（同一组内的索引），由捕获语句的事实自动推导（`facts::must_order`）；单元在构造时就捕获，值直接从闭包返回（[`weave-capture.md`](weave-capture.md)） |
| `class` | 资源类：`Mfma` / `Valu` / `LdsRead` / `Dma` / `Salu` / `Branch` / `Setup` |
| `latency` | 名义的发射→结果延迟（周期） |
| `regs` | 寄存器画像：live-in / live-out / temps（dword，估计值），结果能否放 AGPR |
| `hint` | 交错提示：`Skeleton`（骨架，如 MFMA）/ `Filler`（可填缝的独立工作）/ `LatencySensitive`（宜早发：读、拷贝）/ `Anchor`（相对阶段边界固定：setup、分支） |

现已改成单元形式的件（P1 第一步，按声明顺序发射，与原代码逐字节相同）：

| 件 | 单元 |
|---|---|
| QK（写入 S 变量） | `qk.setup`（slot 绑定的 K / K-scale 分区；`biasonce` 的那一条 bias MFMA）、`qk.block{0..3}`（K 片段 + scale 字 + 两条缩放 MFMA）、`qk.end`（无 IR） |
| PV | `pv.setup`（V 分区、两个 P 片段）、`pv.rowsum`（两条 smfmac 行和）、`pv.h{hh}e{e}`（一个 tr 读 + 一条 MFMA，8 个） |
| buffer 形式的 K / V / K-scale DMA | `dma.<buf>.setup`（soffset、M0 的 readfirstlane）、`dma.<buf>.{q}`（每条 DMA 指令一个，各自在 slot 的写标记下） |
| softmax | `decide`（含惰性回滚分支，整体一个 `Branch` 单元）、`convert`（P 字） |

### 7.2 第二层：weaver

```rust
Weaver::new(units)            // 一个或多个 stage 的单元（Units::append 合并，可加跨组依赖）
    .phase("vs128.m_phase")   // 标签：报告 / 错误 / 后端的区域名
    .order(ids)               // 显式顺序：权威
    // 或 .seq()               // 声明顺序
    // 或 .spread(skel, fillers, per)  // 可选的自动策略：骨架单元之间均匀插入填充单元
    .emit(b);                 // 先检查：每个单元恰好一次、依赖在前（拓扑序），再发射
```

- **显式顺序是权威的**。自动策略（`spread`，以后的调度器策略）只是生成一个顺序，同样要通过拓扑检查。
- **每个形状的参数留在调用点**：选哪个顺序、哪个策略、每个骨架后插几个填充，库不做决定。
- **输出是普通的 region IR**：与今天一样走 gpu_verify（含 GV-STAGE）、lowering、调度、寄存器分配。weave 只决定 IR 中语句的先后。
- **phase / role**：weave 发生在 `Pipeline` 的阶段回调里（`m_phase` / `v_phase` / `window`），role 仍由 `Roles::each` 给出。weaver 不跨阶段边界（barrier）移动单元。

### 7.3 第三层：后端

- 寄存器规划（backend-llvm-tail，分支 `weaver-regs`）：
  - pin（`pin_vgpr_at` / `pin_agpr*`）；
  - 复用提示 `reg_share` 与硬分离 `reg_separate`；
  - 区域上限 `reg_cap(region, vgprs)`，按 weave 区域 / stage 名，按分配后的真实占用检查；
  - 寄存器环 `reg_ring`；
  - `region_report` 给出每区域的真实峰值。

  单元的 `regs` 估计与它对照。
- 调度 pass（sched）：现有的 `ldsink` / `wsink` / `hzfill` / `ovl` / search 改造成 weave 区域上的可选策略。kernel 显式顺序给出的区域保持不变，或只做后处理。

### 7.4 `_vs128` 新旧对照

| 旧（直接发射） | 新（单元 + weaver） |
|---|---|
| `m_phase`：`set_prio(1)`；`self.pv(b, cur, &w)`；`if staged { self.qk(b, next, Vars(&s), bias) }` | `set_prio(1)`；`let mut u = self.pv_units(cur, w); if staged { u.append(self.qk_units(next, s, bias), &[]) }`；`Weaver::new(u).phase("vs128.m_phase").seq().emit(b)` |
| `v_phase`：`set_prio(0)`；`decide_lazy`；`convert(0..4)` | `set_prio(0)`；单元 `decide`、`convert`；`Weaver::new(..).phase("vs128.v_phase").seq().emit(b)` |
| stage：`kdma.issue_buffer(...)`、`vdma.issue_buffer(...)` | `issue_buffer_units(...)`：setup + 每条 DMA 一个单元，可被 weave 进 QK 的 MFMA 之间（例如以前试过的 vlate），而不用改 stage 代码 |

验收：`.seq()` 的 weave 与旧代码**汇编逐字节相同**（全部出厂 kernel 的 md5 对照）。之后，同一份单元可以用别的顺序 weave 出实验变体，而 stage 代码不变。

---

## 8. 实现状态（已在 Main）

| 阶段 | 内容 | 验证 |
|---|---|---|
| P0 | `StageContract` 扩展：<br>• `StageVal.layout`（TV 布局文本）+ `storage` 检查；<br>• `scalars_in/out`（含 owner）；<br>• `outstanding`（vm/lgkm 入口/出口容忍）；<br>• `resources`（barriers 精确检查，寄存器/LDS 为估计）；<br>• `ANY_PHASE`（版本化/角色分支与流水循环内的 stage）。<br>GV-STAGE：池化常量不算跨 stage 读；在任何 stage 之外绑定的值无需生产者。IR 文本 codec 覆盖新字段 | `crates/amdgpu-compiler/tests/stage_ir.rs`、`crates/amdgpu-kernels/tests/ir_text.rs` |
| P1 | `amdgpu-tile/src/stage.rs`：`Prim`、`StageInst`（units + contract + handover + prebind）、`merge` / `compose`、registry（17 个 key，`tests/stage_registry.rs`）。<br>出厂 kernel 均由 stage 实例构成，并位于 `Node::Stage` 区域内：<br>• MLP：`gemm1_act` / `gemm2`；<br>• GEMM v4：`gemm.ab_stage` / `gemm.tile`；<br>• FMHA：`kv.stage` / `v_phase` / `m_phase`（`mma.pv` + `mma.qk` 合并）/ `window`。<br>交接方式：<br>• FMHA 的共享 slot 偏移以 `SlotIdx::offset_scalars` 声明为标量交接，不预绑定（预绑定会改变 IR 顺序与汇编）；<br>• GEMM 用 `prebind`。 | 汇编逐字节相同；证书 24/100/0；strict 0 错误（`_vs128` / `_kvsplit` stage-handoff 12，`_t128` 4） |
| P2 | `amdgpu-tile/src/prims.rs`：E（act / affine / bf16 解包）、R（lane + wave 归约）、Q（`fp8_scale` / `fp8_pack`）。<br>新 kernel：<br>• `rmsnorm_quant_fp8_n<N>`（4 个 stage）；<br>• GEMM epilogue 链 `mlp_fp8_ws_g1q_{relu,silu}`。 | 对 CPU 参考 bad=0；数字见 [`current-status.md`](current-status.md) §4.8 |
| P3 | 算子图、切分、骨架生成（sched，`amdgpu-graph/src/opgraph/`，[`opgraph.md`](opgraph.md)）：出厂 FMHA / GEMM / MLP / RMSNorm 都从图生成骨架，slot 里只放编排 | 骨架 + 原编排 → 汇编逐字节相同 |

**清理**（2026-11-07）：删除了重构后不再使用的代码。

| 位置 | 删除的内容 |
|---|---|
| `Pipeline` 开关 | `TwoSlot`、`ring_var_last`、`mid_lds_drain`、`late_stage_b` / `PipeBody::defer_stage`、`top_vm`、`prologue_vm`、`pre_staged`、`ablate_no_sync`、`tile1_after_barrier`、`key_late`；`RegStaged` 现在固定为"两 tile 一轮、tile 1 晚发" |
| softmax | `decide_scaled` 与其种子路径 |
| 其他未用的 pub 项 | `quant_fp8_row`、`with_hint` 等 |

被删的实验记录保留在 git tag `archive/*` 中。