# 当前状态总览（authoritative snapshot）

本文件是仓库的**唯一权威当前状态快照**，对应 **v7.0**（2026-11 仓库整理之后：实验变体、实验开关与旧 kernel 已删除，crate 按
IR / tile / kernels / compiler 分层）。数字来自 MI355X / gfx950（`mi355-gpu-53`，GPU 6）上的同轮硬件测量，复现命令见 §8。
厂商事实在 [`vendor-spec.md`](vendor-spec.md)，设计规格在 [`design.md`](design.md)；历史逐轮记录与各代设计研究在
[`archive/`](archive/README.md)（其中的数字、kernel 名与开关只代表当时的状态）。

---

## 1. 一句话状态

Rust 类型化 kernel DSL + 自研编译器（**寄存器分配与指令调度始终是我们自己的**，LLVM 只做 post-RA tail），
在 gfx950 上运行厂商 ABI 兼容的 MXFP8 FMHA、MXFP8 blockscale GEMM 与融合 MLP：

- **出厂 FMHA = v3 布局层 kernel**：默认路径 `fmha_fwd_hd128_mxfp8_auto`（主机侧按形状选择）由
  `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}` 组成（`crates/amdgpu-kernels/src/fmha/`）。每个地址、DMA、epilogue 与
  KV-split 暂存都由 `amdgpu-layout` 的布局推导，`amdgpu-isa` 的 atom 描述指令，`amdgpu-tile` 的组件生成 IR；主机合同由张量布局导出。
  asm md5（`--backend own`）：`_vs128` `8938adfa`、`_t128` `d7e02b0f`、`_kvsplit` `41f151fc`、`_kvsplit_combine8h` `24a95a51`（Main `c4d59df8`）。
- **FMHA 性能（真实数据，`REF_DATA=normal`，2026-11-06，GPU 5）**：`port_gate`：B1H8S2048 默认路径（KV-split）**742 TF/s（1.03× 厂商）**；
  铺满网格 B4H8S2048 **1723（0.904×）** / B1H8S8192 **2414（0.988×）**；`_vs128` 每 CU 64 blocks **648（0.894×）**；`_t128` B2H8S2048 layout 中位数 **1110（0.862×）**。
  25 形状矩阵（[`perf-matrix.md`](perf-matrix.md)）：0.897–1.100×，几何平均 **0.980**。调色板数据（`REF_DATA=lcg`，旧口径）下 softmax 几乎每 tile 回滚、掩盖了
  快速路径成本，旧的 0.92× 口径偏乐观；这一轮的改进（懒更新、≥ 4 tile 用厂商配方、真实数据分派、bias 单次 MFMA 且无零片段拷贝、`ldsink` / `wsink`）
  与否定结果（copy 路径、3 槽环、厂商 copy 形态、重平衡）见 [`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md) 与 [`lgkm-lookahead.md`](lgkm-lookahead.md)。
- **MXFP8 blockscale GEMM**：`gemm_mxfp8_bs_v4`（`crates/amdgpu-kernels/src/gemm/`）对照 aiter PR #6696 4-wave pin-AGPR，4096×4096×8192：
  **0.988–0.994×** 厂商（`tools/gemm_gate.sh` PASS，§4.7）；asm md5 `14df28e4`。
- **融合 MLP**：`mlp_fp8_ws_{fused,g1,g2}`（`crates/amdgpu-kernels/src/mlp/`），布局层 P 交接证明见 [`fusion-layout-handoff.md`](fusion-layout-handoff.md)。
- **AGPR 已支持**（寄存器类 A1–A16、acc_cd MFMA、`ds_read_b128 → AGPR`、`v_accvgpr_*`、统一 512 寄存器预算与 hazard）：[`agpr.md`](agpr.md)。
- **主机侧生成**：`amdc gen-host` 从 kernel 合同生成 C++ 启动/ABI 头与 JSON（[`host-gen.md`](host-gen.md)）。
- 精度：**厂商参数校准（vendor parity）**——每个形状、每条路径同轮同输入下 mean_err ≤ 1.10× 厂商且 bad ≤ 厂商（§4.3）；确定性与 split 压力全过。
- `cargo test` **87 个测试目标 / 366 项全绿**。
- 严格 GPU 语义证书：`--shipped` **24 trusted assumes / 100 trusted contracts / 0 unproven**；GEMM v4 **3 / 19 / 0**；`mlp_fp8_ws_fused` **0 / 9 / 0**。

---

## 2. 编译器流水线

```
DSL(Rust) ──► typed IR（区域树：Loop / If / Region；Op 89 种 / Stmt 50 种）──► IR 验证器 + gpu_verify（共享符号算术 amdgpu-sym + 协议）
            ──► lowering（profile 选项：scalarize / scalarize2 / waveuni / imm / edgepad …）──► 我们的 MIR（124 个 MOp，属性表 op_attrs）
            ──► 我们的调度（块内 list scheduler、LDS 读提升、slotfill、sched_group 相位）
            ──► 我们的线性扫描 RA（两阶段 + LDS spill + S4 描述符类 + AGPR 类）
            ──► post-RA MIR 导出（物理寄存器、已排序）
                 ├─ llvm-tail：llc -start-before=gcn-create-vopd … 只跑 tail pass
                 └─ own：我们自己的 hazard / wait / descriptor 实现
            ──► .s ──► ROCm clang 只做汇编 ──► .co ──► HIP 启动（主机 ABI 由 amdc gen-host 生成）
```

| 环节 | 归属 | 说明 |
|---|---|---|
| DSL / IR / 验证 | 自研 | 类型化、无 unsafe、区域树（[`region-ir.md`](region-ir.md)）、循环变量顺序语义（[`region-ir.md`](region-ir.md)）；布局层在构建期证明覆盖/连续/操作数匹配 |
| Lowering / MIR | 自研 | 每个 MOp 的属性在一处（[`op-attributes.md`](op-attributes.md)）；一致值进 SGPR；死值消除含 SALU / `readfirstlane`（SCC 活性） |
| **寄存器分配** | **自研** | 线性扫描、两阶段（path-aware reaching-writer 检查）、LDS spill、S4 描述符类、AGPR 类与统一 512 预算 |
| **指令调度** | **自研** | 块内 list scheduler、LDS 读提升（`hoistchain`）、`slotfill`（GEMM v4）、依赖图 / 效果模型 / 等待模型（[`scheduling.md`](scheduling.md)） |
| hazard / wait / descriptor | llvm-tail 或 own | 两个后端指令流逐条一致（`tail_invariance` 守护） |

编译策略按 kernel profile 固定（`crates/amdgpu-compiler/src/lib.rs`）：FMHA `_vs128`/`_kvsplit` = `dce,cmpfold,scalarize,retarget2,imm,sgpr,zeroinit`，
`_t128` 另加 `addrfold,saddrlds,edgepad,salubr`；GEMM = `dce,cmpfold,lgkmn,zeroinit,scalarize,scalarize2,waveuni,edgepad,kabatch,imm` + M0 复用。
保留的调试开关只有 `AMDC_SCHED`（0 / 1 / ld）与 `AMDC_SCHED_OPTS`、`AMDC_SCHED_VERBOSE`、`AMDC_RA_DEBUG_FILE`、`AMDC_KEEP_MIR`、`AMDC_BACKEND`、
`AMDC_GPU_VERIFY`、`AMDC_DIALECT`、`AMDC_LLC_ARGS` 与 LLVM 定位变量。

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`）。ROCm 10 不自带 `llc`，用
`tools/llvm_build_llc.sh` 从同一 commit 链接 `libLLVM.so` 得到；ROCm 7.0.2（`llvm20`）作为备选方言。
找不到 pin 的工具目录时 **llvm-tail 直接报错**，不静默降级。

**指令覆盖**：124 个 MOp 全部有 own emit；其中 106 个在 `llvm20.tbl` **和** `llvm24.tbl` 都有行。
LDS-DMA、`buffer_load … lds`、DPP、`s_cselect`、`s_load_dwordx4` 等为 own-only（使用它们的 kernel 自动选 own 后端）。

### 2.1 GPU 语义验证（`gpu_verify`，`AMDC_GPU_VERIFY=off|warn|strict`）

编译器**感知 GPU 指令语义**：在 DSL IR 层做符号区间分析（共享的 `amdgpu-sym`：多项式 + 约束上下文 + 循环不动点 + 合同 + 分支事实），在最终 MIR
层检查寄存器类别与**路径敏感**的等待/生命周期。文档：[`gpu-semantics-verifier.md`](gpu-semantics-verifier.md)（模型与边界）、
[`gpu-semantic-protocol.md`](gpu-semantic-protocol.md)（四轴协议）、[`sym.md`](sym.md)（共享符号分析）。

```bash
amdc gpu-verify <kernel>|--all|--shipped [-v] [--strict]   # 报告 proven/发现/trusted 清单/证书
```

**已证明（出厂 v3，`--strict` 0 错误 0 警告）**：

| 类别 | 证明内容 | 出厂覆盖（`_vs128` / `_t128` / `_kvsplit` / combine） |
|---|---|---|
| **LDS** | 每次 DS 访存/DMA（含 64 lane）在声明段内；环形/双缓冲槽位由不动点证明；无下溢/回绕 | 760 / 336 / 760 / – |
| **LDS tile 生命周期** | tile 分区/别名、produce/consume/version（发布于 barrier）、跨角色 stale/race/pending；周期门；计数版 `vmcnt(n)` 完成规则 | 299,520 / 36,800 / 299,520 次读 |
| **全局/buffer** | 访问落在声明 extent 内（`arg_tensor` + `ka_is/grid_is/ka_one_of`）；描述符不得高估 `num_records` | `crates/amdc/tests/contract_host_truth.rs` 校验合同与 host 公式一致 |
| **VGPR/SGPR/AGPR** | 保留寄存器、元组对齐、上限；MFMA A/B 与 D 不重叠；AGPR 统一预算（arch VGPR + AGPR ≤ 512） | 全部 kernel 默认强制 |
| **寄存器生命周期** | 路径敏感 use-before-wait（任一路径、含循环携带在途值） | 默认强制 |
| **barrier 协议** | 波分歧 if/else 两侧动态 barrier 数相等 | 18 / 8 / 18 |

区间分析的位级规则（`x ^ y`、`x & m` 的可能置位位）与盒子检验都有攻击测试（`crates/amdgpu-compiler/tests/gv_bits.rs`、`sym_branch.rs`）。

**信任预算（证书逐条列出）**：`--shipped`（v3 四个）24 / 100 / 0；`gemm_mxfp8_bs_v4` 3 / 19 / 0；`mlp_fp8_ws_fused` 0 / 9 / 0。
每次运行打印 `certificate: N trusted assumes, M trusted contracts, K unproven`。

**未证明/范围外（诚实清单）**：跨工作组数据竞争与别名、i32 回绕（A3 假设）、从内存加载的指针/索引、safe-asm 块内部访存、
tile 内字节级重叠、角色数 2 / 迭代数相等假设；`gpu_protocol` 的模拟器仍按线性化下标工作。

**v4 协议模型**：**到达**（`s_barrier`）、**完成**（`wait_vm/wait_lds/fence_release`，`vmcnt(n)` 按"其后至少 n 条更新的 LDS-direct DMA"计数）、
**可见**（LDS 经 barrier；全局 release→barrier→acquire + `sc1`）、**代际**（仿射/周期 version-slot 代数 + 周期门）；四条义务 O1–O4 静态 discharge，
判不出的进可计数信任账本。全局内存实例（`buf_region/...`）与 last-arriver 融合 combine 模式见 [`last-arriver-memory-model.md`](last-arriver-memory-model.md)。

---

## 3. DSL、库与 crate 地图

| 维度 | 内容 |
|---|---|
| 入口 | `FnBuilder` → `Function`（kernarg ABI；kernel 体是区域树，`Function::stmts()` 是其线性化） |
| 值模型 | `ValueId` / `VarId`（循环变量）/ `Opnd` / `DType`（`Frag2/4/8/16` 等）；表达式层 `I32/F32/Cond/Ptr` 与叶子 `ev/varx/ox`（[`dsl-expression-layer.md`](dsl-expression-layer.md)） |
| 计算 | scaled 32x32x64 与 16x16x128 f8f6f4 MFMA（两者都可用 op_sel 选 E8M0 字节）、C 操作数形式、unscaled MFMA、`v_smfmac`、f16/bf16 MFMA；AGPR 形式 |
| 向量/转换 | VALU int/fp、`pk_*`、`perm`、`permlane32_swap`、DPP `quad_perm`、`pknorm`、`scalef32`、`cvt_pk_fp8`、`cvt_pk_bf16`、`exp/rcp/log`、`max3`、`readfirstlane`、`v_accvgpr_*` |
| 内存 | FLAT `global_load/store`、`global_load_lds`（含 SADDR）；MUBUF 描述符、`buffer_load/store x1/x2/x4`、`buffer_load … lds`；LDS `ds_read/write_b32/b64/b128`、`ds_read_b64_tr_b8` |
| 控制流 | `begin_loop`/`end_loop`、`begin_if`（wave 级）及闭包形式 `loop_`/`if_`/`if_else`/`produce_region`/`stage`、`Stmt::VarAsm(N)` 桥接 safe-asm |
| 布局层 | `amdgpu-layout`（`Ix`、`IntTuple`、`Layout` 代数、GF(2) `Swizzle`、TV 布局、证明）；入门 [`layout-tour.md`](layout-tour.md) |
| tile 层 | `amdgpu-tile`：`TiledCopy`/`TiledMma`/`partition_*`、LdsPlan/标记、Pipeline、softmax、行 DMA（`rows`）、合同（[`dsl-tile-abstractions.md`](dsl-tile-abstractions.md)）；stage 库（`stage.rs`：`StageInst` + registry + `merge`；`prims.rs`：通用 E/R/Q）、发射单元与 weaver（`weave.rs`）、算子图与骨架生成（`opgraph/`）——[`fusion-stage-library-design.md`](fusion-stage-library-design.md)、[`opgraph.md`](opgraph.md) |
| ABI | 与厂商**逐字节兼容的 656 字节 kernarg + 网格/数据布局**；主机侧头由 `amdc gen-host` 生成 |

**crate 地图**（全部 `#![forbid(unsafe_code)]`、零第三方依赖）：

| crate | 内容 | 依赖 |
|---|---|---|
| `amdgpu-sym` | 共享符号算术（`Q`/`Poly`/约束上下文/区间/位支撑分析） | – |
| `amdgpu-layout` | 布局代数 | sym |
| `amdgpu-isa` | gfx950 copy / MMA atom | layout |
| `amdgpu-safeasm` | safe-asm 窗口（[`safe-asm-design.md`](safe-asm-design.md)） | – |
| `amdgpu-dsl` | IR（区域树）、`FnBuilder`、表达式层、DSL IR 打印与结构相等（`print.rs`） | sym、safeasm、layout |
| `amdgpu-tile` | 全部 tile 抽象（TiledCopy/TiledMma/partition、LdsPlan、Pipeline、softmax、frag、行 DMA、合同、handoff）、stage 库与通用原语、weaver、算子图 / 骨架生成 | dsl、layout、isa |
| `amdgpu-kernels` | 全部 kernel：`fmha/`（v3 + FMHA I/O）、`gemm/`、`mlp/`、`norm/`，每个都是"算子图（`graph.rs` + `*.graph.json`）→ 生成的骨架（`skeleton_*.rs`）→ slot 实现 + stage 实例"；`probes/`、`verifier_cases/`、`ir_proof.rs`、注册表 `registry.rs` | dsl、tile、layout、isa、safeasm |
| `amdgpu-compiler` | lowering、调度、RA、hazard/wait、emit、LLVM MIR 桥、`gpu_verify`/`gpu_protocol`/函数证明、MIR 打印（不依赖 kernels/tile） | sym、dsl、safeasm |
| `amdc` | CLI（emit / check / list / print-ir / backend / simmodel / gpu-verify / gen-host）+ 库 `amdc::hostgen` | dsl、compiler、kernels、tile |

需要具体 kernel 的测试在 `crates/amdgpu-kernels/tests/`（主机合同对照在 `crates/amdc/tests/`）；编译器自身的测试只用测试内构造的 kernel。

---

## 4. Kernel 状态

### 4.1 默认出厂路径：`fmha_fwd_hd128_mxfp8_auto`

| 形状类 | 选择 | v7.0 门禁 |
|---|---|---|
| **B1H8S2048**（基准） | KV-split s=4 + `combine8h` | **685.9 TF/s（1.27× 厂商）** |
| 铺满网格 B4H8S2048 / B1H8S8192 | `_vs128` | **1543.9（0.918×）** / **2206.4（0.921×）** |
| 中等形状 | `_t128` / split（guard 选择） | B2H8S2048 `_t128` **1067.4（1.05×）** |
| 单 kernel 每 CU（64 blocks） | `_vs128` | **493.1（0.917×）** |

### 4.2 Kernel 清单（`amdc list` 41 个 + 按名的出厂 kernel）

| Kernel | 状态 |
|---|---|
| `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}` | **出厂**（`crates/amdgpu-kernels/src/fmha/`：算子图 + 骨架 + stage 实例 `kv.stage` / `v_phase` / `m_phase` / `window`；导读 [`fmha-dsl-tour.md`](fmha-dsl-tour.md)） |
| `gemm_mxfp8_bs_v4` | MXFP8 blockscale GEMM（`crates/amdgpu-kernels/src/gemm/`）：厂商结构（寄存器暂存、AGPR 原地重载、slotfill）；stage 实例 `gemm.ab_stage` / `gemm.tile` |
| `mlp_fp8_ws_{fused,g1,g2}` | 融合 MLP 与两段对照（`crates/amdgpu-kernels/src/mlp/`；stage 实例 `gemm1_act` / `gemm2`） |
| `mlp_fp8_ws_g1q_{relu,silu}` | GEMM + 通用 epilogue 链（alpha、bias、激活、行 amax、逐行 fp8 + scale）；见 §4.8 |
| `rmsnorm_quant_fp8_n<N>`（N = 512…8192） | RMSNorm + 逐 token fp8 量化，4 个 stage（load → rms → quant → store）；见 §4.8 |
| `probe_*` | 硬件/ISA 探针（`crates/amdgpu-kernels/src/probes/`），各自配 `tools/*_probe.sh` |
| `gv_attack_*`、`gv_audit_*`、`probe_lds_oob` | 验证器必须拒绝 / 必须列入信任清单的用例（`crates/amdgpu-kernels/src/verifier_cases/`） |

旧手写集、上一版 tile 库集（`dsl_*`）、`v3x`/`v3s` 实验变体、GEMM v3 与旧 MLP 变体已在 2026-11 整理中删除（结论保留在 [`archive/`](archive/README.md)）。

### 4.3 精度与确定性

- **门禁准则（用户决定 2026-11-04，"2+3"）**：ACC64 以**厂商对齐**判定——每个形状、每条路径同轮同输入下，我们的 mean_err ≤ 1.10× 厂商且 bad ≤ 厂商；
  旧的严格准则（bad = 0）仍以 INFO 打印，`ACC_STRICT=1` 可恢复（`tools/port_gate.sh`）。
- **参考算法（用户决定 2026-11-05，"accept vendor accuracy"）**：`_vs128` 的长序列副本（≥ 9 tile）使用厂商的参考配方（目标 s91 = 0.24799132、
  懒回滚阈值 s92 = 0.26135647、转换比例 0.25、无 LUT 校正、首 tile 播种；`OnlineSoftmax::set_vendor_ref`）；短序列副本保留精确配方 + LUT 校正；
  `_t128` 与 `_kvsplit`（出厂 split 走短副本）不受影响。递推与朴素求和的一致性：`crates/amdgpu-compiler/tests/functional_proofs.rs`。
  采用时的同轮测量（历史记录）：精度与厂商同分布（例如 B4H8S2048 2 / 2.848e-3 vs 厂商 2 / 2.859e-3），速度 B1H8S8192 +5.5%、其余 +0.7..0.9%。
- 曾测过的"快速参考"变体（`scaleconv`/`hi512`）没有一个既更快又满足厂商对齐，已删除（数据见 [`archive/`](archive/README.md)）。
- 确定性 5/5 × 3 次；split 形状重复启动（`port_gate` 的 STRESS 段）。

### 4.4 优化里程

292 → 367（每 CU 差 1.41×）→ 505（t128 分块）→ 605（KV-split）→ 622（kddma）→ 674–705 TFLOP/s（第 75 轮，旧手写集）
→ tile 库集 `dsl_*`（≤3% 内）→ v3 布局层（`_t128` SADDR DMA、`edgepad`、`salubr`）→ v6 厂商参考配方、K-scale 字节选择 → v7.0 标量死值消除。
逐轮记录：[`archive/status.md`](archive/status.md)。

### 4.5 门禁方法

`tools/port_gate.sh` 没有参考 kernel 集：性能对同轮厂商测量，并与记录的出厂比值比较（`PERF_TOL=0.97`）。
代码布局敏感性：同一 kernel 前置 0..7 条 `s_nop` 使 per-CU `_vs128` 有约 ±2% 的变化，单一布局对比不可信；门禁同时报告 8 布局中位数。

### 4.6 v7.0 FMHA 门禁（`tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3`，GPU 6）

| 项 | TF/s | ×厂商（同轮） |
|---|---|---|
| B1H8S2048 出厂路径（split + combine） | **685.9** | 1.27× |
| per-CU 64 blocks（`_vs128`） | **493.1** | 0.917× |
| B4H8S2048 / B1H8S8192（`_vs128`） | **1543.9** / **2206.4** | 0.918× / 0.921× |
| B2H8S2048 `_t128` | **1067.4** | 1.05× |

BUILD / VERIFY（0 错误）/ ACC16 / ACC64（厂商对齐，四条路径）/ DET / STRESS 全部 PASS：**OVERALL PASS**。
（v7.0 门禁时的 asm md5 为 `763142f9` / `91b26e5d` / `e1fe92a4` / `24a95a51`；当前出厂 `_vs128` / `_t128` / `_kvsplit` / `combine8h` 为 `8938adfa` / `d7e02b0f` / `41f151fc` / `24a95a51`，stage / weaver / 骨架重写前后逐字节相同。）`_t128` 默认打开 `edgepad` 与 `salubr`（同轮 A/B 布局中位数 +3.3%）及 SADDR 形式的
LDS-DMA（+1.8..2.1%）；它们在 `_vs128`/`_kvsplit` 上为负，不使用（A/B 记录见 [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) §9.4）。

### 4.7 MXFP8 blockscale GEMM

| 项 | 数值（v7.0） |
|---|---|
| kernel / 形状 | `gemm_mxfp8_bs_v4`，M=N=4096、K=8192，256×256×128 CTA tile，4 wave，`v_mfma_scale_f32_16x16x128_f8f6f4`（swap_ab），A/B 片段与部分 C 在 AGPR |
| 性能 vs 厂商（同轮） | **0.988–0.994×**（交替 7 进程中位数）；分析见 [`mxfp8-gemm-4wave-analysis.md`](mxfp8-gemm-4wave-analysis.md)、[`gemm-issue-comparison.md`](gemm-issue-comparison.md) |
| 门禁 | `tools/gemm_gate.sh`：BUILD / VERIFY（strict 0 错误，3 trusted assumes / 19 contracts / 0 unproven）/ ACC（4096²×8192、8192×7168×7168、1280×6144×7168 GEMM_OK）/ DET 3/3 / PERF —— **PASS** |
| 编译器配套 | AGPR 寄存器类与 acc_cd MFMA（[`agpr.md`](agpr.md)）、SALU 地址链 `scalarize2`、kernarg 批量加载、M0 复用、`slotfill`、DPP |

### 4.8 stage 库新 kernel（P2，GPU 5，同机 hipMemcpy D2D = 5.03 TB/s 作带宽基线；节点上没有 aiter / torch）

| kernel | 正确性（对 CPU fp32 参考） | 性能 |
|---|---|---|
| `rmsnorm_quant_fp8_n4096`，M = 16384 | bad = 0（反量化误差 ≤ 半个 e4m3 ulp；scale 相对误差 ≤ 3.5e-7），N 1024 / 4096 / 8192 × M 7 / 4096 / 16384 | 5.14 TB/s（1.02× D2D 拷贝）；N8192 4.69 TB/s，N1024 4.23 TB/s |
| `mlp_fp8_ws_g1q_relu` / `_silu`，M = 65536 | bad = 0 | 412 / 339 TF/s，为无 epilogue 链的 `mlp_fp8_ws_g1`（514 TF/s）的 0.80× / 0.66×：链在 MMA 之后串行发射，尚未交织 |

主机测试：`tests/gpu/rmsnorm_quant_host.cpp`、`tests/gpu/gemm_epi_host.cpp`；严格验证 0 错误。

---

## 5. 已量化关闭的方向（都带反例与归因，详见 [`archive/`](archive/README.md)）

| 方向 | 结论 |
|---|---|
| per-CU 微优化（VALU 削减） | VALU −8.7% 后 wave +1.2%；去掉的槽变成依赖停顿，非指令受限 |
| 循环不变量提升（`licm`）、M0 预写（`m0hoist`） | 指令数下降但 −4..−8% / 中性：保留常量占住调度所需的 VGPR 余量；已删除 |
| bias MFMA 移除（biasacc/biasc/biasv/biasonce/vpre/kpre2/qkfirst/p4-unscaled） | −0.8% … −15%；那 128 XDL 周期是 barrier 后 LDS 读延迟的遮盖 |
| pinned M-window / K-ahead ring / V 预读 | −5…−7% / −10.5% / −3.5..−5% |
| MUBUF 转换 | 正确性中立；上界 +0.1…1.0% |
| 4 waves/SIMD 占用率 | 不适合 VGPR/LDS 预算；厂商同样 1 block / 8 wave per CU |

---

## 6. 修复过的真实正确性 bug

| bug | 影响 | 修复 |
|---|---|---|
| 物理 WAW（MFMA 在途结果寄存器被 RA 复用） | 非确定输出 | `pad_physical_war` 跟踪在途 MFMA 目标区间；`crates/amdgpu-compiler/tests/phys_waw.rs` |
| 两阶段 RA reaching-writer 误报 | 持久循环变量编译失败 | path-aware 检查；`crates/amdgpu-compiler/tests/ra_reaching.rs` |
| VMEM 宽存储数据 WAR | `global_store_dwordx4` 后数据寄存器被过早覆盖 → split 路径偶发损坏 | `pad_physical_war` 记录宽存储数据区间；STRESS 段回归 |
| LDS-DMA 操作数等待放置（AGPR 工作中发现） | 静默误编译 | 见 [`agpr.md`](agpr.md) "Wait placement" 一节 |

---

## 7. 验证与工具

- **测试**：`cargo test` **87 个目标 / 366 项全绿**。编译器：`hazard_*`、`phys_waw`、`ra_reaching`、`pin*`、`gv_bits`、`sym_branch`、`mfma16x16x128`、
  `lds_b64`、`scalarize2`、`dpp`、`var_slice`、`rust_safety` 等；kernel 相关（`crates/amdgpu-kernels/tests/`）：`tail_invariance`、`mir_invariants`、
  `register_accounting`、`asm_assembles`、`agpr`、`gpu_*`（语义/协议/攻击）、`gemm_mxfp8`、`tile_l*`、`region_ir`、`sched_kernels`；
  布局层 `crates/amdgpu-layout/tests/`、tile 层 `crates/amdgpu-tile/tests/`；safe-asm `tests/ui` 29 个用例。
- **门禁**：`tools/port_gate.sh`（FMHA：BUILD / VERIFY / ACC16 / ACC64 / DET / STRESS / PERF）、`tools/gemm_gate.sh`（GEMM）。
- **md5 守护**：`tools/md5_kernels.sh`（全部 `amdc list`）与 `tools/md5_compare.py`（对基线逐 kernel 比较，列出差异与已删除项）。
- **工具**（`tools/`）：厂商对照（`vendor_disasm.sh`、`vendor_analyze.py`）、汇编对照（`asm_diff.py`、`asm_regmap.py`、`loop_stats.py`、`nop_attrib.py`）、
  GEMM（`gemm_ref.sh`、`prof_gemm.sh`、`att_*.py`）、A/B（`ab_fmha.sh`、`ab_gemm.sh`）、探针（`*_probe.sh`）、LLVM（`llvm_build_llc.sh`、
  `llvm_conformance.sh`、`llvm_feature_audit.py`）、文档链接检查 `check_doc_links.py`。
- **测量协议**：`.s`/`.co` 溯源、≥7 轮交错 A/B 或 8 布局中位数、同轮厂商、split 形状反复启动；GPU 隔离（每人固定 `HIP_VISIBLE_DEVICES`）。

---

## 8. 复现命令（容器内）

```bash
# 构建与测试
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                 # 期望 87 targets / 366 tests 全绿

# 发射 kernel 并汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --backend=own --out vs128.s   # md5 763142f9…
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 严格证书
bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify --shipped --strict   # 24 / 100 / 0

# 主机侧生成
bash tools/container/cargo.sh run -q --bin amdc -- gen-host --shipped --out tests/gpu/generated

# 正确性 + 性能（同轮跑我们的、厂商的、CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh

# 门禁
HIP_VISIBLE_DEVICES=6 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3
HIP_VISIBLE_DEVICES=6 bash tools/gemm_gate.sh
```

---

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| `current-status.md` | **本文件：权威当前快照** |
| [`design.md`](design.md) | Rust DSL 与自研编译器设计规格 |
| [`layout-tour.md`](layout-tour.md) | 布局层入门教程 |
| [`fmha-dsl-tour.md`](fmha-dsl-tour.md) | FMHA 逐段教程（DSL 特性 + Rust safe 保证） |
| [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) | 路线 B 设计与 §9 as-built |
| [`dsl-tile-abstractions.md`](dsl-tile-abstractions.md) | tile 层设计 |
| [`dsl-expression-layer.md`](dsl-expression-layer.md) | 运算符表达式层 |
| [`region-ir.md`](region-ir.md) / [`op-attributes.md`](op-attributes.md) / [`sym.md`](sym.md) | 区域树 IR / MOp 属性表 / 共享符号分析 |
| [`fmha-perf-patterns.md`](fmha-perf-patterns.md) / [`fmha-vs-vendor-asm-diff.md`](fmha-vs-vendor-asm-diff.md) | FMHA 性能承重模式 / 与厂商汇编的逐项对照 |
| [`gpu-semantics-verifier.md`](gpu-semantics-verifier.md) / [`gpu-semantic-protocol.md`](gpu-semantic-protocol.md) | GPU 语义验证器 / 四轴协议模型 |
| [`last-arriver-memory-model.md`](last-arriver-memory-model.md) | last-arriver 原语与内存序契约 |
| [`agpr.md`](agpr.md) | AGPR 支持 |
| [`host-gen.md`](host-gen.md) / [`hostgen-json-schema.md`](hostgen-json-schema.md) | 主机侧生成 |
| [`rust-safety.md`](rust-safety.md) / [`rust-safety-by-example.md`](rust-safety-by-example.md) | Rust 安全 |
| [`safe-asm-design.md`](safe-asm-design.md) | safe-asm 设计 |
| [`scheduling.md`](scheduling.md) / [`slotfill.md`](slotfill.md) / [`llvm-tail-integration.md`](llvm-tail-integration.md) / [`llvm-tail-conformance.md`](llvm-tail-conformance.md) / [`llvm-feature-audit.md`](llvm-feature-audit.md) | 调度器、LLVM tail、CodeGen 覆盖审计 |
| [`vendor-spec.md`](vendor-spec.md) / [`fmha-reference.md`](fmha-reference.md) | 厂商 FMHA 事实与认证表、参考 kernel |
| [`mxfp8-gemm-4wave-analysis.md`](mxfp8-gemm-4wave-analysis.md) / [`gemm-issue-comparison.md`](gemm-issue-comparison.md) | 厂商 4-wave GEMM 分析与逐指令对比 |
| [`fusion-layout-handoff.md`](fusion-layout-handoff.md) | 融合 MLP 的布局交接 |
| [`archive/README.md`](archive/README.md) | 历史归档索引（逐轮记录、各代设计研究、终局报告、审计记录） |
