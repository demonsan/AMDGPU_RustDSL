# 当前状态总览（authoritative snapshot）

本文件是仓库的**唯一权威当前状态快照**，最后更新于路线 B 完成后（`Main` = `7f937f8` + integrate；出厂集 = v3 布局层 kernel）。
所有数字都来自同轮硬件测量（MI355X / gfx950，`mi355-gpu-53`），复现命令见 §8。
历史逐轮记录在 `docs/status.md`（25 轮），供应商事实在 `docs/vendor-spec.md`，设计规格在 `docs/design.md`。

---

## 1. 一句话状态

Rust 类型化 kernel DSL + 自研编译器（**寄存器分配与指令调度始终是我们自己的**，LLVM 只做 post-RA tail），
在 gfx950 上运行厂商 ABI 兼容的 MXFP8 FMHA：

- **基准形状 B1H8S2048：674–705 TFLOP/s，同轮厂商 536–539 → 1.25–1.31x 更快**
- 单 kernel 每 CU（64 blocks）：**7.07–7.10 TF/CU**，厂商 8.41 → **1.19x 差距**（本设计下限）
- 铺满网格（B4H8S2048 / B1H8S8192）：**0.846x / 0.841x 厂商**（结构线已量化关闭）
- 精度 **mean 8.3e-4**（厂商 1.5–2.9e-3，好 2–3 倍），15/15 形状、64 行 bad=0、确定性全过、1200/1200 次启动单一 md5
- `cargo test` **282 项 / 56 个测试目标全绿**（Main@7f937f8）
- **出厂集 = v3 布局层 kernel（路线 B，tag `v4.1-layout-p3` 起）**：默认路径 `fmha_fwd_hd128_mxfp8_auto` 的 4 个 kernel 是
  `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}`（`crates/amdgpu-tile/src/fmha.rs`）。K/V/K-scale/Q 的每个地址、
  K/V/K-scale DMA、bf16 epilogue 与 KV-split 转置暂存都由 `amdgpu-layout` 的布局推导，`amdgpu-isa` 的 atom 描述指令，`amdgpu-tile` 的
  `TiledCopy`/`TiledMma`/`partition_*` 生成 IR；主机合同由张量布局导出。全套门禁对旧手写集与对 tile 库集均通过（§4.6），证书与两者相同。
  上一版（tile 库集 `dsl_*`）保留为 `fmha_fwd_hd128_mxfp8_dsl_auto`，旧手写集保留为 `fmha_fwd_hd128_mxfp8_legacy_auto`，两者 md5 不变。
- **主机侧生成（P5）**：`amdc gen-host` 从 kernel 合同生成 C++ 启动/ABI 头与 JSON（[`host-gen.md`](host-gen.md)）；出厂 `ref_host` 使用生成的命名字段。
- **MXFP8 blockscale GEMM（P4）**：`gemm_mxfp8_bs_v3` 对照 aiter PR #6696 4-wave pin-AGPR —— **_（待 kernel-opt 填写最终数字）_**；见 §4.7。
- 上一版说明（2026-11-02）：默认路径 `fmha_fwd_hd128_mxfp8_auto` 的 4 个 kernel 曾为
  `fmha_fwd_hd128_mxfp8_dsl_{vs128,t128,kvsplit,kvsplit_combine8h}`（`crates/amdgpu-dsl/src/fmha_tile.rs`：kernel 函数 21/19/23/6 行 + 注意力
  `AttnBody` 111 行；FMHA ABI/块设置/K-V 环/epilogue/combine 是库组件 `tile/fmha.rs`）。与旧手写集同轮对照：精度/确定性/压力全过，
  `_vs128` B1H8S2048 bf16 输出 **逐位相同**（`faadf00c…`），split 路径 300/300 单一 md5 与旧集相同；性能见 §4.5。旧手写集保留为
  `fmha_fwd_hd128_mxfp8_legacy_auto`（`amdc gpu-verify --legacy`），asm md5 不变。
---

## 2. 编译器流水线

```
DSL(Rust) ──► typed IR (Op 83 种 / Stmt 40 种) ──► IR 验证器
            ──► lowering ──► 我们的 MIR
            ──► 我们的跨基本块调度（list / hoist / swp / taildup）
            ──► 我们的线性扫描 RA（两阶段 + LDS spill + S4 描述符类）
            ──► post-RA MIR 导出（物理寄存器、已排序）
                 ├─ llvm-tail（默认）：llc -start-before=gcn-create-vopd … 只跑 tail pass
                 └─ own：我们自己的 hazard / wait / descriptor 实现
            ──► .s ──► ROCm clang 只做汇编 ──► .co ──► HIP 启动
```

| 环节 | 归属 | 说明 |
|---|---|---|
| DSL / IR / 验证 | 自研 | 类型化、无 unsafe、单赋值循环变量（`var_lint`） |
| Lowering / MIR | 自研 | 内部 SSA-ish MIR + 效应/副作用模型 |
| **寄存器分配** | **自研** | 线性扫描、两阶段（含 path-aware reaching-writer 检查）、LDS spill、S4 四元组描述符寄存器类 |
| **指令调度** | **自研** | 块内 list scheduler、superblock/hoist、软件流水 `swp`、`taildup`、依赖图 / 效果模型 / 等待模型（`sched/` 15 个模块约 2,700 行） |
| hazard / wait / descriptor | llvm-tail 或 own | 两个后端指令流逐条一致（`tail_invariance` 守护） |

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`）。ROCm 10 不自带 `llc`，用
`tools/llvm_build_llc.sh` 从同一 commit 链接 `libLLVM.so` 得到；ROCm 7.0.2（`llvm20`）作为备选方言。
找不到 pin 的工具目录时 **llvm-tail 直接报错**，不静默降级。

**指令覆盖**：101 个 MOp 在 `llvm20.tbl` **和** `llvm24.tbl` 都有行；own emit 全覆盖。
DMA、LDS 直达、`buffer_load … lds` 等为 own-only（使用它们的 kernel 自动选 own 后端）。

### 2.1 GPU 语义验证（`gpu_verify`，`AMDC_GPU_VERIFY=off|warn|strict`）

编译器现在**感知 GPU 指令语义**：在 DSL IR 层做符号区间分析（多项式 + 循环不动点 + 合同），在最终 MIR 层检查寄存器类别与
**路径敏感**的等待/生命周期。文档：[`gpu-semantics-verifier.md`](gpu-semantics-verifier.md)（模型与边界）、
[`gpu-semantics-audit.md`](gpu-semantics-audit.md)（独立对抗审计）。命令行：

```bash
amdc gpu-verify <kernel>|--all|--shipped [-v] [--strict]   # 报告 proven/发现/trusted 清单/证书
cargo run --bin amdc -- gpu-verify --shipped --strict      # 出厂 kernel 的严格证书
```

**已证明（出厂 kernel，`--strict` 0 错误）**：

| 类别 | 证明内容 | 出厂覆盖 |
|---|---|---|
| **LDS** | 每次 DS 访存/DMA（含 64 lane）在声明段内；环形/双缓冲槽位由不动点证明；无下溢/回绕 | 600（`_vs128`/`_vs5`）、760（`_kvsplit`）、336/226/424/318 |
| **LDS tile 生命周期** | tile 分区/别名、produce/consume/version（发布于 barrier）、跨角色 stale/race/pending；周期门（period ≤ horizon） | `_vs128`/`_vs5` **58,880** 次读、`_kvsplit` **240,640** 次读（tile-cover 600/760），0 错误 0 警告 |
| **全局/buffer** | 访问落在声明 extent 内（`arg_tensor` + 形状关系 `ka_is/grid_is/ka_one_of`）；描述符不得高估 `num_records` | 106/92/78/3/88/97/318/424 次访问；`tests/contract_host_truth.rs` 校验合同与 host 公式一致 |
| **VGPR/SGPR** | v0、s0–s7 保留；元组对齐（VGPR 偶数、SGPR 对偶数/四元组×4）、≤256/≤102；MFMA A/B 与 D 不重叠 | 全部 88 个 kernel 0 错误（默认强制） |
| **寄存器生命周期** | 路径敏感 use-before-wait（任一路径、含循环携带在途值） | 默认强制；出厂 0 错误 |
| **barrier 协议** | 波分歧 if/else 两侧动态 barrier 数相等 | `_vs128` 18、`_t128` 8、`_kvsplit` 18、`_t` 11 |

**信任预算（证书里逐条列出）**：出厂 8 个 kernel `--strict`：**0 错误 0 警告**；合计 **44 条 trusted assumes**（10 条理由，全部 `assume_trusted` 带理由）、
**216 条 trusted host contracts**、**0 unproven**。每次运行打印 `certificate: N trusted assumes, M trusted contracts, K unproven`。

**未证明/范围外（诚实清单）**：跨工作组数据竞争与别名、i32 回绕（A3 假设）、从内存加载的指针/索引、safe-asm 块内部访存、
tile 内字节级重叠、`_kvsplit` 的 160 处 epilogue scratch（界已证、生命周期未证，Warn）、角色数 2 / 迭代数相等假设。
独立审计发现并修复了 2 个 soundness 漏洞（`assume` 归纳违反应为 Error；仿射长周期 marker 逃逸 horizon），
并用硬件实测校正了 MFMA 对齐规则（VGPR 偶数，不是 16 对齐）。
**v4 通用协议模型**（[`gpu-semantic-protocol.md`](gpu-semantic-protocol.md)）：把上述检查统一成 **四轴分离** 的协议 —
**到达**（`s_barrier` tick，只表示到达）、**完成**（`wait_vm(0)/wait_lds(0)/fence_release`，每 wave 队列计数）、
**可见**（LDS 用 barrier 传递；全局用 release→barrier→acquire + `sc1`）、**代际**（仿射/周期 version-slot 代数 + 周期门）。
参与者是 **多生产者**（同角色多 wave 协作填充 / 多分支 / 多 block）与 **多消费者**（每个角色、每个 wave 独立消费），
编译器静态 discharge 四条双向义务：**O1 完成先于发布**、**O2 发布先于读**、**O3 读完先于覆写**、**O4 代际一致**；
判不出的进 **可计数信任账本**（`assume_trusted` / `tile_trust_periodic` / `region_trust`，证书行逐项打印）。
全局内存也有实例（`buf_region/buf_produce/buf_consume`，v1 限工作组内）+ 显式 **last-arriver 融合 combine 模式**
（release → ticket atomic → acquire → 读所有 split）：`probe_last_arriver` 的 full/lite/plain_fenced 三种模式 `--strict` 0 错误，
`nofence/plain` 负例被拒。LDS 实例的 verdict 保留旧 `GV-TILE-*` 为主名、新增 `GV-PROTO-{UNPUBLISHED,NOT-VISIBLE,STALE-GEN,
RACE,ARRIVAL-MISMATCH,NO-COMPLETION,UNROLL-LIMIT,TRUST}` 别名，全部既有测试不变、判定未放宽。

---

## 3. DSL 设计

| 维度 | 内容 |
|---|---|
| 入口 | `FnBuilder`（140 个公开方法）→ `Function`（kernarg ABI） |
| 值模型 | `ValueId` / `VarId`（循环变量）/ `Opnd` / `DType` / `Frag8` / `Frag16` |
| 计算 | 普通与 `scaled` 32x32x64 f8f6f4 MFMA（含 **C 操作数形式**）、**unscaled** MFMA（zero-C / 累加 / 独立 C）、`v_smfmac` f8/f16、f16/bf16 MFMA |
| 向量/转换 | VALU int/fp、`pk_add/pk_mul/pk_sub`、`perm`、`permlane32_swap`、`pknorm`、`scalef32`、`cvt_pk_fp8`、`exp/rcp/log`（trans token）、`max3`、`readfirstlane` |
| 内存-FLAT | `global_load/store`、`global_load_lds`（global→LDS 直达） |
| 内存-MUBUF | `buffer_rsrc` / `buffer_desc` / **`buffer_desc_dyn`（运行时 num_records）**、`buffer_load/store x1/x2/x4`、`buffer_load … lds`（硬件验证越界零填充）、m0 / imm12 / soffset |
| 内存-LDS | ring / 双缓冲 API、`ds_read_b64_tr_b8` 转置读 |
| 控制流 | `begin_loop`/`end_loop`、`begin_if`（wave 级）、peel last tile、双 code copy（`svcorr`）、`StoreF32x2/x4`、`Stmt::VarAsm(N)` 桥接手写 asm |
| 实验面 | **`FMHA_OPT` 108 个 token**：每个优化与负结果都可 A/B、可复现 |
| ABI | 与厂商**逐字节兼容的 656 字节 kernarg + 网格/数据布局**；`_auto` 按形状 guard 调度 |
| 布局层（路线 B） | `amdgpu-layout`：`Ix`（静态/动态符号）、`IntTuple`、`Layout` 与 FlyDSL 对齐的代数、GF(2) `Swizzle`、TV 布局、`IssuePlan`/`layout_to_offsets`、构建期证明；`amdgpu-isa`：copy / MMA atom 表；`amdgpu-tile`：`TiledCopy`/`TiledMma`/`partition_{lds,dma,buffer_dma,global}`、布局导出的合同。入门：[`layout-tour.md`](layout-tour.md) |

**crate 地图**：

| crate | 内容 |
|---|---|
| `amdgpu-layout` | 布局代数（零依赖） |
| `amdgpu-isa` | gfx950 copy / MMA atom（依赖 layout） |
| `amdgpu-tile` | TiledCopy / TiledMma / partition、v3 FMHA kernel（依赖 layout、isa、dsl） |
| `amdgpu-kernels` | MXFP8 GEMM（P4，随 integrate 合入） |
| `amdgpu-dsl` | IR、`FnBuilder`、上一版 tile 库（`src/tile/`）、dsl 与旧手写 kernel |
| `amdgpu-compiler` | lowering、调度、RA、hazard/wait、emit、`gpu_verify`、host-gen（`hostgen`） |
| `amdgpu-safeasm` | safe-asm 窗口 |
| `amdc` | CLI（emit / gpu-verify / gen-host / …） |

### safe-asm 子系统（`crates/amdgpu-safeasm`，`#![forbid(unsafe_code)]`）

手写热点窗口用的安全 Rust asm DSL：typed 寄存器、所有权 token（`Pending` / `Mfma` / `Trans` / `Pw<L,H>` 半写状态）、
等待态验证器、常量总线 const-assert、`Seq` 部分 `lgkmcnt`、**drain-closed 循环体契约**、**op 白名单**、
**`Seq::pin()` + `pin_check`**（pinned 段 padding 弱于编译器规则即编译失败）。经 `Op::Asm`/`Stmt::VarAsm` 接入编译器。
覆盖：29 生成 + 27 手写 typed 模板，51/102 backend 行为 raw-only；`tests/ui` 共 29 个用例（28 个编译失败 + 1 个通过对照）。

---

## 4. Kernel 状态

### 4.1 默认出厂路径：`fmha_fwd_hd128_mxfp8_auto`

| 形状类 | 选择 | 性能（同轮厂商） |
|---|---|---|
| **B1H8S2048**（基准） | KV-split s=4 + `combine8h` | **674–705 TFLOP/s vs 536–539 = 1.25–1.31x** |
| 铺满网格 B4H8S2048 / B1H8S8192 | `_vs128` | 1443.6/1454.8 vs 1706.8（**0.846x**）；1998.6/2002.7 vs 2376.5（**0.841x**） |
| 中等形状 | `_t128` / split（7 个 gate 形状选 split） | B2H8S2048 ~1057（1.00x）、B1H8S4096 ~1301（1.07x）、B3H3S768 ~235（1.11x）、B1H16S256 ~131（1.16x）、B1H8S1024 ~227（1.01x） |
| 单 kernel 每 CU（64 blocks） | `_vs128` | **453–455 TFLOP/s = 7.07–7.10 TF/CU**（厂商 8.41 → 1.19x） |

### 4.2 Kernel 清单

| Kernel | 状态 |
|---|---|
| `fmha_fwd_hd128_mxfp8_auto` | **默认**（guard 选择下面三者）：`v3_vs128` / `v3_t128` / `v3_kvsplit` + `v3_kvsplit_combine8h`（布局层，`crates/amdgpu-tile/src/fmha.rs`） |
| `fmha_fwd_hd128_mxfp8_dsl_auto` | 上一版 tile 库集 `dsl_*`（`crates/amdgpu-dsl/src/fmha_tile.rs`），A/B 对照用 |
| `fmha_fwd_hd128_mxfp8_v3s_*` | v3 的别名（同代码），配合 `FMHA_OPT_SCOPE=v3s` 对实验 token 做同轮 A/B |
| `gemm_mxfp8_bs_v3` / `_v4` | MXFP8 blockscale GEMM（P4，`crates/amdgpu-kernels`，随 integrate 合入）；v4 = 厂商结构（RegStaged、AGPR 原地重载） |
| `fmha_fwd_hd128_mxfp8_legacy_auto` | 旧手写集（下面四行），A/B 对照用 |
| `_vs128` | 单 kernel 快路径（铺满网格 / 64-block 基准），256-row block（旧手写） |
| `_t128` | 128-row block，中等/混合形状的 fallback（旧手写） |
| `_kvsplit` + `combine{4,4h,8h,16h}` | KV-split 两段启动（split 形状）；`combine8h` 为默认 combine（旧手写） |
| `_t` | 精确 recipe kernel（llvm-tail，311 TFLOP/s；冻结，仅作正确性对照） |
| `_g2/_g3/_g3o/_pp/_sv/_vs5/_vs6/_vs9/_vsm` | 历史/实验变体，opt-in 或仅按名（负结果记录） |
| `biasacc` / `p4` / `mubufd` / `cut1` / `saddrlds` / `pmdead` / `epdirect` / `eplds,epw` | opt-in 实验（多数中性或负；`epw` 是 VMEM 宽存储 WAR 的灵敏度探测器） |

### 4.5 tile 库出厂集 vs 旧手写集（同轮，`tools/port_gate.sh fmha_fwd_hd128_mxfp8_dsl`，GPU 6）

| 项 | tile 库（dsl） | 旧手写 | 比值 |
|---|---|---|---|
| B1H8S2048 出厂路径（split+combine） | 684.5 | 672.6 | 1.018 |
| per-CU 64 blocks，8 种代码布局中位数 | 447.0 | 451.2 | 0.991 |
| B4H8S2048 / B1H8S8192（`_vs128`） | 1409.6 / 1972.3 | 1432.0 / 1993.5 | 0.984 / 0.989 |
| B2H8S2048 `_t128`，8 布局中位数（24 次） | 1004.3 | 1027.3 | 0.978 |

**代码布局敏感性**：同一 kernel 前置 0..14 条 `s_nop` 使 per-CU `_vs128` 在 444–456 TF/s 间变化（旧、新集相同），单一布局对比不可信；门禁用布局中位数。
`gpu-verify --shipped --strict`（dsl 四个）：0 错误 0 警告，24 trusted / 100 contracts / 0 unproven，proven 计数与旧集逐项相同。

### 4.6 v3 布局层出厂集（同轮，`tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3`，GPU 6）

| 项 | v3 | 旧手写（同轮） | dsl（另一轮同轮，`SHIP_REF=fmha_fwd_hd128_mxfp8_dsl`） |
|---|---|---|---|
| B1H8S2048 出厂路径（split+combine） | 681.1 | 677.1 | 679.0 vs 687.2 |
| per-CU 64 blocks（单一布局） | 445.1 | 453.1 | 446.4 vs 445.1 |
| per-CU 8 布局中位数 | 448.1 | 451.8 | 449.0 vs 447.3 |
| B4H8S2048 / B1H8S8192 | 1418.7 / 1974.8 | 1446.7 / 1986.7 | 1408.0 vs 1413.8 / 1970.9 vs 1967.8 |
| B2H8S2048 `_t128` 8 布局中位数 | 1035.7（SADDR DMA） | 1040.7 | 1002.4 vs 997.9（SADDR 之前） |

VERIFY 0 错误（trusted 6/29、4/29、12/36、2/6，与 dsl/legacy 相同）、ACC16 15/15、ACC64 四路径 bad=0、DET 5/5、STRESS 300/300 + 100/100：**OVERALL PASS**。
v3 asm md5：`_vs128` `68d2832b`、`_t128` `f5a43b09`、`_kvsplit` `e4fc18b1`、`_kvsplit_combine8h` `24a95a51`。
`_t128` 默认用 SADDR 形式的 LDS-DMA（两轮 A/B 布局中位数 +1.8% / +2.1%），`_vs128`/`_kvsplit` 上为 −0.2..−1.4%，保持 opt-in；数字见
[`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) §9.4。

### 4.7 MXFP8 blockscale GEMM（P4）

| 项 | 数值 |
|---|---|
| kernel / 形状 | `gemm_mxfp8_bs_v3`，M=N=4096、K=8192，256×256×128 tile，4 wave，`v_mfma_scale_f32_16x16x128_f8f6f4`（swap_ab） |
| 性能 vs 厂商（同轮） | **_（待 kernel-opt 填写）_** |
| 门禁 | `tools/gemm_gate.sh`：BUILD / VERIFY（strict）/ ACC（3 形状 GEMM_OK）/ DET / PERF —— **_（待填写）_** |
| 编译器配套 | AGPR 寄存器类与 acc_cd MFMA（sched）、SALU 地址链 `scalarize2`（2771 → 2867 TF/s）、`ds_read_b64`/`ds_write_b64`、验证器 possible-bits 与 box 规则（去掉仅为验证器加的掩码） |

### 4.3 精度与确定性

- 参考算法：每 128-key 单元**精确参考重基准**（阈值 120/512）、长序列**无 LUT 校正**、短序列第二份代码副本 + `svcorr`
- mean err **8.3e-4**（厂商 1.5–2.9e-3）；15/15 形状、16/64 行 parity 15/15、our bad = 0
- 确定性 5/5 × N 次；**uneven-tile split 形状 1200/1200 次启动单一 md5**

### 4.4 优化历里程

292 → 367（每 CU 差 1.41x）→ 505（t128 分块）→ 605（KV-split）→ 622（kddma）→ **674–705 TFLOP/s**
（最后一轮：Q 16B 载入 +6–7%、8B epilogue 存储 +8–9%、combine8h +1%、guard 重拟合 +13% 于 B1H16S256）

---

## 5. 已量化关闭的方向（都带反例与归因）

| 方向 | 结论 |
|---|---|
| per-CU 微优化（VALU 削减） | VALU −8.7% 后 wave **+1.2%**；去掉的槽变成依赖停顿（WAIT_INST_ANY +14.8%），非指令受限 |
| bias MFMA 移除（biasacc/biasc/biasv/biasonce/vpre/kpre2/qkfirst/p4-unscaled） | **−0.8% … −15%**；那 128 XDL 周期是 barrier 后 LDS 读延迟的遮盖 |
| pinned M-window / K-ahead ring | −5…−7% / −10.5%（in-order 发射 + padding + VGPR 压力 / 相位失衡） |
| **gen10 三项相位设计**（持久 C、V 预读、相位重平衡） | **硬门槛失败**：最好有效变体 wave 1453 / XDL 1130；XDL 地板 1130（分离-C 19 + 4-vs-2 smfmac 23） |
| MUBUF 转换 | 正确性中立、opt-in；纯 buffer 上界 **+0.1…1.0%**（地址 VALU 早已被遮盖） |
| 铺满网格对齐 | **0.846x / 0.841x**；占用率与厂商相同（1 block/CU、2 waves/SIMD），差距纯属每 CU 关键路径 |

**未来若要铺满对齐**，round 75 文档给出三项必要条件：匹配厂商 2-smfmac 行和结构、消除分离-C 的 19 周期、
重新设计半 tile 相位偏移解决两组 wave 的 barrier 汇合。

---

## 6. 本会话修复的真实正确性 bug

| bug | 影响 | 修复 |
|---|---|---|
| 物理 WAW（MFMA 在途结果寄存器被 RA 复用） | opt-in 组合下非确定输出 | `pad_physical_war` 跟踪在途 MFMA 目标区间，跨基本块传播；MIR 回归测试 |
| 两阶段 RA reaching-writer 误报 | 任何持久循环变量都编译失败（88 violations） | 改为 path-aware（history）检查，仍拒绝真实 clobber；`tests/ra_reaching.rs` |
| **VMEM 宽存储数据 WAR** | `global_store_dwordx4` 发后 2 等待态内被 VALU 覆盖 → **出厂 split 路径偶发数据损坏（~5%，时序扰动下 97%）** | `pad_physical_war` 记录宽存储数据区间；仅 `_kvsplit` +4 nop；`ohat_repro.sh` 300 次压力回归 |

---

## 7. 验证与工具

- **测试**：`cargo test` **40 个目标 / 238 项全绿**；22 个测试文件（`tail_invariance`、`mir_invariants`、`rust_safety`、
  `register_accounting`、`asm_assembles`、`hazard_classes`、`hazard_oracle`、`phys_waw`、`ra_reaching`、
  `pin_check`、`pinning`、`dead_loads`、`var_lint`、`smfmac_hazards`、`dma_hazards`、`lite_contract` …）
- **硬件探针**（`tools/`，**29 个脚本**）：`vendor_validate.sh`、`vendor_counters.sh`、`vendor_disasm.sh`、
  `ab.sh`、`opt_ab.sh`、`bit_exactness.sh`、`ohat_repro.sh`、`buffer_probe.sh`、`cvtchain_probe.sh`、
  `mfma_unscaled_probe.sh`、`mfma_cform_probe.sh`、`store_x2_probe.sh`、`last_arriver{,8}_probe.sh`、
  `safeasm_window_validate.sh`、`race_repro.sh`、`llvm_build_llc.sh`、`llvm_conformance.sh` …
- **环境开关**：`AMDC_BACKEND/DIALECT/SCHED/SCHED_OPTS/LOWER_OPTS/RA_*/KEEP_MIR/PIN_VMWAIT/…` 共 18 个
- **LLVM 测试覆盖审计**（`docs/llvm-feature-audit.md` + `tools/llvm_feature_audit.py`）：对 LLVM AMDGPU CodeGen 的
  **4996 个测试**（3450 `.ll` + 1546 `.mir`）逐一核对"能否用我们的 Rust DSL 构建并编译正确"→ COVERED 16 / PARTIAL 68 /
  OUT-OF-SCOPE 4912；可执行侧 `tools/llvm_opcode_sweep.sh` 独立复跑 **141/141 次运行发射+汇编通过、90 个助记符全部汇编、
  MOp 变体 108/114 可达**；`tools/llvm_feature_probe.sh` 函数级特性 **9/9 PASS**；与 `llc` 对照 **5 IDENTICAL / 17
  SAME_MNEMONIC / 0 DIFFER**。
- **调度预设**：`_t` / `_sv` 家族用 `ld` 预设（scoreboard 部分 wait + `ds_read` 提升）；`_vs128` / `_vs5/6/9` /
  `_kvsplit` / `_vsm` / `_t128` / `_g*` 用完整重排 list scheduler；`AMDC_SCHED=0` 可关。
- **测量协议**（写进验收规则）：`.s`/`.co` 溯源核对每条变体、≥8–12 轮交错 A/B、同轮厂商、
  split 形状必须反复启动而非单次 5/5

---

## 8. 复现命令（容器内）

```bash
# 构建
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                 # 期望 29 targets / 117 tests 全绿

# 发射 kernel 并汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_vs128 --out vs128.s
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 正确性 + 性能（同轮跑我们的、厂商的、CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_WARMUP=150 REF_ITERS=100 ./build/ref_host <vendor.co>

# 确定性 / split 形状压力
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh      # _legacy_auto = 旧手写集
RUNS=300 bash tools/ohat_repro.sh                                       # 300 次启动 md5（OHAT_PREFIX=fmha_fwd_hd128_mxfp8 = 旧集）
HIP_VISIBLE_DEVICES=6 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_dsl  # 全套验收 + 同轮旧集对照 + 布局中位数
```

---

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| `current-status.md` | **本文件：权威当前快照** |
| `dsl-expression-layer.md` | **运算符表达式层**：`I32/F32/Cond/Ptr` + Rust 运算符，纯数据树经 `bind_*` 1:1 降低到现有 Op；demo 与手写版 asm 逐字节相同；不能自动做的（fma/packed/fragment/协议）及原因 |
| `fmha-dsl-tour.md` | **FMHA 逐段教程**：真实 kernel 逐段特性 + Rust safe 保证（面向没写过 Rust kernel 的读者） |
| `rust-safety.md` | **Rust 安全性如何保证**（全 crate `#![forbid(unsafe_code)]`、零依赖、类型层 ISA 规则、验证器、panic 策略、安全边界） |
| `rust-safety-by-example.md` | **逐项 + 真实代码**：DSL kernel 写法下编译器如何保证 Rust safe（A1–A10），手写 asm 时 safe-asm 如何保证（B1–B16），29 个编译失败负例 |
| `llvm-feature-audit.md` | **LLVM AMDGPU CodeGen 测试覆盖审计**（4996 个测试逐一核对 + 可执行验证 + llc 对照） |
| `gpu-semantics-verifier.md` | **GPU 语义验证器**：global/LDS/VGPR/SGPR 的越界与生命周期证明（模型、合同、错误分类、trust 预算） |
| `gpu-semantics-audit.md` | **独立对抗审计**（v4 审计 of record）：攻击表、发现的 soundness 漏洞与修复、四轴分离 verdict、信任账本与剩余边界 |`n| `gpu-semantic-protocol.md` | **四轴协议模型（v4）**：到达/完成/可见/代际分开建模；多生产者/多消费者双向义务 O1–O4 静态 discharge；全局 region + last-arriver 模式；信任账本 |
| `design.md` | Rust DSL 与自研编译器设计规格 |
| `status.md` | 25 轮逐轮实现记录（含所有正负结果与归因） |
| `vendor-spec.md` | 厂商 kernel 事实 + §7.5–§7.11 认证表与归因 |
| `final-report.md` | 目标 → 现状 → 证据 → 差距 的总结 |
| `verification.md` | 独立验证（寄存器记账、后端等价、DMA、正确性 bug 线） |
| `scheduling.md` / `llvm-tail-integration.md` / `llvm-tail-conformance.md` | 调度器与 LLVM tail 分工/一致性 |
| `safe-asm-design.md` | safe-asm 设计与 4 轮对抗评审 |
| `last-arriver-memory-model.md` | last-arriver 原语与内存序契约（含 8-wave 探针结果） |
| `kernel-gen2-design.md` / `gen3-design.md` / `gen4-tiling-design.md` / `gen5-window-spec.md` / `gen6-*.md` / `gen9-vendor-mirror-design.md` / `unscaled-pv-design.md` / `v-window-cut-plan.md` | 各代设计/实验记录（均已关闭，见各自状态行） |
| `kvsplit-design.md` / `kvsplit-results.md` | KV-split 设计与结果（已落地为默认路径之一） |
| `occupancy.md` / `fmha-reference.md` / `vendor-structure.md` | 占用率研究、参考 kernel 事实、结构对照 |
