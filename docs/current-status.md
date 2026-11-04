# 当前状态总览（authoritative snapshot）

本文件是仓库的**唯一权威当前状态快照**，对应 tag `v5.0-route-b`（路线 B 完成：布局层出厂、MXFP8 GEMM、AGPR、主机侧生成）。
数字来自 MI355X / gfx950（`mi355-gpu-53`）上的同轮硬件测量，复现命令见 §8。
厂商事实在 [`vendor-spec.md`](vendor-spec.md)，设计规格在 [`design.md`](design.md)；历史逐轮记录（第 1–75 轮）与各代设计研究在
[`archive/`](archive/README.md)（其中的数字只代表当时的状态）。

---

## 1. 一句话状态

Rust 类型化 kernel DSL + 自研编译器（**寄存器分配与指令调度始终是我们自己的**，LLVM 只做 post-RA tail），
在 gfx950 上运行厂商 ABI 兼容的 MXFP8 FMHA 与 MXFP8 blockscale GEMM：

- **出厂 FMHA = v3 布局层 kernel**：默认路径 `fmha_fwd_hd128_mxfp8_auto` 的 4 个 kernel 是
  `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}`（`crates/amdgpu-tile/src/fmha.rs`）。K/V/K-scale/Q 的每个地址、
  K/V/K-scale DMA、bf16 epilogue 与 KV-split 转置暂存都由 `amdgpu-layout` 的布局推导，`amdgpu-isa` 的 atom 描述指令，
  `amdgpu-tile` 的 `TiledCopy`/`TiledMma`/`partition_*` 生成 IR；主机合同由张量布局导出。asm md5（`--backend own`）：
  `_vs128` `68d2832b`、`_t128` `e4c7e75d`、`_kvsplit` `e4fc18b1`、`_kvsplit_combine8h` `24a95a51`。
  上一版 tile 库集（`dsl_*`，`fmha_fwd_hd128_mxfp8_dsl_auto`）与旧手写集（`fmha_fwd_hd128_mxfp8_legacy_auto`）保留供 A/B，md5 不变。
- **FMHA 性能**（最近一次 `port_gate`，§4.6）：B1H8S2048 默认路径 **672.7 TF/s**（同轮厂商约 536–539，≈1.25×）；
  `_vs128` 每 CU 64 blocks **446.4**（8 布局中位数 449.9，旧手写 453.7）；铺满网格 B4H8S2048 **1421.3** / B1H8S8192 **1978.6**（≈0.84× 厂商）；
  B2H8S2048 `_t128` **1025.0**。
- **MXFP8 blockscale GEMM**：`gemm_mxfp8_bs_v4`（`crates/amdgpu-kernels`）对照 aiter PR #6696 4-wave pin-AGPR，4096×4096×8192：
  **3148 vs 3278 TF/s = 0.960×**（v5.2 门禁；rocprof-report 诊断轮见分析文档 §13），`tools/gemm_gate.sh` PASS（§4.7）。
- **AGPR 已支持**（寄存器类 A1–A16、acc_cd MFMA、`ds_read_b128 → AGPR`、`v_accvgpr_*`、统一 512 寄存器预算与 hazard）：[`agpr.md`](agpr.md)。
- **主机侧生成**：`amdc gen-host` 从 kernel 合同生成 C++ 启动/ABI 头与 JSON（[`host-gen.md`](host-gen.md)）；出厂 `ref_host` 使用生成的命名字段。
- 精度 **mean 8.3e-4**（厂商 1.5–2.9e-3）；15/15 形状、64 行 bad=0；确定性与 split 压力全过。
- `cargo test` **67 个测试目标 / 314 项全绿**。
- 严格 GPU 语义证书：`--shipped` **24 trusted assumes / 100 trusted contracts / 0 unproven**；`--legacy` **44 / 216 / 0**；GEMM v3 / v4 **3 / 19 / 0**。

---

## 2. 编译器流水线

```
DSL(Rust) ──► typed IR (Op 84 种 / Stmt 47 种) ──► IR 验证器 + gpu_verify（符号区间 + 协议）
            ──► lowering（含 scalarize / scalarize2：一致的整数链进 SGPR）──► 我们的 MIR（123 个 MOp）
            ──► 我们的跨基本块调度（list / hoist / swp / taildup）
            ──► 我们的线性扫描 RA（两阶段 + LDS spill + S4 描述符类 + AGPR 类）
            ──► post-RA MIR 导出（物理寄存器、已排序）
                 ├─ llvm-tail（默认）：llc -start-before=gcn-create-vopd … 只跑 tail pass
                 └─ own：我们自己的 hazard / wait / descriptor 实现
            ──► .s ──► ROCm clang 只做汇编 ──► .co ──► HIP 启动（主机 ABI 由 amdc gen-host 生成）
```

| 环节 | 归属 | 说明 |
|---|---|---|
| DSL / IR / 验证 | 自研 | 类型化、无 unsafe、单赋值循环变量（`var_lint`）；布局层在构建期证明覆盖/连续/操作数匹配 |
| Lowering / MIR | 自研 | 内部 SSA-ish MIR + 效应/副作用模型；一致值进 SGPR（`scalarize`、`scalarize2`、`waveuni`） |
| **寄存器分配** | **自研** | 线性扫描、两阶段（path-aware reaching-writer 检查）、LDS spill、S4 描述符类、AGPR 类与统一 512 预算 |
| **指令调度** | **自研** | 块内 list scheduler、superblock/hoist、软件流水 `swp`、`taildup`、依赖图 / 效果模型 / 等待模型（[`scheduling.md`](scheduling.md)） |
| hazard / wait / descriptor | llvm-tail 或 own | 两个后端指令流逐条一致（`tail_invariance` 守护） |

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`）。ROCm 10 不自带 `llc`，用
`tools/llvm_build_llc.sh` 从同一 commit 链接 `libLLVM.so` 得到；ROCm 7.0.2（`llvm20`）作为备选方言。
找不到 pin 的工具目录时 **llvm-tail 直接报错**，不静默降级。

**指令覆盖**：123 个 MOp 全部有 own emit；其中 117 个在 `llvm20.tbl` **和** `llvm24.tbl` 都有行。
DMA、LDS 直达、`buffer_load … lds`、16x16x128 scaled MFMA 等为 own-only（使用它们的 kernel 自动选 own 后端）。

### 2.1 GPU 语义验证（`gpu_verify`，`AMDC_GPU_VERIFY=off|warn|strict`）

编译器**感知 GPU 指令语义**：在 DSL IR 层做符号区间分析（多项式 + 循环不动点 + 合同 + 声明的上下界），在最终 MIR 层检查寄存器类别与
**路径敏感**的等待/生命周期。文档：[`gpu-semantics-verifier.md`](gpu-semantics-verifier.md)（模型与边界）、
[`gpu-semantic-protocol.md`](gpu-semantic-protocol.md)（四轴协议）；历史审计记录：[`archive/gpu-semantics-audit.md`](archive/gpu-semantics-audit.md)。

```bash
amdc gpu-verify <kernel>|--all|--shipped|--legacy [-v] [--strict]   # 报告 proven/发现/trusted 清单/证书
```

**已证明（出厂 v3，`--strict` 0 错误 0 警告）**：

| 类别 | 证明内容 | 出厂覆盖（`_vs128` / `_t128` / `_kvsplit` / combine） |
|---|---|---|
| **LDS** | 每次 DS 访存/DMA（含 64 lane）在声明段内；环形/双缓冲槽位由不动点证明；无下溢/回绕 | 600 / 336 / 760 / – |
| **LDS tile 生命周期** | tile 分区/别名、produce/consume/version（发布于 barrier）、跨角色 stale/race/pending；周期门；计数版 `vmcnt(n)` 完成规则 | 73,600 / 36,800 / 299,520 次读 |
| **全局/buffer** | 访问落在声明 extent 内（`arg_tensor` + `ka_is/grid_is/ka_one_of`）；描述符不得高估 `num_records` | 106 / 92 / 78 / 3；`crates/amdgpu-compiler/tests/contract_host_truth.rs` 校验合同与 host 公式一致 |
| **VGPR/SGPR/AGPR** | 保留寄存器、元组对齐、上限；MFMA A/B 与 D 不重叠；AGPR 统一预算（arch VGPR + AGPR ≤ 512） | 全部 kernel 默认强制 |
| **寄存器生命周期** | 路径敏感 use-before-wait（任一路径、含循环携带在途值） | 默认强制 |
| **barrier 协议** | 波分歧 if/else 两侧动态 barrier 数相等 | 18 / 8 / 18 |

区间分析的位级规则：`x ^ y`、`x & m` 用"可能置位的位"收窄（swizzle 基址 `(row << 7) ^ (chunk << 4)` 可证 ≤ 2032）；
多项式非负判定使用 `ka_range` / `ka_set` / `sym_range` 的上界（盒子检验）。两条规则都有攻击测试（`crates/amdgpu-compiler/tests/gv_bits.rs`）。

**信任预算（证书逐条列出）**：`--shipped`（v3 四个）24 / 100 / 0；`--legacy`（旧手写 8 个）44 / 216 / 0；`gemm_mxfp8_bs_v3`、`_v4` 各 3 / 19 / 0。
每次运行打印 `certificate: N trusted assumes, M trusted contracts, K unproven`。

**未证明/范围外（诚实清单）**：跨工作组数据竞争与别名、i32 回绕（A3 假设）、从内存加载的指针/索引、safe-asm 块内部访存、
tile 内字节级重叠、角色数 2 / 迭代数相等假设。

**v4 协议模型**：**到达**（`s_barrier`）、**完成**（`wait_vm/wait_lds/fence_release`，`vmcnt(n)` 按"其后至少 n 条更新的 LDS-direct DMA"计数）、
**可见**（LDS 经 barrier；全局 release→barrier→acquire + `sc1`）、**代际**（仿射/周期 version-slot 代数 + 周期门）；四条义务 O1–O4 静态 discharge，
判不出的进可计数信任账本。全局内存实例（`buf_region/...`）与 last-arriver 融合 combine 模式见 [`last-arriver-memory-model.md`](last-arriver-memory-model.md)。

---

## 3. DSL 与库

| 维度 | 内容 |
|---|---|
| 入口 | `FnBuilder`（约 160 个公开方法）→ `Function`（kernarg ABI） |
| 值模型 | `ValueId` / `VarId`（循环变量）/ `Opnd` / `DType`（`Frag2/4/8/16` 等）；表达式层 `I32/F32/Cond/Ptr`（[`dsl-expression-layer.md`](dsl-expression-layer.md)） |
| 计算 | scaled 32x32x64 与 **16x16x128** f8f6f4 MFMA（op_sel 选 E8M0 字节）、C 操作数形式、unscaled MFMA、`v_smfmac`、f16/bf16 MFMA；AGPR 形式（acc_cd） |
| 向量/转换 | VALU int/fp、`pk_*`、`perm`、`permlane32_swap`、`pknorm`、`scalef32`、`cvt_pk_fp8`、`cvt_pk_bf16`、`exp/rcp/log`、`max3`、`readfirstlane`、`v_accvgpr_*` |
| 内存-FLAT | `global_load/store`、`global_load_lds`（含 SADDR 形式） |
| 内存-MUBUF | `buffer_rsrc` / `buffer_desc` / `buffer_desc_dyn`、`buffer_load/store x1/x2/x4`、`buffer_load … lds`、m0 / imm12 / soffset |
| 内存-LDS | `ds_read/write_b32/b64/b128`、`ds_read_b64_tr_b8` 转置读、ring / 双缓冲 / 私有区 |
| 控制流 | `begin_loop`/`end_loop`、`begin_if`（wave 级）、`versioned` 双代码副本、`Stmt::VarAsm(N)` 桥接手写 asm |
| 布局层（路线 B） | `amdgpu-layout`（`Ix`、`IntTuple`、`Layout` 代数、GF(2) `Swizzle`、TV 布局、`IssuePlan`/`layout_to_offsets`、证明）；`amdgpu-isa`（atom 表）；`amdgpu-tile`（`TiledCopy`/`TiledMma`/`partition_{lds,dma,buffer_dma,global}`、布局导出的合同）。入门 [`layout-tour.md`](layout-tour.md)，实施记录 [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) §9 |
| tile 库（上一版） | `amdgpu_dsl::tile` L1–L6（[`dsl-tile-abstractions.md`](dsl-tile-abstractions.md)）；LdsPlan/标记、Pipeline、softmax、combine 仍被 v3 复用 |
| 实验面 | `FMHA_OPT` 实验 token（全部 opt-in）；`FMHA_OPT_SCOPE` 把 token 限定到别名 kernel（如 `fmha_fwd_hd128_mxfp8_v3s_*`）做同轮 A/B |
| ABI | 与厂商**逐字节兼容的 656 字节 kernarg + 网格/数据布局**；`_auto` 按形状 guard 调度；主机侧头由 `amdc gen-host` 生成 |

**crate 地图**（全部 `#![forbid(unsafe_code)]`、零第三方依赖）：

| crate | 内容 |
|---|---|
| `amdgpu-layout` | 布局代数（零依赖） |
| `amdgpu-isa` | gfx950 copy / MMA atom（依赖 layout） |
| `amdgpu-tile` | TiledCopy / TiledMma / partition、合同、v3 FMHA kernel（依赖 layout、isa、dsl） |
| `amdgpu-kernels` | MXFP8 blockscale GEMM（依赖 dsl、layout、isa、tile） |
| `amdgpu-dsl` | IR、`FnBuilder`、表达式层、tile 库（`src/tile/`）、dsl FMHA、旧手写 kernel |
| `amdgpu-compiler` | lowering、调度、RA、hazard/wait、emit、LLVM MIR 桥、`gpu_verify`、`hostgen`、kernel 注册 |
| `amdgpu-safeasm` | safe-asm 窗口（[`safe-asm-design.md`](safe-asm-design.md)） |
| `amdc` | CLI（emit / check / list / dump / backend / simmodel / probe / gpu-verify / gen-host） |

### safe-asm 子系统（`crates/amdgpu-safeasm`）

手写热点窗口用的安全 Rust asm DSL：typed 寄存器、所有权 token（`Pending` / `Mfma` / `Trans` / `Pw<L,H>` 半写状态）、
等待态验证器、常量总线 const-assert、`Seq` 部分 `lgkmcnt`、drain-closed 循环体契约、op 白名单、
`Seq::pin()` + `pin_check`。经 `Op::Asm`/`Stmt::VarAsm` 接入编译器。`crates/amdgpu-safeasm/tests/ui/` 共 29 个用例（28 个编译失败 + 1 个通过对照）。

---

## 4. Kernel 状态

### 4.1 默认出厂路径：`fmha_fwd_hd128_mxfp8_auto`

| 形状类 | 选择 | 性能（最近一次门禁；厂商为历次同轮测量） |
|---|---|---|
| **B1H8S2048**（基准） | KV-split s=4 + `combine8h` | **672.7 TF/s**（厂商 536–539 → ≈1.25×） |
| 铺满网格 B4H8S2048 / B1H8S8192 | `_vs128` | **1421.3** / **1978.6**（≈0.85× / 0.84× 厂商） |
| 中等形状 | `_t128` / split（guard 选择） | B2H8S2048 `_t128` **1025.0**（≈1.0× 厂商） |
| 单 kernel 每 CU（64 blocks） | `_vs128` | **446.4**；8 种代码布局中位数 **449.9**（旧手写 453.7） |

### 4.2 Kernel 清单

| Kernel | 状态 |
|---|---|
| `fmha_fwd_hd128_mxfp8_auto` | **默认**（guard 选择）：`v3_vs128` / `v3_t128` / `v3_kvsplit` + `v3_kvsplit_combine8h`（`crates/amdgpu-tile/src/fmha.rs`） |
| `fmha_fwd_hd128_mxfp8_dsl_auto` | 上一版 tile 库集 `dsl_*`（`crates/amdgpu-dsl/src/fmha_tile.rs`），A/B 对照用 |
| `fmha_fwd_hd128_mxfp8_v3s_*` | v3 的别名（同代码），配合 `FMHA_OPT_SCOPE=v3s` 对实验 token 做同轮 A/B |
| `fmha_fwd_hd128_mxfp8_legacy_auto` | 旧手写集（`_vs128` / `_t128` / `_kvsplit` / `combine8h`，`crates/amdgpu-dsl/src/kernels.rs`），A/B 对照用 |
| `gemm_mxfp8_bs_v4` / `_v3` | MXFP8 blockscale GEMM（`crates/amdgpu-kernels`）；v4 = 厂商结构（寄存器暂存、AGPR 原地重载），v3 = 不做原地重载的变体 |
| `_t`、`_g2/_g3/_g3o/_pp/_sv/_vs5/_vs6/_vs9/_vsm` | 历史/实验变体，仅按名（负结果，见 [`archive/`](archive/README.md)） |

### 4.3 精度与确定性

- 参考算法：每 128-key 单元精确参考重基准（阈值 120/512）、长序列无 LUT 校正、短序列第二份代码副本 + `svcorr`
- mean err **8.3e-4**（厂商 1.5–2.9e-3）；15/15 形状、16/64 行 parity 15/15、bad = 0
- 确定性 5/5 × N 次；split 形状重复启动单一 md5（`tools/ohat_repro.sh`，`port_gate` 的 STRESS 段）

### 4.4 优化里程

292 → 367（每 CU 差 1.41×）→ 505（t128 分块）→ 605（KV-split）→ 622（kddma）→ 674–705 TFLOP/s（第 75 轮，旧手写集）
→ tile 库集 `dsl_*`（≤3% 内）→ v3 布局层（≤3% 内，`_t128` SADDR DMA +2%）。逐轮记录：[`archive/status.md`](archive/status.md)。

### 4.5 v3 vs tile 库集 vs 旧手写集

`tools/port_gate.sh` 同轮对照，`SHIP_REF` 选参考集（默认旧手写集，`SHIP_REF=fmha_fwd_hd128_mxfp8_dsl` 对照 tile 库集）。
代码布局敏感性：同一 kernel 前置 0..7 条 `s_nop` 使 per-CU `_vs128` 在约 440–456 TF/s 间变化，单一布局对比不可信；门禁用 8 布局中位数。

### 4.6 最近一次 FMHA 门禁（`tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3`，GPU 6）

| 项 | v3 | 旧手写（同轮） |
|---|---|---|
| B1H8S2048 出厂路径（split + combine） | 672.7 | – |
| per-CU 64 blocks（单一布局） | 446.4 | – |
| per-CU 8 布局中位数 | **449.9** | **453.7** |
| B4H8S2048 / B1H8S8192 | 1421.3 / 1978.6 | – |
| B2H8S2048 `_t128`（单一布局 / 8 布局中位数） | 1025.0 / **1022.3** | – / **1028.6** |

BUILD / VERIFY（0 错误）/ ACC16 / ACC64（四条路径 bad=0）/ DET / STRESS 全部 PASS，性能在 `PERF_TOL=0.97` 内：**OVERALL PASS**。
v3 asm md5：`68d2832b` / `e4c7e75d` / `e4fc18b1` / `24a95a51`。`_t128` 另外默认打开 `edgepad`（MFMA 冒险窗口跨块传递，不在块尾补 `s_nop`）与 `salubr`（波内一致的分支条件走 `s_cmp` + `s_cbranch_scc`）：同轮 A/B 布局中位数 +3.3%（单独 +0.7% / +2.1%），采用后门禁对旧手写集 1061.95 vs 1023.06；这两项在 `_vs128`/`_kvsplit` 上为 −0.7..−1.5%，保持 opt-in（`FMHA_OPT=edgepad,salubr`）。`_t128` 默认用 SADDR 形式的 LDS-DMA（两轮 A/B 布局中位数 +1.8% / +2.1%），
`_vs128`/`_kvsplit` 上为 −0.2..−1.4%，保持 opt-in（`FMHA_OPT=saddrlds`）；A/B 数字见 [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) §9.4。

### 4.7 MXFP8 blockscale GEMM

| 项 | 数值（截至 `v5.0-route-b`） |
|---|---|
| kernel / 形状 | `gemm_mxfp8_bs_v4`，M=N=4096、K=8192，256×256×128 CTA tile，4 wave，`v_mfma_scale_f32_16x16x128_f8f6f4`（swap_ab），A/B 片段与部分 C 在 AGPR |
| 性能 vs 厂商（同轮） | **3148 vs 3278 TF/s = 0.960×**（v5.2 门禁，交替 7 进程中位数）；稳定区间 0.95–0.97×，追平尝试、rocprof-report 诊断与上限证据见 §12–§13；演进与分析见 [`mxfp8-gemm-4wave-analysis.md`](mxfp8-gemm-4wave-analysis.md) §11、[`gemm-issue-comparison.md`](gemm-issue-comparison.md) |
| 门禁 | `tools/gemm_gate.sh`：BUILD / VERIFY（strict 0 错误 0 警告，3 trusted assumes / 19 contracts / 0 unproven）/ ACC（4096²×8192、8192×7168×7168、1280×6144×7168 GEMM_OK）/ DET 3/3 / PERF —— **PASS** |
| 编译器配套 | AGPR 寄存器类与 acc_cd MFMA（[`agpr.md`](agpr.md)）、SALU 地址链 `scalarize2`、`ds_read_b64`/`ds_write_b64`、验证器 possible-bits 与盒子规则、`Pipeline::vm_inflight` + 计数版 `vmcnt(n)` |

---

## 5. 已量化关闭的方向（都带反例与归因，详见 [`archive/`](archive/README.md)）

| 方向 | 结论 |
|---|---|
| per-CU 微优化（VALU 削减） | VALU −8.7% 后 wave +1.2%；去掉的槽变成依赖停顿（WAIT_INST_ANY +14.8%），非指令受限 |
| bias MFMA 移除（biasacc/biasc/biasv/biasonce/vpre/kpre2/qkfirst/p4-unscaled） | −0.8% … −15%；那 128 XDL 周期是 barrier 后 LDS 读延迟的遮盖 |
| pinned M-window / K-ahead ring | −5…−7% / −10.5%（in-order 发射 + padding + VGPR 压力 / 相位失衡） |
| gen10 三项相位设计（持久 C、V 预读、相位重平衡） | 硬门槛失败（XDL 地板 1130） |
| MUBUF 转换 | 正确性中立、opt-in；上界 +0.1…1.0% |
| 4 waves/SIMD 占用率 | 不适合 VGPR/LDS 预算；厂商同样 1 block / 8 wave per CU |
| 铺满网格对齐 | ≈0.84×；差距在每 CU 关键路径（2-smfmac 行和结构、分离-C 19 周期、两组 wave 的 barrier 汇合） |

---

## 6. 修复过的真实正确性 bug

| bug | 影响 | 修复 |
|---|---|---|
| 物理 WAW（MFMA 在途结果寄存器被 RA 复用） | opt-in 组合下非确定输出 | `pad_physical_war` 跟踪在途 MFMA 目标区间；MIR 回归测试 |
| 两阶段 RA reaching-writer 误报 | 持久循环变量编译失败 | path-aware 检查；`crates/amdgpu-compiler/tests/ra_reaching.rs` |
| VMEM 宽存储数据 WAR | `global_store_dwordx4` 后数据寄存器被过早覆盖 → split 路径偶发损坏 | `pad_physical_war` 记录宽存储数据区间；`tools/ohat_repro.sh` 压力回归 |
| LDS-DMA 操作数等待放置（AGPR 工作中发现） | 静默误编译 | 见 [`agpr.md`](agpr.md) "Wait placement" 一节 |

---

## 7. 验证与工具

- **测试**：`cargo test` **67 个目标 / 314 项全绿**。主要测试文件：`tail_invariance`、`mir_invariants`、`rust_safety`、`register_accounting`、
  `asm_assembles`、`hazard_*`、`phys_waw`、`ra_reaching`、`pin*`、`var_lint`、`gpu_*`（语义/协议/攻击）、`gv_bits`、`agpr`、`mfma16x16x128`、
  `lds_b64`、`scalarize2`、`gemm_mxfp8`、`tile_l*`；布局层 `crates/amdgpu-layout/tests/{algebra,fmha_vs128}.rs`、
  `crates/amdgpu-tile/tests/{address_equality,dma_equality,bufdma}.rs`（发射 IR 逐 lane 对照旧 reader/copy）；safe-asm `tests/ui` 29 个用例。
- **门禁**：`tools/port_gate.sh`（FMHA：BUILD / VERIFY / ACC16 / ACC64 / DET / STRESS / PERF，`SHIP_REF` 选参考集）、
  `tools/gemm_gate.sh`（GEMM：BUILD / VERIFY / ACC / DET / PERF）。
- **工具**（`tools/`）：厂商对照（`vendor_validate.sh`、`vendor_counters.sh`、`vendor_disasm.sh`）、GEMM（`gemm_ref.sh`、`prof_gemm.sh`、`att_*.py`）、
  A/B（`ab.sh`、`opt_ab.sh`）、压力（`ohat_repro.sh`、`race_repro.sh`）、探针（`*_probe.sh`）、LLVM（`llvm_build_llc.sh`、`llvm_conformance.sh`、
  `llvm_feature_audit.py`）、`md5_kernels.sh`、文档链接检查 `check_doc_links.py`。
- **LLVM 测试覆盖审计**（[`llvm-feature-audit.md`](llvm-feature-audit.md)）：LLVM AMDGPU CodeGen 4996 个测试逐一核对。
- **测量协议**：`.s`/`.co` 溯源、≥8 轮交错 A/B 或 8 布局中位数、同轮厂商、split 形状反复启动；GPU 隔离（每人固定 `HIP_VISIBLE_DEVICES`）。

---

## 8. 复现命令（容器内）

```bash
# 构建与测试
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                 # 期望 67 targets / 314 tests 全绿

# 发射 kernel 并汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --out vs128.s
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 严格证书
bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify --shipped --strict   # 24 / 100 / 0
bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify --legacy --strict    # 44 / 216 / 0

# 主机侧生成
bash tools/container/cargo.sh run -q --bin amdc -- gen-host --shipped --out tests/gpu/generated

# 正确性 + 性能（同轮跑我们的、厂商的、CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh      # _legacy_auto = 旧手写集，_dsl_auto = tile 库集
RUNS=300 bash tools/ohat_repro.sh

# 门禁
HIP_VISIBLE_DEVICES=6 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3                                  # 对旧手写集
HIP_VISIBLE_DEVICES=6 SHIP_REF=fmha_fwd_hd128_mxfp8_dsl bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3  # 对 tile 库集
HIP_VISIBLE_DEVICES=6 bash tools/gemm_gate.sh gemm_mxfp8_bs_v4
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
| [`dsl-tile-abstractions.md`](dsl-tile-abstractions.md) | tile 库（L1–L6）设计 |
| [`dsl-expression-layer.md`](dsl-expression-layer.md) | 运算符表达式层 |
| [`fmha-perf-patterns.md`](fmha-perf-patterns.md) | FMHA 性能承重模式清单 |
| [`gpu-semantics-verifier.md`](gpu-semantics-verifier.md) / [`gpu-semantic-protocol.md`](gpu-semantic-protocol.md) | GPU 语义验证器 / 四轴协议模型 |
| [`last-arriver-memory-model.md`](last-arriver-memory-model.md) | last-arriver 原语与内存序契约 |
| [`agpr.md`](agpr.md) | AGPR 支持 |
| [`host-gen.md`](host-gen.md) / [`hostgen-json-schema.md`](hostgen-json-schema.md) | 主机侧生成 |
| [`rust-safety.md`](rust-safety.md) / [`rust-safety-by-example.md`](rust-safety-by-example.md) | Rust 安全 |
| [`safe-asm-design.md`](safe-asm-design.md) | safe-asm 设计 |
| [`scheduling.md`](scheduling.md) / [`llvm-tail-integration.md`](llvm-tail-integration.md) / [`llvm-tail-conformance.md`](llvm-tail-conformance.md) / [`llvm-feature-audit.md`](llvm-feature-audit.md) | 调度器、LLVM tail、CodeGen 覆盖审计 |
| [`vendor-spec.md`](vendor-spec.md) / [`fmha-reference.md`](fmha-reference.md) | 厂商 FMHA 事实与认证表、参考 kernel |
| [`mxfp8-gemm-4wave-analysis.md`](mxfp8-gemm-4wave-analysis.md) / [`gemm-issue-comparison.md`](gemm-issue-comparison.md) | 厂商 4-wave GEMM 分析与逐指令对比 |
| [`archive/README.md`](archive/README.md) | 历史归档索引（逐轮记录、各代设计研究、终局报告、审计记录） |
