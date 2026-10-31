# 当前状态总览（authoritative snapshot）

本文件是仓库的**唯一权威当前状态快照**，最后更新于 `Main` = `8fec359`（tag `v2.2-round75-final`）。
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
- `cargo test` **111 项 / 28 个测试目标全绿**

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
| `fmha_fwd_hd128_mxfp8_auto` | **默认**（guard 选择下面三者） |
| `_vs128` | 单 kernel 快路径（铺满网格 / 64-block 基准），256-row block |
| `_t128` | 128-row block，中等/混合形状的 fallback |
| `_kvsplit` + `combine{4,4h,8h,16h}` | KV-split 两段启动（split 形状）；`combine8h` 为默认 combine |
| `_t` | 精确 recipe kernel（llvm-tail，311 TFLOP/s；冻结，仅作正确性对照） |
| `_g2/_g3/_g3o/_pp/_sv/_vs5/_vs6/_vs9/_vsm` | 历史/实验变体，opt-in 或仅按名（负结果记录） |
| `biasacc` / `p4` / `mubufd` / `cut1` / `saddrlds` / `pmdead` / `epdirect` / `eplds,epw` | opt-in 实验（多数中性或负；`epw` 是 VMEM 宽存储 WAR 的灵敏度探测器） |

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

- **测试**：`cargo test` **28 个目标 / 111 项全绿**；21 个测试文件（`tail_invariance`、`mir_invariants`、
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
bash tools/container/cargo.sh test                 # 期望 28 targets / 111 tests 全绿

# 发射 kernel 并汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_vs128 --out vs128.s
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 正确性 + 性能（同轮跑我们的、厂商的、CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_WARMUP=150 REF_ITERS=100 ./build/ref_host <vendor.co>

# 确定性 / split 形状压力
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh
bash tools/ohat_repro.sh fmha_fwd_hd128_mxfp8_kvsplit          # 300 次启动 md5
```

---

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| `current-status.md` | **本文件：权威当前快照** |
| `llvm-feature-audit.md` | **LLVM AMDGPU CodeGen 测试覆盖审计**（4996 个测试逐一核对 + 可执行验证 + llc 对照） |
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
