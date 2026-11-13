# 当前状态总览（authoritative snapshot）

本文件是仓库的**唯一权威当前状态快照**（v11，Main `a65de21a`：在 v10 四层重构之上，本轮按 [`two-weavers-plan.md`](two-weavers-plan.md) 建成“四层 + 两个 Weaver”结构，
并做了 W4 性能基线、W5 厂商 FMHA 复刻与 W6 落后族优化，见 §1.0；v9.2 的记录保留在 §1.1）。数字来自 MI355X / gfx950（`mi355-gpu-53`）上的同轮硬件测量，复现命令见 §8。
厂商事实在 [`vendor-spec.md`](vendor-spec.md)，设计规格在 [`design.md`](design.md)；历史逐轮记录与各代设计研究在
[`archive/`](archive/README.md)（其中的数字、kernel 名与开关只代表当时的状态）。

---

## 1. 一句话状态

Rust 类型化 kernel DSL + 自研编译器（**寄存器分配与指令调度始终是我们自己的**，LLVM 只做 post-RA tail），
在 gfx950 上运行厂商 ABI 兼容的 MXFP8 FMHA、MXFP8 blockscale GEMM 与融合 MLP：

- **出厂 FMHA = v3 布局层 kernel**：默认路径 `fmha_fwd_hd128_mxfp8_auto`（主机侧按形状选择）由
  `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}` 组成（`crates/amdgpu-kernels/src/fmha/`）。每个地址、DMA、epilogue 与
  KV-split 暂存都由 `amdgpu-layout` 的布局推导，`amdgpu-isa` 的 atom 描述指令，`amdgpu-tile` 的组件生成 IR；主机合同由张量布局导出。
  asm md5（`--backend own`，Main `9cffb86f`）：`_vs128` `ff51fd09`、`_t128` `a1ee38ee`、`_kvsplit` `11e2b907`、`_kvsplit_combine8h` `24a95a51`（v11：`_vs128` 改为 W5 复刻配置，再经 SCC 修复重排，见 §1.0；v11.1：VCC 规则改为 1 个等待周期（用户决定接受 vs128 B4 S2048 −0.5..0.8%），57 个 kernel 变化，perf_ab 全族 PASS、`port_gate` PASS）。
- **FMHA 性能（真实数据，`REF_DATA=normal`，2026-11-08，GPU 5）**：`port_gate`：B1H8S2048 默认路径（KV-split）**740 TF/s（1.025× 厂商）**；
  铺满网格 B4H8S2048 **1804（0.976×）** / B1H8S8192 **2416（0.988×）** / B2H8S2048 **1279（0.978×）**；`_vs128` 每 CU 64 blocks **694（0.959×）**；`_t128` B2H8S2048 layout 中位数 **1173（0.901×）**。
  25 形状矩阵（[`perf-matrix.md`](perf-matrix.md)）：0.929–1.063×，几何平均 **0.992**（之前 0.980）。最近一步是厂商的 XCD 感知 block 顺序（K/V 的 L2 命中率 20 % → 77 %，[`fmha-wave-org.md`](fmha-wave-org.md)）。调色板数据（`REF_DATA=lcg`，旧口径）下 softmax 几乎每 tile 回滚、掩盖了
  快速路径成本，旧的 0.92× 口径偏乐观；这一轮的改进（懒更新、≥ 4 tile 用厂商配方、真实数据分派、bias 单次 MFMA 且无零片段拷贝、`ldsink` / `wsink`）
  与否定结果（copy 路径、3 槽环、厂商 copy 形态、重平衡）见 [`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md) 与 [`lgkm-lookahead.md`](lgkm-lookahead.md)。
- **MXFP8 blockscale GEMM**：`gemm_mxfp8_bs_v4`（`crates/amdgpu-kernels/src/gemm/`）对照 aiter PR #6696 4-wave pin-AGPR，4096×4096×8192：
  **0.978–0.994×** 厂商（`tools/gemm_gate.sh` PASS，§4.7；v9 加入 epilogue 前的 `wait_vm(0)` 与 lowering 窥孔后 0.978–0.981×）；asm md5 `e4369386`。
- **融合 MLP**：`mlp_fp8_ws_{fused,g1,g2}`（`crates/amdgpu-kernels/src/mlp/`），布局层 P 交接证明见 [`fusion-layout-handoff.md`](fusion-layout-handoff.md)。
- **AGPR 已支持**（寄存器类 A1–A16、acc_cd MFMA、`ds_read_b128 → AGPR`、`v_accvgpr_*`、统一 512 寄存器预算与 hazard）：[`agpr.md`](agpr.md)。
- **主机侧生成**：`amdc gen-host` 从 kernel 合同生成 C++ 启动/ABI 头与 JSON（[`host-gen.md`](host-gen.md)）。
- 精度：**厂商参数校准（vendor parity）**——每个形状、每条路径同轮同输入下 mean_err ≤ 1.10× 厂商且 bad ≤ 厂商（§4.3）；确定性与 split 压力全过。
- **组件化与编织**：每个出厂 kernel 都由"算子图 → 生成的骨架 → slot 实现 + stage 实例"构建（[`opgraph.md`](opgraph.md)、
  [`fusion-stage-library-design.md`](fusion-stage-library-design.md)）；编织区域带寄存器规划（[`weaver-regs.md`](weaver-regs.md)）与按区域调度策略
  （[`weaver-sched.md`](weaver-sched.md)），出厂 asm 不变。
- **DSL IR 文本**：`amdc print-ir --stage dsl` 输出完整可解析的 `irv1`，`--from-ir` 编译手改文本；全部 kernel 往返后双后端 asm 与证书逐字节相同（[`ir-text.md`](ir-text.md)）。
- **aiter 基础算子覆盖（v9）**：计算密集的 GEMM 族、attention（bf16 prefill / paged decode / MLA）、fused MoE、attention 反向、implicit-GEMM 卷积见
  [`op-coverage-design.md`](op-coverage-design.md) §7（同进程同数据对照 aiter `.co`）；访存受限的逐元素 / norm / 量化 / RoPE / KV cache / MoE 辅助 / 采样 / top-k 见
  [`op-coverage-membound.md`](op-coverage-membound.md) 的总表与 aiter 覆盖表（have / missing / why）；汇总见 §4.2。
- **tile-op 层**（[`op-coverage-tileops.md`](op-coverage-tileops.md) §6.6）：访存宽度族、原子、跨 lane、类型转换（含 fp4 / fp6）、unscaled f8f6f4 / bf16 MFMA、
  `v_pk_fma_f32`、片段视图与原地拼接、`tileop::ewise / apply2 / allreduce`，每条都过“属性表 → 双后端 → hazard oracle → 硬件探针”闭环。
- **64 位寻址**：`ptr_mad` 把描述符基址移到 tile / expert / slot（> 4 GiB 张量），验证器证明移动后的描述符仍在 extent 内。
- **结构化诊断**：`--json`、`amdc summary`、`amdc srcmap`（[`diagnostics.md`](diagnostics.md)）。
- `cargo test` **144 个测试目标 / 586 项全绿**（Main `a65de21a`）；md5 清单 **240 个 kernel**、0 差异；Python 节点检查 `python/tests/run_all.sh` OVERALL PASS：62 / 62 算子检查 + 可达 kernel 编译 / 汇编扫描 + hazard oracle + **每个可达 kernel 的严格 gpu-verify**（v11 新增的守卫，`crates/amdgpu-kernels/tests/verify_reachable.rs`）。
- 严格 GPU 语义证书（Main `a65de21a`）：`--shipped` **21 trusted assumes / 100 trusted contracts / 0 unproven**；GEMM v4 **1 / 19 / 0**（v11.1 后 amdgpu-sym 非负性检验证明了原先 2 个信任假设）；`mlp_fp8_ws_fused` **0 / 9 / 0**；融合 decode `pa_decode16_bf16_hd128_p16_f` **2 / 18 / 0**（与 v9.2 相同）。
  信任清单与每一项的去留见 [`trust-inventory.md`](trust-inventory.md)。

### 1.0 v11 本轮（2026-11-11 / 12）：四层 + 两个 Weaver

**结构（已建成，[`two-weavers-plan.md`](two-weavers-plan.md)，接口设计 [`two-weavers.md`](two-weavers.md)，v10 分层 [`refactor-4layer-plan.md`](refactor-4layer-plan.md)）**

| 部分 | crate | 本轮状态 |
|---|---|---|
| L1 图 | `amdgpu-graph` | 算子图、融合与分派规则（`dispatch.rs`）、`compose_facts.rs`（由每个 kernel 的 Weaver 计划生成，`tools/regen_compose_facts.sh`） |
| L2 stage | `amdgpu-stage` | stage 实例与合同、流水线模板（`pipeline.rs`：`TwoPhaseOffset` / `OneWindow` / `RegStaged`，`unroll2`、`ahead2`）、前端 Weaver（`weave.rs`：单元捕获、`free` / `interleave` / 预算 / 环声明的计划 API） |
| L3 功能块 | `amdgpu-tile`、`amdgpu-layout`、`amdgpu-isa` | tile op、片段、copy / DMA、softmax、归约、epilogue、指令 atom |
| L4 IR + 后端 Weaver | `amdgpu-dsl`、`amdgpu-codegen`、`amdgpu-target` | 计划（`plan.rs`：phase / unit / ring / sync / 寄存器约束）、后端 Weaver：B4 自由 phase 内的指令级交织、B5 跨迭代放置（环槽令牌、按依赖重算等待）、B6a phase 寄存器预算、B6b `reg_rotate`、B6c 按 phase 的分配回退；验证器（`gpu_verify`，本轮新增 `GV-PLAN-RING`） |

全部 12 个 crate 零 `unsafe`（10 个声明 `#![forbid(unsafe_code)]`；`amdgpu-graph` / `amdgpu-stage` 源码中没有 `unsafe`）。

**W4 性能基线**（[`perf-ab.md`](perf-ab.md)，[`perf-v11-vs-v10.md`](perf-v11-vs-v10.md)）：`tools/perf_ab.sh` 对每个集成 kernel 族做同轮新旧 A/B（含厂商列），每个改变行为的合并都过它。v11 对 v10 的全族同轮对照：**没有任何族变慢**；gemm 1.040（组合搜索的规则更新）、mla 1.139、decode 1.011、fmha_mxfp8 1.015，其余族在噪声内。本轮新增 `gemv` 族（小 M GEMM，60 例）。

**本轮改变的出厂 md5 与门禁**（数字与证据见所引文档）：

| kernel | 变化 | 门禁 |
|---|---|---|
| `fmha_fwd_hd128_mxfp8_v3_vs128` | W5 复刻配置（无缩放 PV、两 tile 一趟、跳转线程化、冷回滚块、厂商指令形式），再经 SCC 修复（`b7a3a2ac`），v11.1 VCC 规则后 `ff51fd09` | `port_gate` PASS；perf_ab fmha_mxfp8 1.017（[`fmha-replica.md`](fmha-replica.md) §4.9） |
| `mla_decode_bf16_p*` / `_hp_p*`、`attn_combine_bf16_hd512` | 每 kernel 打开带重排的 list 调度器（依赖 SCC 修复） | perf_ab mla 1.139，4/4 IMPROVED；正确性标记（[`w6-laggards.md`](w6-laggards.md) §6） |
| `pa_decode16_bf16_hd128_p16_c` | 同上，仅 last-arriver 形式 | perf_ab decode 1.012–1.015（§6） |
| 7 个 split-K GEMM（`*_64_splitk`、`*_128_splitk`、`gemm_a8w8_blockscale_splitk`） | fp32 部分和只写含真实行的 16 行块 | 全行检查 0 bad、严格验证、`run_all` PASS；perf_ab gemv 1.060（[`w6-laggards.md`](w6-laggards.md) §7.1） |
| 一批由 list 调度器排序的 kernel（含 `_vs128`） | 后端：描述符伪指令声明 SCC 写入后调度重排（正确性修复） | backend 的门禁（perf_ab、`port_gate`、`run_all`）；变化的 kernel 见 `tests/golden/md5_manifest.tsv` 的提交历史 |

新增但不分派的 kernel：`fmha_fwd_hd128_mxfp8_v3_ring2`（距离 2 的拷贝环 + B5，测试载体）、`*_64_splitk_lc`（kernel 内 last-arriver 归约，4 个 dtype）。

**与厂商的比值（同轮，比 = 厂商时间 / ours）**，在 §1.1 的表上本轮的变化：

| 族 | v11 |
|---|---|
| 出厂 MXFP8 FMHA（`port_gate`，`_vs128` 复刻配置） | 每 CU 0.971×、B4H8S2048 0.983×、B1H8S8192 0.991×、B2H8S2048 0.973×、8 布局中位数 0.972×（[`fmha-replica.md`](fmha-replica.md) §4.9）；KV-split 与 `_t128` 本轮未变（§1.1 的数字） |
| MLA decode | 对 v10 快 1.07–1.23×（几何 1.139） |
| paged decode16 | 对 v10 快 1.02–1.04×（split 形状） |
| GEMM 族 | 对 v10 几何 1.040（规则）；小 M（M ≤ 64）再 +2–14 %（gemv 几何 1.060）；MXFP4 仍 0.85–0.90×（已关闭，原因见 [`mxfp4-gemm-profile.md`](mxfp4-gemm-profile.md) W6 节） |
| bf16 / hd192 prefill、反向、MoE、访存受限 | 与 v10 持平（噪声内）；hd192 小网格 0.77–0.81× 仍为最大差距（[`w6-laggards.md`](w6-laggards.md) §5） |

**W5：厂商 MXFP8 FMHA 1:1 复刻（[`fmha-replica.md`](fmha-replica.md)）**
- 工具：`tools/isa_diff.py`、`isa_forms.py`、`isa_cfg.py`。
- 复刻 kernel `fmha_fwd_hd128_mxfp8_v3_replica`：热循环 437 对厂商 439 条指令。
- 采用为出厂 `_vs128` 的五步：无缩放 PV、两 tile 一趟、厂商指令形式、跳转线程化、冷回滚块。
- 测得为负、未采用：钉住寄存器 −4…−12 %，厂商 M 阶段顺序 −4…−8 %。
- 距离 2 拷贝环（ring2）：B5 放置后仍 −2.5…−6.5 %。加上 B6b 寄存器轮转与厂商 M 阶段顺序后为每 CU 0.829× 厂商（出厂 0.982×）。结论：厂商的组合需要它自己的寄存器计划，这条线在当前寄存器计划下关闭（§4.10、§5）。
- 差异分类表见 §5。

**W6：落后族（[`w6-laggards.md`](w6-laggards.md)）**
- 采用：
  - MLA / decode16 的调度器策略（+13.9 % / +1.5 %）；
  - 小 M split-K 只写真实行（+6 %）。
- 作为构件合入、不分派：
  - decode16 / MLA / bf16 v3 的 Weaver 单元；
  - kernel 内 last-arriver 归约，以及 `buf_last_arriver` 对 buffer 形式的验证扩展；
  - `Pipeline::ahead2` 与 `KvRing3::new_kv`；
  - Weaver 计划 API。
- 测得为负、记录原因后关闭：
  - lowering token 用在落后族上（中性）；
  - hd192 分派变体（−12…−17 %）；
  - bf16 v3 把 PV 移入 softmax 的 B4 交织（−7…−8 %）；
  - 单组 in-wave 流水原型（−9…−20 %）；
  - 更深的 GEMV 权重流（−0…−4 %；稳态流已达约 8.7 TB/s，小 M 差距是每次调用约 8.8 µs 的固定部分）；
  - MXFP4 的 B 直进寄存器原型（−10…−15 %）。

**验证与工具**
- `gpu_tiles` 的屏障变异测试在 debug 下从 >25 分钟降到 191 秒：协议检查按 trace 缓存，并带变异用的迭代上限。
- 退出决策的路径上限从 64 提到 256：修复了 `fmha_fwd_hd128_fp8pt_v3_causal` 被推过上限的问题。
- 新守卫 `verify_reachable`：每个可达 kernel 严格验证 0 错误。

### 1.1 v9.2（2026-11-09 / 10，历史记录）

- **与厂商的比值（同进程同数据，比 = 厂商时间 / ours）**，详见 [`op-coverage-design.md`](op-coverage-design.md) §7.4–§7.8 的汇总表与 [`op-coverage-membound.md`](op-coverage-membound.md) 的总表：

  | 族 | 比值 |
  |---|---|
  | 出厂 MXFP8 FMHA（`port_gate`，本轮最后一次） | KV-split B1H8S2048 1.028×；`_vs128` B4H8S2048 0.955×、B1H8S8192 0.981×、每 CU 0.952×（8 布局中位数 0.943×）；`_t128` 0.900× |
  | GEMM 族（§7.4） | int8 per-token 1.01–1.63×；blockscale 0.91–1.45×（split-K 小 M 1.45×）；bf16 0.96×；MXFP4 0.86–0.90× |
  | attention prefill（§7.5） | fp8 per-tensor 1.07–1.74×（小网格走 KV split）；bf16 v3 hd128 0.89–1.07×，causal 0.80–1.23×；MLA prefill hd192 0.73–0.98× |
  | decode（§7.5） | paged decode16 0.67–1.24×（小 B 长 ctx 走 last-arriver `_c`）；MLA decode H16 约 0.63× |
  | fused MoE（§7.6） | fp8 per-token stage 1 0.93–1.14× |
  | attention 反向（§7.7） | 0.77–0.97× |
  | 访存受限（membound 总表） | 逐元素 / norm / 量化 / RoPE 达流带宽的 90–103 %；与 aiter 同轮：silu_mul 1.07–1.18×、fp8 per-token 1.08×、group128 1.10×、int8 1.04–1.26× |

- **正确性修复**（每条都有回归测试或硬件探针，§6.3）：GEMM LDS epilogue 复用 ring 前未等末轮预取 DMA 的竞态；VALU 写 SGPR 后 VMEM / VALU 读的冒险（`_vs128` 等 8 个 kernel 缺等待，硬件复现）；`moe_sorting` / `moe_scan` 的 LDS 竞态；描述符 SGPR 上限（`next_free_sgpr` 104 不能汇编）。
- **信任清单与写不相交证明**（[`trust-inventory.md`](trust-inventory.md)）：每个普通全局 / buffer 写要么被证明跨工作组不相交，要么计为"已查越界、未查竞争"。出厂 FMHA、GEMM、MLP 全部写已证明；结构性事实 `wg_bijection` / `wg_union` / `wg_winner` 与数据互异事实都计入账本；循环退出事实与持久循环剩余类把出厂证书从 26 降到 21；`amdc check-launch` 供非 Python 调用方检查主机契约；线程独占 LDS 列（`lds_owned`）与按外层迭代的周期检查去掉了 `moe_scatter_ordered` 的全部信任。
- **冒险规则审计**：逐条对照 LLVM `GCNHazardRecognizer` 的 gfx950 规则（[`trust-inventory.md`](trust-inventory.md) §5.1），只有 VALU 写 SGPR 两条缺失，已补上并有变异测试。
- **Python 算子层**（[`python-jit.md`](python-jit.md)）：`launch(kernel, …)` 可调用任意注册 kernel 并按已验证的契约检查；`ops_mem.py` / `ops_more.py` 覆盖 norm、量化、RoPE、KV cache、MoE、GEMM、prefill、MLA decode、采样、top-k、卷积、反向；预备启动（`_hip.Prepared`）把每次调用的主机时间从 23–57 µs 降到 5.5–9.3 µs。

---

## 2. 编译器流水线

```
DSL(Rust) ──► typed IR（区域树：Loop / If / Region / Stage + 编织区域标记；Op 109 种 / Stmt 57 种；irv1 文本可打印/解析）
            ──► IR 验证器 + gpu_verify（共享符号算术 amdgpu-sym + 协议）
            ──► lowering（profile 选项：scalarize / scalarize2 / waveuni / imm / edgepad …；指令按编织区域打标）──► 我们的 MIR（413 个 MOp，属性表 op_attrs）
            ──► 我们的调度（块内 list scheduler、LDS 读提升、slotfill、sched_group 相位；按区域策略 authored / free）
            ──► 我们的线性扫描 RA（两阶段 + LDS spill + S4 描述符类 + AGPR 类 + 寄存器规划：pin / separate / share / ring / 区域上限）
            ──► [opt-in] hzfill：post-RA 用独立指令填 hazard 等待窗口
            ──► post-RA MIR 导出（物理寄存器、已排序）
                 ├─ llvm-tail：llc -start-before=gcn-create-vopd … 只跑 tail pass
                 └─ own：我们自己的 hazard / wait / descriptor 实现
            ──► .s ──► ROCm clang 只做汇编 ──► .co ──► HIP 启动（主机 ABI 由 amdc gen-host 生成；Python JIT 见 python-jit.md）
```

| 环节 | 归属 | 说明 |
|---|---|---|
| DSL / IR / 验证 | 自研 | 类型化、无 unsafe、区域树（[`region-ir.md`](region-ir.md)）、循环变量顺序语义（[`region-ir.md`](region-ir.md)）；布局层在构建期证明覆盖/连续/操作数匹配 |
| Lowering / MIR | 自研 | 每个 MOp 的属性在一处（[`op-attributes.md`](op-attributes.md)）；一致值进 SGPR；死值消除含 SALU / `readfirstlane`（SCC 活性） |
| **寄存器分配** | **自研** | 线性扫描、两阶段（path-aware reaching-writer 检查）、LDS spill、S4 描述符类、AGPR 类与统一 512 预算 |
| **指令调度** | **自研** | 块内 list scheduler、LDS 读提升（`ldshoist`）、`slotfill`（GEMM v4）、依赖图 / 效果模型 / 等待模型（[`scheduling.md`](scheduling.md)） |
| hazard / wait / descriptor | llvm-tail 或 own | 两个后端指令流逐条一致（`tail_invariance` 守护） |

编译策略按 kernel profile 固定（`crates/amdgpu-codegen/src/lib.rs`）：FMHA `_vs128`/`_kvsplit` = `dce,cmpfold,scalarize,retarget2,imm,sgpr,zeroinit`，
`_t128` 另加 `addrfold,saddrlds,edgepad,salubr`；GEMM = `dce,cmpfold,lgkmn,zeroinit,scalarize,scalarize2,waveuni,edgepad,kabatch,imm` + M0 复用。
保留的开关：调度 `AMDC_SCHED`（0 / 1 / ld）、`AMDC_SCHED_OPTS`、`AMDC_SCHED_ADD`（在 kernel 预设上追加调度 token，含 `hzfill`）、`AMDC_SCHED_REGION`
（按区域策略的 A/B）、`AMDC_SCHED_VERBOSE`；分配器 `AMDC_RA_DEBUG_FILE`、`AMDC_RA_LDS_SPILL(_TARGET)`（强制 LDS spill，回归探针用）；后端 `AMDC_BACKEND`、`AMDC_KEEP_MIR`、`AMDC_DIALECT`、`AMDC_LLC_ARGS`、
`AMDC_BACKEND_LOCK` 与 LLVM 定位变量；验证 `AMDC_GPU_VERIFY`。

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`）。ROCm 10 不自带 `llc`，用
`tools/llvm_build_llc.sh` 从同一 commit 链接 `libLLVM.so` 得到；ROCm 7.0.2（`llvm20`）作为备选方言。
找不到 pin 的工具目录时 **llvm-tail 直接报错**，不静默降级。

**指令覆盖**：413 个 MOp 全部有 own emit；381 个在 `llvm20.tbl` **和** `llvm24.tbl` 有行，其余 32 个为 own-only（属性表 `Backend::OwnOnly`：LDS-DMA、DPP 等）。
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

区间分析的位级规则（`x ^ y`、`x & m` 的可能置位位）与盒子检验都有攻击测试（`crates/amdgpu-codegen/tests/gv_bits.rs`、`sym_branch.rs`）。

**信任预算（证书逐条列出）**：`--shipped`（v3 四个）21 / 100 / 0；`gemm_mxfp8_bs_v4` 1 / 19 / 0；`mlp_fp8_ws_fused` 0 / 9 / 0（逐项清单 [`trust-inventory.md`](trust-inventory.md)）。
每次运行打印 `certificate: N trusted assumes, M trusted contracts, K unproven`。

**跨工作组写**：普通全局 / buffer 写的跨工作组不相交由符号线性偏移 + 混合进制检查证明（出厂 kernel 全部写已证明；未证明的写计为 `global-write-untracked`，见 [`trust-inventory.md`](trust-inventory.md) §4）。

**未证明/范围外（诚实清单）**：原子与 last-arriver 以外的跨工作组读写依赖、i32 回绕（A3 假设）、从内存加载的指针/索引（以计数的数据互异事实代替）、safe-asm 块内部访存（计为 `asm_opaque`）、
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
| `amdgpu-target` | 目标描述（纯数据，[`target-design.md`](target-design.md)）：`gfx950` 的波宽、寄存器堆、LDS、计数器、hazard / 延迟 / 调度模型常量；逐指令表（按 `MOp`）在 `amdgpu-codegen/src/target/gfx950/` | – |
| `amdgpu-dsl` | **L4 IR**：区域树 + 编织区域标记、寄存器规划 `RegMark`、区域策略 `RegionDecl`、`CompilePolicy`、`FnBuilder`、表达式层、形状特化（`specialize.rs`）、unit 捕获容器（`units.rs`：`Unit` / `Units`）、irv1 文本打印/解析（`ir_text.rs` + 生成的 `ir_codec.rs`）与 α 结构相等（`print.rs`） | sym、safeasm |
| `amdgpu-codegen` | **L4 后端 + Weaver**：lowering、调度（含按区域策略）、RA + 寄存器规划检查与区域报告（`regplan.rs`）、hazard/wait、`hzfill`、emit、LLVM MIR 桥、`gpu_verify`/`gpu_protocol`/函数证明、MIR 打印、pass 管理器（`pass/`，`amdc passes`）（不依赖 kernels/tile） | sym、dsl、safeasm、target |
| `amdgpu-tile` | **L3 函数层**：tile 抽象（TiledCopy/TiledMma/partition、LdsPlan、softmax、frag、行 DMA 与 GEMM DMA 计划 `dma_plan.rs`、epilogue、规约原语、`TileIt`/`Roles`）、通用原语、handoff / 主机合同 | dsl、layout、isa、target |
| `amdgpu-stage` | **L2 编排层**：stage 库与组合（`stage.rs`）、stage 合同（`contract_stage.rs`）、pipeline 驱动（`pipeline.rs`：`Cadence`）、KV 切分 / 网格组件（`components.rs`）、**前端 Weaver**（`weave.rs`：`Weaver` / `rotate` / `check_order`；`weave_auto.rs`：按代价的 unit 顺序）、流水线模板（`templates/`：ring 对、split、grid、select、scan）、家族 body（`attention/`：`KvRing3`、`AttnBody3`、`Operands`；`gemm/`：寄存器暂存 body、`Elem` / `Scales`） | dsl、layout、isa、tile、target（测试：codegen） |
| `amdgpu-graph` | **L1 图层**：算子图（`opgraph/`：割、预算、JSON、骨架生成器）与 dispatch 规则（`dispatch.rs`，`amdc dispatch`）、组合规划（`compose.rs`，`amdc compose`，[`compose.md`](compose.md)） | layout、tile、stage |
| `amdgpu-kernels` | 全部 kernel：`fmha/`（v3 + FMHA I/O）、`gemm/`（配置、ABI、epilogue、骨架 slot）、`mlp/`、`norm/` …，每个都是"算子图（`graph.rs` + `*.graph.json`）→ 生成的骨架（`skeleton_*.rs`）→ slot 实现 + stage 实例"；`probes/`、`verifier_cases/`、`ir_proof.rs`、注册表 `registry.rs`、可达名清单 `reachable.rs` | dsl、layout、isa、tile、stage、graph、codegen、safeasm |
| `amdc` | CLI（list / emit / print-ir / ir-check / regplan / jit-info / gpu-verify / gen-host / opgraph / skeleton / dispatch / passes / md5-manifest / backend / simmodel / mops / mir-roundtrip；`--from-ir`、`--specialize`）+ 库 `amdc::hostgen` / `amdc::jit` | dsl、codegen、kernels、tile、graph |

分层与迁移记录：[`refactor-4layer-plan.md`](refactor-4layer-plan.md)、[`crate-split.md`](crate-split.md)（§6 状态）。

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

### 4.2 Kernel 清单（`amdc list` 65 个 + 按名注册的参数化族）

| Kernel | 状态 |
|---|---|
| `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}` | **出厂**（`crates/amdgpu-kernels/src/fmha/`：算子图 + 骨架 + stage 实例 `kv.stage` / `v_phase` / `m_phase` / `window`；导读 [`fmha-dsl-tour.md`](fmha-dsl-tour.md)） |
| `gemm_mxfp8_bs_v4` | MXFP8 blockscale GEMM（`crates/amdgpu-kernels/src/gemm/`）：厂商结构（寄存器暂存、AGPR 原地重载、slotfill）；stage 实例 `gemm.ab_stage` / `gemm.tile` |
| `mlp_fp8_ws_{fused,g1,g2}` | 融合 MLP 与两段对照（`crates/amdgpu-kernels/src/mlp/`；stage 实例 `gemm1_act` / `gemm2`） |
| `mlp_fp8_ws_g1q_{relu,silu}` | GEMM + 通用 epilogue 链（alpha、bias、激活、行 amax、逐行 fp8 + scale）；见 §4.8 |
| `rmsnorm_quant_fp8_n<N>`（N = 512…8192） | RMSNorm + 逐 token fp8 量化，4 个 stage（load → rms → quant → store）；见 §4.8 |
| GEMM 族、MoE、卷积（`gemm/`） | `gemm_a8w8_rowcol_fp8*`、`gemm_a16w16_bf16*`、`gemm_a8w8_blockscale*`、split-K + `gemm_splitk_reduce_s<S>`、`gemm_a4w4_mxfp4*`、`gemm_a16w4_mxfp4`；`gemm_moe{1,2}_*`（bf16 / fp8 / fp8bs / fp4 / bf16w4、`_oss`）+ `moe_sum`；`conv{2,3}d_bf16_*`。数字见 [`op-coverage-design.md`](op-coverage-design.md) §7.4 / §7.6 / §7.8 |
| attention（`fmha/`） | `fmha_fwd_hd<64/128/256>_bf16*` 与 `_v3[_causal][_ks]`、`pa_decode_bf16_*`、`pa_decode16_bf16_hd128_p<P>[_k64][_mtp][_f]`、`mla_decode_bf16_[hp_]p<P>`、`attn_combine_*`、`fmha_bwd_*`。§7.5 / §7.7 |
| 访存受限（`eltwise/ rows/ norm/ dequant/ rope/ cache/ moe/ topk/ sampling/`） | 逐元素与激活、rmsnorm / layernorm / softmax 及融合量化、per-token / group / mx 量化与反量化、RoPE（含 qk-norm 融合）、KV cache 写（flash / 分块 / fp8 / MLA）、moe_sorting / reduce / combine / fused_gate、top-k、采样、causal conv1d。[`op-coverage-membound.md`](op-coverage-membound.md) |
| `probe_*` | 硬件/ISA 探针（`crates/amdgpu-kernels/src/probes/`），各自配 `tools/*_probe.sh` / `tools/tileop_probe.sh` |
| `gv_attack_*`、`gv_audit_*`、`probe_lds_oob` | 验证器必须拒绝 / 必须列入信任清单的用例（`crates/amdgpu-kernels/src/verifier_cases/`） |

旧手写集、上一版 tile 库集（`dsl_*`）、`v3x`/`v3s` 实验变体、GEMM v3 与旧 MLP 变体已在 2026-11 整理中删除（结论保留在 [`archive/`](archive/README.md)）。

### 4.3 精度与确定性

- **门禁准则（用户决定 2026-11-04，"2+3"）**：ACC64 以**厂商对齐**判定——每个形状、每条路径同轮同输入下，我们的 mean_err ≤ 1.10× 厂商且 bad ≤ 厂商；
  旧的严格准则（bad = 0）仍以 INFO 打印，`ACC_STRICT=1` 可恢复（`tools/port_gate.sh`）。
- **参考算法（用户决定 2026-11-05，"accept vendor accuracy"）**：`_vs128` 的长序列副本（≥ 9 tile）使用厂商的参考配方（目标 s91 = 0.24799132、
  懒回滚阈值 s92 = 0.26135647、转换比例 0.25、无 LUT 校正、首 tile 播种；`OnlineSoftmax::set_vendor_ref`）；短序列副本保留精确配方 + LUT 校正；
  `_t128` 与 `_kvsplit`（出厂 split 走短副本）不受影响。递推与朴素求和的一致性：`crates/amdgpu-codegen/tests/functional_proofs.rs`。
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
（v7.0 门禁时的 asm md5 为 `763142f9` / `91b26e5d` / `e1fe92a4` / `24a95a51`；v8.1 出厂为 `622a6c0d` / `dc210951` / `41f151fc` / `24a95a51`（stage / weaver / 骨架重写前后逐字节相同；之后 `_vs128` / `_t128` 加入 XCD 顺序，见 [`fmha-wave-org.md`](fmha-wave-org.md)）；v9 为 `80f4a72c` / `5e3d12aa` / `c9dbf396` / `24a95a51`。）`_t128` 默认打开 `edgepad` 与 `salubr`（同轮 A/B 布局中位数 +3.3%）及 SADDR 形式的
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
| 物理 WAW（MFMA 在途结果寄存器被 RA 复用） | 非确定输出 | `pad_physical_war` 跟踪在途 MFMA 目标区间；`crates/amdgpu-codegen/tests/phys_waw.rs` |
| 两阶段 RA reaching-writer 误报 | 持久循环变量编译失败 | path-aware 检查；`crates/amdgpu-codegen/tests/ra_reaching.rs` |
| VMEM 宽存储数据 WAR | `global_store_dwordx4` 后数据寄存器被过早覆盖 → split 路径偶发损坏 | `pad_physical_war` 记录宽存储数据区间；STRESS 段回归 |
| LDS-DMA 操作数等待放置（AGPR 工作中发现） | 静默误编译 | 见 [`agpr.md`](agpr.md) "Wait placement" 一节 |

### 6.1 v9 新增的编译器正确性守卫（每条都有回归测试 / 探针）

| 守卫 | 防住的错误 | 位置 |
|---|---|---|
| 在途读的物理 WAW（`pad_inflight_waw`） | 一条路径上结果已死的 load 仍在途时，RA 把它的寄存器交给别的值，晚到的 load 覆盖之（MLA decode 的 if / else、Vs128 的 `_ks`、死 load） | post-RA，`crates/amdgpu-codegen/tests/inflight_waw.rs` |
| `ds_bpermute` / `ds_permute` / `ds_swizzle` 的 lgkm 等待 | 调度后的等待重算看不到这三类 LDS 返回值，结果未等就被读 | `op_attrs`（按 LDS 读计入 lgkmcnt），`probe_lane_p1` |
| `any_lane` 极性 | 对逐 lane 条件取反再折叠进分支极性，"没有 lane 为真"被当成"有 lane 为假" | `lower_condition`，`probe_anylane`（9 种极性） |
| 分支条件必须 wave 一致 | `begin_if` / `break_if` 的条件不一致时，wave 级分支的语义与作者意图不符；有意的"任一 lane"写 `any_lane(c)` | IR 验证器 `check_uniform_branches` |
| 启动寄存器预算 | 大工作组时每 SIMD 的 wave 共享 512 个寄存器，超预算的 kernel 不能启动 | `regalloc::check_workgroup_budget`，调度器按预算重试 |
| buffer 哨兵回绕（GV-BUF-WRAP） | 用 −16 这类哨兵屏蔽 lane 时，指令立即数把有效偏移绕回 0，被屏蔽的 lane 写进缓冲区 | `gpu_verify`，`gv_attack_buf_wrap` / `gv_ok_buf_sentinel` |
| cvt scale 只用指数（E8M0） | scalef32 转换的常量 scale 不是 2 的幂时结果静默错误 | IR 验证器 |
| fp4 / fp8 打包的目的选择 hazard | 写目的寄存器某一部分的 `v_cvt_scalef32_pk_*` 后紧跟读者需要 1 个等待态 | `hazards.rs`，`probe_fp4_chain` |
| 视图的偏移（view snapshots） | `frag4_lane(var_slice4(v, k), i)` 丢掉切片偏移；循环变量的视图在变量被写后仍按 SSA 语义读取快照 | lowering，`tileops_ir.rs` |
| LDS 存储复用（GV-TILE-ALIAS-RACE） | `LdsPlan::tile_over` / overlays 复用的字节对其它 wave 仍活着；并找到 GEMM `lds_epilogue` 末轮预取 DMA 未等就复用 ring 的真实竞态 | `gpu_protocol`，`gpu_tiles.rs` |
| MIR 导出的 `implicit-def dead` | 被当成读，导致循环内构造描述符时 SCC 的 live-in 错误、llc 中止 | `llvm_mir.rs` |

### 6.2 v9 的 lowering 窥孔（出厂 md5 因此改变，门禁全过）

- `select(cmp(x, y), x, y)`（i32 / u32 / f32，lt / le / gt / ge，两种顺序）→ 一条 `v_min` / `v_max`：去掉 v_cmp、vcc 与 cmp → cndmask 的等待态；`min(max(x, lo), hi)`（常量 lo ≤ hi）→ `v_med3`，min / max 链 → `min3` / `max3`。
- LDS-DMA 的 M0 写变成显式 MIR 指令：单次使用的基址加法（含连续的常量加）折进 `s_add_u32 m0, x, k`，连同单次使用的 SALU 源上提到前一个 M0 使用者之后；只有中间没有指令时才保留 `s_nop 0`。
- 指令数 / `s_nop` 个数 / nop 等待态（前 → 后）：`pa_decode16_bf16_hd128_p16` 963 / 55 / 143 → 883 / 23 / 79；`mla_decode_bf16_p1` 2286 / 121 / 427 → 2174 / 83 / 385；`_t128` 4911 / 141 / 318 → 4864 / 96 / 274；`_kvsplit` 10025 / 230 / 640 → 9987 / 193 / 602；`_vs128` 9365 / 214 / 604 → 9359 / 210 / 600；GEMM v4 1800 / 36 / 76 → 1788 / 24 / 64。
- 门禁：`port_gate` OVERALL PASS（B1H8S2048 742 TF/s 1.017×、B4H8S2048 1855 0.975×、B1H8S8192 2415 0.983×、每 CU `_vs128` 699 0.954×、`_t128` 1172 0.902×）；`gemm_gate` PASS 0.978×。


### 6.3 v9.2 的正确性修复（每条都有回归测试 / 探针）

| 修复 | 错误 | 回归 |
|---|---|---|
| GEMM LDS epilogue 等待末轮预取（`wait_vm(0)`） | `lds_epilogue` 复用 ring 存储时末轮的预取 DMA 仍在途（由 GV-TILE-ALIAS-RACE 发现） | `gpu_tiles.rs`（GV-TILE-ALIAS-RACE）；`gemm_gate` PASS |
| VALU 写 SGPR 的冒险（`hazards::pad_valu_sgpr`） | `v_readfirstlane` 等写 SGPR 后 VMEM 读需 5 个等待态、VALU 读需 2 个；`_vs128`、bf16 v3 / `_ks`、反向 dq / dkdv、`reshape_cache`、`topk_per_row` 等 8 个 kernel 缺等待。硬件探针 `probe_valu_sgpr`：不加等待时 2048 个值错 1984 个 | `tests/hazard_valu_sgpr.rs`（含去掉该 pass 后检查必须失败的变异测试）；`_vs128` md5 `cd403880 → dd009e44`，`port_gate` PASS |
| `moe_sorting` / `moe_scan` 的 LDS 竞态 | 跨 wave 的游标读前缺 barrier；`moe_scan` 的偏移写在发布 barrier 前未完成；原地前缀更新与读者无版本分隔 | LDS 改走声明的 tile（协议验证），barrier 变异测试；`lds_owned` 之后去掉了额外 barrier |
| 描述符 SGPR 上限 | `next_free_sgpr` 按 8 取整到 104 > 102，`fmha_fwd_hd128_bf16_v3_causal_ks` 无法汇编 | `dispatch_compile.rs` 检查描述符字段并对每个可达 kernel 汇编 + 链接 |
| M0 显式写的悬空使用 | 未定义的 M0 使用让一个寄存器从入口一直活着，bf16 v3 causal SGPR 不够 | `dispatch_compile.rs` |

---

## 7. 验证与工具

- **测试**：`cargo test` **121 个测试目标 / 519 项全绿**。编译器：`hazard_*`、`phys_waw`、`ra_reaching`、`pin*`、`gv_bits`、`sym_branch`、`mfma16x16x128`、
  `lds_b64`、`scalarize2`、`dpp`、`var_slice`、`var_semantics`、`regplan`、`sched_regions*`、`rust_safety` 等；kernel 相关（`crates/amdgpu-kernels/tests/`）：
  `tail_invariance`、`mir_invariants`、`register_accounting`、`asm_assembles`、`agpr`、`gpu_*`（语义/协议/攻击）、`gemm_mxfp8`、`tile_l*`、`region_ir`、
  `ir_text`（全部 kernel 的 irv1 往返）、`ir_print`、`hzfill`、`sched_kernels`；`amdc` 的 `jit`、`contract_host_truth`；布局层 `crates/amdgpu-layout/tests/`、
  tile 层 `crates/amdgpu-tile/tests/`；safe-asm `tests/ui` 编译失败对照用例。
- **门禁**：`tools/port_gate.sh`（FMHA：BUILD / VERIFY / ACC16 / ACC64 / DET / STRESS / PERF）、`tools/gemm_gate.sh`（GEMM）。
- **md5 守护**：`tools/md5_kernels.sh`（全部 `amdc list`）与 `tools/md5_compare.py`（对基线逐 kernel 比较，列出差异与已删除项）；`tools/ir_roundtrip_check.sh`（每个 kernel 经 irv1 文本往返后双后端 asm 与证书相同）。
- **工具**（`tools/`）：厂商对照（`vendor_disasm.sh`、`vendor_analyze.py`）、汇编对照（`asm_diff.py`、`asm_regmap.py`、`loop_stats.py`、`nop_attrib.py`）、
  GEMM（`gemm_ref.sh`、`prof_gemm.sh`、`att_*.py`）、A/B（`ab_fmha.sh`、`ab_gemm.sh`、`ab_layout.sh`；变体可为 `kernel@spec` 或 `x.ir`）、探针（`*_probe.sh`）、LLVM（`llvm_build_llc.sh`、
  `llvm_conformance.sh`、`llvm_feature_audit.py`）、文档链接检查 `check_doc_links.py`。
- **测量协议**：`.s`/`.co` 溯源、≥7 轮交错 A/B 或 8 布局中位数、同轮厂商、split 形状反复启动；GPU 隔离（每人固定 `HIP_VISIBLE_DEVICES`，只能是 GPU 0–3；所有 GPU 脚本 source `tools/gpu_guard.sh`，未设置或使用 4–7 即拒绝运行）。

---

## 8. 复现命令（容器内）

```bash
# 构建与测试
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                 # 期望 121 个测试目标 / 519 项全绿

# 发射 kernel 并汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --backend=own --out vs128.s   # md5 dd009e44…
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 严格证书
bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify --shipped --strict   # 21 / 100 / 0

# 一次具体启动的主机契约检查（非 Python 调用方）
bash tools/container/cargo.sh run -q --bin amdc -- check-launch gemm_mxfp8_bs_v4 --ka 0x18=4096,0x1c=4096,0x20=8192 --grid 16,16,1

# DSL IR 文本：打印 → 编辑 → 检查 → 编译（docs/ir-text.md）
bash tools/container/cargo.sh run -q --bin amdc -- print-ir fmha_fwd_hd128_mxfp8_v3_t128 --stage dsl --out t128.ir
bash tools/container/cargo.sh run -q --bin amdc -- ir-check t128.ir --verify
bash tools/container/cargo.sh run -q --bin amdc -- emit --from-ir t128.ir --out t128.s

# 寄存器规划报告（区域峰值 = 分配器真值）
bash tools/container/cargo.sh run -q --bin amdc -- regplan fmha_fwd_hd128_mxfp8_v3_vs128

# 主机侧生成
bash tools/container/cargo.sh run -q --bin amdc -- gen-host --shipped --out tests/gpu/generated

# 正确性 + 性能（同轮跑我们的、厂商的、CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh

# 门禁
HIP_VISIBLE_DEVICES=0 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3
HIP_VISIBLE_DEVICES=0 bash tools/gemm_gate.sh
```

---

## 9. 文档索引

| 文档 | 内容 |
|---|---|
| `current-status.md` | **本文件：权威当前快照** |
| [`design.md`](design.md) | Rust DSL 与自研编译器设计规格 |
| [`fmha-case-study.md`](fmha-case-study.md) | 以 FMHA 为例：kernel 怎样构建与优化（新作者 / agent 入口） |
| [`op-coverage-design.md`](op-coverage-design.md) / [`op-coverage-membound.md`](op-coverage-membound.md) / [`op-coverage-memtemplates.md`](op-coverage-memtemplates.md) / [`op-coverage-tileops.md`](op-coverage-tileops.md) | aiter 基础算子覆盖：迁移设计与批次状态 / 访存受限实测与覆盖表 / 访存模板 / tile-op 覆盖矩阵 |
| [`diagnostics.md`](diagnostics.md) | 结构化诊断（`--json`、`summary`、`srcmap`） |
| [`weave-capture.md`](weave-capture.md) | weave 捕获单元 |
| [`layout-tour.md`](layout-tour.md) | 布局层入门教程 |
| [`fmha-dsl-tour.md`](fmha-dsl-tour.md) | FMHA 逐段教程（DSL 特性 + Rust safe 保证） |
| [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) | 路线 B 设计与 §9 as-built |
| [`dsl-tile-abstractions.md`](dsl-tile-abstractions.md) | tile 层设计 |
| [`dsl-expression-layer.md`](dsl-expression-layer.md) | 运算符表达式层 |
| [`region-ir.md`](region-ir.md) / [`op-attributes.md`](op-attributes.md) / [`sym.md`](sym.md) | 区域树 IR / MOp 属性表 / 共享符号分析 |
| [`ir-text.md`](ir-text.md) / [`ir-printing.md`](ir-printing.md) | DSL IR 文本（irv1）打印-编辑-解析-编译 / 打印与 α 结构相等 |
| [`opgraph.md`](opgraph.md) / [`fusion-stage-library-design.md`](fusion-stage-library-design.md) | 算子图与骨架生成 / stage 组件库 |
| [`weaver-regs.md`](weaver-regs.md) / [`weaver-sched.md`](weaver-sched.md) | 编织区域的寄存器规划 / 按区域调度策略 |
| [`jit.md`](jit.md) / [`python-jit.md`](python-jit.md) | 形状特化与 `jit-info` / Python JIT 入口 |
| [`fmha-perf-patterns.md`](fmha-perf-patterns.md) / [`fmha-vs-vendor-asm-diff.md`](fmha-vs-vendor-asm-diff.md) | FMHA 性能承重模式 / 与厂商汇编的逐项对照 |
| [`fmha-wave-org.md`](fmha-wave-org.md) / [`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md) / [`lgkm-lookahead.md`](lgkm-lookahead.md) / [`fmha-4wave-design.md`](fmha-4wave-design.md) / [`fmha-weaver-parity.md`](fmha-weaver-parity.md) | FMHA 设计记录：wave 组织与 XCD 顺序 / 厂商 copy 流 / lgkm 等待策略 / 4-wave 原型（搁置）/ weaver 铺满场景实验（均为负） |
| [`functional-proofs.md`](functional-proofs.md) / [`stamps.md`](stamps.md) / [`mfma-ceiling.md`](mfma-ceiling.md) / [`mfma-dep-probe.md`](mfma-dep-probe.md) / [`isa-probes-smfmac-tr.md`](isa-probes-smfmac-tr.md) | 功能证明 / kernel 内时间戳 / MFMA 发射上限 / MFMA 依赖探针 / ISA one-hot 探针 |
| [`tirx-comparison.md`](tirx-comparison.md) / [`ir-review-tirx-sftensor.md`](ir-review-tirx-sftensor.md) / [`ir-text-roundtrip-design.md`](ir-text-roundtrip-design.md) | 与 TVM TIRx 的对照（当前状态）/ IR 梳理（TIRx + SF Tensor）/ IR 文本往返设计 |
| [`gpu-semantics-verifier.md`](gpu-semantics-verifier.md) / [`gpu-semantic-protocol.md`](gpu-semantic-protocol.md) / [`trust-inventory.md`](trust-inventory.md) | GPU 语义验证器 / 四轴协议模型 / 信任清单、写不相交证明与冒险规则审计 |
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
