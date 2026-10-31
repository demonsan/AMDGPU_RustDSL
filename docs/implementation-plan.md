# 实施计划：计划 vs 现状（第 75 轮）

> 元数据：Main `8fec359`，tag `v2.2-round75-final`。本文把最初的实施计划逐项对照到最终状态；每一项给出状态（**完成 / 采纳（改用别的实现）/ 关闭（负面）/ opt-in / 未做**）和记录它的文档。总览见 `docs/current-status.md`，终局报告见 `docs/final-report.md`。

## 0. 当前状态

不再是"设计基线"：Rust DSL、自研编译器（SSA → MIR → RA → 调度 → asm，默认 `llvm-tail` 后端）、safe-asm 序列库、MUBUF / unscaled MFMA 等指令族已在 gfx950（MI355X）真机运行；发布路径 `_auto` 在 B1H8S2048 上是同轮厂商的 1.25–1.31 倍，单 CU 0.84 倍，填满网格 0.84–0.85 倍；精度、确定性（含 1200/1200 重复启动）、`cargo test`（28 个目标、110 项）全部通过。剩余性能差距经五类有界实验关闭（`docs/final-report.md` §6）。

## 1. 仓库结构：目标 vs 实际

| 目标目录 | 实际 | 状态 |
|---|---|---|
| `crates/dsl/` | `crates/amdgpu-dsl`（类型化 SSA + 循环变量 + fragment 类型 + 所有指令族；FMHA kernel 家族在 `kernels.rs`） | 完成 |
| `crates/ir/`（Semantic / Resource / Async IR） | 合并进 `amdgpu-dsl` / `amdgpu-compiler` 的 SSA 与 MIR；完整 Resource / Async IR 未做 | 采纳（简化实现） |
| `crates/compiler/` | `crates/amdgpu-compiler`（lowering、tuple-aware RA、hazards、`sched/` 调度器与模型、emit、LLVM tail 后端表 `llvm20.tbl` / `llvm24.tbl`） | 完成 |
| `llvm-bridge/`（项目自有 C++/C ABI） | 未做 C++ bridge；改为 **MIR 导出 + 复用 LLVM 后 RA tail pass**（不修改 LLVM），见 `docs/llvm-tail-integration.md` | 采纳（不同实现） |
| —（计划外） | `crates/amdgpu-safeasm`：安全 Rust 手写汇编序列（`Ready` / `Pending` 证明、`pin()`、`pin_check`、部分 `lgkmcnt` 令牌），经 `Stmt::VarAsm[N]` 嵌入 | 完成（计划外增项），见 `docs/safe-asm-design.md` |
| `examples/` | FMHA frontend 即 `kernels.rs` 中的 `fmha_*` 构造器与 `tests/gpu/`、`tools/` 脚本 | 完成 |
| `tests/`、`fixtures/` | `cargo test` 28 个目标 110 项（GPU-free）；GPU 回归脚本 `tests/gpu/*.sh`、`tools/*probe*.sh`、`tools/gen9_gate.py`；独立 witness 夹具未做 | 完成（witness 未做） |
| `docs/` | 设计 / 状态 / 结果文档，见 §6 导航 | 完成 |

## 2. 阶段：M0–M5

| 项 | 原计划 | 状态 | 记录 |
|---|---|---|---|
| M0 Cargo workspace、out-of-tree pass driver | 建立 workspace；不修改 LLVM | **完成**：workspace + `amdc` 命令行；LLVM 未修改（tail 走外部 `llc` / 汇编器） | `README.md`、`docs/llvm-tail-integration.md` |
| M0 target capability 数据来源与版本记录 | 固化 | **完成**：厂商 `.co` 反汇编事实与 ISA 硬件验证在 `vendor-spec.md`；LLVM 行表按方言版本（20 / 24）固定 | `docs/vendor-spec.md`、`crates/amdgpu-compiler/src/backend/` |
| M0 出口：空 kernel 走通管线生成可加载 `.co` | 是 | **完成**（`tests/gpu/run.sh`、probe kernels） | `docs/status.md` 前 12 轮 |
| M1 `Tile` / `Accumulator` / `OnlineState` / `Slot` / `PendingReadBatch` / `ReadCompleteLease`；structured `loop_by_move` / `if_wave_by_move` | 仿射资源 API | **采纳（简化）**：类型化 SSA 值 + 循环携带变量（"读一次、写一次"契约）+ `begin_if`；资源与 lease 只在 safe-asm 序列里以 `Ready` / `Pending` / 独占借用的形式实现 | `docs/design.md`、`docs/safe-asm-design.md` |
| M1 Semantic SSA + Resource SSA + Async Event IR | 完整三层 IR | **未做**（只有 LDS 区域与"待完成加载"的 wait 跟踪） | `docs/final-report.md` §8 |
| M1 compile-fail 负例与 IR verifier | 全部拒绝 move-after-use 等 | **部分**：DSL / MIR 验证器、RA 后检查、safe-asm 的类型级拒绝（`tests/safeasm.rs`）；完整 compile-fail 套件未做 | `cargo test` |
| M2 Recipe expansion、MachineDAG | 展开 | **采纳（改用 lowering）**：`Op` / `Stmt` 直接 lowering 为 MIR | `crates/amdgpu-compiler/src/lower.rs` |
| M2 tuple-aware RA、pressure-aware scheduling | 内部实现 | **完成**：tuple-aware RA（紧凑 / 窄操作数回退 / 两阶段 / remat / LDS spill 重试）、列表调度器（`AMDC_SCHED`）；两阶段 RA 的 reaching-writer 缺陷已修 | `docs/scheduling.md`、`docs/status.md` |
| M2 hard pin / `named_reg` / Strict | 支持 | **完成**（safe-asm `pin()`；`pin_check` 对最终 MIR 复核补空档） | `docs/safe-asm-design.md` |
| M2 post-RA dependency 与 encoding 验证 | 支持 | **完成**，含三类真实 hazard 修复：物理 WAW、两阶段 RA reaching-writer、VMEM 宽存储数据 WAR | `docs/final-report.md` §5，`docs/vendor-spec.md` §7.11 |
| M2 出口：vector-add 与单 wave MFMA 在 gfx950 通过 CPU reference | 是 | **完成** | `docs/status.md` |
| M3 无 physical 编号的 FMHA frontend、fast / rollback、PV/QK、half drain | 1:1 复刻参考 | **采纳（功能与接口 1:1，结构不同）**：我们的 `_vs128` 等是另一种实现（128-key 单元、一个 rollback、两个 barrier），与厂商逐元素对拍 | `docs/fmha-reference.md`、`docs/final-report.md` §4 |
| M3 ReferenceWitness 复现 1,692 条参考指令 | witness | **未做**（改为反汇编对账：`vendor-spec.md`、`gen9-vendor-mirror-design.md`） | `docs/vendor-spec.md` |
| M3 同一 frontend 切换 optimized plan | 优化计划 | **完成（以 `FMHA_OPT` 令牌 + 按名 kernel 的形式）**：默认 `_auto`，所有实验为 opt-in | `docs/current-status.md` |
| M3 出口：`fmha-reference.md` §7.3 全部条目 | 满足 | **部分 / 关闭**：接口、数值、15 形状对拍满足；指令级 1:1 经 gen5/gen6/gen9/gen10 有界实验关闭（vendor-order M 窗口负） | `docs/gen9-vendor-mirror-design.md`、`docs/status.md` 第 68–75 轮 |
| M4 JIT 与框架（FlyDSL 式调用、Python 绑定、SGLang / DeepInfer） | 全部 | **未做**（范围外） | — |
| M5 通信（MORI LSA / SDMA / GDA）、gfx1250 Expert2 | 全部 | **未做**（范围外；gfx1250 语义证据未补齐） | — |

## 3. 测试矩阵

| 项 | 计划 | 状态 |
|---|---|---|
| 3.1 Rust compile-fail | 跨 graph brand 等 | **部分**（safe-asm 的借用 / 证明级拒绝已测；完整套件未做） |
| 3.2 IR verifier 负例 | 错误 epoch / 遗漏 release 等 | **部分**（DSL / MIR 验证器负例；Async Event IR 未实现） |
| 3.3 机器层负例 | pin 冲突、MFMA hazard、资源少报 | **完成**（`cargo test` 110 项，含 hazard 与 `pin_check` 回归） |
| 3.4 端到端 | ASM / byte differential、CPU reference、真机重复运行、边界 shape | **完成**：15 形状对拍（16 / 64 行）、确定性 5/5、重复启动压力 1200/1200、`tail_invariance`（两后端指令流一致） |
| 3.5 性能 | 冷启动、disk hit、loaded hit、graph replay 分别统计 | **部分**：编译延迟已实测（自研编译侧 5.7 ms / 中等 kernel，端到端经子进程 clang 不满足 ≤10 ms）；缓存路径未做（无 JIT） |

## 4. 明确不做（现仍成立）

任意 Rust/MIR 前端、修改 LLVM 源码、通用设备调用 / panic / 动态分配、任意 raw asm 的 safe 认证、完整框架移植、未经测量的性能承诺。首版 scratch spilling：**已被 LDS spill 重试取代**（`ldsspill`，仅在寄存器不够时使用）。

## 5. 未验证项（原阻塞表）

| 项 | 原阻塞 | 现状 |
|---|---|---|
| gfx1250 Expert2 `va_vdst` / `vm_vsrc` 完整语义 | M5 | 未做（范围外） |
| LLVM 锁定版本 late pass 的 operand / MSB 改写 | M2 | 已解决为"复用 LLVM tail + 指令流不变量测试" |
| MORI CCO 目标版本 queue / packet / visibility | M5 | 未做 |
| 参考 descriptor `kernarg_size=0` 与 metadata 656 的差异 | M3 | 以 656 字节 kernarg 对拍通过（实测） |
| gfx950 真机可用窗口与授权 | M2 起 | 解决（MI355X 节点，GPU 0 / 5） |

## 6. 文档导航

- 总览与路线清单：`docs/current-status.md`；终局报告：`docs/final-report.md`；逐轮记录：`docs/status.md`。
- 厂商事实与硬件规则：`docs/vendor-spec.md`、`docs/fmha-reference.md`；设计：`docs/design.md`、`docs/safe-asm-design.md`、`docs/scheduling.md`、`docs/llvm-tail-*.md`。
- 已关闭 / 采纳的 kernel 设计文档：`kernel-gen2-design`、`gen3-design`、`gen4-tiling-design`（采纳）、`gen5-window-spec`、`gen6-rebalance` / `gen6-results`、`gen9-vendor-mirror-design`、`unscaled-pv-design`、`v-window-cut-plan`、`occupancy`、`kvsplit-design`（采纳）/ `kvsplit-results`。
