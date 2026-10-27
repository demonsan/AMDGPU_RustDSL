# 实施计划

## 0. 当前状态

设计基线。无可运行 compiler，无 GPU 验证，无性能数据。

## 1. 仓库结构（目标）

```text
crates/dsl/          Rust typed SSA 与 affine resource API
crates/ir/           Semantic / Resource / Async IR 与验证
crates/compiler/     内部优化、recipes、RA/schedule、witness、encoding
llvm-bridge/         项目自有 C++/C ABI 适配（不修改 LLVM）
examples/            无 physical 编号的 FMHA frontend
tests/               compile-fail、IR/protocol、ASM/byte differential
fixtures/            参考 metadata 与独立 witness
docs/                设计规格
```

## 2. 阶段

### M0 — 基础设施与 LLVM 边界

- 建立 Cargo workspace 与 out-of-tree pass driver 骨架。
- 确认锁定 LLVM 版本可供使用的公共 API；不修改 LLVM 源码、不要求补丁。
- 固化 target capability 数据来源与版本记录。

**出口条件**：空 kernel 能走通内部管线并生成可加载 `.co`；构建过程不触碰 LLVM 安装目录。

### M1 — 前端与 IR

- `Tile` / `Accumulator` / `OnlineState` / `Slot` / `PendingReadBatch` / `ReadCompleteLease`。
- structured `loop_by_move` / `if_wave_by_move` / `if_wg_by_move`。
- Semantic SSA + Resource SSA + Async Event IR。
- compile-fail 负例与 IR verifier。

**出口条件**：move-after-use、Pending 当 Ready、跨 graph、branch 复制 writer 均在 Rust 或 verifier 阶段被拒绝。

### M2 — 后端

- Recipe expansion、MachineDAG。
- 内部 tuple-aware RA、pressure-aware scheduling。
- hard pin / named_reg / Strict。
- post-RA dependency 与 encoding 验证。

**出口条件**：vector-add 与单 wave MFMA 在 gfx950 真机通过 CPU reference 对比。

### M3 — FMHA 1:1 复刻

- 无 physical 编号的 FMHA frontend，覆盖 fast / rollback 编码、PV/QK、half 中间 drain exit。
- ReferenceWitness 复现 1,692 条参考指令。
- 同一 frontend 切换 optimized plan。

**出口条件**：满足 `docs/fmha-reference.md` 第 7.3 节全部条目。

### M4 — JIT 与框架

- 类 FlyDSL 调用：signature → loaded hit → artifact hit → compile → load → launch。
- Rust API 与 Python 薄绑定，共享核心。
- SGLang / DeepInfer 薄调用示例。

**出口条件**：新 specialization 的编译耗时按阶段实测并报告；缓存命中路径无编译、无工具启动。

### M5 — 通信与 gfx1250

- MORI CCO 单机 LSA / SDMA / GDA(PSD, FULL QP) 三条路径。
- RemoteView、completion facts、credit 协议、quarantine。
- gfx1250 Expert2 依赖模型（需先补齐硬件语义证据）。

## 3. 测试矩阵

### 3.1 Rust compile-fail

跨 graph brand、move-after-use accumulator/slot、Pending 当 Ready、branch 复制 writer、忘记 rest batch、跨 region view。

### 3.2 IR verifier 负例

错误 epoch/event、遗漏 release、partial complete 冒充 whole ready、提前 reclaim、alias/OOB、divergent barrier、loop backedge 状态不一致、phase 冒充 payload 完成。

### 3.3 机器层负例

pin 冲突/对齐/越界、reserved clobber、资源少报、错误 counter、MFMA hazard、Strict 无法兑现、MSB 状态不一致。

### 3.4 端到端

ASM/MIR/byte differential、CPU reference、gfx950 真机重复运行、边界 shape 与 tail。

### 3.5 性能

分别统计：冷进程首次编译、disk hit、loaded hit、graph replay。10ms 目标是**新 specialization** 的 p95，包含内部 passes、bridge、最终验证与 MC/link；不以缓存命中代替。

## 4. 明确不做

- 任意 Rust/MIR 前端。
- 修改 LLVM 源码或依赖未公开符号。
- 通用设备调用、panic、动态分配。
- 首版 scratch spilling。
- 任意 raw asm 的 safe 认证。
- 完整框架移植。
- 未经测量的性能承诺。

## 5. 未验证项（阻塞相关阶段）

| 项 | 阻塞 |
|---|---|
| gfx1250 Expert2 `va_vdst` / `vm_vsrc` 完整语义 | M5 |
| LLVM 锁定版本 late pass 的 operand/MSB 改写细节 | M2 |
| MORI CCO 目标版本 queue/packet/visibility 细节 | M5 |
| 参考 descriptor kernarg_size=0 与 metadata 656 差异来源 | M3 |
| gfx950 真机可用窗口与授权 | M2 起 |
