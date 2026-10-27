# 设计规格：Rust DSL 与自研 AMDGPU Compiler

## 1. 一句话模型

**Rust 源码表达值、所有权与协议；内部 IR 表达可优化的 SSA 与资源/异步关系；物理寄存器与指令顺序由独立 witness 约束；LLVM 只提供 MIR/target/ABI/编码发射基础设施。**

这三层各司其职，不能互相冒充：

| 层 | 负责 | 不负责 |
|---|---|---|
| Rust 类型系统 | 构图期非法描述（move-after-use、重复 writer、Pending 当 Ready、跨 graph 混用） | GPU 异步 hazard、alias、可见性 |
| 内部 IR verifier | 所有设备执行路径上的 SSA 支配、资源版本、effects、completion、epoch、收敛 | 真实堆内存寿命、外部框架 race |
| Runtime / adapter | allocation、module、ABI/context、stream ordering、in-flight 资源 | 编译期正确性 |

## 2. 编译管线

```text
Rust typed DSL（kernel 源码）
        │  值 / 所有权 / 协议，无物理寄存器编号
        ▼
Semantic SSA（数值图）
        │
        ▼
Resource SSA + Async/Event（存储版本、读取租约、完成事实）
        │
        ▼
内部优化：StateSROA / Fragmentation / PackCombine / ResourceLifetime /
          PipelineFormation / Fusion / Dataflow CSE-DCE
        │
        ▼
Recipe Expansion → MachineDAG（virtual tuple，无固定编号）
        │
        ▼
AllocationPlan + SchedulePlan（可由 witness 固定）
        │
        ▼
Physical MIR → post-RA 依赖/编码/mode 收尾 → 内部最终验证
        │
        ▼
LLVM AsmPrinter / MC → .o → lld → .co
```

### 2.1 为什么不是 LLVM 全后端

LLVM 的 AMDGPU RA 与指令调度不满足本项目需求，且 LLVM 源码不可修改。因此：

- 生产路径**不运行** LLVM 的 SGPR/WWM/VGPR allocator，也不运行 pre/post machine scheduler。
- 复用 LLVM 中能直接满足合同的现成能力：机器表示、target/operand/register 描述、必要合法化、ABI 准备、部分 post-RA 修复、AsmPrinter/MC、lld。
- 会破坏 hard pin、Strict 顺序或通信证明的 late pass 由内部 pass 接管，而不是修改 LLVM。

## 3. Rust 前端

### 3.1 类型角色

| 类型 | Rust 层行为 | GPU 语义 |
|---|---|---|
| `Tile` | 可共享 immutable SSA handle | 数值，无固定 physical register |
| `Accumulator` / `OnlineState` | affine，不 `Clone`；更新消费旧值返回新值 | loop-carried 数值与显式 update 链 |
| `Slot<Ready/Free/Writing>` | affine capability，writer 不可复制 | physical storage 生命周期 |
| `PendingReadBatch` | move-only，拥有 source read capability | 已 issue 未完成的 fragment reads |
| `ReadyFragment` | immutable value，不再借用 source slot | 指定 fragment 的读取已完成 |
| `ReadCompleteLease` | 不可伪造、move-only | 全部所需 source reads 结束，可 release |
| `Pred<Wave>` / `Pred<Workgroup>` | 不是 Rust `bool` | device 条件与 uniformity 范围 |

graph / region 使用 invariant brand，禁止跨 kernel、跨 region 混用。物理寄存器编号不出现在这些类型中。

### 3.2 异步读取合同

```rust
fn issue_read(slot: Slot<Ready>, plan: ReadPlan) -> PendingReadBatch;
fn complete_subset(batch: PendingReadBatch, group: FragmentGroup)
    -> (ReadyFragment, PendingReadBatch);
fn complete_remaining(batch: PendingReadBatch) -> ReadCompleteLease;
fn release_reads(done: ReadCompleteLease) -> Slot<Reusable>;
```

必须成立的语义：

- source slot 在 issue 时被 move 进 batch，读取完成前不可重写。
- `complete_subset` 只把对应 fragment 标为 ready；**partial `lgkmcnt(4)` 不等于整 tile ready**。
- `ReadyFragment` 不再借用 LDS lease：读取完成后 LDS 可回收，register compute 可继续。
- `complete_*` 只构建 completion 义务；最终必须由展开后的机器图与 wait/interlock 证据满足。
- 不采用「`ReadDone` 借用 lease，再 move lease 去 release」的自相矛盾签名。

### 3.3 控制流按 move 构造

```rust
b.loop_by_move(init, |body, state| { ... });
b.if_wave_by_move(pred, state, then_fn, else_fn);
b.if_wg_by_move(pred, state, then_fn, else_fn);
```

- body 是 `FnOnce`，只在构图期调用一次；运行时迭代由 SSA block argument / backedge 表示。
- 互斥分支由内部 builder 派生 block-arg capabilities，用户不需要 `Clone` affine state。
- join 检查每条路径的初始化、资源状态、receipts 与 versions 一致。
- reference kernel 在 half 中间有 drain exit，因此使用显式 `guard_or_drain` / `LoopOutcome`，不自动生成额外 loop header 或 end guard。

### 3.4 Role 与 paired barrier

- `Role<0,4>` / `Role<4,4>` 仅表示 wave 集合，不表示集合内 predicate 相同。
- overflow 条件是 `Uniform<Wave>`；drain / continuation 条件需要 `Uniform<Workgroup>` 或经验证的 phase-equivalent 协议。
- GroupProtocol 在外层声明 participants 与 phase 序列，把唯一 endpoint/ticket 分配给两个 role closure。
- rendezvous receipt 只证明参与与到达，不证明 payload 完成或 cache 可见。

## 4. 内部 IR

### 4.1 Layer A：Semantic SSA

```text
Value { id, dtype, shape, layout, uniformity, numerical_contract }
Op    { id, kind, inputs, outputs, source_span }
Region{ block_args, operations, yields, role, control_kind }
```

无 VGPR/SGPR 编号。核心 op 包含 tile view、native max/compare、packing、scaled dot、PV dot、row reduction、state projection、structured if/loop。

### 4.2 Layer B：Resource SSA / Async Graph

```text
Resource  { id, space, extent, layout, owner_domain }
Access    { resource, version/epoch, footprint, R/W, source_lease }
Event     { id, producer, completion_domain, fragment_set }
Transition{ issue, complete, publish, acquire, release, reclaim }
Phase     { participants, arrival_sites, memory_handoff }
```

按 resource / subrange 建 effects，不用单一全局 token 串行化。**phase receipt、payload completion、visibility 是三个独立事实。**

### 4.3 Layer C：MachineDAG

Recipe expansion 产生具体 opcode、virtual tuple、partial write、implicit registers、def/use 与 memory/event edge。每个节点带：

```text
MachineOpKey = (SemanticOpId, Role, Half, Fragment, RecipeStep)
```

### 4.4 Layer D：Physical IR

AllocationPlan / SchedulePlan 决定 physical tuple、顺序、wait placement 和 mode。此时才出现 `v52` 之类的编号，并保留 origin、resource/event 与 constraint sidecar。

### 4.5 数据结构为低延迟设计

arena + index vector、u32 ID、interned dtype/layout、紧凑 operand array 与 use-list。不使用每节点 `Arc`/`Mutex`。基础 IR 不可变，agent plan 以 delta 表达，各 pass 结果按 fingerprint 缓存。

## 5. 优化 Pass

| Pass | 内容与安全条件 |
|---|---|
| Type/Layout Propagation | 消除无指令 view/bitcast；不混淆 FP8、scale 与 opaque bits |
| StateSROA | 拆分 OnlineState 为独立 state 链，避免整 struct 串行化 |
| Fragmentation | 把 PV/QK 分解为实际 MMA fragment，使 partial readiness 可独立调度 |
| PackCombine | 纯 SSA low/high 合成带精确 partial-write 的 machine recipe |
| ResourceLifetime | 按实际 read-complete 缩短 LDS/source 占用 |
| PipelineFormation | issue/complete/compute/release 组成 stage graph，跨迭代用 epoch SSA |
| Fusion/Placement | 合并 producer/consumer 并选择 Reg/LDS/Global，保留多 reader 义务 |
| Dataflow CSE/DCE | 仅在同一 value/version/effect/数值合同下进行 |
| Bank/RA/Schedule | 基于 virtual value、physical alias 与 hardware reservation |
| FinalDependency/Protocol | 验证最终 wait、mode、NoSpill、pin 与传输生命周期 |

## 6. Pin、命名寄存器与 Strict

```text
Pin(value/fragment, bank, base, width, region, Must|Prefer, spill_policy)
```

- `Must`：最终物理 tuple 必须兑现，否则编译失败；不得静默降级或自动 spill。
- `Prefer`：可调整，但报告真实 placement 与原因。
- `named_reg`：无 GPU 内存地址的版本化固定槽，读写带 version 与 loop phi。
- 循环累加读取必须由 backedge 新值决定，不能折叠回循环前初始值。
- MFMA bank form 逐指令选择，一个 pinned AGPR 不得迫使无关 accumulator 迁移。
- Strict schedule 只在允许边界补 wait/NOP；内部 hazard 无法解决则报错。
- post-RA 的 copy、operand 交换、MSB 状态插入都会使旧证明失效，必须重新验证。

## 7. LLVM 复用边界

复用：MIR/target/ABI 基础设施、AsmPrinter、MC、lld、必要 late 修复。
不复用：LLVM RA、pre/post machine scheduler、会改变 placement/order/CFG 的优化、设备 LLVM IR 生成。
禁止：应用 LLVM 补丁、修改安装目录、依赖未公开内部符号。

LLVM 不认识的 opcode/modifier：优先降级为已支持的指令序列；确需原始编码时，必须在完成内部依赖与资源分析后，经现有 INLINEASM / MC raw-byte 通道承载，并显式登记 physical def/use/clobber 与 memory effects。**不能因为装进 opaque asm 就声称 LLVM 的 hazard 分析仍然完整。**

## 8. 跨卡通信（后续阶段）

基于 MORI CCO，首版单机多卡，LSA / SDMA / GDA(PSD, 同机 FULL QP) 三条路径。核心要求：

- RemoteView 是版本化 capability，不是裸指针。
- completion facts 分离：submitted / committed / source-safe / delivered / GPU-visible / consumer-complete。
- publish 必须汇合全部 writer；全部 reader Ack 后才能 reclaim。
- timeout / abort 不等于停止访问，资源 quarantine 直到真实 quiescence。
- 无前进性保证的阻塞等待环拒绝。

## 9. 待验证事项

以下在实现前必须补齐证据，当前不得当作契约：

- gfx1250 Expert2 的完整 `va_vdst` / `vm_vsrc` 硬件语义（公开 CDNA5 文档未完整覆盖）。
- LLVM 锁定版本中 late pass 对 operand / MSB / clause 的具体改写行为。
- MORI CCO 在本项目目标版本的 queue/packet/visibility 细节。
- 参考 kernel descriptor 中 kernarg_size=0 与 metadata 656 的差异来源。
