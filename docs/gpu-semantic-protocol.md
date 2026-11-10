# GPU 语义协议模型（v4）：到达 / 完成 / 可见 / 代际 四件事分开建模

> 实现：`crates/amdgpu-codegen/src/gpu_protocol.rs`（由原 tile 模拟器重构而来的通用协议引擎）；DSL 标记：`lds_tile/lds_ring/lds_private/lds_alias`、`tile_produce/tile_consume/tile_end`（LDS 实例）与 `buf_region/buf_produce/buf_consume/buf_last_arriver/buf_trust`（全局内存实例）；测试：`tests/gpu_tiles.rs`、`tests/gpu_protocol.rs`。
> 本文是**形式模型**。`docs/gpu-semantics-verifier.md` 讲边界/地址/寄存器；本文讲“多个生产者、多个消费者之间的同步协议”到底在证明什么。风格不变：**声明式标记，没有新的类型/struct API**。

## 1. 为什么要把四件事分开

GPU 上“数据好了”被混成一句话，实际是四件互相独立的事。混在一起就会出现“等待做过了所以可见”“barrier 过了所以写完了”这类错误：

| 轴 | 事件 | 谁做 | 保证什么 | **不**保证什么 |
|---|---|---|---|---|
| **到达 Arrival** | `s_barrier` | 工作组内所有 wave | 所有 wave 都到过这个点（相位 +1）；只是**到达** | 不保证任何写已完成，也不保证可见 |
| **完成 Completion** | `wait_vm(0)`、`wait_lds(0)`、`fence_release`（含 vm 等待）；`wait_vm(n)` 对 LDS-direct DMA：其后至少有 n 条更新的 DMA 时才算完成 | **每个 wave 各自**（队列计数；vm 按序，标量 lgkm 可乱序，所以部分 lgkm 计数不算完成） | 本 wave 先前发出的访存已完成 | 不保证别的 wave 的写、不保证其他 CU/XCD 可见 |
| **可见 Visibility** | LDS：barrier（写必须先完成）；全局：`sc1` 写穿 / `fence_release`（`buffer_wbl2`）→ 到达 → `fence_acquire`（`buffer_inv`）/ `sc1` 读 | 生产者 release，消费者 acquire | 其他 wave（或其他 block）能读到这份数据 | 不保证是**哪一代**数据 |
| **代际 Generation** | 版本/槽位代数（循环计数器的仿射/周期函数） | 生产者与消费者共同遵守 | 读到的是期望的那一代，且下一代不会在读的过程中落到同一位置 | — |

## 2. 参与者

* **多生产者**：一个生产角色的**多个 wave**（协作填充同一代的不同切片），或多个分支（then/else 角色、运行时 if 的两种结局），或多个 block（last-arriver 模式）。
* **多消费者**：**每个角色的每个 wave 独立消费**——共享区里**没有“自产自销”**：一个角色在模拟里代表它的所有 wave，自己写完并等待只说明“我这个 wave 完成了”，其他 wave 的那一份可能还没写。只有**私有区**（`lds_private`，整个 wave 写自己的切片再自己读）可以自产自销。
* 因此每个**读事件**对**每个同代生产者**都有义务，每个**写事件**对**每个读窗口**都有义务：这是**双向义务**，由编译器静态 discharge。

## 3. 四个义务

设一个共享区（tile / 环的槽 / 全局 region）的某一代 g，生产者集合 P(g)，消费者集合 C(g)。

| 义务 | 内容 | 违反时 |
|---|---|---|
| **O1 COMPLETE-BEFORE-PUBLISH** | P(g) 中**每个**写在发布事件之前都已**完成**：DMA/全局写要 `wait_vm(0)`（或 `fence_release`），DS 写要 `wait_lds(0)`；发布事件是完成之后的第一个 barrier（全局区还要求完成与 barrier 之间有 `fence_release`） | `GV-PROTO-NO-COMPLETION`（写从未完成）/ `GV-PROTO-NOT-VISIBLE`（全局写未 release） |
| **O2 PUBLISH-BEFORE-READ** | C(g) 的**每个读**严格晚于 P(g) **所有**生产者的发布；可见性：LDS = barrier；全局 = release（生产者）+ barrier（到达）+ acquire（消费者，`inv` 或配 `sc1` 读） | `GV-PROTO-UNPUBLISHED`（版本未发布/协作部分未发布）/ `GV-PROTO-NOT-VISIBLE`（缺 acquire） |
| **O3 READ-BEFORE-OVERWRITE** | 下一代 g' 对同一位置的写，只能在 C(g) **所有**读都完成（`wait_*`）并被 barrier 隔开之后才发出（写窗口 `[发出相位, 发布相位)` 不得与读窗口 `[读相位, 完成相位]` 相交；不同 wave 并发写不同代也是竞争；读在下一个 barrier 前必须完成） | `GV-PROTO-RACE`；读未完成就过 barrier：`GV-PROTO-NO-COMPLETION` |
| **O4 GENERATION-CONSISTENCY** | 所有参与者对“第 i 轮是哪一代、哪个槽”的映射一致：槽/代是循环计数器的**仿射/周期**函数，周期上界由表达式锥推出并**在模拟序列上验证**；读到的版本必须恰为期望 | `GV-PROTO-STALE-GEN`；超出周期闸门：`GV-PROTO-UNROLL-LIMIT` |

另有 `GV-PROTO-ARRIVAL-MISMATCH`（到达轴：工作项相关的 if/else 两侧 barrier 数不等，或 last-arriver 没有 ticket / 读没有被 ticket 保护）与 `GV-PROTO-TRUST`（被信任项，Info，进入账本）。

## 4. 错误分类与旧名（别名）

`Finding.code` 是稳定的主名；`Finding.alias` 是协议名（报告里显示为 `[主名|别名]`）。**全局内存实例没有旧名，主名就是 GV-PROTO-\***。

| 旧名（LDS 实例，主名） | 协议名（别名） | 条件 |
|---|---|---|
| `GV-TILE-STALE` | `GV-PROTO-UNPUBLISHED` | 读时没有已发布的版本 / 协作部分未发布（缺 barrier） |
| `GV-TILE-STALE` | `GV-PROTO-NO-COMPLETION` | 写从未完成（无 `wait_*` / release） |
| `GV-TILE-STALE` | `GV-PROTO-NOT-VISIBLE` | 全局：未 release / 读发生在 acquire 之前 |
| `GV-TILE-STALE` | `GV-PROTO-STALE-GEN` | 最新已发布版本 ≠ 期望版本 |
| `GV-TILE-RACE` | `GV-PROTO-RACE` | 读窗口与别代写窗口相交；不同 wave 并发写不同代 |
| `GV-TILE-PENDING` | `GV-PROTO-NO-COMPLETION` | 读跨过 barrier 且之后没有收尾的 `wait_lds(0)`（Warn）；有收尾等待时，读的生命周期 `[发出, 收尾]` 参与竞争检查 |
| `GV-TILE-INDEX` | `GV-PROTO-STALE-GEN` | 槽/代不是循环计数器的编译期函数 |
| `GV-BAR` | `GV-PROTO-ARRIVAL-MISMATCH` | 到达计数不一致 |
| `GV-TILE-UNROLL-LIMIT`、`GV-TILE-UNSUPPORTED` | `GV-PROTO-UNROLL-LIMIT` | 周期/展开闸门；循环内运行时 if 等无法建模 |
| `GV-ASSUME`（TRUSTED） | `GV-PROTO-TRUST` | 被信任项 |

其余旧码（`GV-TILE-OVERLAP/ESCAPE/KIND/PRIVATE-ESCAPE/UNTAGGED/EMPTY`）是地址/声明层面的检查，没有对应的 O 义务，保持原名。

## 5. Discharge 算法

四条腿，全部静态：

1. **角色路径枚举**：工作项相关、含协议事件的 `if`（then = 角色 0，else = 角色 1）；波一致的运行时 `if` 两个结局都探索（≤64 条路径，工作组一致条件在两个角色取同一结局）；循环内运行时 if 含协议事件 → `UNROLL-LIMIT`（不静默跳过）。
2. **相位代数**：相位 = 动态 barrier 序号。每个事件带（发出相位、完成相位、发布相位）。发布相位 = 完成之后第一个 barrier 的相位 + 1（全局区还要求其间有 `fence_release`）。读窗口 `[读相位, 完成相位]`，写窗口 `[发出相位, 发布相位)`，所有 O1–O3 判定都是这些区间上的比较。
3. **代际周期性**：槽/代表达式必须是计数器的仿射/周期函数（常量、`+ - * << & ^` by 常量；计数器 `+const`/`^const` 步进）；周期上界 `2^M`（`& mask` 位宽、`^ 常量` 位宽、`>> k`、翻转计数器）；模拟 `max(10, 2×槽数+4, 2×2^M+2)` 轮（≤70）并覆盖每个退出奇偶；声明的周期在最长模拟轨迹上**逐标记语句验证**（槽每 p 轮重复、版本每 p 轮前进常量）。
4. **循环不动点**：循环变量区间不动点（环形计数器、`xor`、选择）支撑地址证明；`assume` 必须归纳，否则 Error。

## 6. 全局内存实例（v1：工作组内）

* 声明：`b.buf_region(name, arg, base, size)`（`arg` 的字节区间 `[base, base+size)`，用与 `arg_tensor` 同一套 `Ext` 代数）。标记：`b.buf_produce(name, gen)` / `b.buf_consume(name, gen)` / `b.tile_end()`。区间内的访问必须被证明落在 region 内（`GV-TILE-ESCAPE`）。
* **生产者**：标记作用域内的全局写；**完成** = `wait_vm(0)` 或 `fence_release`；**发布** = 完成后第一个 barrier，且完成与 barrier 之间必须有 `fence_release`（否则写没有被推到全局可见层：`NOT-VISIBLE`）。
* **消费者**：读必须晚于发布；并且在发布 barrier 之后、读之前有 `fence_acquire`（`inv`），或 acquire 的 lite 形式 + 所有读都是 `sc1` 的一致性加载（否则 `NOT-VISIBLE`）。
* **O3**：第二代写在消费者读窗口内 → `RACE`；**O4**：代号错 → `STALE-GEN`。
* 范围：**工作组内**（同一个 block 既生产又消费）；跨 kernel（两次 launch 的 split）不在范围内。

## 7. 显式模式：融合 KV-split combine（last-arriver）

声明 `b.buf_last_arriver(region, counter_arg, splits)`，并用 `buf_produce(region, split_id)` 标记每个 block 写自己切片的存储、用 `buf_consume(region, …)` 标记最后到达者读所有切片的加载。验证器检查（`verify_last_arrivers`）：

| 义务 | 检查 | 失败码 |
|---|---|---|
| O1 | 最后一个 produce 存储与 ticket 原子之间有 `fence_release`；若是 lite 释放（无 L2 写回），所有 produce 存储必须是写穿（`store_*_wt`） | `NO-COMPLETION` / `NOT-VISIBLE` |
| 到达 | 在最后存储之后确有对 `counter_arg` 的 ticket 原子；所有消费读都在由 ticket 结果守卫的分支里 | `ARRIVAL-MISMATCH` |
| O2 | ticket 之后、首个消费读之前有 `fence_acquire`；lite 形式要求所有读是 `sc1` 加载 | `NOT-VISIBLE` |
| O4 | produce 标记的代号的表达式锥里含 `WorkgroupId`（split id 来自 block id）；最后到达者读多于一个切片（循环或多次加载） | `STALE-GEN` |

出厂的 `probe_last_arriver`、`_lite`、`_plain_fenced` 通过；`_nofence`、`_plain`（本来就是负对照）被拒绝。此类区域不进入工作组内模拟器（它们的生产者在别的 block 里）。

## 8. 信任账本（trust ledger）

每一个逃生口都被**计数并打印**，认证 = “0 个未证明 + 评审接受这份账本”：

| 逃生口 | 计数 | 报告 |
|---|---|---|
| `assume_trusted`（循环计数器范围/未被蕴含的值 assume） | `assume_trusted` | `GV-ASSUME … TRUSTED` |
| `tile_trust_periodic` | `tile_trust_periodic` | 同上 |
| `buf_trust(region, reason)`（该 region 的协议不被检查） | `region_trust` | 同上 |
| 主机合同（`arg_tensor*`、`ka_range`、`ka_is`、`ka_one_of`、`grid_is`、`sym_range`） | `trusted contracts` | — |

`amdc gpu-verify` 每个 kernel 后打印：`certificate: N trusted assumes (a assume_trusted, p tile_trust_periodic, r region_trust), M trusted contracts, K unproven`，最后给出合计行。API：`Report::ledger()`。

## 9. 范围与诚实边界

* 全局区只证明**工作组内**协议和**显式 last-arriver 模式**；跨 kernel 的两次 launch split 不在范围内；其它跨 block 协议需要 `buf_trust`（计入账本）。
* last-arriver 的“最后到达者读遍所有 splits”只检查“多于一个切片”，不检查切片总数等于 `splits`（这是 `splits` 合同 + 数据相关循环的事，计入主机合同）。
* 角色只有两个；所有循环迭代数相同；LDS 同 tile 内的字节级部分重叠不区分；safe-asm 内部访存不检查；i32 不回绕（A3）。
* 软件/编译器与硬件之间的缺口（例如我们对 `buffer_wbl2`/`buffer_inv` 语义的建模）由 `tests/gpu/` 的硬件 probe（`probe_last_arriver*`、`probe_la8*`）实测支撑，不由验证器证明。