# 机器指令属性表（`op_attrs`）

> 代码：`crates/amdgpu-compiler/src/op_attrs.rs`；测试：`op_attrs.rs` 内的 `tests`（枚举完整性、LLVM 覆盖表）。
> 设计依据：[`ir-review-tirx-sftensor.md`](ir-review-tirx-sftensor.md) 的 A 项（TIRx 的算子属性：`FInferType`、`TCallEffectKind`、
> `FLowerIntrinsic`、`FLegalize`——pass 查属性，不枚举算子名）。

## 1. 为什么

以前新增一条 MIR 指令（`MOp`）要改十来个地方：调度器代价表（`sched/model.rs`）、效应模型（`sched/effects.rs`）、`vcc` 定义/消费（`mir.rs`）、
值级与物理级冒险（`hazards.rs` 里 8 个分类函数 + 3 处内联清单）、AGPR 合法性（`agpr.rs`）、寄存器分配的 MFMA 检查（`regalloc.rs`）、
验证器的矩阵指令判定（`gpu_verify.rs`）、`M0` 来源（`sched/m0.rs`）、`ds_read` 提升（`sched/hoist.rs`）、相位调度（`sched/phase.rs`）、
半写检查（`pin_check.rs`）、DCE 清单（`lower.rs`）、后端选择（`lib.rs`）。每处都是一份手写的算子名清单，彼此稍有不同，漏改一处就是静默错误。

现在这些"类别"问题都由一张表回答：每个 `MOp` 一行，`attrs_of` 是没有通配分支的穷举 `match`——新算子不写这一行就编译不过。

## 2. 属性

| 属性 | 含义 | 读者 |
|---|---|---|
| `mnemonic` | 主汇编助记符（伪指令：展开后的第一条） | 文档、诊断 |
| `cost` | 流水线、发射占用、结果延迟、证据等级 | `sched::model::cost` → list scheduler、模拟器 |
| `fence` / `fence_only` | 栅栏类别（Nop / WaitVm / WaitLgkm / Barrier / SetPrio）；`fence_only` 表示没有寄存器效应 | `sched::effects` |
| `mem` | 访存：空间（LDS / Global）、读/写、基址（源操作数 k，或 MUBUF 的"未知基址"）、字节数；偏移取 `imm` | `sched::effects`、`is_lds_read`（hoist） |
| `smem` | 标量内存读（lgkmcnt，乱序返回） | `sched::effects` |
| `regfx` | 子寄存器读/写（片段 lane）、16 位半写（目的寄存器也被读）、`permlane32_swap`（两个源都被写） | `sched::effects`、AGPR 的读改写判定 |
| `vcc_read/vcc_write`、`defines_vcc/consumes_vcc` | 效应层的隐式 `vcc`（含指针加法伪指令的进位）；`mir` 层必须相邻的显式 `vcc` 生产者/消费者 | `sched::effects`、`mir::MOp::{defines,consumes}_vcc`、DCE、swp、cfg |
| `scc_read/scc_write` | 隐式 `scc` | `sched::effects`（SALU 分支 / `s_cselect` 不会被调度器拆开） |
| `m0_src` | LDS-DMA 形式中给出 `M0` 的源操作数 | `sched::m0` |
| `mfma` | 矩阵指令：结果等待态、同算子原地累加链、scaled 族、零累加器（无 C）、稀疏 `smfmac` | `hazards`、`agpr::is_mfma`、`regalloc` 操作数检查、`gpu_verify`、`phase`、`hoist` |
| `transcendental` | `v_exp/v_log/v_rcp` 的结果延迟 | `hazards` |
| `movable` | 冒险填充调度器的候选 | `hazards` |
| `valu_writer_delay` | VALU 结果被 MFMA / permlane / readfirstlane / DPP 读之前需要的等待态（2，进位加法 4） | `hazards` |
| `reads_valu_hazard` | 读取受上一行约束的操作数 | `hazards` |
| `wide_store_data` | 宽（> 8 字节）VMEM 存储的数据操作数下标（发射后 2 个等待态内不得被 VALU 改写） | `hazards::pad_physical_war` |
| `agpr_dst` / `agpr_srcs` | 目的、各源能否是 AGPR | `agpr`（AGPR 修复 pass） |
| `half` | 16 位半写的高/低半 | `pin_check` |
| `dce` | Keep / Load（结果未用即可删）/ Pure（纯 ALU） | `lower::dce_full` |
| `backend` | `Llvm`（两张 `.tbl` 都有行）、`OwnOnly`（使用它的 kernel 走 own 尾端）、`NoRow`（无 LLVM 行、也未强制 own：已知缺口） | `lib::uses_own_only_ops` |

## 3. 规则

- **类别问题查属性**：任何"这是不是一条 MFMA / 一次 LDS 读 / 一个 vcc 写者"的问题都通过 `op_attrs::attrs(op)` 回答，不写算子名清单。
- **身份判断可以保留**：针对某一条具体指令的判断（`op == MOp::SNop`、`lower.rs` 构造指令、`swp.rs` 识别循环计数模式里的 `v_add`/`v_cmp`）不是类别，留在原地。
- **新增 `MOp` 的步骤**：在 `mir.rs` 的枚举加变体 → 在 `op_attrs.rs` 加一行（编译器强制）→ 加入 `backend::ALL_MOPS`（`op_attrs::tests::every_variant_is_listed` 强制）→ `emit.rs` 写发射文本 →
  若 `backend` 为 `Llvm`，在 `llvm20.tbl`/`llvm24.tbl` 加行。

## 4. 迁移与验证

迁移是行为保持的重构：

- 迁移时（分支 `op-attrs`）曾用被替换实现的逐字副本对每个算子做等价性测试，并确认全部 kernel 的汇编 md5 不变；迁移完成后该临时测试已删除。
- 全部 kernel（`amdc list`，own 后端）的汇编 md5 在重构前后逐字节相同（`tools/md5_kernels.sh`）。

## 5. 范围说明

- `emit.rs` 仍是每个算子一段发射代码：发射文本依赖操作数形状（立即数 / SGPR / 元组、`off` 形式、`op_sel`），表里只记录主助记符。
- DSL 层（`amdgpu_dsl::Op` / `Stmt`）已有 `result_ty`、`operands` 这类按算子的属性函数；`gpu_verify` 的区间传递函数仍是按算子的 `match`，未纳入此表。
- `backend = NoRow` 的 6 个算子（`v_cmp_*_e64` 掩码比较 4 个、`v_cndmask_b32_e64`、`ds_read_b64`/`ds_write_b64`、`s_mov` 取低半、`s_memtime`）没有 LLVM 行，
  也没有被路由到 own 尾端：它们出现在 llvm-tail 编译的 kernel 里会报错（现状，未改变行为）。
