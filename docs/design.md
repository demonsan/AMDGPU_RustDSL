# 设计规格：Rust DSL 与自研 AMDGPU Compiler（最终架构）

> 本文是**设计规格**（做什么、为什么、边界在哪），不是状态记录。当前数字与结论见 [`current-status.md`](current-status.md)，逐轮证据（第 1–75 轮）归档在 [`archive/status.md`](archive/status.md)。路线 B（布局代数层、GEMM、AGPR、主机侧生成）之后的库结构见 §3.1。
> 2026-10-27 的初版描述的是一个更大的目标（Resource SSA / Async Graph、MachineDAG、跨卡通信）；最终落地的是下面的架构。初版中**未采纳**的部分与原因集中在 §13。

## 1. 一句话模型

**Rust 类型化 DSL 表达值、所有权与协议；内部 typed IR 经验证后降到我们自己的 MIR；寄存器分配与指令调度始终是我们自己的；LLVM 只作为 post-RA tail（hazard / wait / descriptor / 编码）；`own` 后端是不依赖 LLVM 的完整实现，同时充当 tail 的 oracle。**

| 层 | 负责 | 不负责 |
|---|---|---|
| Rust 类型系统（DSL、safe-asm） | 构图期非法描述（move-after-use、Pending 当 Ready、写只读端口、跨 token 混用）；`#![forbid(unsafe_code)]` | GPU 异步 hazard 的最终判定、数值正确性 |
| IR verifier（`amdgpu-dsl`） | 类型/uniformity/循环变量单赋值（`var_lint`）/立即数范围/描述符类型 | 物理寄存器、指令顺序 |
| MIR 层（`amdgpu-compiler`） | 效应模型、调度、RA、hazard/wait 插入、`pin_check`、`mir::verify` | 指令选择之外的 LLVM 级优化 |
| 后端（`llvm-tail` / `own`） | post-RA 的 hazard、waitcnt、descriptor、编码；两者指令流逐条一致（`tail_invariance`） | 寄存器与顺序决策 |
| Runtime / 测试 harness | 模块加载、ABI、同轮厂商对照、确定性压力 | 编译期正确性 |

## 2. 编译管线

```text
DSL(Rust) ─► typed IR（Op 84 种 / Stmt 47 种）─► IR 验证器 + gpu_verify
          ─► lowering（uniform 分析、remat、zeroinit、cmpfold、retarget…）─► 我们的 MIR
          ─► 调度栈（list / hoist / swp / taildup；等待重算）
          ─► pin_check（含 pinned 段的 hazard / wait / 半写校验）
          ─► hazard 填充 / 值级 hazard pass（fix_value_hazards）
          ─► 寄存器分配（线性扫描 → 两阶段 + path-aware 检查；LDS spill；S4 描述符类；AGPR 类 A1–A16）
          ─► 物理级 hazard pass（pad_physical_war）
          ─► post-RA MIR（物理寄存器、已排序）
               ├─ llvm-tail（默认）：导出 MIR → llc -start-before=gcn-create-vopd（只跑 tail pass）
               └─ own：自己的 hazard / wait / descriptor 实现与 emit.rs 文本发射
          ─► .s ─► ROCm clang 仅做汇编 ─► .co ─► HIP 启动
```

### 2.1 所有权分界：我们的 vs LLVM tail 的

| 事项 | 归属 |
|---|---|
| 寄存器分配（含 hole-aware 打包、LDS spill、remat） | **我们** |
| 指令顺序、跨基本块调度、软件流水、尾部复制、ds_read 提升 | **我们** |
| wait 计数的语义来源（scoreboard 模型、partial `lgkmcnt/vmcnt`） | 我们（`waits.rs`）；tail 阶段由 LLVM（`SIInsertWaitcnts`）或 own 实现重新插入 |
| 值/物理级 hazard 的**判定** | 我们的 pass 先行；LLVM tail / own 各自补齐并必须与之一致 |
| descriptor（`.amdhsa_*`、kernarg 元数据）、汇编文本、编码 | LLVM tail 或 own |
| LLVM 的 RA、MachineScheduler、IGLP | **不使用**（它们按基本块工作；LLVM 源码不可修改） |

**后端选择**：`amdc emit --backend auto|llvm-tail|own`（`AMDC_BACKEND`）。`auto` 对含 own-only 操作（DMA / LDS 直达 / `buffer_load … lds`）的 kernel 选 own，其余选 llvm-tail。找不到 pin 的 LLVM 工具时 llvm-tail **直接报错**，不静默降级为 own。

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`，默认）；ROCm 7.0.2（`llvm20`）作为第二方言（`AMDC_DIALECT`，由 `llc --version` 探测）。ROCm 10.x 不带 `llc`，`tools/llvm_build_llc.sh` 用同一 commit 的 `llc.cpp` 对已安装头文件编译并链接已安装的 `libLLVM.so`（2 个 TU，数秒）。

## 3. DSL 与 typed IR

- **入口**：`FnBuilder`（约 160 个公开方法）→ `Function`（kernarg ABI；与厂商 kernel 逐字节兼容的 656 字节布局由 kernel 层负责）。
- **值模型**：`ValueId`（单次定义）、`VarId`（循环变量）、`Opnd`、`DType`（`F16/F32/I32/U32/I1/Ptr/Frag2/4/8/16/Rsrc`）。`Rsrc` 只存在于 SGPR 四元组。
- **控制流**：`begin_loop … end_loop` 是 do-while，循环变量以 `begin_loop(&[(DType, init)])` 声明、`set_var` 更新（`var_lint` 检查单赋值与作用域）；`begin_if` 是 **wave 级**分支（对 `vcc` 的 any 判断，**没有 exec 掩码**），因此体内整 wave 执行。
- **Uniformity**：lowering 的 `uniform_analysis` 判定波内一致的整数值与循环变量，落入 SGPR（`scalarize`；`scalarize2` 另把一致的 `min`、一致比较上的 `select`（`s_cmp` + `s_cselect`）与 `readfirstlane` 结果留在 SGPR；`waveuni` 把 `tid >> 6` 视为一致），其余走 VGPR。`s_operand` 对非一致值报错而不是静默降级。
- **指令族**：MFMA（scaled 32x32x64 f8f6f4 的累加 / zero-C / **C 操作数** 三形式；scaled **16x16x128** f8f6f4（op_sel 选 E8M0 字节）；AGPR（acc_cd）形式；**unscaled** 同三形式；f16/bf16；`v_smfmac`）、VALU（int/fp、`pk_add/mul/sub`、`perm`、`permlane32_swap`、`pknorm`、`scalef32`/`cvt_pk_fp8` 半写、`exp/rcp/log`、`max3`、`readfirstlane`）、内存（见 §9）、同步（`barrier`、`wait_vm/lds`、`fence_release/acquire` 与 coherent/write-through 访问）。
- **实验面**：`FMHA_OPT` token 做 A/B（`FMHA_OPT_SCOPE` 把 token 限定到别名 kernel 做同轮对照）；每个优化与负结果都可复现。

### 3.1 库分层（路线 B 之后）

| crate | 层 |
|---|---|
| `amdgpu-layout` | 布局代数：`Ix`、`IntTuple`、`Layout`、GF(2) `Swizzle`、TV 布局、`IssuePlan` / `layout_to_offsets`、构建期证明（零依赖） |
| `amdgpu-isa` | gfx950 copy / MMA atom 表（宽度、立即数域、计数器、LDS-direct 形式；A/B/C 的 TV 布局与 provenance） |
| `amdgpu-tile` | `TiledCopy` / `TiledMma` / `partition_{lds,dma,buffer_dma,global}`、布局导出的主机合同、出厂 v3 FMHA |
| `amdgpu-kernels` | MXFP8 blockscale GEMM |
| `amdgpu-dsl` | IR、`FnBuilder`、表达式层、tile 库 L1–L6（上一版出厂集仍可构建）、旧手写 kernel |

布局层只决定**地址与每次操作发射哪些指令**；指令顺序仍由 kernel 作者与调度器决定，安全仍由 `gpu_verify` 证明。入门：[`layout-tour.md`](layout-tour.md)；实施记录：[`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) §9。主机侧：`amdc gen-host` 从 kernel 合同生成 C++ 头与 JSON（[`host-gen.md`](host-gen.md)）。
- **IR 验证器**在 lowering 之前运行：类型、立即数位宽（如 MUBUF `imm` 12 位、DS offset 对齐）、描述符操作数必须是 `Rsrc`、soffset 必须是一致整数。

## 4. MIR 与效应模型

- `MFunction { blocks, vregs: Vec<RClass>, alias, loops, … }`；`RClass` = `V1/V2/V4/V8/V16`（VGPR 元组）、`A1/A2/A4/A8/A16`（AGPR 元组，见 [`agpr.md`](agpr.md)）、`S1/S2/S4`（SGPR；`S4` 为 4 对齐的 MUBUF 描述符四元组）。
- `MInst { op: MOp, dst, srcs: Vec<Src>, imm, auto, pin }`。`auto` 标记由 wait 重算 pass 管理的等待；`pin` 标记手写（safe-asm）指令。
- **效应模型**（`sched/effects.rs`）是调度、RA 活跃性与 wait 重算共用的唯一事实来源：每条指令的读/写 **dword 槽**（`Slot = vreg·16 + dword`，含 `vcc/scc` 槽）、内存访问描述（空间 Global/Lds、读写、基址槽、偏移、大小）、fence 类别（Barrier / WaitVm / WaitLgkm / SetPrio / Nop）。MUBUF 访问以“未知基址”建模（恒 may-alias）；LDS 直达形式 = 全局读 + `Fence::Barrier`，无 VGPR 目的。
- **MIR 验证器**（`mir::verify`）检查操作数类别与范围（如 `SBufferRsrc` 必须由 S2 构造 S4、buffer imm ≤ 4095）。

## 5. 调度栈

`sched/` 共 15 个模块、约 2,700 行：

| 模块 | 作用 |
|---|---|
| `effects` / `order` | 效应与内存/fence 排序规则（图与验证器共用） |
| `depgraph` | 块内依赖图（寄存器、内存、fence、pinned 屏障边） |
| `model` | 机器模型表（延迟、发射、资源类；证据等级 Verified/Experimental/Unknown） |
| `list` | 压力感知 list scheduler（多窗口，取不超压力上限的最优，否则保持原序） |
| `hoist` | `ds_read` 上提（受压力与距离上限约束） |
| `swp` / `taildup` / `cfg` | 软件流水；if/else 菱形尾部复制；CFG 清理 |
| `waits` | scoreboard 等待重算：剥离 `auto` 等待，按 vmcnt/lgkmcnt 队列重新插入 **partial** 等待（SMEM 乱序时退化为 0） |
| `verify` / `sim` / `liveness` | 独立验证器（顺序置换、等待充分性）、简易模拟、活跃性 |

- **默认关闭**（`AMDC_SCHED` 未设）；FMHA kernel 由 `lib.rs` 按名称给出预设：`ld`（只重算 wait + `ds_read` 提升，不重排）用于 `_t`/`_sv`，`_vs128/_t128/_kvsplit/_g*` 使用完整 list scheduler（含各自的压力上限/跨 barrier 设置）。`AMDC_SCHED=0` 关闭。
- **失败回退**：出现 “out of VGPRs” 时依次收缩 hoist 距离、压力上限，最后回到未调度代码。
- **pinned 屏障语义**：`MInst.pin` 的指令在依赖图中是屏障节点——run 内保持原序，run 之前的所有指令先于 run 的首条，run 的末条先于其后的所有指令；hazard 填充与 `sink_defs/coalesce` 等 lowering pass 也不得移动或跨越 pinned 指令。
- 调度器对 **ALU 受限** 的假设已被量化证伪（§12）：它保留为可选工具，不是性能杠杆。

## 6. 寄存器分配

- **分配策略链**：线性扫描（紧凑）→ 单寄存器从高位打包 → **两阶段**（宽元组优先，窄值按 dword 级精确活跃段填进空洞）；仍不够时由 lowering 驱动 remat（`remat1..4`）与 **LDS spill**（`ldsspill8..64`）。每一步失败才进入下一步；不会静默产生 spill 之外的降级。
- **活跃性**：块级 + dword 级精确段（`precise_segments`），带“可达定义”剪枝（无可达定义的槽不占寄存器）；循环回边把 header 的 live-in 延伸到 latch。
- **reaching-writer 检查**（两阶段结果的守卫，`check_reaching`）：路径感知——每个物理寄存器记录“当前写者 → 在该路径上先于它的写者集合（history）”。读槽 r 违规，当且仅当 (a) r 自己的值从未到达而他者到达，或 (b) 存在他者写者其 history 含 r（r 被写后在同一路径上被覆盖，即真实 clobber）。“他者的路径上 r 从未被写”（循环体内定义、循环后读取时经零趟路径）**不是**违规：该路径上读本就是未定义值。平坦的“写者集合 ≤ 1”判据会误拒任何持久循环变量。
- **MFMA 操作数检查**（`check_mfma_operand_aliasing`）：A/B 与 D 不得重叠，C/D 同址或不相交，起址对齐。
- **S4 描述符类**：SGPR 四元组、4 对齐，参与普通 SGPR 分配与复用；描述符是普通 SSA 值，在值变化处重新构造。

## 7. Hazard / Wait 模型

分层（每层只做自己能证明的事）：

1. **lowering 等待**：按 `pending_vmem/lgkm` 在读取前插入 `auto` 等待。
2. **wait 重算**（`waits.rs`）：调度后剥离 `auto` 等待并按 scoreboard 重新插入；显式等待保留。存储与 LDS 写也占队列槽，因此 partial 计数精确。
3. **值级 hazard pass**（`fix_value_hazards`，按虚拟寄存器）：MFMA 结果延迟（f16/fp8 16×16/32×32×16 为 12，scaled / unscaled 32×32×64 与 `smfmac` 为 20；strict 模式 +1）、同 opcode 就地累加链豁免、VALU 写 → 矩阵操作数/`readfirstlane`/`permlane` 需 2 等待态（进位输出 4）、超越函数 5 槽、packed fp32 → 超越函数 2 槽、`vcc` 生产-消费、`s_nop N` 计 N+1 态；可选 stall 填充（pinned 指令不参与）。
4. **物理级 hazard pass**（`pad_physical_war`，RA 之后，因为 RA 会重用寄存器，值级 pass 看不到）：
   - 在途 MFMA 的 **C 源** WAR；
   - 在途 MFMA 的 **目的** WAW/RAW 对*另一虚拟寄存器*（19/21 槽，沿 CFG 传播到后继块）；
   - **宽存储数据 WAR**：`global_store_dwordx4`/`buffer_store_dwordx4` 发射后 2 个等待态内不得有 VALU 写其数据寄存器（对应 LLVM `has12DWordStoreHazard`，gfx940+ 为 2）。
5. **`pin_check`**：对含 pinned 指令的 kernel，在调度后的 MIR 上用编译器自己的 hazard / wait / 半写模型复查——如果这些 pass 需要在 pinned run **内部**补 nop/wait，则编译失败（边界处的 padding 属正常）。

**三类真实 bug（均由硬件现象驱动，均有 MIR 级回归测试）**

| 类别 | 现象 | 根因 | 修复 |
|---|---|---|---|
| 物理 WAW vs 在途 MFMA 目的 | opt-in 组合下输出非确定（~2.7% 元素，6 次 6 个 md5） | 结果已死的 MFMA 其目的寄存器被 RA 分配给新值，MFMA 在 ~19–21 槽后落笔覆盖 | `pad_physical_war` 跟踪在途目的区间 |
| VMEM 宽存储数据 WAR | 出厂 split 路径偶发数据损坏（~5%，时序扰动下 97%） | `global_store_dwordx4` 异步读数据 VGPR，后随 VALU 在 1 等待态内覆盖 | 宽存储数据区间 + 2 等待态（`_kvsplit` 仅 +4 `s_nop`） |
| reaching-writer 误报 | 任何持久循环变量使两阶段 RA 编译失败 | 平坦写者集合把零趟路径当成歧义 | 路径感知检查（§6） |

**一致性守卫**：`tail_invariance`（llvm-tail 与 own 的指令流除 nop/wait/`delay_alu` 外逐条相同）、`hazard_oracle`（约 1 万对指令的 own 模型 vs LLVM 插入量）、`hazard_classes`、`phys_waw`、`pin_check`、`pinning`、`smfmac_hazards`、`dma_hazards`。

## 8. 指令覆盖

- **123 个 MOp** 都有 own emit；其中 **117 个**在 `llvm20.tbl` 和 `llvm24.tbl` 中有行（`Key[:kinds] | 签名 | MIR 文本`，`kinds` 区分寄存器/立即数源，最具体的行胜出），其余为 own-only（DMA / LDS 直达 / 16x16x128 scaled MFMA 等）。
- **own-only**：DMA（`global_load_lds*`）、LDS 直达的 `buffer_load … lds`、以及含它们的 kernel——LLVM MIR 对这些要么无法表达要么与我们的 m0 序列模型不符；`uses_own_only_ops` 据 MIR 实际内容判定，`auto` 后端据此选择。
- 每个新增 MOp 的最小闭环：DSL op/stmt → lowering → emit → 效应/代价/hazard → 两个 `.tbl` 行（或登记 own-only）→ 探针 kernel 进 `kernel_names()`（覆盖 `asm_assembles`、`tail_invariance`、`llvm_tail`、`register_accounting`）→ 硬件探针。
- 已知限制：`llvm20` 行是 `llvm24` 行的拷贝，仅由 llvm24 的 `llc` 做机器验证；hazard-oracle 跳过 S4 描述符操作（128 位 SGPR live-in 无法在其按块 live-in 中表达），这些操作由逐行机器验证和硬件探针覆盖。

## 9. 内存路径

| 路径 | 要点 |
|---|---|
| FLAT（`global_*`） | 64 位 VGPR 地址；offset 13 位有符号；`sc1` 写穿/一致性读、`global_atomic_add`（返回旧值，设备作用域）；SADDR 形式（标量基址 + 32 位 VGPR 偏移） |
| LDS | `ds_read_b32/b64/b128/b64_tr_b8`、`ds_write_b32/b64/b128`；offset 为编译期常量并检查对齐（`x + 常数` 链由 lowering 折进立即数）；读写都占 lgkm 队列槽；`ds_read_b128` 可直接写 AGPR |
| 全局→LDS DMA | `global_load_lds_dwordx4/dword`：LDS 目的 = `M0` + imm + lane·size（`s_mov m0` + `s_nop 0`）；vmcnt 跟踪，效应上带 Barrier fence；own-only |
| **MUBUF** | 见下 |
| last-arriver / 内存序 | 完整契约见 [`last-arriver-memory-model.md`](last-arriver-memory-model.md) |

**MUBUF**（`buffer_load/store_dword{,x2,x4}`、`buffer_load_dword{,x4} … lds`）：

- **描述符**：4 个连续 SGPR `[base lo, base hi & 0xffff, num_records, 0x00020000]`（dword3 在 gfx950 硬件验证）；由标量 kernarg 指针经 `s_mov/s_and/s_mov/s_mov` 构造；`buffer_rsrc`（size = 0xFFFFFFFF，关边界检查）、`buffer_desc(ptr, bytes)`、**`buffer_desc_dyn(ptr, runtime_bytes)`**（dword2 取自一致的 SGPR 值/循环变量，可在每块/每 tile 重新构造）。
- **寻址**：`vaddr`（逐 lane 32 位字节偏移，或省略为 `off` 形式）+ `soffset`（一致 SGPR 值/循环变量，或 0..64 的内联常量）+ 12 位 `imm` + 需要时 `m0`；LDS 形式中 `imm` 同时加到内存地址与 LDS 目的。
- **越界语义（gfx950 硬件验证）**：偏移 ≥ `num_records` 的读返回 0；写被丢弃；**LDS 直达形式把越界 lane 以 0 写入 LDS**（不是丢弃）。range check **不含 soffset**，大小须已包含意图中的基偏移。该语义使“最后一个 partial tile”无需 clamp 或分支。
- **Hazard**：与 FLAT 同属 VMEM 类、占 vmcnt 槽；`buffer_store_dwordx4` 数据适用宽存储 WAR（数据是 MIR 源 0）；LDS 形式无 VGPR 目的、走与 `global_load_lds` 相同的 vmcnt 记账。
- **LLVM 行**：`SBufferRsrc(Dyn)` 与全部 load/store（OFFEN/OFFSET，操作数序 `vaddr, srsrc, soffset, offset, cpol, swz`，内存操作数 `addrspace 8`）；两个 LDS 直达形式为 own-only。

## 10. safe-asm 桥与 pin

手写热点窗口用 `crates/amdgpu-safeasm`（详见 [`safe-asm-design.md`](safe-asm-design.md)）：

- **接入**：`Op::Asm`（值形式）、`Stmt::VarAsm`（一个 in/out 循环变量）、`Stmt::VarAsmN`（多 in/out，按声明顺序绑定）。`AsmBlock` 是纯数据（指令 + 端口 + 常量 + pin/loop_body 标志），经 `lower_asm` 降成普通 MIR，**之后与 DSL 代码一起**走调度、RA、hazard、tail。
- **类型层**：端口（`In<N>` 只读、`inout`、输出）、所有权 token（`Pending`、`Mfma`、`Smfmac`、`Trans`、`Pw<L,H>` 半写 type-state）、`Ready` 证明、const-assert 常量总线、部分 `lgkmcnt/vmcnt` 由 scoreboard 精确给出。
- **验证器**：直线、自带时钟；含 **op 白名单**（raw-only 操作直接拒绝）。
- **循环契约**：循环体 `Seq::finish_loop_body()` 必须 **drain-closed**（入口与出口都无在途矩阵结果、无未完成 DS/VMEM、不得 `output_exit`），这样直线验证器对每次迭代都成立。
- **pin**：`Seq::pin()` 使整段成为调度器屏障（§5）；编译器在调度后的 MIR 上运行 `pin_check`（§7.5）。

## 11. 验证体系

| 守卫 | 保证 |
|---|---|
| `cargo test`（66 个目标 / 311 项） | 全部单元/集成/硬件无关测试（`tail_invariance`、`mir_invariants`、`register_accounting`、`asm_assembles`、`llvm_tail`、hazard 系列、`ra_reaching`、`pin*`、safe-asm ui compile-fail …） |
| `register_accounting` / `mir_invariants` | 描述符声明了每个用到的寄存器；RA 从不 clobber 活值；已知错误类能被检出 |
| 硬件探针（`tools/*_probe.sh`） | 新原语的硬件语义（越界、半写、等待态、last-arriver、MUBUF、unscaled MFMA …） |
| 测量协议 | 溯源核对 `.s/.co`、交错 A/B、同轮厂商、split 形状反复启动比对 md5 |

## 12. 设计结论（已量化）

- **指令数与关键路径**：削减 VALU 8.7% 只带来 wave 周期 +1.2%——kernel 不是 VALU/指令受限，去掉的槽变成依赖停顿；因此调度器微调与地址 VALU 削减（含 MUBUF，纯 buffer 上界 +0.1…1.0%）对性能中性。这是把调度栈保留为工具而非杠杆的依据。
- **正确性优先**：本设计里最有价值的部分是“能把每一类硬件 hazard 变成编译期/测试期可复现的模型”（§7），而非某个优化。

## 13. 初版中未采纳的部分

| 初版设想 | 结局 |
|---|---|
| Resource SSA / Async Graph、`Event/Transition/Phase` 全套资源版本模型 | 未实现；由“效应模型 + scoreboard + 显式 `Pending` token（safe-asm）+ hazard pass”取代，覆盖了实际踩到的全部 bug 类 |
| MachineDAG / AllocationPlan + SchedulePlan witness | 未实现；RA 与调度直接作用于 MIR；`pin_check` 承担“手写顺序必须被保留且被校验”的职责 |
| 复用 LLVM late pass 并接管破坏 pin 的 pass | 简化为“导出 post-RA MIR，只跑 tail pass”；pinned 顺序由我们的 hazard 层保证 |
| 跨卡通信（MORI CCO、RemoteView capability） | 不在本项目范围 |
| gfx1250 / CDNA5 硬件语义 | 不在本项目范围（目标为 gfx950） |

初版中**保留下来的原则**：类型系统管构图期非法描述、验证器管全路径事实、LLVM 源码不可修改、`Must` 语义的 pin（pinned 指令顺序不得被悄悄改写）、被替换的证明必须重新验证（post-RA 的任何改写都要重新过 `pin_check`/hazard）。