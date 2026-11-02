# GPU 语义验证器（gpu_verify v1）：全局内存 / LDS / VGPR / SGPR 的越界与生命周期证明

> 实现：`crates/amdgpu-compiler/src/gpu_verify.rs`；DSL 合同 API：`crates/amdgpu-dsl/src/lib.rs`（`Ext`、`arg_tensor`、`arg_tensor_slack`、`ka_range`、`ka_is`、`grid_is`、`sym_range`、`assume`）；测试：`crates/amdgpu-compiler/tests/gpu_semantics.rs`；命令行：`amdc gpu-verify <kernel>|--all|--shipped [-v] [--strict]`。
> 本文是**语义模型与边界**。它说明 v1 证明了什么、依赖什么假设、哪些**没有**被证明（第 8 节），避免“编译器保证安全”被过度解读。

## 0. 一句话

编译器在 **DSL IR 层**对每一次全局/LDS/buffer 访问做符号区间分析，在 **最终 MIR 层**检查寄存器合法性与**路径敏感**的等待/生命周期。任何不能证明的性质都报告为 `Error`（带符号表达式），除非 kernel 作者用显式合同（`arg_tensor` / `ka_range` / `assume`）声明；合同是**主机契约**（host contract），验证器把它们当作前提，并在报告里逐条列出被信任的 `assume`。

## 1. 主机契约与基本假设（A1–A6）

| 编号 | 假设 | 谁负责 |
|---|---|---|
| A1 | `arg_tensor(name, nbytes)` / `arg_tensor_slack(name, logical, alloc)` 声明的分配大小**由调用者保证**（内存真实可读写这么多字节）。loads 以 `alloc` 为界，stores/atomics 以 `logical`（缺省 `alloc`）为界。 | 主机 |
| A2 | `ka_range`、`ka_is`（kernarg 标量等式）、`grid_is`（网格维度等式）、`sym_range` 为真；工作组 id ∈ `[0, grid-1]`，工作项 id ∈ `[0, wg-1]`。 | 主机 |
| A3 | **i32 运算不回绕**（每个地址算术结果 < 2^31）；验证器不逐运算证明，而是把它记为全局假设。 | 主机（张量 < 2 GiB） |
| A4 | `Stmt::Assume`（`b.assume(target, lo, hi)`）是 kernel 作者的断言：对循环变量**归纳检查**（初值与每个 `set_var` 是否落在范围内），能证明则标 proved，否则标 TRUSTED 并作为 `GV-ASSUME`（Info）列出；对普通值，若分析本身已蕴含则标冗余，否则 TRUSTED。 | kernel 作者 |
| A5 | MUBUF 越界语义是硬件定义的（gfx950 实测：越界读返回 0、越界写被丢弃，包括 `... lds` 形式向 LDS 写 0）；有界描述符的访问因此不需要额外证明，**但描述符不得高估分配**（见 4.3）。 | 硬件 + 验证器 |
| A6 | 单次 launch 内**工作组间无数据竞争**、全局内存别名不在本验证器范围（见第 8 节）。 | 作者 |

## 2. 地址模型（global / buffer / LDS 共用）

### 2.1 符号与多项式

* **符号**：`ka(0x..)`（32 位 kernarg 标量，按字节偏移）、`grid_x/y/z`（launch 网格，工作组数）、命名符号 `Ext::sym("B")`（例如不是 kernarg 的批次数）。
* **代数**：每个符号都是非负整数；有下界表（`ka_range` 的 lo，缺省：kernarg 0、网格/命名符号 1）。`ka_is`/`grid_is` 把符号**替换**为表达式（因此 split 启动 `grid_z = B * splits` 之后，`B*splits` 与 `B` 的乘积在证明里精确相消，不需要除法算子）；`lo == hi` 的 `ka_range` 把符号视为常数。
* **多项式** `Poly`：符号单项式的有理系数组合（`Q = n/d`，用于 `>> k`）。

### 2.2 区间

每个整数值的抽象是 `Itv { lo: [候选下界], hi: [候选上界] }`（空 = 无界；最多 4 个候选，取**任一**可证者）。运算规则（`gpu_verify.rs::An::eval`）：

| 操作 | 规则 |
|---|---|
| 常量 / kernarg 标量 / 工作项 id / 工作组 id | 点 / `[lb,+inf)` 的符号点 / `[0, wg-1]` / `[0, grid-1]` |
| `add` | 逐候选相加 |
| `mul` | 一边是常量：缩放（负数翻转）；两边均可证非负：下界×下界、上界×上界；否则无界 |
| `shl k`、`lshr k` | 乘以 2^k；`lshr` 要求非负，上界 `hi/2^k`（对常量向下取整） |
| `and` | 一边为非负常量 m：`[0, m]`（再加另一边的上界）；两边非负：`[0, min(hi)]` |
| `xor` | 与常量 c：高位取自 c、低位自由（`[c_hi, c_hi + 2^k - 1]`，k 为另一边位宽）；两边为小的非负数值：`[0, 2^k-1]` |
| `min` | 任一操作数的上界都是它的上界；下界必须同时是两边下界 |
| `select(cond,a,b)` | 若 `cond` 是 `x < k` / `x >= k`，**按条件细化 x 后重新求值两个分支**（环形计数器 `v < N ? v+1 : 0` 因此有界）；否则取两分支的 join |
| `readfirstlane`、指针 `ptr_add` | 区间透传 / 指针偏移相加 |
| 其它（浮点转换、位转换、无规则的运算） | 无界（Top） |

### 2.3 证明义务与判定

对一次访问（偏移区间 `[lo, hi]`、大小 `size`、上限 `limit`）：

```
lo >= 0                      （任一下界候选可证非负）
hi + size <= limit           （存在 hi 候选 h 与 limit 下界 l，使 l - h - size >= 0）
```

多项式非负的**充分判定**：把每个符号 `s` 代换为 `lb(s) + t`（`t >= 0`），展开后所有系数非负即证。它是**充分不必要**的：不能证 ≠ 越界，只表示“本分析证不出”，报告会打印符号残差让作者补合同。

### 2.4 循环

循环（`begin_loop … end_loop`）对循环变量做**不动点**：`cur = init`，每轮静默执行一遍循环体收集每个 `set_var` 的值，`cur = norm(join(cur, 值))`；最多 6 轮，第 5 轮起仍变化则**放宽为无界**。有 `assume` 的变量固定为假设范围，并在报告轮里对每个 `set_var` 做归纳检查。循环退出后变量保持循环超包络（它只可能持有初值或某个 `set_var` 值）。**计数器型变量**（`kv += 128`）不会收敛，需要作者写 `b.assume(var, lo, hi)`。

### 2.5 全局指针与描述符

* 指针 = `(kernarg 参数, 字节偏移区间)`；只由 `KernargPtr`、`PtrAdd`、同参数的指针 `Select` 构成。其它来源（加载得到的指针、safe-asm 产物）→ `GV-GLB-PROV`。
* 无合同的参数 → `GV-GLB-NOEXT`（不是静默通过）。
* **buffer 描述符**：`buffer_rsrc/buffer_desc/buffer_desc_dyn` 在构造点检查 `num_records.hi <= extent.lo`（描述符不得高估分配，否则“越界读返回 0”会变成“读到别的张量的数据”）→ `GV-BUF-OVER`；`0xFFFFFFFF` 哨兵（无界）的访问按全局访问证明；有界描述符的访问记为 `buffer(hw-guarded)`。

## 3. LDS 模型

* 声明段 `b.lds(bytes)`（`Function.lds_bytes`）；**每个** DS 访问 `[addr, addr+size)`（`ds_read_b32/b128/b64_tr_b8`×1/×4、`ds_read_b128`×2、`ds_write_b32/b128`）以及 **DMA**（`global_load_lds`、`buffer_load … lds`：目的地址 `M0 + imm + lane*size`，**64 个 lane 全部计入**，即检查 `[M0+imm, M0+imm+64*size)`）必须落在声明段内；下溢/回绕（地址下界可能为负）也报错。
* ring / 双缓冲：槽位变量由循环不动点（2.4）界定；`xor 1`、`select(v<N, v+1, 0)` 类更新自动有界，其它写 `assume`。
* MIR 层补充：`m.lds_bytes`（含 LDS 溢出区）≤ 160 KiB 且 ≥ DSL 声明值。
* **v1 不证明**：tile 的版本/生命周期与 barrier 的对应（谁在何时写、读者在哪个 barrier 之后）。v1 只保证“不越界”，不保证“读到的是正确版本”（第 8 节）。

## 4. 寄存器类别合法性（MIR，RA 之后）

对每个被指令引用的虚拟寄存器（`verify_mir`）：

| 检查 | 规则 | 错误码 |
|---|---|---|
| 保留寄存器 | VGPR 不得分配到 `v0`（工作项 id）；SGPR 不得分配到 `s0..s7`（kernarg 指针、工作组 id） | `GV-REG-RESERVED` |
| 对齐 | 宽度 n≥2 的 VGPR 元组起点为 n 的倍数（ISA 要求 64 位及以上为偶数、MFMA 操作数；我们的策略更严）；SGPR 对（偶数）与 4 元组描述符（4 的倍数） | `GV-REG-ALIGN` |
| 上限 | VGPR ≤ 256/wave；SGPR ≤ 102；整个 kernel 的 `max_v/max_s` 也受此限 | `GV-REG-LIMIT` |
| 矩阵操作数 | MFMA/SMFMA 的 A、B 与目的 D 不得重叠；D 的宽度/对齐沿用上面规则 | `GV-REG-MFMA` |

（RA 是否让两个同时活跃的值共用寄存器，由已有的 `regalloc::checked`/`mir_invariants` 负责；本验证器检查的是**类别规则与声明一致**。）

## 5. 路径敏感的等待 / 生命周期（MIR）

复用 `sched/waits.rs` 的 **scoreboard**（vm 队列与 lgkm 队列，每个在途项记 `after` = 其后同队列已发射的操作数），`verify_waits` 对整个 CFG 做数据流：

* 汇合点取各路径状态的**并集、`after` 取最小**（悲观）；等待 `vmcnt(n)/lgkmcnt(n)` 完成 `after >= n` 的项；循环回边迭代到不动点。
* 读/写任何有在途加载的寄存器（RAW/WAW）而没有足够的等待 = 错误：**任何一条 CFG 路径**（含分支一侧漏等、**跨迭代**——上一轮发射的加载在下一轮使用前没有等待）都会被抓到。
* `Fence::WaitLgkm` 遇到在途 SMEM 时不把部分计数当作完成（标量加载可乱序返回）。

错误码 `GV-WAIT`，消息含块号与指令序号。对未使用调度器的 kernel，此前并不运行这项检查；现在它是**默认强制**项（`AMDC_GPU_VERIFY` 未设时 MIR 级错误使编译失败）。

## 6. 角色 / barrier 协议（A/B 平衡）

`begin_if` 是 **wave 级**分支（无 exec 掩码）。条件若依赖工作项 id（污点传播：`WorkitemId` → 所有由它派生的值与循环变量），则不同 wave 走不同分支，两侧必须执行**相同次数**的 `s_barrier`，否则死锁。v1 把每侧的动态 barrier 计数表示为“循环迹数”的多项式（系数 1、N、N·M…；同一嵌套深度的循环在两个角色里被假设迭代次数相同），要求两侧多项式相等；不等则 `GV-BAR`。（barrier 的**位置**可以错开——例如角色 B 相位偏移——只要总数一致。）不依赖工作项 id 的 `if`（工作组一致）不检查。

## 7. 可证明 vs 必须由运行时守卫/合同承担

| 性质 | v1 | 说明 |
|---|---|---|
| LDS 访问在声明段内（含 DMA 64 lane、环形变量） | **证明** | 常量/掩码/异或/环形计数器全部可证 |
| 全局/buffer 访问在声明范围内 | **在合同下证明** | 需要 `arg_tensor` + 形状关系（`ka_range/ka_is/grid_is`）；`min` 夹紧被理解；不可证则 `GV-GLB-OOB` 并打印残差 |
| 描述符不高估 | **证明**（在合同下） | 静态与运行时 `num_records` |
| 寄存器合法性、MFMA 重叠 | **证明** | 对最终 MIR 全量检查 |
| 使用前必须等待（所有路径、含跨迭代） | **证明** | scoreboard 不动点 |
| 角色 barrier 平衡 | **证明**（假设等迭代数） | |
| 循环计数器区间 | **需要 `assume`** | 归纳检查，TRUSTED 项逐条列出 |
| 运行时守卫 | `buffer_desc_dyn`（硬件范围检查）、显式 `min` 夹紧 | 不是“证明”，而是让不可证的访问变成硬件定义的行为 |

### 7.1 `assume` 的审计

`amdc gpu-verify <kernel> -v` 打印每条 `GV-ASSUME`：`proved`（归纳成立）、`already implied`、`TRUSTED`。**kernel 的“被信任清单”就是这些行**；评审时只需要审阅它们。

## 8. 不可证明 / 不在 v1 范围内（诚实清单）

1. **LDS 数据正确性**：tile 版本、生产者/消费者与 barrier 的对应、跨 wave 的写后读、DMA 完成（`vmcnt`）与读者之间的顺序（v1 只证明“不越界”）。
2. **工作组间/线程间数据竞争、别名**（两个指针指向同一分配、重叠写）、原子序；全局内存的**初始化**与浮点数值。
3. **i32 回绕**（A3）：作为假设，不逐运算证明。
4. **从内存加载的整数/指针**（运行时索引表等）：区间无界；`LoadU16` 是唯一有规则的加载（`[0, 65535]`）。
5. **变量位移/除法/取模、两边符号不明的 `and/xor`**：无规则 → 无界。
6. **safe-asm 块内部**的访存不被检查（其地址输入来自被检查的 IR 值；报告里有 `GV-ASM` Info）。
7. **LLVM tail 重新插入的等待**：v1 检查的是我们自己的 MIR（tail 之前）；tail 与 own 后端的一致性仍由 `tail_invariance` 守护。
8. 证明是**充分判定**：`GV-GLB-OOB` 表示“证不出”，可能是分析过弱（需要补合同）而不是真的越界；反之，合同写错（例如声明了过大的分配）会使证明失去意义——这是 A1 的代价。
9. 寄存器活跃重叠、SGPR 的 VCC/exec/m0 等隐式寄存器由其它 pass 负责。

## 9. 错误分类（编译器拒绝什么、消息是什么）

| 码 | 严重度 | 触发条件 | 消息（节选） | 修复 |
|---|---|---|---|---|
| `GV-LDS-OOB` | Error | DS/DMA 地址下界可能 < 0 或上界超过声明段；`m.lds_bytes` 超过 160 KiB 或小于声明 | `stmt N: ds_write_b128: LDS access [addr, addr+16) not provably inside the declared 512-byte segment: upper bound not provably <= limit - 16: address [0, 1008] vs limit [512]` 并附 `unbounded because: …` / `address = …` | 增大 `lds()`、修地址、给环形变量写 `assume` |
| `GV-GLB-OOB` | Error | 全局访问不能证明在范围内 | `load of 16 bytes through \`ptr_q\` not provably inside its allocated extent: upper bound not provably <= limit - 16: address [0, 16 + ka(0x60)*ka(0x70) - …] vs limit [ka(0xA0)*grid_z]` | 补 `ka_range/ka_is/grid_is`，或写显式 `min` 夹紧 / `assume` |
| `GV-GLB-NOEXT` | Error | 指针来自没有 `arg_tensor` 的参数；或描述符所基于的参数没有合同 | `kernarg pointer \`p\` has no declared extent (arg_tensor / arg_tensor_slack)` | 声明 `arg_tensor` |
| `GV-GLB-PROV` | Error | 指针不是由 kernarg 指针经 `ptr_add` 派生；描述符基址不是纯 kernarg 指针 | `pointer provenance unknown …` | 用 `ptr_add(kernarg_ptr, offset)` 表达 |
| `GV-BUF-OVER` | Error | 描述符 `num_records` 不能证明 ≤ 分配 | `buffer descriptor over \`p\` declares num_records … which is not provably <= the allocation …` | 减小 `num_records` / 修正运行时大小表达式 |
| `GV-REG-RESERVED/ALIGN/LIMIT/MFMA` | Error | 第 4 节 | `vreg 7 (V16) at v40: a 16-dword VGPR tuple must start at a multiple of 16 …` | 编译器 bug（RA 输出非法）——报告给编译器组 |
| `GV-WAIT` | Error | 任一路径上，指令读/写了有在途加载的寄存器 | `block 3 inst 0 (VMovB32) touches a register with a load in flight (on at least one control-flow path, including loop back edges)` | 加 `wait_vm/wait_lds` 或调整调度 |
| `GV-BAR` | Error | 工作项 id 相关的 `if/else` 两侧动态 barrier 数不同 | `wave-divergent if/else … executes a different number of barriers: then [3, 4] vs else [3, 4, …]` | 平衡两侧 barrier |
| `GV-CONTRACT` | Error | 合同引用的参数不是指针 kernarg | `extent declared for \`x\` which is not a pointer kernarg` | 修正合同 |
| `GV-ASSUME` | Info | 每条 `assume`；`set_var` 不在假设范围内时也记录 | `assume var3 in [0, 1]: initial value proven inside; every set_var is checked against it` | 审阅 TRUSTED 项 |
| `GV-ASM` | Info | kernel 含 safe-asm 块 | `safe-asm blocks present: their internal memory accesses are NOT bounds-checked by v1` | — |

## 10. 策略与接入

`AMDC_GPU_VERIFY` 由 `compile_with` 在最终 MIR 上读取：

| 值 | 行为 |
|---|---|
| 未设（Default） | 收集全部发现；**MIR 级错误（`GV-REG-*`、`GV-WAIT`）使编译失败**；IR 级（边界/描述符/屏障）发现静默收集，不阻断（kernel 先补合同） |
| `warn` | 同上但把所有发现打印到 stderr，从不失败 |
| `strict` | 任一 `Error` 使编译失败（`Err` 带全部错误） |
| `off` | 关闭 |

`amdc gpu-verify <kernel>|--all|--shipped [-v] [--strict]` 打印每个 kernel 的发现数、已证明的访问数（`proven`）与全部非 Info 发现。接口：`gpu_verify::{verify_ir, verify_mir, analyse, enforce, Mode, Report, Finding, Severity}`。

## 11. v1 在出厂 kernel 上的结果（Main + 本分支，AMDC_GPU_VERIFY=warn，commit 见 PR）

* 全部 88 个 `kernel_names()` kernel + 出厂集：**寄存器合法性与路径敏感等待 0 错误**（即默认强制检查全绿；出厂 asm md5 不变：`_vs128` 00d34086、`_t128` f83c02ce、`_kvsplit` 148b329d、`_combine8h` 19bdd5df）。
* **LDS 访问全部被证明**（无需任何 `assume`）：`_vs128` 600、`_vs5` 600、`_kvsplit` 760、`_t128` 320、`_t` 225、`_vs9` 424、`_vs` 310 个访问；其中 DMA 目的区间按 64 lane 计。
* **角色 barrier 平衡**：`_vs128` 18、`_t128` 8、`_kvsplit` 18、`_t` 11 个 wave 分歧的 if/else 全部平衡。
* 全局访存：未标注合同的 kernel 报 `GV-GLB-NOEXT`（这是预期的缺口清单——合同由 kernel 作者补）；已标注者（kernel-opt 分支）的残差见其报告。

## 12. 测试（`tests/gpu_semantics.rs`，22 项）

正例：LDS 声明段内、全局带形状下界、`min` 夹紧（n≥1）、描述符不高估（静态/运行时）、环形 `xor` 变量、平衡的角色 barrier、合法 MFMA 寄存器、两分支都等待、循环体内等待、出厂 kernel 的强制检查与默认编译。
反例（每类一个）：LDS 溢出、LDS 下溢、全局无形状下界（打印 `ka(0x8)` 残差）、全局无合同、`min` 夹紧但 n 可能为 0、写入 zeroed slack、描述符高估（静态与运行时 +16）、无界增长的环形变量、不平衡的角色 barrier、MFMA 目的错位、MFMA 乘数与目的重叠、S4 描述符错位/SGPR 超限/VGPR 超限、分支一侧漏等待、循环携带的在途值跨迭代使用前未等待。