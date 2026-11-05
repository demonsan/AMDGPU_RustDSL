# GPU 语义验证器（gpu_verify v1）：全局内存 / LDS / VGPR / SGPR 的越界与生命周期证明

> 实现：`crates/amdgpu-compiler/src/gpu_verify.rs`；DSL 合同 API：`crates/amdgpu-dsl/src/lib.rs`（`Ext`、`arg_tensor`、`arg_tensor_slack`、`ka_range`、`ka_is`、`grid_is`、`sym_range`、`assume`）；测试：`crates/amdgpu-kernels/tests/gpu_semantics.rs`；命令行：`amdc gpu-verify <kernel>|--all|--shipped [-v] [--strict]`。
> 本文是**语义模型与边界**。它说明 v1 证明了什么、依赖什么假设、哪些**没有**被证明（第 8 节），避免“编译器保证安全”被过度解读。

## 0. 一句话

编译器在 **DSL IR 层**对每一次全局/LDS/buffer 访问做符号区间分析，在 **最终 MIR 层**检查寄存器合法性与**路径敏感**的等待/生命周期。任何不能证明的性质都报告为 `Error`（带符号表达式），除非 kernel 作者用显式合同（`arg_tensor` / `ka_range` / `assume`）声明；合同是**主机契约**（host contract），验证器把它们当作前提，并在报告里逐条列出被信任的 `assume`。

## 1. 主机契约与基本假设（A1–A6）

| 编号 | 假设 | 谁负责 |
|---|---|---|
| A1 | `arg_tensor(name, nbytes)` / `arg_tensor_slack(name, logical, alloc)` 声明的分配大小**由调用者保证**（内存真实可读写这么多字节）。loads 以 `alloc` 为界，stores/atomics 以 `logical`（缺省 `alloc`）为界。 | 主机 |
| A2 | `ka_range`、`ka_is`（kernarg 标量等式）、`grid_is`（网格维度等式）、`sym_range` 为真；工作组 id ∈ `[0, grid-1]`，工作项 id ∈ `[0, wg-1]`。 | 主机 |
| A3 | **i32 运算不回绕**（每个地址算术结果 < 2^31）；验证器不逐运算证明，而是把它记为全局假设。 | 主机（张量 < 2 GiB） |
| A4 | `b.assume(var, lo, hi)`（循环变量）**必须是归纳的**：初值和每个 `set_var` 都要被证明落在范围内，**否则是 Error**（`GV-ASSUME-VIOLATED`，`--strict` 不能认证；见 kernel-occ 的 `gv_attack_assume_loop`：assume 为 [0,0] 而每轮 +1，会在第 2 轮越界）。确实由分析器看不见的守卫（例如循环退出条件）保证的计数器，必须写 `b.assume_trusted(var, lo, hi, "原因")`：它不检查归纳，作为 TRUSTED（Info）逐条列出，评审时只审这些行。对普通**值**的 `b.assume(value, lo, hi)`：若分析已蕴含则标冗余，否则 TRUSTED（Info）。 | kernel 作者 |
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

### 3.1 LDS tile 生命周期 / 版本模型（v1.1，`gpu_tiles.rs`）

v1 只证明“不越界”。v1.1 证明**错误生命周期使用**：读到过期/未发布的版本、读一个已被下一轮覆盖的槽、缺 barrier、DMA 未等待就发布、两个 live tile 重叠。

**合同 API（均不生成代码）**

| API | 含义 |
|---|---|
| `b.lds_tile(name, base, size)` / `b.lds_ring(name, base, slot_size, slots)` | 声明命名 tile / 环（槽 k = `[base + k*slot_size, +slot_size)`） |
| `b.lds_alias(a, b)` | 允许两个 tile 重叠（时间复用同一存储） |
| `b.tile_produce(name, slot, version)` | 此后（至 `tile_end`/下一个标记/barrier/控制流边界）**本 wave 发出的 LDS 写与 DMA** 产生 `(name, slot)` 的版本 `version` |
| `b.tile_consume(name, slot, version)` | 此后本 wave 的 **LDS 读** 期望 `(name, slot)` 的版本 `version` |
| `b.tile_end()` | 结束标记作用域（barrier 与 if/loop 边界也会结束它） |

`slot`/`version` 是 IR 值，必须是**循环计数器与常量的编译期函数**（环形计数器：`it & 1`、`it + 1` …）；否则 `GV-TILE-INDEX`。

**静态检查**：(a) 声明的 tile 都在 `lds()` 段内，且两两不相交，除非 `lds_alias`（`GV-TILE-OVERLAP`）；(b) 标记作用域内每个 LDS 访问（含 DMA 的 64 lane 目的区间）必须落在被标记的 tile/环内（`GV-TILE-ESCAPE`，复用 v1 的区间分析）；写不能出现在 `tile_consume` 下、读不能出现在 `tile_produce` 下（`GV-TILE-KIND`）；作用域外的 LDS 访问计入 `GV-TILE-UNTAGGED`（Warn：其生命周期未检查）。

**协议模拟**：对 kernel 的控制骨架按**波角色**（工作项 id 相关的 `if` 的 then/else 两侧）分别**具体执行**，每个循环展开 1..=4 次（`BreakIf` 条件已知为真则提前退出），以**动态 barrier 序号**（phase）给每个事件编号；slot/version 表达式按循环变量的具体值求值。

* **发布**：一次 `tile_produce` 在其写**完成**（DMA 须 `wait_vm(0)`，DS 写须 `wait_lds(0)`；部分计数不视为完成）之后的**第一个 barrier** 处对其他 wave 发布（可见相位 = 该 barrier 序号 + 1）；同一 wave 自己读则只需完成。
* **读**（`tile_consume`）必须：(1) 看到**已发布的**最新版本，且版本号 **恰为期望值**（否则 `GV-TILE-STALE`，消息给出看到的版本、生产者角色/语句、发布相位）；(2) 同版本的所有协作生产者（多 wave 各写一部分）都已发布；(3) 其读在下一个 barrier 前**完成**（`wait_lds(0)`），否则 barrier 不能把它与下一个写者隔开（`GV-TILE-PENDING`，Warn）；(4) 其他 wave 对同一槽写**别的版本**的窗口 `[发出相位, 发布相位)` 不得与读窗口 `[读相位, 完成相位]` 重叠（否则 `GV-TILE-RACE`：例如单缓冲被下一轮覆盖、读旧版本时下一版本已落地）。
* **写写**：不同 wave 并发写同一槽**不同版本** → `GV-TILE-RACE`；同一版本（协作填充）允许。
* **每条 producer→consumer 边在所有角色上都跨 barrier**：由“发布相位 ≤ 读相位”与两侧动态 barrier 计数相等（v1 的 `GV-BAR`）共同保证。

**展开上限与“周期性闸门”（`GV-TILE-UNROLL-LIMIT`）**：模拟只展开有限次迭代（`max(10, 2×最大槽数+4)`），所以一个只在前几轮正确、之后才竞争的协议（kernel-occ 的 `gpu_tile_tripcount::late_race`：第 0..4 轮正确，第 5 轮起覆盖消费者的槽）不能靠“模拟过了”被认证。闸门：**每个 tile 标记的 `slot`/`version` 表达式必须是循环计数器的仿射/周期函数**——只由常量、计数器的加减、乘/移位/`&`/`^` 常量构成，且用到的每个计数器初值为常量、只被 `+ 常量` 或 `^ 常量` 更新（`min`、比较、`select`、两个计数器的乘积都不允许）。满足时，所有相对量（槽号 mod 环大小、版本差、相位偏移）对迭代数是周期的，违例只依赖相邻 ≤ 环大小+1 轮，展开 ≥ 2×槽数+4 轮即覆盖任意迭代数；不满足则 `GV-TILE-UNROLL-LIMIT`（Error，`--strict` 不可认证），消息指出依赖的运算，并说明协议只模拟了 N 轮。若结构确实“按构造周期”但不是上述形式，作者必须显式写 `b.tile_trust_periodic("原因")`：作为 TRUSTED（Info）列出。这是对“迭代抽象”的一个**保守、可判定**的实现（不展开到任意轮的符号证明）；假设为：循环迭代数 ≥ 1、所有循环同一迭代数。

**已证明 / 假设计数**：报告的 `proven` 里 `tile-reads` 为已证明的读事件数（按展开场景累计），`tile-cover` 为证明落在标记 tile 内的访问数；对 `slot/version` 的唯一假设是“循环最多展开 4 次、各循环同一迭代数”，以及角色只有两个（then/else）。

**控制流（v1.3）**：(1) **角色分裂**只指“工作项 id 相关、且分支内含 barrier/tile 标记的、尚未处于某个角色内的 `if`”（then = 角色 0，else = 角色 1）；其余工作项相关的嵌套 `if`（如 `wave_live`）和所有**波一致的运行时 `if`**（如 `is_tail`、`is_last`，条件取决于 kernarg/运行时值而模拟器不可知）都**探索两个结局**：每个结局是一条独立的路径（对每个角色做深度优先的路径枚举，上限 64 条），两个角色的路径两两配对，**工作组一致的条件在两个角色里必须取同一结局**；每对路径都做完整的协议检查（因此 `is_tail` 的两个迭代副本、epilogue 的 `wave_live`/`is_last` 都被分别验证）。(2) 循环**内部**含 barrier/标记/LDS 访问的运行时 `if` 不展开（路径数随迭代数指数增长）：`GV-TILE-UNSUPPORTED`（**Error**，不可认证），消息写明是哪个 `if`；两路径总数超过 64、或模拟超过 200 万条语句同样是 Error。

**周期与展开范围（v1.3 修正）**：展开次数必须不小于标记表达式的**周期**。周期上界由表达式锥推出：每个 `& mask` 贡献其位宽（msb+1），每个 `>> k` 贡献 k（`x >> k` 的第 j 位是 x 的第 j+k 位），被 `^ 常量` 翻转的计数器贡献 1；周期整除 `2^M`（`M = Σ>> + max 掩码位宽`）。模拟迭代数取 `max(10, 2×最大槽数+4, 2×2^M+2)`（不超过 70）。若 `2×2^M+2 > 70`（例如 kernel-occ 的 `affine_period_256`：槽 `(nxt&1)+((nxt>>8)&1)` 在第 255 轮才让生产者落到消费者的槽上，M=9）→ `GV-TILE-UNROLL-LIMIT`（Error），消息写明周期上界与 horizon 上限；只有显式 `tile_trust_periodic(原因)`（作为 TRUSTED 计入证书）才能通过。真正的二周期环（M=1）报告 `GV-TILE-PERIOD`（Info）：“周期整除 2，模拟 10 轮 × 每种退出奇偶，覆盖任意迭代数”。

**出厂 kernel 的结果（v1.6 共享可见性规则下）**：`_vs128`/`_vs5`：**58,880 个读事件**被检查，0 个 tile 错误；`_kvsplit`：**240,640** 个读事件，0 错误（另有 160 个未带标记的访问 `GV-TILE-UNTAGGED`，Warn）；`_vs5` 同 `_vs128`。变异测试：逐个删除 `_vs128` 的 30 个 barrier，**24 个**被协议检查或 `GV-BAR` 发现；未被发现的 6 个（barrier 序号 6、7、14、21、22、29，语句 3106/3214/6246/8940/9048/11600，集中在 `is_tail` 的两个迭代副本和 kernel 末尾）要么不保护任何已标记的 tile，要么该区域的标记不完整——应由 kernel 作者复核。

**私有（每 wave）区域（v1.4）**：epilogue 转置之类的暂存区只被**同一个 wave** 先写后读（例如 `base + wave*4096`），不需要也不应要求其他 wave 的发布。`b.lds_private(name, base, size_per_wave, waves)` 声明 `waves` 个互不相交的切片；用 `tile_produce/tile_consume(name, wave_key, version)` 标记，其中 `wave_key` 必须是**地址所用的同一个值**（例如 `wid`，地址 = `wid*size + lane*16 + base`）。语义：(1) **自己的切片**：验证器把地址与 `wave_key` 都线性化（`+`、乘/左移常量，其余值作为原子），要求 `地址 − wave_key×size − base ∈ [0, size − 访问宽度]`；读到别的 wave 的切片（如 `wid ^ 1`）或地址不是 wave_key 的线性函数 → `GV-TILE-PRIVATE-ESCAPE`（Error）；(2) **未标记**的 LDS 访问若不能证明与任何私有区域不相交 → 同一错误码；(3) **生命周期按 wave 本身**：读者必须读到**本 wave 自己已完成**（`wait_lds(0)`/`wait_vm(0)`）的最新版本，且版本号相符，否则 `GV-TILE-STALE`（“not completed by this wave” 或版本不符）；不需要 barrier，也不与其他 wave 的事件配对；(4) 私有区域与任何共享 tile 重叠 → `GV-TILE-OVERLAP`；(5) 私有标记的 wave_key 不参与循环周期分析。
**周期声明必须是真周期（v1.5）**：周期上界 `2^M` 的 M 由标记表达式锥推出：每个 `& mask` 贡献其位宽，**每个 `^ 常量`贡献该常量的位宽**（`version = nxt ^ 256` 的相邻版本关系每 512 轮变化一次，M=9），每个 `>> k` 贡献 k，被 `^` 翻转的计数器贡献 1；**每个标记操作数各自成锥取最大值**（而不是把无关节点相加）。`<< k` / `* 常量` 不增加周期位，但会让 32 位值回绕：锥内乘数/移位总量 ≥ 2^16 时 `GV-TILE-UNROLL-LIMIT`（消息给出可能回绕的迭代数；否则按假设 A3 不回绕）。报告的周期不是“声明了就算”：模拟器在最长的模拟轨迹上**逐标记语句验证**——槽序列每 p 轮重复、版本每 p 轮前进一个常量——验证不通过则 `GV-TILE-UNROLL-LIMIT`；通过时 `GV-TILE-PERIOD`（Info）写明 “claim is VERIFIED on the simulated sequence”。因此 `nxt ^ 256` 被闸门拒绝，`nxt << 8`、`nxt * 256`（每轮常量步进 256，相对结构周期为 1）被接受并报告经验证的周期，真正的二周期环仍报告 “dividing 2”。
**共享 tile 的可见性规则（v1.6 修正）**：共享 tile 的数据由**其他 wave** 生产，每个 wave **独立消费**——不是“自产自销”。一个角色（role）在模拟里代表该角色的**所有** wave，所以**同角色不豁免**：消费者看到版本 v，当且仅当 v 的**每一个**生产者都已被 barrier **发布**（`发布相位 ≤ 读相位`）；生产者自己的 `wait_vm(0)`/`wait_lds(0)` 只是让**下一个 barrier** 能发布它，本身**永远不足以**让共享 tile 可见（同一 wave 写完自己的切片、等待、然后读整块 tile 而中间没有 barrier，会读到其他 wave 还没写的部分 → `GV-TILE-STALE`，消息 “a shared tile is only visible after a barrier publishes it”）。竞争检查同理：读窗口 `[读相位, 完成相位]` 与**任何** wave（含同角色）写入其他版本的窗口 `[写相位, 发布相位)` 相交即 `GV-TILE-RACE`。**只有私有区域（`lds_private`）允许自产自销**：整个 wave 的 64 个 lane 写自己的切片、`wait_lds(0)`、同一个 wave 读，不需要 barrier，因为没有其他 wave 触及该切片。

**v1.3 的边界**：(i) 同一 tile 内按**字节**的部分重叠（两个 wave 写同一槽的不同子区间）不被区分——它们靠“同版本协作”语义被允许；(ii) 角色只有两个，且所有循环迭代数相同；(iii) 未带标记的 LDS 访问（`GV-TILE-UNTAGGED`）不被检查；(iv) 跨 workgroup 与跨 kernel 的别名、数据竞争不在范围内。

## 4. 寄存器类别合法性（MIR，RA 之后）

对每个被指令引用的虚拟寄存器（`verify_mir`）：

| 检查 | 规则 | 错误码 |
|---|---|---|
| 保留寄存器 | VGPR 不得分配到 `v0`（工作项 id）；SGPR 不得分配到 `s0..s7`（kernarg 指针、工作组 id） | `GV-REG-RESERVED` |
| 对齐（**gfx950 汇编器实测**） | 任意宽度（2/4/8/16）的 VGPR 元组（MFMA 的 A/B/C/D、`global/ds/buffer` 数据、packed 运算）起点必须是**偶数**（`v[2:17]`、`v[6:21]`、`v[14:29]` 合法，奇数起点被拒）；SGPR 对起点为偶数，4 元组（buffer 描述符、`s_load_dwordx4`）和 8 元组（`s_load_dwordx8`）起点为 4 的倍数（`s2/s10` 被拒，`s4/s8` 合法）。我们的 RA 额外把元组对齐到其宽度：偏离只作为 `GV-REG-POLICY`（Info），不是错误 | `GV-REG-ALIGN`（Error）、`GV-REG-POLICY`（Info） |
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

### 7.1 信任预算（certificate）与 `assume` 的审计

`amdc gpu-verify <kernel>|--all|--shipped` 在每个 kernel 后打印 `certificate: N trusted assumes, M trusted contracts, K unproven`（最后还有合计行）：**N** = 显式被信任的假设（`assume_trusted`、对普通值的未被蕴含的 `assume`、`tile_trust_periodic`），**M** = 声明的主机合同条数（`arg_tensor*`、`ka_range`、`ka_is`、`ka_one_of`、`grid_is`、`sym_range`），**K** = 未证明的性质（Error 数）。K = 0 且 N、M 被评审接受才算“认证”；`--strict` 在 K > 0 时失败。

### 7.2 `assume` 的审计

`amdc gpu-verify <kernel> -v` 打印每条 `GV-ASSUME`：`proved`（归纳成立）、`already implied`、`TRUSTED`。**kernel 的“被信任清单”就是这些行**；评审时只需要审阅它们。

## 8. 不可证明 / 不在 v1 范围内（诚实清单）

1. **LDS 数据正确性**：tile 版本、生产者/消费者与 barrier 的对应、跨 wave 的写后读、DMA 完成（`vmcnt`）与读者之间的顺序（v1 只证明“不越界”）。
2. **工作组间/线程间数据竞争、别名**（两个指针指向同一分配、重叠写）、原子序；全局内存的**初始化**与浮点数值。
3. **i32 回绕**（A3）：作为假设，不逐运算证明。
4. **从内存加载的整数/指针**（运行时索引表等）：区间无界；`LoadU16` 是唯一有规则的加载（`[0, 65535]`）。
5. **变量位移/除法/取模、两边符号不明的 `and/xor`**：无规则 → 无界。
6. **safe-asm 块内部**的访存**明确不被检查**（`Op::Asm`/`Stmt::VarAsm`/`VarAsmN`）：其地址输入来自被检查的 IR 值，但块内的 `ds_*`/`global_*` 偏移、宽度与顺序只由 safe-asm 自己的类型/验证器负责；报告里有 `GV-ASM` Info，不能把含 safe-asm 的 kernel 的 LDS/全局证明读成覆盖了块内访问。
6b. **合同缺口（由 kernel 作者补，验证器能表达但不会代劳）**：(a) 任何指针 kernarg（包括计数器/信号区，如 `kvs_counters`）都用 `arg_tensor` 声明它的分配大小，否则 `GV-GLB-NOEXT`；(b) 取值受限的标量用 `ka_one_of(offset, &[2, 4, 8])`（例如 KV 切分数；分析使用其 min/max，**幂次/整除性质仍由 `assume` 的作者负责**）；(c) 由形状推出的标量用 `ka_is` 写成等式（例如 `rows_total = B * S * H`：`b.ka_is(0x284, Ext::sym("B") * Ext::ka(S_off) * Ext::ka(H_off))`）；(d) 合同的真实性（声明的分配是否等于主机真正传入的大小）是 A1 的内容，需要主机侧测试（kernel-occ 提议的 `contract_host_truth`）核对，验证器本身无法检查。
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
| `GV-ASSUME-VIOLATED` | Error | 循环变量的 `assume` 不是归纳的：初值或某个 `set_var` 不能被证明落在假设范围内 | `set_var var0 = [1, 1] is not provably inside the assumed range [0, 0]: the assumption is not inductive …` | 放宽 assume，或在守卫确实存在时用 `assume_trusted(.., 原因)` |
| `GV-TILE-OVERLAP` | Error | 声明的 tile 超出 LDS 段，或两个 live tile 相交且未 `lds_alias` | `live tiles \`a\` [0, 2048) and \`b\` [1024, 3072) overlap and are not declared as aliases` | 调整布局或声明别名 |
| `GV-TILE-ESCAPE` | Error | 标记作用域内的 LDS 访问不在被标记的 tile/环内 | `LDS access [..] is not provably inside tile \`buf\` (slot interval …)` | 修地址或标记 |
| `GV-TILE-KIND` | Error | 读出现在 `tile_produce` 下或写/DMA 出现在 `tile_consume` 下 | `an LDS write / DMA is issued under a tile_consume marker` | 修标记 |
| `GV-TILE-STALE` | Error | 读者看不到期望版本：未发布（缺 barrier/未等待完成）、版本不符（槽已被覆盖/下一版本未到）、协作生产者未全部发布 | `read of \`buf\`[1] expecting version 3 at phase 4 finds NO published version: version 3 is produced by role 0 … but is only published at phase 5` | 补 barrier / 等待，或修 slot/version 表达式 |
| `GV-TILE-RACE` | Error | 其他 wave 在读窗口内写（或仍在落地）同一槽的别的版本；或不同 wave 并发写不同版本 | `read of \`buf\`[0] (version 1, phases 1..1) overlaps the write of version 2 by role 0 … no barrier separates them` | 双缓冲/加 barrier |
| `GV-TILE-UNROLL-LIMIT` | Error | tile 标记依赖的运算不是循环计数器的仿射/周期函数，协议只模拟了有限轮，更多轮的启动**未被覆盖** | `the tile marker depends on Min, which is not an affine/periodic function … a launch with more iterations is NOT covered` | 用 `+ 常量`/`^ 常量` 更新的计数器表达 slot/version，或 `tile_trust_periodic(原因)` |
| `GV-TILE-PRIVATE-ESCAPE` | Error | 访问私有（每 wave）区域的地址不能证明落在 wave_key 对应的自己的切片内；或未标记访问可能触及私有区域 | `LDS access is not provably inside the private slice of wave key … another wave's slice may be touched` | 用同一个 wave_key 计算地址并标记；让其他访问与私有区域分离 |
| `GV-TILE-INDEX` | Error | slot/version 不是循环计数器与常量的编译期函数，或槽越界 | `the slot / version of the \`buf\` marker is not a compile-time function of the loop counters` | 用环形计数器表达 |
| `GV-TILE-UNSUPPORTED` | Error | 循环内部的运行时 `if` 含 protocol 事件；路径数 > 64；模拟步数超预算 | `a runtime if INSIDE a loop contains barriers / tile markers / LDS accesses …` | 把 `if` 提到循环外或让两臂一致 |
| `GV-TILE-PERIOD` | Info | 周期与模拟范围、路径组合数、已检查读事件数 | `tile markers are periodic with a period dividing 2 iteration(s); 10 iterations … cover any trip count` | — |
| `GV-TILE-PENDING/UNTAGGED/EMPTY` | Warn | 见 3.1 | | |
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

* 全部 88 个 `kernel_names()` kernel + 出厂集：**寄存器合法性与路径敏感等待 0 错误**（即默认强制检查全绿；当时的出厂 asm md5 不变：`_vs128` 00d34086、`_t128` f83c02ce、`_kvsplit` 148b329d、`_combine8h` 19bdd5df）。
* **出厂集再次切换（路线 B，tag `v4.1-layout-p3` 起）**：默认路径为布局层 kernel `fmha_fwd_hd128_mxfp8_v3_*`，当时 asm md5 `68d2832b` / `e4c7e75d` / `e4fc18b1` / `24a95a51`（v7.0：`763142f9` / `91b26e5d` / `e1fe92a4` / `24a95a51`）；`--shipped --strict` 0 错误 0 警告，**24 / 100 / 0**（与 dsl/legacy 相同）；proven：LDS 600 / 336 / 760，tile-reads 73,600 / 36,800 / 299,520。GEMM `gemm_mxfp8_bs_v4`：3 / 19 / 0（`_v3` 已在 2026-11 整理中删除）。
* **之后加入的规则**（均有攻击测试）：计数版 `vmcnt(n)` 完成规则（`tests/tile_l123.rs`）；`x ^ y` / `x & m` 的可能位收窄与声明上界参与的多项式盒子检验（`tests/gv_bits.rs`）；AGPR 寄存器类与统一 512 预算检查（`tests/agpr.rs`，[`agpr.md`](agpr.md)）。
* **出厂集切换（2026-11-02，`fmha-port`）**：默认路径改为 tile 库组合的 4 个 kernel，asm md5（`--backend own`，`fmha-port` e7cc33e）：`dsl_vs128` 438452c1、`dsl_t128` ef706387、`dsl_kvsplit` ef8c37ab、`dsl_kvsplit_combine8h` ab3d6b09；`amdc gpu-verify --shipped --strict`：0 错误 0 警告，24 trusted assumes / 100 contracts / 0 unproven（proven 计数与旧集逐项相同；`dsl_t128` 另有 tile 协议证明 tile-reads 29440，无 periodic trust）。旧手写集 `--legacy`（已移除） 的 asm md5 不变（上行 4 个值），供 A/B。
* **LDS 访问全部被证明**（无需任何 `assume`）：`_vs128` 600、`_vs5` 600、`_kvsplit` 760、`_t128` 320、`_t` 225、`_vs9` 424、`_vs` 310 个访问；其中 DMA 目的区间按 64 lane 计。
* **角色 barrier 平衡**：`_vs128` 18、`_t128` 8、`_kvsplit` 18、`_t` 11 个 wave 分歧的 if/else 全部平衡。
* 全局访存：未标注合同的 kernel 报 `GV-GLB-NOEXT`（这是预期的缺口清单——合同由 kernel 作者补）；已标注者（kernel-opt 分支）的残差见其报告。

## 12. 测试（`tests/gpu_semantics.rs`，22 项）

正例：LDS 声明段内、全局带形状下界、`min` 夹紧（n≥1）、描述符不高估（静态/运行时）、环形 `xor` 变量、平衡的角色 barrier、合法 MFMA 寄存器、两分支都等待、循环体内等待、出厂 kernel 的强制检查与默认编译。
反例（每类一个）：LDS 溢出、LDS 下溢、全局无形状下界（打印 `ka(0x8)` 残差）、全局无合同、`min` 夹紧但 n 可能为 0、写入 zeroed slack、描述符高估（静态与运行时 +16）、无界增长的环形变量、不平衡的角色 barrier、MFMA 目的错位、MFMA 乘数与目的重叠、S4 描述符错位/SGPR 超限/VGPR 超限、分支一侧漏等待、循环携带的在途值跨迭代使用前未等待。
## 13. 四类"共享 / 边界"资源的保证方式（跨 wave LDS、buffer rsrc、atomic、TDM）

本节回答一个常见问题：LDS 这种会被不同 wave 共享的空间、MUBUF 描述符（`buffer_rsrc`）的边界、atomic、以及 TDM，
在 Rust 里各自**如何保证 safe 使用**、保证到哪一层、边界在哪。总原则：**DSL 侧是安全 Rust 的类型化 API（无 unsafe），
语义安全由"声明式合同 + 编译期证明义务 + 硬件实测语义"三者合成，任何不能证明的都报 `Error` 或进入可计数的 trust 清单。**

### 13.1 跨 wave 共享的 LDS（tile 协议）

LDS 是块内 8 个 wave 共享的地址空间，硬件**不做任何边界或竞争检查**（实测：越界读不回零、不 fault；写被丢弃）。
因此"共享正确性"必须由编译器建模：

* **声明**（不产生代码）：`b.lds_tile(name, base, size)` / `b.lds_ring(name, base, slot_size, slots)` /
  `b.lds_private(name, base, size_per_wave, waves)` / `b.lds_alias(a, b)`；
* **标记**：`tile_produce(name, slot, version)` / `tile_consume(name, slot, version)` / `tile_end()`，
  槽位/版本必须是循环计数器的**仿射函数**（常数、`+ - * << & ^` by 常数；计数器初值常数、步进 `+ const` / `^ const`）；
* **证明**（`gpu_tiles.rs` 模拟器，按角色路径 DFS、运行时 if 的两种结果都探索、相位按动态 barrier 计数）：
  * **共享 tile：数据由其他 wave 生产，每个 wave 独立消费；可见性一律以 barrier 发布为准，同角色不豁免；私有区（`lds_private`）才允许自产自销。**（一个角色代表该角色的全部 wave：自己写完切片并 `wait_lds(0)` 后读整块 tile、中间没有 barrier = 读到其他 wave 尚未写的部分，报 `GV-TILE-STALE`。）
  * `GV-TILE-STALE`：读到未发布/版本不匹配的数据（生产者必须在 barrier 后才算发布；DMA 要 `wait_vm(0)`，
    DS 写要 `wait_lds(0)`）；
  * `GV-TILE-RACE`：读窗口内另一个 wave 写了不同版本（单缓冲被下一轮覆盖 = 典型竞态）；
  * `GV-TILE-PENDING`：读未完成就进入下一个 barrier；
  * `GV-TILE-OVERLAP` / `GV-TILE-ESCAPE` / `GV-TILE-INDEX` / `GV-TILE-UNROLL-LIMIT`；
  * **私有区**（`lds_private`）：每 wave 一段，读只能读自己那段（`GV-TILE-PRIVATE-ESCAPE` 拒绝读别人/非线性地址），
    生命周期按"自己 produce + 自己 wait"判定，不需要别的 wave 发布；
  * **未标记的 LDS 访问** = `GV-TILE-UNTAGGED` Warn（界已证、生命周期未证）——出厂 kernel 已全部标记（0 warning）。

**实测结果**：`_vs128`/`_vs5` tile-cover 600/600（58,880 次读事件）、`_kvsplit` 760/760（全部 LDS，240,640 次读事件），
`--strict` 0 错误 0 警告。**边界**：只建模 2 个角色、各循环迭代数相等、tile 内字节级重叠、跨工作组/跨 kernel 别名。

### 13.2 MUBUF `buffer_rsrc` 的边界

* **类型化**：描述符是 DSL 的 `Rsrc` 值（4 个 SGPR 四元组）；`buffer_rsrc`（静态 `num_records`）与
  `buffer_desc_dyn`（运行时 `num_records`，如 `min(nbytes, …)/stride`）都走同一套 IR。
* **证明 1（描述符不得高估）**：在描述符构造点检查 `num_records.hi <= extent.lo` → `GV-BUF-OVER`。
  理由：MUBUF 越界读返回 0、越界写被丢弃（硬件实测 `tools/buffer_probe.sh` 的 rw/lds/dyn 全 `BUFFER_OK`），
  所以"有界描述符"的访问是**硬件定义的安全**；但如果描述符高估了分配，越界就会静默读到别的张量的数据。
* **证明 2（无界哨兵）**：`num_records = 0xFFFFFFFF`（`buffer_rsrc` 的 no-bounds 形式）不享受硬件范围检查，
  其访问按普通 global 访问证明（需要 `arg_tensor` 与形状关系）。
* **证明 3（可见性/顺序）**：`sc1` 写穿透 / `sc0` atomic 纪律 + `buffer_wbl2`/`buffer_inv` 的顺序由
  `asm_check::check_lite_contract` 检查（last-arriver 融合路径），保证跨工作组看到的是排空后的数据。
* **边界**：运行时 `num_records` 表达式若写错（区间不足以证明 ≤ 分配）会被拒绝；跨工作组数据竞争不建模。

### 13.3 atomic

当前 DSL 只有 `global_atomic_add`（device scope `sc0`，返回旧值），`Op::AtomicAddU32` / `b.atomic_add_u32`。

* **边界**：`check_global(..., size=4, store=true)` —— atomic 是读-改-写，因此**按 store 检查 logical extent**
  （不允许打到 `arg_tensor_slack` 的 zeroed slack）；无 extent → `GV-GLB-NOEXT`，越界 → `GV-GLB-OOB`。
  单测：`tests/gpu_atomic.rs` 的 4 个界测试。
* **返回值生命周期**：`global_atomic_add` 的旧值经 VMEM（`vmcnt`）返回。效果模型是
  `MemAccess { space: Global, write: true }` → 进入 **vm 队列**；等待 pass 在任何路径使用它之前插入 `s_waitcnt`，
  `gpu_verify::verify_waits` 做**路径敏感**检查（分支一侧漏等、循环携带都会报 `GV-WAIT`）。
  单测：`tests/gpu_atomic.rs` 的 `using_the_atomic_result_before_a_wait_is_rejected`。
* **作用域/顺序纪律**：`check_lite_contract` 要求 `global_atomic_*` 用 `sc0`、不得用 `sc1`，且 barrier 前必须 vmcnt 排空。
* **边界（诚实）**：没有 acquire/release/模型序证明；跨工作组数据竞争、原子性的"证明"不在范围内
  （原子性由硬件保证，我们只约束地址、等待与 scope）；目前只有 1 种 atomic 形式（无 `buffer_atomic_*`、无 system scope）；
  safe-asm 窗口内的 atomic 只在 `GV-ASM` Info 中标注为"内部未检查"。

### 13.4 TDM（Tensor Data Mover）

**当前不支持，且目标 ISA 上不存在**：`gfx950` 没有 TDM；整个 crate 图里 TDM 相关 op 数为 **0**
（`grep -r "tdm\|TDM" crates/` = 0）。因此：

* DSL 无法表达 TDM（没有 builder API）；`Op::Asm` 的 raw 路径受 safe-asm **op 白名单**限制，无法偷偷塞入未知助记符；
  编译器也没有对应 MOp / emit 行 → **不存在"能写出来但没被检查"的 TDM 路径**。
* 若将来支持 `gfx1250` 的 TDM，架构上是复用的：TDM 描述符（base/size/stride）用 `arg_tensor`/extent 合同 +
  `GV-BUF-OVER` 式"描述符不得高估"检查；TDM 写入 LDS 的落点用 `tile_produce` + `wait_vm(0)` 发布 +
  `ds` 消费者用 `tile_consume` 验证版本；越界/回绕用同一区间分析。即：**ISA 支持是缺的，模型不缺**。

### 13.5 汇总表

| 资源 | Rust 侧保证 | 证明/检查 | 硬件实测语义 | 诚实边界 |
|---|---|---|---|---|
| 跨 wave LDS | 类型化 tile API（安全 Rust，无 unsafe） | `gpu_tiles` 协议模拟：STALE/RACE/PENDING/OVERLAP/ESCAPE/PERIOD（出厂 0 错误 0 警告） | LDS 无边界检查、越界不回零 | 2 角色/等迭代数/字节级重叠/跨 WG 别名 |
| buffer rsrc | 类型化 `Rsrc` 描述符 | `GV-BUF-OVER`（静态+运行时 num_records ≤ extent）+ 无界哨兵走 global 证明 + lite 契约（sc1/sc0/排空） | 越界读 0 / 写丢弃 / lds 形式零填充（`buffer_probe.sh`） | 运行时 num_records 表达式写错会被拒；跨 WG 竞争不建模 |
| atomic | 类型化 `AtomicAddU32`（device scope） | store 语义的 extent 检查 + vm 队列 + 路径敏感等待 + lite 契约 scope 纪律 | 单指令原子性（硬件） | 无 acquire/release 证明；无跨 WG 竞争建模；仅 1 种形式 |
| TDM | **不支持**（DSL/MIR/emit 均无；raw 被白名单拒绝） | — | — | 目标 gfx950 无此 ISA；模型设计可复用 |
### 13.6 `buffer_load … lds`（MUBUF global→LDS DMA）覆盖情况

`buffer_load_dwordx1/x4 … lds` 与 `global_load_lds` 一样是 **DMA 类**，两侧分别检查、生命周期按 produce 处理：

| 侧 | 检查 | 错误码 |
|---|---|---|
| **全局侧（源）** | 描述符 `num_records` 不得高估分配（静态/运行时） | `GV-BUF-OVER` |
| **全局侧（无界哨兵）** | `0xFFFFFFFF` 不享受硬件范围检查 → 按普通 global 访问证明（需 extent/形状关系） | `GV-GLB-NOEXT` / `GV-GLB-OOB` |
| **LDS 侧（目的）** | `M0 + imm + lane*size` 的**全部 64 lane** 落在声明 LDS 段内 | `GV-LDS-OOB` |
| **tile 内** | DMA 目的区间落在其 tile/ring 内，不允许越出 | `GV-TILE-ESCAPE`（别名 `GV-PROTO-*`） |
| **生命周期** | DMA 是 **produce** 事件：发布需要 `wait_vm(0)` **完成** + barrier；barrier 单独不能发布在途 DMA | `GV-TILE-STALE|GV-PROTO-NO-COMPLETION` |
| **消费/代际** | 消费者在发布相位之后读、代际一致（同 LDS tile 规则） | `GV-TILE-STALE|GV-PROTO-{STALE-GEN,RACE}` |
| **调度/完成轴** | effects 模型把 DMA 记为 Global 读进入 **vm 队列**（`BufferLoadDwordX4Lds`/`GlobalLoadLdsB128`），等待 pass 跟踪 issue order | `GV-WAIT` |

代码位置：`gpu_verify.rs` 的 `Stmt::BufferLoadLds`（同时调 `check_buffer` 与 `check_lds(… 64*size)`）；
`gpu_protocol.rs::is_lds_access` 把 `Stmt::BufferLoadLds` 映射为 `(dma=true, write=true)`；`sched/effects.rs` 把 DMA 记为 Global 读。

**测试**（`tests/gpu_buffer_lds.rs`，6 项全过）：descriptor 高估 → `GV-BUF-OVER`；无 extent → 拒绝；64-lane 目的越界 → `GV-LDS-OOB`；
合法 DMA 证明通过；DMA produce 在 `wait_vm(0)` 后发布 → 通过；**未 wait 的 DMA 即使有 barrier 也不能发布** → `GV-TILE-STALE`。

**验证现状（诚实）**：出厂默认路径用 `global_load_lds`，`--shipped --strict` 全绿；
当时 `FMHA_OPT=mubuf/mubufd` 的 MUBUF DMA 变体（已删除）**未认证**——运行检查器会报 `GV-BUF-OVER`（K/V 描述符的 num_records 表达式在未声明形状关系时不能证明 ≤ 分配）
以及 `mubuf` 变体的大量 `GV-GLB-OOB`/屏障问题。这些变体本就不在出货路径（此前的 A/B 也证明它们无收益），
要认证需要像默认 kernel 一样补 `ka_is`/extent 合同或修正描述符公式；检查器本身对 `buffer_load … lds` 的覆盖由上述 6 项测试证明。
### 13.7 gfx1250 的 TDM 与 split barrier：是否有独立建模？

**结论：没有，而且当前 target（gfx950/CDNA4）上不需要——因为它们不在这个 ISA 里。** 实测工具链能力表（`llvm-mc -mcpu=…`，两条路径一致）：

| 指令 | gfx942 (CDNA3) | **gfx950 (CDNA4，我们的 target)** | gfx1200 | gfx1250 |
|---|---|---|---|---|
| `s_barrier`（完整块屏障） | ✅ | ✅ | ❌ | ❌ |
| `s_barrier_signal N` | ❌ | **❌** | ✅ | ✅ |
| `s_barrier_wait N` | ❌ | **❌** | ✅ | ✅ |
| `s_barrier_leave` / `s_barrier_init m0` | ❌ | **❌** | ✅ | ✅ |
| `s_barrier_signal_isfirst N` / `s_get_barrier_state` | ❌ | **❌** | ✅ | ✅ |
| TDM 指令族 | ❌ | **❌** | ❌ | ✅（MI400） |

也就是说：
1. **split barrier（named barrier：init/signal/wait/leave/isfirst/get_state）是 gfx12xx / gfx1250 的族**。
   我们的编译器只有 `MOp::SBarrier` → `s_barrier` 一种屏障（`emit.rs`、`effects.rs` 的 `Fence::Barrier`），
   协议引擎的"相位 = 动态 `s_barrier` 计数"、"所有 wave 都到达同一个块屏障"也都建立在这唯一的语义上。
   **所以：gfx950 上不存在需要独立建模的 split barrier**；我们的模型对它是"不可表达"而非"未检查"——
   `Op::Asm` 的 raw 路径受 safe-asm op 白名单限制，未知助记符写不进来。
2. **TDM 同样不存在于 gfx950**，crate 图里 TDM 相关 op = 0（见 §13.4）；不存在"能写出来但绕过检查"的路径。
3. 历史上我们**探查过**这条路：`tools/gen3_sync_probe.sh`（gen3 设计的一次能力探针）Part 1 就是
   "gfx12 named/split barrier opcodes vs gfx950 (CDNA4)" 的逐指令接受表，Part 2 在硬件上对比了
   `block s_barrier` vs 软件 4-wave 组屏障（LDS 原子计数 + 自旋）vs 无同步；结论是 CDNA4 只有块屏障，
   "更细粒度同步"的 gen3 设计最终关闭（软件组屏障也慢且有风险，见 `docs/archive/gen3-design.md`、`kernel-gen2-design.md`）。

**若将来做 gfx1250 后端，需要新增的独立建模（按四轴映射）**：

| 原语 | 到达轴 | 完成轴 | 可见轴 | 代际轴 | 额外义务 |
|---|---|---|---|---|---|
| split/named barrier | `s_barrier_signal(token)` 记一次到达（可带计数） | `s_barrier_wait(token)` 才表示"到达完成" | barrier 本身**不**传递内存可见性（仍要 fence/等待，与 LDS 不同） | token 的多实例相位：`(token → signal_count, wait_count, generation)` 取代"全局 barrier 计数" | init/leave 配对；signal 后重复 init；`isfirst` 的"最后一个到达者"语义；计数不匹配（死锁/早发布） |
| TDM | TDM 完成事件（是否接 vmcnt 需硬件证据） | DMA 完成 + `wait_vm(0)` 才发布 | TDM 写入 LDS/global 的可见性规则 | 与 tile/region 的 version 代数复用 | 描述符 base/size/stride 的"不得高估"检查（复用 `GV-BUF-OVER` 式规则）；OOB 语义（TDM 是否零填充需实测） |

注意 gfx1250 上连 `s_barrier` 都被替换成 init/signal/wait 族，因此**即使只想支持"完整屏障"也要重做屏障表示**；
另外 gfx1250 的 `s_waitcnt` 拆成独立 `S_WAIT_*`（`docs/llvm-tail-integration.md` §117 已列为待办），
VALU 重配对（`gcn-create-vopd`）也需要重新确认。所以这不是"补两个 op"，而是一个新的 target 后端 + 协议引擎相位泛化。

**当前文档口径**：gfx950 只建模 `s_barrier` + waitcnt + fence + version 四轴；TDM 与 split barrier 明确列为
"目标 ISA 不存在 / 不在范围"，且有可执行能力表作为证据（本节）。多生产者/多消费者义务 O1–O4 与信任账本
与资源种类无关，新增原语时复用。