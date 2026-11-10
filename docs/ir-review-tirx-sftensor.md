# IR 梳理：TIRx 与 SF Tensor 形式化正确性能借鉴什么

> 参考：[TIRx IR Overview](https://tvm.apache.org/tirx/ir_guides/overview.html)、
> [Operators, Calls, and Regions](https://tvm.apache.org/tirx/ir_guides/operators_calls_and_regions.html)、
> [Symbolic Integer Analysis](https://tvm.apache.org/tirx/ir_guides/symbolic_integer_analysis.html)、
> [Structural utilities](https://tvm.apache.org/tirx/ir_guides/structural_utilities.html)、
> [Data Layout](https://tvm.apache.org/tirx/programming_guides/data_layout.html)；
> [SF Tensor: Proving Kernels Correct Instead of Testing Them](https://sf-tensor.com/news/formal-correctness)。
> 本文只是分析与建议，未改动代码。

## 1. 我们现在的 IR 是什么样

| 层 | 现状 | 位置 |
|---|---|---|
| DSL IR | 封闭枚举 `Op`（值，SSA `ValueId`）+ `Stmt`（副作用语句）；**扁平语句列表**，结构靠成对标记：`BeginLoop/EndLoop`、`BeginIf/Else/EndIf`、`TileProduce…TileEnd` | `crates/amdgpu-dsl/src/lib.rs`（`enum Op` L129，`enum Stmt` L561） |
| 上层库 | `expr.rs`（纯数据表达式树）、`amdgpu-layout`（`Ix`/`Layout`/`Swizzle`/证明）、`amdgpu-tile`（TiledCopy/TiledMma/Pipeline） | 生成上面的 `Op/Stmt` |
| MIR | 封闭枚举 `MOp`（约 125 个）→ 调度 → RA → hazard/wait → emit / LLVM tail | `crates/amdgpu-codegen/src/` |
| 验证 | `gpu_verify`（区间 + possible-bits + box 规则、合同 `Ext`）+ 协议引擎（角色路径 / 相位 / 代际） | `gpu_verify.rs`、`gpu_protocol.rs` |

三个已经反复付出代价的结构性问题：

1. **加一个指令要改 9 个文件。** 例如 `VMfmaScale16x16x128Fp8` 出现在 `lower.rs / mir.rs / regalloc.rs / agpr.rs / hazards.rs / sched/mod.rs / sched/model.rs / gpu_verify.rs / emit.rs`（LLVM tail 表另算）。SCC 建模、"own-only" 标记、M0 写入这类遗漏，都出在"某一处忘了改"。
2. **作用域靠成对标记。** 循环、分支和 tile 协议区段都是 begin/end 标记，每个 pass（验证器、协议引擎、调度、wait 插入）都要自己重建作用域；"标记没有配对"只能在运行期检查。
3. **三套各自为政的符号整数推理**：合同 `Ext`、布局 `Ix`（含 BitForm）、验证器的区间/位集/box。`ka_is` 声明顺序依赖、`256·KT` 在 `sym_range` 下不实例化、XOR/AND 界限这几类 bug 都是"同一个事实在另一套引擎里不认识"。

## 2. TIRx 可直接借鉴的五点

| TIRx 做法 | 对我们的意义 | 建议 |
|---|---|---|
| **算子 + 属性表**：新计算 = 注册一个 `Op`，行为由属性给出（`FInferType`、`TCallEffectKind`、`<target>.FLowerIntrinsic`、`FLegalize`…）；pass 按属性找算子，**从不枚举算子名单** | 正好对上问题 1 | 为每个 `Op`/`MOp` 建一张**属性表**（Rust 里用 trait + `const` 表实现）：类型规则、效果（内存空间、读写、队列、fence/SCC/M0）、hazard 类别、验证器传递函数、降级、emit、LLVM 行。再加穷尽性测试，保证"新 op 每个属性都填了"。enum 保留不动，属性表从一处生成 |
| **区域语句 `RegionStmt(op, args, body)`**：`device_entry` 等作为"带 body 的算子"，作用域天然成立 | 正好对上问题 2 | 把 `BeginLoop/EndLoop`、`BeginIf/Else/EndIf` 改成嵌套的 `Loop{body}` / `If{then, else}`；把 tile 协议标记改成区域：`TileProduce(tile, slot, gen) { body }`、`Private(…) { body }`。builder API 不变（闭包式 `b.region(|b| …)`），配对由构造保证 |
| **统一的符号整数分析器 + ConstraintContext**：一个 `Analyzer` 同时做化简和 `can_prove`；遍历时自动收集事实（循环范围、"循环会执行"、分支条件），pass 回调直接拿到当时的事实 | 正好对上问题 3 | 合并 `Ext` / `Ix` / 验证器区间为一个 `sym` crate：同一表示，同时支持化简（给布局降级 / BitForm 用）、证明（给验证器用）和合同事实；在区域化 IR 上用 walk-with-constraints 自动带入 `ka_range`、循环范围、分支谓词 |
| **不可变 IR + 结构相等/哈希 + 打印/解析（TVMScript、`format_tree`）** | 我们目前判断"等价"只能比 asm md5；调试和 agent 读 IR 只能看 asm | 给 DSL IR 和 MIR 加文本打印（必要时再加解析）和结构相等。测试可以直接断言"两种写法构造出同一棵 IR"，agent 也能逐层看 IR |
| **TileLayout 命名物理轴**：`S[(32,4) : (1@laneid, 1)]`、副本 `R[2 : 1@warpid]`、偏移 `+4`、`ComposeLayout` swizzle | 与 `amdgpu-layout` 的 TV layout 基本同构；我们的"广播 TV"（`make_tiled_copy_bcast`）、副本、wave 维都是隐式的 | 在 `Layout` 上引入**命名轴**（`lane` / `wave` / `m` / `slot`）和显式 replica，打印成同样的记法；证明（覆盖、单射）按轴说明 |

不建议照搬的部分：TIRx 是 Python 前端 + TVM-FFI 对象系统 + 插件化 pass，规模比我们需要的大得多；我们的单 crate Rust 实现、零依赖、`forbid(unsafe)` 应保持。

## 3. SF Tensor 形式化正确性：补上我们缺的那一层

他们做的事：把 PTX 或 AMDGPU 方言降到一个共享 IR，配一份"证明合同"（参考规格），对程序做**符号执行**得到数学模型，然后问 Z3 有没有输入能让 `kernel ≠ spec`：UNSAT 即证明，SAT 即给出反例。几个要点：

- **在实数上证明代数等价**，不证明 IEEE 逐位一致；误差另外界定。
- **barrier 建模成相位边界**：少一个 barrier，证明就失败。
- **不展开 lane/CTA**：只对"任意一个 lane"做符号推理，再加一个 α-重命名的第二个 CTA 检查跨 CTA 干扰。
- **online softmax 的证明**：用循环不变式（分母和累加器始终处在当前 max 的参照系）。

与我们的对照：

| | 我们（`gpu_verify` + 协议引擎） | SF Tensor |
|---|---|---|
| 内存安全：越界、提供方、描述符 | ✅ 符号证明 + 可计数信任账本 | 部分（安全检查） |
| 同步与协议：barrier 发布、代际、竞争、等待 | ✅ 四轴协议，角色路径枚举 | ✅ 相位 + 同步验证器 |
| **功能正确：kernel 是否真在算 spec** | ❌ 只靠测试（15 形状门禁、CPU 参考、确定性） | ✅ 实数上符号执行 + SMT |
| 反例 | 部分（打印符号残差） | ✅ 具体输入 |

我们最近两个只被运行期参考比对抓到的 bug，正好落在这个缺口里：

- GEMM 的"tile 1 被延迟阶段吞掉"：verifier 全部通过，结果错误；
- 编译器 wait 缺失导致的静默错算：由运行期参考比对发现。

对我们来说，这一层比 SF Tensor 更好做：我们的 IR 有类型、规模小，所有地址都由布局推出（索引关系是精确的）。

建议分两步，不引入 Z3（保持零依赖）：

1. **索引级功能证明（GEMM/FMHA 都适用，纯组合问题）**：对每个输出元素 `C[m,n]`，证明它在所有 tile 和 MFMA 上累加了且只累加了一次 `A[m,k]·B[k,n]·sa·sb` 的每个 `k`。也就是"每个 K 块恰好被每个输出用一次、来自正确的 slot 和代际"。方法是在布局层把 DMA 写入、LDS 读取、MFMA 操作数 TV 布局和累加器 TV 布局组合起来，判断它们构成的映射是否为双射。这一步就能静态抓住"跳过 tile 1"这类 bug。
2. **数值递推的代数证明（FMHA）**：对单行、任意长度，在实数上证明 online softmax 加我们的 rebase/rollback 配方（阈值只改变参照系）等价于朴素 softmax。证明方式是循环不变式，可以先作为独立的小证明器实现，不依赖 SMT。

## 4. 结论与建议顺序

**有帮助，而且正好对上我们已经付出代价的三个结构问题，外加一个缺失的验证层。** 建议顺序（每步都应保持出货 kernel 的 asm md5 不变）：

1. **属性表 + 穷尽性测试**：收益最大、风险最低，只改元数据组织方式，不改语义。
2. **区域化 IR**：`Loop{body}` / `If{…}` / tile 协议区域，builder API 不变；验证器和协议引擎改为树遍历。
3. **统一的 `sym` 分析器 + 遍历时自动收集事实**：合并 `Ext` / `Ix` / 验证器区间。
4. **打印器 + 结构相等**：服务调试、测试和 agent 读 IR。
5. **功能验证层**：先做索引级证明（GEMM），再做 FMHA 递推证明。

TIRx 的命名轴 layout 记法可以随第 3、4 步一起引入。
