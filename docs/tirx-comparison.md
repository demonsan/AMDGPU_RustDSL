# 本仓库 vs Apache TVM TIRx：异同对照

> 资料来源（2026-11-05 查阅）：[apache/tvm-tirx README](https://github.com/apache/tvm-tirx)、
> [设计博文](https://tvm.apache.org/2026/11/03/rethinking-abstractions-in-the-agentic-era)、
> [Programming Overview](https://tvm.apache.org/tirx/programming_guides/index.html)、
> [IR Overview](https://tvm.apache.org/tirx/ir_guides/overview.html)、
> [Operators/Regions](https://tvm.apache.org/tirx/ir_guides/operators_calls_and_regions.html)、
> [Symbolic Analysis](https://tvm.apache.org/tirx/ir_guides/symbolic_integer_analysis.html)、
> [Data Layout](https://tvm.apache.org/tirx/programming_guides/data_layout.html)、
> [Tile Operands](https://tvm.apache.org/tirx/programming_guides/tile/tile_patterns.html)、
> [Compiling](https://tvm.apache.org/tirx/programming_guides/compiling.html)、
> [IKET](https://tvm.apache.org/tirx/programming_guides/iket_profiling.html)、
> [Rust 扩展](https://tvm.apache.org/tirx/advanced/extending_ir_in_rust.html)。
> 本仓库状态：Main `v7.1`（见 [current-status.md](current-status.md)）。早先的 IR 梳理见 [ir-review-tirx-sftensor.md](ir-review-tirx-sftensor.md)。

## 1. 一句话

两者都在回答同一个问题："agent 写 kernel 的时代，编译器层还应提供什么"，而且都选择了**低而显式的抽象边界 + 可选的 tile 便利层 + 结构化 IR 供分析与验证**。

不同之处在于：
- TIRx 是**通用、可插件扩展、多语言的 IR 与编译基础设施**，目前面向 CUDA，代码生成交给 NVRTC/LLVM；
- 本仓库是**单目标（gfx950）、端到端自研后端、内建验证器的 kernel 编译器**，有调度、寄存器分配、hazard 与 wait 插入，直接产出生产级 kernel 并与厂商同轮对比。

## 2. 逐项对照

| 维度 | TIRx | 本仓库 | 异同 |
|---|---|---|---|
| 定位 | 最小、稳定、可扩展的 DSL 与编译基础；"分析 + 验证需要结构化 IR" | 面向 MI355X 的 kernel DSL + 自研编译器；FMHA、GEMM、MLP 出货并对标厂商 | 理念相同（为 agent 提供可分析可验证的 IR）；TIRx 做平台，我们做"平台 + 产品" |
| 目标硬件 | CUDA（sm_75 起；Hopper TMA、Blackwell tcgen05 有专章） | AMD gfx950（CDNA4）；gfx1250 仅有调研文档 | 不重叠 |
| 前端语言 | Python TVMScript（`@T.function`），核心 C++，Rust 扩展（cdylib + TVM-FFI） | 纯 Rust 嵌入式 builder（`FnBuilder` + `expr` 层），元编程即 Rust | TIRx 多语言跨 FFI；我们单语言、零三方依赖、`forbid(unsafe)` |
| IR 形态 | 不可变 AST 树；`RegionStmt(op, args, body)`；所有节点注册结构化 hook | 区域树：`Loop/If/Region/Stage` + `Op`(89)/`Stmt`(50) 封闭枚举 | 都是"带 body 的区域"（我们的 region IR 正是借鉴它的结果）；TIRx 节点不可变、开放，我们的是封闭枚举 |
| 扩展机制 | "一切皆插件"：新类型、算子、pass 由独立包动态加载，算子靠属性（类型推断、效果、各目标 lowering）驱动 | 新 op 要改 crate；MIR 层有算子属性表（`op_attrs.rs`，穷尽匹配），DSL 层尚无 | 最大差异。我们是单仓单体，换来可控性与零依赖 |
| 抽象边界 | 低而显式：流水结构、同步、角色、内存放置、后端 intrinsic 都写在源码里；可选 tile 层（执行 scope、tensor layout、tile 分派） | 同样低而显式：角色分支、流水 cadence、tile 协议标记都由作者写；tile 库提供 TiledCopy/TiledMma/pipeline/softmax 等 | 高度相似。TIRx 的 tile op 是可分派的 IR 算子（按 scope、layout、target 选实现）；我们的 tile 层是 Rust 函数，展开后才进 IR |
| 数据布局 | `TileLayout`：`S[shape : stride@axis]`，命名物理轴（`laneid`/`warpid`/内存 `m`）、副本 `R[...]`、偏移、swizzle 组合、指令专用布局 | CuTe 式 `Layout`/`IntTuple`/`Swizzle`/TV 布局、`IssuePlan`、`BitForm`、`convert_layout` 分档，以及构建期证明（单射、bank、tr_b8 源） | 同源（都继承 CuTe 思想）。TIRx 记法更直观（命名轴、显式副本）；我们多了构建期证明和布局转换代价分档 |
| 符号分析 | `sym.Analyzer`：化简、`can_prove`；`ConstraintContext` 在遍历时自动带入循环范围和分支事实 | `amdgpu-sym`：多项式、约束上下文、区间、位集；验证器按区域树遍历并带入均匀分支事实 | 同构（我们的统一 sym 也是按它的思路做的）；TIRx 还有整除、迭代映射、区域分析 |
| 打印 / 解析 / 结构工具 | TVMScript 打印与解析往返；`structural_walk/map/equal/hash` 由编译 hook 驱动；`format_tree` | `amdc print-ir --stage dsl\|mir\|mir-ra`；α-等价结构相等与哈希；**无解析器** | 我们缺解析器和通用 walk/map 框架（各 pass 自行递归） |
| 代码生成 | 交给成熟工具链：CUDA C → NVRTC（寄存器分配、调度、PTX→SASS 由 NVIDIA 完成），host 走 LLVM | **全自研后端**：lowering → MIR → list/slotfill 调度 → 线性扫描 RA（含 AGPR）→ hazard/wait → asm；LLVM 只作可选的 post-RA 尾段或汇编器 | 根本差异。AMD 上要拿到厂商汇编级性能，必须自己控制 AGPR、`s_waitcnt`、MFMA/VALU 编排、M0/SCC；TIRx 在 CUDA 上可以依赖 nvcc |
| 同步与异步协议 | "同步由调用方负责"：tile op 不插 barrier；`cp.async`/TMA/`tcgen05` 的完成协议由作者照文档写 | 作者写 barrier 与 tile 协议标记，**编译器自动插 `s_waitcnt`**，验证器证明协议（到达/完成/可见/代际四轴） | 都让作者显式写同步结构；我们额外有编译器插等待 + 静态证明 |
| 正确性 | 博文目标：结构化 IR + 高效 checker（独立项目）+ 远期形式化（如 Lean）；文档里 lowering 会检查 tile 操作数视图是否满足指令约束 | 内建 `gpu_verify`（越界、寄存器类与对齐、路径敏感等待、协议、stage 契约）+ 严格证书（可计数的信任账本）+ 功能证明（GEMM 索引级与 IR 符号执行、FMHA 递推）+ 门禁（精度、确定性、压力、性能） | 我们在"验证"上已落地得更深；TIRx 把它作为方向和外部 checker |
| 性能剖析 | IKET：kernel 内 `range_push/pop`、`mark` → Perfetto 时间线 | `s_memtime` stamps（按阶段、按 tile）+ ATT 分析脚本 + rocprof-report 工作流 | 思路相同（kernel 内打点）；IKET 产出时间线 UI，我们产出周期拆分表 |
| 主机互操作 | TVM-FFI 自动生成绑定，直接接 PyTorch tensor | `amdc gen-host`：C++ ABI 头（`Args`/`validate()`）+ JSON schema | TIRx 面向 Python 生态；我们面向 C++ 启动与厂商 ABI 兼容 |
| 面向 agent | 多语言、动态扩展，agent 可在独立包里做 pass 和 analysis | 证据驱动：同轮门禁、md5 不变式、证书、`ab_*.sh` / `perf_matrix.sh`，agent 改动必须过门禁 | 侧重点不同：TIRx 重"可扩展"，我们重"可证伪、可回归" |

## 3. 我们可以继续借鉴的

1. **DSL 层算子属性化**：把 `Op`/`Stmt` 的类型规则、效果、lowering、验证器传递函数挂成属性。MIR 层已经这样做了，DSL 层仍是封闭枚举，这样可降低加 op 的成本。
2. **tile op 进 IR**：把 `TiledCopy`/`TiledMma` 这类调用做成可分派的 IR 算子（按 scope、layout、atom 选实现），而不是在 Rust 里展开成指令序列。这样 IR 层能看到 tile 粒度，便于分析、证明和融合（例如 `convert_layout` 可以在 IR 层自动插入）。
3. **命名物理轴的布局记法**：给 `Layout` 加 `lane`/`wave`/`m` 轴标注和显式副本，让打印输出与证明报告更可读。
4. **IR 解析器与通用 walk/map**：打印→解析往返，让 agent 能直接编辑 IR 文本做实验；通用遍历框架可以减少各 pass 的重复递归代码。
5. **IKET 式时间线输出**：把 stamps 的结果导出成 Perfetto 时间线。

## 4. TIRx 目前没有、我们有的

- 自研的寄存器分配、调度与 wait 插入：AMD 上性能的关键，也是 slotfill、AGPR、`edgepad` 这些优化能做的前提。
- 内建的静态协议验证与严格证书，以及功能等价证明（含"跳 tile"反例）。
- 同轮对比厂商的门禁体系，以及出货 kernel 的 md5 不变式。

## 5. 结论

方向一致：结构化 IR、低而显式的边界、可选 tile 层、符号分析、为 agent 的分析与验证服务。我们早先的区域 IR、统一 sym、打印与结构相等，都是对 TIRx 的直接借鉴。

差异来自目标不同：TIRx 是跨硬件的通用基础设施，把后端交给成熟工具链；本仓库是针对一个 AMD 目标的全栈编译器，后端和验证器都自研。如果要进一步靠拢，最有价值的是 §3 的第 1、2 项：DSL 层算子属性化，以及 tile op 作为 IR 一等公民。
