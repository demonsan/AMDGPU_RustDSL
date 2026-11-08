# 本仓库 vs Apache TVM TIRx：异同对照（当前状态）

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
> 本仓库状态：Main v8.1（`f92dd8d9`，见 [current-status.md](current-status.md)）。一个 kernel 今天如何构建与优化、FMHA 上的具体例子见
> `fmha-case-study.md`；早先的 IR 梳理见 [ir-review-tirx-sftensor.md](ir-review-tirx-sftensor.md)。

## 1. 一句话

两者都在回答"agent 写 kernel 的时代，编译器层还应提供什么"，都选择了**低而显式的抽象边界 + 可选的 tile 便利层 + 结构化 IR 供分析与验证**。
TIRx 是通用、可插件扩展、多语言的 IR 基础设施（面向 CUDA，代码生成交给 NVRTC/LLVM）；本仓库是单目标（gfx950）、后端全自研、内建验证器的
kernel 编译器，出货 FMHA / GEMM / MLP / RMSNorm 并与厂商同轮对比。过去两周里，TIRx 的结构化 IR 思想我们大多已经落地（§3），后端与验证上则有意走了另一条路（§5）。

## 2. 逐项对照（当前状态）

| 维度 | TIRx | 本仓库（v8.1） | 状态 |
|---|---|---|---|
| 定位 | 最小、稳定、可扩展的 DSL 与编译基础 | gfx950 kernel DSL + 自研编译器 + 验证器；产品 kernel 对标厂商 | 理念相同，我们是"平台 + 产品" |
| 前端 | Python TVMScript，核心 C++，Rust 扩展经 TVM-FFI | 纯 Rust builder（`FnBuilder` + `expr` 层），零三方依赖、`forbid(unsafe)` | 有意不同（§5） |
| IR 形态 | 不可变 AST；`RegionStmt(op, args, body)` | 区域树 `Loop / If / Region / Stage` + 编织区域标记；封闭枚举 `Op`(82) / `Stmt`(48) | 已继承（§3.3） |
| 扩展机制 | 一切皆插件；算子属性驱动类型推断、效果、lowering | MIR 层算子属性表（`op_attrs.rs`，穷尽匹配）；DSL 层仍是封闭枚举，编解码与 id 访问由生成器产出 | 部分（§4.1） |
| 抽象边界 | 低而显式：流水、同步、角色、内存放置写在源码里；可选 tile 层 | 同样低而显式；编排写在**生成的骨架源码**里（算子图 → 骨架 → slot + stage 实例） | 已继承（§3.7） |
| 数据布局 | `TileLayout`（命名物理轴、副本、swizzle） | CuTe 式 `Layout` / `IntTuple` / `Swizzle` / TV 布局 / `IssuePlan` + 构建期证明 | 已继承；命名轴未做（§4.3） |
| 符号分析 | `sym.Analyzer`、`ConstraintContext` 带入循环与分支事实 | `amdgpu-sym`（`Q` / `Poly` / `Ctx` / `Itv` / 位支撑）；验证器按区域树带入分支事实 | 已继承；整除/迭代映射未做（§4.4） |
| 打印 / 解析 / 结构工具 | TVMScript 往返；`structural_equal/hash`、`structural_walk/map` | irv1 文本打印与解析往返；α 规范化的结构相等/哈希；生成的 id 访问器 | 已继承；通用 map 部分（§3.4–3.5、§4.2） |
| tile op | 可分派的 IR 算子（按 scope / layout / target 选实现） | tile 层与 stage 库是 Rust 函数，展开后进 IR；stage 边界以 `Node::Stage` + `StageContract` 进 IR | 部分（§4.5） |
| 代码生成 | CUDA C → NVRTC；寄存器分配与调度由 NVIDIA 完成 | 全自研：lowering → MIR → 调度（list / slotfill / 按区域策略）→ RA（AGPR、寄存器规划）→ hazard / wait → asm；LLVM 仅可选 post-RA 尾段 | 有意不同（§5） |
| 同步 | 同步由调用方写；完成协议照文档 | 作者写 barrier 与 tile 协议标记；编译器插 `s_waitcnt`；验证器证明四轴协议 | 有意不同（§5） |
| 正确性 | 结构化 IR + 外部 checker + 远期形式化 | 内建 `gpu_verify` + 严格证书（`--shipped` 26 / 100 / 0）+ 功能证明 + 门禁 | 有意不同（§5） |
| 剖析 | IKET → Perfetto 时间线 | `s_memtime` stamps + ATT 脚本 + rocprof 工作流 | 推迟（§6） |
| 主机互操作 | TVM-FFI 绑定，直接接 PyTorch | `amdc gen-host`（C++ 头 + JSON）；Python JIT 以子进程调 `amdc` + ROCm clang | 有意不同（§5） |

## 3. 继承并已实现

| TIRx 思想 | 我们的落地 | 代码 / 文档 |
|---|---|---|
| **3.1 布局代数与 TV 布局** | `Layout` / `IntTuple` / GF(2) `Swizzle` / TV 布局 / `IssuePlan` / `BitForm`，`convert_layout` 按代数分档；构建期证明（单射、覆盖、bank、tr_b8 源、操作数匹配）；gfx950 copy 与 MMA atom 表 | `crates/amdgpu-layout`、`crates/amdgpu-isa`；[layout-tour.md](layout-tour.md) |
| **3.2 算子属性表** | 每个 MOp 的效果、hazard 类、调度代价、AGPR 合法性、后端可表达性在一张穷尽 `match` 表里，各 pass 查表而非枚举算子 | `crates/amdgpu-compiler/src/op_attrs.rs`；[op-attributes.md](op-attributes.md) |
| **3.3 区域 IR ≈ `RegionStmt`** | kernel 体是 `Loop / If / Region / Stage` 树，线性视图由树派生（配对错误无法表示）；`Node::Stage` 带 `StageContract`（融合边界契约，GV-STAGE）；编织区域 `Stmt::RegionMark` 给寄存器规划与按区域调度用 | `crates/amdgpu-dsl/src/lib.rs`；[region-ir.md](region-ir.md)、[weaver-regs.md](weaver-regs.md)、[weaver-sched.md](weaver-sched.md) |
| **3.4 结构相等 / 哈希 + α 规范化** | `dsl_structural_eq/hash`：ValueId / VarId 按首次出现结构化重编号（语句、op 操作数、注解），函数名不参与 | `crates/amdgpu-dsl/src/print.rs`、`ir_text::alpha_canonical`；[ir-printing.md](ir-printing.md) |
| **3.5 IR 文本往返 ≈ TVMScript** | `irv1`：`print-ir --stage dsl` 完整可解析，`ir-check` / `--from-ir`（emit、gpu-verify、jit-info、regplan）、A/B 脚本与 Python JIT 接受 `.ir`；全部 kernel 往返后 `Function` 逐字段相等、双后端 asm 与证书逐字节相同 | `crates/amdgpu-dsl/src/ir_text.rs`；[ir-text.md](ir-text.md) |
| **3.6 符号分析 + 事实上下文 ≈ `Analyzer` / `ConstraintContext`** | 合同、布局与验证器共用一套多项式 + 约束上下文 + 区间 + 位集；按区域树遍历时带入一致分支事实与循环不动点 | `crates/amdgpu-sym`；[sym.md](sym.md) |
| **3.7 显式编排在源码里** | 算子图（`*.graph.json`）→ 生成的 Rust 骨架（`skeleton_*.rs`，编排槽位可读可改）→ slot 实现 + stage 实例；weaver 把发射单元按显式顺序编织；出货 kernel 经此构建且 md5 不变 | `crates/amdgpu-tile/src/{opgraph/,stage.rs,weave.rs}`；[opgraph.md](opgraph.md)、[fusion-stage-library-design.md](fusion-stage-library-design.md) |
| **3.8 tile 便利层建在显式 IR 之上** | `TiledCopy` / `TiledMma` / `partition_*`、pipeline、softmax、epilogue、stage 库通用原语（E / R / Q）——全部展开成普通 IR，作者随时可以下到 IR 级 | `crates/amdgpu-tile`；[dsl-tile-abstractions.md](dsl-tile-abstractions.md) |
| **3.9 pass 工具** | 生成的 id 访问器（`IrIds`：所有 ValueId / VarId 的 walk）、`edit_stmts` / `parse_flat`（线性视图编辑后重建树）、`specialize`（标量 kernarg 常量化）、`regplan::region_report`（按区域的分配器真值） | `ir_codec.rs`（`tools/gen_ir_codec.py`）、`specialize.rs`、`regplan.rs`；[jit.md](jit.md) |

## 4. 部分采纳

1. **DSL 层算子属性化**：MIR 层已是属性表；DSL 的 `Op` / `Stmt` 仍是封闭枚举，类型规则（`result_ty`）、验证、lowering 各自 `match`。
   已有的一步：IR 文本的解码器与 id 访问器由生成器从枚举定义产出，加 op 时打印 / 解析 / α 规范化不需要手写分支（漏了重新生成会编译失败）。
2. **通用 walk / map**：有生成的 id 访问器，但没有 TIRx 那样的通用 `structural_map`；各 pass 仍自行递归区域树。
3. **命名物理轴的布局记法**（`lane` / `wave` / `m` 轴、显式副本）：未做；布局打印仍是 CuTe 式 shape : stride。
4. **符号分析的整除与迭代映射**：`amdgpu-sym` 有多项式、区间、位支撑与约束上下文，没有 TIRx 的整除推理与 iter-map 归一化；验证器靠区间 + 位集 + 契约事实足以证明出货 kernel。
5. **tile op 作为 IR 一等公民**：tile 操作在 Rust 里展开；IR 层能看到的 tile 粒度是 `Stage` 节点（带契约）和 tile 协议区域（`Region`），不是可分派的 tile 算子。`convert_layout` 的分档在 Rust 层由布局代数决定。

## 5. 有意不同

| 方面 | 我们的选择 | 原因 |
|---|---|---|
| 后端 | 自研调度（list / slotfill / 按区域 authored-free）、线性扫描 RA（AGPR、寄存器规划）、hazard 与 wait 插入、`hzfill`；LLVM 只作可选 post-RA 尾段 | AMD 上拿到厂商汇编级性能要直接控制 AGPR、`s_waitcnt`、MFMA/VALU 编排、M0/SCC；没有等价于 nvcc 的工具链可依赖 |
| 正确性 | 内建静态验证（越界、寄存器类、路径敏感等待、四轴协议、stage 契约）+ **严格证书**（信任账本逐项计数，`--strict` 有未证明项即失败）+ 功能证明 + 攻击用例回归（`verifier_cases.rs`） | 验证是编译的一部分而非外部工具；agent 的每个改动都必须拿到同样的证书 |
| 语言与安全 | 单语言 Rust，九个 crate 全部 `#![forbid(unsafe_code)]`、零三方依赖、无 FFI | 可审计、可复现；ISA 规则可编码进类型（safe-asm） |
| JIT | Python 包以**子进程**调 `amdc emit` / `jit-info` + ROCm clang，ctypes 加载 `.co`，磁盘缓存；不做 FFI 绑定 | 保持编译器零 FFI；编译期开销被缓存吸收 |
| 证据体系 | 同轮厂商对照的门禁（`port_gate` / `gemm_gate`）、出货 md5 不变式、8 布局中位数 A/B | 优化必须可证伪、可回归 |

## 6. 推迟

- **IKET 式 Perfetto 时间线**：stamps 已能按阶段 / 按 tile 拆周期，ATT 给逐指令视图；时间线 UI 推迟到做 **mono kernel**（多 stage 融合的单 kernel）时，按 stage 打点再接。
- **mono kernel**：stage 库与算子图已能表达多 stage 融合（融合 MLP、RMSNorm 量化、g1q epilogue 链）；整模型级的单 kernel 编排未开始。

## 7. 早期建议的落实情况

上一版本文（v7.1 时）§3 的五条建议：

| # | 建议 | 状态 |
|---|---|---|
| 1 | DSL 层算子属性化 | **未做**（MIR 层已有；DSL 层只把编解码与 id 访问生成化，见 §4.1） |
| 2 | tile op 进 IR（可分派） | **未做**；改走"算子图 → 骨架 → stage 实例"，IR 层以 `Stage` + `StageContract` 表达边界（§3.7、§4.5） |
| 3 | 命名物理轴的布局记法 | **未做** |
| 4 | IR 解析器与通用 walk / map | **解析器已做**（irv1 往返，§3.5）；walk 部分（生成的 id 访问器），通用 map 未做 |
| 5 | IKET 式时间线 | **推迟**（§6） |

## 8. TIRx 没有、我们有的

- 自研寄存器分配、调度与 wait 插入，以及在其上的寄存器规划（pin / separate / share / ring / 区域上限，按分配器真值报告）与按区域调度策略。
- 内建的静态协议验证、严格证书与攻击用例回归，功能等价证明（含"跳 tile"反例）。
- 出货 kernel 的 md5 不变式：IR 文本往返、重构、编排生成化都以"asm 逐字节不变"为验收。
- 同轮对比厂商的门禁与 8 布局中位数 A/B 协议。

## 9. 结论

结构化 IR 一侧，TIRx 的主要思想已经落地：区域 IR、算子属性表（MIR）、统一符号分析、结构相等与 α 规范化、IR 文本往返、显式编排与可选 tile 层。
仍未做的是 DSL 层属性化、通用 map、命名轴布局记法和 tile op 一等化——它们降低扩展成本，但不阻塞当前 kernel。
后端、验证、语言与 JIT 方式是有意不同的选择，由 AMD 目标与"可证伪、可回归"的工作方式决定。
