# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

本项目目标是让 kernel 作者与自动化 agent 能在一个**类型安全、可验证、可快速迭代**的模型里表达 GPU 数据流、资源生命周期和通信协议，同时绕开现有 LLVM AMDGPU 后端中低效或不满足需求的寄存器分配与指令调度。

## 状态

本仓库当前处于**设计基线阶段**。`docs/` 是可执行的规格，`crates/`、`llvm-bridge/`、`examples/`、`tests/` 将在后续提交中按 `docs/implementation-plan.md` 的里程碑落地。

本仓库**尚无可运行 compiler**，也未在 GPU 上验证任何候选 kernel。文档中所有性能、硬件语义和 1:1 复刻结论都标注了证据等级。

## 分支

- 主开发分支：`Main`（大写 M）。
- 所有 Rust DSL、内部 IR、compiler passes、LLVM bridge、测试和文档都在本仓库。

## 核心约束

1. **LLVM 源码不可修改。** 不应用 LLVM 补丁、不修改安装目录、不要求用户重建 LLVM。所有 LLVM 未提供或不满足合同的能力，由本项目的内部 compiler passes 实现。
2. **不复用 LLVM 的设备端 CodeGen 决策。** 保留自己的语义 IR、effects、RA、调度和最终验证；LLVM 仅作为 MIR/target/ABI/编码与发射的基础设施。
3. **安全不依赖 Drop。** Rust 的 `mem::forget` 可以跳过析构，因此 GPU completion、资源释放和协议义务由独立 ledger 与 IR verifier 检查。
4. **Rust 类型系统只负责构图期约束。** GPU 异步 hazard、跨设备可见性、alias、epoch 与硬件 encoding 由内部 verifier 证明。
5. **1:1 复刻是编译结果的验收约束。** 物理寄存器编号与指令顺序属于独立的 witness，不进入 kernel 源码。

## 非目标（当前阶段）

- 不实现完整的 Rust/MIR 前端或改写 rustc。
- 不实现通用 AMDGPU 后端以替代 LLVM。
- 不做完整框架移植（DeepInfer / SGLang 只提供薄适配）。
- 不承诺未经测量的 10ms 编译延迟已达标。

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/design.md`](docs/design.md) | Rust 安全前端、Semantic/Resource/Async IR、MachineDAG、Witness、LLVM 边界 |
| [`docs/fmha-reference.md`](docs/fmha-reference.md) | gfx950 MXFP8 FMHA 参考 kernel 的已核实事实与 1:1 验收方法 |
| [`docs/implementation-plan.md`](docs/implementation-plan.md) | 阶段划分、测试矩阵、未验证项 |

## 证据约定

文档中的事实按来源区分：

- **Verified**：由固定 commit 的源码、二进制解析或官方规范直接支持，并记录来源。
- **Experimental**：有源码依据但未在目标硬件上验证。
- **Unknown / 待确认**：无充分证据，禁止在实现中当作契约。

## 许可

本仓库尚未声明许可证；在明确决定前不复制第三方实现。

## Status

The compiler pipeline works end to end: a typed Rust DSL lowers to internal MIR, goes through this project's own register allocator and assembly emitter, and produces gfx950 code objects that run on hardware. The sample ttention_fwd kernel is numerically verified against a CPU reference on an MI355X (max abs error ~1e-4). See [docs/status.md](docs/status.md) for evidence and [docs/design.md](docs/design.md) for the architecture; the next milestone is the 1:1 FMHA reference core loop.
