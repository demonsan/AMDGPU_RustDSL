# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950 / MI355X，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

kernel 作者与自动化 agent 在一个**类型化、可验证、可快速迭代**的 Rust DSL 里写 GPU 数据流；自研 compiler 把它降到内部
IR / MIR，自己完成**寄存器分配**与**跨基本块指令调度**，插入 hazard / wait，发射 `.s`，再由 ROCm 自带 `clang`
**只做汇编**得到 `.co`。LLVM 源码不改，LLVM 的设备端 CodeGen（RA、调度）不参与。

> **权威当前状态见 [`docs/current-status.md`](docs/current-status.md)**。本 README 只做入口与快速上手；
> 厂商事实与认证表在 [`docs/vendor-spec.md`](docs/vendor-spec.md)；历史设计研究与逐轮记录归档在 [`docs/archive/`](docs/archive/README.md)。

---

## 当前状态（MI355X / gfx950 实测；tag `v5.0-route-b`）

| 项 | 结果 |
|---|---|
| **出厂 FMHA（v3，布局层）** | 默认路径 `fmha_fwd_hd128_mxfp8_auto` 的 4 个 kernel `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}`（`crates/amdgpu-kernels/src/fmha/mod.rs`）建在布局层上：每个 LDS/全局地址、DMA、epilogue 存储与 KV-split 转置暂存都由 `amdgpu-layout` 的布局推导，主机合同由张量布局导出。asm md5（`--backend own`）：`68d2832b` / `e4c7e75d` / `e4fc18b1` / `24a95a51`。上一版 tile 库集 `fmha_fwd_hd128_mxfp8_dsl_*`（`fmha_fwd_hd128_mxfp8_dsl_auto`）与手写旧集（`fmha_fwd_hd128_mxfp8_legacy_auto`）保留供 A/B |
| **布局层（路线 B）** | `amdgpu-layout`（`Layout`/`IntTuple`/`Swizzle`/TV 布局/`IssuePlan`/构建期证明）→ `amdgpu-isa`（copy 与 MMA atom 表）→ `amdgpu-tile`（`TiledCopy`/`TiledMma`/`partition_*`）；入门 [`docs/layout-tour.md`](docs/layout-tour.md)，实施记录 [`docs/dsl-layout-refactor-design.md`](docs/dsl-layout-refactor-design.md) §9 |
| **MXFP8 blockscale GEMM** | `gemm_mxfp8_bs_v4`（`crates/amdgpu-kernels`，4096×4096×8192，对照 aiter PR #6696 4-wave pin-AGPR）：**3219 vs 3269 TF/s = 0.985×**（v5.4 门禁，同轮交替；演进与证据见 [`docs/mxfp8-gemm-4wave-analysis.md`](docs/mxfp8-gemm-4wave-analysis.md) §12–§14）；`tools/gemm_gate.sh` PASS（strict 0/0、3 形状 GEMM_OK、确定性 3/3）。AGPR（累加寄存器文件）已支持，见 [`docs/agpr.md`](docs/agpr.md) |
| **主机侧生成** | `amdc gen-host <kernel>` / `--shipped`：从 kernel 合同生成 C++ 启动/ABI 头（打包 `Args`、`validate()`）与 JSON 元数据，出厂 `ref_host` 用生成的命名字段；见 [`docs/host-gen.md`](docs/host-gen.md) |
| **FMHA 性能（最近一次 `port_gate`，v3 vs 同轮旧手写集）** | B1H8S2048 默认路径（KV-split + `combine8h`）**672.7 TF/s**（同轮厂商约 536–539，≈1.25×）；`_vs128` 每 CU 64 blocks **446.4**，8 种代码布局中位数 **449.9 vs 453.7**；B4H8S2048 **1421.3**、B1H8S8192 **1978.6**（铺满网格约 0.84× 厂商）；B2H8S2048 `_t128` **1025.0**，布局中位数 **1022.3 vs 1028.6** |
| 精度 | mean err **8.3e-4**（厂商 1.5–2.9e-3）；15/15 形状、64 行 bad=0 |
| 确定性 | 5/5 × N；uneven-tile split 形状重复启动单一 md5 |
| 测试 | `cargo test` **82 个目标 / 362 项全绿**；safe-asm 另有 29 个 ui 用例（28 编译失败 + 1 通过对照） |
| GPU 语义验证 | 编译器感知指令语义：LDS/global/buffer 越界、VGPR/SGPR/AGPR 类别与对齐、路径敏感等待、LDS tile 生命周期与同步协议均被证明。严格证书：出厂 4 kernel `--shipped` **24 trusted assumes / 100 host contracts / 0 unproven**；旧手写集 `--legacy`（已移除） **44 / 216 / 0**；GEMM v3/v4 **3 / 19 / 0**；见 [`docs/gpu-semantics-verifier.md`](docs/gpu-semantics-verifier.md) |
| Rust 安全 | **八个 crate 全部 `#![forbid(unsafe_code)]`、零第三方依赖、无 FFI/裸指针**；ISA 规则编码进类型系统；见 [`docs/rust-safety.md`](docs/rust-safety.md) |

功能（FMHA）：head dim 128，fp8(e4m3) Q/K/V，E8M0 1×32 Q/K 缩放，标量 V 缩放，BF16 输出，在线 softmax，
任意 query/key 长度（含不满一个 tile），GQA（2 的幂 ≤ 16），任意 batch/head。**与厂商 656 字节 kernarg ABI 和网格/数据布局逐字节兼容。**

---

## 快速上手

```bash
# 构建与测试（容器脚本会挂载 ROCm 与 pinned LLVM 工具）
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                    # 期望 82 targets / 362 tests 全绿

# 发射一个 kernel 的汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --out vs128.s
bash tools/container/cargo.sh run -q --bin amdc -- list            # 所有 kernel 名
bash tools/container/cargo.sh run -q --bin amdc -- emit <k> --backend own --sched ld

# 汇编成 .co
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# GPU 语义证书（严格：任一 Error → 退出码 1）
bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify --shipped --strict

# 主机侧 ABI 头 + JSON
bash tools/container/cargo.sh run -q --bin amdc -- gen-host --shipped --out tests/gpu/generated

# 正确性 + 性能（同一输入同轮跑 我们 / 厂商 / 双精度 CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_WARMUP=150 REF_ITERS=100 ./build/ref_host <vendor.co>

# 门禁
HIP_VISIBLE_DEVICES=6 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3     # FMHA 全套验收（同轮对照旧手写集；SHIP_REF=fmha_fwd_hd128_mxfp8_dsl 对照 tile 库集）
HIP_VISIBLE_DEVICES=6 bash tools/gemm_gate.sh gemm_mxfp8_bs_v4            # GEMM（BUILD / VERIFY / ACC / DET / PERF 对照厂商）

# 确定性 / split 压力 / 厂商对照
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh        # fmha_fwd_hd128_mxfp8_legacy_auto = 旧手写集
bash tools/ohat_repro.sh                                                  # OHAT_PREFIX=fmha_fwd_hd128_mxfp8 = 旧手写集
bash tools/vendor_validate.sh          # 厂商 kernel 复核（disasm + 计数 + 64 行 bad 检查）

# 文档链接检查
python tools/check_doc_links.py
```

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`）。ROCm 10 不自带 `llc`，用
`tools/llvm_build_llc.sh` 从同一 commit 链接 `libLLVM.so` 得到。ROCm 7.0.2（方言 `llvm20`）可通过
`ROCM_LLVM_BIN` / `AMDC_DIALECT` 选用。pin 缺失时 **llvm-tail 直接报错**，不静默降级（`--backend=own` 用完整自研 tail）。

---

## 目录

```
crates/amdgpu-layout    布局代数（零依赖）：Ix / IntTuple / Layout / Swizzle / TV 布局 / IssuePlan / 构建期证明
crates/amdgpu-isa       gfx950 atom 表：copy atom（宽度、立即数域、计数器、LDS-direct 形式）与 MMA atom（A/B/C 的 TV 布局、provenance）
crates/amdgpu-tile      TiledCopy / TiledMma / partition_{lds,dma,buffer_dma,global}、布局导出的主机合同、出厂 v3 FMHA（src/fmha.rs）
crates/amdgpu-kernels   MXFP8 blockscale GEMM（gemm_mxfp8_bs_v3 / _v4）
crates/amdgpu-dsl       类型化 kernel DSL（Op 84 / Stmt 47 / FnBuilder 约 160 个方法）+ 表达式层（src/expr.rs）+ tile 库（src/tile/）+ dsl FMHA（src/fmha_tile.rs）+ 旧手写 kernel（src/kernels.rs）
crates/amdgpu-compiler  内部 MIR（123 个 MOp）、lowering、调度器、线性扫描 RA（含 AGPR）、hazard/wait、asm 发射、LLVM MIR 桥、gpu_verify、host-gen
crates/amdgpu-safeasm   手写热点窗口用的 safe Rust asm DSL（typed 寄存器 + 所有权 token + 验证器 + pin）
crates/amdc             CLI（emit / check / list / dump / backend / simmodel / probe / gpu-verify / gen-host）
tests/gpu/              端到端：ref_host（我们 vs 厂商 vs CPU）、run_ref、determinism、split 压力、gemm_ref_host、生成的主机 ABI（generated/）
tools/                  门禁（port_gate / gemm_gate）、厂商对照、计数器与 ATT、探针、A/B、LLVM 构建、容器 cargo、文档链接检查
docs/                   当前状态、设计、教程、验证器、厂商 spec；历史文档在 docs/archive/
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/current-status.md`](docs/current-status.md) | **权威当前快照**（架构、DSL、kernel、数字、证书、已关闭方向、复现命令） |
| [`docs/design.md`](docs/design.md) | Rust DSL 与自研 compiler 设计规格 |
| [`docs/op-attributes.md`](docs/op-attributes.md) | **算子属性表**：每个 MOp 的效果/hazard/调度代价/AGPR/后端等属性单一来源（穷尽 match），各 pass 查表而非枚举算子 |
| [`docs/region-ir.md`](docs/region-ir.md) | **区域化 IR**：函数体为 Loop/If/Region/Stage 树，旧标记序列为派生线性化，配对错误构造期即不可能 |
| [`docs/sym.md`](docs/sym.md) | **统一符号分析器** `amdgpu-sym`：合同/布局/验证器共用的多项式、约束上下文、区间与位集；验证器按区域树遍历并带入分支事实 |
| [`docs/slotfill.md`](docs/slotfill.md) | **填空档调度器**：MFMA 骨架 + 截止时间队列 + 周期预算 + 寄存器压力模型；GEMM v4 默认，取代手工搬数布置 |
| [`docs/ir-printing.md`](docs/ir-printing.md) | **IR 打印与结构相等**：`amdc print-ir --stage dsl|mir|mir-ra`，α-等价比较/哈希 |
| [`docs/functional-proofs.md`](docs/functional-proofs.md) | **功能证明层**：GEMM 索引级证明（含对已构建 IR 的符号执行 E3 与"跳 tile"反例）、FMHA online softmax 递推的实数证明 |
| [`docs/fusion-layout-handoff.md`](docs/fusion-layout-handoff.md) | **融合边界**：`convert_layout` 按布局代数分档、MMA 方向搜索（自动推出 FMHA S→P）、`Node::Stage` 资源契约与 GV-STAGE、融合 MLP 示例 |
| [`docs/stamps.md`](docs/stamps.md) | **in-kernel 时间戳**（`s_memtime`）：按阶段/按 tile 拆分 kernel 周期 |
| [`docs/mfma-ceiling.md`](docs/mfma-ceiling.md) | **MFMA 发射上限微基准**：各操作数形式 32.25 周期/条，真实数据下的时钟与功耗墙 |
| [`docs/ir-review-tirx-sftensor.md`](docs/ir-review-tirx-sftensor.md) | **IR 梳理**：对照 TIRx（算子属性表、区域语句、统一符号分析、打印/结构相等、命名轴 layout）与 SF Tensor 形式化正确性，提出属性表 / 区域化 IR / 统一 sym / 功能证明层的改造顺序 |
| [`docs/layout-tour.md`](docs/layout-tour.md) | **布局层入门**：`Layout` / `Swizzle` / TV 布局 / `IssuePlan` / 证明，用出厂 FMHA 与 GEMM 的真实代码讲 |
| [`docs/fmha-dsl-tour.md`](docs/fmha-dsl-tour.md) | **FMHA 逐段教程**：每一段的 DSL 特性与 Rust safe 怎么保证 |
| [`docs/dsl-layout-refactor-design.md`](docs/dsl-layout-refactor-design.md) | 路线 B（布局代数重构）设计与 §9 as-built |
| [`docs/dsl-tile-abstractions.md`](docs/dsl-tile-abstractions.md) | tile 库（L1–L6）设计；哪些层已被布局层替代 |
| [`docs/dsl-expression-layer.md`](docs/dsl-expression-layer.md) | 运算符表达式层 |
| [`docs/fmha-perf-patterns.md`](docs/fmha-perf-patterns.md) | FMHA 性能承重模式清单（移植 parity checklist） |
| [`docs/gpu-semantics-verifier.md`](docs/gpu-semantics-verifier.md) | **GPU 语义验证器**：地址模型、LDS tile 协议、寄存器规则、路径敏感等待、trust 预算、诚实边界 |
| [`docs/gpu-semantic-protocol.md`](docs/gpu-semantic-protocol.md) | 四轴协议模型（到达 / 完成 / 可见 / 代际） |
| [`docs/last-arriver-memory-model.md`](docs/last-arriver-memory-model.md) | last-arriver 融合 combine 的内存序原语 |
| [`docs/agpr.md`](docs/agpr.md) | AGPR（累加寄存器）支持：寄存器类、acc_cd MFMA、预算、hazard |
| [`docs/host-gen.md`](docs/host-gen.md) / [`docs/hostgen-json-schema.md`](docs/hostgen-json-schema.md) | 主机侧生成（C++ 头 + JSON） |
| [`docs/rust-safety.md`](docs/rust-safety.md) / [`docs/rust-safety-by-example.md`](docs/rust-safety-by-example.md) | Rust 安全：原则与边界 / 逐项真实代码 |
| [`docs/safe-asm-design.md`](docs/safe-asm-design.md) | safe-asm 设计与对抗评审 |
| [`docs/scheduling.md`](docs/scheduling.md) | 指令调度器 |
| [`docs/llvm-tail-integration.md`](docs/llvm-tail-integration.md) / [`docs/llvm-tail-conformance.md`](docs/llvm-tail-conformance.md) / [`docs/llvm-feature-audit.md`](docs/llvm-feature-audit.md) | LLVM tail 集成、一致性与 CodeGen 测试覆盖审计 |
| [`docs/vendor-spec.md`](docs/vendor-spec.md) / [`docs/fmha-reference.md`](docs/fmha-reference.md) | 厂商 FMHA kernel 事实、认证表与参考 |
| [`docs/mxfp8-gemm-4wave-analysis.md`](docs/mxfp8-gemm-4wave-analysis.md) / [`docs/gemm-issue-comparison.md`](docs/gemm-issue-comparison.md) | 厂商 4-wave GEMM 分析与逐指令对比 |
| [`docs/archive/README.md`](docs/archive/README.md) | **历史归档**：逐轮实现记录（`status.md`）、各代设计研究（gen2–gen9）、KV-split 设计与结果、第 75 轮终局报告、独立审计记录等 |

## 边界与已知限制

- 基于 `gfx950`；`gfx1250` 只在架构层面预留（未验证）。
- `llvm20` 方言的表行是 `llvm24` 的拷贝，仅 pinned `llvm24` 的 `llc` 做过机器验证。
- DMA / LDS 直达 / `buffer_load … lds` 等为 own-only；使用它们的 kernel 自动选 own 后端。
- `_t` kernel 有 14 条 loop-var lint 警告（wave 级分支的潜伏风险，已冻结并记录，不是活跃 bug）。
- `FMHA_OPT` 实验 token 都是 opt-in（`FMHA_OPT_SCOPE` 可把它们限定到别名 kernel 做同轮 A/B）；默认路径不含任何 opt-in 负结果项。

## 许可

内部研究项目；第三方 kernel（aiter `fwd_hd128_mxfp8.co`、aiter GEMM code object）仅用于对照，不随仓库分发。
