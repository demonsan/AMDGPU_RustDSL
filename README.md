# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950 / MI355X，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

kernel 作者与自动化 agent 在一个**类型化、可验证、可快速迭代**的 Rust DSL 里写 GPU 数据流；自研 compiler 把它降到内部
IR / MIR，自己完成**寄存器分配**与**跨基本块指令调度**，插入 hazard / wait，发射 `.s`，再由 ROCm 自带 `clang`
**只做汇编**得到 `.co`。LLVM 源码不改，LLVM 的设备端 CodeGen（RA、调度）不参与。

> **权威当前状态见 [`docs/current-status.md`](docs/current-status.md)**。本 README 只做入口与快速上手；
> 逐轮记录在 [`docs/status.md`](docs/status.md)（25 轮），厂商事实与认证表在 [`docs/vendor-spec.md`](docs/vendor-spec.md)。

---

## 当前状态（MI355X / gfx950 实测；出厂集 = v3 布局层 kernel，tag `v4.1-layout-p3` 之后）

| 项 | 结果 |
|---|---|
| **出厂 kernel 的写法（v3，布局层）** | 默认路径的 4 个 kernel（`fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}`，`crates/amdgpu-tile/src/fmha.rs`）建在 **布局代数层**上：`amdgpu-layout`（CuTe/FlyDSL 布局代数 + GF(2) swizzle + 构建期证明）、`amdgpu-isa`（gfx950 copy/MMA atom 表）、`amdgpu-tile`（TiledCopy/TiledMma/partition、DMA 分区、由布局导出的主机合同）——K/V/K-scale/Q 的每个地址、DMA、epilogue 存储和 KV-split 转置暂存均由布局推导；见 [`docs/dsl-layout-refactor-design.md`](docs/dsl-layout-refactor-design.md)。上一版 tile 库集 `fmha_fwd_hd128_mxfp8_dsl_*`（`fmha_fwd_hd128_mxfp8_dsl_auto`）与手写旧集（`fmha_fwd_hd128_mxfp8_legacy_auto`）保留供 A/B |
| **布局层（路线 B）** | `amdgpu-layout`（`Layout`/`IntTuple`/`Swizzle`/TV 布局/`IssuePlan`/构建期证明）→ `amdgpu-isa`（copy 与 MMA atom 表）→ `amdgpu-tile`（`TiledCopy`/`TiledMma`/`partition_*`、v3 FMHA）；入门教程 [`docs/layout-tour.md`](docs/layout-tour.md)，实施结果 [`docs/dsl-layout-refactor-design.md`](docs/dsl-layout-refactor-design.md) §9 |
| **主机侧生成** | `amdc gen-host <kernel>` / `--shipped`：从 kernel 合同生成 C++ 启动/ABI 头（打包 `Args`、`validate()`）与 JSON 元数据，出厂 `ref_host` 用生成的命名字段；见 [`docs/host-gen.md`](docs/host-gen.md) |
| **MXFP8 blockscale GEMM（P4）** | `gemm_mxfp8_bs_v3`（4096×4096×8192，对照 aiter PR #6696 4-wave pin-AGPR）：**_（待 kernel-opt 填写最终数字：TF/s、相对厂商、门禁）_** |
| **上一版（tile 库）** | 4 个 kernel（`fmha_fwd_hd128_mxfp8_dsl_{vs128,t128,kvsplit,kvsplit_combine8h}`）全部由 **tile 抽象库**（`crates/amdgpu-dsl/src/tile/`：张量视图+自动合同、LDS 计划/环、DMA copy、流水线、片段/MMA、在线 softmax、epilogue/combine、FMHA ABI 组件）组合而成：`crates/amdgpu-dsl/src/fmha_tile.rs` 每个 kernel 函数 25–45 行；原手写版（`fmha_fwd_hd128_mxfp8_{vs128,…}`）保留为 `fmha_fwd_hd128_mxfp8_legacy_auto` 供 A/B。见 [`docs/dsl-tile-abstractions.md`](docs/dsl-tile-abstractions.md)、[`docs/fmha-perf-patterns.md`](docs/fmha-perf-patterns.md) 附录 B |
| **基准形状 B1H8S2048**（默认 `fmha_fwd_hd128_mxfp8_auto` = KV-split + `combine8h`） | **674–705 TFLOP/s**；同轮厂商 536–539 → **1.25–1.31x 更快** |
| 单 kernel 每 CU（`_vs128`，64 blocks） | **453–455 TFLOP/s = 7.07–7.10 TF/CU**（厂商 8.41 → 1.19x 差距） |
| 铺满网格 B4H8S2048 / B1H8S8192 | **0.846x / 0.841x** 厂商（结构线已量化关闭） |
| 中等形状 | ≥ 厂商：B2H8S2048 1.00x、B1H8S4096 1.07x、B3H3S768 1.11x、B1H16S256 1.16x |
| 精度 | mean err **8.3e-4**（厂商 1.5–2.9e-3，好 2–3 倍）；15/15 形状、64 行 bad=0 |
| 确定性 | 5/5 × N；uneven-tile split 形状 **1200/1200 次启动单一 md5** |
| 测试 | `cargo test` **56 个目标 / 282 项全绿**（Main@7f937f8；`salu` 分支另加 `scalarize2` / `lds_b64` / `gv_bits` 三个目标）；布局层：`amdgpu-layout` 19 项（CuTe/FlyDSL 例子、`_vs128` 手写地址复现）、`amdgpu-tile` 地址等价测试（发射 IR 逐 lane 对照旧 reader/copy）；safe-asm 另有 29 个 ui 用例（28 编译失败 + 1 通过对照） |
| GPU 语义验证 | **编译器感知指令语义**：LDS/global/buffer 越界、VGPR/SGPR 类别与对齐、路径敏感等待、LDS tile 生命周期均被证明；出厂 4 kernel（`amdc gpu-verify --shipped --strict`）**0 错误 0 警告**（24 trusted assumes / 100 host contracts / 0 unproven）；手写旧集 `--legacy` 8 kernel 不变（44 / 216 / 0）；见 [`docs/gpu-semantics-verifier.md`](docs/gpu-semantics-verifier.md) |
| Rust 安全 | **七个 crate（含 amdgpu-layout / amdgpu-isa / amdgpu-tile）全部 `#![forbid(unsafe_code)]`、零第三方依赖、无 FFI/裸指针**；ISA 规则编码进类型系统；见 [`docs/rust-safety.md`](docs/rust-safety.md) |
| 正确性 | 本会话修复 3 个真实编译器/内核 bug（物理 WAW、两阶段 RA reaching-writer、**VMEM 宽存储数据 WAR**） |

功能：head dim 128，fp8(e4m3) Q/K/V，E8M0 1×32 Q/K 缩放，标量 V 缩放，BF16 输出，在线 softmax，
任意 query/key 长度（含不满一个 tile），GQA（2 的幂 ≤ 16），任意 batch/head。**与厂商 656 字节 kernarg ABI 和网格/数据布局逐字节兼容。**

---

## 快速上手

```bash
# 构建与测试（容器脚本会挂载 ROCm 与 pinned LLVM 工具）
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                    # 期望 56 targets / 282 tests 全绿（Main@7f937f8）

# 发射一个 kernel 的汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --out vs128.s
bash tools/container/cargo.sh run -q --bin amdc -- list            # 所有 kernel 名
bash tools/container/cargo.sh run -q --bin amdc -- emit <k> --backend own --sched ld

# 汇编成 .co
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 正确性 + 性能（同一输入同轮跑 我们 / 厂商 / 双精度 CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_WARMUP=150 REF_ITERS=100 ./build/ref_host <vendor.co>

# 确定性 / split 压力 / 厂商对照
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh        # fmha_fwd_hd128_mxfp8_legacy_auto = 旧手写集
bash tools/ohat_repro.sh                                                  # OHAT_PREFIX=fmha_fwd_hd128_mxfp8 = 旧手写集
HIP_VISIBLE_DEVICES=6 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3     # 全套验收（同轮对照旧手写集；SHIP_REF=fmha_fwd_hd128_mxfp8_dsl 对照 tile 库集）
bash tools/container/cargo.sh run -q --bin amdc -- gen-host --shipped --out tests/gpu/generated   # 生成主机 ABI 头 + JSON
HIP_VISIBLE_DEVICES=6 bash tools/gemm_gate.sh gemm_mxfp8_bs_v3   # GEMM 门禁（P4；BUILD/VERIFY/ACC/DET/PERF 对照厂商）
bash tools/vendor_validate.sh          # 厂商 kernel 复核（disasm + 计数 + 64 行 bad 检查）
bash tools/vendor_counters.sh          # 逐轮 PMC
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
crates/amdgpu-kernels   MXFP8 blockscale GEMM（P4，kernel-opt；随 integrate 合入）
crates/amdgpu-dsl       类型化 kernel DSL（Op 83 / Stmt 40 / FnBuilder 140 方法）+ tile 抽象库（src/tile/）+ 出厂 FMHA（src/fmha_tile.rs）+ 旧手写 kernel（src/kernels.rs）
crates/amdgpu-compiler  内部 IR/MIR、lowering、调度器、线性扫描 RA、hazard/wait、asm 发射、LLVM MIR 桥
crates/amdgpu-safeasm   手写热点窗口用的 safe Rust asm DSL（typed 寄存器 + 所有权 token + 验证器 + pin）
crates/amdc             CLI（emit / check / list / backend / simmodel / dump / probe / gpu-verify / gen-host）
tests/gpu/              端到端：ref_host（我们 vs 厂商 vs CPU）、run_ref、determinism、split 压力
tools/                  厂商对照、计数器、探针、A/B、LLVM 构建、容器 cargo 等约 26 个脚本
docs/                   设计、逐轮状态、厂商 spec、认证与归因（见下）
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/current-status.md`](docs/current-status.md) | **权威当前快照**（架构、DSL、kernel、数字、已关闭方向、复现命令） |
| [`docs/gpu-semantics-verifier.md`](docs/gpu-semantics-verifier.md) | **GPU 语义验证器**：global/LDS/VGPR/SGPR 越界与生命周期证明（地址模型、LDS tile 协议、寄存器规则、路径敏感等待、barrier 协议、trust 预算、诚实边界） |
| [`docs/gpu-semantics-audit.md`](docs/gpu-semantics-audit.md) | **独立对抗审计**：攻击表、发现的 soundness 漏洞与修复、最终 verdict（v4 四轴审计） |`n| [`docs/gpu-semantic-protocol.md`](docs/gpu-semantic-protocol.md) | **四轴协议模型（v4）**：到达/完成/可见/代际分开建模；多生产者/多消费者双向义务 O1–O4 的静态 discharge；全局内存 region + last-arriver 模式；信任账本 |
| [`docs/dsl-expression-layer.md`](docs/dsl-expression-layer.md) | **运算符表达式层**：地址/整数/浮点计算直接用 Rust 运算符书写，1:1 降低到现有 Op（逐字节相同的 asm）；以及为什么 fma/packed/fragment/协议仍需显式 |
| [`docs/fmha-dsl-tour.md`](docs/fmha-dsl-tour.md) | **FMHA 逐段教程**：用真实 kernel 讲每一段的 DSL 特性与 Rust safe 怎么保证（面向没写过 Rust kernel 的读者） |
| [`docs/rust-safety.md`](docs/rust-safety.md) | **Rust 安全性如何保证**（`#![forbid(unsafe_code)]`、零依赖、类型层 ISA 规则、验证器、panic 策略、安全边界） |
| [`docs/rust-safety-by-example.md`](docs/rust-safety-by-example.md) | **逐项 + 真实代码**：写 DSL kernel 时编译器如何保证 Rust safe（A1–A10），手写 asm 时 safe-asm 如何保证（B1–B16），29 个编译失败负例 |
| [`docs/design.md`](docs/design.md) | Rust DSL 与自研 compiler 设计规格 |
| [`docs/status.md`](docs/status.md) | 25 轮逐轮实现记录（所有正负结果 + 归因） |
| [`docs/vendor-spec.md`](docs/vendor-spec.md) | 厂商 kernel 事实 + §7.5–§7.11 认证表与归因 |
| [`docs/final-report.md`](docs/final-report.md) | 目标 → 现状 → 证据 → 差距 |
| [`docs/verification.md`](docs/verification.md) | 独立验证与正确性 bug 线 |
| [`docs/safe-asm-design.md`](docs/safe-asm-design.md) | safe-asm 设计与 4 轮对抗评审 |
| [`docs/kvsplit-design.md`](docs/kvsplit-design.md) / [`docs/kvsplit-results.md`](docs/kvsplit-results.md) | KV-split（默认路径之一） |
| 各代设计记录 | `kernel-gen2-design.md`、`gen3-design.md`、`gen4-tiling-design.md`、`gen5-window-spec.md`、`gen6-*.md`、`gen9-vendor-mirror-design.md`、`unscaled-pv-design.md`、`v-window-cut-plan.md`（均已关闭/落地，见文件头状态） |

## 边界与已知限制

- 基于 `gfx950`；`gfx1250` 只在架构层面预留（未验证）。
- `llvm20` 方言的表行是 `llvm24` 的拷贝，仅 pinned `llvm24` 的 `llc` 做过机器验证。
- DMA / LDS 直达 / `buffer_load … lds` 为 own-only；使用它们的 kernel 自动选 own 后端。
- `_t` kernel 有 14 条 loop-var lint 警告（wave 级分支的潜伏风险，已冻结并记录，不是活跃 bug）。
- `FMHA_OPT` 有 108 个实验 token；默认路径不含任何 opt-in 负结果项。

## 许可

内部研究项目；第三方 kernel（aiter `fwd_hd128_mxfp8.co`）仅用于对照，不随仓库分发。
