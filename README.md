# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950 / MI355X，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

kernel 作者与自动化 agent 在一个**类型化、可验证、可快速迭代**的 Rust DSL 里写 GPU 数据流；自研 compiler 把它降到内部
IR / MIR，自己完成**寄存器分配**与**跨基本块指令调度**，插入 hazard / wait，发射 `.s`，再由 ROCm 自带 `clang`
**只做汇编**得到 `.co`。LLVM 源码不改，LLVM 的设备端 CodeGen（RA、调度）不参与。

> **权威当前状态见 [`docs/current-status.md`](docs/current-status.md)**。本 README 只做入口与快速上手；
> 逐轮记录在 [`docs/status.md`](docs/status.md)（25 轮），厂商事实与认证表在 [`docs/vendor-spec.md`](docs/vendor-spec.md)。

---

## 当前状态（MI355X / gfx950 实测，`Main` = tag `v2.2-round75-final`）

| 项 | 结果 |
|---|---|
| **基准形状 B1H8S2048**（默认 `fmha_fwd_hd128_mxfp8_auto` = KV-split + `combine8h`） | **674–705 TFLOP/s**；同轮厂商 536–539 → **1.25–1.31x 更快** |
| 单 kernel 每 CU（`_vs128`，64 blocks） | **453–455 TFLOP/s = 7.07–7.10 TF/CU**（厂商 8.41 → 1.19x 差距） |
| 铺满网格 B4H8S2048 / B1H8S8192 | **0.846x / 0.841x** 厂商（结构线已量化关闭） |
| 中等形状 | ≥ 厂商：B2H8S2048 1.00x、B1H8S4096 1.07x、B3H3S768 1.11x、B1H16S256 1.16x |
| 精度 | mean err **8.3e-4**（厂商 1.5–2.9e-3，好 2–3 倍）；15/15 形状、64 行 bad=0 |
| 确定性 | 5/5 × N；uneven-tile split 形状 **1200/1200 次启动单一 md5** |
| 测试 | `cargo test` **28 个目标 / 111 项全绿**；safe-asm 另有 29 个 ui 用例（28 编译失败 + 1 通过对照） |
| 正确性 | 本会话修复 3 个真实编译器/内核 bug（物理 WAW、两阶段 RA reaching-writer、**VMEM 宽存储数据 WAR**） |

功能：head dim 128，fp8(e4m3) Q/K/V，E8M0 1×32 Q/K 缩放，标量 V 缩放，BF16 输出，在线 softmax，
任意 query/key 长度（含不满一个 tile），GQA（2 的幂 ≤ 16），任意 batch/head。**与厂商 656 字节 kernarg ABI 和网格/数据布局逐字节兼容。**

---

## 快速上手

```bash
# 构建与测试（容器脚本会挂载 ROCm 与 pinned LLVM 工具）
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                    # 期望 28 targets / 111 tests 全绿

# 发射一个 kernel 的汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_vs128 --out vs128.s
bash tools/container/cargo.sh run -q --bin amdc -- list            # 所有 kernel 名
bash tools/container/cargo.sh run -q --bin amdc -- emit <k> --backend own --sched ld

# 汇编成 .co
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 正确性 + 性能（同一输入同轮跑 我们 / 厂商 / 双精度 CPU 参考）
REF_ROWS=64 OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto REF_CO=<vendor.co> bash tests/gpu/run_ref.sh
REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_WARMUP=150 REF_ITERS=100 ./build/ref_host <vendor.co>

# 确定性 / split 压力 / 厂商对照
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh
bash tools/ohat_repro.sh fmha_fwd_hd128_mxfp8_kvsplit
bash tools/vendor_validate.sh          # 厂商 kernel 复核（disasm + 计数 + 64 行 bad 检查）
bash tools/vendor_counters.sh          # 逐轮 PMC
```

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`）。ROCm 10 不自带 `llc`，用
`tools/llvm_build_llc.sh` 从同一 commit 链接 `libLLVM.so` 得到。ROCm 7.0.2（方言 `llvm20`）可通过
`ROCM_LLVM_BIN` / `AMDC_DIALECT` 选用。pin 缺失时 **llvm-tail 直接报错**，不静默降级（`--backend=own` 用完整自研 tail）。

---

## 目录

```
crates/amdgpu-dsl       类型化 kernel DSL（Op 83 / Stmt 40 / FnBuilder 140 方法）+ 全部 kernel
crates/amdgpu-compiler  内部 IR/MIR、lowering、调度器、线性扫描 RA、hazard/wait、asm 发射、LLVM MIR 桥
crates/amdgpu-safeasm   手写热点窗口用的 safe Rust asm DSL（typed 寄存器 + 所有权 token + 验证器 + pin）
crates/amdc             CLI（emit / check / list / backend / simmodel / dump / probe）
tests/gpu/              端到端：ref_host（我们 vs 厂商 vs CPU）、run_ref、determinism、split 压力
tools/                  厂商对照、计数器、探针、A/B、LLVM 构建、容器 cargo 等约 26 个脚本
docs/                   设计、逐轮状态、厂商 spec、认证与归因（见下）
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/current-status.md`](docs/current-status.md) | **权威当前快照**（架构、DSL、kernel、数字、已关闭方向、复现命令） |
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
