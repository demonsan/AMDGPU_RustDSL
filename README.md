# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

kernel 作者和自动化 agent 在一个**类型化、可验证、可快速迭代**的 Rust DSL 里写 GPU 数据流；自研 compiler 把它降到内部 SSA / MIR，自己做寄存器分配、hazard / wait 插入和汇编发射，输出 `.s`，再由 ROCm 自带的 `clang` **只做汇编**得到 `.co`。LLVM 源码不修改，LLVM 的设备端 CodeGen（寄存器分配、调度）不参与。

## 当前状态（已在 MI355X / gfx950 真机上验证）

| 内容 | 状态 |
|---|---|
| Rust 类型化 DSL、IR 验证器、MIR、线性扫描 RA、hazard / wait 插入、asm 发射 | 已实现，`cargo test` 11 项 |
| 端到端：DSL → `.s` → `.co` → HIP 启动 → 数值比对 | 已打通，见 `tests/gpu/` |
| `attention_fwd`（fp16，示例） | 对 CPU 参考最大绝对误差 ≈ 6e-5 |
| MXFP8 FMHA：`fmha_fwd_hd128_mxfp8_t` | 与厂商 `fwd_hd128_mxfp8.co` 使用**相同的 656 字节 kernarg ABI 和数据布局**，15 个形状逐元素对拍通过；B1H8S2048 上 291 TFLOP/s（llvm-tail；厂商 514，差 1.76 倍；快速路径 `fmha_fwd_hd128_mxfp8_vs128` 见下），对 CPU 的平均误差 1.25e-3（优于厂商 2.42e-3） |

`fmha_fwd_hd128_mxfp8_t` 的功能：head dim 128，fp8(e4m3) Q/K/V，E8M0 1×32 Q/K 缩放，标量 V 缩放，BF16 输出，`exp2` softmax，在线 softmax，任意 query / key 长度（含不满一块的 tile），GQA（2 的幂 ≤ 16），任意 batch / head 数。不支持 LSE 输出。

对拍结果（`tests/gpu/run_ref.sh`，同一输入同时运行我们的 kernel、厂商 kernel 和双精度 CPU 模型）：

- 我们 vs CPU：平均误差 1.1–1.8e-3（厂商 kernel vs CPU 为 1.8–2.8e-3）
- 我们 vs 厂商：全部输出逐个比较，平均差异 1.3–2.8e-3；Q = 0 与 SK = 1 时逐位相同
- 差异来源：P 量化到 fp8 的舍入方式不同（固有）

**性能（如实）**：B=1, H=8, S=SK=2048 上默认的精确 recipe kernel `fmha_fwd_hd128_mxfp8_t` **367 TFLOP/s**（`fmha_fwd_hd128_mxfp8_vs128`，own 后端：DMA 暂存；厂商结构）／**291**（`fmha_fwd_hd128_mxfp8_t` 精确 recipe，llvm-tail）（默认对该 kernel 启用 scheduler 的 `ld` 预设：scoreboard 部分 wait + `ds_read` 提升，`AMDC_SCHED=0` 可关）（与 CPU 的 mean err 1.25e-3，优于厂商 2.42e-3；15 个形状在两个后端均通过，确定性 5/5），厂商 kernel 约 515 TFLOP/s，**慢约 1.40 倍**。opt-in 实验：`_sv`（厂商式 softmax，加 svcorr 后 15/15，254 TFLOP/s，mean err 1.75e-3；不加 svcorr 285 TFLOP/s 但 13/15 形状有离群点）、`_pp`（ping-pong，205）、`FMHA_OPT=dma`（global->LDS DMA，仅 own 后端，H>=2 仍错误，未启用）。详见 `docs/status.md` 第 44/45 轮（占用率、barrier、swizzle 等负结果）。 DMA staging（`FMHA_OPT=dma`，仅 own 后端）已修好（原 H>=2 错误是 own 后端寄存器分配器 SGPR 计数 bug，见 `docs/status.md` 第 46 轮），在默认 scheduler 下无额外收益，保持 opt-in。

**不是 1:1 复刻**：ABI、网格 / block 形状、1×32 缩放用法、`v_mfma_scale_f32_32x32x64_f8f6f4`、`ds_read_b64_tr_b8`、A/B 两组 wave 的分工都和参考 kernel 一致，但指令级结构不同（P 量化用 `v_cvt_pk_fp8_f32`，参考用 pknorm + scalef32 pack；PV 用 32×32×64 MFMA，参考用 `v_smfmac_f32_16x16x128_fp8_fp8`；每轮 1 个 barrier（双缓冲），参考 4 个 phase；没有 rollback 出口）。对照表见 `docs/final-report.md`。

## 后端分工：默认 LLVM tail（第 3 轮）

`amdc emit` 的**默认后端是 `llvm-tail`**（`--backend=own` 或环境变量 `BACKEND=own` 回到完整的自研 tail；`AMDC_BACKEND` 也可设置）。分工如下：

| 阶段 | 谁负责 |
|---|---|
| DSL、lowering、MIR、**寄存器分配**、**指令调度**（块内 list scheduler、swp、taildup、`--sched` / `--sched=auto`） | **始终是我们自己的**（两个后端完全一致） |
| 硬件 wait（`s_waitcnt`）、hazard nop / `s_delay_alu`、memory legalizer、kernel descriptor / metadata | `llvm-tail`：我们导出 post-RA MIR（物理寄存器、已排好序），`llc -start-before=gcn-create-vopd` 只跑 tail 的 pass；`own`：我们自己的实现 |

LLVM 不改源码；pin 的版本记录在 `backend.lock`（ROCm 10.1.0，AMD LLVM 24.0.0git；ROCm 10 不带 `llc`，用 `tools/llvm_build_llc.sh` 从同一 commit 的 `llc.cpp` 链接其 `libLLVM.so` 得到）。找不到 pin 的工具目录时 `llvm-tail` **直接报错**并建议 `--backend=own`，不会静默换成别的 LLVM。ROCm 7.0.2（dialect `llvm20`）仍可通过 `ROCM_LLVM_BIN` / `AMDC_DIALECT` 选用。

**“RA 和调度仍然是我们的”有自动化回归保护**：`crates/amdgpu-compiler/tests/tail_invariance.rs`（`cargo test`）对每个 kernel 同时用 own 与 llvm-tail 编译，要求指令流（opcode、物理寄存器、顺序）在忽略 `s_nop` / `s_waitcnt` / `s_delay_alu` / fallthrough `s_branch` 与等价的操作数写法后**完全一致**，导出的 MIR 仅含物理寄存器（`noVRegs: true`），descriptor 与我们的一致；可在 `AMDC_SCHED=1`、`swp`、`taildup,swp`、`auto` 下重复。详见 `docs/llvm-tail-integration.md` §9 与 `docs/llvm-tail-conformance.md`。
## 目录

| 路径 | 内容 |
|---|---|
| `crates/amdgpu-dsl` | DSL（`Op` / `Stmt` / `FnBuilder`）、类型检查与验证器、所有 kernel（`kernels.rs`） |
| `crates/amdgpu-compiler` | lowering → MIR、coalescing、DCE、hazard / wait pass、线性扫描 RA、asm emit、`amdc` 命令行 |
| `tests/gpu` | HIP 主机程序（`attn_host.cpp`、`ref_host.cpp`）与 `run.sh`、`run_ref.sh` |
| `tools/container` | 在容器里运行 `cargo` 的脚本（`FMHA_ABLATE` 调试开关会被传入） |
| `docs` | `final-report.md`（总结与对照）、`status.md`（按轮次的验证记录与硬件事实）、`design.md`、`fmha-reference.md`、`implementation-plan.md` |

## 如何复现

```bash
# 编译器测试（容器内，不需要 GPU）
bash tools/container/cargo.sh test

# 示例 attention_fwd：DSL -> asm -> .co -> GPU -> CPU 对比（需要 gfx950 与 ROCm）
bash tests/gpu/run.sh

# 与厂商 fwd_hd128_mxfp8.co 逐元素对拍（厂商 .co 不在仓库里，需自行取得 aiter 提交 7061cbd 的该文件）
REF_CO=/path/to/fwd_hd128_mxfp8.co OUR_KERNEL=fmha_fwd_hd128_mxfp8_t bash tests/gpu/run_ref.sh
```

`amdc emit <kernel> --out x.s` 可单独生成汇编；`amdc` 的 kernel 列表见 `crates/amdgpu-compiler/src/lib.rs` 的 `kernel_names()`。

## 核心约束（保持不变）

1. **LLVM 源码不可修改**：不打补丁、不改安装目录。LLVM 没有的能力由内部 pass 实现。
2. **不复用 LLVM 的设备端 CodeGen 决策**：自己的 SSA / MIR、RA、wait / hazard 插入和 asm 发射；LLVM 只做汇编。
3. **证据约定**：文档里的结论分 Verified（真机或固定 commit 的源码 / 二进制直接支持）、Experimental、Unknown。性能与数值结论都来自 `tests/gpu` 里的可复现命令。

## 已知限制

- 调度能力：块内 list scheduling、超块尾部复制（`taildup`）、单块循环软件流水（`swp`）、`ds_read` 提升（`ldshoist`）；默认对 FMHA kernel 开 `ld` 预设，其余默认关闭。仍没有全局（跨函数/跨 region）调度。\n- 性能：`fmha_fwd_hd128_mxfp8_vs128`（厂商结构：2 barrier/128 键、1 次 rollback、DMA 暂存、128 键单位）约 **367 TFLOP/s**（own 后端）；`fmha_fwd_hd128_mxfp8_t`（精确 recipe）约 290（llvm-tail）/ 280（own）；厂商手写 kernel 约 515，差约 1.43 倍；剩余差距主要是 staging+barrier、exp 与惰性重缩放（详见 `docs/status.md` 第 44/45 轮）。
- 没有跨卡通信（MORI LSA / SDMA / GDA）、gfx1250 后端、JIT / 框架适配的实现；`docs/design.md` 里这些仍是设计。
- 设计文档中的 Resource / Async IR 只落地了一部分（LDS 区域、待完成加载的 wait 跟踪）。
- 没有测量 ≤10 ms 的编译延迟，不做这个声明。

## 最终 FMHA 架构（第 57 轮）
- **快速路径** `fmha_fwd_hd128_mxfp8_vs128`（仅 own 后端：global->LDS DMA 暂存）：厂商结构（128 键单位、每 128 键 2 个 barrier、DMA 暂存、双 wave 组互补相位）+ 精确参考 recipe（每单位重对齐的 rollback 阈值 120/512、两字 f16 bias、长序列不用 LUT 修正、短序列 <9 tile 用 LUT 修正的第二份代码）+ smfmac 行和 + Q 常驻寄存器。B1H8S2048 **367 TFLOP/s**（厂商 ~514，差 1.40 倍），对 CPU 平均误差 6.9e-4（厂商 2.42e-3），15/15 形状、64 行下 bad=0、确定性 5/5。
- **精确默认** `fmha_fwd_hd128_mxfp8_t`（llvm-tail 可用）：292 TFLOP/s，mean err 1.25e-3。
- **实验** `fmha_fwd_hd128_mxfp8_g2`（仅按名字注册，own 后端）：QK(t+1) 放进 V 相位（3-slot ring），输出与 `_vs128` 逐位一致，333/344 TFLOP/s（低于 `_vs128`）；结论与第二代前置条件见 `docs/kernel-gen2-design.md` §6-7 与 `docs/status.md` 第 57 轮。
- **结论**：FMHA 快速路径为 367 TFLOP/s = 距厂商 1.40 倍；其后测过的所有调度/结构变体均为负或在噪声内，文档化的后续工作需要算法/分块（tile）层面的改变（tile 几何、基于 AGPR 的 512 寄存器 wave、不同的 softmax recipe），而不是调度（见 `docs/gen3-design.md` 第 7-8 节与 `docs/status.md` 第 58 轮）。
