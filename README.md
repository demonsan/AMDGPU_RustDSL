# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

kernel 作者和自动化 agent 在一个**类型化、可验证、可快速迭代**的 Rust DSL 里写 GPU 数据流；自研 compiler 把它降到内部 SSA / MIR，自己做寄存器分配、hazard / wait 插入和汇编发射，输出 `.s`，再由 ROCm 自带的 `clang` **只做汇编**得到 `.co`。LLVM 源码不修改，LLVM 的设备端 CodeGen（寄存器分配、调度）不参与。

## 当前状态（已在 MI355X / gfx950 真机上验证）

| 内容 | 状态 |
|---|---|
| Rust 类型化 DSL、IR 验证器、MIR、线性扫描 RA、hazard / wait 插入、asm 发射 | 已实现，`cargo test` 11 项 |
| 端到端：DSL → `.s` → `.co` → HIP 启动 → 数值比对 | 已打通，见 `tests/gpu/` |
| `attention_fwd`（fp16，示例） | 对 CPU 参考最大绝对误差 ≈ 6e-5 |
| MXFP8 FMHA：`fmha_fwd_hd128_mxfp8_t` | 与厂商 `fwd_hd128_mxfp8.co` 使用**相同的 656 字节 kernarg ABI 和数据布局**，15 个形状逐元素对拍通过 |

`fmha_fwd_hd128_mxfp8_t` 的功能：head dim 128，fp8(e4m3) Q/K/V，E8M0 1×32 Q/K 缩放，标量 V 缩放，BF16 输出，`exp2` softmax，在线 softmax，任意 query / key 长度（含不满一块的 tile），GQA（2 的幂 ≤ 16），任意 batch / head 数。不支持 LSE 输出。

对拍结果（`tests/gpu/run_ref.sh`，同一输入同时运行我们的 kernel、厂商 kernel 和双精度 CPU 模型）：

- 我们 vs CPU：平均误差 1.1–1.8e-3（厂商 kernel vs CPU 为 1.8–2.8e-3）
- 我们 vs 厂商：全部输出逐个比较，平均差异 1.3–2.8e-3；Q = 0 与 SK = 1 时逐位相同
- 差异来源：P 量化到 fp8 的舍入方式不同（固有）

**性能（如实）**：B=1, H=8, S=SK=2048 上约 186 TFLOP/s，厂商 kernel 约 518 TFLOP/s，**慢约 2.7 倍**。没有指令调度器，循环是线性 lowering，两个 wave 之间的 ping-pong 重叠没有实现。详见 `docs/status.md` 的消融实验。

**不是 1:1 复刻**：ABI、网格 / block 形状、1×32 缩放用法、`v_mfma_scale_f32_32x32x64_f8f6f4`、`ds_read_b64_tr_b8`、A/B 两组 wave 的分工都和参考 kernel 一致，但指令级结构不同（P 量化用 `v_cvt_pk_fp8_f32`，参考用 pknorm + scalef32 pack；PV 用 32×32×64 MFMA，参考用 `v_smfmac_f32_16x16x128_fp8_fp8`；每轮 2 个 barrier，参考 4 个 phase；没有 rollback 出口）。对照表见 `docs/final-report.md`。

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

- 只有块内局部调度（hazard 空档填充），没有全局 / 跨块调度；性能落后厂商 kernel 约 2.7 倍。
- 没有跨卡通信（MORI LSA / SDMA / GDA）、gfx1250 后端、JIT / 框架适配的实现；`docs/design.md` 里这些仍是设计。
- 设计文档中的 Resource / Async IR 只落地了一部分（LDS 区域、待完成加载的 wait 跟踪）。
- 没有测量 ≤10 ms 的编译延迟，不做这个声明。