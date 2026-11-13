# AMDGPU_RustDSL

面向 AMD GPU（首发 gfx950 / MI355X，架构上考虑 gfx1250）的 Rust kernel DSL 与自研 compiler。

kernel 作者与自动化 agent 在一个**类型化、可验证、可快速迭代**的 Rust DSL 里写 GPU 数据流；自研 compiler 把它降到内部
IR / MIR，自己完成**寄存器分配**与**跨基本块指令调度**，插入 hazard / wait，发射 `.s`，再由 ROCm 自带 `clang`
**只做汇编**得到 `.co`。LLVM 源码不改，LLVM 的设备端 CodeGen（RA、调度）不参与。

> **权威当前状态见 [`docs/current-status.md`](docs/current-status.md)**。本 README 只做入口与快速上手；
> 厂商事实与认证表在 [`docs/vendor-spec.md`](docs/vendor-spec.md)；历史设计研究与逐轮记录归档在 [`docs/archive/`](docs/archive/README.md)。
>
> **新 kernel 作者 / agent 推荐从这里开始：[`docs/fmha-case-study.md`](docs/fmha-case-study.md)**——以 FMHA 为例讲今天的 kernel 怎样构建（算子图 → 骨架 → slot → stage → weaver → 编译器 → 证书 → 主机）、
> 怎样优化（真实数据、归因、消融上界、同轮 A/B、门禁，以及 FMHA 的完整路程），以及从 TIRx 借鉴了什么。

---

## 当前状态（MI355X / gfx950 实测；Main v11 `a65de21a`：四层 + 两个 Weaver 之后）

| 项 | 结果 |
|---|---|
| **出厂 FMHA（v3，布局层）** | 默认路径 `fmha_fwd_hd128_mxfp8_auto` 的 4 个 kernel `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}`（`crates/amdgpu-kernels/src/fmha/`）建在布局层上：每个 LDS/全局地址、DMA、epilogue 存储与 KV-split 转置暂存都由 `amdgpu-layout` 的布局推导，主机合同由张量布局导出。asm md5（`--backend own`，Main `9cffb86f`）：`ff51fd09` / `a1ee38ee` / `11e2b907` / `24a95a51`（v11 的 `_vs128` = W5 复刻配置；v11.1 起 VALU 写 VCC→读 VCC 为 1 个等待周期）。≥4 个 tile 用厂商参考值配方（懒惰回滚，精度与厂商一致），更短的序列保留精确重定基 |
| **v11：四层 + 两个 Weaver** | L1 `amdgpu-graph`（图、融合、分派）→ L2 `amdgpu-stage`（stage、流水线模板、前端 Weaver 与计划 API）→ L3 `amdgpu-tile` / `amdgpu-layout` / `amdgpu-isa` → L4 `amdgpu-dsl` / `amdgpu-codegen`（后端 Weaver：自由 phase 交织、跨迭代放置与环槽等待、phase 寄存器预算与轮转）；每个改变行为的合并都过同轮全族 A/B（[`docs/perf-ab.md`](docs/perf-ab.md)），v11 对 v10 没有族变慢，gemm 1.040×、MLA 1.139×、decode 1.011×（[`docs/perf-v11-vs-v10.md`](docs/perf-v11-vs-v10.md)）；厂商 FMHA 复刻见 [`docs/fmha-replica.md`](docs/fmha-replica.md)，落后族见 [`docs/w6-laggards.md`](docs/w6-laggards.md)；总览 [`docs/current-status.md`](docs/current-status.md) §1.0 |
| **布局层（路线 B）** | `amdgpu-layout`（`Layout`/`IntTuple`/`Swizzle`/TV 布局/`IssuePlan`/构建期证明）→ `amdgpu-isa`（copy 与 MMA atom 表）→ `amdgpu-tile`（`TiledCopy`/`TiledMma`/`partition_*`）；入门 [`docs/layout-tour.md`](docs/layout-tour.md)，实施记录 [`docs/dsl-layout-refactor-design.md`](docs/dsl-layout-refactor-design.md) §9 |
| **MXFP8 blockscale GEMM** | `gemm_mxfp8_bs_v4`（`crates/amdgpu-kernels/src/gemm/`，对照 aiter PR #6696 4-wave pin-AGPR）：4096×4096×8192 **0.98–1.00×**（v7.0 门禁 3248 vs 3289 TF/s = 0.988×）；6 个形状几何平均 0.982×（[`docs/perf-matrix.md`](docs/perf-matrix.md)）；`tools/gemm_gate.sh` PASS。演进与证据见 [`docs/mxfp8-gemm-4wave-analysis.md`](docs/mxfp8-gemm-4wave-analysis.md) §12–§14；AGPR 见 [`docs/agpr.md`](docs/agpr.md) |
| **算子覆盖（对照 aiter kernel zoo，v9.2）** | 计算密集（[`op-coverage-design.md`](docs/op-coverage-design.md) §7 的汇总表，同进程同数据对照 aiter `.co`，全部 `bad=0`、strict 0 错误；比 = 厂商时间 / ours）：**GEMM 族** §7.4（int8 per-token 1.01–1.63×、blockscale 0.91–1.45×、bf16 0.96×、MXFP4 0.86–0.90×）；**attention** §7.5（fp8 per-tensor prefill 1.07–1.74×，bf16 v3 hd128 0.89–1.07× / causal 0.80–1.23×，MLA prefill hd192 0.73–0.98×，paged decode16 0.67–1.24×，MLA decode 约 0.63×）；**fused MoE** §7.6（fp8 per-token stage 1 0.93–1.14×）；**attention 反向** §7.7（0.77–0.97×）；**implicit-GEMM 卷积** §7.8（2d / 3d bf16，无可运行厂商对照）。访存受限（[`op-coverage-membound.md`](docs/op-coverage-membound.md) 总表与 aiter 覆盖表）：逐元素 6.3–6.7 TB/s（流带宽的 97–103 %）、norm / 量化（含 mxfp4 / mxfp6）/ RoPE / KV cache / MoE 排序与合并 / 采样 / top-k / causal conv1d；aiter 同轮可对照的项 silu_mul 1.07–1.18×、fp8 per-token 1.08× 等 |
| **tile-op 层（TIRx 式基础算子）** | [`op-coverage-tileops.md`](docs/op-coverage-tileops.md) §6.6：访存宽度族（b8–b128、d16、ds_read2 / tr_b16）、LDS / global / buffer 原子、跨 lane（DPP、bpermute、ballot、readlane、permlane16 swap）、类型转换（fp8 / bf8 / fp4 / fp6 / int、E8M0 scale）、unscaled f8f6f4 与 bf16 MFMA、`v_pk_fma_f32`、片段视图（`frag_slice` / `frag_cat` / `pack2` 原地写）、`tileop::ewise / apply2 / allreduce`；每条指令都经过“属性表 → 双后端 → hazard oracle → 硬件探针”闭环 |
| **64 位寻址（> 4 GiB 张量）** | `ptr_mad`（64 位基址 = ptr + a·b）把 buffer 描述符移到 tile / expert / slot 上；验证器证明移动后的描述符在声明的 extent 内。实机：9 GiB 缓冲探针、MoE E160（W1 4.7 GB，旧版 `bad1=224929`，新版 0）、`reshape_cache` 任意大小 cache |
| **结构化诊断（面向 agent）** | `amdc emit / gpu-verify / ir-check / regplan --json`：每条发现带 code、语句、编织区域、**创建该 IR 节点的 Rust 源码位置**（`#[track_caller]`）、相关值与修复建议；`amdc summary <k>`（寄存器 / LDS / 每区域指令构成 / wait 与 nop / 证书计数）、`amdc srcmap <k> [--line N]`（asm 行 → IR 值 → 源码）。见 [`docs/diagnostics.md`](docs/diagnostics.md) |
| **主机侧生成** | `amdc gen-host <kernel>` / `--shipped`：从 kernel 合同生成 C++ 启动/ABI 头（打包 `Args`、`validate()`）与 JSON 元数据，出厂 `ref_host` 用生成的命名字段；见 [`docs/host-gen.md`](docs/host-gen.md) |
| **FMHA 性能（真实分布数据 `REF_DATA=normal`，同轮对照厂商）** | 门禁（v11，`_vs128` 复刻配置，[`docs/fmha-replica.md`](docs/fmha-replica.md) §4.9）：B4H8S2048 **0.983×**、B1H8S8192 **0.991×**、B2H8S2048 **0.973×**、`_vs128` 每 CU **0.971×**（8 布局中位数 0.972×）；KV-split（v9.2 门禁 B1H8S2048 1.028×）与 `_t128`（0.900×）本轮未变；25 形状矩阵 0.929–1.063×，几何平均 **0.992×**（[`docs/perf-matrix.md`](docs/perf-matrix.md)）。旧的 LCG 调色板数据会让双方都几乎每 tile 走回滚分支，高估了我们的相对位置，门禁已改为真实分布 |
| 精度 | 门禁判据 = 厂商持平（每形状每路径 mean ≤ 1.10× 厂商且 bad ≤ 厂商，同轮同输入）：15/15 ×3 路径 + split 13/13；旧 bad=0 判据作为信息仍为 0 |
| 确定性 | 5/5 × N；uneven-tile split 形状重复启动单一 md5 |
| **组件化与编织** | 出厂 kernel 由"算子图 → 生成的骨架 → slot 实现 + stage 实例"构建（[`docs/opgraph.md`](docs/opgraph.md)、[`docs/fusion-stage-library-design.md`](docs/fusion-stage-library-design.md)）；编织区域可声明寄存器规划（pin / separate / share / ring / 区域上限，[`docs/weaver-regs.md`](docs/weaver-regs.md)）与按区域调度策略（[`docs/weaver-sched.md`](docs/weaver-sched.md)）；出厂 asm 不变 |
| **DSL IR 文本** | `amdc print-ir --stage dsl` 输出完整可解析的 `irv1`；手改后 `amdc ir-check` / `emit --from-ir` / `gpu-verify --from-ir`（默认严格验证），A/B 脚本与 Python JIT 接受 `.ir`；全部 kernel 往返后双后端 asm 与证书逐字节相同（[`docs/ir-text.md`](docs/ir-text.md)） |
| 测试 | `cargo test` **144 个目标 / 586 项全绿**（Main `a65de21a`）；md5 清单 240 个 kernel；`python/tests/run_all.sh`：62 / 62 算子检查 + 可达 kernel 编译 / 汇编扫描 + hazard oracle + 每个可达 kernel 的严格 gpu-verify；safe-asm 另有 ui 用例（编译失败对照）；本轮的正确性修复（GEMM epilogue 竞态、VALU 写 SGPR 冒险、moe_sorting / moe_scan LDS 竞态、SGPR 上限）见 [`docs/current-status.md`](docs/current-status.md) §6.3 |
| GPU 语义验证 | 编译器感知指令语义：LDS/global/buffer 越界、VGPR/SGPR/AGPR 类别与对齐、路径敏感等待、LDS tile 生命周期与同步协议、融合 stage 边界契约均被证明。跨工作组写不相交（出厂 kernel 全部写已证明）。严格证书：出厂 4 kernel `--shipped` **21 trusted assumes / 100 host contracts / 0 unproven**；GEMM v4 **1 / 19 / 0**（v11.1 后 amdgpu-sym 非负性检验证明了原先 2 个信任假设）；融合 MLP **0 / 9 / 0**；见 [`docs/gpu-semantics-verifier.md`](docs/gpu-semantics-verifier.md) 与信任清单 [`docs/trust-inventory.md`](docs/trust-inventory.md)（每一项信任、写不相交证明、LLVM 冒险规则审计） |
| **Python 算子层** | 纯标准库：`launch(kernel, …)` 调用任意注册 kernel 并按已验证的契约检查；norm / 量化 / RoPE / KV cache / MoE / GEMM / prefill / decode / 采样 / top-k / 卷积 / 反向的算子封装；预备启动后每次调用主机时间 5.5–9.3 µs（之前 23–57 µs）；见 [`docs/python-jit.md`](docs/python-jit.md) |
| Rust 安全 | **12 个 crate 零 `unsafe`**（10 个声明 `#![forbid(unsafe_code)]`，`amdgpu-graph` / `amdgpu-stage` 源码无 `unsafe`）、零第三方依赖、无 FFI/裸指针；ISA 规则编码进类型系统；见 [`docs/rust-safety.md`](docs/rust-safety.md) |

功能（FMHA）：head dim 128，fp8(e4m3) Q/K/V，E8M0 1×32 Q/K 缩放，标量 V 缩放，BF16 输出，在线 softmax，
任意 query/key 长度（含不满一个 tile），GQA（2 的幂 ≤ 16），任意 batch/head。**与厂商 656 字节 kernarg ABI 和网格/数据布局逐字节兼容。**

---

## 快速上手

> **GPU 规则（2026-11-11）**：GPU 测试只允许使用 GPU 0–3，任何节点都不得使用 4–7。所有跑 GPU 的脚本都 source [`tools/gpu_guard.sh`](tools/gpu_guard.sh)：`HIP_VISIBLE_DEVICES`（或 `ROCR_VISIBLE_DEVICES`）未设置或含 0–3 以外的卡时直接退出 1。

```bash
# 构建与测试（容器脚本会挂载 ROCm 与 pinned LLVM 工具）
bash tools/container/cargo.sh build
bash tools/container/cargo.sh test                    # 期望 121 个目标 / 519 项全绿

# 发射一个 kernel 的汇编
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --out vs128.s
bash tools/container/cargo.sh run -q --bin amdc -- list            # 所有 kernel 名
bash tools/container/cargo.sh run -q --bin amdc -- emit <k> --backend own --sched ld

# DSL IR 文本：打印 → 手改 → 检查 → 编译（docs/ir-text.md）；寄存器区域报告（docs/weaver-regs.md）
bash tools/container/cargo.sh run -q --bin amdc -- print-ir fmha_fwd_hd128_mxfp8_v3_t128 --stage dsl --out t128.ir
bash tools/container/cargo.sh run -q --bin amdc -- ir-check t128.ir --verify
bash tools/container/cargo.sh run -q --bin amdc -- emit --from-ir t128.ir --out t128.s
bash tools/container/cargo.sh run -q --bin amdc -- regplan fmha_fwd_hd128_mxfp8_v3_vs128

# 结构化诊断（docs/diagnostics.md）：JSON 发现 / 紧凑摘要 / asm 行到源码
bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify gemm_mxfp8_bs_v4 --json --strict
bash tools/container/cargo.sh run -q --bin amdc -- summary fmha_fwd_hd128_mxfp8_v3_vs128
bash tools/container/cargo.sh run -q --bin amdc -- srcmap fmha_fwd_hd128_mxfp8_v3_vs128 --line 400

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
HIP_VISIBLE_DEVICES=0 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3     # FMHA 全套验收（精度=厂商持平；性能=同轮厂商比 ≥ 0.97×记录的出厂比；ACC_STRICT=1 用旧 bad=0 判据）
HIP_VISIBLE_DEVICES=0 bash tools/gemm_gate.sh gemm_mxfp8_bs_v4            # GEMM（BUILD / VERIFY / ACC / DET / PERF 对照厂商）

# 确定性 / 性能矩阵
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto bash tests/gpu/determinism.sh
HIP_VISIBLE_DEVICES=0 bash tools/perf_matrix.sh                         # FMHA 25 形状 + GEMM 6 形状，同轮对照厂商

# 文档链接检查
python tools/check_doc_links.py

# Python JIT（纯标准库，subprocess 调 amdc + ROCm clang，磁盘缓存，默认按形状特化；docs/python-jit.md）
python3 -c "import sys; sys.path.insert(0, 'python'); import amdgpu_jit as aj; help(aj.gemm_mxfp8_blockscale)"
HIP_VISIBLE_DEVICES=0 AMDGPU_JIT_AMDC=target/release/amdc python3 python/tests/node_check.py   # 与 C++ harness 逐位一致、缓存、开销、特化 vs 通用
```

**LLVM pin**：ROCm 10.1.0（AMD LLVM 24.0.0git，方言 `llvm24`）。ROCm 10 不自带 `llc`，用
`tools/llvm_build_llc.sh` 从同一 commit 链接 `libLLVM.so` 得到。ROCm 7.0.2（方言 `llvm20`）可通过
`ROCM_LLVM_BIN` / `AMDC_DIALECT` 选用。pin 缺失时 **llvm-tail 直接报错**，不静默降级（`--backend=own` 用完整自研 tail）。

---

## 目录

```
crates/amdgpu-sym       统一符号分析（零依赖）：有理数、多项式、约束上下文、区间、位集；合同/布局/验证器共用
crates/amdgpu-layout    布局代数：Ix / IntTuple / Layout / Swizzle / TV 布局 / IssuePlan / BitForm / convert_layout 分档 / 构建期证明
crates/amdgpu-isa       gfx950 atom 表：copy atom 与 MMA/smfmac atom（A/B/C 的 TV 布局、provenance，含硬件探针得到的映射）
crates/amdgpu-dsl       类型化 kernel IR（区域树：Loop/If/Region/Stage + 编织区域、寄存器规划、区域策略）+ FnBuilder + 表达式层 + 形状特化 + irv1 文本打印/解析 + α 结构相等
crates/amdgpu-tile      全部 tile 抽象：TiledCopy/TiledMma/partition_*、行拷贝、pipeline、softmax、reduce、epilogue、DMA/bufdma、handoff；stage 库与通用原语、weaver、算子图/骨架生成
crates/amdgpu-graph     L1 图层：模型级算子图 + 融合规则（算子子图 → 已有融合 kernel 名）+ 启动计划（`amdc graph plan`）；dispatch 规则（GEMM / prefill / MXFP8 / decode / MLA / MoE / 反向，按形状选 kernel 与参数，Python 入口与计划共用同一条规则）；kernel 内部的 stage 级算子图与骨架生成
crates/amdgpu-kernels   全部 kernel（算子图 → 骨架 → slot + stage 实例）：fmha/（v3 出厂四件 + auto、bf16 prefill / decode / MLA / 反向）、gemm/（MXFP8 v4 与 GEMM 族、MoE、卷积）、mlp/、norm/、rows/ eltwise/ dequant/ rope/ cache/ moe/ topk/ sampling/（访存受限算子）、probes/、verifier_cases/ + 按名注册表；IR 功能证明执行器
crates/amdgpu-codegen  编译器 + 验证器：MIR（算子属性表）、lowering、调度器（list/slotfill、按区域策略）、线性扫描 RA（含 AGPR、寄存器规划）、hazard/wait、hzfill、asm 发射、LLVM MIR 桥、gpu_verify/协议引擎/功能证明；不依赖 kernels/tile
crates/amdgpu-safeasm   手写热点窗口用的 safe Rust asm DSL（typed 寄存器 + 所有权 token + 验证器 + pin）
crates/amdc             CLI（list / emit / print-ir / ir-check / regplan / summary / srcmap / jit-info / gpu-verify / gen-host / opgraph / skeleton / dispatch / graph plan / md5-manifest / backend / simmodel / mops / mir-roundtrip；`--json`）+ hostgen、jit 库
tests/gpu/              端到端：ref_host（我们 vs 厂商 vs CPU）、run_ref、determinism、split 压力、gemm_ref_host、mlp_host、探针 host、生成的主机 ABI（generated/）
python/amdgpu_jit/       Python JIT 入口（ctypes + libamdhip64；GEMM / FMHA / prefill / decode / MoE 等算子，按 dispatch 规则选 kernel；`graph.plan` / `graph.run` 执行 L1 计划；磁盘缓存；kernarg 打包 + 主机合同校验）
tools/                  门禁（port_gate / gemm_gate）、性能矩阵、A/B、ATT/计数器/时间戳分析、asm 对比、探针、LLVM 构建、容器 cargo、文档链接检查
docs/                   当前状态、设计、教程、验证器、厂商 spec；历史文档在 docs/archive/
```

## 文档

| 文档 | 内容 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | **四层架构（实际实现）**：crate 依赖图与各层内容、两个贯穿各层的例子、gfx950 pass 流水线与目标相关性、gfx1250 状态、不变量 |
| [`docs/design-to-kernel.md`](docs/design-to-kernel.md) | **设计如何落到 kernel 上**：四层 + 两个 Weaver 各自对应的 crate / 函数、作者决定什么与系统决定什么；三个端到端例子（出厂 MXFP8 `_vs128`、MLA decode、小 M GEMM 经图 / 组合路径）附实跑的 `amdc` 输出与门禁；加新 kernel 的清单；默认与 opt-in、测得为负的方向 |
| [`docs/l1-graph.md`](docs/l1-graph.md) | **L1 图层**：算子词表、融合规则（rmsnorm+quant、add+rmsnorm(+quant)、silu_mul+quant、qknorm+rope(+kvcache)…）、多 kernel 计划（attention split→combine、MoE、GEMM split-K）、`amdc graph plan`、golden 计划与数值检查 |
| [`docs/python-jit.md`](docs/python-jit.md) | **Python JIT 入口**：amdc 子进程编译 + 磁盘缓存 + 形状特化（默认开）、kernarg 打包与主机合同校验、GEMM / FMHA 高层算子（含出厂路径选择） |
| [`docs/current-status.md`](docs/current-status.md) | **权威当前快照**（架构、DSL、kernel、数字、证书、已关闭方向、复现命令） |
| [`docs/op-coverage-design.md`](docs/op-coverage-design.md) / [`docs/op-coverage-membound.md`](docs/op-coverage-membound.md) / [`docs/op-coverage-memtemplates.md`](docs/op-coverage-memtemplates.md) / [`docs/op-coverage-tileops.md`](docs/op-coverage-tileops.md) | **aiter 基础算子覆盖**：迁移设计与批次状态（§7：GEMM / attention / MoE / 反向 / 卷积）/ 访存受限算子实测与 aiter 覆盖表 / 访存模板 T1–T7 / TIRx 式 tile-op 覆盖矩阵 |
| [`docs/diagnostics.md`](docs/diagnostics.md) | **结构化诊断**：`--json` 发现（code / 语句 / 区域 / Rust 源码位置 / 修复建议）、`amdc summary`、`amdc srcmap` |
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
| [`docs/fusion-stage-library-design.md`](docs/fusion-stage-library-design.md) | **融合 stage 组件库设计**：按 aiter 算子分类分解为 9 个基础 stage 原语 + 组合规则、StageContract、边界代价分级、轻量算子图、示例与分期计划（设计稿） |
| [`docs/weaver-auto.md`](docs/weaver-auto.md) | **基于代价模型的自动编织策略**（可选，未采用）：单 wave 发射模型 + list scheduling / 局部搜索 + 寄存器护栏，单元级与语句级，FMHA 实测与 t128 的合法性缺口 |
| [`docs/weaver-regs.md`](docs/weaver-regs.md) / [`docs/weaver-sched.md`](docs/weaver-sched.md) | **Weaver 后端层**：weave 区域的寄存器规划（pin/share/separate/cap/ring、region_report）与按区域的调度策略（authored/free） |
| [`docs/opgraph.md`](docs/opgraph.md) | **算子图与骨架生成器（P3）**：图 / 边界类 / 切分的数据模型与 JSON、`amdc opgraph` / `amdc skeleton`、三种骨架形式（顺序 / 流水线 / 分派）、出货 kernel 经由生成骨架构建（md5 不变） |
| [`docs/fmha-weaver-parity.md`](docs/fmha-weaver-parity.md) | 用 weaver 冲铺满场景的全部实验与结论（均未超过出货编排） |
| [`docs/ir-text.md`](docs/ir-text.md) / [`docs/ir-text-roundtrip-design.md`](docs/ir-text-roundtrip-design.md) | **DSL IR 文本（irv1）**：打印-编辑-解析-编译工作流、格式与校验 / 设计（已实现） |
| [`docs/jit.md`](docs/jit.md) | 形状特化（`--specialize`）与 `amdc jit-info`（Python JIT 的编译接口） |
| [`docs/tirx-comparison.md`](docs/tirx-comparison.md) | **与 Apache TVM TIRx 的异同对照（当前状态）**：继承并已实现 / 部分采纳 / 有意不同 / 推迟，早期建议的落实情况 |
| [`docs/ir-review-tirx-sftensor.md`](docs/ir-review-tirx-sftensor.md) | **IR 梳理**：对照 TIRx（算子属性表、区域语句、统一符号分析、打印/结构相等、命名轴 layout）与 SF Tensor 形式化正确性，提出属性表 / 区域化 IR / 统一 sym / 功能证明层的改造顺序 |
| [`docs/layout-tour.md`](docs/layout-tour.md) | **布局层入门**：`Layout` / `Swizzle` / TV 布局 / `IssuePlan` / 证明，用出厂 FMHA 与 GEMM 的真实代码讲 |
| [`docs/fmha-dsl-tour.md`](docs/fmha-dsl-tour.md) | **FMHA 逐段教程**：每一段的 DSL 特性与 Rust safe 怎么保证 |
| [`docs/dsl-layout-refactor-design.md`](docs/dsl-layout-refactor-design.md) | 路线 B（布局代数重构）设计与 §9 as-built |
| [`docs/dsl-tile-abstractions.md`](docs/dsl-tile-abstractions.md) | tile 库（L1–L6）设计；哪些层已被布局层替代 |
| [`docs/dsl-expression-layer.md`](docs/dsl-expression-layer.md) | 运算符表达式层 |
| [`docs/fmha-perf-patterns.md`](docs/fmha-perf-patterns.md) | FMHA 性能承重模式清单（移植 parity checklist） |
| [`docs/fmha-wave-org.md`](docs/fmha-wave-org.md) / [`docs/fmha-vendor-stream-design.md`](docs/fmha-vendor-stream-design.md) / [`docs/lgkm-lookahead.md`](docs/lgkm-lookahead.md) | FMHA 设计记录：铺满形状的 wave 组织与 XCD 感知 block 顺序（v8.1）/ 厂商 copy 流对照（真实数据）/ LDS 读与 lgkm 等待的调度策略 |
| [`docs/fmha-vs-vendor-asm-diff.md`](docs/fmha-vs-vendor-asm-diff.md) / [`docs/fmha-4wave-design.md`](docs/fmha-4wave-design.md) | 与厂商汇编的逐项对照（寄存器 + 编排）/ 4-wave 原型（已搁置） |
| [`docs/isa-probes-smfmac-tr.md`](docs/isa-probes-smfmac-tr.md) / [`docs/mfma-dep-probe.md`](docs/mfma-dep-probe.md) | ISA 探针：`v_smfmac` 与 `ds_read_b64_tr_b8` 的 one-hot 映射 / MFMA 依赖与操作数复用对发射间隔、时钟、功耗的影响 |
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
- 编译策略按 kernel profile 固定（`crates/amdgpu-codegen/src/lib.rs`）；实验开关、别名 kernel 与负结果变体已在 2026-11 整理中删除（结论在 `docs/archive/`，实验分支归档为 `archive/*` tag）。

## 许可

内部研究项目；第三方 kernel（aiter `fwd_hd128_mxfp8.co`、aiter GEMM code object）仅用于对照，不随仓库分发。
