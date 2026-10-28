# 最终总结与对照（第 37 轮）

本文把整个工作按"目标 → 现状 → 证据 → 差距"整理一遍。逐轮的验证记录和硬件事实在 `status.md`，设计在 `design.md`。所有数值结论都能用 `README.md` 里的命令复现。

## 1. 目标逐项对照

| 目标 | 现状 | 证据 |
|---|---|---|
| Rust AMDGPU kernel DSL | 已实现：类型化 `Op` / `Stmt`、`FnBuilder`、循环 / 分支 / 循环携带变量、fragment 类型（`Frag2/4/8/16`）、MFMA / LDS / 全局访存 / pack 指令、lane / 子寄存器操作 | `crates/amdgpu-dsl`，`cargo test` |
| 内部 SSA / 资源 IR | SSA 值 + 循环变量；资源部分只落地了 LDS 区域和"待完成加载"的 wait 跟踪，设计文档里的完整 Resource / Async IR **未实现** | `docs/design.md` 与实现的差距见 §4 |
| 验证 | DSL 验证器（类型、可见性、循环变量作用域）、MIR 验证器（寄存器类、MFMA 操作数、vcc 邻接）、RA 后检查（MFMA 对齐 / 重叠）；GPU-free 回归测试 10 项 | `cargo test` |
| RA / 调度 | 线性扫描 RA（紧凑策略失败时改用窄操作数从顶部分配的回退策略）、copy coalescing、DCE、sub-register 别名、hazard / wait 插入（`s_nop`、`s_waitcnt vmcnt(n)` / `lgkmcnt`）。**没有指令调度器** | `lower.rs`、`regalloc.rs`、`hazards.rs` |
| asm 发射，不修改 LLVM | 发射 `.s`，由 ROCm `clang -x assembler` 汇编成 `.co`；LLVM 源码未改 | `emit.rs`、`tests/gpu/run.sh` |
| 在 gfx950 真机运行的 attention kernel，并通过数值验证 | **达成**：`attention_fwd`（fp16）与 `fmha_fwd_hd128_mxfp8_t`（MXFP8，厂商 ABI）都在 MI355X 上运行并对拍通过 | `tests/gpu/run.sh`、`tests/gpu/run_ref.sh` |
| 推进 FMHA 参考 core loop 的 1:1 复刻 | **部分**：见 §2。ABI、网格、缩放用法、主要指令族、wave 分工一致；指令级结构和性能不一致 | §2、§3 |

## 2. 与参考 kernel（aiter `fwd_hd128_mxfp8.co`，commit 7061cbd）的对照

| 项 | 参考 kernel | 我们的 `fmha_fwd_hd128_mxfp8_t` | 一致？ |
|---|---|---|---|
| kernarg ABI | 656 字节，固定槽位 | 相同槽位（逐字节读取同一布局） | 是（实测对拍） |
| grid / block | `(ceil(S/256), H, B)`，512 线程 | 相同 | 是 |
| 数据格式 | e4m3 Q/K/V，E8M0 1×32 Q/K 缩放，标量 V 缩放，BF16 输出 | 相同 | 是 |
| softmax 约定 | `exp2(S)`，忽略 kernarg 里的 softmax scale | 相同（实测发现） | 是 |
| 缩放 MFMA | `v_mfma_scale_f32_32x32x64_f8f6f4`（16 字节编码，VGPR 缩放） | 相同指令，K 排列与缩放字节选择经真机验证 | 是 |
| LDS 转置读 | `ds_read_b64_tr_b8` / `ds_read_b128` | 相同指令，lane 映射经真机测量 | 是 |
| wave 分工 | `s62 < 4` 分 A / B 两条独立循环 | wave 0–3 / 4–7 各跑独立循环（A 搬 K，B 搬 V），同一 barrier 日程 | 结构类似，功能不同 |
| barrier | 每轮 4 个 phase，成对 | 每轮 2 个 | 否 |
| P 量化 | `v_cvt_pknorm_u16_f32` ×32 + `v_cvt_scalef32_pk_fp8_bf16` ×64 | `v_cvt_pk_fp8_f32` ×32，`P·16` 折进 exp2 参数 | 否 |
| PV 乘法 | `v_smfmac_f32_16x16x128_fp8_fp8`（稀疏 MFMA，原地累加）+ 缩放 MFMA | 32×32×64 缩放 MFMA | 否 |
| 行最大值 | `v_max3_f32` 链 + `v_cmp_nge_f32` 触发的 rollback 出口 | `v_max3_f32` 链 + 与 `lane ^ 32` 交换；无 rollback（试验过惰性重缩放，收益 ≤3%，已回退） | 部分 |
| 资源 | 246 VGPR，AGPR 0，LDS 84,224 B | 208 VGPR，LDS 34.8 KB | 否 |
| 吞吐（1×8×2048×2048） | ≈ 518 TFLOP/s | ≈ 186 TFLOP/s | 慢约 2.7× |

结论：**我们没有逐指令复刻参考 kernel**。复刻的是它的接口契约、数据布局、缩放 / softmax 约定和主要硬件指令族，并且与它在 15 个形状上逐元素对拍。核心循环本身是另一种实现（转置形式 `Sᵀ = K Qᵀ`，分数全在寄存器里）。

## 3. 性能归因（第 31–36 轮）

- 第一版（行 per lane，分数经 LDS）约 15 TFLOP/s；转置版 88；去掉大部分 `v_mov`（sub-register 别名）118；批量 16 字节加载 + 预取 + 地址折叠 190。
- 消融实验：暂存 + barrier 约 23%，PV 约 10%，exp 约 6%，其余一半以上是 S 路径（QKᵀ MFMA + hazard 补 nop + 最大值交换 + `sub` / `cvt`）。
- 否定的优化：先发四个 score tile（VGPR ↑，更慢）、惰性重缩放（≤3%）、前置 K 缩放加载（更慢）。
- 剩余差距需要调度级改动（两个 wave 在同一 SIMD 上的 ping-pong 重叠、MFMA 与 VALU 交错），当前线性 lowering 无法表达。

## 4. 编译延迟（实测，第 37 轮）

release 构建，整个进程（启动 + DSL 构建 + 编译 + 写文件），取 5 次最小值：

| kernel | 汇编行数 | Rust 编译侧耗时 |
|---|---|---|
| `attention_fwd` | 1264 | 1.8 ms |
| `fmha_fwd_hd128_mxfp8_t` | 4413 | 5.7 ms |
| `probe_fp8_fmha_hd128c` | 6569 | 14.5 ms |

再加上 `clang -x assembler`（独立进程，进程启动占主要部分）：`attention_fwd` 16 ms，`fmha_fwd_hd128_mxfp8_t` 22 ms。所以**自研编译侧在 10 ms 目标之内（对中等大小的 kernel），但端到端经子进程 clang 汇编不满足 ≤10 ms**；要达标需要进程内汇编（设计里的 LLVM bridge，未实现）。

## 5. 未实现 / 设计中的部分

- 完整的 Resource / Async IR、provenance、witness 驱动的 1:1 复刻管线（`design.md` §Witness）：只有设计，没有实现。
- 指令调度器。
- 跨卡通信（MORI LSA / SDMA / GDA）、gfx1250 后端、JIT 与框架适配（DeepInfer / SGLang）。
- LSE 输出；`+0x80` KV 进度更新与 rollback 出口。

## 6. 发现并修复的编译器问题（价值最高的部分）

这些问题都是先在真机上表现为"随机错误"或"寄存器不够"，再定位到编译器：

1. MFMA 结果读取缺少延迟补 nop（第 12 轮）。
2. V1/V2 加载作为 scale 操作数被消费时没有 `s_waitcnt`（第 14 轮）。
3. 控制流两侧共享"待完成加载"状态：一侧发的 wait 让另一侧漏等（第 23 轮）。
4. 常量 / 转换缓存没有按控制流区域隔离：一侧物化的常量被另一侧当作已存在（第 23 轮）。
5. 空 block 把跨过它的值的活跃区间拉到位置 0，造成大量虚假寄存器压力（第 28 轮）。
6. DSL 变量遮蔽（`g8` / `g16`）：不是编译器 bug，但被真机对拍立即发现（第 34 轮）。

每一项都有对应的回归（GPU-free 测试或真机 probe）。