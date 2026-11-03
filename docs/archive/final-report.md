# 最终总结与对照（第 75 轮）

> 元数据：Main `8fec359`，tag `v2.2-round75-final`，第 75 轮收官。主快照见 `docs/current-status.md`（含已关闭路线的完整清单 §5）；逐轮验证记录、硬件事实和所有数值在 `docs/archive/status.md`；设计在 `docs/design.md`；厂商 kernel 的逐条事实在 `docs/vendor-spec.md`。所有数值结论都能用 `README.md` 里的命令复现。

本文按"目标 → 现状 → 证据 → 差距"整理整个工作。第 37 轮版本只覆盖了最初的 `fmha_fwd_hd128_mxfp8_t`（186 TFLOP/s，厂商的 0.36 倍）；之后的 38 轮把它推进到发布路径 1.25–1.31 倍厂商、单 CU 0.84 倍厂商，并把"能不能再进一步"的问题用测量关闭。

## 1. 目标逐项对照

| 目标 | 现状 | 证据 |
|---|---|---|
| Rust AMDGPU kernel DSL | 已实现：类型化 `Op` / `Stmt`、`FnBuilder`、循环 / 分支 / 循环携带变量、fragment 类型、MFMA（含缩放 / 非缩放 fp8、f16 偏置）/ SMFMAC / LDS（含转置读、DMA）/ 全局访存 / MUBUF（描述符、`buffer_*`、LDS 直通形式）/ 原子 / pack 指令 | `crates/amdgpu-dsl`，`cargo test` |
| 自研编译器（SSA → MIR → 寄存器分配 → 调度 → asm） | 已实现，默认后端是 `llvm-tail`（自研 lowering / RA / 调度，LLVM 只负责 wait / hazard / memory legalizer）；`--backend=own` 保留完整自研 tail | `docs/llvm-tail-integration.md`，`tests/tail_invariance.rs` |
| 验证 | DSL / MIR 验证器、RA 后检查、safe-asm 序列的 `pin_check`（对最终 MIR 重跑 value-hazard pass），`cargo test` 29 个目标共 117 项通过 | `cargo test` |
| 在 gfx950 真机运行并通过数值验证的 attention kernel | **达成并超过**：`attention_fwd`（fp16）与 MXFP8 厂商 ABI 的 FMHA 家族在 MI355X 上运行，15 个形状逐元素对拍 | `tests/gpu/run.sh`、`tests/gpu/run_ref.sh`、`tools/gen9_gate.py` |
| 推进 FMHA 参考 core loop 的 1:1 复刻 | **功能与接口 1:1，指令级结构不同**：ABI、网格、缩放用法、数值约定、主要硬件指令族一致，并已通过 15 形状对拍与真机计数器对账；核心循环是另一种实现（`Sᵀ = K Qᵀ`，分数全在寄存器，128-key 单元，一个 rollback，每 tile 两个 barrier）。结构层面的逐指令复刻（vendor-order M 窗口、持久偏置 C、V 预读）做了完整的有界实验，结论为负（§6、§7） | `docs/archive/gen9-vendor-mirror-design.md`、`docs/archive/status.md` 第 68/70/73/75 轮 |
| 性能 | 发布路径（`_auto`）B1H8S2048 **674–705 TFLOP/s = 厂商同轮 536–539 的 1.25–1.31 倍**；单 CU 453–455 TFLOP/s = 7.07–7.10 TF/CU，厂商 8.41 TF/CU（我们是它的 0.84 倍）；填满网格 B4H8S2048 0.846 倍、B1H8S8192 0.841 倍；中等形状 ≥ 厂商（1.00–1.16 倍） | §2 |

## 2. 认证数字（Main `8fec359`）

### 2.1 吞吐（TFLOP/s，同轮厂商对照，own 后端，GPU 0 / GPU 5）

| 形状 / 口径 | 我们 | 厂商 | 比值 | 说明 |
|---|---|---|---|---|
| B1H8S2048，发布路径 `_auto`（64 个 256 行块，自动走 KV-split） | **674–705** | 536–539 | **1.25–1.31×** | 欠填充网格：128 行分块 + KV-split 补满 CU |
| B1H8S2048，`_vs128` 单 CU（64 块 = 64 CU） | 453–455（7.07–7.10 TF/CU） | 8.41 TF/CU | 0.84× | 每 CU 关键路径差距 1.19× |
| B4H8S2048（256 块，填满） | 1443.6 / 1454.8 | 1706.8 | **0.846×** | 一块一 CU，与厂商占用率相同 |
| B1H8S8192（256 块，填满） | 1998.6 / 2002.7 | 2376.5 | **0.841×** | 同上 |
| 中等形状（B2H8S2048、B1H8S4096、B1H8S1024 等） | — | — | **1.00–1.16×** | `_t128` / `_vs128` / KV-split 按守卫自动选择 |

占用率与厂商相同：厂商 LDS 84224 B / vgpr 246 / sgpr 99 / 512 线程，我们 LDS 102400 B / vgpr 249 / sgpr 48 / 512 线程；任一方都放不下第二个块（2×84224、2×102400 都超过 163840 B；2 个 wave/SIMD × ~250 VGPR 占满 512），填满网格里每个 CU 恰好一块 8 个 wave。因此填满网格的差距就是单 CU 关键路径差距，不是占用率。

### 2.2 精度与正确性

| 项 | 结果 |
|---|---|
| 精度（对 CPU 参考，REF_ROWS=64） | 平均误差 **8.3e-4**（厂商 1.5–2.9e-3，我们约好 3 倍）；`bad = 0`（厂商自己在 3 个形状上各有 1–5 个坏点） |
| 15 个形状 | 16 行 / 64 行门禁均 15/15，vendor-parity 15/15（`tools/gen9_gate.py`） |
| 确定性 | `determinism.sh` 5/5 ×多轮；**重复启动压力 1200/1200**（4 个不均匀 tile 数的 KV-split 形状，每个 300 次，逐次比较完整输出 md5）；`FMHA_OPT=eplds,epw` 敏感度探测器 100–200/200 干净 |
| 单元测试 | `cargo test`：29 个目标、117 项全部通过；发布路径汇编 md5 在所有非 KV-split 改动中保持不变 |

### 2.3 发布路径（`_auto`）的组成

见 `docs/fmha-reference.md` 末节"我们的实现状态"：`_auto`（默认，按守卫模型选 `_vs128` / `_t128` / `_kvsplit`+combine）、`_vs128`（256 行块，填满网格与单 CU 基准）、`_t128`（128 行块，欠填充形状）、`_kvsplit` + `_kvsplit_combine8h`（序列并行 + 精确合并）、`_t`（冻结的精确参考实现）。

## 3. 优化历程（B1H8S2048，发布路径 TFLOP/s）

| 里程碑 | TFLOP/s | 主要来源 |
|---|---|---|
| 第 37 轮 `_t` 初版 | 186 | 转置形式 `Sᵀ = K Qᵀ`，批量 16 字节加载，地址折叠 |
| `_t` + LLVM tail + 调度默认 | **292** | LLVM tail 接管 wait / hazard；XOR swizzle，LDS bank conflict 524288 → 8704（第 4 轮） |
| `_vs128`（128-key 单元、一个 rollback、两个 barrier、smfmac 行和、Q 常驻寄存器） | **367** | 第 49–56 轮 |
| `_t128`（128 行块） | **485–505** | 欠填充网格的分块收益（第 59 轮） |
| + KV-split（fp16 partial、LDS 转置 16 字节存储、combine 合并） | **605–613** | 第 60–61 轮 |
| + K-scale LDS-DMA（`kddma`） | **622–635** | 第 61 轮 |
| + Q 加载向量化、8 字节尾部存储、`combine8h`、三个编译器 bug 修复 | **674–705** | 第 65–69 轮 |

单 CU 基准（`_vs128`，64 块）：389.6 → 413（Q 加载 `load_b128`）→ 449–455（8 字节输出存储 `global_store_dwordx2`）。

## 4. 与厂商 kernel（aiter `fwd_hd128_mxfp8.co`）的对照（现状）

| 项 | 厂商 | 我们（`_vs128`） | 一致？ |
|---|---|---|---|
| kernarg ABI / grid / block / 数据格式 / softmax 约定 | 656 字节、`(ceil(S/256),H,B)`、512 线程、e4m3 + E8M0、`exp2` | 相同 | 是（逐元素对拍） |
| 缩放 MFMA、LDS 转置读、smfmac 行和、pknorm + scalef32 的 P 量化 | 是 | 是（P 量化与 rollback 的数值路径已对账） | 是 |
| 参考 / rollback | f32 参考，rollback 时 `r += s93·s26·δ` | 精确 hi/lo f16 参考对（约 2⁻²²），同样的惰性 rollback；我们的 P 近似误差更小（平均误差约好 3 倍） | 数值等价或更优 |
| 每 tile 矩阵工作 | 16 个 64 周期 MFMA + 2 smfmac + 1 偏置 MFMA = 1088 XDL 周期 | 16 个 64 周期 MFMA + 4 smfmac + 4 偏置 MFMA = 1216 | 否（见 §6） |
| 每 tile VALU | ≈ 251 | ≈ 353（热路径循环体 ≈ 222） | 否，但已证明不是限制因素（§6） |
| 占用率 | 1 块 / 8 wave 每 CU | 相同 | 是 |
| 吞吐（填满网格） | 1706.8（B4H8S2048） | 1443.6–1454.8 | 0.846× |

## 5. 发现并修复的三个真实编译器正确性 bug（价值最高的部分）

这三个 bug 都是先在真机上表现为"随机错误"，再追到编译器；每一个都有 GPU-free 回归测试或真机重现脚本。

1. **物理寄存器 WAW（physical WAW）**：MFMA / 长延迟指令的写回与后续对同一物理寄存器的写之间缺少排序，在 own 后端的某些 DSL 结构下产生随机输出。修复：由 backend 在 hazard 层对物理寄存器的 WAW / WAR 补齐等待（`AMDC_NO_PHYS_WAR=1` 保留为 A/B 开关）。
2. **two-phase RA 的 reaching-writer 错误**：两阶段寄存器分配在循环携带 / 条件写路径上选错了到达写者，导致把仍活跃的值覆盖（`biasacc,vpre` 组合在全重排调度器下非确定性，其他调度配置下确定）。修复由 backend（`backend-race` 分支）在两阶段 RA 中完成，`AMDC_RA_FORCE_TWOPHASE` 保留为回归开关。
3. **VMEM 宽存储数据 WAR（第 69 轮，影响最大）**：`global_store_dwordx4` 在发射之后的数个周期里才读取它的数据 VGPR；我们自己的 tail 没有为"宽存储之后 2 个 wait state 内对同一数据寄存器的 VALU 写"补 `s_nop`（LLVM 的 `has12DWordStoreHazard` 有）。KV-split 的 fp16 partial 尾部在 4 个 lane（j = 25/27/29/31）的 `ec = 1` 字上间歇性被新值污染：均匀 tile 数形状无感，不均匀 tile 数（B1H2S200 SK1337）0–5% 的启动出错，时序漂移时升到 30%。修复：`hazards::pad_physical_war` 记录宽存储数据范围并为后续 VALU 写补 `s_nop`（`_kvsplit` 仅多 4 条 `s_nop`，成本 0.3%，噪声级），`AMDC_NO_PHYS_STORE=1` 保留为 A/B 开关。附带结论：单次 5/5 确定性测试不足以认证，必须做重复启动完整输出 md5 压力（1200/1200）并保留 `epw` 探测器；`emit_fmha_vs9_role` 是 vs128 role 的逐字节副本，补丁必须落在 vs128 role。

早期修复的编译器问题（MFMA 结果读取缺 nop、scale 操作数缺 `s_waitcnt`、控制流两侧共享 pending-load、常量缓存未按区域隔离、空 block 拉长活跃区间等）仍保留回归测试，见 `docs/archive/status.md` 前 40 轮。

## 6. 已关闭的负面路线（每条一句归因；完整数据见 `docs/archive/status.md` 与 `docs/current-status.md` §5）

| 路线 | 结果 | 一句归因 |
|---|---|---|
| gen2 / gen3：单 wave 内 VALU 与 MFMA 重叠、合并窗口、barrier 位置偏移（第 57–58 轮） | 负 | 两个 wave 组本来就处于互补相位，进一步合并只增加寄存器压力和调度脆弱性 |
| dsplit（延迟 DMA wait）、`dmah`、`dmin`（第 61 轮） | −6..−11% / −0.9% / −1.0% | DMA 延迟不在关键路径上，付代价的是 DMA 发射与 barrier 对齐 |
| 融合 last-arriver combine（第 62 轮） | 最终负（默认关闭，代码保留） | 单发射的 combine 需要跨 XCD 一致性 fence，成本高于它省下的 ~2.5 µs 启动底 |
| 占用率路线：4 wave/SIMD 的维度分割 O（第 46 轮起） | 负 | VGPR / LDS 预算放不下，且厂商是同样的 1 块/CU 占用率 |
| gen5 安全汇编窗口（第 63 轮） | Block C −2.5%，Block M 中性，Block B 在 2 槽 K 环下无法构建 | 手写顺序不比列表调度器更优，关键路径是依赖链而不是指令顺序 |
| gen6 K-ahead 环 + 合并窗口 + 再平衡（第 64 轮） | −10.5% / −12..−17% | 在 M 窗口里塞暂存 VALU 加长了关键路径 |
| 偏置路线：`biasacc` / `biasv` / `biasc` / `biasonce` / `vpre` / `kpre2` / `qkfirst`（第 66–68、73、75 轮） | −0.8%..−15% | 省下的 XDL 周期原样变成屏障 / 相位会合等待（`biasacc`：XDL −86、WAIT_ANY +91） |
| gen9 vendor 风格 M 窗口（固定顺序、V 预读、持久偏置 C，第 68–70 轮） | 未达 460 TF 门槛（−5..−8%） | 固定顺序的块强制 MFMA 顺序发射、`pin_check` 补 17 个 wait state、VGPR 255–256 |
| `p4` 非缩放 PV（第 70 轮，需要 `v_mfma_f32_32x32x64_f8f6f4`） | 单独 −1.2..−2.2%（保留 opt-in） | 数值正确但本身不提速，只是前置条件 |
| VALU 削减：`saddrlds` / `pmdead` / `mubufd` / `cut1`（第 67、71 轮） | 全部 ±1%；组合 `mubufd,cut1` 墙钟周期 +1.2–1.7%（VALU −8.7% 但 WAIT_INST_ANY +14%） | 循环不是 VALU 发射受限，移除的 VALU 槽位变成依赖停顿 |
| MUBUF K/V DMA（第 71 轮） | 位等价、中性（`mubufd` −0.5%，带分支版 −1.3..−2.1%） | 地址 VALU 已被隐藏；运行时大小描述符 + 硬件零填充是有价值的编译器新能力，内核转换保持 opt-in |
| gen10 有界周期（第 75 轮） | 硬门槛失败：墙钟周期与 XDL 需同时 ≥ 8% 下降，所有有效变体墙钟 +5..+15%，XDL 地板 1130 | 省下的 XDL 周期以屏障会合等待的形式回来：`qkzero`（XDL 1111、VALU 310）的墙钟仍是 1355 ≈ 基线 |

## 7. 对任何未来相位 / XDL 重设计的三项要求

第 73–75 轮的计数器给出了明确结论：这个 kernel 受**两个 wave 组的相位会合（barrier + LDS 等待）**限制，不是受 XDL / VALU / VGPR / 指令数限制。因此任何后续设计必须同时满足：

1. **必须让会合等待本身变短，而不只是移除矩阵或向量工作。** 省下的工作若不重排两个角色的相位长度，只会让该 wave 更早到 barrier 然后等待（WAIT_ANY 上升）。验收看**墙钟周期**（`SQ_WAVE_CYCLES`）而不是 XDL 或 VALU 计数：硬门槛 = 墙钟周期与 XDL 同时 ≥ 8% 低于同轮对照（B1H8S2048，对照 wave ≈ 1354–1373 / XDL 1216），并且 `SQ_WAIT_ANY` 不能同比上升。
2. **必须在 VGPR ≤ 256、调度器不回退的条件下成立，并避免强制顺序块的发射顺序与 wait-state 惩罚。** 已测量的限制：pinned safe-asm 窗口内 MFMA 按序发射、`pin_check` 在寄存器复用处补 17 个 wait state（71 对 22 个 NOP 状态 / tile）、VGPR 255–256 触发 `falling back to the unscheduled code`；kernel 无调度回退时才有可比性。
3. **必须保持逐位或精度门禁，并通过重复启动确定性认证。** 不改变数学的改动要求输出 md5 逐位一致；改变舍入的改动要求 15/15、`bad ≤ 厂商`、平均误差 ≤ 2.6e-3；任何触及宽存储 / 尾部的改动要通过 KV-split 不均匀形状的完整输出 md5 重复启动压力（≥ 300 次 × 4 形状）和 `epw` 探测器，而不是单次 5/5。

## 8. 未实现 / 范围外

- 完整的 Resource / Async IR、witness 驱动的 1:1 复刻管线（`design.md` §Witness）：仍只是设计。
- 跨块指令调度器、软件流水线的通用化（只有块内调度、`swp`、`taildup`）。
- 跨卡通信（MORI LSA / SDMA / GDA）、gfx1250 后端、JIT 与框架适配（DeepInfer / SGLang）。
- LSE 输出的对外接口（KV-split 内部使用 lse）；厂商的 `+0x80` KV 进度更新结构。
- 进程内汇编（LLVM bridge）：端到端经 `clang -x assembler` 子进程，不满足 ≤10 ms 目标；自研编译侧耗时在目标之内。

## 9. 复现

```bash
bash tools/container/cargo.sh test                      # 29 个目标、117 项
OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto BACKEND=own REF_ROWS=64 REF_CO=<厂商 .co> bash tests/gpu/run_ref.sh
python3 tools/gen9_gate.py <run_ref 日志>               # 精度 / vendor-parity 门禁
BACKEND=own bash tests/gpu/determinism.sh               # 5 个形状；认证还需要重复启动完整输出 md5 压力
```

重复启动压力与 `epw` 探测器的用法见 `docs/vendor-spec.md` §7.11 与 `docs/archive/status.md` 第 69 轮。
