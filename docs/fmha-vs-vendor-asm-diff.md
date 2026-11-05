# FMHA `_vs128` / GEMM v4 与厂商汇编的逐项对照（寄存器 + 指令编排）

> 基线：Main `7eb79b2`。我们的汇编来自 `amdc emit fmha_fwd_hd128_mxfp8_v3_vs128 --backend=own`（md5 `f129690c8be7`）和
> `amdc emit gemm_mxfp8_bs_v4 --backend=own`（md5 `54c6aee3e432`）。厂商汇编来自对 `ref_mxfp8.co`（aiter FMHA）和 aiter PR #6696 4-wave
> pin-AGPR GEMM `.co` 的 `llvm-objdump -d --mcpu=gfx950` 反汇编；描述符取自 `llvm-readobj --notes`。
>
> 工具（本分支新增）：
>
> - `tools/asm_diff.py <ours.s> <vendor.dis>`：找出所有回边循环，选定稳态循环，按指令类、操作码、等待计数（含部分计数）、`s_nop` 等待状态统计每单元（128 key / 一个 K 段）的指令数，并给出压缩时间线。
> - `tools/asm_regmap.py <file> <loop>`：只根据操作数位置推断寄存器角色（O / S / K / Q / V / P / 缩放字 / 行和 …）。
>
> 下文的静态数字都来自这两个脚本；动态周期来自前一轮 warm ATT（同一默认 kernel，B4H8S2048，iteration [4]，CU 1，8 个 wave）。
> 不做推测：凡是没有测过的结论都会明确标注。

## 0. 一句话结论

- **GEMM v4：稳态循环与厂商几乎一一对应。** 两边的指令计数和顺序基本一样：每轮都是 128 个 scaled 16x16x128、68 个 `ds_read`、32 个 DMA、2 个 barrier、4 个 waitcnt。
  剩余差异是 SALU 地址推进的写法（73 vs 57 条），以及 acc 在 AGPR 与 VGPR 之间的划分（96/160 vs 84/172）。性能 0.985×，两边都受功耗墙限制。
- **FMHA `_vs128`：每 128 key 的稳态指令数为 388 vs 厂商 236（role A），VALU 254 vs 113。**
  - 差额的主要来源是**每 tile 都执行的 rebase**：32 条 `v_pk_add_f32`（S += δ）和 32 条 `v_pk_mul_f32`（O *= corr），ATT 计约 730 周期。厂商把 rollback 放在循环外，很少执行。
  - 其次是矩阵指令的组合：我们 16 个 scaled + 4 个 f16 bias + 4 个 smfmac，厂商 8 个 scaled（QK）+ 8 个 unscaled（PV）+ 1 个 bf16 + 2 个 smfmac。
  - 再次是约 45 条地址 / 常量 VALU。
  - 寄存器分配几乎同构：O 64、S 64、K/V 共享 32、Q 常驻 16、P 16。

## 1. 寄存器

### 1.1 描述符

| 项 | FMHA 我们 | FMHA 厂商 | GEMM 我们 | GEMM 厂商 |
|---|---|---|---|---|
| `.vgpr_count`（含 AGPR） | 252 | 246 | 424 | 456 |
| `.agpr_count` / `accum_offset` | 0 / 252 | 0 / — | 224 / 200 | 212 / — |
| `.sgpr_count` | 48 | **99** | 56 | **104** |
| LDS（group segment） | **102400 B** | 84224 B | 154112 B | 152064 B |
| scratch / spill | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

### 1.2 FMHA 稳态循环中的寄存器角色（`asm_regmap.py`）

数量单位为 VGPR，括号内是寄存器区间。

| 角色 | 我们 | 厂商 |
|---|---|---|
| O 累加器（PV 的 D） | 64（v16–79） | 64（v116–179） |
| S tiles（QK 的 D，`v_max3` 读） | 64（v80–143） | 64（v52–115） |
| K 片段（QK A） | 32（v160–175, 184–199） | 32（v20–51） |
| V 片段（PV A / `ds_read_b64_tr_b8` 目的） | 24，与 K 共享（v160–175, 184–191） | 32，与 K 共享（v20–51） |
| Q 片段（QK B，常驻） | 16（v8–15, 176–183） | 16（v4–19） |
| P（PV B，fp8 字） | 16（v144–159） | 16（v52–67，复用已转换完的 S） |
| MFMA 缩放字 | **12**（每个 S tile 单独抽出的 K scale 字） | **6**（scale 字整体保留，靠 `op_sel` 选字节） |
| 行和累加器（smfmac D） | 4（v200–203） | 4（v224–227） |
| smfmac A（ones）+ 索引 | 6 | 5 |
| bias / helper MFMA 操作数 | 4（f16 bias 两字） | 4（bf16 helper） |
| DMA voffset | 4 | 3 |
| 其余（softmax 状态、地址、常量、临时） | 19 | 8 |
| 循环内触及的 VGPR | 234（最高 v250） | 206（最高 v245） |
| 循环内触及的 SGPR | 19 | 20（另有大量 SGPR 用于 ring 地址：99 vs 48） |
| AGPR | 0 | 0 |

**对照要点（全部为静态事实）：**

- **两边都不用 AGPR。**
- **P 的存放不同。** 厂商把 P 写进已经转换完的 S 的低 16 个寄存器（v52–67）；我们的 P 单独占 16 个（v144–159）。
- **K scale 的取法不同。** 厂商在 scaled MFMA 上用 `op_sel_hi` 直接选 scale 字节，只保留 6 个缩放字。
  我们给每个 S tile 用 VALU 抽出单独的 scale 字（`v_lshrrev` 多 4 条/tile），占 12 个 VGPR。
- **SGPR 用法不同。** 厂商有 99 个 SGPR，ring 槽位与 M0 用 `s_cselect_b32` / `s_add_u32` 在 SALU 里轮转。
  我们只有 48 个 SGPR，一部分 ring 地址仍在 VALU 里（`v_add_u32` 21 vs 3/tile）。

### 1.3 GEMM 稳态循环

| 角色 | 我们 | 厂商 |
|---|---|---|
| acc（MFMA D） | AGPR 96（a128–223）+ VGPR 160（v32–191） | AGPR 84（a128–211）+ VGPR 172（v20–191） |
| A / B 操作数 | AGPR（原地重载，如 `a[120:127]`, `a[56:63]`） | AGPR（如 `a[64:71]`, `a[0:7]`） |
| 缩放字 | 3 个 VGPR（v1 v4 v16） | 8 个（v206–209, 212–215，`op_sel_hi` 选字节） |
| DMA voffset | 12 | 4 |
| 触及寄存器 | v 182 / a 224 / s 26 | v 204 / a 212 / s 47 |

## 2. 稳态循环指令计数

### 2.1 FMHA：每 128 key、每 wave（role A；厂商循环 2× 展开，已除以 2）

静态计数，长序列代码副本（不含 LUT 校正）。role B 的数字几乎相同（我们 370 vs 厂商 228）。

| 类 | 我们 | 厂商 | 备注 |
|---|---|---|---|
| **总指令** | **388** | **236** | |
| MFMA scaled 32x32x64 | 16 | 8 | 我们的 PV 也用 scaled（固定缩放 127/123）；厂商的 PV 用 unscaled |
| MFMA unscaled 32x32x64 | 0 | 8 | |
| bias/helper MFMA | 4 × f16 32x32x8 | 1 × bf16 32x32x8 | |
| `v_smfmac_f32_16x16x128` | 4 | 2 | |
| `ds_read_b128` / `ds_read_b32` | 16 / 4 | 16 / 4 | 相同 |
| `ds_read_b64_tr_b8` | 32 | 32 | 相同 |
| DMA（`buffer_load … lds`） | 4（+1 K scale `global_load_lds_dword`） | 4 + 1（K scale 也走 buffer） | |
| **VALU** | **254** | **113** | 见下行明细 |
| └ `v_pk_add_f32`（S += δ） | 32 | 0 | |
| └ `v_pk_mul_f32`（O *= corr） | 32 | 0 | 厂商的 O 重标定在循环外的 `rollback_overflow` 块 |
| └ `v_max3_f32` | 31 | 32 | |
| └ `v_cvt_pknorm_u16_f32` / `v_cvt_scalef32_pk_fp8_bf16` | 32 / 32 | 32 / 32 | 相同 |
| └ `v_mov_b32` | 24 | 1 | 常量重物化约 8、`v_mov v, s` 约 8、复制 |
| └ `v_add_u32` | 21 | 3 | 地址 / ring |
| └ cmp / cndmask / mul_lo / addc / readfirstlane | 约 17 | 约 2 | |
| └ f16 bias 两字更新（cvt_f16 / f32、sub、mul） | 约 17 | — | |
| SALU | 15 | 23 | 厂商的 ring 在 SALU：`s_cselect` / `s_cmp` / `s_add` |
| `s_waitcnt` | 19（`lgkmcnt(0)` 6，部分计数 12，`vmcnt(0)` 1） | 16（`lgkmcnt(0)` 1，部分计数 14，`vmcnt(0)` 1） | |
| `s_nop`（等待状态合计） | 10（31 态） | 1（2 态） | |
| `s_barrier` | 2 | 2 | 相同 |
| `s_setprio` | 2 | 3 | |
| 分支 | 5 | 2.5 | 我们循环内有 rollback 的 if |

**动态（warm ATT，同一默认 kernel）：** 每 tile 我们 4350 周期，厂商 3648 周期（+19%）。按每条指令发射到下一条发射之间的周期计：

| 项 | 我们 | 厂商 |
|---|---|---|
| `v_pk_add_f32` | 578 | 0 |
| MFMA 类合计 | 727 | 456 |
| `v_add_u32` | 132 | 30 |
| `v_readfirstlane` | 92（之后已改为每 tile 一次） | 0 |
| `s_nop` | 94 | 38 |
| `s_waitcnt` | 171 | 95 |
| barrier | 724 | 793（相当） |

### 2.2 GEMM：每轮循环（2 个 K 段，两边相同的展开）

| 类 | 我们 | 厂商 |
|---|---|---|
| 总指令 | 330 | 324 |
| scaled 16x16x128 | 128 | 128 |
| `ds_read`（b128 / b64 / b32） | 68（b64 2、b32 2） | 68（b64 4） |
| DMA | 32 | 32 |
| VALU | 12 | 5 |
| SALU | 73（`s_add_u32` 39） | 57（`s_mov_b32` 34、`s_add_i32` 11） |
| `s_waitcnt` | 4（`lgkmcnt(0)` 2、`vmcnt(0)` 2） | 4（另有 `lgkmcnt(14)` 2） |
| `s_nop`（等待状态） | 8 | 26 |
| barrier | 2 | 2 |

## 3. 编排（每 tile 时间线）

### 3.1 FMHA role A

记法：`x` 后面是连续条数；`wait[...]` 是 `s_waitcnt`。

**我们：**

```
softmax 相位:  4×DMA（K/V，buffer…lds；每条前有 s_mov m0 + s_nop）| setprio 0
             | VALU×63（max3 链 + 跨 lane max）| 分支（rollback if）| VALU×71（rollback：S+=δ 32 pk_add, O*=corr 32 pk_mul, l, bias 两字）
             | wait vmcnt(0) | wait lgkmcnt(0) | VALU×63（pknorm + scalef32 转换）
barrier → setprio 1
MFMA 相位:   4×f16 bias MFMA，与 K / kd / V 的 ds_read 交错（lgkmcnt 7→4）
             | 4×[wait lgkmcnt(0) → PV scaled MFMA → 4×ds_read_tr → smfmac]     ← 每个 PV 前都等到 0
             | QK：scaled MFMA ×12，K 读 2 条 / V tr 读 4 条，在 MFMA 之间以 lgkmcnt(2..8) 等待
             | wait lgkmcnt(0) | MFMA×4 | smfmac
barrier
```

**厂商（每 128 key 一遍；循环里有两遍）：**

```
softmax 相位:  setprio 0 | VALU×35（max3）| s_nop | 分支（阈值，罕见时跳到循环外 rescaler）| VALU×64（pknorm + scalef32）
             | 8×ds_read_tr（提前读本 tile 的 V）| wait vmcnt(0)
setprio 1 → barrier
MFMA 相位:   wait lgkmcnt(4) | 2×smfmac | 8×[unscaled PV MFMA, 每条之后 4×ds_read_tr，wait lgkmcnt(4)]   ← 始终领先一组
             | 1×bf16 helper MFMA
             | 8×[scaled QK MFMA；K 读 2 条，wait lgkmcnt(2)；其间穿插 5×DMA（下一 tile 的 K / V / kscale，每条前 s_add m0 + s_nop）]
             | wait lgkmcnt(0) | 最后 1 条 MFMA
setprio 0 → barrier
```

**编排差异（静态事实）：**

1. **DMA 位置。** 我们把 4 条 DMA 集中放在 softmax 相位开头（setprio 0 之前）；厂商把 5 条 DMA 分散在 QK MFMA 之间。
2. **LDS 读领先量。** 厂商的 PV 读始终领先一组（`lgkmcnt(4)`），每个 tile 只等到 0 一次。
   我们在 PV 段前 4 个 MFMA 前各等一次 `lgkmcnt(0)`，每 tile 共 6 次等到 0：读出的数据马上被消费，与 MFMA 之间没有间隔。
3. **V 读的提前量。** 厂商在 softmax 相位末尾就发出 8 条 V 的 tr 读，越过 barrier 提前。我们的 V 读全部在 barrier 之后。
4. **rollback 位置。** 我们的 rollback 是循环内的 if，约 71 条 VALU，实测 15/16 个 tile 都会执行。厂商的 rescaler 在循环外，只在越过阈值时跳过去。
5. **s_nop。** 我们在每条 DMA 前加 `s_nop`（M0 冒险），并在 MFMA 之间有 2–3 个 `s_nop`，每 tile 共 31 个等待状态。
   厂商每 tile 只有 2 个等待状态：DMA 前的 M0 用 SALU 计算，紧邻处没有插 nop。

### 3.2 GEMM

两边的时间线在结构上一致：

```
barrier 后 wait(vmcnt0, lgkm0) → MFMA×3–5 → 16×DMA 均匀穿插在 MFMA 之间（每条前 SALU + s_nop）
→ 后半段 ds_read×2 / MFMA×1–4 交替 → 回边
```

区别：

- 我们回边处多 6–19 条 SALU 地址推进（`s_add_u32` 39 vs 厂商 0）。
- 厂商在循环顶部有 `wait lgkmcnt(14)`（部分等待）；我们是 `wait vmcnt(0)` + `lgkmcnt(0)`，各 2 次。

## 4. 剩余差异（按测得 / 估计的周期排序）

| # | 差异 | 量 | 状态 |
|---|---|---|---|
| 1 | 每 tile rebase：`S += δ`（32 `pk_add`）+ `O *= corr`（32 `pk_mul`）+ l / bias，循环内执行（15/16 tile） | ATT 约 578 + 约 150 周期/tile | **已关（精度取舍）**：快路径（转换缩放折叠、整数 δ）快 1–1.4%，但不满足厂商一致性（bad 20 vs 2）。混合方案满足一致性但慢 2–4%。我们 3× 于厂商的精度正来自每 tile 对齐最大 key（`docs/current-status.md` §4.3 表） |
| 2 | 矩阵组合：PV scaled vs unscaled、4 f16 bias vs 1 bf16、4 smfmac vs 2 | XDL 1216 vs 1088/wave/tile；ATT MFMA 类 727 vs 456 | bias 去除（biasacc/biasonce 等）**已关，负**（−0.8…−15%）；unscaled PV（`p4`）**已关，负**（−1.2…−3.5%）；2 smfmac **未做**（需要厂商 P 打包 + V tr 布局建模，预计 ≤1%，选项 A） |
| 3 | 地址 / 常量 VALU：`v_mov` 24 vs 1、`v_add_u32` 21 vs 3、cmp / cndmask 等 | 约 45 条/tile，ATT 约 250 周期 | LICM（backend）**已关，负**（−3…−8%，保持活跃的代价更大）；MUBUF DMA 和每 tile 一次 readfirstlane **已做**；ring 地址全部移到 SALU（厂商 99 SGPR 的做法）**未做** |
| 4 | LDS 读等待：每 tile 等到 0 的次数 6 vs 1，PV 读没有领先一组 | ATT `s_waitcnt` 171 vs 95 周期 | `pvpipe` / `qkfirst`（第 52 轮）**已关，负**（−0.5…−5%）；按厂商形状（V 读跨 barrier 提前、PV 领先一组）**未在当前结构上重测** |
| 5 | DMA 位置：集中在 softmax 相位开头 vs 分散在 QK MFMA 之间 | 未单独测 | **未做**；slotfill（sched）在 vs128 上无改动 |
| 6 | `s_nop` 等待状态 31 vs 2 | ATT 94 vs 38 周期 | 归因（`tools/nop_attrib.py`，`_vs128` 稳态循环 31 态）：循环尾 15（MFMA 结果跨回边，需调度把 MFMA 前移）、MFMA→`ds_read_b128` 11（寄存器复用：ds_read 写入仍在飞的 MFMA 正在读的物理寄存器，`pad_physical_war`）、M0 写→LDS-DMA 5。新增 `FMHA_OPT=m0reuse`（GEMM 的 M0 复用）+ `m0hoist`（M0 写上移到独立指令之前，免 `s_nop 0`）：`_vs128` 31→28、`_t128` 10→2 态；port_gate 全过但性能中性（layout-median −0.1% / −0.2%，KV-split +2.8%）→ **保持 opt-in** |
| 7 | K scale 抽字：每 S tile 用 VALU 抽字（12 VGPR）vs `op_sel` 选字节（6 VGPR） | 约 4 VALU/tile + 6 VGPR | **编译器已支持**：`var_mfma_scale_sel(var, a, b, sa, sb, byte_a, byte_b)` / tile `gemm(.., byte_a, byte_b)` 对 32x32x64 生成 `op_sel/op_sel_hi`（own + LLVM 两个尾端；硬件探针 `tools/mfma_opsel_probe.sh` 两个尾端均 64/256/16/1024 精确）；内核侧改用待 kernel-opt |
| 8 | P 寄存器复用 S（厂商）vs P 单独 16 VGPR（我们） | 16 VGPR | **未做**（寄存器压力并非当前瓶颈：252 vs 246） |
| 9 | LDS 102400 vs 84224 B | 每 CU 只驻留 1 个 block，不影响占用 | 信息 |
| 10 | 尾声 | 已对齐（8 × 16 B 行存储） | **已做**（+3.9%） |
| — | GEMM：SALU 地址推进 73 vs 57、nop 8 vs 26、acc 划分 | 0.985× 厂商，功耗墙 | 收尾，无已知可测收益 |

**当前门禁数字**（`port_gate`，厂商一致性准则，GPU 5）：B4H8S2048 0.882×，B1H8S8192 0.859×，per-CU 0.879×，B1H8S2048 split 1.26×，B2H8S2048 `_t128` 1.03×。
