# `mfmapair`：同累加器 MFMA 成对相邻（opt-in 调度器模式）

`crates/amdgpu-compiler/src/sched/mfmapair.rs`，开关：sched token `mfmapair` / `mfmapair=<预算系数>`（`Function::sched_opts`），或实验开关
`AMDC_MFMAPAIR=1`（所有内核）/ `AMDC_MFMAPAIR=<内核名或策略名子串>[:预算系数]`（`cargo.sh` 已转发）。默认关闭；关闭时汇编逐字节不变。测试：`crates/amdgpu-compiler/tests/mfmapair.rs`（7 项）。

## 做什么

用户方向："MFMA 两条一组，先用上 forwarding，然后第二条后面去插 VALU/SALU"。对每个块的最终顺序（list scheduler / slotfill 之后、等待重算之前）：
对每条 MFMA `i` 和它之后最近的、srcC 为 `i` 的 vDst 的 MFMA `j`：
* `i`、`j` 之间的指令分成两类：**j 依赖的**（它的其它操作数：ds_read、地址链……）上提到 `i` 之前；**其余的**（包括别的累加器的独立 MFMA）下沉到 `j` 之后。结果 `i`、`j` 之间什么都没有。
* 不做的情况：`j` 需要的某条指令依赖 `i` 的结果（无法相邻）；上提超过 12 条；`j` 之后到下一条 MFMA 之间的非 MFMA 工作量（模型发射代价）超过预算（`预算系数 × 第二条 MFMA 的流水线占用`，默认 1.0）——
  探针（docs/mfma-dep-probe.md §3）显示成对时第一条 MFMA 的影子是空的，能藏非 MFMA 的容量只有第二条的一个发射间隔。
* 依赖 / 冒险 / M0 / SCC / VCC / 栅栏全部走 `depgraph`；整块结果再用 `verify::check_perm` 校验，失败则整块保持原样；等待与冒险之后照常重算 / 插入。

## 静态效果（`tools/mfma_reuse.py`）

| 内核 | 相邻 MFMA 对：C=D(prev) 比例 基线 → mfmapair | 说明 |
|---|---|---|
| `fmha_fwd_hd128_mxfp8_v3_vs128` | 6.5 % → **48.8 %**（91 + 14 / 215） | 厂商 51 %（184 对） |
| `fmha_fwd_hd128_mxfp8_v3_t128` | 16 % → 30 %（27 + 9 / 119） | 预算跳过了一部分对 |
| `fmha_fwd_hd128_mxfp8_v3_kvsplit` | → 51.6 %（97 + 14 / 215；基线未单独记录） | |

## 硬件结果（GPU 4；`tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3`，`AMDC_MFMAPAIR=fmha_fwd_hd128_mxfp8_v3`，同一轮里先 mfmapair 后基线）

正确性：ACC16 / ACC64（4 条路径）/ DET / STRESS 全部通过（bit-identical 3×，与基线 md5 相同）。**性能全面变差**：

| 形状 / 内核 | 基线 TF/s | mfmapair TF/s | 变化 |
|---|--:|--:|--:|
| percu64（`_vs128` 强制） | 487.6 | 439.8 | −9.8 % |
| B4H8S2048（`_vs128`） | 1560.5 | 1453.5 | −6.9 % |
| B1H8S8192（`_vs128`） | 2212.1 | 2020.3 | −8.7 % |
| B1H8S2048（kvsplit） | 693.5 | 666.5 | −3.9 % |
| B2H8S2048（`_t128`） | 1050.5 | 1021.1 | −2.8 % |

预算系数扫描（`_vs128` 强制，5 轮交替，TF/s，基线 / 0.25 / 0.5 / 1.0 / 2.0 / 4.0 / 无限）：
* percu64：487.9 / 461.7 / 461.1 / 438.2 / 440.5 / 441.5 / 441.3
* B4H8S2048：1561.5 / 1522.3 / 1513.6 / 1454.6 / 1459.4 / 1460.0 / 1464.7
* B1H8S8192：2199.1 / 2115.2 / 2121.3 / 2008.2 / 2027.2 / 2015.6 / 2019.7

成对的对数越多越慢，没有任何预算比基线好；预算 0.25–0.5（只有 VALU 很少的对才成对）损失 3.5–5.4 %，≥ 1 时损失 8–10 %。

**解读**：成对让 MFMA 之间的 VALU/SALU 只能挤在第二条之后，丢掉了第一条 MFMA 的影子；探针也显示这一点（s32 在 k=16 时成对每对多 18 周期）。FMHA 的 softmax VALU 很重，把它们与不同累加器的
MFMA 交错（现状）比成对更好，forwarding 的时钟收益（s32 上 ≈ 0）也补不回来。结论：不建议作为默认；保留为 opt-in，供 `_t128` / `_vs128` 做 A/B（`AMDC_MFMAPAIR=<内核名>:<预算>`）。

另：之前"slotfill 在 FMHA v3 上中性"的实验里，`AMDC_SLOTFILL=fmha_fwd_hd128_mxfp8_v3` 因策略名不含 `_v3` 对 `_vs128` 实际没有生效（只有 `_t128` / kvsplit 生效）。已把开关改成同时匹配内核名，
重测 `_vs128`（强制，5 轮交替）：slotfill 487.5 / 1572.8 / 2202.1 → 486.1 / 1569.1 / 2207.8 TF/s（−0.3 / −0.2 / +0.3 %），仍然中性。