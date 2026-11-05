# 文档归档（historical records）

这里的文件是**历史记录**：设计研究、逐轮实现日志、已关闭的各代实验、阶段性审计与终局报告。它们记录"当时为什么这么做、测到了什么"，
数字与结论只代表写作时的状态，**不再与当前代码同步**。当前状态以 [`../current-status.md`](../current-status.md) 为准；
文件内部的相对链接已改为指向仍在 `docs/` 下的文档（`../x.md`）或本目录内的其他归档文件。

## 文件清单

| 文件 | 是什么 | 时间 / 轮次 | 结局 | 被什么取代 |
|---|---|---|---|---|
| [`status.md`](status.md) | 逐轮实现记录（所有正负结果与归因，184 KB） | 第 1–75 轮（2026-10-27 至 2026-11-01） | 收官于 tag `v2.2-round75-final` | [`../current-status.md`](../current-status.md)（权威快照）；之后的工作记录在各设计文档的 as-built 节 |
| [`final-report.md`](final-report.md) | 第 75 轮终局总结：目标 → 现状 → 证据 → 差距 | 第 75 轮，Main `8fec359` | 旧手写集为出厂集时的终局 | [`../current-status.md`](../current-status.md)（出厂集已换为 v3） |
| [`implementation-plan.md`](implementation-plan.md) | 最初实施计划逐项对照到第 75 轮状态 | 第 75 轮 | 计划已完成或关闭 | [`../design.md`](../design.md)、[`../current-status.md`](../current-status.md) |
| [`kernel-gen2-design.md`](kernel-gen2-design.md) | gen2：单 wave 内 VALU/MFMA 重叠（`_g2`） | 第 57 轮 | 关闭（正确但不更快） | — |
| [`gen3-design.md`](gen3-design.md) | gen3：每 tile 合并单窗口 + 偏移 barrier（`_g3`） | 第 57–58 轮 | 关闭（更慢） | — |
| [`gen4-tiling-design.md`](gen4-tiling-design.md) | gen4：128 行 block 分块 | 第 59 轮 | **采纳**：出厂为 `_t128`（今天的 v3 `_t128` 沿用该分块） | v3 `_t128`（`crates/amdgpu-kernels/src/fmha/mod.rs`） |
| [`gen5-window-spec.md`](gen5-window-spec.md) | gen5：手写（safe-asm）内层 tile 窗口 `_vs5` | 第 63 轮 | 关闭（Block C −2.5%，M 中立） | — |
| [`gen6-rebalance.md`](gen6-rebalance.md) | gen6 输入：厂商 V/M 划分与 VALU 再平衡 | 第 64 轮 | 关闭（见 gen6-results） | — |
| [`gen6-results.md`](gen6-results.md) | gen6：K-ahead ring + 合并窗口 + 再平衡（`_vs6`） | 第 64 轮 | 关闭（−10.5% / −12..−17%） | — |
| [`gen9-vendor-mirror-design.md`](gen9-vendor-mirror-design.md) | gen9：按厂商反汇编镜像的 FMHA 设计 | 第 70–73 轮 | 关闭（−5..−8%；gen10 硬门槛失败） | — |
| [`occupancy.md`](occupancy.md) | 4 waves/SIMD 占用率假设（`_t`） | 第 46 轮（及 71–72） | 关闭（预算不允许，厂商同为 1 block/CU） | — |
| [`v-window-cut-plan.md`](v-window-cut-plan.md) | V 窗口 VALU 审计与削减计划 | 第 71–72 轮 | 关闭（VALU −8.7%，wave 周期 +1.2%） | — |
| [`unscaled-pv-design.md`](unscaled-pv-design.md) | 不带缩放的 PV 数值设计（`FMHA_OPT=p4`） | 第 70 轮 | opt-in 原型（单独 −1.2..−2.2%） | — |
| [`kvsplit-design.md`](kvsplit-design.md) | KV-split 设计研究（序列并行 + 精确跨块合并） | 第 60 轮前后 | **采纳**：KV-split + `combine8h` 是默认路径之一 | v3 `_kvsplit` / `_kvsplit_combine8h` |
| [`kvsplit-results.md`](kvsplit-results.md) | KV-split Step A/B 结果（含 last-arriver 融合 combine 的负结果、epilogue 损坏根因） | 第 60–62、69 轮 | 落地；融合 combine 关闭 | v3 KV-split；[`../last-arriver-memory-model.md`](../last-arriver-memory-model.md) |
| [`vendor-structure.md`](vendor-structure.md) | 厂商 FMHA 相位结构与当时 `_vs128` 重写的对照 | 第 60 轮前后 | 结论已并入 vendor-spec | [`../vendor-spec.md`](../vendor-spec.md) |
| [`verification.md`](verification.md) | 独立验证：寄存器记账、后端等价、DMA（kernel-occ） | 第 47 轮，Main `ae003a9` | 发现的 bug 已修复并有回归测试 | [`../current-status.md`](../current-status.md) §6–§7 |
| [`dsl-tile-audit.md`](dsl-tile-audit.md) | tile 库移植（`dsl_*`）的独立审计 | 2026-11-02，`port-audit` | 审计通过，`dsl_*` 出厂（后被 v3 取代） | [`../dsl-layout-refactor-design.md`](../dsl-layout-refactor-design.md) §9 |
| [`gpu-semantics-audit.md`](gpu-semantics-audit.md) | gpu_verify v4 四轴协议模型的独立对抗审计记录 | Main `7b8c5e0` | 发现的 2 个 soundness 漏洞已修复 | [`../gpu-semantics-verifier.md`](../gpu-semantics-verifier.md)、[`../gpu-semantic-protocol.md`](../gpu-semantic-protocol.md) |

## 已删除

本次清理没有删除任何文件：所有被取代的文档都还有历史价值（实验的反例与归因），因此归档而不是删除。
