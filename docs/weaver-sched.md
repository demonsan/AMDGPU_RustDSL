# 按编织区域（weave region）选择调度策略

状态：已实现并通过测试（分支 `weaver-sched2`，基于 weaver2 = Main + weaver + weaver-regs）。**没有声明区域时一切不变**（50 个 kernel 的 asm md5 逐个相同）。

## 区域从哪来

区域标记由 backend 的 `weave_region_begin(name)` / `weave_region_end()` / `weave_region(name, f)`（`Stmt::RegionMark`，不生成代码，可嵌套，也可以是 stage 名）产生，Weaver 对每个带标签的编织都会发出它。
lowering 给区域内生成的每条 MIR 指令打上区域号 `MInst::region`（`MFunction::reg_hints.regions` 的 1-based 下标，**每个作用域实例一个号**）。

## 策略按**名字**声明

```rust
b.declare_region("m_phase", /*authored*/ true, /*tokens*/ "ldsink=1");
```

- `authored = true`：编织出来的顺序是权威。区域内**不做** list 调度、`slotfill`、`ldshoist`（token 也不能把它们打开）；仍然重算计数等待（`rewait`），并允许保序的后处理（`ldsink`、`wsink`、`ldahead`、`ldahead32`）。
- `authored = false`：区域的 token（`AMDC_SCHED_OPTS` 词汇，新增 `reorder` = 打开 list 调度）叠加在 kernel 的选项之上，只作用于该区域的指令。
- 没有声明的区域继承它所嵌套的区域的策略（最外层 = kernel 默认）。

`Options::regions[k-1]` 是第 k 个声明，`Options::region_of[id]` 把 lowering 的区域号映射到策略号（`amdgpu-compiler/src/lib.rs` 的 `region_strategies`）。

## 调度器怎么用

- list 调度器：一个基本块里按**策略**切成 run（同一声明的相邻实例是一个 run），每个 run 单独调度（块的活跃集合给出 run 两端的活跃值，压力模型仍然精确）；authored / `rewait` 的 run 保持顺序。
- 后处理（`ldsink` / `ldahead` / `ldahead32` / `wsink`）：按指令所属策略的选项生效，**遇到策略变化就停**，不跨区域边界移动。
- `ldshoist` / `slotfill`（整块 pass）：只处理其中所有指令的策略都开启它们的块。
- 区域可以包含控制流（回滚 `if`）：区域覆盖其子树 lower 出来的各个块，每个块有自己的 run。

## 实验开关

`AMDC_SCHED_REGION=<区域名>:<authored|free>[:<tokens>][;或|...]`：不改 builder 就能声明策略（附加在 kernel 声明之后，同名后者覆盖；`cargo.sh` 已转发；`|` 也能做分隔符，便于 `tools/lgab.sh` 这类按 `;` 切分的脚本）。
例：`AMDC_SCHED_REGION=m_phase:authored|v_phase:authored`。

## 首批数据：`_vs128` 的 authored 区域（REF_DATA=normal，GPU 4，同一轮交替 5 轮，精度 `bad=0`）

| 声明 | per-CU B1H8S2048 | B4H8S2048 | B1H8S8192 |
|---|--:|--:|--:|
| （无，kernel 默认 list 调度） | 652.4 | 1744.9 | 2465.2 |
| `m_phase:authored` | 585.2 (-10.3 %) | 1690.4 (-3.1 %) | 2346.4 (-4.8 %) |
| `v_phase:authored` | 674.1 (+3.3 %) | 1738.7 (-0.4 %) | 2448.3 (-0.7 %) |
| `m_phase:authored` + `v_phase:authored` | 671.4 (+2.9 %) | 1759.5 (+0.8 %) | 2434.7 (-1.2 %) |

也就是说：Weaver 的声明顺序做 m_phase 时比 list 调度器差 3–10 %（list 调度器在该区域里确实有用），`v_phase`（decide + convert）authored 对 per-CU 有 +3 % 但对大形状略负。
`pexact` / `search` / `ovl` 只在 1-wave kernel 上有收益，尚未移植到 Main（`w4sched2`）；`hzfill` 还在 `fmha-4w2`，有了之后按区域选择它同样是区域选项。

测试：`crates/amdgpu-compiler/tests/sched_regions.rs`（run 划分、authored / free、token 只在区域内生效、后处理不跨边界、authored 不能被 token 解锁、策略号映射）、`tests/sched_regions_lower.rs`（按名字声明、嵌套继承、两个实例同一策略）。