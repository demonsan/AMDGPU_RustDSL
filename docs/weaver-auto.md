# 基于代价模型的自动编织策略（`weave_auto`）

状态：**可选策略，未采用为默认**（分支 `weave2-auto`；代码 `crates/amdgpu-tile/src/weave_auto.rs`，测试 `crates/amdgpu-tile/tests/weave_auto.rs`）。默认行为不变（所有 kernel 的 asm md5 与 Main 相同）；全部开关都是环境变量，只在编织的 `emit` 里读取。

## 做了什么

输入是一个编织相位的单元（`Units`）或被 capture 的语句（`Captured`，[`weave-capture.md`](weave-capture.md)）；输出是一个满足依赖的顺序。

- **机器模型**（`Model::gfx950()`）：单 wave 顺序发射，每条指令占 `issue_gap = 4` 周期；MFMA 管线占用 64 周期（32x32x64，按 XDL passes × 4），结果 84 周期后可读；`ds_read` 结果 125–145 周期，按顺序返回，在飞超过 8 条后加 200 周期的队列惩罚；VALU 结果 8 周期，`cvt_scalef32` / trans 占两个发射槽；LDS-DMA 发射 24 周期；barrier / 计数等待 / 分支（`fence`、`drain`）先等所有在飞的操作完成。参数来自 `docs/lgkm-lookahead.md` 的校准。
- **规划**（`plan_items`）：贪心 list scheduling（选发射后最早完成的就绪项），再做局部搜索（把单个项移到别的合法位置，窗口 40，目标 = 模型周期，超出寄存器预算则重罚）；声明顺序是兜底。`fixed` 项（anchor、分支）保持声明相对顺序，不被搜索移动。
- **寄存器护栏**：单元级用 `RegProfile` / 估计的活跃增量；语句级是**值级**的（每个值定义时占 `dwords`，最后一个使用者发射后释放），峰值不得超过声明顺序的峰值 + `reg_slack`（16 dword）。
- **单元级**：`auto_order(&units, &Model)`，`Unit::cost`（`Units::cost`）给出每个单元的指令构成（没有时用按名字 / 类别的临时估计）；`AMDC_WEAVE_AUTO=<相位标签>[,..]|*` 让一个声明顺序的编织改用它（显式 `Weaver::order` 永不被覆盖）。
- **语句级**：`auto_order_nodes(f, &captured, &Model)`；合法性完全来自 `facts::must_order`（数据、循环变量、anchor、内存、分支）；weave-region 标记对保持连续；`AMDC_WEAVE_AUTO_STMT=<标签>[,..]|*` 在 `Weaver::emit` 里 capture 该相位的所有单元、重排其语句、再 splice。调试用：`AMDC_WEAVE_AUTO_VERBOSE=1`（打印计划与模型周期 / 峰值）、`AMDC_WEAVE_AUTO_PREFIX=N`（只采用计划的前 N 项，其余按声明顺序；`0` = 恒等，用来二分）。

## 实测（`REF_DATA=normal`，同一轮交替，GPU 4，TF/s 中位数；"基线" = 声明顺序 + 完整 list scheduler）

`_vs128`（`ROWS_PER_BLOCK=256`，B4H8S2048 / B1H8S8192）：

| 变体 | B4H8S2048 | B1H8S8192 |
|---|--:|--:|
| 基线 | 1893 | 2462 |
| 单元级 `AMDC_WEAVE_AUTO=m_phase` | 1887 | 2466 |
| 单元级 + `m_phase:authored` | 1880 | 2353 |
| 语句级 `m_phase` | 1857 (−2.4 %) | 2392 (−2.9 %) |
| 语句级 `m_phase` + authored | 1832 | 2347 |
| 语句级 `v_phase` | 1920 (+0.9 %，噪声内) | 2462 (±0) |
| 仅 authored（声明顺序） | 1856 | 2355 |

结论：模型里 m_phase 的周期有明显下降，硬件上单元级持平、语句级变差、authored 版本与声明顺序的 authored 同量级（慢 4–5 %）：模型在 ≲3800 周期 / 迭代以下不再预测硬件（单 wave 的 FMHA 循环受发射数与共享 LDS 读带宽约束，见 `lgkm-lookahead.md`）。`v_phase` 只有一条依赖链（decide → convert）几乎没有自由度，语句级的收益在噪声内，且编译时间从 5 秒升到 128 秒（语句数 > 500 时局部搜索代价大）。

`_t128` 的 `window`：单元级（声明依赖）与语句级（`must_order` 推导的依赖）**都产生错误结果**（B2H8S2048 `bad=656…1895`，gpu-verify 仍是 0 错误），所以速度数字（+3.5 % 单元级）无效。二分（`AMDC_WEAVE_AUTO_PREFIX`）：只采用计划的前 48 项正确，前 52 项开始错误 —— 第 49–51 位把三条 `BitcastF32(Frag16Lane{Var(6), 9..11})` 挪到对应三条 `Frag16Lane`（对循环变量 6 的寄存器视图，lowering 里是 `alias_vreg`）之后。把提取与 bitcast 绑成一个组、把"视图的所有使用者必须先于变量的下一次写"加成依赖，都没有消除错误：说明还有 `facts` 没有表达的约束（或 lowering / 分配对这类视图的假设）。这是自动语句级编织的**合法性缺口**，需要 backend 侧确认后才能把语句级策略用于有 `if` 回滚区域 / 变量视图的相位。

GEMM tile：当前单元带 `last` 链依赖（`mma.i*j*` 一条链），声明依赖里没有可重排的自由度；语句级需要上述缺口先关闭，未测。

## 采用与否

不采用。单元级策略在模型里更好、在硬件上不比声明顺序好（vs128 持平）；语句级在 vs128 上更差，在 t128 上暴露合法性缺口。作为可选策略保留（默认关、环境变量开启、有测试）：对没有 LDS 带宽瓶颈的新 kernel（例如 epilogue / 归一化相位）可以先用 `AMDC_WEAVE_AUTO=*` 试，再决定是否固化顺序。

## 接口

- `Model::gfx950()`；`UnitCost`（指令构成）；`Item` / `plan_items`（通用规划器）；`auto_order`（单元）；`items_from_nodes` / `auto_order_nodes` / `reorder`（语句）；`simulate` / `simulate_items` / `simulate_nodes`（模型周期与峰值）。
- `Unit::cost` 与 `Units::cost`；`Weaver` 的 `explicit` 标志（`order` / `spread` 给出的顺序不被开关覆盖）。