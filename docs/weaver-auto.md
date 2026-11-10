# 基于代价模型的自动编织策略（`weave_auto`）

状态：**可选策略，未采用为默认**（分支 `weave2-auto`；代码 `crates/amdgpu-codegen/src/weave_auto.rs`，测试 `crates/amdgpu-codegen/tests/weave_auto.rs`）。默认行为不变（所有 kernel 的 asm md5 与 Main 相同）；全部开关都是环境变量，只在编织的 `emit` / `plan` 里读取。

## 做了什么

输入是一个编织相位的单元（`Units`）：每个单元都是被 capture 的 IR，带语句事实（`Unit::stmts`、`Unit::summary`）和由 `facts::must_order` 推导的依赖（[`weave-capture.md`](weave-capture.md)）；输出是一个满足依赖的顺序。`Weaver::fine()` 把单元拆成语句组后，同一个规划器就是语句级的。

- **机器模型**（`Model::gfx950()`）：单 wave 顺序发射，每条指令占 `issue_gap = 4` 周期；MFMA 管线占用 4 × passes 周期（32x32x64 = 64），结果 +84 周期可读；`ds_read` 结果 125–145 周期，按顺序返回，在飞超过 8 条后加 200 周期的队列惩罚；VALU 结果 8 周期，trans 占两个发射槽；LDS-DMA 发射 24 周期；barrier / 计数等待 / 分支先等所有在飞的操作完成。参数来自 [`lgkm-lookahead.md`](lgkm-lookahead.md) 的校准。
- **规划**（`plan_items`）：贪心 list scheduling（选发射后最早完成的就绪项），再做局部搜索（把单个项移到别的合法位置，窗口 40，目标 = 模型周期，超出寄存器预算则重罚）；声明顺序是兜底；anchor 和分支（`fixed`）保持声明相对顺序、不被搜索移动。
- **寄存器护栏**：值级——单元定义的值按其 footprint 占寄存器，最后一个使用者发射后释放；峰值不得超过声明顺序的峰值 + `reg_slack`（16 dword）。
- **接口**：`auto_order(&units, &Model)`、`simulate(&units, &order, &Model)`、`items_from_units`、`plan_items`。单元的指令构成来自它的语句类别（`ResClass`），不再有按名字的估计。
- **开关**（`Weaver` 的显式 `order` / `spread` 永不被覆盖；标签 = 相位名，即 stage 名）：`AMDC_WEAVE_AUTO=<标签,..|*>` 单元级；`AMDC_WEAVE_AUTO_STMT=<标签,..|*>` 语句级（`fine()` 之后）；调试：`AMDC_WEAVE_AUTO_VERBOSE=1` 打印计划与模型周期 / 峰值，`AMDC_WEAVE_AUTO_PREFIX=N` 只采用计划的前 N 项（其余按声明顺序，`0` = 恒等，用来二分）。

## 一条合法性教训

语句级第一次在 `_t128` 的 `window` 上产生错误结果（`bad=656…1895`，gpu-verify 0 错误）。二分（`AMDC_WEAVE_AUTO_PREFIX`）定位到循环变量的寄存器视图：`Frag16Lane{Var}` 与它的 `BitcastF32` 在 lowering 里是对变量寄存器的 alias，**所有**使用者（传递地）在使用位置读该变量；facts 只记录了提取处的读。backend 的修复（`FnBuilder::facts` 把被视图引用的变量加进每个使用者的 `var_reads`，`view_vars`）合入后，`window` 在语句级的随机合法顺序全部 `bad=0`。这类"语句事实没有表达的别名"正是自动重排的风险点；任何新的自动策略都应先用 `AMDC_WEAVE_AUTO_PREFIX` 二分和实机精度比较验证。

## 实测（`REF_DATA=normal`，同一轮交替，GPU 4，TF/s 中位数；基线 = 声明顺序 + 完整 list scheduler）

`_vs128`（`ROWS_PER_BLOCK=256`，7 轮）：

| 变体 | B4H8S2048 | B1H8S8192 |
|---|--:|--:|
| 基线 | 1860–1878 | 2461–2462 |
| 单元级 `m_phase` | 1892 | 2443 (−0.7 %) |
| 语句级 `m_phase` | 1849–1871 | 2390–2392 (−2.9 %) |
| 单元级 `v_phase` | 1900 | 2462 |
| 语句级 `kv.stage` | 1907（+2.5 %，在基线噪声 1800–1896 之上） | 2454 (−0.35 %) |
| 语句级 `v_phase` | ≈ 持平，编译 129 秒（语句数多，局部搜索贵） | ≈ 持平 |

`_t128`（`ROWS_PER_BLOCK=128`，7 轮，B2H8S2048 / B1H8S4096）：基线 1197 / 1419；单元级 `window` 1193 / 1400（−0.6 / −1.4 %，精度 `bad=0`）；语句级 `window` 1150 / 1345（**−3.9 / −5.2 %**，精度 `bad=0`）；语句级 `window` + `window:authored` 1179 / 1403（−1.5 / −1.1 %）。

GEMM v4（4096 × 4096 × 8192，7 轮，`GEMM_OK 7/7`，证书不变）：基线 3049.5；单元级 `gemm.tile` 3026.9，语句级 `gemm.tile` 3025.9，单元级 `gemm.ab_stage` 3044.1，语句级 `gemm.ab_stage` 3028.1（−0.2 … −0.8 %，噪声边缘）。

RMSNorm n4096（M = 16384，5 轮，VALU 多但内存带宽受限，5.3–5.5 TB/s）：基线 5327 GB/s，单元级 `*`（asm 与基线相同，无自由度）持平，语句级 `*` 5360（+0.6 %，噪声内）。

结论：模型里周期总是明显下降（例如 t128 `window` 4088 → 2156），但硬件上没有一项在多形状上稳定变好；语句级对 `_t128` 是 −4…−5 %，`kv.stage` 语句级在 B4H8S2048 上 +2.5 % 但在 B1H8S8192 上 −0.35 %，不足以采用。模型在 ≲3800 周期 / 迭代以下不再预测硬件：单 wave 的 FMHA 循环受发射数与共享 LDS 读带宽约束（见 `lgkm-lookahead.md`），而 GEMM / RMSNorm 的瓶颈是 MFMA / 内存，不是单元顺序。

## 采用与否

不采用。保留为可选策略（默认关、环境变量开启、有测试）：对没有 LDS 带宽 / 内存瓶颈、阶段内自由度大的新 kernel 可以先试 `AMDC_WEAVE_AUTO=*` 看模型和硬件是否一致，再决定是否固化顺序。