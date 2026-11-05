# LDS 读领先量实验（`_vs128`，真实数据 `REF_DATA=normal`）：`ldahead` / `ldsink`

实现：`crates/amdgpu-compiler/src/sched/ldahead.rs`（调度后处理，作用于每个基本块的最终顺序，之后照常重算等待）。token：`ldahead=D`（每条 `ds_read` 上移，直到它与首个消费者之间有 D 条 MFMA；`ldslack=S`
允许比原峰值多用 S 个 VGPR dword，默认 0）、`ldsink=D`（反方向：下沉到与消费者之间至多 D 条 MFMA）。依赖 / 冒险 / 栅栏走 `depgraph`，结果用 `verify::check_perm` 复核。
无环境变量用法：`Function::sched_opts` 的 token；注册了 `fmha_fwd_hd128_mxfp8_v3x_vs128_ldsink{1,2,..}` 与 `..._v3x_kvsplit_ldsink{N}`（与出厂 v3 同码，仅多一个 token）。
实验开关：`AMDC_SCHED_ADD=<内核名子串>:<tokens>`（叠加在内核自己的出厂调度预设之上；`cargo.sh` 已转发）。工具：`tools/lgab.sh`（`tools/ab_fmha.sh` 加每变体的 emit 环境：`kernel%VAR=val;..`）。

## 现状（Main `b6b2ca59` 附近）

`_vs128` 稳态循环的 MFMA 相位（`s` = scaled MFMA，`T` = `ds_read_b64_tr_b8`，`R` = `ds_read_b128`）：`... T×8 W(8) Ms T×4 W(8) Ms T×4 W(8) Ms ... W(4) Ms ... R.. W(5) .. W(3) ..`：PV 段已经是部分等待
（`lgkmcnt(8)`），整块只剩 2–3 个 `lgkmcnt(0)`。**count-based 的部分等待早已是现状**（scoreboard 按发射顺序算），所以实验 (2) 没有可调的余量；剩下的是发射**顺序**。
厂商是每条 PV MFMA 之后读"下一组"，等待 `lgkmcnt(4)`（一组在飞）；我们是 `lgkmcnt(8)`（两组在飞，即**更深**）。lgkmcnt 只有 4 位（≤ 15），在飞读超过 15 条时等待只能退化成 `lgkmcnt(0)`。

## 结果（GPU 4，`REF_DATA=normal`，同一轮交替；TF/s 中位数，7 或 11 轮；全部精度检查 `bad=0`，证书 6 / 29 / 0 不变）

| 变体 | B4H8S2048 | B1H8S8192 | 备注 |
|---|--:|--:|---|
| 基线（出厂 `_vs128`） | 1696.3 / 1700.6 / 1704.1 | 2369.7 / 2375.8 / 2373.6 | VGPR 256，`lgkmcnt(0)` 49 个 |
| `ldshoist` / `hoist=128`（调度前的 ds_read 上提） | 1673.9 / 1679.7 | 2329.4 / 2334.7 | −1.3 … −1.7 %，VGPR 244 |
| `limit=248`（调度器压力上限） | 1682.5 | 2339.1 | −0.8 … −1.3 % |
| `ldahead=1 / 2`（更深：上移到 1 / 2 条 MFMA） | 1693.7 / 1694.4 | 2374.6 / 2364.7 | 持平（−0.2 … −0.4 %） |
| `ldahead=4` | 1667.1 | 2323.0 | −1.7 % |
| `ldahead=8` | 1609.1 | 2242.2 | −5.1 %（`lgkmcnt(0)` 49 → 202：超过 15 条在飞，等待退化） |
| `ldahead=4,ldslack=4` | 1668.5 | 2323.7 | 同 `ldahead=4` |
| **`ldsink=1`（更浅：与厂商同样一组在飞）** | **1706.8 / 1707.4（+0.2…+0.4 %）** | **2388.4 / 2391.5（+0.5…+0.8 %）** | VGPR 256 → 248，**每轮 ≥ 基线** |
| `ldsink=2` | 1702.0 / 1698.9 | 2381.4 / 2380.1 | 持平 |
| `ldsink=3` | 1697.6 | 2369.8 | 持平 |

per-CU（B1H8S2048，`ROWS_PER_BLOCK=256` 强制 `_vs128`，11 轮）：基线 628.6，`ldsink=1` **633.3（+0.75 %）**，`ldsink=2` 628.0。

**结论**：更深的读领先量只会变差（寄存器已在 256，且 lgkmcnt 4 位上限）；方向反过来 —— 读再贴近消费者、与厂商一样只留一组在飞（`ldsink=1`）—— 有小而稳定的收益（+0.2 … +0.8 %，并释放 8 个 VGPR），
作为 `fmha_fwd_hd128_mxfp8_v3x_vs128_ldsink1` 提供给 A/B + `port_gate`，**尚未设为默认**。