# LDS 读与 lgkm 等待的调度策略（`ldsink` / `ldahead32` / `wsink`）与已淘汰的实验

实现：`crates/amdgpu-codegen/src/sched/ldahead.rs`（调度后处理，作用于每个基本块的最终顺序，之后照常重算等待；依赖 / 冒险 / 栅栏走 `depgraph`，结果用 `verify::check_perm` 复核，区域边界不跨越，见 [`weaver-sched.md`](weaver-sched.md)）。

| token | 作用 | 状态 |
|---|---|---|
| `ldsink=D` | 每条 `ds_read` 下沉，直到它与首个消费者之间至多 D 条 MFMA（厂商的 `lgkmcnt(4)`：只留一组在飞） | **出厂**（FMHA 三个 kernel，`D = 1`） |
| `ldahead32=D` | 只把 32-bit scale 字的 `ds_read_b32` 上移，直到与消费者之间有 D 条 MFMA | **出厂**（`D = 1`） |
| `wsink` | 把 kernel 显式的 `s_waitcnt lgkmcnt(n)` 下沉到不碰内存 / 栅栏的计算之后，让前面的 MFMA 拿到记分板的计数等待 | **出厂** |
| `hoist=N`（`ldshoist`） | 调度前把 `ds_read` 上提到至多 N 条之前（受寄存器预算约束） | GEMM profile 预设 |

用法：`Function::sched_opts` 的 token，或叠在预设上的实验开关 `AMDC_SCHED_ADD=<内核名子串>:<tokens>`；工具 `tools/lgab.sh`（`tools/ab_fmha.sh` 加每变体的 emit 环境：`kernel%VAR=val;..`，同一轮交替）。token 也可以按 weave 区域声明（`AMDC_SCHED_REGION`，见 `weaver-sched.md`）。

## 出厂策略的测量（`REF_DATA=normal`，同一轮交替，TF/s 中位数；证书不变，精度 `bad=0`）

`_vs128` 稳态循环的 `lgkmcnt` 不是退化成 `(0)` 的：PV 段早已是部分等待，厂商是每条 PV MFMA 之后读"下一组"并等 `lgkmcnt(4)`（一组在飞），我们是 `lgkmcnt(8)`（两组在飞）；lgkmcnt 只有 4 位（≤ 15），在飞读超过 15 条时只能退化成 `(0)`。

| 变体 | B4H8S2048 | B1H8S8192 | 备注 |
|---|--:|--:|---|
| 基线 `_vs128` | 1696.3 / 1700.6 / 1704.1 | 2369.7 / 2375.8 / 2373.6 | VGPR 256，`lgkmcnt(0)` 49 个 |
| `ldsink=1` | 1706.8 / 1707.4（+0.2…+0.4 %） | 2388.4 / 2391.5（+0.5…+0.8 %） | VGPR 256 → 248 |
| `ldsink=2 / 3` | 持平 | 持平 | |
| `wsink` | +0.2 % | +0.3 %（per-CU +0.6 %） | |
| **`wsink,ldahead32=1`** | **+1.8 %** | **+1.1 %**（per-CU +1.0 %） | M 相位中段两处 `lgkmcnt(0)` 变计数等待，VGPR 248 → 240 |

两处 `lgkmcnt(0)` 的来源：(a) 循环末尾 QK 最后 4 条 MFMA 之前是内核（tile pipeline）显式的收尾 `wait_lds(0)`，它是 depgraph 的栅栏，K 的读全堆在它前面（`wsink` 把它沉到最后一条 MFMA 之后）；(b) 取 K-scale 字的 `ds_read_b32` 刚发出就被消费（`ldahead32=1` 提前一条 MFMA）。
出厂对比 `ldsink=1`（kernel-opt，7 次）：`_vs128` −0.1 / +1.2 / +1.3 %，`_kvsplit` +1.8 / +1.1 / +1.2 %，`_t128` +0.8 / −0.2 / +1.5 %。

## 预设必须叠加而不是替换

`scheduler_options`（`crates/amdgpu-codegen/src/lib.rs`）先取 compile profile 的预设（GEMM profile：记分板等待 + `ldshoist`；`T128` profile：ALU 跨栅栏 + 压力上限 248），**再**把 `Function::sched_opts` 的 token 叠在其上。曾经 `_t128` 带 `sched_opts = "ldsink=1"` 后绕过了预设，丢了 +2.2…+2.9 %（layout 中位数）；回归测试 `crates/amdgpu-kernels/tests/sched_preset.rs`。

## 实验过但已淘汰（代码已删除；实验分支已归档为 `archive/*` tag）

| 策略 | 结果（同一轮交替；`_vs128` 除非注明） | 为什么没用 |
|---|---|---|
| `ldahead=D`（所有 `ds_read` 上移 D 条 MFMA，`ldslack` 放宽寄存器） | D=1/2 持平（−0.2…−0.4 %），D=4 −1.7 %，D=8 −5.1 %（`lgkmcnt(0)` 49 → 202） | 寄存器已在 256，且 lgkmcnt 4 位上限；只对 b32 有用，保留为 `ldahead32` |
| `ldshoist` / `hoist=128` 用在 `_vs128` | −1.3…−1.7 %（VGPR 244） | 完整 list scheduler 之前的上提不如它自己的顺序 |
| `limit=248` 用在 `_vs128` | −0.8…−1.3 % | 同上（`_t128` 例外，见预设） |
| `hoistchain`（上提时带上地址的 ALU 生产者） | 4-wave 原型上与 `hoist=24` 持平（405.2 vs 404.7 TF/s） | 无收益 |
| `mfmapair`（成对发射 MFMA，含对内间隙） | 预算 0.25…∞ 全部 ≤ 基线（−0.3…−1.9 %），带间隙 −0.4…−2.9 % | 无收益 |
| `slotfill` 用在 `_vs128` | −1.5…−3.9 % | 只在 GEMM v4 出厂（见 [`slotfill.md`](slotfill.md)） |
| `ovl`（无 barrier 区段里把与 MFMA 无依赖的 VALU/SALU 均匀摊到各 MFMA 之后）+ `ovlsink` | 4-wave 单 wave 原型：`rewait,hoist=24,ovl` 404.7 / 1321.1 / 1926.7（vs `rewait` 376.8 / 1268.1 / 1822.0）；`ovlsink` 持平 | 只在 1-wave 原型有效；该原型的瓶颈是 LDS 读带宽与指令发射，不是调度（见下） |
| `pexact`（压力模型 = 分配器真实峰值）、`search`（模拟器引导的局部搜索）、`hjoin`（跨稀有分支提 MFMA 前缀） | `pexact` list 方案 381.9 / 1278.5 / 1848.8（低于 `ovl`）；`hjoin=8…200` +0.2 → −9.5 %；`search` 模拟周期 −20 % 而硬件 ≤ +1…+3 % | 模拟器在 ≲3800 周期后不再预测硬件：单 wave 循环受发射数与共享 LDS 读带宽约束（约 4200 周期/tile 的常数项，硬件 ≈ 4200 + 0.2 × 模拟） |

4-wave 原型（`fmha_fwd_hd128_mxfp8_v4w_proto`）的结论：循环的下限由发射数（每 tile 约 425 VALU + 106 `ds_read` + 68 `v_accvgpr_*` + 49 `s_nop`）和 LDS 读带宽决定，重排只能带来 +1…+3 %；要追平厂商需要减少指令数 / LDS 字节，而不是更好的调度。

## 其它

- `hzfill`（post-RA 的 `s_nop` 窗口填充）保留为 weave 区域选项，见 [`weaver-sched.md`](weaver-sched.md)。
- 模拟器 `amdc simmodel` 与它的校准（`sched/sim.rs`）见 [`scheduling.md`](scheduling.md)。