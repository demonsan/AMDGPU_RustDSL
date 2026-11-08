# 访存受限算子（batch 2）：实现状态与实测

设计与 roofline 方法见 [`op-coverage-memtemplates.md`](op-coverage-memtemplates.md)（流式拷贝基准 [`tests/gpu/membw_survey.cpp`](../tests/gpu/membw_survey.cpp)：同读写比的实测上限约 5.7–6.2 TB/s，验收线 = 其 90 %）。每个算子：CPU 参考对拍（含尾部 / 奇数尺寸）、`gpu-verify --strict` 0 错误、实机带宽、已注册（`kernel_by_name`）、IR 文本往返。**aiter 同轮对比：节点上没有 torch / aiter 环境（`import torch` 失败），所以只做 roofline 对比，不做 aiter 同轮对比。**

验证方法：`tests/gpu/<op>_host.cpp`（`hipModuleLoad` 加载 `amdc emit --backend=own` 汇编出的 co，CPU 双精度参考，bf16 判据 = 误差 ≤ 1 bf16 ulp + 1e-6 的绝对下限，并检查输出末尾之外的字节没有被写），带宽 = 读 + 写字节 / 单次 kernel 时间（50 次连续发射的平均，工作集约 1 GB 以避开 Infinity Cache）。

## 当前状态总表（以此为准；下面各节的第一版表格用的是 regular 缓存策略，数值已被「非临时访问」一节取代）

流式 kernel 默认带 `nt`（`_rt` 后缀选回 regular）；流式读写的实测上限 ≈ 6.5 TB/s，下表的百分比按它算。`tools/b2_matrix.sh` 重跑全部正确性与带宽（每个 kernel 一行）。

| 族 | kernel | 当前带宽 / 延迟 | 说明 |
|---|---|---|---|
| 逐元素 | silu_mul / gelu_tanh_mul / add_bf16 / cvt | 6.3–6.7 TB/s（97–103 %） | 与 aiter 同轮：silu_mul 1.07–1.18×，gelu_tanh_mul 1.23–1.27× |
| 行内 | rmsnorm N = 4096 / 1000 / 200 | 6.30 / 5.91 / 5.22–5.28 TB/s | N < 512 用 wave 内分组（N = 64 / 128 / 256：6.3–6.4）；aiter N = 1000 的 1.17× |
| 行内 | add_rmsnorm / layernorm / softmax（N = 4096） | 6.47 / 5.60 / 6.01 TB/s | layernorm 约 1000 条 VALU / 行（rmsnorm 约 590），softmax N = 8192 5.5；短行更低（N = 200：3.9 / 4.3） |
| 量化 | per-token / group128 fp8、int8、静态 fp8、silu_mul_quant | 5.7–6.4 TB/s | 与 aiter 同轮：fp8 per-token 1.08×（N = 1000 为 1.45×）、group128 1.10×、silu_mul_quant 1.11×、int8 1.04–1.26×；静态 fp8 0.97× |
| 反量化 | dequant fp8 → bf16（static / token / group128） | 6.0–6.4 TB/s（N = 1000：5.3） | scalef32 转换只吃 2 的幂 scale，一般 scale 用显式乘法 |
| RoPE / cache | RoPE neox / gptj（含部分旋转） | 6.3–6.8 TB/s；H = 1, D = 64 / 128：7.1 / 9.2（按含 cos / sin 表的流量） | 小 H 一个 wave 装多个 token（见下） |
| RoPE / cache | KV-cache 写（flash / vLLM 分块） | 5.8 / 0.9–1.5 TB/s | 分块布局的 value 是 2 字节散写，regular 策略，不是流式 |
| MoE | topk_softmax（含 renorm） | E = 128, K = 8：T = 65536 为 21 µs（DPP 归约后 +9–19 %） | 小 T 比 aiter 快 1.6–1.8×，大 T 0.87–0.92× |
| MoE | moe_sorting（单 block / 多 block 原子 / 多 block 稳定顺序） | 10–18 µs（≤ 几千对）/ 31 µs（26 万对）/ 105 µs（稳定，26 万对） | 格式与 aiter `moe_sorting` 一致，附 `tile_start` |
| 采样 | argmax / 温度采样 / top-k + top-p | 4.3 TB/s / 0.7–1.3 TB/s 等效 / 36–124 µs 每行（V = 32000） | top-k / top-p 用 bf16 序键的两遍基数选择，不排序 |

未做 / 已知差距：layernorm / softmax 的短行与 N = 400–511 的行布局；采样的性能版本（直方图 top-k / top-p，一行多 block）；> 4 GiB 的 cache / 张量（要 64 位寻址，B1）；vLLM 分块 cache 的整行转置写；aiter 同轮对比没有覆盖 layernorm、row softmax、RoPE、cache、moe_sorting、采样（无法脱离 torch 独立编译）。
## 流式逐元素（模板 T1）

| kernel | 语义 | 配置 | VGPR | 带宽（GB/s，≈1 GB 流量） | 占 roofline（6.0 TB/s） |
|---|---|---|--:|--:|--:|
| `silu_mul_bf16_d<D>` | `out[m][d] = silu(x[m][d]) · x[m][D+d]`，bf16 `[M][2D]` → `[M][D]` | 每 lane 每输入一个 b128，`threads = min(4, ⌈D/512⌉)·64`，grid = (列 tile, M) | 28 | D = 4096：5917；D = 1024（M = 65535）：5909；D = 11008：5820；D = 14336：5737 | 96–99 % |
| `gelu_tanh_mul_bf16_d<D>` | `gelu_tanh(x[m][d]) · x[m][D+d]`（`x·sigmoid(2u)` 形式，无抵消） | 同上 | 28 | D = 4096：5965；D = 14336：5785 | 96–99 % |
| `add_bf16`（flat） | `out = a + b`，n 个元素（n % 8 = 0），RNE 到 bf16 | 256 线程，每 lane 每输入一个 b128 | 24 | n = 1.8 亿：5736（读写比 2 : 1） | 96 % |

正确性（`tests/gpu/eltwise_host.cpp`）：D ∈ {24, 1024, 4096, 11008, 14336}，M ∈ {17 … 65535}（含 D = 24 的整 wave 内半数 lane 越界、D = 11008 的末尾列 tile 不满），`bad = 0`，输出末尾之外 0 字节被写；`max_rel` 3.9e-3（≈ 1 bf16 ulp）。限制：D 是 8 的倍数（8 … 65536）、M ≤ 65535（行在 grid y）、张量 < 4 GiB（buffer 描述符 32 位）。

实现要点：越界行尾 lane 的 store 偏移取 `-16`（超出任何 `num_records`，硬件丢弃），loads 不需要掩码（读到下一行的数据 / 零，无害）；gelu 用 `x / (1 + 2^(−2u·log2 e))`（与 silu 同构），避开 `x − x/(1+e)` 在大负 x 处的抵消。

## 行内归约族（模板 T2 / T4；一行一 wave）

`crates/amdgpu-kernels/src/rows/mod.rs`；一行一 wave，lane `l` 持有 `8 (l + 64 i) + k`，行驻寄存器（HBM 只读一次），统计量 `wave_reduce`；短行（N < 1024）每 wave 同时处理 2 / 4 行（`rows_per_wave`）凑在途字节。N 是 8 的倍数（8 … 8192），不是 512 倍数时最后一个 chunk 在统计量里被屏蔽、存储偏移取 `-16` 被硬件丢弃。主机测试 `tests/gpu/rows_host.cpp`（bf16，误差 ≤ 1 ulp + 1e-6；`add_rmsnorm` 的 `res_out` 要逐位相同；输出末尾之外不能被写）。

| kernel | 语义 | VGPR | 带宽（GB/s，≈1 GB 流量） | 占 6.0 TB/s |
|---|---|--:|--:|--:|
| `rmsnorm_bf16_n<N>` | `x · rsqrt(mean x² + eps) · γ` | 124 / 64 / 60 | N = 4096：5830；1000：5655；200：3681 | 97 % / 94 % / **61 %** |
| `add_rmsnorm_bf16_n<N>` | `res_out = bf16(x + res)`，`out = rmsnorm(res_out)·γ` | 156 | N = 4096：5475 | 91 % |
| `layernorm_bf16_n<N>` | `(x − mean)·rsqrt(var + eps)·γ + β`（两遍） | 160 / 76 | N = 4096：5242；776：4736 | **87 %** / **79 %** |
| `softmax_bf16_n<N>` | 行 softmax（max → Σexp → 归一，行驻寄存器） | 92 / 60 / 160 | N = 4096：5384；1000：5068；8192：5229 | 90 % / **84 %** / **87 %** |

正确性：N ∈ {8, 200, 776, 1000, 4096, 8192}，M ∈ {7 … 41}（含 N = 8 一行只有 1 个 lane 有数据，N = 1000 / 776 / 200 的部分 chunk），`bad = 0`，`res_out_bad = 0`，输出末尾之外 0 字节被写。

**未达 90 % 目标的项和原因**（诚实记录）：N = 200 的 rmsnorm（行 400 B，一个 wave 只有 25 个 lane 有数据，即使 4 行 / wave 在途也不够）→ 需要"几行共用一个 wave 的不同 lane 组"的亚 wave 行布局（待做）；layernorm / softmax：VGPR 160（3 wave / SIMD）加上两遍 VALU（x、d 两份 f32），在途量与占用都低 → 用 packed f32（`PkMulF32` / `PkAddF32`）减半 VALU、统计量用 bf16 打包重算降寄存器（待做）；N = 8192 softmax 同理。

## 量化族（模板 T2 / T3）

`crates/amdgpu-kernels/src/rows/quant.rs`；同样一行一 wave，fp8 e4m3 每 lane 每 chunk 存 8 B。主机测试 `tests/gpu/quant_host.cpp`（反量化值与 CPU 参考的误差 ≤ 半个 e4m3 ulp，scale 相对误差 < 1e-4，输出末尾之外不能被写）。

| kernel | 语义 | VGPR | 带宽（GB/s） | 占 roofline |
|---|---|--:|--:|--:|
| `quant_token_bf16_fp8_n<N>` | per-token 动态：`scale[m] = max|x|/448`，`q = fp8(x/scale)` | 92 / 44 | N = 4096：5533；1000：5615（读 : 写 = 2 : 1） | 92–94 % |
| `quant_group128_bf16_fp8_n<N>` | per-group（128 元素 = 16 lane）动态；scale f32 `[M][N/128]`；组内最大值 = 2 步 DPP quad + 4 槽 LDS | 84 / 40 | N = 4096：5611 | 94 % |
| `silu_mul_quant_fp8_d<D>` | `silu(x[:, :D])·x[:, D:]` 后 per-token fp8（融合，读 : 写 = 4 : 1） | 100 / 52 | D = 4096：5109；1000：5422 | 85 % / 90 %（读密集的 roofline 约 6.3 TB/s） |

正确性：N ∈ {640, 1000, 4096}，M ∈ {15 … 37}，含离群值（±90），`bad = 0`，`max_rel_scale ≤ 3.5e-7`。限制：N（D）为 8 的倍数（group128 为 128 的倍数），≤ 8192；M 任意；张量 < 4 GiB；短行（N < 512）还没有多行 / wave。

## 类型转换与静态量化（模板 T1）

`eltwise/mod.rs`；主机测试 `tests/gpu/cast_host.cpp`（`cvt_*` 逐位对拍；`quant_static` 半个 e4m3 ulp，含 ±5000 离群值被钳到 ±448）。

| kernel | 语义 | VGPR | 带宽（GB/s） | 占 roofline |
|---|---|--:|--:|--:|
| `cvt_f32_bf16` | f32 → bf16（RNE），n 为 4 的倍数；每 lane 两个 4 元素向量（b128 读、b64 写，全合并） | 16 | 5895 | 98 % |
| `cvt_bf16_f32` | bf16 → f32 | 16 | 5974 | 99 % |
| `quant_bf16_fp8_static` | `fp8(clamp(x / scale, ±448))`，`scale` 为 kernarg f32，n 为 8 的倍数（`med3` 钳位） | 20 | 5923（读 : 写 = 2 : 1） | 98 % |

`add_bf16` 的 `ka_is` 事实改为 `n = N`（grid = ⌈N / 2048⌉），不再声称 n 是 2048 的倍数。

## RoPE（模板 T7）

`rope/mod.rs`：`x [T][H][D]` bf16，`positions [T]` i32，`cos_sin [P][D]` bf16（前半 cos、后半 sin），非原地 `out`；一个 grid x 一个 token，一个 block = `min(4, ⌈H·L/64⌉)` 个 wave，每个 wave `64/L` 个 head 同时处理，全部加载先于计算发射；位置 ≥ P 读到 0（cos = sin = 0）。主机测试 `tests/gpu/rope_host.cpp`（CPU 双精度参考，用同一份 bf16 表；含一个位置越界的 token）。

| kernel | 语义 | L（lane / head 行） | VGPR | 带宽（GB/s） | 占 roofline |
|---|---|--:|--:|--:|--:|
| `rope_neox_bf16_h<H>_d<D>` | rotate-half，每 lane 两个 b128 | D / 16 | 44 | H = 32, D = 128：6118；H = 8, D = 128：6495（T = 262144，部分落在 Infinity Cache）；**H = 1, D = 64：1300** | 100 % / 100 %+ / **22 %** |
| `rope_gptj_bf16_h<H>_d<D>` | 交错对 (2i, 2i+1)，每 lane 一个 b128 | D / 8 | 32 | H = 32, D = 128：5997 | 100 % |

D ∈ {64, 128, 256}（全旋转维），H ∈ 1…256；正确性：T ∈ {9 … 37}，H ∈ {1, 8, 32, 40}（H = 40 的最后一个 pass 部分 head 越界）。**限制**：H · L < 64 的内核（如 MQA 的 H = 1）一个 wave 里只有一部分 lane 有 head，带宽低（H = 1, D = 64 只有 22 %）→ 需要"一个 wave 处理多个 token"的变体（待做）；部分旋转见下。

## KV-cache 写（模板 T7）

`cache/mod.rs`，flash 布局：`k, v [T][Hk][D]` → `key_cache, value_cache [S][Hk][D]`，按 `slot_mapping [T]`（i32，负数 = padding token，什么都不写）；一个 token 用 64 / 128 / 192 / 256 个 lane 拷贝一行（`Hk·D·2` 字节，每 lane 每 16 B 一个 b128，K、V 同一轮加载），一个 block 含 4 / 2 / 1 / 1 个 token。主机测试 `tests/gpu/cache_host.cpp`（逐字节对拍，未被写的槽保持哨兵值）。

| kernel | VGPR | 带宽（GB/s，随机槽位的置换） | 占 roofline |
|---|--:|--:|--:|
| `reshape_cache_flash_bf16_h<Hk>_d<D>` | 20 | Hk = 8, D = 128：4909；Hk = 8, D = 64：4716；Hk = 16, D = 128：4860 | 79–82 % |

限制：cache < 4 GiB（32 位 buffer 偏移；需要后端的 64 位地址乘加 B1 才能去掉）。低于 90 % 的原因：随机槽位的 2 KB 行写（相邻 token 的目的地不连续），在途量已按行 / 块给够，剩下的差距在 DRAM 页局部性；量化写（fp8 cache + scale）、NHD 以外的分块布局（vLLM 的 `[num_blocks][Hk][D/x][block_size][x]`）待做。

## 非临时访问（`nt`）

aiter 的 kernel 在每次访存上带 `AUX` / `GROUP_NT` 缓存策略，而我们的第一版没有。同轮对比里 aiter 在 fp8 输出族上领先 4–6 %，把"每 lane 写更宽（16 B）"当假设试过（更慢，已撤销），之后 backend 加了 `CachePolicy`（`nt` / `sc0` / `sc1`，gfx950 汇编里的 `nt` 位），在流式加载 / 存储上加 `nt`（只对 x / residual / 输出；γ、β、cos / sin 不加）。同一轮的前后对比（≈ 1 GB 流量，GB/s）：

| kernel | regular | `nt` | 变化 |
|---|--:|--:|--:|
| silu_mul D = 4096 / 1024 / 14336 | 5977 / 5899 / 5738 | **6558 / 6723 / 6508** | +10 % / +14 % / +13 % |
| gelu_tanh_mul D = 4096 | 5939 | 6424 | +8 % |
| add_bf16 | 5736 | 6502 | +13 % |
| cvt_f32_bf16 / cvt_bf16_f32 | 5895 / 5974 | 6581 / 6293 | +12 % / +5 % |
| quant_bf16_fp8_static | 5979 | 6122 | +2.4 % |
| rmsnorm N = 4096 / 1000 | 5856 / 5726 | 5915 / 5716 | +1 % / 0 |
| add_rmsnorm N = 4096 | 5510 | 5900 | +7 % |
| layernorm N = 4096 / 776 | 5242 / 4736 | 5348 / 4760 | +2 % / +0.5 % |
| softmax N = 4096 / 1000 / 8192 | 5384 / 5068 / 5229 | 5893 / 5658 / 5483 | +9 % / +12 % / +5 % |
| quant_token N = 4096 / 1000 | 5543 / 5615 | **6357** / 5879 | +15 % / +5 % |
| quant_group128 N = 4096 | 5595 | **6387** | +14 % |
| silu_mul_quant D = 4096 / 1000 | 5113 / 5422 | **5927** / 5735 | +16 % / +6 % |
| rope_neox H = 32 / 8，rope_gptj H = 32（D = 128） | 6118 / 6495 / 5997 | 6418 / 6773 / 6347 | +5 % / +4 % / +6 % |
| reshape_cache_flash H = 8, D = 128 | 4909 | 4933 | 0（随机槽位写，瓶颈不在这里） |
| topk_softmax | — | — | 0（延迟 / VALU 主导） |

结论：**流式 kernel 默认带 `nt`**（`_rt` 后缀选回 regular，用于 A / B；单个 kernel 的汇编里能数到 `nt` 位）；实测流式读写的上限是 ≈ 6.6 TB/s（读 : 写 = 2 : 1 的 silu_mul、cvt），不是 5.7–6.2 TB/s 的旧结论——后者来自 HIP 内建 `nt` 在展开拷贝上没有差别的基准，形态不同。所有 kernel 的正确性检查（含尾部、奇数尺寸）在 `nt` 版上全部重跑通过（`tools/b2_matrix.sh`）。已有的 `rmsnorm_quant_fp8_n<N>`（出厂 kernel，md5 不动）没有改成 `nt`，候选的 opt-in 改进。

## 与 aiter 的同轮对比

方法：aiter 的 HIP 源（`csrc/kernels/{activation,quant,rmsnorm_quant,topk_softmax}_kernels.cu`）在节点上无 torch、独立编译成 host + device 目标文件，由 `tests/gpu/vendor_membound_host.cpp` 经 aiter 自己的 C++ 入口（`aiter::silu_and_mul`、`aiter::rmsnorm`、`aiter::dynamic_per_token_scaled_quant`、`aiter::topk_softmax` …，张量用 `AiterTensor` 包住普通 HIP 缓冲）调用；构建见 `tools/vendor_membound.sh`（aiter 的 `csrc/{kernels,include,py_itfs_cu}` 要先解到 `/tmp/aiter_mb`）。我们的内核与 aiter 在同一个作业里交替运行，同样的数据规模（≈ 1 GB 流量，GB/s = 同一公式的读 + 写字节 / 单次时间）、同一块 GPU（GPU 4）。aiter 的数值输出没有逐一对拍，只比时间；输入 bf16。

下表：「我们（regular）」是带 `nt` 之前的同轮数值（两轮），「我们（`nt`）」是默认版本的单独一轮（同一台机器、同一块 GPU、同样的命令，`tools/b2_matrix.sh`），aiter 两轮相差 < 1 %。

| 算子，形状 | 我们（regular） | 我们（`nt`） | aiter | `nt` / aiter |
|---|--:|--:|--:|--:|
| silu_mul，M = 43690, D = 4096 | 5977 / 5947 | 6558 | 5819 / 5901 | **1.11 – 1.13** |
| silu_mul，M = 65535, D = 1024 | 5899 / 5903 | 6723 | 6289 / 6306 | **1.07** |
| silu_mul，M = 12483, D = 14336 | 5738 / 5692 | 6508 | 5515 / 5488 | **1.18** |
| gelu_tanh_mul，M = 43690, D = 4096 | 5939 / 5970 | 6424 | 5051 / 5204 | **1.23 – 1.27** |
| rmsnorm，M = 65536, N = 4096 | 5856 / 5837 | 5915 | 5818 / 5874 | 1.01 – 1.02 |
| rmsnorm，M = 262144, N = 1000 | 5726 / 5688 | 5716 | 4870 / 4866 | **1.17** |
| add_rmsnorm，M = 32768, N = 4096 | 5510 / 5498 | 5900 | 6859 / 6849（见下） | 0.86 |
| rmsnorm_quant（我们的已有 `rmsnorm_quant_fp8_n4096`，未改 `nt`），M = 65536, N = 4096 | 4711 / 4728 | — | 5024 / 5049 | 0.94 |
| quant_token（per-token fp8），M = 131072, N = 4096 | 5543 / 5542 | 6357 | 5906 / 5896 | **1.08** |
| quant_token，M = 524288, N = 1000 | 5615 / 5604 | 5879 | 4050 / 4051 | **1.45** |
| quant_group128（per-group fp8），M = 131072, N = 4096 | 5595 / 5630 | 6387 | 5825 / 5822 | **1.10** |
| quant_static（per-tensor fp8），n = 5.4 亿 | 5979 / 5963 | 6122 | 6292 / 6307 | 0.97 |
| silu_mul_quant（融合，per-row 量化），M = 65536, D = 4096 | 5113 / 5133 | 5927 | 5329 / 5320 | **1.11** |
| topk_softmax（E = 128, K = 8），T = 4096 / 65536 / 524288，µs | — | 4.1 / 21.2 / 141 | 7.3 / 20.9 / 127.5 | 1.8× 快 / 持平 / 0.90 |
| topk_softmax_renorm（E = 256, K = 8），同上 | — | 4.9 / 41.0 / 288.7 | 7.7 / 37.8 / 252.5 | 1.6× 快 / 0.92 / 0.87 |

读法：
- **`nt` 之后，我们在 12 个带宽受限的对比中 10 个领先或持平，仅 quant_static 落后 3 %**；aiter 此前的 4–6 % 领先全部来自缓存策略（"每 lane 写更宽"的假设已经被撤销的实验否定）。
- **add_rmsnorm（aiter 6.86 TB/s）高于测得的流式上限，不能当作有效对比**：可能是 aiter 的 `add_rmsnorm` 对 `residual_out` / 输出的语义不同导致实际流量更小（没有逐元素对拍），待查；在查清之前这一行只保留数据，不用于结论。
- **topk_softmax 不是带宽受限**（524288 × 128 个专家只有 168 MB 流量，~1.2 TB/s）：小 T 我们快（延迟低），大 T 慢 8–13 %（每轮 LDS 往返）；下一步用 DPP 行操作（backend 的 A4）去掉 LDS。
- 没有 aiter 对应的内核：layernorm（aiter 的 layernorm 是 torch 扩展）、row softmax、RoPE / KV cache（`pos_encoding_kernels.cu` 与 `cache_kernels.cu` 的入口依赖 torch 的 int64 positions 等约定，`cache_kernels.cu` 在 gfx950 上不能独立编译：`dtype_float16.cuh` 里的旧 asm 约束），这些仍只和 roofline 比。

## fp8 → bf16 反量化与 int8 量化（backend 的 A2 转换）

`dequant/mod.rs`、`rows/quant.rs`；主机测试 `tests/gpu/dequant_host.cpp`（CPU 参考：随机 fp8 字节，不含 NaN 码，乘 f32 scale，bf16 误差 ≤ 1 ulp）与 `tests/gpu/quant_host.cpp` 的 `token_int8` 模式（反量化值与参考的误差 ≤ 半个 int8 台阶，scale 相对误差 < 1e-4）。

| kernel | 语义 | VGPR | 带宽（GB/s） |
|---|---|--:|--:|
| `dequant_fp8_bf16_static` | `bf16(q · scale)`，一个 f32 scale；每 lane 8 个元素（b64 读、b128 写） | — | 6396 |
| `dequant_fp8_bf16_token_n<N>` | 每行一个 scale `[M]` | — | N = 4096：6064；N = 1000：5321 |
| `dequant_fp8_bf16_group128_n<N>` | 每 128 元素一个 scale `[M][N/128]` | — | N = 4096：6005 |
| `quant_token_bf16_int8_n<N>` | per-token int8：`scale = max|x| / 127`，`q = clamp(rne(x / scale), −127, 127)` | — | N = 4096：6305；N = 1000：5366 |

**一个会踩的坑（已告知 backend）**：`v_cvt_scalef32_pk_{bf16,f16,f32}_fp8` 的 scale 操作数只用到指数（E8M0 风格的 2 的幂）：给 1.63 的效果是 1.0（我们的参考对拍直接发现：输出等于未缩放的 fp8 值）。所以任意 f32 scale 的反量化 = 普通的 fp8 → f32 转换（`PkF32Fp8`）后再乘 scale、再打包成 bf16；`ScalePk…` 转换适用于 mx 格式（E8M0 的 2 的幂 scale）。dequant 内核因此多一次乘法，仍是带宽受限。

## 行内族：行保持打包（寄存器压力）

行内 kernel（rmsnorm / add_rmsnorm / layernorm / softmax）的行在各遍之间保持 bf16 打包（每个 chunk 4 个 dword），在用到的地方再展开成 f32（每对元素两条移位 / 掩码），而不是把 f32 常驻寄存器。VGPR：rmsnorm N = 4096：124 → 92，add_rmsnorm：156 → 120，layernorm N = 4096：160 → 128，softmax N = 8192：160 → 156。带宽变化（`nt` 之后的前后，GB/s）：rmsnorm N = 4096：5915 → 6010，N = 1000：5716 → 5900；add_rmsnorm：5900 → 5804；layernorm N = 4096 / 776：5348 / 4760 → 5333 / 4552；softmax N = 4096 / 1000 / 8192：5893 / 5658 / 5483 → 5952 / 5647 / 5498。结论：占用不是这几个 kernel 的主要限制（变化在 ±4 % 内，有升有降）；剩下的差距来自每行 2 次分组归约的串行延迟链和 VALU，下一步是 DPP 行操作（backend 的 A4）与 packed f32 数学，而不是寄存器。

## KV-cache 写：vLLM / aiter 分块布局

`cache/mod.rs`：`reshape_cache_blocked_bf16_h<Hk>_d<D>_b<BS>`：`k, v [T][Hk][D]` 写入 `key_cache [NB][Hk][D/8][BS][8]` 和 `value_cache [NB][Hk][D][BS]`（`block = slot / BS`，`off = slot % BS`；负 slot = padding token）。key 一侧每 lane 搬一个 16 B chunk（8 个元素，b128 存）；value 一侧是转置的（token 下标是最快维），每个元素一次 16 位存。D ∈ {64, 128, 256}，BS ∈ {8 … 128}（2 的幂），cache < 4 GiB。主机测试 `tests/gpu/cache_blocked_host.cpp`（逐字节对拍，未写字节保持哨兵值；`CONTIG=1` 为 prefill 型的连续槽位）。

| 形状 | 槽位 | 带宽（GB/s，有效读写字节） |
|---|---|--:|
| Hk = 8, D = 128, BS = 16，T = 131072 | 连续（prefill） | 1456 |
| 同上 | 随机置换 | 927 |

这个布局的 value 写本身就是 2 字节散写，不可能接近流式带宽；已有的 aiter `reshape_and_cache` 同布局，同样的问题（不能在节点上独立编译做同轮对比）。**关键一点：这种部分行的 2 字节存不能带 `nt`**——全部带 `nt` 时是 186 GB/s（连续槽位），改成 regular 策略后 1456 GB/s（8 倍；`nt` 绕开了 L2 的写合并）；key 一侧的完整 16 B 存仍然用 `nt`（随机槽位下 `nt` 927 → regular 640）。更好的写法（待做）：同一个 block 的 BS 个连续 token 在 LDS 里转置后整行写。

## MoE 对齐 / 排序：`moe_sorting`（模板 T5 + T7，单 block）

`moe/mod.rs`：`moe_sorting_e<E>_k<K>_b<BM>`（E = 8 … 512 个专家，K = 1 … 16，BM ∈ {16, 32, 64, 128}；M 运行时给，M · K ≤ 2²²）。格式与 kernel-opt 商定，沿用 aiter `moe_sorting` 约定（FlyDSL `moe_sorting_kernel.py` 的头注释）：

| 输出 | 内容 |
|---|---|
| `sorted_token_ids` u32 `[max_padded]` | `(topk_slot << 24) \| token_id`，按专家升序分组，每组补齐到 BM 的倍数，补齐值为哨兵 `(K << 24) \| M`；`max_padded = M·K + E·BM − K` |
| `sorted_weights` f32 `[max_padded]` | 对应的 top-k 权重，补齐处为 0 |
| `sorted_expert_ids` i32 `[max_padded / BM]` | 每个 M tile 的专家号，已用部分之后为 E |
| `num_valid_ids` i32 `[2]` | `[padded_total, M]` |
| `tile_start` i32 `[E + 1]` | 每个专家的第一个 tile（`tile_start[E]` = padded_total / BM），持久化 grouped GEMM 不必再扫 `sorted_expert_ids` |

范围 0 … E 之外的专家号（EP 里的 −1、E）被忽略。专家组内顺序 = 原子到达顺序（多重集，GEMM 行相互独立，结果不变）；确定性的 token 顺序版要等 ballot / mbcnt（已在 B2 里）。

实现（一个 1024 线程的 block，LDS 16 KB）：`ds_add_u32` 做直方图 → 每个线程一个专家的 padded 数 → LDS 里的 Hillis–Steele 前缀扫描（9 步，每步一个 barrier）→ 补齐位置 / `tile_start` / `num_valid_ids` / `sorted_expert_ids`（二分查找扫描结果，每个 tile 一次，`E` 填充）→ 散写：位置 = 该专家游标的 `ds_add_rtn_u32`，`token = i / K` 用 `mul_hi` 魔数（K = 1 用恒等）。**踩到的坑**：DSL 的循环必须是 wave 一致的（分支条件是 VCC 的"任一 lane"），所以每个线程不同的 trip count（补齐项、专家的 tile 数）要改写成统一的上界 + 谓词存储（越界偏移 `0x7fffff00`）。`gpu_verify --strict` 0 错误（LDS 原子地址靠 `min` / `max` 钳位证明）。

主机测试 `tests/gpu/moe_sorting_host.cpp`：与 CPU 参考逐项比较（`num_valid_ids`、`tile_start`、每个 tile 的专家号、每个专家组的 (packed id, 权重) 多重集、补齐值和尾部哨兵、输出末尾之外不被写），含倾斜分布（一半的对去专家 0）与无效专家号（−1、E），`bad = 0`。延迟（单次启动，含启动开销）：

| 形状 | 路由对数 | 延迟 |
|---|--:|--:|
| E = 128, K = 8, BM = 32，M = 37 | 296 | 12.2 µs |
| E = 128, K = 8, BM = 32，M = 1001（倾斜 + 无效号） | 8008 | 14.8 µs |
| E = 64, K = 2, BM = 16，M = 513 | 1026 | 10.5 µs |
| E = 385, K = 6, BM = 32，M = 777 | 4662 | 17.9 µs |
| E = 256, K = 8, BM = 128，M = 4096 | 32768 | 52.2 µs |

限制：只有一个 block，路由对越多越慢（每 1 K 对约 +1.3 µs）；大 M（≳ 16 K 对）需要多 block 版本（全局原子直方图 + 分阶段，aiter 的 multiphase 同样如此）——待做。没有与 aiter 的同轮对比：aiter 的 `moe_sorting` 在本机树里是 CK / FlyDSL 路径，不能脱离 torch 单独编译。

## 部分 RoPE（rotary_dim < D）

`rope_{neox,gptj}_bf16_h<H>_d<D>_r<R>`（R 是 16 的倍数，16 ≤ R < D；表是 `[P][R]`，前半 cos、后半 sin）：每个 head 只有前 R 个元素旋转，其余原样拷贝。实现不另加数据通路：不旋转的 lane 用同一套公式，cos = 1、sin = 0（恒等旋转）——NeoX 里 R 之后的元素成对分给 `D/16 − R/16` 个 lane（每 lane 两个相邻 chunk），GPT-J 里 `lr ≥ R/8` 的 lane 直接是恒等；lane 总数仍是 `D/16`（NeoX）/ `D/8`（GPT-J），所以占用与全旋转版相同。主机测试 `rope_host`（`ROPE_R=<R>`）：D = 128 / R = 64（neox、gptj）、D = 256 / R = 32，`bad = 0`，R 之后的元素逐位等于输入；D = 128 / R = 64 的 neox：5062 GB/s（流量按 R 宽的表计）。R = D 的名字不变，行为不变。

## 采样：`argmax` 与温度采样（模板 T5 / T6，一个 block 一行）

`sampling/mod.rs`：`argmax_bf16_v<V>`（贪心，并列取最小下标）和 `sample_bf16_v<V>`（`p = softmax(logits / T)`，返回累计概率首次达到 `u[b] · 总和` 的最小下标，即带调用方给的均匀随机数 `u ∈ [0, 1)` 的逆 CDF / 多项式采样）。V 为 8 的倍数、8 … 131072；一个 1024 线程的 block 处理一行（grid = B）。ABI：`argmax`：logits @0（bf16 `[B][V]`）、out @8（i32 `[B]`）、B @16；`sample`：logits @0、u @8（f32 `[B]`）、out @16、inv_temperature @24（f32）、B @28。主机测试 `tests/gpu/sampling_host.cpp`（CPU 双精度参考：argmax 逐行对拍，含峰值与每三行一个并列；sample 的下标与参考相同或在累计概率相差 < 2e-4 的边界上相邻）：V ∈ {8, 1000, 32000, 128256}，`bad = 0`。

实现：每个线程拥有 `C = ⌈V / 8192⌉`（向上取 4 的倍数）个连续 16 B chunk；各遍是运行时循环（每次迭代 4 个 chunk 的加载先于计算）：第 1 遍（合并访问）每线程取最大值（严格大于才更新，所以线程内取最小下标）→ block 归约（DPP quad + wave 的 LDS 交换 + 每 wave 一个槽位 + barrier，每次归约一个新槽位，不需要再 barrier）取全局最大，再用"负下标取 Max"取最小下标；`sample` 再加第 2 遍（连续拥有，线程内 `exp` 之和）→ LDS 里的 Hillis–Steele 前缀扫描 → 第 3 遍（连续拥有）找第一个累计值 ≥ 目标的元素（每个线程取负下标的 Max，block 归约取最小下标；舍入导致没有命中时回退到 argmax）。**为什么不把一行留在寄存器里**：128 K 词表需要 160+ VGPR，而 1024 线程的 block 每线程至多 128（超出会 `HSA_STATUS_ERROR_INVALID_ISA` 启动失败）；展开的遍里调度器还会把后几遍的加载全部提到最前。

| 形状 | VGPR | 耗时 | 有效读带宽 |
|---|--:|--:|--:|
| `argmax_bf16_v128256`，B = 4096 | 52 | 246 µs | 4278 GB/s |
| `argmax_bf16_v32000`，B = 16384 | 52 | 275 µs | 3817 GB/s |
| `sample_bf16_v128256`，B = 4096 | 60 | 1460 µs | 720 GB/s |
| `sample_bf16_v32000`，B = 16384 | 60 | 834 µs | 1258 GB/s |

结论：argmax 约为流式上限的 60–65 %（每行只有一个 block 的 16 个 wave，归约与 barrier 的延迟没有被其他行遮盖）；采样是三遍 + 扫描 + `exp`，VALU 与延迟受限，只有 argmax 的 1/3–1/6。没有与 aiter 的同轮对比（`sample_kernels.cu` 的入口依赖 torch 的随机数 / 张量约定）。待做：top-k / top-p 过滤（按阈值二分，不排序）、每个 block 处理多行或把行拆给多个 block 来遮盖归约延迟、DPP 行操作（A4）。

## MoE 排序：多 block 版本与确定性（稳定）顺序

`moe/multi.rs`：输出格式与单 block 的 `moe_sorting_e<E>_k<K>_b<BM>` 完全相同，用于路由对很多的情形，三个 kernel 共用一块 workspace `ws`（u32 `[E][64]`，第 e 行 = 专家 e 在至多 G ≤ 64 个 block 里的计数 / 起点）：

1. `moe_count_e<E>_k<K>`（grid G，1024 线程）：block g 数自己那一段路由对（LDS 原子直方图），写 `ws[e][g]`。
2. `moe_scan_e<E>_k<K>_b<BM>`（grid 1）：每个专家对 G 个 block 求和；padded 前缀扫描；然后 `ws[e][g] = 专家 e 的补齐起点 + 之前各 block 的对数`；同时写 `tile_start`、`num_valid_ids`、`sorted_expert_ids` 和补齐 / 尾部哨兵。
3. `moe_scatter_e<E>_k<K>_b<BM>`（grid G）：block g 放自己的对：位置 = `ws[e][g]` + LDS 游标（`ds_add_rtn_u32`）；block 之间的布局是确定的，block 内同一专家的顺序是原子到达顺序。
3b. `moe_scatter_ord_e<E>_k<K>_b<BM>`：**确定性（稳定）版本**：每个专家的对按 (token, slot) 升序排列，与 token 顺序的计数排序一致。每 1024 个相邻的对为一块，块按顺序处理；块内每个 wave 用 `ballot` / `mbcnt`（B2）匹配相同专家：在 wave 内取 rank（比自己早且同专家的 lane 数）和每个专家的计数，16 个 wave 的计数在 LDS 里按专家做前缀（一个线程一个专家），再推进该专家的游标。

主机测试 `tests/gpu/moe_multi_host.cpp`（`ORDERED=1` 同时逐项检查稳定顺序；其余检查同单 block 版：每个专家组的 (packed id, 权重) 多重集、补齐、尾部哨兵、`tile_start`、`num_valid_ids`、每个 tile 的专家号、输出末尾之外不被写，含倾斜分布与无效专家号），E ∈ {64, 128, 256}，M 4097 … 262144，`bad = 0`。三个 kernel 的 `gpu_verify --strict` 均为 0 错误。耗时（三次启动之和，G = min(64, ⌈对数 / 4096⌉)）：

| 形状 | 路由对数 | 单 block（上一节） | 多 block（原子） | 多 block（稳定，ORDERED） |
|---|--:|--:|--:|--:|
| E = 128, K = 8, BM = 32，M = 4097 | 32776 | ≈ 52 µs（E = 256 同量级） | 20.6 µs | 81.9 µs |
| E = 128, K = 8, BM = 32，M = 20000（倾斜） | 160000 | — | 23.7 µs | 67.8 µs |
| E = 256, K = 8, BM = 128，M = 32768 | 262144 | ≈ 400 µs（外推） | 31.4 µs | 104.9 µs |
| E = 256, K = 8, BM = 128，M = 262144 | 2097152 | — | 98.0 µs | 666.9 µs |
| E = 64, K = 2, BM = 16，M = 100000（倾斜） | 200000 | — | 26.7 µs | 66.0 µs |

三次启动的固定开销约 20 µs（所以小 M 仍应用单 block 版）。稳定版慢 3–7 倍（每个 1024 对的块有 4 次 barrier，wave 内按不同专家逐个匹配）；原子版的 block 内顺序不确定但布局整体确定。限制：G ≤ 64（`ws` 一行 64 个槽）、E ≤ 512。**踩到的坑**：`read_lane` 只接受常量 lane，动态广播要用 `bpermute`；LDS 存储不能像 buffer 存储那样用越界偏移丢弃，非活跃 lane 要写到一个哑单元；`gpu_verify` 对 buffer 存储要求描述符大小是真实的分配大小（`0xffffffff` 的描述符被拒绝），所以 scatter 也要带 `BM`。

## 采样：top-k / top-p 过滤（`sample_topkp_bf16_v<V>`）

在 `sample_bf16_v<V>` 的基础上加 top-k 与 top-p（nucleus）：ABI 在 inv_temperature @24 之后多 `top_p`（f32 @28，1.0 = 关）和 `top_k`（i32 @32，0 = 关），B 移到 @36。语义：先取 top-k（并列在第 k 大值上的都保留），再在保留集合上按 top-p 取最小的前缀（累计概率 ≥ p，并列同样保留），然后在保留集合内重新归一化做逆 CDF 采样（下标升序）。

**不排序，用 bf16 的 16 位序键做两遍基数选择**：logit 是 bf16，原始位翻转后（负数全翻、正数翻符号位）整数序 = 数值序（温度缩放是单调的，不改变序）。阈值 `T` = 满足"权重(key ≥ T) ≥ 目标"的最大键：第 1 遍对高 8 位建 256 个桶的直方图（LDS 原子，`ds_add_u32` 计数 / `ds_add_f32` 求 `exp(x − m)` 之和），下降方向的累计扫描找到累计值首次达到目标的桶 B（wave 内 DPP 包含式扫描：`row_shr` 1 / 2 / 4 / 8 + `row_bcast` 15 / 31，4 个 wave 的总数经 LDS，3 次 barrier），第 2 遍只对高字节 = B 的元素按低 8 位建直方图再扫一次；`T = B · 256 + L`。top-k 用计数（目标 k）；top-p 用质量（目标 `p · S_k`，S_k = top-k 之后的质量，多一遍求和），两者各 2 遍直方图，关掉的那一项（kernarg 控制的一致 `if`）不运行任何东西。最终阈值键 = top-p 的键（它从 top-k 的键起算，所以 ≥ top-k 的键）；第 2、3 遍把键 < T 的元素的分子置 0。**踩到的坑**：不参与计数的元素必须写到每个 lane 各自的"垃圾桶"（`256 + lane`），而不是同一个地址——64 个 lane 打同一个 LDS 原子地址会被串行化，开销最大的就是这个（k-only 62 → 36 µs，p-only 125 → 109 µs）。

主机测试（`sampling_host` 的 top_k / top_p 参数；CPU 参考按上面的集合语义用排序实现）：V = 32000：过滤全关、k = 50、p = 0.9、k = 40 且 p = 0.8、k = 1，V = 1000（k = 20, p = 0.95）和 V = 128256（k = 100, p = 0.9），全部 `bad = 0`；`gpu_verify --strict` 0 错误，72 VGPR。单行延迟（一个 block）：V = 32000：全关 18 µs、仅 k 36 µs、仅 p 109 µs、k 与 p 同时 124 µs；V = 128256 两者同时 465 µs；B = 4096、V = 32000、k = 50 且 p = 0.9：2.02 ms（此前的 26 步二分版本：133 / 133 / 243 µs、4.33 ms）。top-p 仍慢的原因是 `ds_add_f32` 的直方图比整数计数慢得多（约 5 倍）；下一步可以把质量改成定点整数累加，或让每个 wave 先在寄存器里做桶内部分和。

## 行内族：短行的 wave 内分组布局（N < 512）

此前每行占一个 wave 的 64 个 lane，N = 200 只有 25 个 chunk，大半 lane 空闲（rmsnorm N = 200：3.4–3.7 TB/s）。现在 N < 512 的行用 `g` 个 lane（`g` = 不小于 chunk 数的最小 2 的幂，至少 4），一个 wave 并排放 `64 / g` 行，每行的规约在 `g` 个 lane 的组内用 DPP 全规约（`tileop::group_allreduce`：quad_perm、`row_half_mirror` / `row_ror`、`permlane16_swap`），不再经过 LDS；每个 wave 总共处理 `max(8, 64 / g)` 行（先发出全部加载再计算）。N ≥ 512 的布局不变（`g = 64`）。主机测试 `rows_host`（行数 M 较大时 `rows_per_wave` 要与 kernel 一致）：N ∈ {8, 24, 64, 128, 200, 256, 400, 504} 的 rmsnorm、N ∈ {24, 200} 的 layernorm / softmax / add_rmsnorm，`bad = 0`（含奇数的 M = 53）。带宽（M = 10⁶ 行；GB/s）：

| kernel | 之前 | 现在 |
|---|--:|--:|
| rmsnorm N = 200 | 3412–3769 | **5284** |
| rmsnorm N = 64 / 128 / 256 | — | 6284 / 6441 / 6289 |
| rmsnorm N = 400 / 504（g = 64，只是行数更多） | — | 5455 / 5488 |
| add_rmsnorm N = 200 | — | 5628 |
| softmax N = 200 | — | 4326 |
| layernorm N = 200 | — | 3854 |

softmax 与 layernorm 的短行仍然只有 rmsnorm 的 60–80 %（两遍规约 + 更多 VALU，加载的字节更少）；N = 400 – 511 的行仍是一个 wave 的 64 lane 里用 50 个，下一步是为它们加一个"两行占三个 wave"的布局或把 chunk 数不是 2 的幂的行放进 `g = 64` 组里并排两行。

## RoPE：小 H 时一个 wave 装多个 token

一个 token 只用 `H · L` 个 lane（L = 每个 head 的 lane 数：NeoX 为 D / 16、GPT-J 为 D / 8），`H · L < 64` 时大半个 wave 空闲（H = 1, D = 64 只有 4 个 lane：此前 1300 GB/s，22 %）。当 `H · L ≤ 32` 且是 2 的幂时，一个 wave 装 `64 / (H · L)` 个 token（H = 1, D = 64：16 个），一个 block 4 个 wave，grid = ⌈T / (4 · tpw)⌉；位置和 cos / sin 表行因此是向量值（每个 lane 组自己的 token），越界 token 的 store 被屏蔽。名字与 ABI 不变。主机测试 `rope_host`（T = 29 … 41 的奇数 token 数，H ∈ {1, 2, 4, 8}，D ∈ {64, 128}，neox / gptj）`bad = 0`。带宽（GB/s，流量把每个 token 的 cos / sin 行也算上，它们在 L2 里，所以高于真实 HBM 流量）：H = 1, D = 64（neox）7062；H = 1, D = 128：9250；H = 2, D = 64：6193；H = 4, D = 128：7174；H = 1, D = 64（gptj）7419。

## MoE 合并：`moe_reduce` / `moe_combine`（模板 T3）

kernel-opt 的 stage 2 先写"每条路由一行"的输出，再由这个 kernel 合并：`moe_reduce_bf16_k<K>_d<D>`：`out[t][d] = bf16(Σ_k w[t][k] · x[t][k][d])`（x bf16 `[T][K][D]`，w f32 `[T][K]`，out bf16 `[T][D]`）；`moe_combine_bf16_k<K>_d<D>` 是不带权重的和（`moe_buf` / 共享专家的合并；名字区别于 kernel-opt 的 `moe_sum_bf16_k<K>`：那个的输入是 fp32 的每路由中间结果、D 在 kernarg 里）。K ∈ 1 … 16，D 是 8 的倍数（8 … 65536）；grid = (列 tile, T)，每个 lane 一个 b128 / 路由，所有 K 次加载先发出，f32 累加，一次 RNE 到 bf16，流式（`nt`）访问；块的线程数在 64 / 128 / 192 / 256 里选最后一个列 tile 空闲 lane 最少的（D = 7168 → 128）。限制：T ≤ 65535 且 T · K · D · 2 < 2 GiB（32 位偏移；更大要 B1）。主机测试 `tests/gpu/moe_reduce_host.cpp`（CPU 双精度参考；误差判据 = 1 个 bf16 ulp + 1e-6 · Σ|项| 的累加误差）：K ∈ {1, 2, 6, 8, 16}，D ∈ {24, 512, 1000, 2048, 4096, 7168}，T ∈ {17 … 41}（奇数），`bad = 0`，输出末尾之外 0 字节被写，`gpu_verify --strict` 0 错误，VGPR 20 … 100。带宽：见 `tools/b2_matrix.sh` 的 moe_reduce 行（T = 16384 … 65535、K = 6 … 8、D = 2048 … 7168：5.4–6.4 TB/s，D = 7168 的块大小修正前是 5.44）。


## 行内族：γ / β 延后加载；softmax 重算 exp 的实验

单步（N ≥ 512）的行在统计量算完之后才加载 γ / β（此前和 x 一起在最前面加载，在两次归约期间白占 32–64 个 VGPR；它们命中 L2，延迟被其他 wave 遮住）；短行（多步）仍在最前面加载。带宽（GB/s，N = 4096）：add_rmsnorm 5803 → **6468**，layernorm 5349 → 5602，rmsnorm 6304 → 6375；VGPR 只少了几个（layernorm 128 → 120，add_rmsnorm 120 → 116），收益主要来自 γ / β 的加载不再卡在每个 wave 的关键路径之前。softmax 在输出遍里重算 exp、不保存 64 个 f32：N = 4096 反而慢（5836 vs 6007），N = 8192 只快 0.7 %（VGPR 156 → 144），所以只是 opt-in（`AMDC_SOFTMAX_RECOMPUTE_EXP=1`）。周期模型：layernorm N = 4096 约 1005 条 VALU / 行（每条 wave64 指令 2 个周期，4 个 SIMD）≈ 500 个 CU 周期，而一行的 HBM 时间 ≈ 1550 个 CU 周期，所以 VALU 约占三分之一，不是瓶颈；剩下的差距在加载 / 存储与计算的重叠。

## 去掉的实验开关（结果留在这里，代码在归档 tag）

仓库约定 kernel 里不留非默认的环境变量开关，所以每个 kernel 的选择已固定：
- **DPP 全规约 vs LDS 交换**（`AMDC_DPP_REDUCE`，归档 tag `archive/dpp-reduce-env`）：`topk_softmax` 用 DPP（+19 % E = 128 / +9 % E = 256，延迟受限）；行内族（rmsnorm / add_rmsnorm / layernorm / softmax）、量化族、`argmax` / `sample` 的块归约用 LDS 交换（带宽受限，DPP 与 LDS 持平 ±2 %）；N < 512 的短行布局本来就必须用组内 DPP。
- **softmax 重算 exp**（`AMDC_SOFTMAX_RECOMPUTE_EXP`，归档 tag `archive/softmax-recompute-exp`）：N = 4096 5836 vs 6007 GB/s（更慢），N = 8192 只快 0.7 %，VGPR 156 → 144；不采用。

## KV-cache 写：超过 4 GiB 的 cache（B1 的 64 位描述符基址）

`reshape_cache_flash_bf16_h<Hk>_d<D>` 和 `reshape_cache_blocked_bf16_h<Hk>_d<D>_b<BS>` 现在支持任意大小的 cache（slot 到 2³¹ 个，每个 slot 行至多 64 KiB，即单个张量到 ≈ 128 GiB）：缓存描述符被一个 64 位 `ptr_mad` 移到当前 token 的 slot 行 / 块上（`num_records` = 一行 / 一块，所以 32 位偏移不再限制 cache 大小），非法 slot（< 0 或 ≥ S）的存储被屏蔽（backend 补丁，`origin/batch2-ptr64`）。新增主机测试 `tests/gpu/cache_big_host.cpp`：整个 cache 上散布的 slot（含 4 GiB / 8 GiB 之后的，以及首尾 slot、负的和 ≥ S 的非法 slot），每个被写的 slot / 块与 CPU 参考逐字节对比，抽样的未触碰行保持哨兵值；flash 12.2 / 30.5 GiB（h8, d128）和 14.3 GiB（h5, d64）、blocked 12.2 GiB，`bad = 0`。`moe_reduce` / `moe_combine` 同样改成每 token 一个描述符（`T · K · D · 2` 不再有 2 GiB 的限制）：T = 65535, K = 8, D = 7168（x = 7.5 GB）的抽样检查 `bad = 0`，5.73 TB/s；<4 GiB 的带宽与之前持平（6.0 / 5.9 / 5.9 TB/s，在噪声内）。

带宽：< 4 GiB 时没有可测的退步——同一次运行里，新的 64 位版本与旧的 32 位实现（commit 297c4c53 的代码）在 flash h8, d128 上是 4988 vs 4926 GB/s（噪声内），blocked 随机 slot 870 / 连续 1433 GB/s 与之前持平。（此前几轮测得的 5.8 TB/s 是另一个时间点的机器状态：同一份旧代码现在也只有 4.9 TB/s；backend 交替运行新旧时看到的 5 % 差别也在这个噪声之内。）让数据加载紧跟在 slot 加载之后发出（已采用，无害）和把边界检查换成一次无符号比较（通不过 provenance 校验，已撤销）都没有带来可测的变化，所以不需要单独的 < 4 GiB 变体。


## 短行的 layernorm / softmax：VALU 受限（结论）

把短行每个 wave 同时在途的行数从 8 加到 16 没有可测的变化（rmsnorm N = 200：5284 → 5384，layernorm N = 200 3846、softmax N = 200 4363 都不动），所以短行不是延迟 / 在途字节受限，已撤销；它们是 VALU 受限：layernorm N = 200 每 wave 步约 150 条 VALU 处理 2 行（g = 32），softmax 还有每元素一次 `v_exp`（四分之一速率）。N = 400 / 256 的 layernorm（4224 / 4937）和 N = 400 的 softmax（4572）同理；N = 400 – 511 的行用 `g = 64`，一个 wave 里 50 – 63 个 lane 有数据，不是瓶颈。继续优化需要减少每元素的 VALU：用 `v_pk_add_f32` / `v_pk_mul_f32`（DSL 里有 `PkAddF32` / `PkMulF32`，tileop 的 `apply2` 会自动打包）重写 layernorm 的减均值 / 平方 / 缩放，理论上可去掉约三分之一的 VALU；暂未做。


## aiter 内存受限算子覆盖表（have / missing / why）

按 aiter 的分类逐项对照（`csrc/kernels` 与 `aiter/ops`）；"有" = 已有 Rust 实现且过 CPU 参考 + `gpu_verify --strict` + `ir_text` 往返 + `tools/b2_matrix.sh`。按 aiter 派发里的使用频率排序。

| 类别 | aiter 算子 | 状态 | 我们的 kernel / 说明 |
|---|---|---|---|
| 归一化 | rmsnorm | 有 | `rmsnorm_bf16_n<N>`（N = 8 … 16384，短行组内 DPP） |
| | add + rmsnorm（残差） | 有 | `add_rmsnorm_bf16_n<N>` |
| | layernorm | 有 | `layernorm_bf16_n<N>`（N ≥ 512 时 5.5 TB/s；短行 VALU 受限，见上） |
| | rmsnorm + fp8 量化（per token） | 有 | `rmsnorm_quant_*`（已有，shipped kernel）；本批新增行内版本见下行 |
| | **fused add + rmsnorm + per-token fp8 量化**（`add_rmsnorm_quant`） | 有（新） | `add_rmsnorm_fp8_n<N>`：res_out bf16 + out fp8 + scale f32 [M]；N = 4096 4.3 TB/s（计流量），VGPR 128 – 204 |
| | **gated rmsnorm**（`norm(x) * w * silu(z)`，Mamba2 / GDN 输出归一化） | 有（新） | `gated_rmsnorm_bf16_n<N>`：N = 128 6.2 TB/s，N = 4096 6.1 TB/s |
| | **gated rmsnorm + fp8 group 量化**（`gated_rmsnorm_fp8_group_quant`：head_dim = group = 128） | 有（新） | `gated_rmsnorm_fp8_n<N>`：每行一个 scale（N = 128 时一行就是一个 head = 一个 group）；N = 128 5.6，N = 4096 5.2 TB/s（流量更小） |
| | fused qk rmsnorm（`fused_qk_rmsnorm`：q、k 各自整行归一化，一次启动） | 有（两次 rmsnorm 启动） | 只省一次启动，不新增 kernel；N = 1536 / 512 用 `rmsnorm_bf16_n<N>` |
| | groupnorm（`groupnorm.cu`，视觉模型） | 缺 | 非推理热路径（LLM 推理里用不到），按需再做 |
| 量化 | per-token / per-group128 / static fp8，int8 per-token，dequant | 有 | 见前文 |
| | mxfp4 / mxfp8 量化（`quant_mxfp4.cu`，E8M0 缩放，32 元素块） | 有（Main，kernel-opt） | `quant_mxfp4_bf16_k<K>`、`quant_mxfp8_bf16_k<K>`（`rows/mxquant.rs`，不在本分支，合入 Main 后可用） |
| | mxfp6 量化 | 缺 | 与 mxfp4 同构（e2m3 / e3m2 打包），使用面窄 |
| | **融合进 norm 的 mxfp4 量化**（`fused_rms_mxfp4_quant`） | 有（新） | `rmsnorm_quant_mxfp4_bf16_k<K>`：RMSNorm（f32，不经过 bf16 中间值）后直接按 `quant_mxfp4_bf16_k<K>` 的规则（kernel-opt 的 amax 舍入、E8M0、`v_cvt_scalef32_pk_fp4_f32`）量化，输出 fp4 `[M][K/2]` 和 shuffled a4w4 scale 布局；K % 512 = 0，≤ 16384，M % 16 = 0；K = 2048 / 4096 / 7168 都是 2.7 – 2.8 TB/s（只计一次 x 读 + q + scales） |
| 融合激活 | silu_mul / gelu_tanh_mul（`[M][2D]`） | 有 | `silu_mul_bf16_d<D>`、`gelu_tanh_mul_bf16_d<D>`（6.5 TB/s） |
| | silu_mul + fp8 量化 | 有 | `silu_mul_quant_fp8_d<D>` |
| | **silu_and_mul 带 limit**（`x = min(x, limit)`, `y` 夹到 ±limit） | 有（新） | `silu_mul_limit_bf16_d<D>`，limit 是 kernarg（≤ 0 = 不限）；6.55 TB/s |
| | **swiglu_and_mul**（gpt-oss：限幅 7、alpha = 1.702、`(lin + 1)`） | 有（新） | `swiglu_oai_mul_bf16_d<D>`；6.58 TB/s |
| | **一元激活**（gelu_fast = tanh 形式、relu2、silu、sigmoid） | 有（新） | `gelu_fast_bf16` / `gelu_tanh_bf16` / `relu2_bf16` / `silu_bf16` / `sigmoid_bf16`（6.2 – 6.4 TB/s） |
| | gelu（erf 精确形式）、gelu_and_mul_bias、swiglu_and_mul_bias（MoE 逐专家 bias）、situv2 | 缺 | erf 需要多项式近似；带 bias 的两个需要 expert id 的 gather；使用频率低 |
| | scaled_silu_and_mul（输出 fp8，静态 scale） | 有（组合） | `silu_mul_quant_fp8_d<D>` 的静态 scale 变体可由 `quant_bf16_fp8_static` 接在后面；融合版缺 |
| 位置编码 | rope neox / gptj，部分旋转（`_r<R>`），token 打包 | 有 | `rope_{neox,gptj}_bf16_h<H>_d<D>[_r<R>]` |
| | **fused qk-norm + rope**（`fused_qk_norm_rope`，Qwen3 风格逐 head 归一化后旋转） | 有（新） | `qknorm_rope_{neox,gptj}_bf16_h<H>_d<D>`（全旋转）；q、k 各启动一次；H = 8 D = 128 5.7 TB/s |
| | **fused qk-norm + rope + KV cache 写（+ fp8 量化）**（`fused_qk_norm_rope_cache_quant` 的 K / V 部分） | 有（新） | `qknorm_rope_kvcache_{neox,gptj}_{bf16,fp8}_h<H>_d<D>[_i64]`：K 逐 head 归一化后旋转、V 原样（bf16）或按静态 `v_scale` 量化（fp8），都直接写进 flash 布局的 cache（`slot_mapping`，int32 / int64）；Q 用 `qknorm_rope_*` 单独一次启动。H = 8，D = 128：bf16 cache 5.9 TB/s，fp8 cache（i64）4.9 TB/s（计流量）。cache < 2 GiB |
| | mrope（3D 位置）、yarn / 缩放表 | 缺（表由调用方预计算） | 表是 `cos_sin [P][D]` 输入，yarn 只是换表，无需新 kernel；mrope 的按段选位置需要一个小的位置 gather |
| KV cache | reshape_and_cache flash 布局 / vLLM 分块布局（任意大小 cache） | 有 | `reshape_cache_flash_bf16_h<Hk>_d<D>`、`reshape_cache_blocked_bf16_h<Hk>_d<D>_b<BS>` |
| | **fp8 KV cache 写（静态 per-tensor scale）** | 有（新） | `reshape_cache_flash_fp8_h<Hk>_d<D>`：k_scale / v_scale kernarg；5.9 TB/s |
| | **fp8 cache 写 + per-token（per-head）动态 scale，分块布局**（`reshape_and_cache_with_pertoken_quant`，非 asm 布局，x = 16） | 有（新） | `reshape_cache_blocked_fp8_h<Hk>_d<D>_b<BS>[_i64]`：`scale[h][slot] = max(amax, floor) / 448`，key `[NB][Hk][D/16][BS][16]`、value `[NB][Hk][D][BS]`（fp8），`k_scales / v_scales f32 [Hk][S]`；0.93 TB/s（value 是 1 字节散写，与 bf16 分块版同量级） |
| | fp8 分块版的 per-block scale（`with_block_quant`）、asm 布局（`asm_layout=true`） | 缺 | per-block scale 要对整块做 amax（写入后二次扫描），asm 布局只是另一套偏移；使用面窄 |
| | slot_mapping 为 int64（vLLM / SGLang 实际传的类型） | 有（新） | 所有 `reshape_cache_*` 都有 `_i64` 后缀版本：读每个条目的低 32 位（slot < 2³¹，−1 = 0xffffffff 仍是 padding） |
| | MLA cache（`concat_and_cache_mla`） | 缺 | 两段拼接写（kv_c + k_pe）：和 flash 版同构，缺的只是名字 / 布局约定 |
| 采样 / top-k | argmax、sample（温度）、top-k + top-p、topk_softmax | 有 | 见前文 |
| | **moe_fused_gate**（DeepSeek-V3 分组 topk：sigmoid + bias → 组内 top-2 之和 → 丢掉 G − TG 个最低的组 → 余下专家里取 top-K → 权重 = s / Σs） | 有（新） | `moe_fused_gate_bf16_e<E>_g<G>_tg<TG>_k<K>`（G ∈ {4, 8, 16, 32, 64}，E / G ∈ {8, 16, 24, 32}，K ≤ 32）：一个 lane 持有一个 (行, 组) 的 E / G 个专家，所有规约是 G 个 lane 上的 DPP 全规约；并列取低下标（组选择同样）；NaN → 0。E = 256, G = 8, TG = 4, K = 8：M ≤ 4096 约 10 µs，M = 131072 118 µs（VALU 受限，0.9 µs / 千行）。不含 shared-expert 融合与 pad 行 |
| | topk_gating | 缺 | 与 `topk_softmax` 的 renorm 变体等价的一个子集，未单列 |
| | **topk_per_row**（decode：`seq_lens` + `next_n` 给出每行长度；prefill：逐行 `[start, end)`；稀疏注意力的 indexer topk，K ≤ 2048，f32 logits） | 有（新） | `topk_per_row_decode_f32_k<K>` / `topk_per_row_prefill_f32_k<K>`：四遍 8 位基数选择（f32 的保序 32 位键）+ 一遍发射；输出为被选列的绝对下标（无序；第 K 大值上的并列任取），行长 ≤ K 时原序返回整行，之后补 −1；一行一个 1024 线程的块（行不拆到多个 CU）。R = 256 行 × 64K 94 µs，R = 4096 行（1/4 为短行）× 32K 0.54 ms；不含 `values` 输出与 `stable`（升序）模式 |
| | topk_plain（无行范围的版本） | 缺 | 同一 kernel 的 `start = 0, end = stride` 特例；按需再加名字 |
| MoE 辅助 | moe_sorting（单 block / 多 block / 稳定序）、**moe_reduce / moe_combine** | 有 | `moe_sorting_*`、`moe_{count,scan,scatter,scatter_ord}_*`、`moe_reduce_bf16_k<K>_d<D>`、`moe_combine_bf16_k<K>_d<D>` |
| 其他 | **causal_conv1d_update**（decode，W = 2 – 4，状态窗口移位，可选 silu 与连续批处理的 state_indices，pad = 负） | 有（新） | `causal_conv1d_update_bf16_w<W>_s<SL>[_idx][_silu]`：输出与新状态均与 CPU 参考逐位一致；B = 2048 dim = 8192 3.1 TB/s（2 字节访问，没有向量化） |
| | **causal_conv1d fwd**（prefill，变长序列，`split_qkv`） | 有（新） | `causal_conv1d_fwd_bf16_w<W>[_silu]`（输出 `[T][dim]`）与 `causal_conv1d_fwd_split_bf16_w<W>[_silu]`（特征 `[0,k)` → q、`[k,2k)` → k、其余 → v，对应 aiter `causal_conv1d_fwd_split_qkv`）：channel-last x、`cu_seqlens`、`cache_indices`（负 = 跳过）、`has_initial`、初始 / 最终状态；W = 2 – 4；长度一致的序列 2.6 – 2.9 TB/s |
| | gated delta rule（`chunk_gated_delta_rule_fwd_h`、`gdr_decode_packed`、`fused_split_gdr_update`） | 缺 | 带小矩阵递推的状态更新（不是纯带宽），属于 GEMM 类；不在内存受限计划内 |
| | custom_all_reduce / quick_all_reduce / reduce_scatter | 缺 | 多卡通信，本机单卡无法验证 |
| | mhc / mega_mhc / mha_v4_quant / dsv4_*（DeepSeek-V4 专用融合） | 缺 | 模型专用的大融合，使用面窄 |
| | binary / unary 算子（add、mul、tanh、sigmoid…） | 部分 | `add_bf16`、`sigmoid_bf16`、`silu_bf16` 等；其余是同一模板的换函数 |

**精度判据**：bf16 输出 ≤ 1 ulp（+ 小绝对量），fp8 输出按 e4m3 的半步长（相对 1/16）加 scale · 2⁻⁹（次正规），scale 相对 1e-3；cache / state 类的输出按字节逐位比较。


## 每行 top-K（`topk_per_row_*`）

`topk_per_row_{decode,prefill}_f32_k<K>`（K = 1 … 2048）：对 f32 logits 的每一行取最大的 K 个，输出列下标。decode 版的行 `b · next_n + j` 有 `L = seq_lens[b] − next_n + j + 1` 个有效列（grid = (B, next_n)）；prefill 版逐行 `[row_starts, row_ends)`，输出绝对列下标。选择方法与"采样"一节的 top-k 相同，只是键是 f32 的 32 位保序键，所以是四遍 8 位的基数选择（每遍一次 LDS 直方图 + 256 桶的降序扫描）求出第 K 大的键 `T` 与还需要的与 `T` 相等的元素个数 `rem`，最后一遍发射所有 `> T` 的元素（LDS 计数器分配位置）和前 `rem` 个 `== T` 的元素；不参与计数的元素写到每个 lane 各自的垃圾槽。每行读 5 遍（第 1 遍来自 HBM，其余多半来自 L2）。

验证（`tests/gpu/topk_per_row_host.cpp`）：列必须互不相同、在行内、且所选值的多重集合恰好是该行最大的 K 个值（并列任取）；行长 ≤ K 的行必须按列序返回整行再补 −1。覆盖 decode（next_n = 1 / 2 / 3，长度不一，含短于 K 的行）、prefill（行首非零、行长 < K / = K / > K），K = 1、64、100、128、2048，带大量并列值的数据；`bad_rows = 0`，`gpu_verify --strict` 0 错误，48 VGPR。性能（一行一个块，不拆到多个 CU）：256 行 × 64K：94 µs；256 行 × 32K：46 µs；4096 行（1/4 短行）× 32K：0.54 ms。长行拆到多个 CU（aiter 的 multi-block 路径）是后续工作，只对行数少于 CU 数时有意义。


## causal_conv1d 前向（prefill / 变长）

`causal_conv1d_fwd_bf16_w<W>[_silu]` 与 `causal_conv1d_fwd_split_bf16_w<W>[_silu]`：`out[t][f] = bias[f] + Σ_{i < W} weight[f][i] · x[t − (W − 1) + i][f]`，序列起点之前的输入取该序列的初始状态（`has_initial` 为真）或 0，可选 silu；序列结束后 `conv_states[cache_idx][f][0 .. W−1)` 写成该序列最后 W − 1 个输入（序列短于 W − 1 时保留更早的初始状态值）。x 是 channel-last 的 `[T][dim]`（SGLang prefill 的布局），`cu_seqlens [B + 1]` 给出序列边界，`cache_indices [B]` 给出状态槽（负 = 跳过整个序列）；状态布局 `[slots][dim][W − 1]`，与 decode 的 `causal_conv1d_update_*_s<W−1>` 完全一致，所以 prefill 写的状态 decode 可以直接接着用。split 版本把特征 `[0, k)`、`[k, 2k)`、`[2k, dim)` 分别写进 q、k、v 三个张量（行距 k、k、dim − 2k；k % 8 = 0）。

布局：一个 lane 负责 8 个连续特征（b128 访问），一个块 256 lane = 2048 个特征，grid = (⌈dim / 2048⌉, B, ⌈max_len / 8⌉)：每个块顺序处理 8 个 token，前 W − 1 个输入留在寄存器里（块内第一个块从初始状态读，其余块从 x 读前几个 token）；8 次加载在第一个输出前全部发出；超出序列末尾的块不碰内存。限制：`T · dim · 2 < 2 GiB`（32 位偏移；更大要逐序列的 `ptr_mad` 描述符）、dim % 8 = 0。验证（`tests/gpu/conv1d_fwd_host.cpp`）：长度 1、2、W、随机的混合序列，一部分有初始状态、一部分没有、一部分状态槽为负（输出与状态都必须不被写），dim = 776 / 1000 / 1536 / 2048 / 4104 / 8192，W = 2 / 3 / 4，silu 开 / 关，split；输出（≤ 1 bf16 ulp）与最终状态（逐位）都与 CPU 参考一致，`gpu_verify --strict` 0 错误。性能：B = 128、每序列 512 token、dim = 8192：2.8 TB/s（计 x 读 + out 写）；序列长度很不均匀时（最长是平均的 4 倍，网格按最长序列铺）约 1 TB/s，因为 74 % 的块在序列末尾之外空转——要避免就需要 aiter 那样的（序列, 块）调度表，留作后续。VGPR 120 – 176。


## fp8 分块 cache（per-token scale）与 int64 slot_mapping

`reshape_cache_blocked_fp8_h<Hk>_d<D>_b<BS>[_i64]`（D ∈ {64, 128, 256}，BS ∈ {16, 32, 64, 128}）：每个 (token, head) 取 D 个元素的 amax（一个 head 的 D / 8 个 lane 上的 DPP 全规约），`scale = max(amax, floor) / 448`，`q = fp8(x / scale)`；key 写成 `[NB][Hk][D/16][BS][16]`（每个 lane 一次 8 字节存储），value 写成转置的 `[NB][Hk][D][BS]`（每个元素 1 字节存储），scale 写到 `k_scales / v_scales [Hk][S]`（下标 `head · S + slot`，head 的第一个 lane 存）；`S = NB · BS`，`Hk · S · 4 < 2 GiB`。缓存描述符按块移动（64 位 `ptr_mad`），所以 cache 本身可以任意大。主机测试 `tests/gpu/cache_blocked_fp8_host.cpp`：每个写入的 (token, head) 的字节按 e4m3 的半步长对 x / scale 校验、scale 对 amax / 448 校验（相对 1e-3）、未写的 slot 的 scale 与 cache 字节保持哨兵值、末尾之后 0 字节被写；Hk = 1 / 4 / 8，D = 64 / 128 / 256，BS = 16 / 32 / 64，含 padding slot；`bad = 0`；`gpu_verify --strict` 0 错误，44 VGPR。

**int64 slot_mapping**：vLLM 与 SGLang 的 slot_mapping 是 int64 张量，所以 `reshape_cache_flash_{bf16,fp8}_h<Hk>_d<D>`、`reshape_cache_blocked_{bf16,fp8}_h<Hk>_d<D>_b<BS>` 都有 `_i64` 后缀的变体（名字最后加 `_i64`，其余 ABI 不变，slot_mapping 每项 8 字节）：只读每项的低 32 位，要求 slot < 2³¹（−1 = 0xffffffff 仍是 padding）。flash bf16 / fp8 和 blocked bf16 的 `_i64` 版本用 `cache_big_host` / `cache_fp8_host`（新增的 `i64` 参数）在 12 GiB 的 cache 上验证过，`bad = 0`。


## 融合的 qk-norm + RoPE + KV cache 写

`qknorm_rope_kvcache_{neox,gptj}_{bf16,fp8}_h<Hk>_d<D>[_i64]`：把 `qknorm_rope_*` 的输出阶段换成 KV cache 写。输入 K、V 为 `[T][Hk][D]` bf16；K 逐 head 做 RMSNorm（gamma、eps）后旋转，结果写到 `key_cache[slot_mapping[t]]`；V 写到 `value_cache[slot]`，bf16 cache 原样拷贝、fp8 cache 用静态 `v_scale` 量化（K 用 `k_scale`，e4m3，写入的是 `x / scale`）；flash 布局 `[S][Hk][D]`；`slot_mapping` 负数或 ≥ S 的 token 什么都不写。ABI 在 `qknorm_rope_*` 的基础上加 v @56、key_cache @64、value_cache @72、slot_mapping @80、S @88（i32）、k_scale @92 / v_scale @96（仅 fp8）；`out` 参数保留但不使用。Q 的 norm + rope 仍然是 `qknorm_rope_*` 的一次启动，所以完整的 aiter 融合 = 两次启动（Q、KV）。限制：cache < 2 GiB（逐 lane 的 slot 地址是 32 位偏移）、全旋转（无 `_r<R>`）。

主机测试 `tests/gpu/qknorm_rope_kv_host.cpp`：CPU 参考（范数 → 旋转 → 写 / 量化）逐元素比较 K（bf16 ≤ 1 ulp、fp8 按 e4m3 半步长）和 V（bf16 逐位、fp8 同上），未写的 cache 行保持哨兵、末尾之后 0 字节被写；覆盖 neox / gptj，H = 1（打包 token 模式）/ 2 / 4 / 8，D = 64 / 128 / 256，bf16 与 fp8 cache，int32 与 int64 slot，含负 slot、≥ S 的 slot 和超出 cos / sin 表的位置；`bad = 0`；`gpu_verify --strict` 0 错误，52 – 92 VGPR。


## RMSNorm + MXFP4 融合

`rmsnorm_quant_mxfp4_bf16_k<K>`：把 `rmsnorm_bf16_n<K>` 与 kernel-opt 的 `quant_mxfp4_bf16_k<K>`（`rows/mxquant.rs`）合成一个 kernel，省掉规范化后的行在 HBM 里的 bf16 往返（写 2 B + 读 2 B / 元素）。量化规则、E8M0 scale、fp4 转换指令和 shuffled a4w4 scale 布局完全沿用 mxquant（输出可以直接喂 `gemm_a4w4_mxfp4`）。映射也沿用 mxquant：一个块 = 16 行 × 整行 K，4 个 wave，wave `w` 处理 tile pair `w + 4 i`，lane `16 g + r` 负责行 `16 rb + r` 的第 `16 p + g + 4 j` 组；pass A 求每个 lane 的平方和（一个 tile pair 的 4 组展开，16 次 b128 加载在飞），行内 4 个 lane（`l ^ 16`、`l ^ 32`）用两次 lane swap 合并，4 个 wave 用 256 字节 LDS 合并；pass B 重新读行（L2）、乘 `inv_rms · gamma` 后量化。ABI：x @0、gamma @8、q @16、scale @24、M @32、eps @36。

主机测试 `tests/gpu/mxnorm_host.cpp`：CPU 参考把同一规则用在 f32 的规范化行上；scale 字节必须一致（每百万个 group 里有几个边界 group 差 1，因为设备的 rsqrt 是近似指令，允许 ≤ 1‰），每个反量化值必须在设备自己的 scale 的 fp4 半步长（1 × scale）以内；K = 512 / 2048 / 4096 / 7168，`code_bad = 0`，`gpu_verify --strict` 0 错误，84 VGPR。性能（K = 4096、M = 32768）：123 µs；分开做（rmsnorm 约 90 µs + mxfp4 量化）估计 ≥ 1.6 倍。


## layernorm 的 packed 数学（负结果）

把 layernorm 的"减均值 / 乘 inv / 乘 γ / 加 β"和"偏差平方"换成 `v_pk_add_f32` / `v_pk_mul_f32`（每条处理两个元素）后，VALU 指令数反而**增加**（N = 4096：1005 → 1158，N = 776：684 → 762，N = 400：1723 → 1883，N = 256：652 → 732，N = 200：793 → 873），带宽持平或更低（同一次运行里，N = 4096 5882 → 5873、776 4284 → 3798、400 3623 → 3096、256 5245 → 4916、200 3915 → 3340 GB/s）：packed 操作要求两个 f32 在相邻的 VGPR 里，而 bf16 解包产生的是分开的寄存器，所以每对元素还要一到两条 `v_mov` 来拼 / 拆，吞掉了省下的指令。要让 packed 数学有用，需要解包阶段本身就产生相邻的寄存器对（例如 `v_pk_` 系的 bf16 → f32 或者把 `v_lshlrev / v_and` 的目的寄存器按对分配），这是编译器 / 寄存器分配层面的事；kernel 层面不做。实验代码在归档 tag `archive/pk-layernorm`，不合入。

## mxfp4 规则只有一份

`rows/mxquant.rs` 里的"一个 32 元素组"的步骤（amax 舍入、E8M0、`ScalePkFp4F32` 链、frag4）现在是 `pub(crate) fn mxfp4_group`，`quant_mxfp4_bf16_k<K>` 和融合的 `rmsnorm_quant_mxfp4_bf16_k<K>`（`rows/mxnorm.rs`）都调用它；重构前后两个 kernel 的汇编 md5 完全相同（`quant_mxfp4_bf16_k512 / 4096 / 7168`、`rmsnorm_quant_mxfp4_bf16_k512 / 4096 / 7168`、`quant_mxfp8_bf16_k4096`）。
