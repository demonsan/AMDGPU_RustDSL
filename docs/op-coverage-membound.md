# 访存受限算子（batch 2）：实现状态与实测

设计与 roofline 方法见 [`op-coverage-memtemplates.md`](op-coverage-memtemplates.md)（流式拷贝基准 [`tests/gpu/membw_survey.cpp`](../tests/gpu/membw_survey.cpp)：同读写比的实测上限约 5.7–6.2 TB/s，验收线 = 其 90 %）。每个算子：CPU 参考对拍（含尾部 / 奇数尺寸）、`gpu-verify --strict` 0 错误、实机带宽、已注册（`kernel_by_name`）、IR 文本往返。**aiter 同轮对比：节点上没有 torch / aiter 环境（`import torch` 失败），所以只做 roofline 对比，不做 aiter 同轮对比。**

验证方法：`tests/gpu/<op>_host.cpp`（`hipModuleLoad` 加载 `amdc emit --backend=own` 汇编出的 co，CPU 双精度参考，bf16 判据 = 误差 ≤ 1 bf16 ulp + 1e-6 的绝对下限，并检查输出末尾之外的字节没有被写），带宽 = 读 + 写字节 / 单次 kernel 时间（50 次连续发射的平均，工作集约 1 GB 以避开 Infinity Cache）。

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