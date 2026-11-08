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

D ∈ {64, 128, 256}（全旋转维），H ∈ 1…256；正确性：T ∈ {9 … 37}，H ∈ {1, 8, 32, 40}（H = 40 的最后一个 pass 部分 head 越界）。**限制**：H · L < 64 的内核（如 MQA 的 H = 1）一个 wave 里只有一部分 lane 有 head，带宽低（H = 1, D = 64 只有 22 %）→ 需要"一个 wave 处理多个 token"的变体（待做）；部分旋转（rotary_dim < D）还没有。

## KV-cache 写（模板 T7）

`cache/mod.rs`，flash 布局：`k, v [T][Hk][D]` → `key_cache, value_cache [S][Hk][D]`，按 `slot_mapping [T]`（i32，负数 = padding token，什么都不写）；一个 token 用 64 / 128 / 192 / 256 个 lane 拷贝一行（`Hk·D·2` 字节，每 lane 每 16 B 一个 b128，K、V 同一轮加载），一个 block 含 4 / 2 / 1 / 1 个 token。主机测试 `tests/gpu/cache_host.cpp`（逐字节对拍，未被写的槽保持哨兵值）。

| kernel | VGPR | 带宽（GB/s，随机槽位的置换） | 占 roofline |
|---|--:|--:|--:|
| `reshape_cache_flash_bf16_h<Hk>_d<D>` | 20 | Hk = 8, D = 128：4909；Hk = 8, D = 64：4716；Hk = 16, D = 128：4860 | 79–82 % |

限制：cache < 4 GiB（32 位 buffer 偏移；需要后端的 64 位地址乘加 B1 才能去掉）。低于 90 % 的原因：随机槽位的 2 KB 行写（相邻 token 的目的地不连续），在途量已按行 / 块给够，剩下的差距在 DRAM 页局部性；量化写（fp8 cache + scale）、NHD 以外的分块布局（vLLM 的 `[num_blocks][Hk][D/x][block_size][x]`）待做。