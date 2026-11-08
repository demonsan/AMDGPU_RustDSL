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