# 信任清单（gpu_verify / IR 检查依赖但不证明的东西）

本轮有三个潜在错误在验证下漏过：
- decode16 `_f` 的 LSE 写进别的 wave 仍在读的 V 区；
- GEMM `lds_epilogue` 复用 ring 时末轮预取 DMA 未等；
- MLA / Vs128 的在途 load 物理 WAW。

前两个属于"声明了就被信任"的 LDS 复用，现已由 GV-TILE-ALIAS-RACE 检查（见 §2）；第三个在编译器后端（RA 之后），不在 gpu_verify 的模型里，由 `pad_inflight_waw` 修复（见 §5）。

本文把剩下的信任面全部列出：每一项写明是否被检查、能否检查、现在怎样计数，以及本轮加的检查或攻击测试。

**原则：一个逃逸要么报错，要么在证书里计数，不能静默。** 本轮新增的计数：
- 账本（ledger）加入 `asm_opaque`；
- `gpu-verify` 输出新增一行 `bounds-checked, not race-checked`，按种类计数只做了越界检查、没做竞争检查的访问。

证书行格式：

```
certificate: N trusted assumes (a assume_trusted, b tile_trust_periodic, c region_trust, d asm_opaque), M trusted contracts, U unproven
bounds-checked, not race-checked (docs/trust-inventory.md): {"global-write-untracked": G, "lds-untracked": L}
```

固定这些行为的测试在 [`trust_inventory.rs`](../crates/amdgpu-kernels/tests/trust_inventory.rs)：
- 无标记的 LDS 竞争被计数而不是静默通过；
- 所有 workgroup 写同一全局地址被计数；
- safe-asm 块进入账本；
- 一个错误的 `assume_trusted` 只被计数、不被当作已证明。

## 1. 账本里的信任（计入 "trusted assumes"）

### `assume_trusted`

代码中共 34 处调用，生产 kernel 每个 0–12 条（出厂 `--shipped` 合计 26 条）。分两类。

**(a) 主机数据事实**：值来自显存里的数据，静态分析无法知道。例子：
- decode / MLA 的 context length ≤ max_pages·P；
- varlen 的 `cu_seqlens` 落在 [0, total] 内。

能否检查：
- 运行时可以检查：Python 包装可以在主机侧检查（需要一次 D2H，代价高），或 kernel 自己 clamp（多几条指令）；
- 静态不可能检查。

**(b) 区间引擎推不出的算术事实**：
- 变量移位，如 `head >> log2(gqa) < HK`（需要关系 H = HK·gqa）；
- 变量掩码与 ceil-div；
- XCD 双射（逻辑 tile / head 是物理编号的置换）；
- 由循环退出测试界定的计数器（`tile::Pipeline`、`combine_splits`）。

能否检查：可以，但要换更强的抽象域——关系域、循环退出条件进入归纳。现在逐条计数。

### 循环退出事实（#4，本轮新增）

- `break_if(c)` 之后，同一语句表的其余部分只在 c 为假时执行（break 条件是 wave 一致的）。`x < k` / `x >= k` 的操作数在此期间被收紧。
- `set_var` 在其语句表末尾生效，所以同一表里稍后的 break 也收紧回边上的值。
- 分析能证明的受信循环变量范围不再计入账本：只有初值或某一步仍未证明时才计数。
- 效果：`tile::Pipeline` 的计数器（`_vs128` / `_t128` / GEMM）和 `combine_splits` 的计数器已被证明。出厂证书 29 → 22，GEMM v4 4 → 3。`_kvsplit` 的计数器仍受信（计数）。

### 其余账本项

| 项 | 用途 | 现状 |
|---|---|---|
| `tile_trust_periodic` | slot / version 表达式非仿射时，声明协议按迭代周期性成立 | 生产 kernel 0 处 |
| `buf_trust` | 一个全局区域的可见性论证在验证器之外 | 生产 kernel 0 处（只有 `verifier_cases` 的一个示例） |
| `asm_opaque`（本轮新增） | safe-asm 块（`b.asm`、var asm）内部的访存不建模（GV-ASM） | 以前只是 Info、不计数；现在每个块计 1。只有两个探针用到 |

## 2. 被检查的声明（不是信任）

**`lds_reuse` / `LdsPlan::tile_over` / `private(.., overlays)`**：
- 一侧的写与另一侧被其他 wave 的访问，必须被 barrier 相位隔开，否则报 GV-TILE-ALIAS-RACE；
- 例外：能证明写落在写者自己 wave 的私有切片内（`own_reuse` 证明）。

**`lds_alias`（普通别名）**：
- 只由 `LdsPlan::commit` 产生：声明范围重叠、但真实字节足迹不相交（交错 ring 组），这是构造即证明的布局事实；
- 真实足迹重叠却没有声明 overlay 时，`commit` 直接 panic；
- kernel 代码中不再手写 `lds_alias`。

**`lds_private`**：每次访问都要证明落在本 wave 自己的 key 切片内（tile-cover）。

**`buf_last_arriver`**：
- 检查模式本身：fence / ticket / acquire 的位置、读多个切片；
- lite 变体再由 `asm_check::check_lite_contract` 检查（sc1、ticket sc0、barrier 前排空）。

**`any_lane`**：
- `if` / `break_if` 的条件必须可证 wave 一致；逐 lane 条件必须显式包上 `any_lane`（IR 验证器 `check_uniform_branches`）；
- 极性由 `probe_anylane` 的 9 种组合在硬件上固定。

**`inplace` / `frag_cat` 原地写**：
- 编译器只在"新鲜、单次使用、中间无读者"时才原地写，否则退回拷贝；
- 在途 load 的物理 WAW 由 `pad_inflight_waw` 在 RA 之后守住。

**`pin_agpr` / 寄存器计划**：RA 检查冲突（重复钉、类型不符、预算）。

**stage 契约**：由 `verify_stages` 检查。

## 3. 主机契约（计入 "trusted contracts"）

包括：`arg_tensor`（分配大小）、`ka_is` / `ka_range` / `ka_one_of`、`grid_is`、`sym_range`。

**编译期**：
- kernel 在这些前提下被证明；
- 前提本身由调用方保证，静态无法验证。

**运行期**：
- Python `amdgpu_jit.launch` 在每个调用签名的第一次启动时逐条求值并检查，违反时抛出 `ContractError` 并指出是哪一条；
- C++ 主机程序（`tests/gpu/*_host.cpp`）和外部启动器不检查。

**非 Python 调用方（本轮新增）**：`amdc check-launch <kernel> --ka OFF=V,... --grid X,Y,Z --size ARG=BYTES,... [--sym NAME=V,...]` 对一次具体启动逐条求值主机契约：
- 符号从 grid / kernarg 的直接定义或 `sym·c` 形式解出；
- 输出 `VIOLATED`（违反）、`unchecked`（缺输入）和汇总；
- 有违反时退出码为 1。

例如 GEMM v4 在 grid (16, 8) 下报告 `kernarg 0x18 = 4096, the contract requires MT·256 = 2048`。

## 4. 做了越界检查、没做竞争检查（本轮新增计数）

**跨 workgroup 的全局写不相交（本轮新增证明）**：
- 范围：所有不在 `buf_region` 标记内的普通全局 / buffer 写（原子写不需要）；
- 做法：把每个写地址化成符号线性式，原子是 workgroup id、lane 派生值、循环变量；结构相同的重算表达式共用一个原子。各写的常数差按单项式拆成伪数字位。
- 判据：所有数字位按系数排序后构成混合进制，即每个系数 ≥ 更小数字位的总跨度 + 访问大小；并且每个可能多于一个 workgroup 的轴都是一个数字位。满足时两个不同 workgroup 不可能写同一字节，计入 `proven["global-write-disjoint"]`；不满足的写仍计入 `global-write-untracked`；
- 合同里未定义、kernel 也不读的 grid 轴按 1 个 workgroup 处理（与 grid 范围同属主机契约）；
- 掩码写的哨兵臂（≥ 2^30 或负常数，被描述符范围检查丢弃，这一点已由越界证明确立）不算写；
- `AMDC_GW_DEBUG=1 amdc gpu-verify <k>` 逐参数打印失败原因和原子签名。

补充规则：
- **掩码上限**：掩码写的活臂只在 `x < bound` 时执行；夹紧写 `min(x, bound - 1)` 也算，但要求 x 唯一的 workgroup 数字位在最高位，且 c_u·hi_u ≤ bound - 1。若偏移里含 k·x、x 自身对其原子是单射，就把 x 的原子合成一个数字位（系数 k，取值 [min x, bound - 1]）。各写的 x 常数可以不同，会从各自常数中扣除。
- **`b.wg_bijection(outs, axes, reason)`**：受信事实，计入账本。声明 `outs` 这个元组是 workgroup id 在这些轴上的单射（例如 XCD 光栅、块序置换）；证明时把 `outs` 当作这些轴的数字位。只在它真正换来证明的地方使用：GEMM 光栅、mxfp8 FMHA 的 XCD 块序。

其余规则：
- **夹紧**：`min(x, y)` 也按夹紧处理。当 x 的 workgroup 数字位所在轴的 grid = ceil(bound / c) 时，夹紧总是落在本 workgroup 内。
- **`b.wg_union(outs, axes, lo, hi, reason)`**：受信事实，计入账本。几类写各自算出的同一个索引（causal 配对的 q tile x 与 nt - 1 - x）在 (workgroup, 类) 上单射，取值 [lo, hi]；证明时把它们当作同一个数字位。
- **decode16 的 G**：G 现为契约符号的点事实（`assume_trusted(q, G, G)`，替换原来的 [1, 16]，计数不变），使按 G 缩放的偏移保持线性。

已证明（出厂 / 常用，全部写）：
- `gemm_mxfp8_bs_v4`、bf16 / blockscale GEMM；
- mxfp8 FMHA `_vs128` / `_t128` / `_kvsplit` / combine；
- bf16 prefill v3（含 causal 配对）、bwd dot / dq / dkdv；
- decode16 `_f`；`_c` 的 12 个写中证明 10 个；
- MLA（两种形式）；
- rmsnorm 系列、量化、激活、rope、`topk_softmax`、`moe_sum` / `moe_reduce`、各 combine、`gemm_splitk_reduce`、argmax / sample。

代价：出厂账本 26 → 29（`_vs128` / `_t128` 的 XCD 块序，`_kvsplit` 的 z 分解），GEMM v4 3 → 4（光栅）。

仍未证明（原因 → 需要什么）：
- decode16 `_c` 的末到者写（ticket 复位与最终输出）：只有末到者执行；需要把已验证的 last-arriver 协议当作该轴的覆盖；
- `bf16_v3_ks`：分片原子的范围；
- MoE GEMM / reshape_cache：行号或 slot 是数据 → 需要数据互异事实（受信、计数）；
- （已解决）持久化 `mlp_fp8_ws_fused`：循环变量从 workgroup id 出发、每步加 grid 大小（v = wg + k·grid），不同 workgroup 走不相交的剩余类，可作为该轴的数字位；32 / 32 已证明。

**`lds-untracked`**：
- 范围：不在任何 tile 标记内的 LDS 访问；
- 越界已证明，跨 wave 的顺序 / 可见性不检查；
- 生产 kernel 中的分布（其余被调查的 kernel 都是 0）：

  | kernel | lds-untracked |
  |---|--:|
  | `sample_topkp_bf16_v*` | 255 |
  | `topk_per_row_*` | 73 |
  | `moe_sorting_*` | 54 |
  | `argmax_bf16_v*` | 20 |
  | `rmsnorm_quant_mxfp4_*` | 5 |

**能否检查**：
- LDS：把这些 kernel 的 LDS 归约区声明为 tile 并加标记（LdsPlan），就会进入协议检查；
- 全局写：可以对"写地址关于 workgroup id 仿射、跨步不小于单个 CTA 的写足迹"做符号证明；对按 slot 散写的 kernel（`reshape_cache` 的 slot_mapping、`moe_sorting`）则只能依赖主机数据前提（slot 互不相同），属于 §1(a) 类。

## 5. 不在验证模型里的东西（编译器 / 硬件信任，未计数）

| 项 | 怎样把关 |
|---|---|
| 冒险模型：`op_attrs` 的延迟、等待态、MFMA 原位链互锁（本轮扩展到 bf16 / f16 / i8 与 zero-C 起始） | 有 tbl 行的 MOp 对照 LLVM hazard oracle；own-only 的 MOp 只靠硬件探针（`probe_mfma_p1` 等） |
| 等待计数 | `verify_waits`：每条路径上的 vmcnt / lgkmcnt |
| 在途 load 的物理 WAW、物理 WAR | RA 之后的 `pad_inflight_waw` / `pad_physical_war`（本轮的 MLA / Vs128 错误发生在这里，不在 gpu_verify 的范围） |
| 均匀性分析（scalarize / waveuni） | 编译器自身的分析，没有独立证明 |
| M0 / LDS-DMA 语义、调度器、LLVM 尾端、汇编器 | 门禁、探针、md5 回归 |
| 可达 kernel 能否编译 | `dispatch_compile` 测试；本轮 `bf16_v3_causal` 的 SGPR 溢出就是这类漏网 |

## 6. 建议（按性价比）

1. 采样 / top-k / moe_sorting 的 LDS 归约改用 LdsPlan tile 并加标记，使 `lds-untracked` 归零。
2. `amdc check-launch`（已做，见 §3）。
3. 全局写不相交证明（已做，见 §4）：下一步补双射事实（XCD 光栅）与步长下界，覆盖 GEMM / attention。
4. 区间引擎接入循环退出条件和 `x >> log2(g)` 这类关系，可去掉 §1(b) 的大部分 `assume_trusted`。
5. 调试模式把 `assume` / `assume_trusted` 编成运行时断言（违反时写调试缓冲区），覆盖 §1(a)。
