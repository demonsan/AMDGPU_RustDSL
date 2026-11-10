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

### 结构性受信事实（全部计入账本）

| 事实 | 含义 | 使用处 |
|---|---|---|
| `wg_bijection(outs, axes, r)` | 元组是 workgroup id 的单射 | GEMM XCD 光栅；mxfp8 FMHA XCD 块序；KV-split z 分解；reshape_cache 分块布局 (block, offset)（数据：slot 互不相同） |
| `wg_union(outs, axes, lo, hi, r)` | 各写类各自算出的同一个索引在 (workgroup, 类) 上单射 | causal 配对 q tile；reshape_cache flash 的 slot；MoE GEMM 输出行（数据：moe_sorting 每个 (token, slot) 只列一次） |
| `wg_winner(cond, axes, r)` | `if cond` 分支在沿 axes 的每组 workgroup 中只有一个执行 | decode16 `_c` 的末到者（同一计数器上 ticket S − 1） |

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

**`lds_owned(name, base, bytes, col_stride, row_stride)`（本轮新增）**：线程 t 拥有 `base + t·col_stride + r·row_stride + [0, col_stride)` 这些字节。
- 标记作用域内的每次访问都会被检查：地址必须是 `base + col_stride·tid + k·row_stride + r`，其中 0 ≤ r ≤ col_stride − size，且 row_stride ≥ 线程数 × col_stride。
- 全部通过的标记是 owner 事件：不同线程的 owner 事件互不冲突；同一线程先写后读按程序顺序，不需要 barrier。
- 访问其他线程的列仍是普通共享事件，照常检查。
- 测试：读自己的列后原地改写、中间不加 barrier → 通过；改写邻居的列 → 报竞争。
- 用途：sched 的 `moe_scatter_ordered`，可去掉为过检加的 barrier 和 `tile_trust_periodic`。

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
- （已解决）decode16 `_c` 的末到者写：`wg_winner`，12 / 12 已证明；
- `bf16_v3_ks`：分片原子的范围；
- （已解决）MoE GEMM、reshape_cache flash / blocked：数据互异事实（计数）+ 已知低位为零的原子按 c·2^z 步进；尚余 `reshape_cache_blocked_fp8` 的逐块 scale 写；
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
| VALU 写 SGPR 的冒险（本轮新增）| `pad_valu_sgpr`（RA 之后）：VMEM 读该 SGPR 需 5 个等待态，VALU 读需 2 个（LLVM GCNHazardRecognizer：checkVMEMHazards、hasVDecCoExecHazard）；`probe_valu_sgpr` 在硬件上复现了不加等待时读到过期 SGPR（2048 个值中错 1984 个）；LLVM 尾端在相同位置插入相同的 s_nop |
| 在途 load 的物理 WAW、物理 WAR | RA 之后的 `pad_inflight_waw` / `pad_physical_war`（本轮的 MLA / Vs128 错误发生在这里，不在 gpu_verify 的范围） |
| 均匀性分析（scalarize / waveuni） | 编译器自身的分析，没有独立证明 |
| M0 / LDS-DMA 语义、调度器、LLVM 尾端、汇编器 | 门禁、探针、md5 回归 |
| 可达 kernel 能否编译 | `dispatch_compile` 测试；本轮 `bf16_v3_causal` 的 SGPR 溢出就是这类漏网 |

### 5.1 冒险规则审计：LLVM GCNHazardRecognizer（gfx950）→ 本编译器

逐条对照 LLVM `GCNHazardRecognizer.cpp`（PreEmitNoopsCommon 与 fixHazards 的全部检查），按 gfx950 的子目标谓词筛选：GFX9 代、
`HasGFX940Insts` / `HasGFX950Insts`、gfx90a 的 MAI 规则。上一行就是本轮漏掉、已在硬件复现的那一条。

| LLVM 规则（等待态） | gfx950 | 本编译器 | 结论 |
|---|---|---|---|
| checkVMEMHazards：VALU 写 SGPR → VMEM 读该 SGPR（5） | 适用（VI 代起） | `pad_valu_sgpr`（RA 后，物理 SGPR，跨块；SALU 中途改写也计） | 已覆盖（本轮新增；`probe_valu_sgpr` 硬件复现，`tests/hazard_valu_sgpr.rs` 回归） |
| checkVALUHazards / VDecCoExec：VALU 写 SGPR 或 VCC → VALU 读（2） | 适用（GFX940 指令集） | SGPR：`pad_valu_sgpr`；VCC：`fix_value_hazards` 的 vccw（2） | 已覆盖 |
| VDecCoExec：VALU 写 VGPR → v_readlane / v_readfirstlane 读（1） | 适用 | `valu_writer_delay` 对 readfirstlane 读者（2） | 已覆盖（比 LLVM 多 1，保守） |
| VDecCoExec：VALU 写 EXEC → readlane / writelane（4） | 适用 | 不用 VALU 写 EXEC（v_cmpx 不生成） | 不适用 |
| checkRWLaneHazards：VALU 写 SGPR 作 readlane / writelane 的 lane 选择（4） | 适用 | `VReadLane` / `VWriteLane` 只用立即数 lane | 不适用 |
| TransForwarding：TRANS 结果 → 非 TRANS 的 VALU 读（1） | 适用（GFX940） | 超越函数结果延迟 `TRANS_LATENCY`（5，所有读者） | 已覆盖（保守） |
| DstSel / CvtScale forwarding：op_sel 高半 / SDWA dst_sel / fp8·fp4 cvt 写 → 任何读或 WAW（1） | 适用（GFX940 / GFX950） | `cvt_dst_sel_write` → hiw（1） | 已覆盖；**有意偏离**：同一 dword 的 Lo / Hi 配对写不加等待（`probe_fp4_chain` 硬件实测）；fp4 SR 变体的 cvt 未生成 |
| 12DWordStore：>8 字节 VMEM store 后 VALU 覆写数据 VGPR（gfx950：SGPR soffset 的 BUF 为 1，其余为 2） | 适用 | `wide_store_data`：2（物理寄存器，`pad_physical_war`） | 已覆盖（统一按 2，保守） |
| checkDPPHazards：VALU 写 VGPR → DPP 读（2）；VALU 写 EXEC → DPP（5） | 适用 | DPP 读者 `valu_writer_delay`（2）；EXEC 只由 SALU 写 | 已覆盖 / 不适用 |
| checkPermlaneHazards：VALU 写 VGPR → v_permlane 读（2）；v_cmpx 写 EXEC → permlane（4） | 适用（GFX950） | permlane 读者 `valu_writer_delay`（2）；不生成 v_cmpx | 已覆盖 / 不适用 |
| checkMAIHazards：VALU 写 VGPR → MFMA 读（2；gfx950 scaled MFMA 再 +1） | 适用 | vw（2）；`scaled` +1（硬件确认） | 已覆盖 |
| checkMAIHazards：XDL 写 → XDL 读 SrcA/B（NumPasses + 3，gfx950 再 +1：8 pass 12、16 pass 20）；SrcC 完全重叠且同 opcode（0） | 适用 | `mfma.latency`（12 / 20）；原位链互锁 | 已覆盖（非链一律满延迟，保守） |
| checkMAIVALUHazards：XDL 写 VGPR → VALU / VMEM / DS 读（同上公式）；→ VALU 覆写 WAW（同上） | 适用 | ready map（所有读者）；在途目的寄存器 `pad_physical_war` | 已覆盖 |
| checkMAIVALUHazards：XDL 读 SrcC → VALU 覆写 WAR（gfx940：16x16 7、32x32 15 量级） | 适用 | `pad_physical_war` 的 SrcC 窗口（12 / 21） | 已覆盖（保守） |
| checkReadM0Hazards：SALU 写 M0 → LDS DMA（1） | 适用（GFX9） | `sched::m0::pad`（1） | 已覆盖 |
| checkReadM0（SendMsg / GDS、MovRel / Interp、LDS direct） | 部分适用 | 不生成 s_sendmsg、GDS、movrel、interp；LDS direct 是 GFX11 | 不适用 |
| checkSMRDHazards（VALU 写 → SMRD 读，4） | 只 SI | — | 不适用 |
| checkDivFMasHazards、Get / SetReg、RFE、FPAtomicToDenorm、NSAtoVMEM | 适用或不适用 | 不生成 v_div_fmas、s_getreg / s_setreg、s_rfe、s_denorm_mode、NSA | 不适用 |
| fixShift64HighRegBug（v_lshlrev_b64 等的位移量在块尾 VGPR） | 适用（gfx90a+） | 不生成 64 位 VALU 移位 | 不适用 |
| checkInlineAsmHazards | — | 没有 inline asm；safe-asm 块由 `pin_check` 检查其内部等待 | 不适用 / 另行检查 |
| fixHazards 的 GFX10 / GFX11 / GFX12 项：vcmpx-permlane、VMEM→SALU 写、SMEM→VALU 写、vcmpx-exec WAR、LDS-branch-VMEM WAR、LDS direct、partial forwarding、trans use、WMMA、mask write、export priority、getreg wait idle、scratch base | 不适用（谓词只对这些代为真） | — | 不适用 |

### 5.2 LLVM 冒险预言机（强制）

`crates/amdgpu-codegen/tests/hazard_oracle.rs` 对表中每一对（生产者，消费者，操作数）比较 pinned LLVM tail 插入的等待态与我们最弱路径（非 strict 的值冒险 + 全部 RA 后冒险 pass）提供的等待态。LLVM 更多而又不在 `allowance` 里的对，测试失败。167406 对中 LLVM 更多的有 2357 对，全部归入以下三类，各有证据：

| 类 | 对数 | 结论与证据 |
|---|---|---|
| VALU 写 VCC → VALU 读 VCC（`v_cmp` / `v_add_co` / `VPtrAdd` 内部 → `v_addc` / `v_cndmask`），LLVM 2 | 660 | gfx950 硬件转发 VCC：`tools/vcc_hazard_probe.sh` 在 0 / 1 / 2 个等待态下全部正确（3 轮 × 12 种）。LLVM `hasVDecCoExecHazard` 对 VCC 过度保守 |
| VOP3P 生产者（`v_pk_*`）→ 任意读者，LLVM 1 | 1569 | pinned LLVM 把打包指令的 op_sel_hi 位当作 VOP3 的目的 op_sel（同一位）。上游 `getDstSelForwardingOperand` 已排除 VOP3P |
| 16x16x128 f8f6f4 的 FP4 / FP6 操作数，LLVM 12，我们 8 | 144 | 硬件：VALU 读者 7 个、VMEM 读者 1 个等待态即正确（`tools/mfma_lat_probe.sh`；FP8 为 11 / 5，我们按 20）。LLVM 对所有格式都按 8 pass 计 |

本轮由预言机发现并修正的两条（manifest 中 12 个 kernel 的 md5 改变，出厂 FMHA / GEMM / MLP 不变；`run_all.sh` 60 / 60 PASS）：
- **软子句（XNACK 重放）**：`hazards::pad_soft_clause`。连续的 VMEM（或 SMEM）构成子句，子句中任一指令不得读子句中写过的寄存器（D16 load 读自己的目的；一条 load 写回前一条 load 的地址寄存器），子句写过寄存器后的 store 另起子句；用 `s_nop 0` 断开。依据 LLVM `checkSoftClauseHazards`：我们的 code object 是 `xnack any`，开启 XNACK 的进程会重放子句。
- **高半转换的伙伴例外**：只豁免合并操作数（绑定的目的），把结果当数据读的转换仍等待 1 个等待态。

结论：除本轮补上的 VALU 写 SGPR 两条外，gfx950 适用的 LLVM 规则都已覆盖，且本编译器的等待数不小于 LLVM；唯一有意偏离是 cvt 的 Lo / Hi 配对，有硬件实测支持。

## 6. 建议（按性价比）

1. 采样 / top-k / moe_sorting 的 LDS 归约改用 LdsPlan tile 并加标记，使 `lds-untracked` 归零。
2. `amdc check-launch`（已做，见 §3）。
3. 全局写不相交证明（已做，见 §4）：下一步补双射事实（XCD 光栅）与步长下界，覆盖 GEMM / attention。
4. 区间引擎接入循环退出条件和 `x >> log2(g)` 这类关系，可去掉 §1(b) 的大部分 `assume_trusted`。
5. 调试模式把 `assume` / `assume_trusted` 编成运行时断言（违反时写调试缓冲区），覆盖 §1(a)。
