# FMHA 参考 Kernel：已核实事实与 1:1 验收

## 1. 参考对象

用户指定参考：

- 仓库：`https://github.com/ROCm/aiter`
- 路径：`hsa/gfx950/fmha_v4_fwd/fwd_hd128_mxfp8.co`
- 固定 commit：`7061cbd88b01ead4fb032ea37d8160f4bed91e37`
- Git blob：`805e5d5020da331d31510eeb2c386b29813e5f64`
- SHA256：`83331484c6abc12b460681de10081bd8925df1da94653784d26dd32fd4243438`
- 大小：63,432 字节

本地副本、远端 blob、SHA256 三者一致（Verified）。

反汇编在用户授权的 AAC16 Job 8705 / `mi355-gpu-53` 上以 CPU-only 只读方式完成：二进制经 stdin 传入，未写远端文件、未运行 GPU kernel、未编译候选代码。`.text` 的 58,836 字节全部连续解码，无 unknown 指令。

## 2. 二进制事实（Verified）

| 项 | 值 |
|---|---|
| 格式 | ELF64 ET_DYN，EM_AMDGPU |
| ELF ABI version | 4（COV6） |
| metadata version | `[1, 0]`（与 ELF ABI 不一致，保留原样不修正） |
| wavefront | 64 |
| VGPR | 246 |
| AGPR | 0 |
| SGPR | 99 |
| LDS | 84,224 B |
| private segment | 0 |
| kernarg | 656 B，align 4，单一 `_kernarg_raw` by-value |
| `descriptor.kernarg_size` | 0（与 metadata 656 不一致，保留原样） |

**该 binary 没有任何 AGPR 分配证据，实现不得预设 AGPR accumulator。**

## 3. Launch 与 ABI 契约

来自 `aiter/csrc/py_itfs_cu/asm_mha_v4_fwd.cu` 与 manifest：

- grid = `(ceil(seqlen_q / 256), nhead_q, batch)`
- block = `(512, 1, 1)` → 8 × wave64
- 当前 HIP stream
- kernarg：41 × 16 B slots = 656 B
  - `ptr_o` @ 0x000，`ptr_q` @ 0x010，`ptr_k` @ 0x020，`ptr_v` @ 0x030
  - `ptr_lse` @ 0x040，scalar @ 0x050
  - `ptr_q_descale` @ 0x200，`ptr_k_descale` @ 0x210，`ptr_v_descale` @ 0x220
- manifest 行：Q/K/V format 3（E4M3 FN），v_pack 0，out 2（BF16），Q/K scale mode 5（E8M0 per 1×32），V scale mode 1（F32 per tensor），head dim 128，tile Q 256，tile KV 128

## 4. Core Loop 范围（Verified）

局部符号 size 均为 0，因此边界取自实际分支目标与反汇编区间，而非符号 size。

| 区域 | VA 区间（左闭右开） | 指令数 | 字节 |
|---|---:|---:|---:|
| A steady loop | `0x2B68..0x3930` | 473 | 3,528 |
| B steady loop | `0x6AA0..0x77EC` | 455 | 3,404 |
| rollback s0 | `0x9D4C..0xA1F4` | 191 | 1,192 |
| rollback s1 | `0xA1F4..0xA69C` | 191 | 1,192 |
| rollback s2 | `0xA69C..0xAB44` | 191 | 1,192 |
| rollback s3 | `0xAB44..0xAFEC` | 191 | 1,192 |

合计：**1,692 条指令，11,700 字节**。

## 5. Wave Role 与 Paired Barrier

原始分支依据 wave id `s62 < 4` 选择 A / B：wave 0–3 走 A，wave 4–7 走 B，各自在自己的 backedge 循环。

每个 outer loop 有两个 `+0x80` 的 KV 进度更新与四个 workgroup barrier。对应同步点为：

| phase | A | B |
|---|---|---|
| 0 | `0x2ED4` | `0x6E0C` |
| 1 | `0x3248` | `0x7144` |
| 2 | `0x35B8` | `0x74B4` |
| 3 | `0x3928` | `0x77E4` |

DSL 必须支持「不同 wave-role 代码路径对应同一 collective phase」。但不能仅因两侧数量相同就认为安全：还需验证 drain / tail 路径的 phase 与控制条件。

## 6. 数值路径特征

hot loop 使用：

- `v_max3_f32` 折叠求 lane max（每个 lane 32 条，含初始化 `v_mov_b32_e32 v193, v52`）
- `v_cmp_nge_f32_e32` 与 `s_nop 1` 后 `s_cbranch_vccnz` 进入 out-of-line rollback
- `v_cvt_pknorm_u16_f32`（32 条）与 `v_cvt_scalef32_pk_fp8_bf16`（64 条，低/高 half 各一次写同一目标）
- `v_smfmac_f32_16x16x128_fp8_fp8`（in-place accumulator 依赖）
- `v_mfma_f32_32x32x64_f8f6f4` 与 `v_mfma_scale_f32_32x32x64_f8f6f4`
- `ds_read_b64_tr_b8` / `ds_read_b128` / `ds_read_b32`
- `s_waitcnt lgkmcnt(4/2/0)`、`s_waitcnt vmcnt(0)`

**scaled MFMA 在此 binary 中为 16 字节编码**（例：`0x30BC` → `D3AC0000 0003D9EE D3AE0834 05920914`）。不能用一个高层 GEMM API 隐藏该形态差异。

反汇编原文（已验证）：
`v_mfma_scale_f32_32x32x64_f8f6f4 v[52:67], v[20:27], v[4:11], v[52:67], v238, v236 op_sel_hi:[0,0,0]`
即 dst 16 dword、A 8 dword、B 8 dword、C 16 dword（in-place），**两个 scale 操作数是 VGPR**
（不是 SGPR），并带 `op_sel_hi` 修饰。gfx950 实测：scale 取 VGPR 低 8 位的 E8M0 指数，
结果为 `2^(a-127) * 2^(b-127) * (A @ B) + C`，A/B 字节按 E4M3 解码。

FP8 pack 的低/高部分写入必须保留 write mask 与旧值依赖，不能简化为「覆盖写」。

## 7. 1:1 验收方法

### 7.1 分层

```text
Semantic SSA + Resource SSA + Async provenance + Layouts
    → fragmentization / recipe expansion
    → MachineDAG（virtual）
    → apply ReferenceWitness（物理 tuple + 顺序 + wait + phase sites）
    → RA / schedule / legalize / verify
    → encoded core comparison
```

### 7.2 Witness 内容

- **RecipeWitness**：选择原 max3 / pack / SMFMAC / MFMA recipe，保留 16 B scaled form。
- **AllocationWitness**：expanded virtual value → 原 physical tuple；包含 in-place 版本与互不重叠的 K/V scratch 复用。
- **ScheduleWitness**：给 `MachineOpKey` 指定原顺序、wait 与 paired phase site。
- **EncodingGolden**：仅用于最终对比，不作为生成手段。

示例映射（只存在于 witness）：`ScoreFragments → v[52:115]`、`PackedIntermediate → v[52:83]`、`PackedProbability → v[52:67]`、`OutputFragments → v[116:179]`、`OperandScratch → v[20:51]`。

witness 必须通过 liveness / alias / read-write reservation 验证；同一个 bank 不是永久属于某个 logical value。

### 7.3 验收标准

1. A/B 的 473 / 455 条与 4×191 条 rollback 全部覆盖。
2. opcode、physical operand、modifier、encoding form、wait/NOP、顺序一致。
3. 非 relocation 字节一致；16 B scaled MFMA 全 DWORD 一致。
4. branch 以相同 logical target / CFG 验证；layout 不同则单独报告位移差异。
5. 若保留原 layout，六个区间 11,700 字节完全一致。
6. 功能验证使用同一批 packed 输入，原 kernel 与 candidate 对比；优先要求 BF16 输出 bit equality。
7. 覆盖 K = 256 / 384、边界 255 / 257 / 511 / 513、GQA 与 tail。
8. 不以内嵌原 `.co`、跳回原 kernel 或 patch 原 binary 冒充 compiler。

### 7.4 明确限制

witness 不能修正语义缺失。以下情况直接报错，不自动插 copy/wait 后仍宣称 1:1：

- generic softmax / whole-tile matmul 模型与参考数值路径不符
- 缺少参考使用的 SMFMAC / pknorm / 16 B scaled form
- 自动新增 loop header 或 end guard
- 把 partial completion 当作 whole-tile ready

## Vendor kernel schedule (disassembly findings)
See docs/status.md, section Kernel optimisation (round 41+), subsection Vendor kernel schedule.
