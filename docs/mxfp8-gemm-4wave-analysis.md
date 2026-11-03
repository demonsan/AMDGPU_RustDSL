# MXFP8 blockscale GEMM「4-wave pin-AGPR」内核分析（aiter PR #6696, gfx950）

目标实例：`opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_u2_gm16_wide1_vf1.co`（kid 14）。
源码：`C:\Users\blyu\Documents\Workspace\aiter_pr6696`（branch `fyz/opus_blockscale_mxfp8_gemm`，shallow）。

引用缩写（相对 aiter_pr6696 根目录）：

| 缩写 | 文件 |
|---|---|
| PIPE | csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh |
| TR | csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh |
| OPUS | csrc/include/opus/opus.hpp |
| LAUNCH | csrc/opus_gemm/include/gfx950/opus_co_launch_gfx950.cuh |
| GEN | csrc/opus_gemm/codegen/gen_instances_gfx950.py |
| COMMON | csrc/opus_gemm/opus_gemm_common.py |
| BUILD | csrc/opus_gemm/gen_co/build_co.py |
| JSON | csrc/opus_gemm/gen_co/co_kernels_gfx950.json |
| BINFO | csrc/opus_gemm/gen_co/gfx950/build_info.json |
| SHUF | aiter/ops/shuffle.py |
| HIPC | csrc/include/aiter_hip_common.h |
| CSV | aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv |
| ELF | 本地用 Python 解析 .co 的 ELF note / kernel descriptor 得到的结果（本机无 llvm-objdump，**未做反汇编**） |

凡标「推断」的内容是根据源码推导、未经 ISA 反汇编验证。

---

## 0. 名字解码

`256x256x256x128` = `BLOCK_SIZE x B_M x B_N x B_K`（COMMON:2901-2906 用 `(block_size, *tile)` 拼名，tile=[256,256,128]，JSON:279）；
`2x2` = wave 布局 T_M×T_N；`16x16x128` = MFMA W_M×W_N×W_K；`1x128x128` = scale 粒度 GROUP_M×GROUP_N×GROUP_K（COMMON:2902-2906, TR:76-78）。
变体后缀是 JSON 中手写字段 `variant`（JSON:281），symbol = 全名，`.co` 文件名/name 会删掉默认段 `sn0/sl0/sched0`（COMMON:2876-2879, 2918-2922）。

| 段 | 含义 | 实现 |
|---|---|---|
| pm | PAD_M（M 尾块）。pm0 → `PadM=false`，m_align=256 | TR:61, 130-133（panel_traits 第 3 模板参数，JSON:287 `"false"`）；PIPE:433-442, 644-652, 1244-1250；JSON:294 |
| sp | SCALE_PANEL_K_CAPACITY：LDS 中缓存的 K128 scale 组数（16/32/64） | TR:120-128；PIPE:548, 713-733 |
| u | LOOP_UNROLL → `#pragma clang loop unroll_count` | TR:134；PIPE:1013, 1017 |
| gm | GROUPED_M（0/4/8/16）CTA 光栅化分组 | TR:148-155；PIPE:405-418 |
| wide | WIDE_OUTPUT：C 的 64 位基址 + 以 tile 为界的 32 位 buffer 偏移 | TR:148,153；PIPE:453-462 |
| sn | STATIC_N（编译期 N），经 `opus_gemm_bpreshuffle_static_shape_traits` | TR:183-188；PIPE:381-387 |
| sl | SCALE_LOAD_MODE：0=scale 原始值放持久数组；1=局部临时数组；2=把 refill 移出内层循环、按 panel 分块循环 | TR:66,171-181；PIPE:656-665, 793, 1007-1019 |
| sched | SCHEDULE_POLICY：0=全部 sched_group_barrier；2=只保留 MFMA(0x008) 组；1=全部关闭（只剩 sched_barrier(0)） | TR:163-169；PIPE:813 等处的 `if constexpr` 条件 |
| vf | 编译选项 `-mllvm --amdgpu-mfma-vgpr-form` 是否存在（vf1 有，vf0 无；逐条核对 JSON 全部 pin_agpr 条目一致） | JSON:282；kid 29（vf0）同 body_args 但无该 flag |

kid 14 的完整 traits：`panel_traits<64, 2, false>` + `4wave_pin_agpr_traits<CoTraits, matrix_view::tile, output::lds, UnsignedThreadIndex=false, WideOutput=true, GroupM=16>`（JSON:286-290）。

---

## 1. 数据布局

### 1.1 A（激活）
- dtype `opus::fp8_t`（PIPE:395），MFMA 格式码 fmt=0 = FP8 E4M3（OPUS:3126, 3146）。
- 行主序 `[M][K]`，`stride_a = K`（元素=字节），host 设定 GEN:3194；batch stride `m*k`（GEN:3195）。
- 全局视图：tile 模式下 buffer 基址 = `ptr_a + batch*stride_a_batch + row*stride_a`，pm0 时 num_records 默认 0xffffffff（无越界保护）（PIPE:430-444, OPUS:1852-1853, 1759-1760）。

### 1.2 B（权重，预 shuffle）
- 逻辑 `[N][K]` fp8 e4m3，host 必须先 `shuffle_weight(WQ, layout=(16,16))`（aiter/ops/opus/gemm_op_a8w8.py:338-343）。
- `shuffle_weight`（SHUF:194-205）：IN=16, IK=16, BK=32, K16=16/elem_size=16；`view(N/16,16,K/32,2,16).permute(0,1,3,4,2,5)`。
- **精确字节映射**（逻辑 (n,k) → 内存偏移）：
  ```
  off_B(n,k) = (n/16)*16*K + (k/16)*256 + (n%16)*16 + (k%16)
  ```
  即每 16 行 N 为一个连续 16·K 字节块；块内按 k16 组排列，每组 256B = 16(n) × 16(k)。
- 内核：`stride_b = K`（GEN:3194），`gb_offset(half,tile_k) = half*128*stride_b + tile_k*128*16`（PIPE:555-560）：一个 N16 块内 K128 段 = 8×256 = 2048B 连续。

### 1.3 A scale（x_scale, 1×128）
- dtype E8M0 字节（`unsigned char`，PIPE:399；host 接受 `fp8_e8m0` 或 `u8`，GEN:3184-3186）；值 v 代表 2^(v-127)（OPUS:3127）。
- 逻辑形状 `[M][K/128]`，但**内存必须是 K-major/“列主序字节”：`sfa_mem[kg*M + m]`**。内核 `stride_sfa = m`（GEN:3197），基址 `ptr_sfa + row`，偏移 `kg*stride_sfa + m_local`（PIPE:471-472, 563, 257-277）。
- 注意：host 检查允许 `x_scale.is_contiguous()` 或 `stride==(1,m)`（GEN:3187-3190），但 contiguous 情形要求数据已是列主序字节（gemm_op_a8w8.py:1125-1131 注释；测试做法 `x_scale.transpose(0,1).contiguous().view(*x_scale.shape)`，op_tests/test_gemm_a8w8_blockscale.py:153）。
- `stride_sfa_batch = m*(k/128)`（GEN:3198）同时用作 buffer num_records：`stride_sfa_batch - row`（PIPE:471-472）——自写 host 必须正确填写，否则越界读返回 0。

### 1.4 B scale（w_scale, 128×128）
- E8M0 字节，逻辑 `[N/128][K/128]`，行主序 contiguous（GEN:3190），`stride_sfb = k/128`、`stride_sfb_batch = (n/128)*(k/128)`（GEN:3197-3198）。
- 每个 wave 的 buffer：基址 `ptr_sfb + (col/128 + wave_id%2)*stride_sfb`，num_records = `stride_sfb`（PIPE:473-476）。

### 1.5 scale 何时/如何生效
- **直接作为 MFMA 的 scale 操作数**，没有 epilogue 反量化：`base_mma{}(a, b, c, sfa, sfb, number<m_repeat>, number<0>)`（PIPE:90-95）。
- `mfma_adaptor_swap_ab` 交换 A/B：硬件 srcA=B 片段、srcB=A 片段，硬件 scale_a=sfb(op_sel 0)、scale_b=sfa(op_sel=m_repeat)（OPUS:3711-3714, 3118-3120 → `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(B, A, c, fmt=0, fmt=0, opsel_sfb=0, sfb, opsel_sfa=e, sfa)`）。
- 一个 K128 scale 广播到硬件的 4 个 K32 lane 组（TR:80-83）。每 K tile（128）换一次 scale；所以 K 方向排列只要 A/B 一致即可。
- 每 lane 的 `v_sfa[h]`（u32）打包 4 个字节 = 本 lane 行在 4 个 m_repeat（e=0..3）下的 scale，`op_sel=e` 选字节（PIPE:84-90, 301-318, TR:73 `VEC_SCALE_SF == E_M`）。
- `v_sfb[h]` 只用 byte0（op_sel 0）；LDS 中写入的是 0 扩展的 u32（PIPE:719-728，注释称“replicated”但代码只写 raw byte，PIPE:342 vs 727）。

### 1.6 输出 C
- bf16 行主序 `[M][N]`，`stride_c = n`（PIPE:397, GEN:3194）。fp32→bf16 用 `static_cast`（gfx950 硬件转换，RNE，OPUS:1218-1220）。
- wide1：buffer 基址 = `ptr_c + int64(row)*stride_c + col`，num_records = `((tile_rows-1)*stride_c + 256)*2`，偏移相对 tile（PIPE:453-462），支持 >2GB 的 C。

---

## 2. 分解

- grid = `dim3(n/256, m/256, 1)`，block = 256，动态 LDS = 0（GEN:3224-3226；pm0 时 rounded_m=m）。M=N=4096 → 16×16 = 256 WG。
- `block_n = blockIdx.x, block_m = blockIdx.y, batch = blockIdx.z`（PIPE:403-404, 422；OPUS:1620-1624）。
- gm16 重映射（PIPE:405-413）：`linear = by*gn + bx`，`block_m = first_m + local % group_rows`，`block_n = local / group_rows`。4096²：gm=gn=16 → `block_m = linear%16, block_n = linear/16`，连续 16 个 WG 共享同一 B 列 panel。（gm0 且 m、n 均为 512 倍数时改为交换 block_m/block_n 最低位的 2×2 swizzle，PIPE:414-418。）
- wave：`wave_id = tid/64`（readfirstlane），`wave_id_m = wave_id%2`，`wave_id_n = wave_id/2`（PIPE:426-427, 478-479）。
- CTA 256×256 分为 2×2 个 128×128 “half”象限（TR:48-49）。每个象限内，wave(wm,wn) 负责 E_M×E_N = 4×4 个 16×16 MFMA 块（TR:57-59, 111），块交错：
  - 行 `= h_m*128 + 32*e + 16*wm + (lane%16)`，列 `= h_n*128 + 32*j + 16*wn + …`（PIPE:1219-1220 给出同一公式）。
  - 每 wave 总计 128×128 输出（4 象限 × 4×4 × 16×16）。
- **每 K step(128) 每 wave：64 条 `v_mfma_scale_f32_16x16x128_f8f6f4`**（4 象限 × 16，PIPE:800-997；E_K=1，TR:59,111）。
- 寄存器预算（每 lane）：
  - 累加器：4 个 CTile × 16 frag × 4 fp32 = 256 个 32 位寄存器（PIPE:531-538, OPUS:3139）。
  - A 片段：2 half × 4 × 8 dword = 64；B 片段：2 × 4 × 8 = 64（每片段 32 字节，OPUS:3137-3138；PIPE:524-526）。
  - AGPR 固定分配（PIPE:14-20）：`a[0:31]`=A half0，`a[32:63]`=A half1，`a[64:95]`=B half0，`a[96:127]`=B half1，`a[128:147]`=C11 frag 11..15（主循环），`a[148:211]`=C01（16 frag）。→ **agpr_count = 212**（BINFO:265；ELF 同）。
  - ArchVGPR：accum_offset = 244（ELF：COMPUTE_PGM_RSRC3 accum_offset 字段）；`.vgpr_count = 456 = 244 + 212`（BINFO:264）。C00+C10+C11[0..10] = 172 个 VGPR，其余为 scale、地址、暂存（`a1_m3_next`/`b1_n3_next` 各 8，PIPE:960-967）等。
  - SGPR 104，无 spill，无 scratch（BINFO:266-271）。
  - scale：`v_sfa[2]`、`v_sfb[2]`、`v_sfa_next`/`v_sfb_next`（各 2 dword）（PIPE:543-546）。

---

## 3. 主循环调度

### 3.1 LDS 中放什么
- **A 和 B 都经 LDS**，双缓冲（2 stage × 2 half）；**两个 scale panel 也在 LDS**（PIPE:496-507）。
- LDS-direct 载入：`async_load<16>` → `__builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc, dst, 16, voff, soff, imm, aux)`，即 gfx950 的 `buffer_load_dwordx4 ... lds`（OPUS:1889-1897）。目的地址按 wave 线性（M0 基址 + lane*16），因此 LDS 以 1024B 的 “slot”（64 lane × 16B）组织，slot 间距 `P = 1024 + 32` 字节（padding 32，TR:85-89；PIPE:572）。注意 imm offset 同时平移源与目的（OPUS:1870），代码在 voffset 中预减 `local*P`（PIPE:594-595）。
- A half 的 slot：`slot s = 4*i0 + 2*wn' + wm'` 存 8 行 × 128B，行 = `32*i0 + 16*wn' + wm' + 2*r`（r=lane/8，k=(lane%8)*16）（PIPE:116-158 prologue 布局；主循环版本 PIPE:573-629，issue (g,L) → slot 4g+L）。每 half 16 slot = 16896B。
- B half 的 slot：`slot = 8*(nb/4) + 4*kh + nb%4`，nb=N16 块号(0..7)，kh=K128 的前/后 64；每 slot = 4 个 k16 组 × 16n × 16k，与全局预 shuffle 的 1024B 连续段一一对应（PIPE:160-203, 609-612）。全局读完全合并（每条指令 1KB 连续）。

### 3.2 全局 load
- 生产者分工：**wave 0/1（wn=0）载 A 的 half wm；wave 2/3 载 B 的 half wm**，每 wave 每 K tile 16 条 `buffer_load_dwordx4 lds`（=16KB）（PIPE:569-590, 600-629）。prologue 中 4 个 wave 共同加载 stage0（PIPE:740-755）。
- `matrix_pair_stride = 16*stride_a` 也用于 B（PIPE:585）→ 要求 `stride_a == stride_b`（host 两者都 = K，GEN:3194）。
- scale 全局 load：SFA 每 lane `buffer_load_dwordx4`（16 个连续 M 字节/一个 kg），每 wave 4 pass（PIPE:638-655, 257-277）；SFB 用 `buffer_load_ubyte`，仅 wave 0/1、每 lane 一个 kg（PIPE:720-723, 320-335）。
- SFA 写 LDS 前在 VGPR 中做 4×4 字节转置：`v_mov_dpp quad_perm 0xb1` + `v_perm_b32` + `v_mov_dpp 0x4e` + `v_perm_b32`（PIPE:676-689），然后 `ds_write_b32`（PIPE:707）。
- LDS 中 SFA 布局（由读写两侧推导）：`s_sfa[(kg&63)*256 + (wm*16+lm)*8 + h*4 + e] = sfa[row=128h+32e+16wm+lm][kg]`（PIPE:565, 301-318, 279-299）；SFB：`s_sfb[(kg&63)*8 + h*4 + 0..3]`（PIPE:566, 337-353）。

### 3.3 ds_read 形式
- A/B 片段：每片段 2 × `ds_read_b128`（VEC 16 字节，PIPE:49-53, 69-73），**目的寄存器为 AGPR**（`[[clang::amdgpu_pin_agpr(...)]]`），每 K tile 每 wave 32 条（A 16 + B 16；TR:95-96,115 中 a/b_ds_read_insts=8 指每 half）。
- lane 内容：A frag e：行 `h*128+32e+16wm+lane%16`，K 字节 `kv*64 + (lane/16)*16 + [0..15]`，kv=0/1 对应片段 dword 0-3/4-7（PIPE:205-227）；B frag j 同样的 K 映射，列 `32j+16wn+lane%16`（PIPE:229-255）。A/B K 排列一致 + scale 在 K128 内统一 ⇒ 结果正确。
- scale：`ds_read_b64`（VEC_SCALE_SF_PAIR=8，一次读两个 half，PIPE:801, 855）；prologue 用 `ds_read_b32`（PIPE:770-773）。
- epilogue：`ds_write_b64`（4×bf16）暂存、`ds_read_b128`（8×bf16）回读（PIPE:1261-1264, 1243）。

### 3.4 迭代结构（每 K tile = 64 MFMA，分 4 个象限相位）
寄存器中持有 tile k，LDS stage `s^1` 持有 tile k+1，本迭代把 tile k+2 写入 stage `s`。（PIPE:790-1006）

| 相位 | MFMA | 交错的其它指令 |
|---|---|---|
| C00 = A0×B0 | 16 | refill 检查（PIPE:793）；`ds_read_b64` 下一 tile 的 SFA（801）；**`s_waitcnt vmcnt(0) lgkmcnt(0)`（804-805）+ `s_barrier`（808）**；然后 prefetch issue 0-4 与 MFMA 交错（812-836） |
| C10 = A1×B0 | 16 | prefetch issue 5-11（844-883）；SFB 下一 tile `ds_read_b64`（855）；最后重载 B0[0]（890） |
| C01 = A0×B1 | 16 | prefetch issue 12-15；重载 B0[1..3]、A0[0..3]（在其最后一次被用之后，897-946）；提交 `v_sfb[0]`、`v_sfa[0]` |
| C11 = A1×B1 | 16 | 重载 A1[0..2]、B1[0..2] 直接到 AGPR；A1[3]、B1[3] 先暂存到 VGPR（960-967），最后一条 MFMA 后再写入 AGPR（999-1000）；提交 `v_sfb[1]`/`v_sfa[1]` |

- LDS 双缓冲深度 2，全局预取距离 2 个 tile（`future_tile = k_tile + 2`，PIPE:795-797）；寄存器片段为“用完即重载”的原地流水（无寄存器双缓冲，A1[3]/B1[3] 例外）。
- 每迭代只有 **1 次 barrier + 1 次 vmcnt(0)/lgkmcnt(0)**（PIPE:804-808）；panel 边界时 refill 额外 2×(lgkmcnt(0)+barrier)+vmcnt(0)（PIPE:715-731）。sp64 且 K≤8192 时 refill 永不触发（`(next_tile & 63)==0` 需 next_tile=64，而 loops=64 时 next_tile≤63）。
- 循环：`for (; tile+2 < loops; ++tile)`（PIPE:1017-1018）→ K=8192 时 62 次（u2 展开），之后一个不发全局 load 的“epilogue 迭代”（PIPE:1021-1169），最后一个 tile 纯 64 MFMA（PIPE:1172-1204）。
- 优先级：`s_setprio 1` 进入主循环（PIPE:786），epilogue 迭代中在 wait/barrier 周围降为 0（1036-1040），最终 MFMA 后 `s_setprio 0`（1205）。
- 调度提示：每个相位末 `__builtin_amdgcn_sched_barrier(0)`（PIPE:803/806/840/893/949），相位内 `__builtin_amdgcn_sched_group_barrier(mask, n, syncid)`，syncid=相位号 0..3；mask 0x008=MFMA、0x100=DS read、0x800 用在每个 buffer→LDS 发射处（PIPE:813-814 等）。0x800 在我所知的上游 LLVM SchedGroupMask 列表（止于 0x400 TRANS）中未出现，含义**无法从源码确认**（构建用的是定制工具链 `yuyzhang512-amdgpu-pin-op-dst`，BINFO 顶层 `llvm_bin`）。迭代末 `sched_barrier(0x004)`（只允许 SALU 跨越，PIPE:1001）。
- 编译选项（JSON:282）：`--amdgpu-mfma-vgpr-form`、`-enable-post-misched=0`（关闭 post-RA 调度，保留手排顺序）、`--lsr-drop-solution=1`、`--amdgpu-kernarg-preload-count=32`（但 ELF kernel descriptor 中 kernarg_preload=0、user SGPR=2，即实际未预加载——ELF）、`-amdgpu-early-inline-all`、`-amdgpu-function-calls=false`、`-fgpu-flush-denormals-to-zero`。
- 空 inline asm `asm volatile("" : "+v"(x))` 用作优化屏障，固定 voffset/lane_id 在 VGPR（PIPE:596, 616, 672, 792, 797, 1232）；没有真正的指令级 inline asm。

---

## 4. AGPR pinning

- 机制：定制 clang 属性 `[[clang::amdgpu_pin_agpr(N)]]` 作用于**赋值语句**，把该语句产生的值放到物理 `a[N...]`（PIPE:51, 58, 71, 89, 92, 999-1000, 1067-1068, 1092）。build 时加 `-Werror=unknown-attributes -Werror=ignored-attributes` 确保属性不被忽略（BUILD:347-348）。
- A/B 片段：`load_operand_chunks_pinned`/`load_operand_fragment_pinned` 把每个 `ds_read_b128` 的结果直接 pin 到 AGPR（PIPE:41-74）；断言范围 < 128（PIPE:46, 66）。构建校验：反汇编中 `ds_read_b128 a[N:` 必须覆盖 `a[0:63]`(A) 与 `a[64:127]`(B)（pin_mask=3，BUILD:310-322；JSON:293）。
- C：只有 C01 整体 pin 到 `a[148:211]`（通过 `pinned_accumulator_array` 类型标签，PIPE:23-38, 532-536, 91-93）；C11 的 frag 11..15 在主循环中显式 pin 到 `a[128:147]`（PIPE:975-994，`EXPLICIT_AGPR_BASE=128, INDEX_BASE=11`）。C00/C10/C11[0..10] 不 pin，在 vf1 下为 VGPR-form MFMA。pin_mask 不含 bit 4，所以“C/D 同一 AGPR”的校验（BUILD:323-337）**对本 kid 未执行**。
- 为什么省掉 `v_accvgpr_read/write`：gfx950 统一 512 寄存器文件，256+64+64 个值无法全放 ArchVGPR。把 MFMA 的 **srcA/srcB 放 AGPR**，并让 `ds_read` 直接写 AGPR，则操作数“LDS→AGPR→MFMA”全程无搬运；累加器主要留在 VGPR（vgpr-form），epilogue 的 bf16 转换可直接读 VGPR。没有 pin 时编译器在 vgpr-form 下会把 A/B 先读进 VGPR，或在 AGPR-form 下需要 `v_accvgpr_*` 拷贝（推断，未反汇编验证）。
- 约束：
  - MFMA 的 srcC 与 vdst 必须同类（同为 AGPR 或同为 VGPR）；C01/C11-tail 的 MFMA 因此是 AGPR-form，其余为 VGPR-form（推断）。
  - VALU 不能直接读 AGPR：epilogue 对 C01（及 C11 tail，若仍在 AGPR）做 `cast<bf16>` 之前需要 `v_accvgpr_read_b32`（推断，PIPE:1301-1311）。
  - epilogue 迭代把下一 tile A0[0] 暂存到 `a[128:135]`（`A0_M0_STAGE_AGPR_BASE=128`，PIPE:18, 1065-1068），与主循环中 C11-tail 的 `a[128:147]` 重叠；尾部 C11 MFMA 不再显式 pin（PIPE:1136-1156）。编译器如何搬移 C11 tail 以避免冲突**无法从源码确定**（需反汇编）。
  - 片段重载必须在该片段最后一次被 MFMA 读取之后发射（见 3.4 的顺序），因为没有寄存器双缓冲。

---

## 5. Epilogue

- 无 scale 运算（scale 已在 MFMA 中）。kid 14 用 `Output::lds` 路径（JSON:290；PIPE:1230-1325）。
- 步骤：`vmcnt(0) lgkmcnt(0)` + barrier（PIPE:1282-1285）→ 每象限：16 frag `cast<bf16>`，以 `ds_write_b64`（4 bf16）写入 LDS，行距 `B_N+8 = 264` 个 bf16（PIPE:541, 1255-1265）；barrier（除 (1,0) 象限外先 vmcnt(0)，PIPE:1270-1278）；每线程 8 次 `ds_read_b128` + `buffer_store_dwordx4`（aux=2，gfx940+ 上 bit1 = NT，推断）（PIPE:1239-1251, 1279）。每象限 128×128×2B = 32KB = 256 线程 × 8 × 16B。
- C 暂存复用 A/B 的 LDS：256 行 × 264 × 2B = 135168B，恰为 matrix LDS 大小（PIPE:497-503）。
- 由于 swap_ab，lane 持有一行的 4 个连续 N（`lane%16` 为行），因此暂存写是 8B 一组（OPUS:3687-3697）。
- 边界：pm0 不做任何 M/N 边界检查（PIPE:1248-1250）；要求 `m%256==0 && n%256==0 && k%128==0`（GEN:3214）。pm1 才对行做 `row+output_row<m` 判断（PIPE:1244-1247）。另有 `Output::direct` 路径用 `v_permlane16_swap` 直接写全局（PIPE:1208-1229），本 kid 不用。

---

## 6. LDS 与占用率

| 区域 | 字节 | 来源 |
|---|---|---|
| A：2 stage × 2 half × 16 slot × 1056 | 67584 | TR:98 |
| B：同上 | 67584 | TR:99 |
| SFA panel：256 × 64 | 16384 | TR:106-107 |
| SFB panel：2 × 64 × 4 | 512 | TR:108 |
| 合计 | **152064** | TR:117；BINFO:272；ELF group_segment_fixed_size |

- gfx950 每 CU LDS 160KB（TR:117 断言 ≤163840）→ **1 WG/CU**。
- `.vgpr_count=456`（>256）→ **每 SIMD 1 wave**；launch_bounds(256, 1)（COMMON:2891-2892, BUILD:276）。即 4 wave/CU，每 SIMD 一个。
- 4096×4096：256 WG，恰好铺满 256 CU 一轮（CU 数取自 CSV 的 cu_num=256）。
- sp16/sp32 变体 LDS 分别为 139392/143616（BINFO 中 kid 6/8），仍为 1 WG/CU。

---

## 7. README / KNOWN_ISSUES / 性能数据

- gen_co/README.md 与 KNOWN_ISSUES.md 只覆盖 gfx1250 a16w16 `.co` 家族（README:1-10；KNOWN_ISSUES:1-12），**没有** gfx950 pin_agpr 的条目。README 中有用的通用规则：`.co` 用 `-D__HIPCC_RTC__` 编译，单个 by-value kernarg，无隐藏参数（README:315-319）——gfx950 同理（BUILD:383；ELF `.args` 只有 1 个 96B by_value）。
- csrc/opus_gemm/README.md:92 表格写 gfx950 `a8w8_blockscale_bpreshuffle` 为 “empty”，与 COMMON:3195-3211 注册 CO kid 的现状不符（文档过时）。
- 分派限制：CO kid 带 `supported_shapes` 白名单时，`a8w8_mxscale_bpreshuffle_supports_shape` 只接受名单内形状（COMMON:3221-3225）。kid 14 名单只含 N∈{6144,7168}、K∈{7168,16384}（JSON:296 起），**不含 4096×4096×8192**；但内核本身对任意 m%256、n%256、k%128 有效（GEN:3214）。默认 max_k=16384（COMMON:3208）。
- 实测数据（CSV，gfx950, cu_num=256）：
  - kid 14：8192×7168×7168 → 328.2 µs，2564.8 TFLOP/s（CSV:734）；49152×7168×16384 → 3908.9 µs，2953.5 TFLOP/s（CSV:819，全表 pin_agpr 最高）。
  - kid 10（sp64 u2 gm0 wide1）：4096×7168×2048 → 61.5 µs，1956.5 TFLOP/s（CSV:507）。
  - 4096×7168×7168 由 flydsl 8-wave 胜出：174.2 µs，2416.5 TFLOP/s（CSV:732）。
  - **4096×4096×8192 在 gfx950 上无数据**（仓库内仅有 gfx1250 flydsl 行，不可比）。
- 正确性注意：x_scale 列主序字节要求（§1.3）；`stride_a==stride_b`（§3.2）；`stride_sfa_batch` 被当作 SFA buffer 长度（§1.3）。

---

## 8. Host launch ABI（独立 HIP 程序）

**代码对象**：`csrc/opus_gemm/gen_co/gfx950/opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_u2_gm16_wide1_vf1.co`（bare ELF，code object v5 / `amdgpu9.50-amd-amdhsa-unknown-gfx950`，ELF；LAUNCH:41-42 只检查 ELF 魔数）。

**kernel 符号**（注意比文件名多 `_sn0_sl0_sched0`，BINFO:270；ELF `.name`）：
```
opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_u2_gm16_wide1_sn0_sl0_sched0_vf1
```

**kernarg**：单个 96 字节 by-value 结构，无隐藏参数（TR:5-26；BUILD:402-403；ELF `.kernarg_segment_size=96`, align 8）：

| off | 类型 | 字段 | 4096²×8192 取值 |
|---|---|---|---|
| 0 | ptr | ptr_a | A fp8 [M][K] |
| 8 | ptr | ptr_b | shuffle 后的 B |
| 16 | ptr | ptr_c | C bf16 [M][N] |
| 24 | i32 | m | 4096 |
| 28 | i32 | n | 4096 |
| 32 | i32 | k | 8192 |
| 36 | i32 | batch | 1 |
| 40 | i32 | stride_a | k = 8192 |
| 44 | i32 | stride_b | k = 8192（必须 == stride_a） |
| 48 | i32 | stride_c | n = 4096 |
| 52 | i32 | stride_a_batch | m*k = 33554432 |
| 56 | i32 | stride_b_batch | n*k = 33554432 |
| 60 | i32 | stride_c_batch | m*n = 16777216 |
| 64 | ptr | ptr_sfa | x_scale，内存 [K/128][M] u8 |
| 72 | ptr | ptr_sfb | w_scale，内存 [N/128][K/128] u8 |
| 80 | i32 | stride_sfa | m = 4096 |
| 84 | i32 | stride_sfb | k/128 = 64 |
| 88 | i32 | stride_sfa_batch | m*(k/128) = 262144（也是 SFA buffer 长度） |
| 92 | i32 | stride_sfb_batch | (n/128)*(k/128) = 2048 |

（取值规则：GEN:3191-3198。）

**launch**：grid = (n/256, m/256, 1) = (16, 16, 1)，block = (256, 1, 1)，dynamic shared = 0（LDS 152064 为静态，GEN:3224-3226；LAUNCH:51-59）。参数通过 `HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE` 传入（HIPC:358-362, 410-420）。无 workspace。
前置条件：m%256==0、n%256==0、k%128==0；m*k、n*k ≤ INT32_MAX（GEN:3214, 3228-3240）。kernel descriptor：tgid x/y/z 启用、仅 workitem_id x、kernarg 段指针为唯一 user SGPR、fp32 denorm flush（ELF）。

**数据准备**：
- A：fp8 e4m3 行主序。
- B：`Bp[(n/16)*16*K + (k/16)*256 + (n%16)*16 + (k%16)] = B[n][k]`。
- x_scale：`sfa[kg*M + m] = e8m0(A 行 m 第 kg 个 128 块)`。
- w_scale：`sfb[(n/128)*(K/128) + kg]`。
- 参考：`C[m][n] = bf16( Σ_kg 2^(sfa-127) * 2^(sfb-127) * Σ_{k∈kg} A[m][k]*B[n][k] )`。

```cpp
struct __attribute__((packed, aligned(8))) Kargs {
  const void* ptr_a; const void* ptr_b; void* ptr_c;
  int m, n, k, batch, stride_a, stride_b, stride_c,
      stride_a_batch, stride_b_batch, stride_c_batch;
  const void* ptr_sfa; const void* ptr_sfb;
  int stride_sfa, stride_sfb, stride_sfa_batch, stride_sfb_batch;
};
static_assert(sizeof(Kargs) == 96);
hipModule_t mod; hipFunction_t fn;
hipModuleLoad(&mod, ".../opus_gemm_gfx950_4wave_pin_agpr_co_..._pm0_sp64_u2_gm16_wide1_vf1.co");
hipModuleGetFunction(&fn, mod, "opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_u2_gm16_wide1_sn0_sl0_sched0_vf1");
Kargs a{dA, dBp, dC, M, N, K, 1, K, K, N, M*K, N*K, M*N, dSfa, dSfb, M, K/128, M*(K/128), (N/128)*(K/128)};
size_t sz = sizeof(a);
void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
hipModuleLaunchKernel(fn, N/256, M/256, 1, 256, 1, 1, 0, stream, nullptr, cfg);
```

---

## 9. 我们的编译器 / DSL 需要的功能清单

| # | 功能 | 必要性 | 依据 |
|---|---|---|---|
| 1 | AGPR 物理寄存器分配 + 显式固定（按 a[N] 基址固定 A/B 片段与部分 C），并输出正确 accum_offset / agpr_count | must | PIPE:14-20, 41-74；BINFO:264-265 |
| 2 | ds_read 直接以 AGPR 为目的（`ds_read_b128 a[n:n+3]`） | must | BUILD:311-322 |
| 3 | `v_mfma_scale_f32_16x16x128_f8f6f4`：srcA/srcB 各 8 dword（可为 AGPR），srcC/vdst 4 dword（VGPR-form 与 AGPR-form 都要支持，同一 kernel 混用），fmt cbsz/blgp=0(E4M3)，scale_src0/1 为 VGPR，op_sel 选择 scale 字节（A scale byte=e∈0..3，B scale byte 0），即 VOP3PX2 编码（`v_mfma_ld_scale_b32` 前缀 + MFMA） | must | OPUS:3118-3120, 3189-3202；PIPE:90 |
| 4 | swap_ab 语义（硬件 A←B 片段，scale 也互换），输出“转置” C 布局 | must | OPUS:3687-3714 |
| 5 | `v_accvgpr_read_b32`/`v_accvgpr_write_b32`（epilogue 读 AGPR 中的 C01、暂存值写回 AGPR） | must | PIPE:999-1000, 1092, 1301-1311（指令由编译器生成，推断） |
| 6 | `buffer_load_dwordx4 ... lds`（LDS-direct，M0=wave LDS 基址，imm offset 同时平移源/目的，soffset 用 SGPR） | must | OPUS:1870-1897；PIPE:604-629 |
| 7 | buffer 资源描述符（base/num_records/flags 0x00020000），`buffer_load_dwordx4`、`buffer_load_ubyte`、`buffer_store_dwordx4`（cache 位 aux=2/NT） | must | OPUS:1747-1760, 1860-1866, 1917-1932；PIPE:1249 |
| 8 | `ds_read_b128`、`ds_read_b64`、`ds_read_b32`、`ds_write_b32`、`ds_write_b64`，支持 16 位 imm offset 折叠 | must | PIPE:770-773, 801, 707, 1261-1264, 1243 |
| 9 | `s_waitcnt vmcnt(0)` / `lgkmcnt(0)` 分开发射；`s_barrier` | must | PIPE:804-808；OPUS:1642, 3069-3072 |
| 10 | 指令级精确顺序控制：等价于 `sched_barrier(0)`/`sched_group_barrier(MFMA/DSread/LDS-DMA, n, id)` 的手排或硬约束；关闭 post-RA 重排 | must（自研编译器可直接按 DSL 顺序发射） | PIPE:803-1001；JSON:282 |
| 11 | `s_setprio` | must | PIPE:786, 1036-1040, 1205 |
| 12 | `v_mov_b32_dpp quad_perm(0xb1/0x4e)` + `v_perm_b32`（SFA 4×4 字节转置） | must（或改为其它 scale 布局） | PIPE:676-689；OPUS:1687-1688 |
| 13 | `v_readfirstlane_b32`（wave_id、prefetch soffset 标量化） | must | PIPE:426, 625, 796 |
| 14 | `v_cvt_pk_bf16_f32`（RNE） | must | OPUS:1218-1220；PIPE:1261 |
| 15 | 512 统一寄存器预算：ArchVGPR≈244 + AGPR 212，SGPR≈104，1 wave/SIMD；静态 LDS 152064 | must | BINFO:264-272；ELF |
| 16 | 96B by-value kernarg，无隐藏参数；tgid x/y/z | must | TR:26；BUILD:402-403；ELF |
| 17 | fp32 denorm flush 模式位 | nice | JSON:282；ELF |
| 18 | `sched_group_barrier` 风格的 MFMA/load 交错模板（按相位 syncid 0..3） | nice | PIPE:813-997 |
| 19 | 寄存器片段“最后使用后原地重载”的活性分析/检查（防 WAR） | nice（强烈建议做静态检查） | PIPE:897-1000 |
| 20 | `v_permlane16_swap`（direct 输出路径） | nice | PIPE:1215-1216 |
| 21 | kernarg preload | nice（本 .co 实际未使用） | ELF |
| 22 | 空 inline asm 屏障等价物（固定值在 VGPR/SGPR） | nice | PIPE:596, 797 |
| 23 | 编译期常量 M/N（sn 变体）、grouped-M 光栅化、wide 64 位输出基址 | nice | TR:183-188；PIPE:405-418, 453-462 |

未能确定（需在有 llvm-objdump 的机器上反汇编 .co 才能确认）：实际 MFMA 是否全部 VGPR-form（C01/C11-tail 除外）、epilogue `v_accvgpr_read` 数量、C11-tail 与 `a[128:135]` 暂存的冲突处理、sched mask 0x800 的确切语义、每迭代真实指令序列与 s_waitcnt 位置。

---

## 10. 实测基线与反汇编事实（2026-11-03，mi355-gpu-53 GPU 6，`tools/gemm_ref.sh`）

**正确性**：kid 14 在 4096×4096×8192 上对 64 行 × 4096 列做双精度 CPU 参考比对：bad=0，最大相对误差 3.9e-3（bf16 输出量化），说明 §8 的 ABI / B 预 shuffle / scale 布局推导正确（白名单 `supported_shapes` 只限制 aiter 分派，内核本身支持此形状）。

**性能（warm 协议：300 次预热，7×100 次计时取中位数）**：kid 14 = **3303 TFLOP/s**（0.0832 ms；三次独立运行 3303.3 / 3302.6 / 3306.5）。全部 34 个 pin_agpr 变体在同一形状上 3198–3305 TFLOP/s（`sn*` 定形变体不适用此形状）；最快 `pm0_sp64_u2_gm0_wide1_vf1` 3304.6，与 kid 14 持平。**移植基线 = 3303 TFLOP/s**。

**反汇编（ROCm llvm-objdump，`.co` 共 3425 条指令）**：
- MFMA：静态 320 条 `v_mfma_scale_f32_16x16x128_f8f6f4`；**225 条 VGPR 形式（dst/srcC = v）、95 条 AGPR 形式（dst/srcC = a）**；srcA/srcB 绝大多数为 AGPR（`dst=v srcA=a srcB=a` 219 条，`dst=a srcA=a srcB=a` 86 条）——印证 §4：操作数 pin 在 AGPR、累加器大部分在 VGPR、C01 + C11 尾在 AGPR。
- AGPR 搬运：`v_accvgpr_write_b32` 173、`v_accvgpr_read_b32` 84、`v_accvgpr_mov_b32` 83（主要在 prologue 初始化 / epilogue 读 C01 / 暂存交换处；主循环内需逐段核对）。
- 其它：`ds_read_b128` 192、`buffer_load_dwordx4` 93（含 LDS 直达形式）、`v_mov_b32_dpp` 104 + `v_perm_b32` 104（SFA 4×4 字节转置）、`v_cvt_pk_bf16_f32` 128、`ds_write_b64` 32 + `buffer_store_dwordx4` 32（epilogue）、`s_nop` 423（MFMA 相关等待态）、`s_barrier` 16、`s_waitcnt` 68。
