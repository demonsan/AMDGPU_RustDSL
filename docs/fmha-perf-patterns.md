# FMHA 性能承重模式清单（tile 抽象库的 parity checklist）

> 用途：新的高层 tile 抽象库（tensor view / LDS ring / copy-pipeline / 寄存器 fragment + tile mma / reduction /
> online_softmax / epilogue / combine）要用 ~100–200 行重写出厂的 4 个 kernel，并保持 **性能在出厂 ±3% 以内**
> （与 `tools/port_gate.sh` 的 `PERF_TOL=0.97` 一致）。本文列出库**必须**复现的每一个模式：指令级形状、代码位置、
> 实测影响、分类、库需要暴露/保证的抽象。
>
> 缩写：`K` = [`crates/amdgpu-dsl/src/kernels.rs`](../crates/amdgpu-dsl/src/kernels.rs)（22.5k 行），
> `C` = [`crates/amdgpu-compiler/src/lib.rs`](../crates/amdgpu-compiler/src/lib.rs)，
> `H` = [`crates/amdgpu-compiler/src/hazards.rs`](../crates/amdgpu-compiler/src/hazards.rs)，
> `S.rN` = [`docs/status.md`](status.md) 第 N 轮，`CS` = [`docs/current-status.md`](current-status.md)，
> `VS` = [`docs/vendor-spec.md`](vendor-spec.md)，`KR` = [`docs/kvsplit-results.md`](kvsplit-results.md)。
> 行号均在分支 `fmha-port`（HEAD `071f9c5`）上核对过。性能数字默认是 B1H8S2048、own 后端、`_vs128` 强制 64 blocks（per-CU 视角），
> 除非另注。不同轮次的绝对值基线不同（299 → 453 TFLOP/s），**看相对变化**。
>
> 分类：**承重** = 去掉/改形就掉 ≥1% 或有量化证据表明是关键路径的一部分；**附带** = 历史遗留或测得中性，库可以不复现
> （但不能引入负向）；**正确性必需** = 不做就会错（数值、越界、竞争、hazard），与性能无关也必须保留；
> **负向承重** = 必须*不*做（做了就慢）。

---

## 0. 出厂 kernel 与入口

| kernel | 入口 | 角色 builder | block | 说明 |
|---|---|---|---|---|
| `fmha_fwd_hd128_mxfp8_vs128` | K:20877-20882 → `fmha_t_build` K:21004-21641 | `emit_fmha_vs128_role` K:9602-11292 | 512 线程 / 256 行 | 铺满网格 + per-CU 快路径 |
| `fmha_fwd_hd128_mxfp8_t128` | K:19595-19600 → `fmha_t_build` | `emit_fmha_t128_role` K:16117-17057（仅 role A 一份，K:21545-21562） | 256 线程 / 128 行 | 未铺满网格（`2·ceil(S/256)·H·B ≤ 256`） |
| `fmha_fwd_hd128_mxfp8_kvsplit` | K:19633-19638 → `fmha_t_build`（`is_kvs`） | **同一个** `emit_fmha_vs128_role`（`vs_role_fn` K:20671-20673） | 512 / 256 | KV-split 第 1 段，fp16 Ohat + lse |
| `fmha_fwd_hd128_mxfp8_kvsplit_combine8h` | K:20707-20788 | 无（独立 builder） | 256 线程 | KV-split 第 2 段，每线程 8 元素 |

`_auto`（host 侧 `tests/gpu/ref_host.cpp`）在三者间分派；验收脚本 `tools/port_gate.sh`（BUILD/VERIFY/ACC16/ACC64/DET/STRESS/PERF）。

---

## 1. 总表

| # | 模式 | 位置 | 实测影响 | 分类 | 库抽象/参数 |
|---|---|---|---|---|---|
| P1 | 两组 wave（A=wave0-3, B=wave4-7）互补相位 ping-pong | K:21291-21334（split0）、K:10075-10127（B 多一个 prologue barrier） | lock-step（split1/2）**-15%**（313.5→266.7/265.4，S.r51） | 承重 | `WaveGroups{split: Contiguous4}` + 「B 组滞后一个 phase」的 prologue 生成 |
| P2 | 每 128 key 恰好 2 个 `s_barrier`（V→M、M→V），barrier 前 `wait_lds(0)` | K:10727-10732、K:10829-10832 | `nobar` -3.5%（S.r56）/ -7%（420.9 vs 452.7，S.r72）；`midbar`(3 个) **-9.7%** | 承重 | pipeline 的 phase/barrier 节奏必须可精确指定为 2/tile，不得自动加 barrier |
| P3 | `s_setprio`：**默认不发** | K:9611-9612、K:10712-10714、K:10820-10822 | 去掉 **+2.8%**（300.0→308.4，S.r50）；vsx 上再开 -0.9%（S.r53） | 负向承重 | 默认不发 setprio；可选开关 |
| P4 | LDS 布局 102400 B：2-slot K/V ring（每 slot 32 KB = K16K+V16K）、KD ring、ones、Q1/Q0 | `fmha_vs_tiles` K:20910-20931、`VS128_LDS_BYTES` K:19589 | 3→2 slot 在 `_vs` 上等速（S.r48）；决定占用率 1 block/CU | 承重（布局）/ 正确性（spill base） | `LdsRing{slots:2, slot_bytes:32768, sub=[K@0,V@16384]}`、`KdRing{512 B/slot}`、显式字节偏移、`spill_base=102400` |
| P5 | DMA staging：`global_load_lds_dwordx4` ×4/wave/tile（每条 1 KB），M0=wave 4 KB 顶 +3072，负 imm，全局地址正补偿 | K:9680-9725、K:9807-9833 | `nodma` +6.5..+7.6%（S.r53/56/61）= 发射侧成本；`dmah` -2%/-0.9%，`dmin` -1.0%，`mubuf` -1.3..-2.1% | 承重（形状）+ 正确性（正补偿） | `copy_g2l_async(tile, ring.slot(t+1))`：每 wave 32 行、4×1 KB、chunk 交换在全局侧 |
| P6 | DMA 时序：A 在 V(t) 开头发 K(t+1)+KD(t+1)，B 发 V(t+1)；V 相位末 `wait_vm(0)`，随后 barrier 发布 | K:10682、K:10699-10701、K:20910-20913 | `dsplit`（推迟 wait）**-6..-11%**（S.r61）；`novmwait` 无增益（S.r53） | 承重 | pipeline stage 描述：issue@V(t)start、wait@V(t)end、publish@barrier1、reuse-guard@barrier2 |
| P7 | KD（K scale）走 LDS-DMA `global_load_lds_dword`（kddma，默认） | K:9884-9907 | **+2.7%**（376.6→387，S.r61） | 承重 | `copy_g2l_async` 支持 dword 粒度、wave 2/3 复写 0/1 |
| P8 | XOR swizzle `s(r)`：128 B 行，chunk c 存于 c^s(r) | K:9684-9701、K:21390-21443 | bank conflict -98%（_t 上，速度不变，S.r45）；_vs128 上未单测 | 承重（布局前提：DMA 无法 padding） | `Swizzle::Xor3(s)` 作为 LDS tile 属性，读写/DMA 三方共享 |
| P9 | Q：每 64 B 组 2×`global_load_dwordx4`（字 0-3、8-11），共 4 条 | K:21181-21210 | **+6.0%**（389.6→412.6-414.1，S.r65） | 承重 | `load_frag(q, vec=16B)`，置换 K 序 |
| P10 | Q fragment 常驻寄存器（b0+b1，16 VGPR），LDS Q1/Q0 只在 prologue 写 | K:9621、K:10038-10046 | **+2.0%**（359.5→366.7，S.r56）；`noqreg` -4.5%（S.r66） | 承重 | `RegFragment` 跨循环常驻 |
| P11 | K fragment：每 S tile 2×(`ds_read_b128`×2) | K:10255-10268、K:21412-21421 | 去掉（`nokread`）反而 -7%（调度变化，S.r56）——不可删，但不是暴露项 | 承重（形状） | `lds_frag_read(K, b128x2)` |
| P12 | V^T fragment：每 fragment 4×`ds_read_b64_tr_b8`，每 tile 8 个 fragment（32 条） | K:10610-10627、K:21336-21345、K:21422-21433 | `pvpipe`/`vpre` 中性~负；`vpre2` **-5.5%**（S.r73） | 承重（形状） | `lds_frag_read_tr_b8(V)`；不要自动预取 |
| P13 | QK：per S tile = 1×`v_mfma_f32_32x32x8_f16`（bias→S）+ 2×`v_mfma_scale_f32_32x32x64_f8f6f4`（in-place） | K:10255-10311 | 去 bias MFMA 的所有方式 **-0.8..-15%**（biasonce/biasacc/biasc/biasv，S.r53/66/73/75） | 承重 | `tile_mma_scaled(acc_init=BiasMfma(ones, bias_word))`，保持 4 个 f16 MFMA 在 M 窗口开头 |
| P14 | scale 操作数：K 用 `kd>>g8`/`kd>>(g8+16)`，Q 用 `(qd>>g8)-9`（S/512 单位）；PV 用 (127, 123)=V×1、P×2^-4 | K:10068-10069、K:10283-10290、K:10667、K:21211-21225 | 数值约定（rollback 阈值、exp2(512δ) 依赖它） | 正确性必需 | scale 作为 fragment 元数据；`scale_bias=-9` 参数 |
| P15 | 行 max：32×`v_max3_f32` + `v_permlane32_swap` + `v_max_f32`（lane l↔l^32） | K:10368-10377；lowering `lower.rs:2494-2510` | `pmdead`（少 1 mov）+0.1%（S.r67） | 承重 | `row_reduce_max(frag, partner=Permlane32)` |
| P16 | 精确参考重基准：每 128 key 一次判定，阈值 120/512，two-word f16 bias，分支内 32 `v_pk_add_f32`（S）+32 `v_pk_mul_f32`（O）+ l 缩放 | K:10378-10499 | vsx 配方 **+10%**（316→352，S.r53）；rollback 全删上界 +16%（S.r51，错误数值） | 承重 + 正确性 | `online_softmax(policy=RebaseExact{thr:120/512, bias:TwoWordF16})`，wave 级分支、未回滚 lane δ=0 |
| P17 | P 转换：`v_cvt_pknorm_u16_f32`×2 + `v_cvt_scalef32_pk_fp8_bf16` lo/hi（每 4 值 4 条） | K:10500-10532；lowering `lower.rs:2477-2493` | svcorr 全开 -9.7%（S.r51） | 承重 | `convert_p_fp8(Schraudolph)` |
| P18 | LUT 校正（svcorr）只在短序列副本：总 tile 数 < 9 选带 LUT 的第二份完整代码 | K:10517-10526、K:21539-21598 | 双副本运行时零成本（`onecopy` 358.0 vs 359.1，S.r55） | 正确性必需（精度 parity） | `CodeVariants{select_once_at_entry(nkv_total<9)}` |
| P19 | 行和：4×`v_smfmac_f32_16x16x128_fp8_fp8` in-place 进 Frag4 循环变量（dword2=16·l） | K:10047-10067、K:10554-10567 | vs ones-MFMA **+2.2%**（351.6→359.6，S.r54）；vs 标量 `smfrs` +4.2% | 承重 | `row_sum(P, impl=Smfmac4InPlace)` |
| P20 | 尾 tile 剥离：主循环无 mask；最后一 tile 按 `is_tail` 分 masked/unmasked 两份 | K:10834-10857 | `peelone`（一份 masked）**-3%**（291 vs 299.3，S.r49） | 承重 | `pipeline.peel_last(masked_and_unmasked)` |
| P21 | bf16 epilogue：16×`global_store_dwordx2`/lane，无 LDS | K:11259-11291（_t128 K:17025-17056） | **+9%**（413→449-454，S.r65）；_t128 +5.4% | 承重 | `epilogue_store(bf16, vec=8B, direct)` |
| P22 | KV-split epilogue：fp16 Ohat 经 per-wave 4 KB LDS 转置 → 8×`global_store_dwordx4`/lane；lse = log2 l − 512 b | K:10887-11043 | fp16 vs fp32 +10%（556→613）；16 B vs 4 B 存储 7.4 vs 17-22 us（KR）；`epdirect` **-11%**（S.r69） | 承重 | `epilogue_store(fp16, via=LdsTranspose{4KB/wave, swz=row&7}, vec=16B)` |
| P23 | combine8h：每线程 8 元素，1×16 B 读/ split，在线合并，1×16 B 写 | K:20707-20788 | combine 5.0→3.9 us，总 +1%（S.r65） | 承重（小） | `combine(elems_per_thread=8)` |
| P24 | KV-split key range（tile 对齐）、split 解码（splits∈{2,4,8}） | K:21099-21119、K:21249-21289 | — | 正确性必需 | `KeyRange::split(tile=128, splits=pow2)` |
| P25 | GQA：`kv_head = head >> log2(gqa)`（select 链） | K:21227-21239 | — | 正确性必需 | `HeadMap::Gqa(pow2 ≤16)` |
| P26 | 行/键钳位：Q 行 select 到 S-1，DMA/KD 行钳到 kv_last，epilogue `wave_live` | K:21163-21173、K:9811-9819、K:10918、K:11266 | `dmin`（一条 min 代替 cmp+select）-1.0% | 正确性必需 | tensor view 的 clamp 语义（select，不用 min） |
| P27 | 编译器预设：lowering `dce,cmpfold,scalarize,retarget2,imm,sgpr,zeroinit`；完整重排 list scheduler；_t128 加 `addrfold` + xbar + max_vgprs 248 | C:71-127 | full vs `ld` +4%（299.3 vs 287.9）；retarget2 +1.8%；`imm` 299.3 vs 281.0；_t128 addrfold 416→494、xbar/248 489→505 | 承重 | 预设按「kernel 类别」而非名字前缀选择（见 §5.1） |
| P28 | VGPR ≤ 252、2 waves/SIMD、1 block/CU；调度器 fallback 即掉速 | S.r71-72、C:128-169 | fallback：t128 4.5k→3.57k cycles/tile；Block M lookahead +8 VGPR -3.1% | 承重 | 库不得增加常驻值；循环变量单写（见 P29） |
| P29 | 循环变量 SSA 纪律：bias word 每迭代读一次写一次；_t128 在窗口**末尾**写 | K:10378-10380、K:16975-16976 | _t128 提前写：410 vs 494（fallback，S.r59） | 承重 + 正确性 | `LoopVar` 写位置可控 |
| P30 | _t128：单组 4 wave、合并窗口、1 barrier/tile、3-slot ring、QK(t+1,s) 紧跟 P(s) 转换 | K:16333-16342、K:16516-16535、K:16918-16985 | 未铺满网格 1.23-1.5x（S.r59） | 承重 | 第二种 pipeline schedule（MergedWindow, ring=3） |
| P31 | 硬件 hazard：VMEM 宽存储数据 WAR、物理 WAW、MFMA SrcC WAR | H:375-406、H:414-602 | `_kvsplit` +4 `s_nop`（-0.3%）；不修则 ~5% 启动损坏 | 正确性必需 | 库生成的代码必须经 `pad_physical_war`；不得绕过编译器 |
| P32 | tile 协议注解（produce/consume/version）+ `gpu-verify --strict` 0 错误 | K:9908-9947、K:20910-20931 | 不发指令 | 正确性必需（验收门 VERIFY） | ring API 自动发注解 |
| P33 | prologue：Q/scale 一批发射 → tile0 DMA → `wait_vm(0)`+`wait_lds(0)` → P0 barrier → QK(0) | K:9948-10127 | prologue ≈3.1 us/启动（S.r61）；Q 向量化 +6% | 承重 | `pipeline.prologue()` 生成，含 A/B 不对称 barrier |

---

## 2. 逐项说明

### P1 两组互补 ping-pong（承重）
- **形状**：8 wave 按 `wave < 4` 分组（`split0`，K:21333）。每 wave 自己算 32 个 query 行（`row0 = qtile·256 + wave·32`，K:21159-21161）的完整 QK→softmax→PV；两组**只在 DMA 分工和相位**上不同：
  A 组 = `P0, QK(0), P1, loop[V, bar, M, bar]`；B 组 = `P0, P1, QK(0), P2, loop[...]`（K:10075-10077 的 `!role_a` 多一个 barrier），
  于是同一 SIMD 上的 wave w 和 w+4 一个在 V（VALU）相位、一个在 M（MFMA/LDS）相位（S.r51 addendum）。
- **证据**：`split1`/`split2` 把同 SIMD 两 wave 放进同组（lock-step）：266.7/265.4 vs 313.5（**-15%**）；硬件 co-issue 探针 MFMA+VALU ≈ max+48 cycles（S.r51）。
  g3（同码无偏移、1 barrier）-6.5%，g3o（偏移 barrier）-9%（S.r57/58）。
- **库必须**：(a) 支持「同一循环体、B 组滞后一个 phase」的 prologue；(b) group-local thread id `rt` 必须是组内 0..255 的置换（K:21292，DMA/KD 地址依赖）；
  (c) 不得在循环中跨回边携带 P fragment（S.r47 bug(2)：A/B 携带 P 曾导致非确定）。
- **组内 barrier 计数相等**：A = P0+P1+2·n+2（剥离尾 2 个），B = P0+P1+P2+2·n+1（尾部省最后一个，`last_barrier = role_a`，K:10849-10855）。gpu-verify 证明 18 个（CS §2.1）。

### P2 barrier 节奏（承重）
- V 相位末：`stage_end`（`wait_vm(0)`）→ `wait_lds(0)` → `s_barrier`（K:10699-10732）；M 相位末：`wait_lds(0)` → `s_barrier`（K:10829-10832）。
- barrier1 发布 K(t+1)/V(t+1)/KD(t+1)；barrier2 保证下一轮 DMA（复用 tile t 的 slot）在 tile t 的最后一次读之后（K:20912-20913）。
- 证据：`nobar` 各轮 -3.5%…-7%（S.r56/61/72）、`midbar` -9.7%（S.r56）；kernel-occ：「两个 barrier 是承重的重叠，不是可删的停顿」（S.r72）。
  注意早期 S.r50（带 setprio）`nobar1/nobar2` 曾 +3%，此后基线变化，**以 r56/61/72 为准**。
- **库必须**：barrier 数/位置由 schedule 显式给出；`wait_lds(0)` 在 barrier 前（循环内无 LDS 写，等待为 DS 读）。

### P3 `s_setprio`（负向承重）
- 代码保留 `set_prio(1)` 于 barrier1 前、`set_prio(0)` 于 M 相位末，但默认 `noprio`（`prio` token 才开）。
- 证据：去掉 +2.8%（S.r50）；vsx 后再开 -0.9%（S.r53）；_t128 同。VS §7.7 记为「waived by measurement」。厂商是用的（gen9 doc §4），**我们不用**。

### P4 LDS 布局与 2-slot ring（承重 / 正确性）
- `VS128_LDS_BYTES = 2·32768 + 2048 + 2048 + 16384 + 16384 = 102400`（K:19589）：
  slot k = [32768k, +32768)：K tile 在前 16 KB、V tile 在后 16 KB；KD ring [65536+512k, +512)（2 KB 块）；ones 67584（2 KB）；Q b1 69632（16 KB）；Q b0 86016（16 KB）。
  tile t 在 slot `t & 1`、version t（`ring` K:10198-10211：`slot_cur = idx·32768`，`slot_nxt = (idx^1)·32768`，`kds = idx·512`）。
- `VS_SPILL_BASE = 102400`（K:19592）：编译器 LDS spill 区从此开始（2 KB/dword，至 163840 上限共 30 dword）——**库若改布局必须同步此常量**（S.r48 addendum）。
- ones 块与 Q1/Q0 在默认配方下**只写不读**（smfv、q0reg/q1reg 默认开）→ **附带**；但 69632..102400 被 KV-split epilogue 当 per-wave 4 KB scratch（`lds_private("epi",69632,4096,8)`，K:20922），**承重**。
- 占用率：102400 B > 163840/2 → 1 block/CU（与厂商相同：84224 B 也放不下第 2 个，S.r71-72）。库不需要缩 LDS，但**不得超过 163840 − spill 需求**。

### P5 DMA staging 形状（承重 + 正确性）
- 每 wave（group-local `wave4`）负责 tile 的 32 行（4 KB）：4 条 `global_load_lds_dwordx4`，第 q 条覆盖行 `32w+8q .. +7`（1 KB 连续）；lane i 落 LDS 槽 i，
  取全局行 `8q + (i>>3)` 的 chunk `(i&7) ^ s(row)`（K:9680-9718）。即 **swizzle 在全局地址侧完成，LDS 写是线性的**。
- M0 = `lds_base + slot + w·4096 + 3072`，指令 imm = `-(1024·(3-q))`，全局地址补偿 `+1024·(3-q)`（K:9719-9725、K:9827-9831）。
  **正补偿是正确性要求**：32 位偏移零扩展，负补偿会回绕到 +4 GB（S.r47 bug(3)）。
- 行钳位：`row = (row_raw < kv_len) ? row_raw : kv_last`（K:9817-9818）。
- 每 wave 每 tile：A 组 K 4 + KD 1 条，B 组 V 4 条（S.r61 addendum）。`nodma` 的收益来自**发射侧**（地址 VALU + VMEM 发射 + readfirstlane/M0），不是延迟（S.r53）。
- 注：用户需求里提到的「1040-byte runs」是**厂商**与旧 `_t/_sv` dma 路径的布局（K:21364-21374，`j128` 公式）；`_vs128/_kvsplit/_t128` 实际使用的是 128 B 稠密 XOR swizzle 布局（`cx.kfx/trx`），1040 公式在 VS 模式下算出的 `kf_base/tr_base` 不被这三个 role 使用。

### P6 DMA 时序（承重）
- iter（K:10675-10833）：`stage_start(key_n2 = 128·min(it+1, nkv-1), slot_nxt)` 在 V 相位**最前**发射 → `valu()` → `stage_end`=`wait_vm(0)` → barrier1。
- 最后一 tile 后无多余 DMA：`nn = select(it+1 < nkv, it+1, nkv-1)`（K:10205-10209）——prologue 和尾部复取最后一 tile 而非越界。
- 证据：`dsplit`（DMA 提前一相位、推迟 wait）-6..-11%；`AMDC_PIN_VMWAIT` 无益（S.r61）。**库不要「自动加深预取」。**

### P7 KD via LDS-DMA（承重）
- `kd_dma`（K:9887-9907）：wave `wlo = wave4 & 1` 负责行 `64·wlo + lane`，`global_load_lds_dword` 到 `65536 + slot/64 + 256·wlo`；wave 2/3 重复写 0/1 的数据（无害重复）。
- 读：`kd_cur = 65536 + 4j`，S tile s 偏移 `+128s`，每 tile 4×`ds_read_b32`（K:10273-10279）。
- 证据：+2.7%（S.r61）。替代方案 `kdpre`/`kdv4`/`kdglob`/`kdearly` 均 -2..-8%（S.r51/53）。

### P8 XOR swizzle（承重：布局前提）
- `s(r)`：`p=r&1, u=(r>>1)&7`，`s2=u2^p, s1=u0, s0=u1^p`（K:9684-9701 / K:21395-21411，暴力搜索得到，对 b128 K 读、`ds_read_b64_tr_b8` V 读、b128 staging 写在 64-bank 和 32-bank 两种模型下零冲突）。
- K 读地址：`kfx[k] = j·128 + ((pblk ^ s(j)) ^ 2k)·16 + slot`，S tile s 再加 `4096·s`（K:21412-21421）。
- V 读地址：`trx[e] = key_part·128 + 8·(lane&1) + ((m_gl ^ s(key_part)) ^ 2e)·16 + 16384 + slot`，
  `key_part = 8·((lane&15)>>3) + 4·pblk + ((lane>>1)&3)`，`m_gl = (lane>>4)&1`（K:21336-21345、K:21422-21433）。
- 证据：_t 上冲突 524288→8704、速度不变（S.r45）；_vs128 计数器 bank conflict = 0（S.r56）。_vs128 上「无 swizzle」未测（DMA 线性写无法 padding，故必须保留）。

### P9 Q 16 B 载入（承重）
- `q_off = row·q_seqs + b·q_bs + h·q_hs + 16·pblk`；对 c∈{0,1}、half∈{0,1}：`load_b128(qp, 64c + 32·half)` → 每 lane 读 64 B 组中的字 0-3 和 8-11（与 K 读的 chunk 置换一致）（K:21195-21202）。
- Q scale 一个 dword：`sq0 = qd >> g8`，`sq1 = qd >> (g8+16)`；V descale 一个 dword（K:21211-21225）。
- 证据：+6.0%（16 个串行 `global_load_dword`+vmcnt(0) → 4 条 dwordx4，S.r65）。剩余：Q 延迟与 tile0 DMA 仍串行（S.r65 末，未做）。

### P10 Q 常驻寄存器（承重）
- `bq[0]/bq[1]`（各 Frag8）整个循环常驻（`q0_reg`/`q1_reg` 默认，K:9621、K:10038）。prologue 仍把 Q 写到 LDS Q0/Q1（K:10002-10019，默认不读，附带）。
- 证据：两者同开 +2.0%，单开反而 -1.2%（S.r56）；LDS 指令 68.8→52.8/wave-iter（=厂商 52.5）。

### P11/P12 Fragment 读（承重：形状）
- K：每 S tile `load_lds_b128x2(a0,a1)`、`load_lds_b128x2(a2,a3)` = 4×`ds_read_b128`，tile 共 16 条（K:10266-10267）。
- V^T：`load_v(hh,e)` = 4×`ds_read_b64_tr_b8` @ `trb[e] + 8192·hh + 2048·r`（r=0..3），8 个 fragment/tile = 32 条（K:10610-10627）。
- 调度器给出部分 `lgkmcnt(4)/(2)` 等待（S.r56）。手工流水 `pvpipe`、`qkfirst`、`kdearly`、`vpre`、`vpre2` 全部中性或负（S.r50-56/73）。**库不要强制 read-ahead 顺序；把顺序交给 list scheduler。**

### P13 QK MFMA 序列与 f16 bias MFMA（承重）
- 每 S tile s（4 个，各 32 key×32 query）：
  1. `v_mfma_f32_32x32x8_f16 S[s] = ones_f16 × bias_word`（`var_mfma_f16_z`，K:10256-10258）——把携带的参考值 b = 4(hi+lo) 写进累加器；
  2. `v_mfma_scale_f32_32x32x64_f8f6f4 S[s] += K8_0 × Q_b0`（scale sk0, sq0−9）；
  3. 同上 `+= K8_1 × Q_b1`（sk1, sq1−9）（K:10307-10310）。
- PV：8×`v_mfma_scale` `O[e] += V^T(hh,e) × P(hh)`，scale (127, 123)（K:10650-10669）。XDL 合计 16×64 + 4×32 + 4×16 = 1216/tile（厂商 1088）。
- **承重原因**：4 个 f16 bias MFMA 是 barrier 后唯一不等 LDS 读的矩阵工作，遮盖 ~190-206 cycles 的 K/V 读延迟（S.r66 closure）；去掉后节省的 XDL 周期以 barrier 汇合等待（WAIT_ANY +91..+118）形式出现（S.r73/75）。
- 证据：`biasonce` -0.8%（S.r53）/ -5%（S.r56）；`biasacc` -5.8%；`biasc` -3.5%；`biasv` -12%（S.r56）/ **-15.2%**（S.r66）；`qkzero` 诊断也无墙钟收益。

### P14 scale 约定（正确性必需）
- Q scale 减 9 → S′ = S/512（K:10068-10069）；所以阈值以 /512 表示，回滚倍率 `exp2(512·δ)`（K:10476-10477），lse 里 `512·b`（K:10896-10901）。
- P 的 MFMA scale 123 = 2^-4；V 的 127 = 1；V descale 在 epilogue `fin = rcp(l)·vd`（K:10879-10886）。
- `p4`（把 2^-4 移进转换）-1.2..-3.5%（S.r75、`unscaled-pv-design.md`），**不做**。

### P15 行 max（承重）
- 64 个分数（4 tile × 16）→ `max3` 链 32 条（K:10368-10376；masked 时先 `select(valid, v, -1e30)`）→ `partner_max_f32`，lowering = 2×`v_mov` + `v_permlane32_swap` + `v_max_f32`（`lower.rs:2494-2510`）。
  lane l 与 l^32 持同一 query 的不同 key 半（S^T = K·Q^T 布局）。
- 旧的 LDS 交换槽（`ex_own/ex_partner`，K:21150-21157）只给 `_t` 用，vs128 不用。

### P16 精确参考重基准 / rollback（承重 + 正确性）
- 判定：`grow = mt > 120/512`（`c_thr`，K:9645）；每 128 key **一次**（一个 tile 一次 `v_exp`）。
- two-word bias（`vs128_vsx()` 默认，K:10403-10429）：`b_old = 4(hi+lo)`，`b_new = b_old + 131/512 − mt`，量化成 hi = f16(b_new/4)、lo = f16(残差)，`δ = b_q − b_old`（仅 grow lane，否则 0）。新 bias word 是 SSA 值，循环变量只在此处写一次（K:10448-10449）。
- 分支体（`begin_if(grow)`，K:10450-10495）：`S[s] += (δ,δ)` 4×8 = 32 `v_pk_add_f32`；`corr = exp2(512δ)`；`O[e] *= corr` 4×8 = 32 `v_pk_mul_f32`；行和 dword2 ×corr。
- **正确性**：`begin_if` 是 wave 级「任一 lane」分支、无 exec mask → 不回滚的 lane 必须拿到 δ=0、corr=1（K:10455）。
- 证据：阈值 120（每单元对齐）去掉了长序列的 LUT 校正：+10%、mean err 6.95e-4（S.r53）；`rbtrim`（去冗余 `&0xffff`）+0.8%（S.r51）；rollback 体 ablation：S shift ~-0.5k、O rescale ~-0.5k cycles/tile（S.r51）。
  `cut1`（f32 携带 b）-0.4%、`svscale`（整数 shift）精度不过门（S.r47/51）——都不做。

### P17 P 转换链（承重）
- 每 4 个分数：`v_cvt_pknorm_u16_f32`×2 → `v_cvt_scalef32_pk_fp8_bf16`（lo）+ 同（hi, `op_sel:[0,0,1]`，读-改-写同一目的），scale = 1.0（`one_f`）（K:10528；`lower.rs:2477-2493`；`emit.rs:267-296`）。
  tile 64 值 → 16 个 P word（两半各 8 → 两个 Frag8）。masked 时输入 `select(valid, v, 0)`（K:10509-10512）。
- 这是 Schraudolph 式 exp2：S′ 已含 bias，pknorm 的 u16 被当作 bf16 位型。

### P18 LUT 副本（正确性必需：精度 parity）
- 带 LUT：每对 `u = pknorm(x0,x1)`；`sel = (u>>4) & 0x00070007`；`pm = v_perm_b32(lut_hi, lut_lo, sel)`（LUT 字节 [0,6,9,11,11,10,7,3]）；`u' = v_pk_sub_u16(u, pm)`；再 `cvt_p8x4_from_u16`（K:10517-10526）。每对 4 个 packed op、每 tile 128 VALU（S.r51，被证明不可再省）。
- 选择：入口处一次 `short = nkv_total < 9`（`cs` 默认 9）→ 两份**完整** kernel 代码（每份内再 A/B 两 role）（K:21574-21598）；KV-split 用**整条序列**的 tile 数（`ntiles_total`），不是本 split 的（K:21576-21580）。_t128 同规则（K:21545-21558）。
- 证据：全序列开 LUT -9.7%；全关则 SK=100/129 等短形状 parity FAIL（S.r53）；双副本 IFETCH/时间不变（S.r55）。

### P19 smfmac 行和（承重）
- 每 lane 常量 A：`a_rs = frag4(a23,a23,a01,a01)`，由 `lane&3 < 2`、`lane&4` 选 0 或 0x38383838（K:10052-10061）；idx01 = 0x44444444，idx23 = 0xeeeeeeee。
- 每 tile：`smfmac(la, a_rs, Pa, idx01)`, `(Pb, idx01)`, `(Pa, idx23)`, `(Pb, idx23)` 原地累加到 Frag4 循环变量（K:10561-10566），**在 PV MFMA 之前**发（`smford` 关）。dword2 = 16·l，epilogue ×0.0625（K:10866-10870）。
- 证据：vs ones-MFMA +2.2%（S.r54）；`smfrs`（每单元清零+提取+标量和）-4%、`smf2` -2.7%。厂商用 2 条（vs 我们 4 条，差 ~23 XDL cycles，S.r75）——「2-smfmac」是已知未实现方向，**不是库的 parity 要求**。

### P20 尾 tile 剥离（承重）
- 主循环跑 `nkv-1` 次全 unmasked iter；循环后 `is_tail = kv_len < 128·it + 128` 选 masked 或 unmasked 的一份 iter（K:10843-10857）。masked 版本对 max 用 -1e30、对 P 用 0。
- 证据：`peelone`（只一份 masked）-3%（S.r49）、+VGPR（256）。

### P21 bf16 epilogue（承重）
- `wave_live = row0 < S` 整 wave 跳过；`fin = rcp(l)·vd`；对 e∈0..3、q∈0..3：4 个 f32 ×fin → 2×`v_cvt_pk_bf16_f32` → 1×`global_store_dwordx2` @ `row·o_seqs + 64e + 16q + 8·pblk`（K:11259-11291）。每 lane 16 条。
  越界行已被钳到 S-1，重复写同值（P26）。**没有 LDS 转置**（LDS 转置只用于 KV-split 的 fp16 partial，P22）。
- 证据：`ost1`（2×dword）→ dwordx2 +9%（_vs128）、+5.4%（_t128）（S.r65）。

### P22 KV-split partial epilogue（承重）
- 前置：所有 wave 一个 barrier（K:10858-10864，`cx.kvs.is_some()` 时）——role 依赖的末 barrier 使先结束的 wave 可能覆写伙伴仍在用的 Q staging；2026-10-31 的 `1d9baea` 修到 vs128 role。
  注意 VS §7.11 记录「对称 barrier 曾使宽存储 WAR 损坏率变高」——那是在 hazard 修复**之前**；现状是 barrier + `pad_physical_war` 同时存在，库必须**完全照此**并通过 `ohat_repro.sh`。
- `lse = log2(l) − 512·b_final`，`b = 4(hi+lo)`（K:10887-10901）；partial 行 `prow = split·rows_total + (b·S + row)·H + h`（K:10902-10907）；每 lane 写一次 lse（两 half 同值）。
- fp16（`p16` 默认，`nop16` 回 fp32）：scratch = `69632 + wid·4096`（wid = wave4 + 4·roleB）；对 pr∈{0,1}：写 2 个 fragment（ec）×4 quarter，每次 2×`ds_write_b32` 到 `j·128 + ((4ec+q) ^ (j&7))·16 + 8·half`；`wait_lds(0)`；
  4×`ds_read_b128` @ 行 `8i + lane>>3`、chunk `(lane&7) ^ row` → 4×`global_store_dwordx4` @ `prow_i·256 + 16·(lane&7) + 128·pr`；`wait_lds(0)`（K:10945-11043）。每 lane 共 8 条 dwordx4，每行 128 B 合并。
- 证据：16 B lane 存储 7.4 us vs 4 B 散写 17-22 us；fp16 vs fp32 556→613 TFLOP/s；`epdirect`（无 LDS，dwordx2）616.8 vs 694.0（-11%）（KR、S.r69）。q3（e4m3 partial）精度不过门（S.r61）。

### P23 combine8h（承重，小）
- grid = `ceil(rows·128/8/256)`，256 线程；`t8` 钳到最后一个；`row = t8 >> 4`（K:20720-20729）。
- 循环 s∈[0,splits)：`load_f32 lse[s·rows+row]`、`load_b128 Ohat[(s·rows·16 + t8)·16]`；在线合并 `m=max, a=exp2(m_old−m), w=exp2(l−m), den=den·a+w, acc_k = acc_k·a + w·v_k`（8 个 f16→f32）；结束 `rcp(den)` → 4×`v_cvt_pk_bf16_f32` → 1×16 B 存储（K:20734-20786）。
- 编译：名字不匹配任何调度预设 → **只有 `zeroinit` lowering、无调度器**（C:91-107）。证据：5.0→3.9 us，总 +1%（S.r65）；16h 相同。

### P24 KV-split key range（正确性必需）
- grid z = `B·splits`，`lg = (s>>1) − (s>>3)`（splits∈{2,4,8}），`bi = z>>lg`，`sp = z & (splits−1)`（K:21099-21119）。
- `ntiles = ceil(SK/128)`，`tps = ceil(ntiles/splits)`，`tile_lo = sp·tps`，`tile_hi = min(ntiles, tile_lo+tps)`，`key_lo = 128·tile_lo`，`key_hi = min(128·tile_hi, SK)`，`kv_eff = key_hi − key_lo`；
  K/V/KD 的 base 偏移加 `key_lo·stride`，之后 kernel 内所有 key 下标相对 key_lo（K:21249-21289）。host guard：`(splits−1)·tps < ntiles`（每 split 非空，KR）。
- 库要求：`KeyRange` 视图必须**平移 base**（而不是在每个地址上加 key_lo），这样主循环代码与 `_vs128` 完全相同。

### P25 GQA（正确性必需）
- `gshift` = 4 段 select 链（gqa>1/2/4/8 → 1..4），`kvh = head >> gshift`（K:21227-21239）；host 合同 gqa 为 2 的幂 ≤16。

### P26 钳位（正确性必需）
- Q 行：`row = row_raw < S ? row_raw : S−1`（K:21163-21173），整条计算和存储对越界行重复最后一行（同值，无越界）。
- K/V DMA 行、KD 行：`< kv_len ? r : kv_last`（K:9817-9818、K:9895-9896）；epilogue 用 `wave_live` 跳过全越界 wave（K:11266）。
- 用 `v_min_i32`（`dmin`）代替 select 反而 -1.0%（S.r61）——库的 clamp 应生成 `cmp+cndmask`，不要「优化」成 min。

### P27 编译器预设（承重）
| kernel | lowering（C:74-90） | scheduler（C:103-127） |
|---|---|---|
| `_vs128`、`_kvsplit` | `dce,cmpfold,scalarize,retarget2,imm,sgpr,zeroinit` | 完整重排 list scheduler：`reorder=true`、`alu_across_barrier=false`、`max_vgprs=240`、无 `ldshoist/swp/taildup`；VGPR 失败时按 `(cut 16/32/48)` 降压重试 |
| `_t128` | 上面 + `addrfold` | 重排 + `alu_across_barrier=true` + `max_vgprs=248` |
| `_kvsplit_combine8h` | `zeroinit` | 无（不匹配预设名） |
| `_t`（对照） | `zeroinit` | `ld` 预设（不重排、xbar、ldshoist） |
- 所有 sched 结果之后都跑 `fix_value_hazards` + `regalloc` + `pad_physical_war`（C:153-161）。
- 证据：`_vs128` full vs `ld` 299.1 vs 287.9（S.r49）；`retarget2` +1.8%（S.r51）；无 `imm` 281.0（vgpr 256）；`sgpr`/`addrfold` 曾误编译 `_vs`（S.r47，已修）；`_t128` addrfold 416→494、xbar+248 489→505（S.r59、C:112-117，「heuristic knob，改动后重测」）。
- 手工顺序 / 预取全部输给调度器自己的结果（S.r50-52）。**库生成的 IR 必须给调度器同等自由度**（不要 pin，不要 safe-asm 窗口：Block C -2.5%，Block M 中性，S.r63）。

### P28 寄存器预算与占用（承重）
- `_vs128` vgpr 249-252、sgpr 48（S.r71-72）；`_kvsplit` vgpr ~231（KR）；`_t128` 252（S.r59）。2 waves/SIMD、1 block/CU（与厂商同）。
- 任何「+N 常驻 VGPR」的改动都要过调度器：`biasonce` vgpr 232→256 -0.8%、Block M lookahead +8 VGPR → fallback -3.1%、`_t128` fallback 时 4.5k vs 3.57k cycles/tile。
- **库必须**：不引入额外常驻值（描述符、预计算地址、常量 → `constres` 也被两阶段 RA 阻塞，VS §7.10.8）；可选地提供 per-kernel VGPR 报告与 fallback 告警（`sched: falling back` 行，C:169）。

### P29 循环变量纪律（承重 + 正确性）
- 循环变量在一个迭代内「读一次（迭代开头）、写一次、写后不读」（K:10378-10380；`Opnd::Var` 在 `set_var` 同迭代读到的是迭代初值，S.r47 bug(1)）。
- `_t128` 把 bias word 写在窗口末尾（K:16975-16976），提前写 → 410 vs 494（fallback，S.r59）。库的 `LoopVar` 需要可控的写点。

### P30 `_t128` 结构（承重）
- 4 wave（256 线程，1 wave/SIMD，**无 ping-pong 伙伴**），只发 role A 一份代码（K:21545-21562），A 组 DMA **同时**取 K 和 V（同一组钳位行共享，K:16287-16308）。
- 3-slot ring（LDS `3·32768 + 2048 + 2048 + 16384 + 16384 = 135168`，K:21014），prologue staged tile 0 和 1（K:16331-16342），iter 在开头发 tile t+2 的 DMA。
- **单窗口**（K:16918-16985）：DMA(t+2) → max+rollback(t)（`valu(..,0,0)` 只做判定）→ 对 s=0..3：转换 P(s) 紧接 `qk(t+1, s)` 写入刚空出的 S 寄存器 → 写 bias 变量 → PV(t)+smfmac → `wait_vm(0)`、`wait_lds(0)`、**1 个 barrier**。
- K scale 走 VGPR（`kd_row_load` + `ds_write_b32`，K:16233-16242、K:16314），**不是** kddma；`hoist`（每窗口地址一次）默认开（`nohoist` 关）。
- 无 tile 协议注解（`fmha_vs_tiles` 只给 `_vs128/_kvsplit/_vs5`，K:21064-21066）。
- 证据：B1H8S2048 367.8→494.1（1.34x），N≤128 时 1.23-1.5x；铺满时 -25%（S.r59）。

### P31 硬件 hazard（正确性必需）
- **VMEM 宽存储数据 WAR**：`global_store_dwordx4`/`buffer_store_dwordx4` 发射后 2 个等待态内 VALU 改写其数据 VGPR 会写坏未读数据（LLVM `has12DWordStoreHazard`）。`STORE_DATA_WAIT=3` slot（H:375-382，H:496-506、H:573-579）。
  出厂效果：只有 `_kvsplit` +4 `s_nop`（180 vs 176），其余 byte-identical；不修时 ~5% 启动（`epw` 下 97%）损坏 fp16 Ohat 首 dword（VS §7.11、S.r69）。回归：`tools/ohat_repro.sh` 300 次 + `FMHA_OPT=eplds,epw` 100 次干净。A/B：`AMDC_NO_PHYS_STORE=1`。
- **物理 WAW**：结果已死的 MFMA 的目的寄存器被 RA 分给新值，MFMA 19/21 个等待态后落笔覆盖（H:398-406、H:507-534，跨基本块传播 H:420-455）。现象：`biasacc,vpre` 6 次 6 个 md5（S.r66）。A/B：`AMDC_NO_PHYS_WAW=1`。出厂 kernel 上的单独代价未测。
- **MFMA SrcC WAR**：在途 MFMA 的累加器源（scaled 21 / 其他 12 等待态；SMFMAC 读 A/B/idx 多拍）不得被改写（H:548-572）。
- 其他已知正确性约束：SMFMAC 后 17 态 padding（S.r72）、VALU→MFMA 2 态、MFMA 结果 20 态（safe-asm 规则）。
- **库要求**：所有代码经编译器正常流水线（不得绕过 `pad_physical_war`）；若库引入新的宽存储/新的 MFMA 目的复用模式，必须跑 `phys_waw`/`hazard_*` 测试与 ohat 压力。

### P32 tile 协议与验证（正确性必需）
- `tile_produce/tile_consume/tile_end` 包住每个 DMA 与 DS 读（K:9908-9947、K:10097-10113、K:10264-10279、K:10621-10624、K:11025-11040），`fmha_vs_tiles` 声明分区/别名（K:20914-20931）。
  gpu-verify `--strict`：`_vs128` 58,880 次读、`_kvsplit` 240,640 次读，0 错误 0 警告（CS §2.1）；barrier 协议 `_vs128`/`_kvsplit` 18、`_t128` 8。
- 约束（`assume_trusted`）：kv_head、split 下标、tps/tile_lo/kv_eff、nkv、循环计数（K:21114-21115、K:21239、K:21263-21279、K:21382、K:10178）。库的 ring/KeyRange 抽象应自动产生这些注解。

### P33 prologue（承重）
- 顺序：kernarg → Q 4×dwordx4 + 2 scale dword 一批（部分 vmcnt）→ tile0 DMA（K、KD / V）→ ones/Q LDS 写 → `wait_vm(0)`、`wait_lds(0)`、P0 barrier（K:9948-10024）→ B 组 P1 → QK(0)（bias f16 MFMA + 2 scaled/tile）→ `wait_lds(0)` → barrier（A:P1、B:P2）（K:10075-10127）。
- prologue ≈ 3.1 us/启动（两次串行内存延迟），KV-split 固定成本的一半（S.r61 addendum）。重叠 Q 与 DMA 延迟最多省一个延迟，未做。

---

## 3. 各 kernel 精确差异

| 维度 | `_vs128` | `_t128` | `_kvsplit` | `combine8h` |
|---|---|---|---|---|
| 线程 / 行 / grid.x | 512 / 256 / `ceil(S/256)` | 256 / 128 / `ceil(S/128)` | 512 / 256 / `ceil(S/256)` | 256 / — / `ceil(rows·128/8/256)` |
| grid.z | B | B | B·splits | — |
| role | A+B（`begin_if(is_a)`） | 只 A | A+B | — |
| ring | 2-slot，K/V 分组 DMA | **3-slot**，A 同时取 K+V | 同 `_vs128` | — |
| LDS | 102400 | 135168 | 102400（`epreloc` +32768） | 0 |
| KD | kddma（LDS-DMA） | VGPR + `ds_write_b32` | kddma | — |
| barrier/tile | 2 | **1** | 2 | — |
| 窗口 | V（max/rollback/转换）+ M（PV+QK(t+1)） | 单窗口：判定 → 4×(转换 s + QK(t+1,s)) → PV | 同 `_vs128` | — |
| key 范围 | [0, SK) | [0, SK) | split 的 tile 对齐子区间，base 平移 | — |
| LUT 副本选择 | `nkv < 9` | `nkv < 9`（只 role A 两份） | **整序列** `ntiles_total < 9` | — |
| epilogue | bf16，16×dwordx2/lane，直接 | 同 | 全体 barrier → lse + fp16 Ohat，LDS 转置，8×dwordx4/lane | 1×16 B 读/split，1×16 B 写 |
| kernarg | 656 B（0x280 `abi_tail`） | 同 | 656 B + 0x270 Ohat、0x278 lse、0x280 splits、0x284 rows_total | 0x00 ohat、0x08 lse、0x10 out、0x18 rows、0x1C d、0x20 splits |
| lowering | 7 项 | 7 项 + `addrfold` | 7 项 | `zeroinit` |
| scheduler | 重排、max 240 | 重排、xbar、max 248 | 重排、max 240 | 无 |
| vgpr | 249-252 | 252 | ~231 | 小 |
| tile 注解 | 有 | 无 | 有 | — |
| hazard padding | 宽存储：无 | 无 | **+4 `s_nop`** | 无 |
| 分派（auto） | 铺满网格（N256 > 128） | `2·N256 ≤ 256` | guard：`N256≤128 && N256·s≤256 && (s−1)·tps<ntiles && T_kv<0.95·T_cur`，s∈{2,4} | 与 `_kvsplit` 配对 |

（N256 = `ceil(S/256)·H·B`；guard 见 KR 与 S.r59/60。）

---

## 4. 已量化关闭的负方向（库**不要**默认做）

| 方向（token / kernel） | 实测 | 来源 |
|---|---|---|
| `s_setprio` 1/0 每相位（`prio`） | -2.8%（r50）、-0.9%（vsx 后） | S.r50/53 |
| lock-step 组（`split1`/`split2`） | **-15%** | S.r51 |
| 第 3 个 barrier（`midbar`） | -9.7% | S.r56 |
| 去 barrier（`nobar`） | -3.5% … -7% | S.r56/61/72 |
| 转换后移到 M 相位（`convlate`） | -6.4% | S.r56 |
| DMA 推迟 wait（`dsplit`） | -6 … -11% | S.r61 |
| DMA 地址 hoist（`dmah`） | -2.0% / -0.9% | S.r51/61 |
| `v_min_i32` 钳位（`dmin`） | -1.0% | S.r61 |
| MUBUF K/V DMA（`mubuf` / `mubufd`） | -1.3..-2.1% / -0.5% | S.r71 |
| SADDR DMA / `pmdead` | +0.5% / +0.1%（<1% 门，opt-in） | S.r67 |
| K scale 变体（`kdpre`/`kdv4`/`kdglob`/`kdearly`/`nokddma`） | -2 … -8% / -2.7% | S.r51/53/61 |
| 手工流水/预取（`pvpipe`/`qkfirst`/`vpre`/`vpre2`/`qonce`） | -0.5 … -5.5%（`vpre` +0.4%、`qonce` +0.2% 噪声内） | S.r50-56/66/73 |
| 去 bias MFMA（`biasonce`/`biasacc`/`biasc`/`biasv`/`bacin`） | -0.8 … -15% | S.r53/56/66/73/75 |
| unscaled PV（`p4`） | -1.2 … -3.5% | S.r75 |
| 携带 f32 参考（`cut1`）、`mubufd,cut1` | -0.4% / -1.2..-1.7%（wave cycles） | S.r71-72 |
| Q 从 LDS 读（`noqreg`）、单个 Q 常驻 | -4.5% / -1.2% | S.r56/66 |
| ones-MFMA 行和（`nosmfv`）、`smfrs`、`smf2` | -2.2% / -4% / -2.7% | S.r54 |
| svcorr 全序列（`novsx` 配方） | -9.7% | S.r51/53 |
| 整数 shift 重基准（`svscale`） | 精度不过 parity | S.r47/51 |
| 单份 masked 尾（`peelone`） | -3%（vgpr 256） | S.r49/66 |
| 2×dword 存储（`ost1`）、16 次串行 Q 载入（`qdw`） | -9% / -6% | S.r65 |
| 合并组 `_vsm`（277-287 vs 308）、gen2/3 `_g2/_g3/_g3o`（-6.5 … -9%）、K-ahead ring `_vs6`（-10.5%） | -6 … -10.5% | S.r50/57/58/64 |
| pinned safe-asm 窗口（Block C、M2 `_vs9`） | -2.5% / -5..-9% | S.r63/68/73 |
| KV-split：直接 dwordx2 partial（`epdirect`）、fp32 partial（`nop16`）、e4m3 partial（`q3`） | -11% / -10% / 精度不过门 | S.r61/69、KR |
| 融合 last-arriver combine（`fused`/`fusedlite`） | 36.6-56 us vs 26.8 us | S.r62 |
| 调度：`ld` 预设代替重排（`_vs128`）、`AMDC_SCHED=0` | -3.8% / 更差 | S.r49/50 |
| 预设 lowering 去 `imm` | 281.0 vs 299.3（vgpr 256） | S.r49 |

**元结论**（S.r71-75、CS §5）：循环不是 VALU-issue bound（VALU -8.7% → wave cycles +1.2%），也不是 XDL bound；限制项是两组的相位汇合 + barrier 后 LDS 读延迟。**库的任何「更聪明」的默认（自动预取、自动合并窗口、自动减少 barrier、自动去冗余 MFMA）都已被测为负**，必须默认关闭。

---

## 5. 给库设计者的陷阱

### 5.1 名字耦合（必须处理，否则直接掉性能）
- 编译器按**名字**选 lowering/scheduler：`_vs128` 的 lowering 需要 `starts_with("fmha_fwd_hd128_mxfp8_vs")`（C:74），scheduler 需要 `starts_with("fmha_fwd_hd128_mxfp8_t"|"_sv"|"_vs")` 或 `ends_with("_t128"|"_g2"|…)`（C:107），重排开关又看 `ends_with("_vs128"|"_kvsplit"|"_t128"…)`（C:119）。
  新库 kernel 若改名（如 `tile_fmha_vs128`）会**失去全部预设**（P27 的 4-19%）。`_kvsplit` 若前缀不匹配会**无调度器**。→ 库应把预设作为 `Function` 元数据显式携带，或沿用名字。
- `ref_host` 从名字里的 `combine8` 推断每线程元素数（`port_gate.sh` 注释）。
- builder 内的 `VS_MODE` 全局 atomic 与 `VS_CORR_FORCE`（K:19568-19587）是**构建期**状态切换（决定 dma/dbuf/kdlds 与 LUT 副本），库若复用旧 builder 片段必须保持同样的 set/reset 顺序。

### 5.2 `FMHA_OPT` 语义
- 默认串 `cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,svpk,svcorr,swz`（K:19547）；**设置 FMHA_OPT 会整串替换默认**（S.r65/67 的 `qdw` 陷阱）。
  `_vs128` 实际依赖的默认开项是由 `!no*` 形式给出的：`kddma`、`smfv`、`q0reg/q1reg`、`rbtrim`、`vsx`、`p16`，以及 VS_MODE 强制的 `dma/dbuf/kdlds`；`cvt4,pksub,lazy,msum,permlane,lacc1,psh2,svpk` 不被 `_vs128/_t128` 读取（只服务 `_t/_sv`）。
- 库的 parity 目标 = **无 FMHA_OPT 环境变量**时的构建。

### 5.3 验收
- `tools/port_gate.sh <prefix>`：VERIFY（strict 0 错）、ACC16/ACC64（15 形状 ×4 路径，bad=0，parity ≤1.10×vendor）、DET、STRESS（300 次 split 形状）、PERF（中位数 ≥ 0.97× 同轮出厂）。
- 输出不变量是 **bf16 输出 md5**（`faadf00c…` B1H8S2048 等），不是 asm md5（S.r67）。

---

## 6. 未确认 / 未测项（诚实清单）

- XOR swizzle 在 `_vs128` 上的单独性能贡献：未测（只有 `_t` 的「冲突 -98%、速度不变」）。
- 物理 WAW padding 对出厂 kernel 的指令/性能影响：文档未单列（`AMDC_NO_PHYS_WAW=1` 可 A/B），未测。
- 早期 `nobar1/nobar2` 的符号：S.r50（带 setprio）为 +3%，S.r61 addendum 称「-9% each earlier」，原始数据未找到；以 `nobar` 的 -3.5..-7% 为准。
- ones 块 / Q0/Q1 LDS 写在默认配方下不被读取：删掉它们的影响未测（预计中性，但会改变 prologue 调度，需重测）。
- KV-split 前置全体 barrier（`1d9baea`，2026-10-31）的性能影响：未单独测量。
- `_t128` 的 tile 协议注解缺失：gpu-verify 对其 LDS 生命周期的覆盖弱于 `_vs128`（CS §2.1 只列 bounds）。

---

## 附录 A. 出厂基线（`tools/port_gate.sh fmha_fwd_hd128_mxfp8 baseline`，2026-11-02，mi355-gpu-53 GPU 6，git `071f9c5`）

| 段 | 项 | 值 | 结论 |
|---|---|---|---|
| VERIFY | `_vs128` / `_t128` / `_kvsplit` / `combine8h` `--strict` | 0 / 0 / 0 / 0 错误（trusted assumes 6 / 4 / 12 / 2） | PASS |
| ACC16 | 15 形状 auto，16 行 | REF_KERNEL_OK 15/15，parity 15/15 | PASS |
| ACC64 | auto / `_vs128` / `_t128` / 强制 split（13 形状），64 行 | bad>0 形状 0 / 0 / 0 / 0；parity 全过；SPLIT_ULP 7/7、12/13（信息） | PASS |
| DET | 5 形状 × 3 次 | 5/5 bit-identical | PASS |
| STRESS | B1H2S200 SK1337 gqa2 split4 ×300；B1H8S2048 auto ×100 | 0 失败；worst diff vs vendor 0.066 / 0.098；md5 各 1 种 | PASS（阈值 0.15） |
| PERF | B1H8S2048 auto（split） | **677.96** TF/s（同轮厂商 536.68，1.263x） | 基线 |
| PERF | per-CU 64 blocks（`_vs128`） | **454.80**（厂商 539.10） | 基线 |
| PERF | B4H8S2048（`_vs128`） | **1433.87**（厂商 1674.39，0.856x） | 基线 |
| PERF | B1H8S8192（`_vs128`） | **1987.42**（厂商 2359.30，0.842x） | 基线 |
| PERF | B2H8S2048（`_t128`，信息） | 1029.19（厂商 1012.24，1.017x） | 基线 |

5 次交错重复取中位数，`REF_WARMUP=150 REF_ITERS=100`。候选 kernel 集运行时，脚本在同一作业里交错跑出厂集合，按「候选中位数 ≥ 0.97 × 同轮出厂中位数」判定。

**命名约定（重要）**：编译器预设按名字选（P27，C:74/107/119）。在库提供显式「预设属性」之前，移植 kernel 统一用前缀
`fmha_fwd_hd128_mxfp8_vs2`：`..._vs2_vs128`、`..._vs2_t128`、`..._vs2_kvsplit`、`..._vs2_kvsplit_combine8h`
（以 `fmha_fwd_hd128_mxfp8_vs` 开头 → 得到 vendor-structure lowering 选项与 full scheduler；后缀 `_t128`/`_kvsplit` 触发各自附加项；
combine 名字含 `combine8` 供 ref_host 推断每线程元素数）。注意 `..._vs2_kvsplit_combine8h` 会比出厂 combine 多拿到 lowering 选项和调度器 —— 需要在 PERF 中确认不回退。
门禁命令：`HIP_VISIBLE_DEVICES=6 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_vs2`（未移植的 kernel 用 `KN_T128=fmha_fwd_hd128_mxfp8_t128` 等回落到出厂）。
