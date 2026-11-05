# DSL tile 抽象层设计（`amdgpu_tile`）

> 状态：**已实现**（`crates/amdgpu-tile/src/`；用它写成的 `fmha_fwd_hd128_mxfp8_dsl_*` 曾是出厂集，2026-11 整理中删除；v3 出厂集仍复用其 LdsPlan / Pipeline / softmax / combine 组件）。本文是 Phase 1 设计稿 + 实现路线（分支 `dsl-tile`）。作者：backend-llvm-tail（DSL 架构 + 库实现）。
> 目标（用户原话）："我专门开 dsl 出来，就是为了让你增加一些抽象来简化上层 kernel 的编写负担。不是把 asm 包一层隐藏起来。"
> 验收（用户批准）：四条出厂 FMHA fwd 路径（`_vs128`、`_t128`、`_kvsplit`、`_kvsplit_combine8h`）全部在新抽象上重写；性能与当前手写 kernel 相差 ≤ ~3%；
> 数值过现有门禁（15 形状、64 行 bad=0、确定性、split 形状重复启动压力）；不要求 bit-identical；`amdc gpu-verify --shipped --strict` 0 错误；`cargo test` 绿；
> 主 kernel 从 ~1600 行降到 ~100–200 行。

> **后续（路线 B，已出厂）**：本文描述的 tile 库（L1–L6，`amdgpu_tile`）是上一版出厂集 `dsl_*` 的基础。之后按
> [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) 把其中**与布局有关的层**换成了布局代数：
>
> | 本文的层 | 路线 B 中的替代 |
> |---|---|
> | L1 张量 `TensorDecl` 自动合同 | 合同从 `amdgpu_layout::Layout` 推出：`amdgpu_tile::contract::compact_tensor` / `same_strides`（`crates/amdgpu-kernels/src/fmha/mod.rs:358-430` 的 `declare_io`） |
> | L3 copy（`RowsCopy`、`DwordRowsCopy`、`RowFrags::load`） | `partition_dma`、`partition_buffer_dma`、`partition_global`（`crates/amdgpu-tile/src/{dma,bufdma,global}.rs`） |
> | L4 片段读（`RowsReader`、`TrReader`、`RowScaleReader`） | `make_tiled_copy_a/_b/_bcast` + `partition_lds`（`crates/amdgpu-tile/src/copy.rs`），atom 来自 `amdgpu-isa` |
> | L4 MMA（`qk_tile`/`pv_tile` 中的 MFMA） | `TiledMma` + `gemm`（`crates/amdgpu-tile/src/mma.rs`；支持 `swap_ab`） |
> | L2 `LdsPlan`/标记、L5 `Pipeline`、L6 softmax / epilogue 语义 / combine | **保留**，v3 直接复用 |
>
> 新的出厂 kernel：`crates/amdgpu-kernels/src/fmha/mod.rs`（`fmha_fwd_hd128_mxfp8_v3_vs128` 在 `:283`，K/V 环在 `KvRing3::new` `:64-101`）。
> 地址等价性由测试保证：`crates/amdgpu-tile/tests/address_equality.rs`（TiledCopy == RowsReader / TrReader，逐 lane）、
> `crates/amdgpu-tile/tests/dma_equality.rs`（partition_dma == RowsCopy）。入门教程：[`layout-tour.md`](layout-tour.md)。

---

## 0. 现状：为什么今天的 kernel 是"带类型的汇编"

逐段读过的真实代码（行号以 `f39eb49` 为准）：

| 位置 | 内容 | 体量 | 本质 |
|---|---|---|---|
| `fmha_t_build`（kernels.rs:21004） | 656 B kernarg 的 30 个 `arg`、Q 片段载入、lane 地址（`kfx`/`trx`/`swzw`）、GQA、KV-split 区间、角色分裂、短/长序列两份代码 | ~640 行 | 全部是手写地址算术（`and/lshr/xor/shl/mul/add`） |
| `fmha_abi_contract`（:20933） | `ka_range/ka_is/grid_is/arg_tensor` | 57 行 | **形状声明被手工展开**成 30 条合同 |
| `fmha_vs_tiles`（:20914） | `lds_ring/lds_tile/lds_private/lds_alias` | 17 行 | **LDS 布局被手工编号**（69632、86016 这类魔数在 role 里又重复出现） |
| `emit_fmha_vs128_role`（:9602–11292） | DMA、ring、QK、softmax/回滚、P 转换、PV、行和、三种 epilogue | ~1690 行 | 每个 lane 地址、每条 MFMA、每个 barrier、每个 `tile_produce/consume` 都手写；80% 是当时的 `FMHA_OPT` 实验分支（历史） |
| `emit_fmha_t128_role`（:16117–17060） | 同一算法的单组/3 槽/单 barrier 版本 | ~940 行 | 与 vs128 大段重复（`swz_s` 函数在文件里被复制了 6 次） |
| `..._kvsplit_combine8h`（:20708） | split 合并 | 80 行 | 在线 max/rescale 手写 |

kernels.rs 全文约 4,800 处地址算术 builder 调用。抽象缺失的具体表现：

1. **形状不是一等公民**：Q 的地址 `row*ka(0x70) + b*ka(0xA0) + h*ka(0x90) + 16*pblk` 在 `fmha_t_build` 里手算，同一个形状关系又在 `fmha_abi_contract` 里以 `ka_is(0x0A0, S*ka(0x70))` 手写一遍。两处必须同步，靠人。
2. **LDS 布局不是一等公民**：`fmha_vs_tiles` 声明 K 环在 `[0, 65536)`，role 里又独立写 `lds_kd = 2*32768`、`ones_base = 2*32768+2048`、`c69632`……
3. **lane→元素映射（MFMA 布局）不是一等公民**：C32x32 累加器的行号公式 `8*(r/4) + 4*(lane/32) + r%4` 在 mask（`valid`）、epilogue（`64e+16q`、`g_bytes8`）、行和里各写一遍。
4. **流水线节拍不是一等公民**：`stage_start`/`stage_end`/两个 barrier/角色错拍/尾迭代剥离写在 `iter` 闭包里，`tile_produce/consume` 的 (slot, version) 用 3 个 `Cell` 在闭包间传递。
5. **算法组件不是一等公民**：在线 softmax 的"参考偏移 + 回滚"配方（双 f16 字 bias、阈值、`exp2(512δ)`）在 vs128、t128、vs9、vs6、g2、g3… 每个 role 各复制一份。

## 1. 设计原则

* **P1 抽象的是"语义"，不是"指令"**。作者写"Q 的这个 tile"、"K 环的第 t 代"、"S += K·Qᵀ"、"行最大值"、"在线 softmax"，库生成地址、标记、等待、合同。不是把 asm 包一层：作者不再写 lane 号、字节偏移、barrier。
* **P2 生成的指令模式与手写 kernel 相同**（性能对等的唯一可靠路径）。每个组件的 lowering 都对照现有手写序列编写，并有"同 op 序列"单测（见 §9）。不做黑盒优化；选择（DMA 形式、读宽度、转换链）由布局/配置**确定性**决定。
* **P3 安全只增不减**。所有 `arg_tensor/ka_is/grid_is/assume_trusted/lds_*/tile_*` 由声明生成；库内部的 trusted assume 带固定理由，进证书账本；`commit` 时对 LDS 真实占用做重叠检查（未声明复用的真重叠 = 构建期 panic）。新 kernel 必须 `gpu-verify --strict` 0 错误。
* **P4 可渐进移植**。库是普通 Rust（`#![forbid(unsafe_code)]` 继承自 crate），所有组件都接受 `&mut FnBuilder` 并返回普通 `ValueId/VarId` 包装：kernel-opt 可以先在手写 kernel 里替换一段，比对 asm，再替换下一段。
* **P5 底层复用 `expr.rs`**。地址公式写成 `I32` 表达式树再 `bind`，与手写 op 一一对应（`expr.rs` 已有"同 asm 同 MIR"的测试先例）。

## 2. 分层总览

```
 L6 组件库      online_softmax / epilogue(bf16, kvsplit partial+lse) / combine_splits / kv_split_range / gqa
 L5 片段运算    row_max(max3+partner) / masked / add_pk / scale_pk / exp2 / to_p8(pknorm|LUT) / row_sum(smfmac)
 L4 片段+MMA    Frag<L>, Acc<L>, AccTile<N>; FragReader(b128x2 | tr_b8x4); mma_tile(scaled 32x32x64)
 L3 搬运+流水   copy(TileView -> SlotRef){GlobalLds|BufferLds}; Pipeline{slots, stages, Cadence, roles}
 L2 共享内存    LdsPlan -> Shared{ring/tile/private, Layout(swizzle)}; SlotRef{slot, gen} -> tile_* 标记
 L1 张量视图    KArgs, Dim, Tensor{dims, strides} -> arg/ka_is/arg_tensor; TileView{origin, shape, clamp}
 L0 现有 IR     FnBuilder + expr.rs (I32/F32/Cond/Ptr)
```

模块树：`crates/amdgpu-tile/src/{mod.rs, lane.rs, tensor.rs, shared.rs, copy.rs, pipeline.rs, frag.rs, reduce.rs, softmax.rs, epilogue.rs}`；
测试：`crates/amdgpu-compiler/tests/tile_*.rs`。

---

## 3. L1 张量视图 `Tensor`

### 3.1 作者写什么

```rust
let mut ka = KArgs::new(&mut b);                       // kernarg 表：声明 + 惰性载入 + 去重
let s   = ka.dim("s_seq_len", 0x060, 1..);             // 运行期维度：Ext::ka(0x60) + 载入的 I32
let skv = ka.dim("s_kv_seq_len", 0x110, 1..);
let h   = ka.dim("s_q_head_num", 0x140, 1..);
let hk  = Dim::sym("HK");                              // 只存在于合同里的符号
let bat = Dim::grid(2);                                // B = grid_z（kvsplit: Dim::sym("B")）
let d   = Dim::c(128);

// BSHD，strides 以字节计，来自 kernarg；`packed()` = 声明 "stride[i] = dims[i+1]*stride[i+1]"
let q = ka.tensor("ptr_q", 0x010, Elem::Fp8, [bat, s, h, d], [0x0A0, 0x070, 0x090, Stride::Unit]).packed();
let k = ka.tensor("ptr_k", 0x020, Elem::Fp8, [bat, skv, hk, d], [0x0E0, 0x0C0, 0x0D0, Stride::Unit]).packed();
let o = ka.tensor("ptr_o", 0x000, Elem::Bf16, [bat, s, h, d], [0x1A0, 0x180, 0x190, Stride::Unit]).packed();
let qs = ka.tensor("ptr_q_descale", 0x200, Elem::U8x4, [bat, s, h], [0x230, Stride::Derived, 0x240]).slack(Ext::c(512*4) * h.ext());
```

### 3.2 生成什么

| 声明 | 生成（与 `fmha_abi_contract` 对照） |
|---|---|
| `ka.dim(name, off, 1..)` | `b.arg(name, I32, off, 4)`、`b.ka_range(off, 1, MAX)`；首次用到时 `KernargI32{off}` |
| `ka.tensor(...)` | `b.arg(ptr)` + 每个 stride kernarg 的 `b.arg`；`b.arg_tensor(ptr, dims[0] * stride[0])` |
| `.packed()` | 对每个非常量 stride：`ka_is(stride_i, dims[i+1] * stride_{i+1})`；对常量 stride（head stride = 128 / bf16 的 256）：`ka_range(off, c, c)` |
| `.slack(extra)` | `arg_tensor_slack(ptr, logical, logical + extra)` |
| `Tensor::view(...)`（非 packed 的别名视图，如 `v` 共享 `k` 的 strides） | `ka_is(v_stride, Ext::ka(k_stride))` |

不能由形状推出的关系（`H = HK*gqa`、`gqa ∈ [1,16]`、`grid.x = ceil(S/256)`）仍由作者显式写，但通过有名字的组件：`gqa_heads(&mut ka, h, hk, 0x0B0)`（生成 `ka_range/ka_is` + 返回 `kv_head` 值 + 那条 trusted assume）、`launch_grid(b, [S.ceil_div(256), H, B])`。

### 3.3 Tile 视图与 per-lane 寻址

```rust
let qt  = q.tile([bidx, row0, hidx, 0], [1, 256, 1, 128]).clamp_rows(1);      // 维 1（seq）越界行 = 最后一个合法行
let row = qt.lane_row(&lane, |l| wave32 + l.j);                                  // 每个 lane 的行
let qp  = qt.row_ptr(&mut b, row);                                               // select(row<S, row, S-1)*stride + ... 生成于此
```

* `TileView` 持有 origin（`I32` 表达式）、shape、每维的 clamp 策略（`Clamp::LastValid(limit)` / `Clamp::None` / `Clamp::Min(limit)`；`limit` 默认是该维 `Dim` 的运行值，KV-split 用 `.with_extent(1, kv_eff)` 换成 split 长度）。
* `row_ptr` 产生的 op 序列与手写一致：`cmp_lt_i32 + Select`（或 `min_i32`）+ `mul_i32` + `add_i32` + `ptr_add`。
* **clamp 是自动的**：只要 tile 的 extent 可能越过维度（shape 不整除 / origin 来自 grid），`row_ptr` 一定生成 clamp；没有 clamp 的访问要求作者写 `.unclamped("reason")`，而且验证器会照常证明（证不出就 `GV-GLB-OOB`）。
* 片段载入：`Frag::<B32x32x64Fp8Kperm>::load_global(&mut b, &qt, row, khalf)` 生成手写的 `2 × global_load_dwordx4 + lane 拆分`（`fmha_t_build` :21195）。布局类型决定读宽度与字节顺序（§6）。

### 3.4 合同如何自动产生、为何安全不退化

验证器只看到与今天相同种类的合同条目（`arg_tensor`、`ka_is`、`ka_range`）——它们由声明一次性生成，地址用同一组 `Dim`/stride 值生成，因此"合同与地址不同步"这一类人为错误消失。合同正确性（与 host 真实分配一致）仍由 `contract_host_truth.rs` 对账；新增测试会把 `Tensor` 生成的合同与 `fmha_abi_contract` 的条目做**集合相等**比较（§9）。

---

## 4. L2 共享内存 `LdsPlan` / `Shared` / `SlotRef`

### 4.1 作者写什么

```rust
let mut lds = LdsPlan::new();
// 两槽环，每槽 = [K 16 KB | V 16 KB]（交错环组），128 行 x 128 B，XOR-16B swizzle
let [ks, vs] = lds.ring_group(2, [("K", Rows(128, 128), Layout::FMHA_SWZ), ("V", Rows(128, 128), Layout::FMHA_SWZ)]);
let kd   = lds.ring("KD", Bytes(512), 2, Layout::Linear);          // K scale 环
let ones = lds.tile("ones", Bytes(2048), Layout::LaneMajor16);
let q1   = lds.tile("Q1", Bytes(16384), Layout::LaneMajor16);
let q0   = lds.tile("Q0", Bytes(16384), Layout::LaneMajor16);
let epi  = lds.private("epi", Bytes(4096), 8).overlays(&[&q1, &q0]);   // 时间复用：epilogue 时 Q 已死
lds.commit(&mut b);                                                     // = b.lds(102400) + 全部 lds_* 声明
```

### 4.2 生成什么

| 声明 | 生成 |
|---|---|
| `ring_group(n, [A, B])` | `lds_ring("K", base, 32768, 2)`、`lds_ring("V", base+16384, 32768, 2)`；交错导致的**声明区间**重叠（V 环的声明范围 `[16384, 81920)` 覆盖后面的块）自动 `lds_alias` |
| `ring/tile` | `lds_ring/lds_tile`，地址按声明顺序 16 B 对齐分配（可 `.at(off)` 固定，用于 bit 级复刻） |
| `private(name, per_wave, waves)` | `lds_private(name, base, per_wave, waves)` |
| `.overlays(&[..])` | 放在被覆盖块的基址；对每个被覆盖块（含其声明区间相交的 ring）`lds_alias` |
| `commit` | `b.lds(total)`；**真实占用**（每槽实际字节区间）两两求交，未经 `overlays` 声明的真实重叠 → 构建期 panic（"`X` 与 `Y` 真实重叠且未声明时间复用"） |

自动别名只用于两种情况：(a) 交错环组的声明范围伪重叠（真实字节不相交——库可证明）；(b) 作者显式的 `overlays`。因此与手写 `fmha_vs_tiles` 等价，但不会把真实冲突静默地别名掉。

### 4.3 布局/swizzle 策略

```rust
pub enum Layout {
    Linear,                                  // offset = byte
    Rows { row_bytes: u32, swz: Swz },       // offset = row*row_bytes + ((chunk ^ swz(row)) << 4) + (byte & 15)
    LaneMajor16 { half_stride: u32 },        // Q/ones 暂存：lane*16，第二个 16 B 在 +1024（无冲突）
}
pub enum Swz { None, Xor16(fn(&I32) -> I32) }
pub const FMHA_SWZ: Swz = Swz::Xor16(fmha_swz); // s2 = u2^p, s1 = u0, s0 = u1^p, u = (r>>1)&7, p = r&1（status.md 第 801 行的蛮力搜索结果）
```

`fmha_swz` 只实现一次（今天被复制了 6 份）。它的 `I32` 表达式 bind 出的 op 序列与手写 `swz_s` 完全一致（and/lshr/xor/shl/add，同顺序）。

### 4.4 `SlotRef`：标记与代际编号

```rust
let cur  = ks.slot(&it.cur);     // it.cur = (ring idx 变量, 代 = 迭代计数 it)
let next = ks.slot(&it.next);    // (idx ^ 1 或 3 槽的 select, it + 1)
cur.read(&mut b, |b| { ... ds_read ... });      // = tile_consume("K", slot, gen) ... tile_end
next.write(&mut b, |b| { ... DMA ... });        // = tile_produce("K", slot, gen) ... tile_end
let base: I32 = cur.base();                     // = slot_idx * slot_stride + tile base（地址由此派生）
```

* (slot, gen) 由 Pipeline 的迭代器产生（§5），与手写的 `stage_tag/qk_tag/pv_tag` 三个 `Cell` 等价：作者不再手传版本号，也就不可能把 `it` 写成 `it+1`（`GV-PROTO-STALE-GEN` 那类错误从源头消失）。
* 私有区：`epi.mine(&wave_id)` 返回 `PrivRef`，地址只能由 `wave_id` 派生（满足 `lds_private` 的"wave_key 就是地址所用的同一个值"规则，见 lib.rs:902），读写自动带 `tile_produce/consume(name, wave_id, gen)`；同 wave 写后读自动插 `wait_lds(0)`。

---

## 5. L3 搬运 `copy` + 流水 `Pipeline`

### 5.1 copy

```rust
let kcp = Copy::rows_to_shared(&mut b, &k_tile, &ks, WavePart { wave: &wave4, waves: 4, rows_per_wave: 32 }, DmaForm::GlobalLds);
// 每 tile：
kcp.issue(&mut b, &next, key0);            // 在 next.write(..) 标记内发 4 条 global_load_lds_dwordx4
```

`Copy::rows_to_shared` 的生成规则（就是手写 `dma_issue_global`）：

* 每 wave 负责 `rows_per_wave = 32` 行；一条 DMA = 64 lane × 16 B = 1 KB = 8 行 × 128 B；`q ∈ 0..4`。
* lane i 负责 tile 行 `8q + (i>>3)` 的 16 B chunk `(i & 7) ^ swz(row)`——**目标 LDS 是线性落位的，所以 swizzle 施加在源地址上**（由 `ks` 的 `Layout` 推出）。
* M0 = wave 4 KB 区段的顶端（`+3072`），指令偏移为负 `-(1024*(3-q))`，全局地址补偿正值（32 位偏移零扩展，负偏移会绕到 +4 GB——手写注释里的坑，库里固化）。
* 行 clamp 来自 `k_tile` 的 clamp 策略（`select(row<kv_len,row,kv_last)`）；行无关部分（`base_off + cbytes + comp`）可选择 hoist（`dmah` 等价开关）。
* 每 wave 的不变量（`lane_hi/lane_lo/s_even/s_odd/cbytes_*`、`lds_tile`）在 `rows_to_shared` 时一次性计算——与手写一致。
* `DmaForm::BufferLds` 生成 `buffer_desc_dyn + load_buffer_lds`（opt-in；验证器尚未认证，见 gpu-semantics-verifier.md 13.6）。
* `Copy::col_dword_to_shared`：K-scale 每 lane 1 dword（`kd_dma`：wave w 与 w^2 重复覆盖 64 行）。

### 5.2 Pipeline：把节拍变成数据

```rust
pub enum Cadence {
    /// `_vs128`/`_kvsplit`：两组错拍（A 搬 K，B 搬 V），每 tile 两个 barrier：[stage(t+1) | V 相位 | wait_vm(0) wait_lds(0) barrier | M 相位 | wait_lds(0) barrier]
    TwoPhaseOffset,
    /// `_t128`：单组 4 wave 同时搬 K、V，3 槽，每 tile 一个 barrier：[stage(t+2) | 窗口 | wait_vm(0) wait_lds(0) barrier]
    OneWindow,
}
pub struct Pipeline { slots: u32, cadence: Cadence, stages: Vec<Stage> }   // Stage{ shared, copy, producer: Role }
pub trait PipeBody {
    fn prologue_compute(&mut self, b: &mut FnBuilder, t0: &TileIt);           // QK(0)
    fn v_phase(&mut self, b: &mut FnBuilder, t: &TileIt) ;                    // softmax(t)（TwoPhaseOffset）
    fn m_phase(&mut self, b: &mut FnBuilder, t: &TileIt) ;                    // PV(t) + QK(t+1)
    fn window(&mut self, b: &mut FnBuilder, t: &TileIt) { ... }               // OneWindow 用
}
pipe.run(&mut b, role, nkv, &mut body);
```

`TileIt` 提供：`it`（计数变量）、`cur/next/n2: SlotIdx`、`key0()`、`masked: bool`（编译期：尾迭代的掩码副本）、`staged: bool`（尾迭代不再 stage）。

`run` 生成（与 `emit_fmha_vs128_role` 的 :9948–10175 + :10675–10857 一致）：

1. 序：stage(tile 0)（含 `wait_vm(0)`）→ 作者的 `prologue_stores`（Q/ones 常量）→ `wait_vm(0) wait_lds(0) barrier`（P0）→ 角色 B 多一个 barrier（P1）→ `prologue_compute`（QK(0)）→ `wait_lds(0) barrier`。
2. 循环：`begin_loop([it, idx, 作者状态…])`，`assume_trusted(it ∈ [0, ceil(SK/128)])`（理由固定文本，进账本），头部 `break_if(it >= nkv-1)`。
3. 每次迭代按 `Cadence` 插入 stage / wait / barrier / `set_prio`（可选），`idx ^= 1`、`it += 1`。
4. 尾迭代剥离：`if kv_len < key0+128 { iter(masked) } else { iter(unmasked) }`，`stage=false`，最后一个 barrier 只在角色 A 执行（两侧 barrier 数相等——这是 `GV-PROTO-ARRIVAL-MISMATCH` 的不变量，库按 cadence 计算并在 debug 构建中断言两角色 barrier 计数相等）。
5. 所有 `tile_produce/consume` 的 (slot, gen) 都来自 `TileIt`，不由作者传。

作者状态（S/O/行和/bias 变量）通过 `pipe.state(DType, init)` 注册为循环变量，返回 `VarId` 包装（`Acc`/`StateVar`）。

### 5.3 角色

```rust
let roles = Roles::split(&mut b, &lane, 4);        // is_a = wave < 4；wave4 = wave & 3；wid = wave
roles.each(&mut b, |b, role| body(b, role));        // begin_if(is_a) body(A) else_ body(B) end_if
```

`Role` 只携带编译期 tag（A/B）+ wave-in-group 值；`Stage{producer: Role::A}` 让 `run` 只在生产者分支里发 DMA。`versioned(&mut b, cond, |b, flag| ..)` 生成"两份代码副本"（vsx 短/长序列 LUT 校正）。

---

## 6. L4 寄存器片段与 tile 级 MMA

### 6.1 布局类型

```rust
pub trait FragLayout { const TY: DType; }
pub struct A32x32x64Fp8;      // Frag8，MFMA scaled 32x32x64 A 操作数
pub struct B32x32x64Fp8;      // Frag8，B 操作数
pub struct C32x32F32;         // Frag16 累加器：lane l 的第 r 个值 = (row 8(r>>2)+4(l>>5)+(r&3), col l&31)
pub struct Smf16x16x128Acc;   // Frag4（smfmac 行和）
pub struct Frag<L>(ValueId, PhantomData<L>);
pub struct Acc<L>(VarId, PhantomData<L>);          // 循环携带
pub struct AccTile<L, const N: usize>([Acc<L>; N]); // S: 4 个 32-key 块；O: 4 个 32-dim 块
```

`mma_scaled(&mut b, &acc, &Frag<A32x32x64Fp8>, &Frag<B32x32x64Fp8>, sa, sb)` 只接受匹配的布局类型——把 A 片段当 B 喂是 **rustc 类型错误**（今天是运行时数值错误）。

`C32x32F32::row(l, r)` / `col(l)` 是唯一的 lane→元素映射；mask（§7）、epilogue（§8）、行和都从它派生。

### 6.2 从共享 tile 读片段：`FragReader`

```rust
let kread = FragReader::rows_b128x2(&mut b, &ks, &lane);     // A = K 行（K 行主序存储），预计算 kfx[0..4]
let vread = FragReader::tr_b8x4(&mut b, &vs, &lane);         // A = Vᵀ（V 行主序，转置读），预计算 trx[0..4]
let k8 = kread.read(&mut b, &cur, s, khalf);                 // 2 × ds_read_b128  (+ tile_consume 标记)
let v8 = vread.read(&mut b, &cur, hh, e);                    // 4 × ds_read_b64_tr_b8
```

选择规则：存储布局（行主序 + swizzle）× 操作数需要的布局 → 读指令形状。行主序且操作数 K 维连续 → `b128x2`；操作数需要列（转置）→ `tr_b8x4`。lane 地址公式（`kfx`：`j*128 + ((pblk ^ s(j)) ^ 2k) << 4`；`trx`：`key_part*128 + 8(l&1) + ((m_gl ^ s(key_part)) ^ 2e) << 4`）由 `Layout` 的 swizzle 推出，读时只加 `slot base + 常量`（与手写同）。

### 6.3 tile 级 MMA

```rust
// S[s] (+)= K(slot)[32s..32s+32, :] · Qᵀ   —— 4 个 S 块 × 2 个 k-half = 8 条 v_mfma_scale_f32_32x32x64_f8f6f4
mma_tile(&mut b, &s_tile, MmaA::Shared(&kread, &next), MmaB::Regs(&qfrag), Scales { a: KScaleSrc::Shared(&kd, &next), b: [sq0, sq1] });
// O[e] += Vᵀ(slot)[e] · P[hh]   —— 2 × 4 = 8 条
mma_tile(&mut b, &o_tile, MmaA::SharedTr(&vread, &cur), MmaB::Words(&p), Scales::constant(127, 123));
```

`mma_tile` 内部是 `for s { for khalf { read A (标记) ; var_mfma_scale } }`，顺序与手写一致；K-scale 字节移位（`>> g8`、`>> g8+16`）由 `KScaleSrc` 产生。

---

## 7. L5 片段逐元素运算与归约

| API | 生成（手写出处） |
|---|---|
| `s_tile.values(&mut b)` | `frag16_lane + bitcast_f32` × 64 |
| `KeyMask::new(key0, kv_len).apply(vals, fill)` | 每元素 `key = key0 + 32s + C32x32F32::row(l,r)`，`cmp_lt + Select(fill)`（`valid` 闭包） |
| `row_max(&mut b, vals)` | `max3_f32` 链（64 值 = 1 + 31 次）+ `partner_max_f32`（lane l^32 交换） |
| `s_tile.add_pk(&mut b, pair)` / `o_tile.scale_pk` / `la.scale_lane(2, c)` | `var_add_pk` / `var_scale_pk` / `var_scale_lane` |
| `to_p8(&mut b, vals, scale, P8Conv::PkNorm)` | `cvt_pknorm_p8x4`（每 4 值 1 字） |
| `to_p8(.., P8Conv::PkNormLut)` | `cvt_pknorm_pair → lshr/and → perm_b32(LUT) → pk_sub_u16 → cvt_p8x4_from_u16`（短序列校正） |
| `row_sum_smfmac(&mut b, &la, &a_rs, [pa, pb])` | 4 × `var_smfmac_fp8`（idx01/idx23），`a_rs` 由 `SmfmacOnes::new(lane)` 生成 |

## 8. L6 组件库

### 8.1 `OnlineSoftmax`（带参考偏移/回滚配方）

```rust
let mut sm = OnlineSoftmax::new(&mut pipe, Recipe::Vsx { thr: 120.0/512.0, b131: 131.0/512.0 });  // 注册 bias2/la 循环变量
let bias = sm.decide(&mut b, &s_tile, mask, &o_tile);   // max → 双 f16 字 bias 链 → if(grow){S+=δ, O*=2^(512δ), l*=…}
sm.bias_mfma(&mut b, &s_tile, bias);                    // QK 前：S = ones ⊗ bias（f16 MFMA，4 条）
let l = sm.row_sum_total(&mut b);                       // 0.0625 * smfmac dword 2
```

`decide` 精确复刻手写 vsx 路径（:10403–10496）：`grow = thr < m`；`delta` 对不回滚的 lane 用 `Select` 置 0（分支无 exec 掩码——注释里的数值不变量由库固化，作者无法遗漏）；`set_var(bias2, nb)` 只写一次且之后用 SSA 值（满足 `var_lint`）。其他配方（单字 bias / svscale）作为 `Recipe` 变体保留。

### 8.2 epilogue

```rust
Epilogue::bf16(&mut b, &o, &o_tile_acc, fin, live_rows);           // O*(fin) → cvt_pk_bf16 → 64 × global_store_dwordx2（:11259）
Epilogue::kvsplit_f16(&mut b, &ohat, &lse, &o_tile_acc, fin, &sm, &epi.mine(&wid), split_row); // LDS 私有区转置 + 16 B 合并写 + lse
```

`kvsplit_f16` 自动产生：入口 barrier（:10862 那个修过真实 bug 的 barrier——私有区与 Q 暂存时间复用，由 `epi.overlays(&[q1,q0])` 推出"复用前必须同步"，库插入）、私有区标记、`wait_lds(0)`。

### 8.3 combine

```rust
combine_splits(&mut b, &ohat, &lse, &out, rows, splits, PerThread(8), PartialElem::F16);   // = combine8h 全部
```

在线 `m/den/acc` 循环（`exp2(m_old-m_new)`）、clamp、trusted assume（行号上界、split 计数器）全部内置。

### 8.4 其他组件

`kv_split_range(&mut b, ntiles, splits_ka)`（KV-split 的 tile 区间 + 4 条 trusted assume）、`gqa_heads`（kv_head + assume）、`q_frags(&mut b, &q_tile, rows)`（Q 片段 + 描述符缩放 sq0/sq1）。

---

## 9. 性能对等与安全的保证方式

| 机制 | 说明 |
|---|---|
| op 序列单测 | 每个组件有一个"手写参照"小 kernel（直接从 kernels.rs 抽出的那段 op 序列）与库版本，比较 IR 语句种类序列 / 编译后 asm 指令直方图（`tile_*.rs`）。 |
| 指令直方图门 | 新 `_vs128` 与旧 `_vs128` 的 asm 按助记符计数（MFMA、ds_read、global_load_lds、v_*）差异列表，作为 PR 附件；VGPR 数对比。 |
| 编译策略 | 编译器目前按**名字**选 lowering/scheduler 选项（`lib.rs:74-127`：`starts_with("fmha_fwd_hd128_mxfp8_vs")`、`ends_with("_vs128")` 等）。新 kernel 需要同样的策略：见 §11 决策 D1。 |
| 合同等价 | `tile_contract.rs`：`Tensor` 生成的合同集合 ⊇ `fmha_abi_contract` 的集合。 |
| 验证 | 每个 demo kernel `gpu_verify::analyse` 0 error；新 FMHA kernel 进入 `--shipped` 列表前必须 0 error；库内部 trusted assume 数可计数（证书账本）。 |

## 10. 新 `_vs128` 的目标源码形状（伪真实 Rust，约 150 行）

```rust
pub fn fmha_fwd_hd128_mxfp8_vs128_tile() -> Function {
    let mut b = FnBuilder::new(NAME, 512);
    let mut ka = KArgs::new(&mut b);
    let fm = FmhaAbi::declare(&mut ka, AbiKind::Single { rows_per_block: 256 });   // 656 B ABI：Q/K/V/O/scales 张量 + GQA + grid（L1 组件）
    let lane = Lane::new(&mut b);                                                     // tid, wave, lane, j, half
    let blk  = fm.block(&mut b, &lane);                                               // bidx/hidx/kv_head, row0, nkv, kv_len（含 clamp/assume）

    // ---- LDS 计划 ----
    let mut lds = LdsPlan::new();
    let [ks, vs] = lds.ring_group(2, [("K", Rows(128, 128), FMHA_SWZ), ("V", Rows(128, 128), FMHA_SWZ)]);
    let kd   = lds.ring("KD", Bytes(512), 2, Layout::Linear);
    let ones = lds.tile("ones", Bytes(2048), Layout::LaneMajor16);
    let q1s  = lds.tile("Q1", Bytes(16384), Layout::LaneMajor16);
    let q0s  = lds.tile("Q0", Bytes(16384), Layout::LaneMajor16);
    let epi  = lds.private("epi", Bytes(4096), 8).overlays(&[&q1s, &q0s]);
    lds.commit(&mut b);

    // ---- 寄存器常驻的 Q 片段 + 片段读取器（lane 不变量一次算好）----
    let q = q_frags(&mut b, &fm.q.tile_rows(&blk), &lane);                // Frag<B32x32x64Fp8> x2 + sq0/sq1
    let kread = FragReader::rows_b128x2(&mut b, &ks, &lane);
    let vread = FragReader::tr_b8x4(&mut b, &vs, &lane);
    let kdread = KScaleReader::new(&mut b, &kd, &lane);

    let roles = Roles::split(&mut b, &lane, 4);
    versioned(&mut b, fm.short_seq(&mut b, &blk), |b, short| {
        roles.each(b, |b, role| {
            let k_tile = fm.k.tile_keys(&blk).clamp_rows(1);
            let v_tile = fm.v.tile_keys(&blk).clamp_rows(1);
            let mut pipe = Pipeline::new(2, Cadence::TwoPhaseOffset)
                .stage(&ks, Copy::rows_to_shared(b, &k_tile, &ks, role.part(32)), Role::A)
                .stage(&kd, Copy::col_dword_to_shared(b, &fm.kscale_tile(&blk), &kd, role.part(64)), Role::A)
                .stage(&vs, Copy::rows_to_shared(b, &v_tile, &vs, role.part(32)), Role::B);
            pipe.prologue_stores(|b| {
                ones.fill_lanes(b, role.wid(), 0x3838_3838);
                q0s.store_frag(b, role.wid(), &q.b0);
                q1s.store_frag(b, role.wid(), &q.b1);
            });
            let s = pipe.acc_tile::<C32x32F32, 4>(b);           // S（初值由 prologue_compute 给）
            let o = pipe.acc_tile_zero::<C32x32F32, 4>(b);
            let mut sm = OnlineSoftmax::new(b, &mut pipe, Recipe::vsx(short));
            let mut body = Vs128Body { s, o, sm: &mut sm, q: &q, kread, vread, kdread, p: None };
            pipe.run(b, role, blk.nkv, &mut body);
            let fin = sm.finish(b) * fm.v_descale(b);
            Epilogue::bf16(b, &fm.o.tile_rows(&blk), &o, fin, blk.live());
        });
    });
    b.finish()
}

struct Vs128Body<'a> { s: AccTile<C32x32F32, 4>, o: AccTile<C32x32F32, 4>, sm: &'a mut OnlineSoftmax, q: &'a QFrags,
                       kread: FragReader, vread: FragReader, kdread: KScaleReader, p: Option<PWords> }

impl PipeBody for Vs128Body<'_> {
    fn prologue_compute(&mut self, b: &mut FnBuilder, t: &TileIt) { self.qk(b, &t.cur(), self.sm.bias_init()); }
    // V 相位：softmax(t)，P(t) 留给 M 相位
    fn v_phase(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let vals = KeyMask::for_tile(t).apply(b, self.s.values(b), f32::NEG_INFINITY.max(-1e30));
        let bias = self.sm.decide(b, &self.s, &vals, &self.o);
        self.p = Some(to_p8(b, &KeyMask::for_tile(t).apply(b, self.s.values(b), 0.0), self.sm.p_scale(), self.sm.conv()).with_bias(bias));
    }
    // M 相位：PV(t) + 行和 + QK(t+1)
    fn m_phase(&mut self, b: &mut FnBuilder, t: &TileIt) {
        let p = self.p.take().unwrap();
        row_sum_smfmac(b, self.sm.l(), &p);
        mma_tile(b, &self.o, MmaA::SharedTr(&self.vread, &t.cur()), MmaB::Words(&p), Scales::constant(127, 123));
        if t.staged { self.qk(b, &t.next(), p.bias()); }
    }
}
impl Vs128Body<'_> {
    fn qk(&mut self, b: &mut FnBuilder, slot: &SlotIdx, bias: BiasWord) {
        self.sm.bias_mfma(b, &self.s, bias);
        mma_tile(b, &self.s, MmaA::Shared(&self.kread, slot), MmaB::Regs(&self.q.frags), Scales { a: self.kdread.at(slot), b: self.q.scales() });
    }
}
```

`_kvsplit` = 同一 body，`AbiKind::KvSplit` + `kv_split_range` + `Epilogue::kvsplit_f16`；`_t128` = 同一 body 方法，`Cadence::OneWindow`、3 槽、单组（`Stage{producer: Role::All}`），`window()` = `v_phase` 的每 S 块转换与 `qk` 交错 + PV；`combine8h` = `combine_splits(..)` 一行加 ABI 声明。

## 11. 需要 lead/用户决策的点

* **D1 编译策略按名字选择**：`amdgpu-compiler/src/lib.rs:74-127` 用名字前缀/后缀决定 lowering 选项（scalarize/retarget2/imm/sgpr/addrfold）与调度器预设（reorder / alu_across_barrier / max_vgprs 248）。新 kernel 若叫 `..._vs128_tile` 会落到错误的预设（`ends_with("_vs128")` 失败 → `reorder=false`）。建议：在 `Function` 上加一个可选字段 `compile_profile: Option<&'static str>`（`"vs128" | "t128" | "kvsplit" | "combine"`），编译器优先用它，名字匹配作为回退；或新 kernel 名用 `fmha_fwd_hd128_mxfp8_vs_<x>_vs128` 这种同时满足前缀/后缀的形式（不推荐）。需要 lead 拍板（改 Function 结构属于共享代码）。
* **D2 `_t128` 目前没有 tile 标记**（不在 `fmha_vs_tiles` 列表里），库会自动加标记——验证器对 3 槽 select 索引的环可能需要 `tile_trust_periodic`（进账本，+1 trusted）。这是安全**增强**，但证书计数会变。
* **D3 实验分支不移植**（当时的 `FMHA_OPT` 开关已全部删除）：新 kernel 只实现出厂默认配方（`kddma`、`smfv`、`q0reg/q1reg`、vsx 双副本、`dwordx2` 输出、fp16 partial）。实验开关如需保留，作为库组件的配置参数而不是 kernel 里的 `if`。
* **D4 命名**：新 kernel 暂定 `fmha_fwd_hd128_mxfp8_{vs128,t128,kvsplit,kvsplit_combine8h}` 的 `_dsl` 变体并行存在，过门禁后切换出厂映射（md5 不变量在切换前不动）。

## 12. 实现路线（分支 `dsl-tile`）

1. L1+L2+L3（`tensor.rs`、`shared.rs`、`copy.rs`、`pipeline.rs`、`lane.rs`）+ 测试 `tile_tensor.rs`、`tile_shared.rs`、`tile_pipeline.rs`（demo：2 角色 DMA 环 kernel，`--strict` 干净；`Tensor` 合同 = 手写合同；`copy` 的 op 序列 = `dma_issue_global`）。
2. L4（`frag.rs`）+ `tile_mma.rs`（demo：QK 一个 tile，asm 中 8 条 `v_mfma_scale_f32_32x32x64_f8f6f4` + 8 对 `ds_read_b128` + 标记）。
3. L5/L6（`reduce.rs`、`softmax.rs`、`epilogue.rs`）+ `tile_softmax.rs`、`tile_epilogue.rs`、`tile_combine.rs`（combine8h 用库重写的 demo 与原 kernel 的数值等价性：host 解释器/同 op 序列）。
4. 每层落地后报告 lead；kernel-opt 在 L1–L4 齐备后开始移植。

---

## 13. 实现状态（as-built，分支 `dsl-tile`）

| 层 | 模块 | 实际 API（与 §3–§8 设计的差异） | 测试 |
|---|---|---|---|
| D1 | `amdgpu_dsl::CompileProfile`、`Function.compile_profile`、`b.compile_profile(..)`；编译器 `policy_name(f)` | profile 映射到对应出厂 kernel 的名字，所有按名字选策略的检查（lowering 选项、调度预设、vcc fill、pin_vmwait）看到的就是该 kernel 的策略；`None` = 旧行为 | 出厂 asm md5 不变 |
| L0 | `expr.rs` | 新增循环变量叶子 `b.i32var(VarId)` / `tile::varx`，`b.bind_opnd`（裸变量根返回 `Opnd::Var`） | `expr_layer` |
| L1 | `tile/tensor.rs` | `KArgs{arg,i32,ptr,dim,tensor}`，`Dim{C,Ka,Sym,Grid}`，`Stride{Unit,Ka(name,off),Derived(expr)}`，`TensorDecl.packed()/same_strides_as()/alloc()/slack()/declare()`；`TileView.clamp(dim,limit,last)/clamp_min` 作用在**相对**索引上，`offset/ptr/row_source`。元素类型用 `elem_bytes` 而不是 `Elem` 枚举 | `tile_l123`：生成的 Q/K/V/O 合同与出厂合同**结构相等** |
| L2 | `tile/shared.rs` | `LdsPlan{ring_group,ring,tile,private(..,overlays),overlay,at,total,commit}`，`Layout{Linear,Rows{row_bytes,swz}}`、`Layout::FMHA_SWZ`、`fmha_swz`；`SlotIdx{slot,gen}`（带按 stride 的绑定偏移缓存）、`SlotRef{off,off_v,base,read,write}`、`PrivRef{base,write,read_back}` | `tile_l123`：复现 `fmha_vs_tiles`（偏移、102400、7 对别名完全一致）；未声明真实重叠 → panic |
| L3 | `tile/copy.rs`、`tile/pipeline.rs`、`tile/lane.rs` | `RowsCopy::new(b,src,&dst,wave,&lane,waves).issue(b,&key0,&slot_ref)`、`DwordRowsCopy`；`Pipeline::new(Cadence,nkv,kv_len,128,bound)`（`ring_var_last` 旋钮）`.run(b,role,&mut body)`，`trait PipeBody{stage,stage_complete,prologue_stores,prologue_compute,loop_state,bind_state,v_phase,m_phase,window}`，`TileIt{it,cur,next,n2,masked,staged}`；`Lane`、`Roles{split,single,each,wave_id}`、`versioned` | `tile_l123`：两角色 DMA 环 0 错误、DMA/barrier 数精确；错代/提前读被拒；OneWindow 3 槽环 0 错误 |
| L4 | `tile/frag.rs` | 布局类型 `A32x32x64Fp8/B32x32x64Fp8/C32x32F32/C16x16F32`，`Frag<L>/Acc<L>/AccTile<L,N>`，`RowFrags::load`（Q），`RowsReader`（kfx）、`TrReader`（trx）、`RowScaleReader`、`ScaleSel`；tile 级 MMA 为 `qk_tile(QkOut::{Vars,Values})` 与 `pv_tile`（取代设计稿里泛化的 `mma_tile`） | `tile_l4`：MFMA/ds_read/tr 计数精确 |
| L5 | `tile/reduce.rs` | `KeyMask::new(b,t,half_base,limit).valid`，`masked_values`，`row_max`，`to_p8(PkNorm/PkNormLut)`+`LutConsts`，`SmfmacOnes{new,row_sum}` | `tile_l56` |
| L6 | `tile/softmax.rs`、`tile/epilogue.rs` | `OnlineSoftmax::new(b,&lane,Recipe::vsx(lut))`，`state/bind/decide/convert/row_sum/l_total/lse`；`store_bf16`、`kvsplit_partial_f16`、`combine_splits(8|16)` | `tile_l56`：只用库组件写成的完整 attention（LUT/非 LUT、bf16/KV-split 尾声）0 错误可编译；库版 combine8h 与出厂指令构成一致 |

验证器的两处增强（都不改变出厂证书：44 trusted / 216 contracts / 0 unproven，proven 计数不变）：
1. `ka_is` 定义按**不动点**代换（与声明顺序无关；kernel-opt 发现：外层 stride 关系先于内层声明时出现伪 `GV-GLB-OOB`）。
2. **有限状态环计数器**：单一更新是"自身 + 常量"的 select/compare 函数的循环变量（如 3 槽环 `idx < 2 ? idx + 1 : 0`），在宿主上求出其循环长度 p；标记的周期声明 = lcm(p, 仿射界)，仍在模拟轨迹上校验。`_t128` 的 3 槽环因此**无需** `tile_trust_periodic`（决策 D2）。
