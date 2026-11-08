# TIRx 式 tile-op 覆盖矩阵（gfx950）

> `op-coverage-design.md` 的 §6（kernel-opt 主笔的总设计；本节由 backend 负责）。目标：以 aiter kernel zoo（`MI_Kernel_Agent/knowledge/aiter-kernel-zoo`：
> attention、gemm、moe、quantization、normalization、rope、sampling-topk、cache-layout、fusion-activation …）为需求，按 **TIRx 的 tile-op 模型**
> 列出 gfx950 指令族的覆盖情况、缺口与优先级，并给出 `tile::apply` / `tile::ewise` 在我们栈上的形态与验证方法。基线：Main `f25294dc`。

## 6.1 TIRx 的模型与我们的对应

TIRx 只有两个 tile 操作（[Tile Operations](https://tvm.apache.org/tirx/programming_guides/tile/index.html)、
[Tile Operands and Execution](https://tvm.apache.org/tirx/programming_guides/tile/tile_patterns.html)）：

- **`tile.apply(instr, dst, srcs…, scope)`**：把**一条指定指令**在操作数视图上重复执行。视图的坐标分三段：**group**（执行组：warp / lane）、
  **repeat**（每线程重复几次）、**atom**（一条指令处理的值，由指令决定）。lowering 检查视图满足指令的对齐 / 连续 / 归属约束；不插同步。
  覆盖向量 load/store、算术（选定舍入）、异步拷贝（cp.async）、TMA、warp MMA、tcgen05。
- **`tile.ewise(f, dst, srcs…)`**：把一个**标量函数**内联到每个寄存器元素上（可能展开成多条指令）；group 之后的坐标都是元素。

我们已有的对应物（都在构建期展开成普通 IR，见 [`tirx-comparison.md`](tirx-comparison.md) §4.5）：

| TIRx | 我们 | 位置 |
|---|---|---|
| 指令 + atom 形状 | `CopyAtom`（宽度、空间、立即数域、计数器、LDS-direct 形式）、`MmaAtom`（A/B/C 的 TV 布局、passes） | `crates/amdgpu-isa` |
| 操作数视图（group / repeat / atom） | TV 布局（lane × value）+ `Layout` 代数；`IssuePlan`（一次发射的 atom 序列） | `crates/amdgpu-layout` |
| `apply`（拷贝 / MMA） | `TiledCopy` / `TiledMma`、`partition_{lds,dma,buffer_dma,global}`：按 TV 布局把 atom 重复到视图上，构建期证明覆盖 / 连续 / 对齐 / bank / 操作数匹配 | `crates/amdgpu-tile/src/{copy,mma}.rs` |
| `apply`（算术，选定指令） | 直接调 DSL op（`Op::PkMulF32` …）或 safe-asm（指定指令形式与等待态） | `crates/amdgpu-dsl`、`crates/amdgpu-safeasm` |
| `ewise` | 表达式层（`I32/F32` 运算符）在片段 lane 上的手写循环；stage 库通用原语 E（逐元素）/ R（归约）/ Q（量化） | `crates/amdgpu-dsl/src/expr.rs`、`crates/amdgpu-tile/src/prims.rs` |
| 同步由调用者负责 | 同；但 `s_waitcnt` 由编译器插入，协议由 `gpu_verify` 证明 | `crates/amdgpu-compiler` |

**差距**：没有一个统一的 `apply(atom, view…)` 入口——拷贝与 MMA 有，**VALU / 转换 / 跨 lane（DPP、permlane、bpermute）没有 atom 化**；`ewise`
没有统一形式（各组件手写 lane 循环），也不会自动把成对元素选成 packed 指令。

## 6.2 指令族覆盖矩阵

状态列：**MIR**（`MOp`，own emit + 两张 LLVM 表行或 own-only）/ **DSL**（`Op` / `Stmt`）/ **atom**（`amdgpu-isa`）/ **safe-asm**（`ops.rs` 有类型化形式）/
**HW**（硬件探针或出货 kernel 已在 GPU 上验证）。✔ 已有，△ 部分，✘ 缺。优先级见 §6.4。

### 6.2.1 全局 / buffer / flat 访存

| 指令族 | gfx950 形式 | MIR | DSL | atom | safe-asm | HW | 缺口 |
|---|---|---|---|---|---|---|---|
| global load dword / x4 | `global_load_dword{,x4}`（含 SADDR、`sc0/sc1/nt`） | ✔ | ✔ `LoadF32`、`LoadGlobalB128` | ✔ X1 / X4 | ✔ x4 | ✔ | — |
| global load x2 / x3 | `global_load_dwordx2/x3` | ✘ | ✘ | △ X2 atom 已有，无 MOp | ✘ | ✘ | **P1** |
| global load 8/16 位 + d16 | `global_load_{ubyte,sbyte,ushort,sshort}`、`_d16`、`_d16_hi` | △ ushort | △ `LoadU16` | ✘ | ✘ | △ | **P1**（量化 / cache-layout 的 int8 / fp8 / fp16 散读） |
| global store | `global_store_{byte,short,dword,dwordx2,x3,x4}`、`_d16_hi` | △ short / dword / x2 / x4 | △ | △ X2 / X4 | ✘ | ✔ | byte / x3 / d16_hi：**P1** |
| 一致 / 写穿访问 | `sc0 sc1 nt`（coherent、streaming、non-temporal） | △ `sc1` load / store | △ `LoadF32Coherent`、`StoreF32Wt` | ✘ | ✘ | ✔ last-arriver | `nt` 流式存储（大输出 GEMM / cache 写）：**P2** |
| buffer load / store | `buffer_load/store_dword{,x2,x4}`（offen / off、soffset、12 位 imm） | ✔ | ✔ `BufferLoad` / `BufferStore` | △ X4 | ✘ | ✔ 越界语义已探针 | x3、8/16 位、d16：**P1**；`idxen`：P3 |
| global → LDS DMA | `global_load_lds_{dword,dwordx3,dwordx4}`、`_ubyte/_ushort` | △ dword / x4（含 SADDR） | ✔ `LoadGlobalLds` | ✔ | ✘ | ✔ | x3、8/16 位：**P2** |
| buffer → LDS DMA | `buffer_load_{dword,dwordx4} … lds` | ✔ | ✔ `BufferLoadLds` | ✔ | ✘ | ✔ | x3 / 8 / 16 位：P2 |
| flat | `flat_load/store_*` | ✘ | ✘ | ✘ | ✘ | ✘ | 不需要（global / buffer 覆盖）：不做 |
| 全局原子 | `global_atomic_{add,sub,max,min,and,or,xor,swap,cmpswap}`（u32 / i32 / u64）、`add_f32`、`pk_add_f16/bf16`、`max/min_f32`（返回 / 不返回） | △ `add` u32 返回 | △ `AtomicAddU32` | ✘ | ✘ | ✔ | **P2**（MoE 排序、top-k、split-K 归约）：add_f32 / pk_add_bf16、max / min、cmpswap、无返回形式 |
| buffer 原子 | `buffer_atomic_*` | ✘ | ✘ | ✘ | ✘ | ✘ | P3 |
| 缓存控制 | `buffer_wbl2`、`buffer_inv` | ✔ | ✔ fence | — | ✘ | ✔ | — |

### 6.2.2 LDS（DS）

| 指令族 | gfx950 形式 | MIR | DSL | atom | safe-asm | HW | 缺口 |
|---|---|---|---|---|---|---|---|
| ds_read / write b32 / b64 / b128 | `ds_{read,write}_b{32,64,128}` | ✔ | ✔ | ✔ | ✔ b32 / b128 | ✔ | — |
| ds b96 | `ds_{read,write}_b96` | ✘ | ✘ | ✘ | ✘ | ✘ | **P1**（hd 96 / 3 元组布局） |
| 双地址 | `ds_read2 / write2 _b32 / _b64`（两个 8 位 offset）、`_st64` | ✘ | ✘ | ✘ | ✘ | ✘ | **P1**（转置写入、行跨步读，省地址 VALU） |
| 8 / 16 位与 d16 | `ds_read_{u8,i8,u16,i16}`、`_d16`、`_d16_hi`、`ds_write_b8/b16`、`_d16_hi` | ✘ | ✘ | ✘ | ✘ | ✘ | **P1**（量化 scale、索引表） |
| 转置读 | `ds_read_b64_tr_b8`（fp8）、`ds_read_b64_tr_b16`（f16 / bf16）、`ds_read_b64_tr_b4`（fp4）、`ds_read_b96_tr_b6`（fp6） | △ tr_b8 | △ `LoadLdsTrB8` / `Tr4B8` | △ TR_B8X4 | ✔ tr_b8 | ✔ tr_b8（one-hot 探针） | tr_b16：**P1**（bf16 / f16 GEMM 与 attention）；tr_b4 / tr_b6：P2（MXFP4 / FP6） |
| LDS 原子 | `ds_add/max/min/and/or/xor_{u32,f32}`、`ds_add_rtn_*`、`ds_cmpst` | ✘ | ✘ | ✘ | ✘ | ✘ | P2（直方图、top-k、MoE 计数） |
| 跨 lane via LDS | `ds_bpermute_b32`、`ds_permute_b32`、`ds_swizzle_b32` | ✘ | ✘ | ✘ | ✘ | ✘ | **P1**（任意 lane 交换：归约、top-k、布局转换） |

### 6.2.3 跨 lane（VALU）

| 指令族 | 形式 | MIR | DSL | atom | safe-asm | HW | 缺口 |
|---|---|---|---|---|---|---|---|
| DPP | `quad_perm`、`row_shl/shr/ror`、`row_mirror`、`row_bcast:15/31`、`wave_shr` …（作为 `v_mov_b32_dpp` 或任意 VOP1/2 的修饰） | △ `v_mov_b32_dpp quad_perm` | △ `DppQuadPerm` | ✘ | ✘ | ✔ quad_perm | row / bcast 模式与"DPP 修饰的算术"（一条指令完成 shuffle+op）：**P1**（warp 归约） |
| permlane | `v_permlane32_swap`、`v_permlane16_swap` | △ 32 | △ `Permlane32Swap` | ✘ | ✔ 32 | ✔ 32 | permlane16_swap：**P1** |
| readlane / writelane | `v_readfirstlane`、`v_readlane`、`v_writelane` | △ readfirstlane | △ | — | ✔ readfirstlane | ✔ | readlane / writelane：P2 |
| 片段内重排 | `v_perm_b32`、`v_alignbit/alignbyte` | △ perm | ✔ `PermB32` | — | ✘ | ✔ | alignbit：P3 |

### 6.2.4 VALU 算术与 packed

| 指令族 | 已有 | 缺口（优先级） |
|---|---|---|
| f32 | `v_add/sub/mul/fma/max/max3_f32`、`v_cmp_lt_f32`、`v_cndmask` | `v_min/min3/med3_f32`、`v_mac/fmac`、`v_ldexp/frexp/fract/floor/ceil/rndne`、`v_div_scale/fixup/fmas`（精确除法）、`v_cmp_*` 其余谓词、`v_cmp_class`（NaN / Inf）：**P1**（norm、softmax、激活、采样） |
| 整数 | `v_add_u32/add_co/addc`、`v_mul_lo_u32`、`v_and/or/xor`、`v_lshl/lshr`、`v_min_i32`、`v_cmp_{lt,ge,gt,ne}_i32` | `v_sub_u32`、`v_mul_hi_u32`、`v_mad_u32_u24`、`v_lshl_add / add_lshl / lshl_or / and_or`、`v_bfe/bfi`、`v_ashr`、`v_max_i32/u32`、`v_min_u32`：**P1**（地址与索引、MoE 排序、量化位操作） |
| packed f32 | `v_pk_add_f32`、`v_pk_mul_f32` | `v_pk_fma_f32`、`v_pk_mov_b32`：**P1** |
| packed f16 / bf16 / i16 | `v_pk_sub_u16` | `v_pk_{add,mul,fma,max,min}_f16`、bf16 packed（gfx950 `v_pk_*_bf16` 若有，否则经 f32）、`v_pk_{add,sub,mul_lo,max,min}_{u16,i16}`、`v_dot2_f32_f16 / bf16`：P2（f16 epilogue、激活） |
| 超越函数 | `v_exp_f32`、`v_log_f32`、`v_rcp_f32` | `v_rsq_f32`、`v_sqrt_f32`（norm）、`v_sin/cos_f32`（RoPE）、`v_exp_f16`、`v_rcp_iflag`：**P1**（rsq / sqrt / sin / cos） |

### 6.2.5 类型转换

| 方向 | 已有 | 缺口（优先级） |
|---|---|---|
| f32 ↔ f16 / bf16 | `v_cvt_f32_f16`、`v_cvt_f16_f32`、`v_cvt_pk_bf16_f32` | `v_cvt_pkrtz_f16_f32`、bf16 → f32（移位，DSL 层可组合）：P2 |
| f32 → fp8 / bf8 | `v_cvt_pk_fp8_f32`（lo / hi 半写）、`v_cvt_scalef32_pk_fp8_f32`（lo / hi） | `v_cvt_pk_bf8_f32`、`v_cvt_scalef32_pk_bf8_f32`、随机舍入 `_sr_`：**P1**（量化） |
| fp8 / bf8 → f32 | ✘ | `v_cvt_f32_fp8 / bf8`（`byte_sel`）、`v_cvt_pk_f32_fp8 / bf8`、`v_cvt_scalef32_{f32,pk_f32}_{fp8,bf8}`：**P1**（反量化） |
| fp4（MXFP4）/ fp6 | ✘ | `v_cvt_scalef32_pk_fp4_f32 / _f16 / _bf16`、`v_cvt_scalef32_pk_f32_fp4`、`v_cvt_scalef32_pk32_{f32,f16,bf16}_{fp6,bf6}` 与反向：**P2**（MXFP4 MoE / GEMM） |
| int ↔ float | ✘（以 bitcast + 位运算替代） | `v_cvt_f32_{i32,u32}`、`v_cvt_{i32,u32}_f32`、`v_cvt_f32_ubyte{0..3}`：**P1**（int8 量化、索引） |
| 归一化打包 | `v_cvt_pknorm_u16_f32` | `v_cvt_pknorm_i16_f32`、`v_cvt_pk_u8_f32`：P3 |

### 6.2.6 矩阵核心

| 族 | gfx950 形式 | MIR | DSL | atom | safe-asm | HW | 缺口 |
|---|---|---|---|---|---|---|---|
| scaled f8f6f4 | `v_mfma_scale_f32_32x32x64_f8f6f4`、`_16x16x128_` | ✔（含 zero-C、op_sel 选 E8M0 字节） | ✔ | ✔ | ✔ 32x32 | ✔ 出货 | **A/B 数据格式选择（cbsz / blgp：fp8、bf8、fp6、bf6、fp4）**：DSL 只有 fp8(e4m3)：**P1**（MXFP4 / bf8） |
| unscaled f8f6f4 | `v_mfma_f32_32x32x64_f8f6f4`、`16x16x128` | △ 32x32 | △ | ✘ | ✔ 32x32 | ✔ | 16x16x128 unscaled、格式选择：P2 |
| fp8 旧形式 | `v_mfma_f32_{16x16x32,32x32x16}_{fp8,bf8}_{fp8,bf8}` | △ fp8_fp8 两形 | ✘ | ✘ | ✘ | ✘ | P3（被 f8f6f4 取代） |
| f16 | `v_mfma_f32_16x16x16_f16`、`32x32x8_f16`、gfx950 新增 `16x16x32_f16`、`32x32x16_f16` | △ 前两种 | △ | △ 32x32x8 | ✔ 32x32x8 zero | ✔（bias） | gfx950 双速率 f16：**P1**（f16 attention / GEMM） |
| bf16 | `v_mfma_f32_{16x16x16,32x32x8}_bf16`、gfx950 `16x16x32_bf16`、`32x32x16_bf16` | ✘ | ✘ | ✘ | ✘ | ✘ | **P1**（bf16 是 zoo 里最常见的 GEMM / attention 精度） |
| i8 | `v_mfma_i32_{16x16x64,32x32x32}_i8`（gfx950） | ✘ | ✘ | ✘ | ✘ | ✘ | P2（int8 GEMM / 量化 attention） |
| f32 / xf32 | `v_mfma_f32_{16x16x4,32x32x2}_f32`、`_xf32` | ✘ | ✘ | ✘ | ✘ | ✘ | P3 |
| f64 | `v_mfma_f64_16x16x4_f64` | ✘ | ✘ | ✘ | ✘ | ✘ | 不做 |
| 稀疏 | `v_smfmac_f32_16x16x128_fp8_fp8`、其他 `smfmac` 形状 / 类型（bf8、f16、bf16、i8、32x32） | △ 1 种 | △ | ✔ | ✔ | ✔ | P3 |
| WMMA | — | gfx950 无 | | | | | 不适用 |
| AGPR 形式 | 所有 MFMA 的 `acc_cd` | ✔ | ✔ `pin_agpr` / `agpr` | — | — | ✔ GEMM v4 | — |

### 6.2.7 标量 / 其他

| 族 | 已有 | 缺口（优先级） |
|---|---|---|
| SMEM | `s_load_dword{,x2,x4}`（kernarg） | `s_load_dwordx8/x16`、`s_buffer_load_dword*`（常量表、MoE 路由表）：**P2** |
| SALU | `s_add/addc/mul/lshl/lshr/and/xor/min/cmp_lt/cselect`、`s_mov` | `s_sub`、`s_max`、`s_or`、`s_bfe`、`s_ff1`（找首个 1：top-k / 稀疏）、`s_bcnt1`、`s_cmp_*` 其余：P2 |
| 同步 / 控制 | `s_barrier`、`s_waitcnt`（vm / lgkm）、`s_setprio`、`sched_group_barrier`、`s_nop`、`s_memtime` | `s_sleep`、`s_memrealtime`、`s_waitcnt expcnt`（无需）：P3 |

## 6.3 `tile::apply` / `tile::ewise` 在我们栈上的形态

新模块 `amdgpu-tile::tileop`（在已有 `TiledCopy` / `TiledMma` 之上，不另起一套布局）：

```rust
/// 一个 atom：一条指令的操作数形状与约束（拷贝 / MMA 已有；新增 VALU / 转换 / 跨 lane atom）
pub enum Atom { Copy(CopyAtom), Mma(MmaAtom), Valu(ValuAtom), Lane(LaneAtom) }
pub struct ValuAtom { pub op: ValuOp, pub width: u8 /* 1 或 2（packed）*/, pub dtype_in: DType, pub dtype_out: DType }

/// 视图 = 存储（寄存器片段 / LDS tile / 全局张量）+ TV 布局（group = lane / wave，repeat，atom 轴）
pub struct View<'a> { pub storage: Storage<'a>, pub tv: TvLayout }

/// TIRx apply：atom 在视图的 repeat 轴上重复；构建期检查对齐 / 连续 / 归属（layout 证明），失败即构建错误
pub fn apply(b: &mut FnBuilder, atom: &Atom, dst: &View, srcs: &[&View]) -> Result<Applied, TileOpError>;

/// TIRx ewise：标量闭包作用到每个寄存器元素；相邻元素在同一 32 位寄存器对中且闭包是单个可 pack 运算时，自动选 packed 指令
pub fn ewise(b: &mut FnBuilder, dst: &RegTile, srcs: &[&RegTile], f: impl Fn(&mut FnBuilder, &[Opnd]) -> Opnd) -> RegTile;
```

- **视图坐标**沿用 TIRx 的 group / repeat / atom 三段；`group` 只有 lane（0..64）与 wave 两种轴（gfx950 无 warp-group 集体指令）。
- **检查**（构建期，失败报告带布局坐标）：拷贝 atom 的字节连续与对齐（DS b96 / b128 对齐、`tr` 读的 8 字节对齐与 lane 分组、buffer 12 位 imm）、
  MMA atom 的 A / B / C TV 布局匹配（已有证明）、VALU atom 的 packed 配对（两元素同寄存器对）、跨 lane atom 的 lane 置换合法（DPP 行内、permlane 半 wave）。
- **同步**：与 TIRx 一样由调用者负责 barrier 与协议标记；`s_waitcnt` 仍由编译器插入，协议仍由 `gpu_verify` 证明。
- `apply` / `ewise` 只是构建器：输出普通 DSL op，capture / weave、IR 文本、验证器全部照常适用。

## 6.4 缺口优先级与实施顺序

以 zoo 的覆盖面（attention、GEMM、MoE、量化、norm、RoPE、top-k、cache-layout）和"tile / pipeline / 访存性能"为准：

| 批次 | 内容 | 理由 |
|---|---|---|
| **P1-a 访存** | global / buffer `x2` / `x3` / 8 / 16 位 / `d16(_hi)` load 与 byte / x3 store；`ds_read/write_b96`、`ds_read2/write2(_st64)`、DS 8 / 16 位；`ds_read_b64_tr_b16` | 几乎所有非 fp8 kernel 的数据通路；转置写入与行跨步读省地址 VALU |
| **P1-b 跨 lane** | `ds_bpermute / swizzle`、DPP row / bcast 模式与 DPP 修饰算术、`permlane16_swap` | warp 归约（norm、softmax、top-k、sampling） |
| **P1-c 转换与算术** | fp8 / bf8 双向（含 scalef32、sr）、int ↔ float、`v_min/min3/med3`、整数地址助手（`mad_u32_u24`、`lshl_add`、`bfe/bfi`、`mul_hi`）、`pk_fma_f32`、`rsq / sqrt / sin / cos` | 量化 / 反量化、norm、RoPE、激活 |
| **P1-d 矩阵** | bf16 MFMA（含 gfx950 双速率形状）、gfx950 双速率 f16、f8f6f4 的 A / B 格式选择（bf8 / fp4 / fp6） | bf16 GEMM / attention、MXFP4 |
| **P2** | 全局原子（f32 / bf16 add、max / min、cmpswap、无返回）、LDS 原子、`tr_b4 / tr_b6`、fp4 / fp6 转换、`nt` 流式存储、`global_load_lds_dwordx3` / 8 / 16 位、i8 MFMA、packed f16 / bf16、SMEM 宽加载 / `s_buffer_load`、SALU 补全 | MoE、top-k、MXFP4、int8 |
| **P3** | buffer 原子、`idxen`、旧 fp8 MFMA 形式、f32 / xf32 MFMA、其余 smfmac、`alignbit`、`s_sleep` | 边缘需求 |

每批在 `tile::apply` / `tile::ewise` 落地之前先补底层（MOp → DSL op → atom），按 §6.5 验证。

## 6.5 每条新指令的验证闭环

1. **属性表一行**（`op_attrs.rs`：效果、hazard 类、调度代价、AGPR 合法性、后端）——穷尽匹配，漏写不能编译。
2. **汇编器探针**：own emit 文本由 ROCm clang 汇编（`asm_assembles`），两张 LLVM 表的行（或登记 own-only）并通过 `tail_invariance` / `llvm_tail`。
3. **hazard 模型**：等待态与 ISA 文档对照，`hazard_oracle` 交叉检查（LLVM 的插入量）。
4. **硬件探针**：每个族一个探针 kernel（`probes/`），one-hot 输入对照 CPU 参考，验证 TV 布局 / 字节选择 / 舍入 / 越界语义（先例：[`isa-probes-smfmac-tr.md`](isa-probes-smfmac-tr.md)、MUBUF 越界探针）；结果写进 atom 的 provenance。
5. **验证器覆盖**：新的地址形式进 `gpu_verify` 的区间模型（越界、对齐），新的原子 / 跨 lane 指令进效果模型；攻击用例进 `verifier_cases`。
6. **DSL 层**：`Op` / `Stmt` 新变体 → 运行 `tools/gen_ir_codec.py`（IR 文本往返）、`facts.rs` 分类（capture / weave 依赖）。

## 6.6 实施状态

按 sched（batch 2）的顺序实施：A1 → A3 → A2 → A4，然后 B1–B4；GEMM 族（bf16 MFMA、f8f6f4 格式选择、fp4）并行推进。

| 批次 | 内容 | 状态 |
|---|---|---|
| **A1 VALU / math** | DSL 通用算子 `Op::Un(UnOp, x)` / `Op::Bin(BinOp, a, b)` / `Op::Tri(TriOp, a, b, c)` / `Op::Cmp { pred, ty, a, b }`（构建器 `un` / `bin` / `tri` / `cmp`）；60 个新 MOp：`floor/ceil/trunc/rndne/fract`、`rsq/sqrt/sin/cos`（`sin/cos` 以"圈"为单位）、`cvt_{i32,u32}_f32` / `cvt_f32_{i32,u32}` / `cvt_f32_ubyte0..3`、`min_f32`、`max/min_{i32,u32}`、`sub_u32`、`mul_u32_u24`、`ashrrev_i32`、`mul_hi_{u32,i32}`、`or_b32`、`min3/med3_f32`、`min3/max3/med3_{i32,u32}`、`mad_u32_u24`、`lshl_add_u32`、`add3_u32`、`bfe_{u32,i32}`、`bfi_b32`、`pk_{add,mul,max,min,fma}_f16`、比较（f32：Lt/Le/Gt/Ge/Eq/Ne(neq)/Lg/Uno/Ord；i32/u32：Lt/Le/Gt/Ge/Eq/Ne）。验证：两张 LLVM 表行（`tail_invariance`、`llvm_tail`）、汇编（`asm_assembles`）、`gpu_verify` 对整数地址算子的区间规则（Add3、LshlAdd、MadU24、Min3、Max、Bfe…）、硬件探针 `probe_valu_p1`（`tools/tileop_probe.sh`：68 个算子 × 64 lane 含 NaN / ±0 / Inf / 边界整数，own 与 llvm-tail 两个后端 **4352 / 4352** 全部与 CPU 参考一致） | **完成** |
| **A3 访存** | DSL：`MemTy { U8, I8, U16, I16, B32, B64, B96, B128 }` 参数化的 `Op::LoadGlobalN / BufferLoadN / LoadLdsN`、`Stmt::StoreGlobalN / BufferStoreN / StoreLdsN`（`hi`：8 / 16 位的 `_d16_hi` 存储）、`Op::LoadGlobalD16 / LoadLdsD16`（`D16Ty`、lo / hi）、`Op::LoadLds2` / `Stmt::StoreLds2`（`ds_read2 / write2[st64]_b32 / _b64`，常量地址折叠进 offset0 / offset1）、`Op::LoadLdsTrB16`；50 个新 MOp（96 位访问用 V4 寄存器，LLVM 行用新 token `{d3}` / `{sNt3}`）。IR 验证器（类型、13 位立即数、buffer 立即数）、`gpu_verify` 对每个新访问做越界证明（全局 / buffer / LDS，read2 两个地址都查）与 vm / lgkm 计数、`gpu_protocol` 把它们计入 LDS / 全局访问，窄加载的值区间（U8 0..255 等）。硬件探针 `probe_mem_p1`（64 个平面 × 64 lane + 全局 / buffer 两个字节区，未写字节保持 0xA5）：两个后端**全部一致**。**发现**：gfx950（SRAM-ECC 打开）上 d16 加载**不保留**另一半（清零）——DSL 语义据此定为"另一半为 0"，去掉了 merge 操作数。`ds_read_b64_tr_b16` 已能编译 / 汇编，硬件布局探针待做 | **完成**（tr_b16 布局待探针） |
| **P1-d MFMA（第一批）** | DSL：`MmaKind { shape: MmaShape, a: MmaTy, b: MmaTy, scaled }` 参数化的 `Op::Mfma { kind, a, b, c: Option, scale }` 与 `Stmt::VarMfma { var, kind, a, b, scale, zero }`（构建器 `mfma` / `var_mfma`）；本批可降级：gfx950 双速率 `v_mfma_f32_16x16x32_bf16`、`32x32x16_bf16`、`16x16x32_f16`、`32x32x16_f16` 与 `v_mfma_i32_16x16x64_i8`、`32x32x32_i8`（C 与零 C 形式、VGPR 与 AGPR(acc_cd) 累加器两套 LLVM 行；其余组合由验证器拒绝）。isa 原子 `MmaAtom::mfma_{bf16,f16}_{16x16x32,32x32x16}()`、`mfma_i8_{16x16x64,32x32x32}()`（布局 = 探针测得：A/B lane l 持有行 l % M、k = e·(l / M) + 0..；C 同 16x16 / 32x32 原子）。硬件探针 `probe_mfma_p1`：6 种 × (零 C、C、AGPR 中的变量累加) 两个后端 **18 / 18** 一致。冒险：16x16 形状按 8 pass（结果延迟 12）、32x32 按 16 pass（20），C 形式可 SrcC 链接（`hazard_oracle` 对照 LLVM 通过）。另加 `Op::VarSlice4`（Frag8 / Frag16 变量的 Frag4 视图，无指令） | **完成** |
| **P1-d MFMA（第二批）** | 块缩放 f8f6f4 的 A / B 格式选择：`MmaKind::scaled_f8f6f4(M16N16K128 | M32N32K64, a, b)`，a / b ∈ {Fp8, Bf8, Fp6, Bf6, Fp4}（可混合，如 a8w4），`mfma_scaled` / `var_mfma_scaled`；36 个 MOp（A / B 寄存器宽 4 / 6 / 8 的组合 × C / 零 C；imm 携带 opsel 与 cbsz / blgp；6 寄存器操作数用 Frag8 的前 6 个，LLVM token `{sNtK}`）。硬件探针 `probe_f8f6f4_p1`（13 种组合含混合格式，两个后端全部精确）**测得**：8 位格式的 k 顺序同既有 fp8 原子（两个 16 元素半段），而 **fp4 与 fp6 / bf6 是连续的** k = 32·(l / M) + v；isa 原子 `MmaAtom::mfma_scale_{16x16x128,32x32x64}_f4()`。冒险延迟暂按 20 保守 | **完成** |
| **A2 转换** | `Op::Cvt { op: CvtOp, srcs: [Opnd; 4], sel }`（构建器 `cvt(op, &srcs, sel)`）：fp8 / bf8 → f32（`v_cvt_pk_f32_{fp8,bf8}` 字选择、`v_cvt_f32_{fp8,bf8}` 字节选择，SDWA 形式）、f32 → bf8 打包（`v_cvt_pk_bf8_f32`，字选择、部分写）、块缩放反量化 `v_cvt_scalef32_pk_{bf16,f16,f32}_{fp8,bf8}` 与 `_{f32,bf16,f16}_fp4`（字 / 字节选择）、块缩放量化 `v_cvt_scalef32_pk_{fp8,bf8,fp4}_f32`（部分写）、`v_cvt_pk_u8_f32`（字节选择）、`v_cvt_pk_{i16_i32,u16_u32}`（饱和）；20 个 MOp。真值来源：`tools/llvm_cvt_probe.ll`（llc gfx950 打印的汇编语法与 MIR 操作数，含 op_sel 位在源修饰符中的编码）。硬件探针 `probe_cvt_p1`（22 种情形，两个后端全部一致）**测得**：缩放量化是 value / scale（反量化 value · scale），`v_cvt_pk_u8_f32` 为四舍六入五成偶并钳位 0..255，i16 / u16 打包饱和 | **完成** |
| **B4 缓存策略** | `CachePolicy { sc0, sc1, nt }`（`CachePolicy::NT` 等），`load_global_c` / `store_global_c` / `buffer_load_c` / `buffer_store_c`；`MInst::cpol`（LLVM CPol 位 sc0 = 1、nt = 2、sc1 = 16），own 发射 `sc0 nt sc1`（与 LLVM 同序），LLVM 尾传 cpol 操作数；探针 `probe_cpol_p1` 两个后端一致 | **完成** |
| **A4 跨 lane** | `Dpp { ctrl: DppCtrl, row_mask, bank_mask, bound_zero }`（`DppCtrl`：quad_perm、row_shl / shr / ror 1..15、wave_shl / shr / rol / ror、row_mirror、row_half_mirror、row_bcast:15 / 31），`b.dpp(dpp, x, old)`（`v_mov_b32_dpp`）与 `b.dpp_alu(DppAluOp, dpp, a, b, old)`（`v_{add,sub,mul,max,min}_f32_dpp`、`v_add_u32_dpp`、`v_{max,min}_{i32,u32}_dpp`、`v_{and,or,xor}_b32_dpp`：一条指令完成 shuffle + 运算，未写的 lane 保留 `old`）；`b.permlane16_swap(a, b)`。硬件探针 `probe_dpp_p1`（20 种 DPP 情形 + permlane16）全部一致；**测得**：row_bcast 不覆盖的行读本 lane（不是"无效源"）。DPP 指令暂为 own-only（LLVM 尾无行，含 DPP 的 kernel 自动走 own 后端）；permlane16_swap 两个后端都有 | **完成** |

探针工具：`tests/gpu/tileop_probe_host.cpp`（通用 `kernel(in, out)` 一块 harness，输出预填 0xA5）、`tools/tileop_probe_check.py`（输入生成与 CPU 参考）、`tools/tileop_probe.sh <probe> [own|llvm-tail]`。
