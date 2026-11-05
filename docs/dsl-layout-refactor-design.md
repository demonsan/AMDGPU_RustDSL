# DSL 布局代数重构设计（"路线 B"）

> 状态：**已实施**（P1–P5 完成，出厂集为 v3 布局层 kernel；实施结果见 §9 As-built）。§0–§8 是评审时的设计稿（分支 `layout-design`，基于 Main `8fc4ca1`），保留原文供对照。作者：backend-llvm-tail。
> 用户决策："路线 B"——保留我们自己的 builder 前端 + 自研寄存器分配/调度/验证器，但按 NVIDIA cuda-oxide 的分层方式重组库；tile 层的设计参照 **FlyDSL**（AMD 上的 CuTe 代数）与 **aiter OPUS**（薄、编译期决议的 C++ 模板库）。
> 参考：cuda-oxide README（crate 划分：`cuda-device` / `ctm-rs` / `cute-layout` / `cute-rs` / `cuda-host` / `cuda-core`）；FlyDSL 仓库的 `layout_system_guide.md`、`kernel_authoring_guide.md`（已读原文）；aiter `csrc/include/opus/opus.hpp`（按 lead 给出的要点；未逐行通读）。
> 前作：`docs/dsl-tile-abstractions.md`（今天的 L1–L6 tile 库）、`docs/fmha-perf-patterns.md`（33 条性能模式）。

---

## 0. 结论先行

1. 新增一个**纯 Rust、零依赖、可在主机上单测**的布局代数 crate `amdgpu-layout`：IntTuple / Layout / Swizzle / 全套 CuTe 代数（与 FlyDSL 对齐）+ OPUS 式"线程维 × 发射维（issue space）"划分 + 偏移预计算与**静态部分折叠进指令立即数** + 一组构建期证明（覆盖、单射、每 lane 连续/向量宽度、tr 读合法性、MFMA 操作数布局匹配、swizzle 的 bank/相位性质）。
2. 把 "指令 + 操作数 TV 布局" 收拢成 **atom**（copy atom / MMA atom），放进扩展后的类型化指令层 `amdgpu-isa`（由 `amdgpu-safeasm` 演进而来）。
3. 今天的 `tile/` 拆成独立 crate `amdgpu-tile`，在 atom + layout 之上提供 FlyDSL 命名的 `TiledCopy` / `TiledMma` / `partition_*` / `copy` / `gemm`，**保留**我们已经验证过的 `LdsPlan` + `Pipeline` 协议（tile 标记与代际自动生成、barrier 节拍、角色错拍）。
4. 我们**不**复制 cuda-oxide 的"unsafe 边界由作者负责 barrier 相位/异步完成"这一点：完成/发布/代际仍由 `gpu_verify` 证明；布局证明是额外的、更早的检查，不减少验证器的工作，也不新增信任条目。
5. 主机侧（需要 FFI = unsafe）给出三个方案，**推荐先做 (b)：从 Rust 合同生成 C++ 启动/ABI 头文件**，零 unsafe；是否要纯 Rust HIP 运行时（方案 (a)，一个隔离、审计过的 `amdgpu-host-sys`）**请用户决定**。
6. 分 5 期迁移，每期有验收门；P3 门 = 现有 `port_gate`（性能与当前 dsl/legacy 相差 ≤3%、精度/确定性/压力、`--strict` 0 错误、不新增信任）。

---

## 1. 分层与 crate 地图

### 1.1 cuda-oxide 的分层 → 我们的对应

| cuda-oxide | 职责 | 我们（gfx950） | 说明 |
|---|---|---|---|
| `cuda-device` / `cuda-intrinsics` | 设备内建（thread、warp、barrier、TMA…） | `amdgpu-dsl`（IR `Op`/`Stmt` + `FnBuilder` + `expr.rs`） | **前端不变**：我们不做 rustc 后端，内建即 IR 操作 |
| `ctm-rs` | 类型化指令层 + ABI + launch contract + typestate（mbarrier、TmemGuard） | `amdgpu-isa`（由 `amdgpu-safeasm` 扩展） | atom 描述（指令 + 操作数 TV 布局 + 立即数范围）、寄存器 token、等待/DMA/MFMA typestate |
| `cute-layout` | 布局代数 + 编译期证明 | **`amdgpu-layout`（新）** | 纯 Rust，零依赖，主机单测；证明在"kernel 构建期"执行 |
| `cute-rs` | Tensor、zipped_divide、slice、local_tile、cooperative/tiled copy、生产者/消费者流水、tiled MMA | **`amdgpu-tile`（由 `amdgpu-tile/src/` 拆出）** | FlyDSL 命名的 TiledCopy/TiledMma + 我们的 LdsPlan/Pipeline 协议 |
| 闭世界导入器 + 全模块验证器 | rustc MIR → 方言；whole-module verifier | `amdgpu-compiler`（lower / sched / regalloc / hazards / `gpu_verify` / emit） | 不变；`gpu_verify` 本来就是全函数验证器 |
| `cuda-host` / `cuda-core` / `cuda-bindings` | 类型化启动、RAII、FFI | `amdgpu-host`（+ 可选 `amdgpu-host-sys`） | 见 §5，需用户决策 |

依赖方向（无环）：

```
amdgpu-layout  (纯代数，无依赖)
     ↑
amdgpu-isa     (atom 描述、typestate；依赖 layout)        amdgpu-dsl (IR + builder + expr)
     ↑                                                        ↑
     └──────────────── amdgpu-tile (依赖 layout + isa + dsl) ──┘
                              ↑
              amdgpu-kernels (fmha/ gemm/ mlp/ probes/ verifier_cases/ + 注册表)
                              ↑
              amdgpu-compiler (lower/RA/sched/gpu_verify/emit)   amdgpu-host (合同 → 主机)
                              ↑
                            amdc
```

`amdgpu-layout` 不依赖 `amdgpu-dsl`：它只产出自己的小型符号表达式 `IxExpr`（§2.1），由 `amdgpu-tile` 翻译成 `expr::I32`（运行值）与 `Ext`（合同）。这样代数与证明可以脱离 IR 单测，也可以被主机侧（合同检查、C++ 头生成）复用。

### 1.2 今天的代码搬到哪里

| 今天 | 去向 | 变化 |
|---|---|---|
| `amdgpu-dsl/src/expr.rs` | 留在 `amdgpu-dsl` | 不变（`IxExpr → I32` 的翻译在 tile 层） |
| `tile/tensor.rs`（KArgs、Dim、Stride、TensorDecl、TileView、clamp） | `amdgpu-tile::tensor` | `Tensor = ptr + Layout`（动态叶子 = kernarg/表达式）；合同由 `cosize(layout)` 与"紧致"关系生成，取代手写的 packed 规则；clamp 变成 copy 的谓词（§3.4） |
| `tile/shared.rs`（LdsPlan、Shared、Layout{Linear,Rows}、`fmha_swz`、SlotIdx/SlotRef/PrivRef） | `amdgpu-tile::shared` | `Shared` 持有一个 `ComposedLayout = Swizzle ∘ Layout`（取代 `Layout::FMHA_SWZ` 枚举）；LdsPlan、标记、代际、偏移缓存**原样保留** |
| `tile/copy.rs`（RowsCopy、DwordRowsCopy、DwordRowsVgpr、issue_rows_shared） | 被 `TiledCopy` 取代 | 手推的 lane 公式（`lane_hi/lane_lo/s_even/s_odd/rowb/M0 顶`）由 atom TV 布局 ∘ 源/目的布局**推导**；"共享行 clamp"变成同一 TiledCopy 作用于两个张量（谓词共享） |
| `tile/frag.rs`（布局标记类型、RowsReader/TrReader/RowScaleReader、qk_tile/pv_tile/qk_block、RowFrags、ScaleSel） | 读取器 → `TiledCopy`（S2R：`DsRead128x2`、`DsReadTrB8x4`、`DsRead32`）；MMA → `TiledMma` + `gemm` | `C32x32F32::row_const` 等手写映射变成 atom 的 `layout_C_tv` 查询 |
| `tile/pipeline.rs`、`tile/lane.rs` | `amdgpu-tile::pipeline`、`::lane` | 不变（这是我们相对 FlyDSL/OPUS 的增值：节拍 + 协议标记自动化） |
| `tile/reduce.rs`、`softmax.rs`、`epilogue.rs`、`components.rs`、`fmha.rs` | `amdgpu-tile::{reduce,softmax,epilogue,components,fmha}` | 改为基于"带布局的 fragment"：KeyMask 由 `layout_C_tv.idx2crd` 推出；epilogue 用 `make_tiled_copy_C` |
| `amdgpu-safeasm`（Seq、Vgpr<N>、Ready、Pending、Mfma token、Pw） | `amdgpu-isa`（重命名或作为其子模块） | 新增 atom 描述表、等待队列/DMA typestate（§4） |
| `kernels.rs`、`fmha_tile.rs` | `amdgpu-kernels` | 已完成（2026-11 整理）：探针在 `probes/`，`fmha_tile.rs` 与旧 builder 已删除；实际 crate 布局见 [`current-status.md`](current-status.md) §3 |

---

## 2. `amdgpu-layout`：布局代数

### 2.1 静态与动态如何在 Rust 里表示

关键观察：**我们的 DSL 是生成器**——kernel builder 本身就是在主机上运行的 Rust 程序。CuTe/FlyDSL 里"编译期常量 vs 运行期 SSA 值"的区分，在我们这里对应"**构建期 Rust 值** vs **GPU 运行期值**"。所以：

* 不需要用类型层编码静态整数（Rust 1.74 的 const generics 无法表达嵌套元组与表达式，`generic_const_exprs` 也不稳定）。静态性是**值层面**的属性：`Int::S(i64)`。
* 动态叶子 `Int::D(Dyn)` 持有一个符号 id；`Dyn` 在符号表里登记三样东西：构建期已知的**区间** `[lo, hi]`、**位支撑**（可能为 1 的位掩码，用于 §2.5 的 XOR/ADD 判定）、以及（在 tile 层）它对应的 `expr::I32` 运行值与 `Ext` 合同表达式。

```rust
pub enum Int { S(i64), D(DynId) }                    // 叶子
pub enum IntTuple { Leaf(Int), Tup(Vec<IntTuple>) }   // 可嵌套：(8, (4, 2))
pub struct Layout { pub shape: IntTuple, pub stride: IntTuple }   // 同构树
pub struct DynTable { syms: Vec<DynInfo> }            // DynInfo { name, lo, hi, bits: u64 }
/// 布局计算的结果：小型符号表达式（常量折叠、规范化成 "静态常数 + Σ 动态项"）
pub enum IxExpr { C(i64), Sym(DynId), Add(..), Mul(..), Div(..), Mod(..), Xor(..), And(..), Shl(..), Shr(..), Select(..) }
```

* 全静态的小布局（atom 的 TV 布局、swizzle）另提供 `const fn` 构造 + `const _: () = assert!(...)` 自检：atom 表写错在 **rustc 编译期**报 E0080（与 safeasm 的 `Imm<K>` 同一手法）。
* 规则（与 CuTe 实践一致）：**tile 内部的模式必须静态**（divide/complement/product 只在静态模式上定义完整），动态只允许出现在最外层"rest"模式（批、头、序列方向的 tile 计数）与 stride 上。违反时构建期 panic，信息指出哪个模式。

### 2.2 代数（FlyDSL 对齐）

| 类别 | API | 备注 |
|---|---|---|
| 构造 | `make_shape`、`make_stride`、`make_layout(shape, stride)`、`make_ordered_layout(shape, order)`、`make_coord` | |
| 映射 | `crd2idx(crd, L) -> IxExpr`、`idx2crd(idx, L)` | 静态输入 → 常数；动态 → IxExpr |
| 查询 | `size`、`cosize`、`rank`、`depth`、`get(i)`、`get_shape/get_stride` | `cosize` 生成合同里的分配大小 |
| 代数 | `composition(A, B)`、`complement(L, n)`、`coalesce`、`right_inverse`、`left_inverse` | `right_inverse` 用于"值 → 线程"反查（谓词、epilogue 写回） |
| 乘积 | `logical_product`、`raked_product`、`blocked_product` | （`zipped/tiled/flat_product` 无使用者，2026-11 删除） |
| 除法 | `logical_divide`、`logical_divide_bymode`、`zipped_divide_bymode`、`local_tile(T, tiler, coord)`、`local_partition` | `local_tile` = `zipped_divide_bymode` + 取 rest 坐标（cute-rs 名）；无使用者的 `zipped/tiled/flat_divide` 变体已删除 |
| 结构 | `select`、`group(b, e)`、`append`、`prepend`、`slice(L, crd)`（`None` 占位 = `_`）、`dice` | |
| 宽度 | `recast_layout(L, old_bits, new_bits)` | fp8(8) / f16,bf16(16) / f32(32) / dword 打包；MFMA 的 fp8 操作数按字节布局、按 dword 寄存器发射 |
| IntTuple 算术 | `add/sub/mul/div/ceil_div/product/product_each` | |

全部是纯函数，算法照搬 CuTe/FlyDSL 的 `LayoutUtils`（Apache-2.0 思想，自行实现；**不引入第三方依赖**——仓库的 `no_third_party_dependencies` 测试继续成立）。

### 2.3 Swizzle：把 FMHA 的 XOR swizzle 写成 `Swizzle ∘ Layout`

CuTe 的 `Swizzle<B, M, S>` 是"把位段 `[M+S, M+S+B)` 异或到 `[M, M+B)`"。我们 FMHA 的 swizzle（`docs/archive/status.md` 第 801 行蛮力搜索的结果）是：128 字节一行，16 字节块号 `c`（字节偏移的位 4..6）异或上 `s(r)`，其中 `r` 是行号（位 7..10），

```
p = r0;  u = (r >> 1) & 7 = (r1, r2, r3)
s2 = u2 ^ p = r3 ^ r0,   s1 = u0 = r1,   s0 = u1 ^ p = r2 ^ r0
```

它**不是**单一的 `Swizzle<B,M,S>`（那只能是 `c ^= r & 7` 一类的移位异或），但它是 GF(2) 上的**线性**变换。所以我们把 Swizzle 定义成一般形式，CuTe 的三参数形式是特例：

```rust
pub struct Swizzle {
    pub src: BitRange,       // 读哪些位（FMHA：字节偏移位 7..10 = 行号低 4 位）
    pub dst: BitRange,       // 异或到哪些位（FMHA：位 4..6 = 16 字节块号）
    pub m: [u16; 16],        // dst 的第 i 位 = ⊕_j m[i]_j · src_j  （GF(2) 矩阵；CuTe Swizzle = 单位矩阵）
}
impl Swizzle {
    pub const fn cute(b: u8, m: u8, s: u8) -> Swizzle;          // FlyDSL 的 Swizzle<B,M,S>
    // 行 m[i] 是 dst 位 i 依赖的 src 位掩码（src 位 j = 行号位 r_j）：c0 ^= r2^r0, c1 ^= r1, c2 ^= r3^r0
    pub const FMHA128: Swizzle = Swizzle { src: bits(7, 4), dst: bits(4, 3), m: [0b0101, 0b0010, 0b1001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] };
}
let k_smem = ComposedLayout::new(Swizzle::FMHA128, make_layout((128, 128), (128, 1)));   // [key, d] 字节
```

证明（构建期，§2.6）：`src ∩ dst = ∅` ⇒ 双射（一定）；对给定访问模式（atom 的 TV 布局）枚举 bank 冲突——把 `status.md` 里"两个 bank 模型下 K 读、tr 读、写入都 0 冲突"的结论变成单测，换 swizzle 时自动复验。

### 2.4 OPUS 式发射空间（issue space）与偏移预计算

OPUS 的 `layout<Shape, Stride, Coord>` 把形状分成**线程维 p**（由 lane/wave 决定）和**每线程发射维 y**（同一线程依次发出的若干条指令/若干个向量），`layout_to_offsets<vec>` 预先算出每次发射的偏移；静态布局时偏移直接成为指令立即数（`layout_imm_offsets`、`layout + number<N>` 平移布局）。哲学是：**作者保留调度权；每个 tile 操作发多少条指令，由 issue space 静态可知。**

我们的对应物是 `TvLayout`（thread-value 布局）的一个**分解视图**：

```rust
pub struct IssuePlan {
    pub thr: Layout,          // (lane, wave, ...) → 偏移的"线程部分"           —— 动态，kernel 开头算一次（lane 不变量）
    pub issue: Layout,        // (i0, i1, ...)  → 偏移的"发射部分"           —— 静态，逐条展开
    pub vec_bytes: u32,       // 每次发射的向量宽度（由 atom 决定：16/8/4 字节）
}
pub struct Offsets {          // layout_to_offsets 的结果
    pub bases: Vec<IxExpr>,   // 需要的不同基址寄存器（线程部分 + 无法折叠的静态部分）
    pub per_issue: Vec<(usize /*base 下标*/, i32 /*立即数*/)>,
}
pub fn layout_to_offsets(p: &IssuePlan, atom: &AtomImm) -> Offsets;   // AtomImm = 立即数字段的范围与对齐
```

**静态部分折叠进立即数的规则**（这是我们 kernel-opt 已测量过的性能模式："slot offset 折进 ds_read 立即数"）：

1. 偏移 = `线程部分(lane) ⊕/＋ 静态部分(i) ＋ 每 tile 部分(slot, it)`。
2. 布局是线性（只有 `+`、`*常数`）时：静态部分直接进立即数，只要落在 atom 的立即数范围内（`ds_read/ds_write`：0..65535 字节；`global_load/store`：gfx950 有符号 13 位 -4096..4095；`buffer_load/store`：0..4095；`global_load_lds`：同 global；超出范围时按范围切分成多个基址）。
3. 有 swizzle（XOR）时：`(x ⊕ c)` 只有在 **c 的位与 x 的可能位不相交**时才等于 `x + c`。`amdgpu-layout` 用 `DynInfo.bits`（位支撑）判定：
   * FMHA K 读：`kfx + 4096·s`——`4096·s` 改的是行号的位 5、6（字节位 12、13），而 swizzle 只读行号位 0..3 → 与线程部分的位支撑不相交 → **可折叠**（今天的 `RowsSlot::read_pair` 还在用 4 条 `v_add` 做这件事，这是一个现成的收益点）。
   * `chunk ^ 2k`：静态位（块号位 1）与 swizzle 后线程部分的位重叠 → **不可折叠** → 需要 4 个基址（正是手写的 `kfx[0..3]`）。
   代数自动得出与手写 kernel 相同的基址个数，并在能折叠的地方比手写更省。
4. 每 tile 部分（slot·stride）：沿用今天的 `SlotIdx` 绑定缓存（`Pipeline.prebind`），它是"一个 SGPR/VGPR 加数"，不进立即数（运行时值）。

前提（IR 扩展，P2 内做）：`Op::LoadLdsB128 / LoadLdsB128x2 / LoadLdsTrB8x4 / StoreLds*` 加 `imm: u16` 字段，lowering 把它发到 `offset:`。`global_load`/`store` 已有 `byte_off`。这会改变 asm（不是 bit 级不变），由 P3 的 port_gate 把关。

### 2.5 构建期证明（cute-layout 风格）

全部在 kernel 构建期执行（= GPU 程序的编译期），失败即构建失败，消息指向布局与具体坐标；每条都有主机单测。

| 证明 | 内容 | 用在哪里 |
|---|---|---|
| **覆盖（coverage）** | TiledCopy 的 TV 布局在 tile 上是满射：每个元素恰好被一个 (线程, 值) 搬运；部分覆盖必须显式声明（例如 KD 的"波 2、3 重复波 0、1"） | 所有 copy；对应验证器的 `tile-cover` |
| **单射（injectivity）** | TV 布局 (t, v) → 元素 是单射（没有两个线程写同一字节），重复写只允许显式 `broadcast` | 写 LDS / 写全局 |
| **每 lane 连续 / 向量宽度** | 每个线程每次发射的 `vec_bytes` 字节在目的（和源）上连续且按宽度对齐 → 可以用 dwordx4 / b128 | 选 atom 宽度；与 OPUS 的 `vec` 参数一致 |
| **tr 读合法性** | `ds_read_b64_tr_b8` 的源：8 字节对齐；同一 8 lane 组的 8 个地址按 atom 要求的行间距排布；转置后得到的寄存器布局 = 目标 MMA 操作数的 A/B 布局 | V^T 读取 |
| **MFMA 操作数匹配** | `make_tiled_copy_A(atom, mma)` 产生的寄存器 TV 布局与 `mma.layout_A_tv` 相等（考虑 K 置换）；手工给的 fragment 布局不匹配 → 构建错误。今天这只靠 `A32x32x64Fp8` 之类的标记类型（只能防"喂错槽"，不能防"lane 映射错"） | QK/PV |
| **swizzle 性质** | 双射；在给定 atom 访问模式下按 gfx950 LDS bank 模型（64 bank × 4B，b128 每 16 lane 一拍、b64 每 32 lane）冲突数 = 0（或 ≤ 声明值）；相位：DMA 写入时的源侧 swizzle 与读侧使用同一 `Swizzle` 对象 | K/V 环 |
| **范围** | `cosize(dst) ≤ slot 大小`；立即数在范围内；动态部分的区间（来自 `DynInfo`）× stride ≤ 合同分配 | 预检，失败比验证器早、信息更好 |

**如何喂给 `gpu_verify`（安全不退化）**：

* 布局证明是**冗余的前置检查**，不是信任来源：验证器仍在 IR 上独立证明每个访问的界、每个读的代际、每个 barrier 的平衡。证书账本不新增条目。
* 生成物不变：TiledCopy 作用于 `Shared` 张量时，照今天 `SlotRef::read/write` 的方式输出 `tile_produce/consume(name, slot, gen)` + `tile_end`；全局张量的合同（`arg_tensor` = `cosize × elem`、紧致关系 `ka_is`）由 Tensor 的 Layout 生成。
* 一个可选的增强（P3 之后再定）：让 TiledCopy 把"本次访问的字节区间 = 布局像"作为提示随标记输出，验证器用它**交叉检查**自己的区间分析（不一致 = 报错）。仍然不是信任。

---

## 3. gfx950 atom 与 TiledCopy / TiledMma

### 3.1 Copy atom

atom = （指令, 每线程向量宽度, 源 TV 布局, 目的 TV 布局, 立即数字段, 完成计数器）。完成计数器决定 typestate/等待（`vmcnt` / `lgkmcnt`）。

| atom | 指令 | 宽度/线程 | 方向 | 完成 | 备注 |
|---|---|---|---|---|---|
| `GlobalLoad128/64/32` | `global_load_dwordx4/x2/dword` | 16/8/4 B | G→R | vm | Q 片段（今天 2×b128/64B 组） |
| `GlobalStore128/64/32` | `global_store_dwordx4/x2/dword` | 16/8/4 B | R→G | vm | O（dwordx2，64e+16q）、Ohat |
| `GlobalLds128/32` | `global_load_lds_dwordx4/dword` | 16/4 B，**目的 = M0 + imm + 16·lane（线性）** | G→S | vm | DMA：目的布局固定为线性，所以 swizzle 只能施加在**源**上（atom 的 dst TV 布局是常量，TiledCopy 由 `right_inverse` 自动把 Shared 的 swizzle 挪到源侧——正是手写 `cbytes_even/odd` 的推导） |
| `BufferLds128/32` | `buffer_load_dword[x4] ... lds` | 同上 | G→S | vm | 带描述符 + `num_records` 越界清零（OPUS `gmem::async_load`）；谓词可省（§3.4） |
| `BufferLoad/Store128/64/32` | `buffer_load/store_dword*` | | G↔R | vm | OPUS `gmem<T>` |
| `DsRead128/64/32`、`DsRead128x2` | `ds_read_b128/b64/b32`、两条 b128 拼 Frag8 | 16/8/4 B | S→R | lgkm | K 片段（kperm：块 `h`, `h+2`） |
| `DsReadTrB8x4` | 4× `ds_read_b64_tr_b8` | 8 B ×4 | S→R（转置） | lgkm | V^T：寄存器 TV 布局 = MFMA A 布局；源合法性见 §2.5 |
| `DsWrite128/64/32` | `ds_write_b128/b64/b32` | | R→S | lgkm | Q/ones 暂存、KD（VGPR 路径）、epilogue 转置 |

`tr_b8` 的精确 lane 分组（哪 8 个 lane 组成一个 8×8 字节转置块）以仓库里的硬件探针 `probe_lds_tr_b8` 为准写成 atom 常量表，并加一个"atom 表 vs 探针 dump"的回归测试。

### 3.2 MMA atom

| atom | 指令 | 形状 | A/B/C TV 布局 | 用途 |
|---|---|---|---|---|
| `MfmaScale32x32x64F8` | `v_mfma_scale_f32_32x32x64_f8f6f4` | M32 N32 K64，A/B 各 8 dword，C 16 dword，E8M0 缩放 | C：lane l、值 r → (行 `8(r>>2)+4(l>>5)+(r&3)`, 列 `l&31`)；A/B：行 `l&31`，K 段由 `l>>5` 选择 | QK、PV |
| `MfmaScale16x16x128F8` | `v_mfma_scale_f32_16x16x128_f8f6f4` | M16 N16 K128 | 按 ISA 表 | 备用（P4 GEMM 比较） |
| `MfmaF16_32x32x8` | `v_mfma_f32_32x32x8_f16`（今天 `mfma_f16_wide`） | | | FMHA 的"ones ⊗ bias"参考偏移（每 32 key 一条，perf-patterns P"保留 4 条 f16 bias MFMA"） |
| `SmfmacF8_16x16x128` | `v_smfmac_f32_16x16x128_fp8_fp8` | 稀疏，idx 字 | A = 每 lane 的"ones"模式（今天 `SmfmacOnes`），dword 2 = 16× 行和 | 行和 |

**K 置换**：FMHA 让 A、B 用同一 K 置换（lane 半 h 读 64B 组里的 `16h..` 与 `32+16h..`）——这就是 FlyDSL `make_tiled_mma(atom, atom_layout, permutation)` 的 permutation 参数。我们把它做成 `KPerm` 布局，`make_tiled_copy_A/B` 自动把它作用到读取侧（今天的 `RowsReader` 与 `RowFrags::load` 各自手写了一遍）。缩放字节的选择（`ScaleSel`：`>> 8h`、`>> 8h+16`）成为 atom 的"scale 操作数布局"，类似 FlyDSL 的 `[frag_A, scales_A]` 分组。

### 3.3 TiledCopy / TiledMma / partition API（Rust，FlyDSL 命名）

```rust
// 构造（构建期值，证明在这里执行）
let tv = make_layout_tv(thr_layout, val_layout);                       // (tiler_mn, layout_tv)
let g2s = make_tiled_copy(GlobalLds128::atom(), tv, tile_mn);          // TiledCopy
let mma = make_tiled_mma(MfmaScale32x32x64F8::atom(), atom_layout, KPerm::fmha());
let s2r_a = make_tiled_copy_A(DsRead128x2::atom(), &mma);              // 与 mma 的 A 划分配套
let s2r_vt = make_tiled_copy_A(DsReadTrB8x4::atom(), &mma);
let r2g_c = make_tiled_copy_C(GlobalStore64::atom(), &mma);

// 每线程视图
let thr = g2s.get_slice(lane_id);           // ThrCopy：线程部分在这里一次性 bind（lane 不变量）
let src = thr.partition_S(&g_tile);         // 发射维 = 静态，线程维 = 已 bind 的值
let dst = thr.partition_D(&s_slot);
copy(b, &g2s, &src, &dst, pred);            // 展开成 issue 次指令；dst 是 Shared 槽 → 自动 tile_produce

let tm = mma.thr_slice(lane_id);
let a = tm.make_fragment_A(&s_k);  let bq = tm.make_fragment_B(&q_regs);  let c = tm.make_fragment_C(&s_acc);
gemm(b, &mma, &c, &a, &bq, &c, Scales(..));
```

层次 block → wave → thread → instruction 保持显式（FlyDSL 的做法）：`atom_layout` 描述 wave 在 tile 上的排布，`thr_layout` 描述 lane，`issue` 维描述单线程的指令序列。**作者保留调度**（OPUS 哲学）：`copy`/`gemm` 按发射维的字典序原样展开，不重排；顺序要改就改布局或拆成多次调用（例如 `_t128` 的"每 32 key 块交错"就是对 `gemm` 的 M 维切片后逐块调用）。

### 3.4 谓词（尾块）

OPUS 的 `load_if/store_if` 对发射空间加谓词；FlyDSL `copy(..., pred=)`。我们支持三种尾块策略，作为 TiledCopy 的参数：

| 策略 | 生成 | 今天的对应 |
|---|---|---|
| `Clamp`（默认） | 越界行索引替换为最后一个合法行（`select`/`min`），读重复数据、计算后丢弃 | 今天 K/V/KD DMA、Q 行、O 行 |
| `Pred` | `begin_if(谓词)` 包住该次发射（波级一致时），或对 lane 级谓词用 `select` 把地址指到安全位置 + 写 0 | epilogue 的 wave_live |
| `HwBound` | buffer 描述符 `num_records` = 合法字节数，硬件清零（`mubufd` 已在硬件上验证 LDS 形式也清零） | 今天是 opt-in |

谓词由 `identity_layout` 经同一 TiledCopy 划分得到坐标张量，再与形状比较（CuTe 惯用法），所以"哪些发射需要谓词"也是静态可知的：完整 tile 的发射不带谓词。

### 3.5 Shared 张量上的 copy 自动产生标记与代际

`Shared`（来自 `LdsPlan`）+ `SlotIdx`（来自 `Pipeline` 的 `TileIt`）→ `s_slot = ks.at(&t.next).tensor()`。`copy(..., dst = s_slot)` 输出 `tile_produce(K, slot, gen)`…`tile_end`；`partition_A(s_slot)` 的读输出 `tile_consume`。私有区（`lds_private`）的 `PrivRef` 同理，wave key 由 `get_slice` 的 wave 值提供（满足"地址由同一个 key 派生"的验证器规则）。**节拍（waits/barriers/角色错拍/尾块剥离）仍由 `Pipeline` 负责**——这是 FlyDSL/OPUS 都留给作者手写、而我们已经自动化并被验证器证明的部分。

---

## 4. 类型化指令层 `amdgpu-isa`：哪些进 typestate，哪些留给 gpu_verify

| 性质 | 放在哪里 | 理由 |
|---|---|---|
| 寄存器宽度、立即数范围、对齐、MFMA 操作数在途不可覆盖（`Ready<'a,N>`、`Mfma<'a,OP>`）、加载结果在 wait 前不可读（`Pending<T>`） | **已在 safeasm**，迁入 `amdgpu-isa` | rustc 编译期错误（E0308/E0382/E0502/E0080） |
| atom 表（指令 + TV 布局 + 立即数字段 + 完成计数器） | `amdgpu-isa::atoms`（`const` 表 + `const` 断言） | 写错 atom 表 = rustc 编译期错误 |
| **等待队列**：`vmcnt`/`lgkmcnt` 是按发射顺序递减的计数器；`Pending` token 带"队列位置"，`wait_vm(n)` 消费所有比第 n 新的更老的 token | `amdgpu-isa`（仅在 safe-asm 窗口内） | 窗口内是直线代码，typestate 可行且精确 |
| **DMA 在途**：`copy` 到 Shared 返回 `InFlight<SlotWrite>`；必须先 `complete(wait)` 才能交给 `publish(barrier)` | `amdgpu-tile`：Rust move 防"重复完成/完成前发布"；"忘了完成"无法由类型系统阻止（Rust 无线性类型）→ `Drop` 记录 + 构建期 panic（同 `Seq::finish`） | 双保险；最终裁决仍在验证器 |
| **barrier 相位/代际**、跨波发布、角色间 barrier 数相等、读完才覆写、循环中的周期性 | **留在 `gpu_verify`**（协议模拟 + 周期证明，含已加入的有限状态环计数器） | 跨分支、跨迭代、跨波的性质无法用局部 typestate 表达；cuda-oxide 在这里把责任交给作者（unsafe），我们**不这样做** |
| 路径敏感的"用前等待"、寄存器合法性 | 留在 MIR 级 `gpu_verify` | 依赖 RA 结果 |
| launch contract（grid、kernarg 关系、分配大小） | `Function.contract`（已有）→ 主机侧检查（§5） | |

---

## 5. 主机侧：类型化启动与 launch contract

现状：kernel 的合同（`arg_tensor`、`ka_is`、`ka_range`、`grid_is`）只给验证器用；主机端 `tests/gpu/ref_host.cpp` 手写 kernarg 打包与形状，靠 `contract_host_truth.rs` 对账。

约束：所有 crate `#![forbid(unsafe_code)]`；Rust 调 HIP 运行时必须 FFI（unsafe）。

| 方案 | 内容 | 优点 | 代价/风险 |
|---|---|---|---|
| **(a) 隔离的 `amdgpu-host-sys`** | 唯一允许 `unsafe` 的 crate（`#![deny(unsafe_op_in_unsafe_fn)]`，约 300–500 行：`hipModuleLoadData`/`hipModuleGetFunction`/`hipModuleLaunchKernel`/`hipMalloc`/`hipMemcpy`，手写 `extern "C"`，不用 bindgen 以保持零第三方依赖）+ 安全封装 `amdgpu-host`：`DeviceBuffer<T>`（RAII）、由 `Function.contract` 生成的 `PreparedLaunch`（启动前在主机上求值 `Ext`：分配大小 ≥ `arg_tensor`、`ka_is` 成立、`grid_is` 成立，不成立返回 `Err`） | 端到端纯 Rust；合同在启动时被强制（cuda-oxide `#[launch_contract]` 的对应物） | 打破"全部 crate forbid(unsafe)"这一项（需在 `rust_safety.rs` 里登记唯一例外 + 审计清单）；链接 ROCm 运行时；CI 需 GPU 节点 |
| **(b) 生成 C++ 头（推荐先做）** | `amdc emit-host-abi <kernel>` 从 `Function.args` + `contract` 生成 `kernel_abi.h`：packed kernarg 结构体（`static_assert(offsetof(...))`）、`check_launch(const Args&, dim3 grid, alloc sizes)`（把 `Ext` 翻译成 C++ 表达式）、grid 计算辅助；`ref_host.cpp`、aiter 调用方 include 它 | **零 unsafe**；立刻消除手写 kernarg 偏移这一类错误；与现有门禁/harness 完全兼容；`contract_host_truth.rs` 从"对账"变成"生成" | 主机仍是 C++；合同检查依赖调用方真的调用 `check_launch` |
| (c) 元数据 + 外部运行时 | 生成 JSON 元数据（参数、合同、grid 公式），由 Python（hip-python / PyTorch 扩展）或 aiter 的现有加载器使用 | 便于集成到推理框架 | 第三套实现；合同检查在 Python 侧 |

**建议**：P5 先做 (b)（1–3 人日，收益确定），同时输出 (c) 的 JSON（几乎免费）；(a) 只有在需要"纯 Rust 端到端产品"时才做，并作为唯一的 unsafe 例外单独审计。**这是需要用户拍板的决定（D-host）。**

---

## 6. 迁移计划

| 期 | 内容 | 验收门 | 估计（人日） |
|---|---|---|---|
| **P1** | `amdgpu-layout`：IntTuple/Layout/Swizzle（含 GF(2) 形式与 `FMHA128`）、全套代数、`IssuePlan`/`layout_to_offsets`（含 XOR/ADD 折叠判定）、`recast_layout`、§2.5 全部证明 | 主机单测：与 FlyDSL/CuTe 文档示例逐条对照（composition/complement/divide/product 的已知结果表）；`FMHA128` 复现 `fmha_swz` 的 2048 个 (行, 块) 映射；bank 冲突单测复现 status.md 结论；构建期证明的负例（覆盖缺口、非单射、tr 不对齐、MFMA 布局不匹配）各至少一个 | 6–9 |
| **P2** | `amdgpu-isa` atom 表（§3.1/3.2，`const` 自检）；`amdgpu-tile` 的 `TiledCopy`/`TiledMma`/`partition_*`/`copy`/`gemm`/谓词；IR 扩展：LDS 操作的 `imm` 字段（lowering + emit + hazards + 验证器的地址模型）；`tr_b8` atom vs 硬件探针回归 | 每个 atom 一个 demo kernel：`--strict` 0 错误、asm 指令计数精确；`TiledCopy` 生成的 K/V DMA、K 读、V^T 读与今天 `RowsCopy`/`RowsReader`/`TrReader` 的地址**在 64 lane × 全部发射上数值相等**（主机求值）；ds 立即数折叠的单测 | 8–12 |
| **P3** | 在新层上重写 `tile/`（tensor/shared/copy/frag 改为基于 layout；pipeline/softmax/epilogue 保留接口），重新移植 4 个 FMHA kernel | **现有 `port_gate`**：性能与当前 dsl/legacy 相差 ≤3%（含 layout 中位数行，±1.5% 的代码布局噪声已知）；15 形状 / 64 行 bad=0 / 确定性 / split 重复启动压力；`gpu-verify --shipped --strict` 0 错误；证书不新增信任条目；`cargo test` 绿 | 8–12 |
| **P4** | **只用新层**写一个新 kernel 证明通用性：MXFP8 GEMM（`v_mfma_scale_f32_32x32x64` 或 16x16x128，DMA 双缓冲环，bf16 输出），对比 hipBLASLt / aiter（若节点上可用） | 正确性对 CPU 参考；`--strict` 0 错误；kernel 源 ≤ 150 行；性能报告（不设硬门，给出与库的差距与原因） | 6–10 |
| **P5** | 主机侧（按用户决定）：(b) C++ ABI 头 + JSON；或 (a) host-sys | (b)：`ref_host.cpp` 改为 include 生成头，全部 GPU 门禁不变；`check_launch` 负例测试 | (b) 2–3；(a) 6–10 |

合计约 30–45 人日（不含 (a)）。P1、P2 可与现网 kernel 工作并行；P3 是唯一会动出厂 kernel 的一期，且新旧 kernel 并存直到门禁通过（与上一轮 `dsl_*` 的做法相同）。

### 6.1 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **性能对等**（最大） | 上一轮已测到：循环变量顺序 ≈2.3%、slot 偏移绑定位置 ≈0.3%、迟写 bias 决定调度器能否 fit（494 vs 410 TF/s）、代码布局 ±1.5%。新层改变的是**指令生成顺序与值的绑定位置**，RA/调度对此敏感 | `copy`/`gemm` 不重排、按发射维字典序展开；lane 不变量在 `get_slice` 一次绑定（与今天 `RowsReader::new` 同位置）；保留 `Pipeline.prebind`/`ring_var_last` 旋钮；P2 的"地址数值相等"测试 + P3 的 asm 直方图对比 |
| **立即数折叠** | 需要 IR 扩展；折叠改变寄存器压力（少了 add，也少了活跃值）与 asm | P2 单独开关（`fold_imm`），P3 A/B 测量后再默认打开 |
| **构建时间** | 证明是枚举式的（64 lane × 发射数 × 每 tile 元素），FMHA 规模约 10^5 次求值，可接受；动态布局的代数要做符号化简 | 只对静态部分枚举；动态部分用区间/位支撑 |
| **动态形状的代数** | complement/divide 在动态形状上需要整除事实 | 规则：tile 内部模式必须静态（§2.1）；动态只在最外层 |
| **tr_b8 / smfmac 等特殊语义** | ISA 文档不足 | 以硬件探针为准，atom 表 vs 探针 dump 回归 |
| **两套 tile 库并存期** | 维护成本 | P3 通过后删除旧 `tile/copy.rs`/`frag.rs` 读取器，旧 `kernels.rs` builder 保留为对照（已是如此） |

### 6.2 刻意不照搬的东西

* cuda-oxide 的 **unsafe 边界**（"barrier 相位、异步完成由作者负责"）：我们继续由 `gpu_verify` 证明完成/发布/代际/barrier 平衡（§4）。
* cuda-oxide 的 **rustc 后端 + 闭世界 MIR 导入**、FlyDSL 的 **MLIR + Python AST 重写**：我们保留 builder 前端与自研后端（路线 B 的前提）。
* FlyDSL/OPUS 的"调度/同步全手写"：我们保留 `Pipeline` 的节拍自动化与协议标记。
* 第三方依赖：代数自行实现，零依赖策略不变。

---

## 7. 对照示例：dsl_vs128 的 QK/PV 与 K/V 暂存

### 7.1 当时（已删除的 `fmha_tile.rs` + `tile::fmha::KvRing` + `tile::frag`，节选，历史）

```rust
// KvRing::new —— LDS 计划 + 读取器 + 拷贝，每个读取器/拷贝内部是手推的 lane 公式
let [ks, vs] = lds.ring_group(slots, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
let kd = lds.ring("KD", 512, slots, Layout::Linear);
lds.commit(b);
let kr = RowsReader::new(b, &ks, lane);     // kfx[k] = j*128 + ((h ^ s(j)) ^ 2k) << 4      —— 手推
let vr = TrReader::new(b, &vs, lane);       // trx[e] = key_part(l)*128 + 8(l&1) + ((m_gl ^ s(key_part)) ^ 2e) << 4 —— 手推
let kdr = RowScaleReader::new(b, &kd, lane);
let ksrc = blk.kv_tile(b, &io.k).row_source(b, 1);
let kcp = RowsCopy::new(b, ksrc, &ks, roles.wave_in_group, lane, 4);  // lane_hi/lane_lo/s_even/s_odd/rowb/M0 顶 —— 手推
let vcp = RowsCopy::new(b, vsrc, &vs, roles.wave_in_group, lane, 4);
let kdc = KdCopy::Dma(DwordRowsCopy::new(b, kdsrc, &kd, roles.wave_in_group, lane));

// AttnBody —— QK / PV
fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, out: QkOut, bias: ValueId) -> Vec<ValueId> {
    let k = self.ring.kr.at(b, &self.ring.ks.at(slot));          // 4 个基址 + slot 偏移
    let kd = self.ring.kdr.at(b, &self.ring.kd.at(slot));
    qk_tile(b, out, Some((self.sm.ones_f16, bias)), &k, &kd, &self.blk.q.frags, self.blk.sq, &self.blk.sel)
    //   └ 每个 s：bias MFMA；4 次 v_add(kfb[k] + 4096 s)；2×ds_read_b128x2；KD 读；ScaleSel；2 条 scaled MFMA
}
fn pv(&self, b: &mut FnBuilder, slot: &SlotIdx, w: &[ValueId]) {
    let v = self.ring.vr.at(b, &self.ring.vs.at(slot));
    let p = [Frag::<B32x32x64Fp8>::from_words(b, &w[0..8]), Frag::<B32x32x64Fp8>::from_words(b, &w[8..16])];
    self.sm.row_sum(b, p[0].v, p[1].v);
    pv_tile(b, &self.o.unwrap(), &v, &p, self.c127, self.c123);   // 每 (hh,e)：4 次 v_add + 4×ds_read_b64_tr_b8 + MFMA
}
fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
    self.kd_pending = self.ring.stage(b, role, key0, slot);       // A：KD + K 的 DMA；B：V 的 DMA
}
```

问题不在行数，而在 `RowsReader`/`TrReader`/`RowsCopy`/`ScaleSel`/`RowFrags` 各自**手推了一遍同一套布局事实**（swizzle、K 置换、C 布局、lane 半），彼此之间、与 MFMA 的硬件布局之间没有机器检查的一致性；换一种 tile 形状或 MFMA（16x16x128）就要重推。

### 7.2 新层（同一语义，布局推导 + 证明）

```rust
// ---- 构建期：布局与 atom（全部证明在这里执行）----
let swz = Swizzle::FMHA128;                                                          // GF(2) 形式的 s(r)
let kv_smem = ComposedLayout::new(swz, make_layout((128, 128), (128, 1)));           // [key, d] 字节，K 与 V 同布局
let mma = make_tiled_mma(MfmaScale32x32x64F8::atom(), make_layout((1, 1), (0, 0)), KPerm::fmha());
// S^T(32 key × 32 query) = K(32 key × 64 d) · Q^T ;  O^T(32 d × 32 query) = V^T(32 d × 64 key) · P^T
let s2r_k  = make_tiled_copy_A(DsRead128x2::atom(), &mma);       // K 行：证明 → 寄存器布局 = mma.layout_A_tv（含 KPerm）
let s2r_vt = make_tiled_copy_A(DsReadTrB8x4::atom(), &mma);      // V^T：tr 读合法性 + 布局匹配
let g2s = make_tiled_copy(GlobalLds128::atom(),
    make_layout_tv(make_layout((4, 64), (64, 1)) /* 4 波 × 64 lane */, make_layout((1, 16), (0, 1)) /* 16 B */),
    (128, 128));                                                 // 覆盖 + 单射证明；swizzle 自动挪到源侧
let kd_g2s = make_tiled_copy(GlobalLds32::atom(), tv_rows_dup(128, 4, /*dup*/ 2), (128, 4));   // 波 2、3 显式重复

// ---- 每 kernel 一次：线程切片（lane 不变量在这里 bind，等价于今天 RowsReader::new 的位置）----
let (tk, tv, tg, tkd) = (s2r_k.get_slice(b, &lane), s2r_vt.get_slice(b, &lane), g2s.get_slice(b, &lane, wave4), kd_g2s.get_slice(b, &lane, wave4));
let tm = mma.thr_slice(b, &lane);
let q = tm.make_fragment_B(b, &io.q.local_tile((bidx, row_blk, hidx), (1, 256, 1, 128)));   // Q 常驻寄存器（2×b128/64B）

// ---- PipeBody ----
fn stage(&mut self, b: &mut FnBuilder, role: Role, key0: &I32, slot: &SlotIdx) {
    let kv = |t: &Tensor| t.local_tile(b, (blk.bidx, key0, blk.kvh), (1, 128, 1, 128)).pred(Pred::Clamp(blk.kv_len));
    match role {
        Role::A => { copy(b, &tkd, &tkd.partition_S(&kv(&io.ksc)), &tkd.partition_D(&kd.at(slot)));
                     copy(b, &tg,  &tg.partition_S(&kv(&io.k)),    &tg.partition_D(&ks.at(slot).view(&kv_smem))); }
        _       =>   copy(b, &tg,  &tg.partition_S(&kv(&io.v)),    &tg.partition_D(&vs.at(slot).view(&kv_smem))),
    }   // 每个 copy：tile_produce(slot, gen) + 4 条 DMA（立即数 = 发射维偏移）+ tile_end
}
fn qk(&self, b: &mut FnBuilder, slot: &SlotIdx, s_acc: &Fragment, bias: ValueId) {
    let a = tk.partition_A(&ks.at(slot).view(&kv_smem));          // (MMA, M=4, K=2)：4096·s 进 ds_read 立即数（§2.4 判定可折叠）
    let sa = tm.partition_scale_A(&kd.at(slot));                  // KD 字 + 字节选择（取代 ScaleSel）
    for s in 0..4 {                                               // 作者保留顺序：bias → 读 → 缩放 → 2 条 MFMA
        gemm_bias(b, &self.bias_mma, s_acc.slice(s), self.sm.ones_f16, bias);
        gemm(b, &mma, s_acc.slice(s), &copy_frag(b, &tk, a.slice((_, s, _))), &q, Scales(sa.slice(s), self.blk.sq));
    }
}
fn pv(&self, b: &mut FnBuilder, slot: &SlotIdx, p: &Fragment /* P 的 B 布局，由 to_p8 产生 */) {
    self.sm.row_sum(b, p);
    let vt = tv.partition_A(&vs.at(slot).view(&kv_smem.transpose()));   // V^T 视图 = 同一存储的转置布局
    gemm(b, &mma, &self.o, &copy_frag(b, &tv, &vt), p, Scales::fixed(127, 123));   // 2 × 4：tr 读 + MFMA，顺序同今天
}
```

差别：swizzle、K 置换、C 布局、lane 半、M0/负立即数补偿等事实**只写一次**（在 `Swizzle`、`KPerm`、atom 表里），读取器与拷贝由代数推导，并在构建期证明彼此一致、与 MFMA 硬件布局一致；换成 16x16x128 MFMA 或 64 行 tile，只改 `mma`/`tile_mn` 两行。

---

## 8. 需要决策的点（汇总）

| 编号 | 问题 | 选项 | 建议 |
|---|---|---|---|
| **D-host** | 主机侧要不要纯 Rust HIP 运行时 | (a) 隔离 unsafe 的 `amdgpu-host-sys` + 安全封装；(b) 从合同生成 C++ ABI 头（零 unsafe）；(c) JSON 元数据给 Python/aiter | **先 (b)+(c)**；(a) 视产品需要，作为唯一 unsafe 例外单独审计（**用户决定**） |
| D-crate | `tile/` 拆成独立 crate `amdgpu-tile`、safeasm 演进为 `amdgpu-isa`、kernel 移到 `amdgpu-kernels` | 拆 / 不拆（只建模块） | 拆：编译时间与依赖边界更清楚；`amdgpu-layout` 必须独立（纯、可单测、主机复用） |
| D-imm | LDS 操作加立即数字段（IR 扩展）并默认折叠 | 默认开 / 开关 / 不做 | P2 加开关，P3 A/B 后定默认 |
| D-static | 静态性用值层表示（不用 const generics 编码形状） | 值层 / 类型层 | 值层（Rust 1.74 限制；我们本来就是构建期生成）；atom 表用 `const fn` + `const` 断言拿到 rustc 编译期检查 |
| D-p4 | P4 通用性 kernel 选哪个 | MXFP8 GEMM / 别的（如 decode attention、MoE GEMM） | MXFP8 GEMM（atom 完全复用，有 hipBLASLt/aiter 可比） |
| D-verify | 是否让布局证明以"提示"形式随标记输出供验证器交叉检查 | 做 / 不做 | P3 之后再定；无论如何不作为信任来源 |


---

## 9. As-built（路线 B 实施结果）

本节记录路线 B 实际落地的样子：做了什么、东西搬到了哪里、用什么门禁验收、做了哪些决定。设计正文（§0–§8）保留原样，供对照。

### 9.1 阶段状态

| 阶段 | 交付 | 状态与证据 |
|---|---|---|
| P1 | `crates/amdgpu-layout`：`Ix`（静态常量 / 带范围与位支撑的动态符号）、`IntTuple`、`Layout`、FlyDSL 对齐的代数（composition、complement、coalesce、left/right inverse、logical/zipped/tiled/flat/blocked/raked product 与 divide、按 mode 的动态外层 divide、`local_tile`、`slice`、`recast_layout`、`make_layout_tv`）、GF(2) `Swizzle`（含 CuTe `Swizzle<B,M,S>` 与 FMHA 128 字节行 swizzle）、`IssuePlan` + `layout_to_offsets`、构建期证明（覆盖/单射/广播、连续性与向量宽度、`tr_b8` 源地址、MFMA 操作数匹配含 K 置换、swizzle 双射、LDS bank 冲突） | 完成。19 个单元测试：CuTe/FlyDSL 文档例子逐个复现；`_vs128` 的 K 片段读（`kfx`）和转置 V 读（`trx`）**逐 lane、逐指令与手写公式一致**，基址寄存器数同为 4 |
| P2 | `crates/amdgpu-isa`（copy / MMA atom 表，带 provenance 与 `const` 断言）、`crates/amdgpu-tile`（`Ctx`、`TiledCopy`、`partition_lds`、`partition_dma`、`partition_buffer_dma`、`partition_global`、`TiledMma`/`gemm`）；IR 新增 `VarMfmaScale16`（`v_mfma_scale_f32_16x16x128_f8f6f4`，op_sel 选 E8M0 字节）、`LoadLdsB64`/`StoreLdsB64` | 完成。在**发射出的 IR** 上用小解释器逐 lane 比对：TiledCopy 的 K 读 = `RowsReader`，转置 V 读 = `TrReader`，`partition_dma` = `RowsCopy`（全局地址、立即数、LDS 目的地，含尾块钳位），全部逐字相等 |
| P3 | `fmha_fwd_hd128_mxfp8_v3_{vs128,t128,kvsplit,kvsplit_combine8h}`（`crates/amdgpu-kernels/src/fmha/mod.rs`），**已成为出厂集** | 完成。全套 `port_gate` 对 legacy 与对 dsl 均 PASS（见 §9.4）；证书与 dsl/legacy 相同（6/29、4/29、12/36、2/6） |
| P4 | MXFP8 blockscale GEMM（aiter PR #6696 4-wave pin-AGPR 结构）：kernel-opt 用同一套 atom / TiledCopy / TiledMma 写成；编译器侧 AGPR（sched）、SALU 地址链 `scalarize2`、`ds_read_b64`/`ds_write_b64`、验证器位/区间规则 | 完成（截至 `v5.0-route-b`）：`gemm_mxfp8_bs_v4`（`crates/amdgpu-kernels`）4096×4096×8192 **3116 vs 3267 TF/s = 0.954×** 厂商，`tools/gemm_gate.sh` PASS，strict 3 / 19 / 0；差距仍在优化（[`current-status.md`](current-status.md) §4.7） |
| P5 | 主机侧 (b)+(c)：`amdc gen-host` 从 kernel 合同生成 C++ 启动/ABI 头与 JSON 元数据，零 unsafe | 完成，见 [`host-gen.md`](host-gen.md)、[`hostgen-json-schema.md`](hostgen-json-schema.md) |

### 9.2 东西搬到了哪里

| 以前（`amdgpu_tile` / `fmha_tile.rs`） | 现在 |
|---|---|
| `RowsReader`（K 片段，手写 `kfx` 公式） | `make_tiled_copy_a(DS_READ_B128X2, mma).partition_lds(..)`，`crates/amdgpu-kernels/src/fmha/mod.rs:83` |
| `TrReader`（转置 V，手写 `trx`） | `make_tiled_copy(DS_READ_TR_B8X4, vt_tr_source(), ..)`，`fmha.rs:30`、`fmha.rs:84` |
| `RowScaleReader`（K-scale 字） | 广播 TV 的 `make_tiled_copy_bcast(DS_READ_B32, kd_read_tv(), ..)`，`fmha.rs:41`、`fmha.rs:85` |
| `RowsCopy` / `DwordRowsCopy`（K/V/K-scale DMA） | `partition_dma`（从目的地侧推导：swizzle 放到源地址，M0 顶置 + 负立即数补偿），`fmha.rs:89` |
| `RowFrags::load`（Q 片段） | `partition_global`（G2R），`fmha.rs:459` |
| `store_bf16` / KV-split 私有转置暂存 | `partition_global`（R2G）`fmha.rs:500`；R2S/S2R + `Swizzle<3,4,3>` `fmha.rs:538` |
| `TensorDecl` 自动合同 | `contract::compact_tensor` / `same_strides`：合同从 `Layout` 推出（分配 = cosize，紧致关系 = 内层外延乘积），`fmha.rs:397` |
| 不涉及布局的部分 | 仍复用 `amdgpu_tile`：`LdsPlan`/标记、`Pipeline`/`Roles`、softmax、combine、`_t128` 的 VGPR K-scale 暂存、ones/Q 常量写入 |

crate 依赖：`amdgpu-layout`（零依赖）← `amdgpu-isa` ← `amdgpu-tile`（再依赖 `amdgpu-dsl`）← `amdgpu-compiler`（注册 kernel）。全部 `#![forbid(unsafe_code)]`，`rust_safety` 测试覆盖。

### 9.3 决定（与 §8 的对照）

| 编号 | 决定 | 实际情况 |
|---|---|---|
| D-host | (b)+(c) | 落地为 `amdc gen-host`（C++ 头 + JSON，带 schema 版本），出厂 `ref_host` 已改用生成的命名字段 |
| D-crate | 拆分 | `amdgpu-layout` / `amdgpu-isa` / `amdgpu-tile` 三个新 crate；GEMM 在 `amdgpu-kernels`（随 integrate 合入） |
| D-static | 值层 | `Ix::C` / `Ix::Sym`；atom 表用 `const` + `const` 断言 |
| **D-imm** | 原计划"P2 加 LDS 立即数 IR 开关，P3 A/B 后默认" | **取消**：实测降低阶段的 `lds_addr` 早已把 `x + const` 链折进 DS 立即数（`_vs128` 的 128 条 `ds_read_b128` 全部带 `offset:`）。规划器只负责生成可折叠的 `base + const` 形式并检查 `ImmRange`，IR 不改 |
| D-p4 | MXFP8 GEMM | 采用 |
| D-verify | 布局证明不进信任账本 | 保持：构建期证明只是提前报错，最终仍由 `gpu_verify` 在 IR 上证明 |

实施中追加的决定：

- swizzle 基址按"高位 + ((低位 ^ s ^ 静态异或) << dst_lo)"生成（与手写 `kfx`/`trx` 同形），区间分析紧致；纯 XOR 形式会让私有暂存区的证明失败。
- `Ctx::bind` 对一个 kernel 内所有 lane 表达式做 CSE：v3 的 lane 初始化比手写公式少 60 余条指令。
- `_t128` 默认采用 SADDR 形式的 LDS-DMA（标量张量基址 + 每 lane 32 位偏移），两轮同轮 A/B 布局中位数 +1.8% / +2.1%；`_vs128` / `_kvsplit` 上为 −0.2..−1.4%，不使用（历史记录：所述开关 / 别名 / 旧 kernel 已在 2026-11 仓库整理中删除，当前状态见 [`current-status.md`](current-status.md)）。

### 9.4 门禁（GPU 6，`tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3`）

对 legacy（同轮，采用 SADDR `_t128` 后的最终一轮）：VERIFY 0 错误；ACC16 15/15；ACC64 四条路径 bad=0；DET 5/5；STRESS 300/300 + 100/100；PERF：B1H8S2048 KV-split 681.1 vs 677.1，每 CU 445.1 vs 453.1，`_vs128` 布局中位数 448.1 vs 451.8，B4H8S2048 1418.7 vs 1446.7，B1H8S8192 1974.8 vs 1986.7，`_t128` 布局中位数 1035.7 vs 1040.7 → **OVERALL PASS**。

对 dsl（`SHIP_REF=fmha_fwd_hd128_mxfp8_dsl`，同轮 PERF）：679.0 vs 687.2、446.4 vs 445.1、布局中位数 449.0 vs 447.3、1408.0 vs 1413.8、1970.9 vs 1967.8、`_t128` 布局中位数 1002.4 vs 997.9（SADDR 之前）→ PASS。

SADDR A/B（当时的 `FMHA_OPT=saddrlds FMHA_OPT_SCOPE=v3s`，候选 `v3s` / 基线 `v3`，两轮）（历史记录：所述开关 / 别名 / 旧 kernel 已在 2026-11 仓库整理中删除，当前状态见 [`current-status.md`](current-status.md)）：

| 行 | 第 1 轮 | 第 2 轮 |
|---|---|---|
| B1H8S2048 KV-split | 686.9 / 680.9 | 688.4 / 690.3 |
| 每 CU `_vs128` | 441.2 / 447.2（−1.4%） | 442.0 / 446.3（−1.0%） |
| `_vs128` 布局中位数 | 446.5 / 447.3（−0.2%） | 447.4 / 448.3（−0.2%） |
| B4H8S2048 | 1400.8 / 1419.7 | 1401.9 / 1405.0 |
| B1H8S8192 | 1964.4 / 1975.7 | 1947.0 / 1963.0 |
| `_t128` 布局中位数 | 1033.1 / 1012.2（+2.1%） | 1019.7 / 1001.5（+1.8%） |

### 9.5 出厂 asm md5（`--backend=own`）

| kernel | md5（前 8 位） | 说明 |
|---|---|---|
| `fmha_fwd_hd128_mxfp8_v3_vs128` | `763142f9` | v7.0（标量死值消除后；此前 `68d2832b`、`df33f835`） |
| `fmha_fwd_hd128_mxfp8_v3_t128` | `91b26e5d` | SADDR DMA + `edgepad` + `salubr`；v7.0（此前 `e4c7e75d`、`f5a43b09`、`985d7a02`） |
| `fmha_fwd_hd128_mxfp8_v3_kvsplit` | `e1fe92a4` | v7.0（此前 `e4fc18b1`、`f6709cd2`） |
| `fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h` | `24a95a51` | |

上一版 `dsl_*` 集与旧手写集（当时作 A/B 对照）已在 2026-11 整理中删除。

### 9.6 编译器与验证器的配套改动（为 P3/P4 服务）

- `scalarize2`：一致的 `min`、对一致比较的 `select`（`s_cmp_lt_i32` + 紧邻的 `s_cselect_b32`，SCC 进依赖图）、`v_readfirstlane` 结果视为一致；GEMM profile 打开 `scalarize,scalarize2,waveuni` 后 2771 → 2867 TF/s，VGPR 408 → 376。FMHA 不受影响（md5 不变）。
- `Pipeline::vm_inflight` 与计数版 `vmcnt(n)`：协议检查器记录每个 `wait_vm` 的计数，DMA 在"其后至少有 n 条更新的 LDS-direct DMA"时才算完成；读下一 tile 的 FMHA 窗口用 `vm_inflight > 0` 会被拒绝。
- `gpu_verify` 的可能位（possible-bits）规则：`x ^ y`、`x & m` 在 bit31 可证为 0 时收窄到 `[0, mask]`（`(row << 7) ^ (chunk << 4)` 证明 ≤ 2032）；声明的最大值（`ka_range` / `ka_set` / `sym_range`）参与多项式非负判定（盒子检验）。均附攻击测试（可能为负的操作数、重叠和的进位、真实越界、无界 kernarg）。

### 9.7 `edgepad` / `salubr` A/B（当时的 `FMHA_OPT_SCOPE=v3s`，候选 `v3s` / 基线 `v3`，GPU 6；历史记录）

| 行 | edgepad + salubr | 仅 edgepad | 仅 salubr |
|---|---|---|---|
| B1H8S2048 KV-split | 675.1 / 685.7（−1.5%） | 690.2 / 690.1 | 678.9 / 683.6（−0.7%） |
| 每 CU `_vs128` | 441.4 / 445.1 | 444.7 / 446.6 | 440.7 / 446.1 |
| `_vs128` 布局中位数 | 442.4 / 447.6（−1.2%） | 445.1 / 449.5（−1.0%） | 442.2 / 448.0（−1.3%） |
| B4H8S2048 / B1H8S8192 | 1402.6 / 1418.2、1960.2 / 1975.8 | 1411.4 / 1415.8、1963.6 / 1968.3 | 1411.9 / 1426.4、1955.9 / 1965.1 |
| `_t128` 布局中位数 | **1051.2 / 1017.3（+3.3%）** | 1032.9 / 1026.2（+0.7%） | 1060.1 / 1038.6（+2.1%） |

结论：两项一起只在 v3 `_t128` 默认打开（当时 md5 `e4c7e75d`）；采用后全套门禁（对当时的旧手写集）PASS，`_t128` 布局中位数 1061.95 vs 1023.06。其余 kernel 不变。
