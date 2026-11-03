# 布局层入门：从 `Layout` 到出厂 FMHA 与 GEMM 的每一个地址

> **读者**：读过（或至少翻过）[`fmha-dsl-tour.md`](fmha-dsl-tour.md)，知道 DSL 里 `FnBuilder`、LDS、DMA、MFMA 大概是什么，但**没接触过 CuTe / FlyDSL 的布局代数**。
> **方法**：每个概念先用一句话讲清楚，再给仓库里真实的代码或测试（`文件:行号`，以写作时的 `Main` 为准；漂移了就按函数名查）。所有例子都能在节点上用
> `bash tools/container/cargo.sh test -p amdgpu-layout -p amdgpu-tile` 跑出来。
> **范围**：三个 crate —— `amdgpu-layout`（布局代数，零依赖）、`amdgpu-isa`（gfx950 指令 atom）、`amdgpu-tile`（把布局变成 IR 的
> `TiledCopy` / `TiledMma` / `partition_*`）。设计与实施记录见 [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md)（§9 是 as-built）。

---

## 目录

- §0 为什么要"布局"
- §1 `Layout` = shape + stride
- §2 静态与动态：`Ix`
- §3 布局代数：composition、complement、divide、product、inverse
- §4 `Swizzle`：把 XOR 交错写进布局
- §5 TV 布局：线程 × 值 → 元素
- §6 发射空间与立即数：`IssuePlan` / `layout_to_offsets`
- §7 构建期证明
- §8 atom 与 `TiledCopy` / `TiledMma` / `partition_*`：真实 FMHA 与 GEMM
- §9 动手练习
- §10 常见误解

---

## §0 为什么要"布局"

GPU kernel 里最容易错、也最难读的代码是**地址**：第 `l` 个 lane 在第 `i` 条 `ds_read_b128` 里读 LDS 的哪个字节？旧的手写 FMHA 里这是一行行公式，比如
K 片段读的基址：

```
kfx[k] = j*128 + (((half ^ s(j)) ^ 2k) << 4) + lds_k        // j = lane & 31, half = lane >> 5, s(j) = 行 j 的 swizzle
```

它是对的，但**不可复用**：换一个 MFMA 形状、换一种 LDS 排布、换一个 swizzle，公式就得重推一遍。

布局代数（来自 NVIDIA CuTe，FlyDSL 是它在 AMD 上的移植）把这件事拆成三块可组合的部件：

1. **数据在哪里**：张量 / LDS 图像的布局（坐标 → 字节）；
2. **谁拿哪个元素**：指令的 TV 布局（lane、值序号 → 元素）；
3. **把它们复合起来**：得到"lane、值序号 → 字节"，再拆成"每个 lane 一次算好的基址 + 每条指令的立即数"。

本仓库里这三步分别是 `Layout`（§1–§3）+ `Swizzle`（§4）、TV 布局与 atom（§5、§8）、`IssuePlan` / `layout_to_offsets`（§6）。
上面那行 `kfx` 公式，现在是 §6 里四行代码的**推导结果**，并且有测试逐 lane 证明二者相等。

---

## §1 `Layout` = shape + stride

一个 `Layout` 是一对同形状的整数元组：**shape**（每一维多大）和 **stride**（这一维走一步，地址走多少）。坐标 `(c0, c1, …)` 的地址是
`Σ ci · stride_i`。一维下标按**列优先**（第 0 维变化最快）拆成坐标。

FlyDSL 文档的第一个例子，原样在测试里（`crates/amdgpu-layout/tests/algebra.rs:14`）：

```rust
let l = make_layout(tup![8, 16], tup![1, 8]);        // 8 x 16，列优先
assert_eq!(l.crd2idx(&tup![3, 5], &d), Ix::C(43));    // 3*1 + 5*8 = 43
assert_eq!(idx2crd(&Ix::C(43), &l.shape, &d), tup![3, 5]);
assert_eq!(l.size(), Ix::C(128));
```

元组可以**嵌套**：`(9, (4, 8))` 是"第 0 维 9，第 1 维又分成 4 和 8"。嵌套让一个维度内部有自己的步长，这正是描述"lane 的低 5 位走行、高 1 位走列"
这种硬件排布的方式。例如 32x32x64 MFMA 的 A 操作数（§5）：

```rust
// crates/amdgpu-isa/src/lib.rs:165（mfma_scale_32x32x64_f8）
make_layout(tup![tup![32, 2], tup![16, 2]], tup![tup![1, 512], tup![32, 1024]])
//            ^ 线程: lane%32 行, lane/32 半    ^ 值: 16 个字节, 2 组
```

读法：元素下标 = `m + 32·k`（A 是 32 行 × 64 字节）。线程 `t` 拆成 `(t%32, t/32)`，前者走行（步长 1），后者让 k 跳 16（16·32 = 512）；
值 `v` 拆成 `(v%16, v/16)`，前者让 k 走 1（步长 32），后者让 k 跳 32（32·32 = 1024）。

---

## §2 静态与动态：`Ix`

布局的叶子不一定是常数。块号、lane、行跨度往往到运行时才知道。`Ix` 是一个小的符号整数（`crates/amdgpu-layout/src/ix.rs`）：

- `Ix::C(k)`：编译期常数；
- `Ix::Sym(s)`：动态符号，在 `Dyns` 里登记**取值范围**（lane ∈ [0, 63]）；
- 以及 `Sum` / `Mul` / `Div` / `Mod` / `Xor` / `And` / `Shl` / `Shr` 组合，构造时自动规范化。

范围让化简更聪明，例如（`tests/algebra.rs:174`）：

```rust
let lane = d.sym("lane", 0, 63);
assert_eq!(lane.div_c(32, &d).mod_c(2, &d), lane.div_c(32, &d));   // lane < 64 ⇒ (lane/32)%2 = lane/32
let e = lane.mod_c(32, &d).mul_c(128).add(&lane.div_c(32, &d).mul_c(16)).add_c(4096);
assert_eq!(e.split_const().1, 4096);                                // 动态部分 + 常数部分
assert_eq!(e.split_const().0.bits(&d), (31 << 7) | 16);             // 动态部分可能置位的 bit
```

`bits()`（可能置位的位）是 §6 立即数折叠的关键：只要动态部分与常数部分**不共享任何 bit**，加法就等于异或，swizzle 就能"穿过"常数。

**原则**：tile 内部的 mode 用静态 shape（指令形状、tile 大小都是常数），动态只出现在外层（第几个 tile、行跨度）。这样所有指令级的拆分都是编译期常数。

`amdgpu-tile` 的 `Ctx` 把符号接到 IR 值上（`ctx.sym("lane", 0, 63, lane.lane())`），`Ctx::bind` 把 `Ix` 降成 IR（2 的幂的除法/取余变成
`lshr`/`and`），并对**整个 kernel 的所有 lane 表达式做 CSE**（`crates/amdgpu-tile/src/lib.rs:88`）。

---

## §3 布局代数

全部在 `crates/amdgpu-layout/src/layout.rs`，与 FlyDSL / CuTe 同名同义，测试逐条复现它们文档里的例子（`tests/algebra.rs`）。

| 运算 | 含义 | 例子（来自测试） |
|---|---|---|
| `composition(A, B)` | 先 B 再 A：`(A∘B)(i) = A(B(i))` | `(6,2):(8,2) ∘ (4,3):(3,1) = ((2,2),3):((24,2),8)` |
| `complement(A, M)` | 在 `[0, M)` 里 A 没覆盖到的"剩余"布局 | `complement(4:2, 24) = (2,3):(1,8)` |
| `logical_divide(A, T)` | 把 A 按 tile T 切成 (tile 内, tile 间) | `(4,2,3):(2,1,8) ÷ 4:2 = ((2,2),(2,3)):((4,1),(2,8))` |
| `logical_product(A, B)` | 把 A 重复 B 次 | `(2,2):(4,1) × 6:1 = ((2,2),(2,3)):((4,1),(2,8))` |
| `right_inverse(L)` | `L ∘ R = id` | `right_inverse((4,8):(8,1)) = (8,4):(4,1)` |
| `coalesce` / `filter` | 合并可合并的维 / 去掉步长 0 | `(2,(1,6)):(1,(6,2)) → 12:1` |
| `zipped/tiled/flat_divide_bymode` | 按维切 tile，外层可以是动态的 | `local_tile` 用它取第 t 个 tile 的布局与偏移 |
| `recast_layout` | 换元素位宽（fp16 → 字节、字节 → dword） | `(4,8):(1,4)` 的 16 位 → 8 位 = `64:1` |
| `make_layout_tv` | 由线程排布与值排布构造 TV 布局 | 覆盖与单射由 `check_tv` 证明 |

**composition 是主力**。FMHA 里"K 操作数 tile 在 LDS 里的字节布局"是 `(32, 64):(128, 1)`（32 行，每行 128 字节，取其中 64 字节），
MFMA 的 A TV 布局是 §1 那个；复合起来（`tests/fmha_vs128.rs:20` 用到）：

```
(32,64):(128,1) ∘ ((32,2),(16,2)):((1,512),(32,1024))  =  ((32,2),(16,2)):((128,16),(1,32))
```

读法：lane 的低 5 位走 128 字节（行），第 6 位走 16 字节（另一半 k），值的低 4 位逐字节，第 5 位跳 32 字节。这就是"每个 lane 读哪些字节"，
不再需要手推。

**right_inverse 的用法**（GEMM 的 DMA，`crates/amdgpu-kernels/src/gemm_mxfp8.rs` 的 `dma_plan`）：LDS-direct DMA 的**目的地固定**为
`M0 + 16·lane`，问题是"LDS 第 d 字节放的是 tile 的哪个元素"——这是 LDS 布局的逆。于是源地址 = `glob ∘ right_inverse(smem)`：

```rust
let sm = smem_tile(rows);                               // LDS 图像：(r, k) → 字节
let g = composition(glob, &right_inverse(&sm));        // LDS 字节 → 全局字节
let gofs = IssuePlan::new(tv, 16).offsets(&g, s_tid, &Ix::C(0), &ctx.dyns);
```

---

## §4 `Swizzle`：把 XOR 交错写进布局

LDS 有 64 个 4 字节 bank。K 片段读时 16 个 lane 同时读 16 行的同一列，行跨度 128 字节恰好让它们落在同一组 bank 上——8 路冲突。
解决办法是 XOR swizzle：把"16 字节块号"异或上行号的几位。

`Swizzle`（`crates/amdgpu-layout/src/swizzle.rs`）是 GF(2) 线性的：目的位段的每一位 = 源位段若干位的异或。

```rust
// swizzle.rs:35（FMHA 128 字节行的 swizzle）
Swizzle { src_lo: 7, src_n: 4, dst_lo: 4, dst_n: 3, rows: vec![0b0101, 0b0010, 0b1001] }
// 块号的 bit0 ^= 行 bit0 ^ 行 bit2；bit1 ^= 行 bit1；bit2 ^= 行 bit0 ^ 行 bit3
```

CuTe 的 `Swizzle<B, M, S>` 是它的特例（`Swizzle::cute(b, m, s)`）；KV-split 的转置暂存区用 `Swizzle<3,4,3>`
（`crates/amdgpu-tile/src/fmha.rs:538`：块号 ^= 行号低 3 位）。

swizzle 与布局复合成 `ComposedLayout { swz, inner }`：`坐标 → inner → 字节 → swz → 实际字节`。测试证明三件事：

- 与旧手写 `swz_s` 对所有 (行, 块) 相同（`tests/algebra.rs:149`）；
- 是双射（`check_swizzle_bijective`）；
- **K 的 b128 读和 V 的 tr_b8 读在 swizzle 下 0 bank 冲突，不 swizzle 时有冲突**（`tests/fmha_vs128.rs:120`，用 `proofs::bank_conflicts`）。

---

## §5 TV 布局：线程 × 值 → 元素

TV（thread-value）布局是一个两维布局：第 0 维是线程（lane，或 `lane + 64·wave`），第 1 维是这个线程持有的值序号；输出是 tile 的元素下标。
它描述**一条指令（或一个 MMA 操作数）的数据归属**。

`amdgpu-isa` 的 MMA atom 给出 A/B/C 三个 TV 布局：

| atom | A/B | C |
|---|---|---|
| `mfma_scale_32x32x64_f8`（`isa/src/lib.rs:165`） | `((32,2),(16,2)):((1,512),(32,1024))`：lane%32 行，k = 16·(lane/32)+0..15 与 32+16·(lane/32)+0..15 | `((32,2),(4,4)):((32,4),(1,8))`：行 8(r/4)+4(lane/32)+r%4，列 lane%32 |
| `mfma_scale_16x16x128_f8`（`:187`） | `((16,4),(16,2)):((1,256),(16,1024))`：lane%16 行，k = 16·(lane/16)+0..15 与 64+16·(lane/16)+0..15 | `((16,4),4):((16,4),1)` |

16x16x128 的布局由厂商反汇编确认（A 片段两条 `ds_read_b128`，偏移 0 与 64，每行 128 字节）。

**`swap_ab`**：厂商 GEMM 把硬件 srcA 接 B 片段、srcB 接 A 片段，结果是 Cᵀ。`make_tiled_mma(atom, true)` 交换 A/B 的 TV 布局并把 C 转置
（`crates/amdgpu-tile/src/mma.rs:40` 的 `c_tv`），`gemm(...)` 自动交换操作数、缩放与字节选择（`mma.rs:68`）。

copy 也有 TV 布局，往往**不是** MMA 布局本身，而是"同一组元素按另一种方式取"。例如转置 V 读的源地址（`fmha.rs:30`）：

```rust
// lane bit0 → 字节 +8, bit1/bit2 → key +1/+2, bit3 → key +8, bit4 → 字节 +16, bit5 → key +4；值：读 r(key +16r)、半 hh(key +64hh)、列组 e(字节 +32e)
make_layout(tup![tup![2, 2, 2, 2, 2, 2], tup![4, 2, 4]], tup![tup![8 * 128, 1, 2, 8, 16 * 128, 4], tup![16, 64, 32 * 128]])
```

以及**广播** TV：K-scale 字两半 lane 读同一个字，线程第 2 维步长为 0（`fmha.rs:41`）：

```rust
make_layout(tup![tup![32, 2], 4], tup![tup![1, 0], 32])     // lane j 与 lane j+32 读第 j + 32s 个字
```

---

## §6 发射空间与立即数：`IssuePlan` / `layout_to_offsets`

一个线程的值被切成若干条指令（每条 `vec` 个元素，例如 b128 = 16 个 fp8）。`IssuePlan`（`crates/amdgpu-layout/src/tv.rs`）就是
"线程维（动态，每 kernel 算一次）× 发射维（静态，每条指令一个）"的拆分（OPUS 的思路）。`offsets(dst, lane, extra)` 对每条指令给出一个
`Ix` 字节偏移。

`layout_to_offsets`（`tv.rs:86`）把这些偏移拆成**少量基址寄存器 + 每条指令的立即数**：

- 无 swizzle：偏移 = 动态部分 + 常数，常数进立即数（在 `ImmRange` 内时；DS 0..65535，global −4096..4095，buffer 0..4095）。
- 有 swizzle：若动态部分 `a` 与常数 `c` 的 bit 不相交，则 `swz(a + c) = swz(a) ^ c'`；`c'` 中落在 swizzle 目的位或 `a` 的位上的部分只能进基址，
  其余（与基址 bit 不相交，所以 XOR = 加法）进立即数。

这正是 `_vs128` K 读（`tests/fmha_vs128.rs:20`）：

```rust
let plan = IssuePlan::new(mfma_a_32x32x64_f8(), 16);                       // ds_read_b128：每条 16 字节
for s in 0..4 { for c in 0..2 {                                            // 4 个 32 行块 × 2 个 k 组
    let tile = make_layout(tup![32, 64], tup![128, 1]);
    exprs.extend(plan.offsets(&tile, &lane, &Ix::C(4096 * s + 64 * c), &d));
}}
let op = layout_to_offsets(&exprs, &Swizzle::fmha128(), ImmRange::DS, &d);
assert_eq!(op.bases.len(), 4);                                             // = 手写的 kfx[0..3]
// 每条指令：基址 = 手写 kfx[2c+i]，立即数 = 4096·s —— 对 64 个 lane 逐个断言相等
```

基址按"高位 + ((低位 ^ s ^ 静态异或) << dst_lo)"生成，和手写公式同形，验证器也能给出紧的区间。

> **说明**：立即数实际由编译器的降低阶段折叠——它本来就会把 `x + 常数` 链折进 DS 立即数。规划器负责的是把地址**写成可折叠的形状**，
> 并在常数超出 `ImmRange` 时把它留在基址里。

---

## §7 构建期证明

构建 kernel（= 生成 GPU 程序）时，`amdgpu_layout::proofs` 会检查布局，不对就 `panic!` 并指出哪个线程哪条指令：

| 证明 | 函数 | 例子 |
|---|---|---|
| 覆盖 + 单射（或均匀广播） | `check_tv(tv, tile, dup)` | TV 布局漏了半个 tile、或两个值撞同一元素 → 拒绝（`tests/algebra.rs:140`） |
| 连续 + 对齐 | `check_contiguous` | 8 字节粒度的 swizzle 会把 16 字节向量拆开 → 拒绝（`tests/fmha_vs128.rs:149`） |
| `tr_b8` 源地址 | `check_tr_b8_source` | 每 lane 8 字节对齐、64 个 lane 读 64 个不同四字 |
| MMA 操作数匹配（含 K 置换） | `check_operand_match` | 手写 K 读反推出的元素恰是 MFMA A 布局；错的 K 顺序被拒（`tests/fmha_vs128.rs:51`、`:178`） |
| swizzle 双射 / bank 冲突 | `check_swizzle_bijective` / `bank_conflicts` | §4 |
| DMA 目的地按 lane 连续 | `partition_buffer_dma` 内 | `M0 + imm + 16·lane` 以外的落点 → 拒绝 |

这些证明只是**提前报错**，不进信任账本：最终仍由 `gpu_verify` 在 IR 上证明每次访存的边界与协议（[`gpu-semantics-verifier.md`](gpu-semantics-verifier.md)）。

---

## §8 atom 与 `TiledCopy` / `TiledMma` / `partition_*`

**atom**（`crates/amdgpu-isa/src/lib.rs`）是一条指令在单个 lane 上的样子：搬几个字节、几条指令为一组、对齐、立即数域、完成计数器（vmcnt / lgkmcnt）、
LDS-direct 的固定落点、是否转置、以及 **provenance**（这个布局来自 ISA、硬件探针、已出厂 kernel 还是厂商推导）。表是 `const` 的，
`const _: () = { … }`（`:124`）在 rustc 编译期检查每一项。

**`TiledCopy`** = atom + TV 布局（`crates/amdgpu-tile/src/copy.rs`）：

- `make_tiled_copy(atom, tv, tile, elem_bits)`：证明覆盖；`make_tiled_copy_a/_b(atom, &mma)` 直接用 MMA 的操作数布局；`make_tiled_copy_bcast` 允许广播。
- `partition_lds(b, &ctx, &lane, &shared, &view, &placements)`：对一块 LDS（`view` = 带 swizzle 的字节布局）求出每个 lane 的基址（**每 kernel 绑定一次**）和每次操作的立即数；
  `.at(b, &slot)` 每个 tile 加一次槽位偏移；`.read(b, op)` / `.write(b, op, vals)` 在槽位标记下发射指令。
- `partition_dma`：global → LDS 的 `global_load_lds`，从**目的地**反推源地址，swizzle 落在源一侧，M0 放在 wave 段顶部、用负立即数补偿。
- `partition_buffer_dma`：MUBUF `buffer_load … lds` 的厂商形式——voffset 每 lane 算一次，每条指令的源增量（与线程无关时）放进 SGPR soffset。
- `partition_global`：寄存器 ↔ 全局（Q 片段加载、bf16 输出存储），行由 kernel 提供（钳位后的行不是仿射的）。

**`TiledMma`** = MMA atom +（可选）`swap_ab`；`gemm(b, &mma, acc, a, b, sa, sb, byte_a, byte_b)` 发射原地累加。

### FMHA（出厂 v3 `_vs128`，`crates/amdgpu-tile/src/fmha.rs:79-90`）

```rust
let mma = make_tiled_mma(MmaAtom::mfma_scale_32x32x64_f8(), false);
let kview = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![32, 64], tup![128, 1]));
let placements: Vec<i64> = (0..4).flat_map(|s| (0..2).map(move |c| 4096 * s + 64 * c)).collect();
let kpart = make_tiled_copy_a(DS_READ_B128X2, &mma).partition_lds(b, &ctx, &l, &ks, &kview, &placements);
let vpart = make_tiled_copy(DS_READ_TR_B8X4, vt_tr_source(), 128 * 128, 8).partition_lds(b, &ctx, &l, &vs, &kv_view(), &[0]);
…
let kdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, ksrc, &ks, &kv_view());
```

每个 tile 的 QK（`fmha.rs:179-187`）：`k.read(b, 2s)`、`k.read(b, 2s+1)`、`kd.read(b, s)`、两次 `gemm`。测试
`crates/amdgpu-tile/tests/address_equality.rs` 和 `dma_equality.rs` 用一个小解释器跑发射出的 IR，证明这些地址与旧手写 reader / copy 对每个 lane、
每条指令逐字相等。

### GEMM（MXFP8 blockscale，`crates/amdgpu-kernels/src/gemm_mxfp8.rs`）

LDS 图像是 16 行 × 16 字节交错（和厂商 B 预洗牌一致，一个 16 lane 组正好扫完 256 个连续字节，所有 bank 各一次）：

```rust
// (r, k) -> (r / 16) 2048 + (k / 16) 256 + (r % 16) 16 + k % 16
fn smem_tile(rows: i64) -> Layout {
    make_layout(tup![tup![16, rows / 16], tup![16, 8]], tup![tup![16, 2048], tup![1, 256]])
}
```

MMA 是 16x16x128、`swap_ab = true`；A 用 FMHA 的 128 字节行 swizzle，B 用上面的交错图像：

```rust
let mma = make_tiled_mma(MmaAtom::mfma_scale_16x16x128_f8(), true);
let copy_a = make_tiled_copy_a(DS_READ_B128X2, &mma);
let copy_b = make_tiled_copy_b(DS_READ_B128X2, &mma);
let a_view = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![16, 128], tup![128, 1]));
let mut a_rd = copy_a.partition_lds(b, &ctx, &s_lane, &a_s, &a_view, &a_place);
let mut b_rd = copy_b.partition_lds(b, &ctx, &s_lane, &b_s, &view, &b_place);
```

DMA 源地址 = `glob ∘ right_inverse(smem)`（§3）。同一套 atom 与代数同时服务 FMHA 与 GEMM——这就是"布局层"的意义。

---

## §9 动手练习

1. 在 `crates/amdgpu-layout/tests/algebra.rs` 里加一个测试：`complement(make_layout(tup![2, 4], tup![1, 8]), 32)` 应该是什么？先手算，再跑。
2. 把 §6 的 K 读例子里的 `Swizzle::fmha128()` 换成 `Swizzle::none()`：基址数会变成几？立即数呢？`bank_conflicts` 报多少？
3. 写一个 TV 布局，让 64 个 lane 各读一行 128 字节中的 16 字节（8 lane 一行），用 `check_tv` 证明它覆盖 [8 行 × 128 字节]，再用
   `check_contiguous` 配合 `Swizzle::cute(3, 3, 4)` 看它为什么被拒。
4. 在 `amdgpu-tile` 的 `address_equality.rs` 里把 `placements` 改错（例如 `64 * c` 写成 `32 * c`），看测试如何指出第几个 lane 第几条指令不等。

---

## §10 常见误解

- **"布局层会改变指令序列/调度"**：不会。`partition_*` 只决定地址和每次操作发射哪些指令；指令顺序由 kernel 作者写的调用顺序与编译器调度决定。
- **"构建期证明通过就安全了"**：不。它们是提前报错；安全边界仍由 `gpu_verify` 在 IR 上证明（越界、协议、等待）。
- **"动态布局会让每条指令多算地址"**：线程维（lane / wave）的表达式每 kernel 只绑定一次（`Ctx::bind` 还会 CSE），每个 tile 只加一次槽位偏移，
  每条指令只剩可折叠的立即数。v3 的 lane 初始化比手写公式还少 60 余条指令。
- **"swizzle 让地址无法分析"**：基址按"高位 + ((低位 ^ s) << lo)"生成，是有界项之和；验证器还有 possible-bits 规则处理 `x ^ y` / `x & m`。
- **"TV 布局就是 MMA 布局"**：copy 的 TV 布局可以不同（转置读、广播读、DMA 的目的地驱动），只要最终寄存器里的元素与 MMA 期望一致——
  这正是 `check_operand_match` 检查的。
