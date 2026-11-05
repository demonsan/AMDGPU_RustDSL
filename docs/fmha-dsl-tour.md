# 用真实 FMHA 读懂 Rust DSL：逐段特性 + Rust safe 是怎么保证的

> **读者**：会一点 C/C++、知道 GPU 大概怎么回事，但**没写过 Rust kernel**、不熟 Rust 所有权/借用、也没见过本仓库 DSL 的人。
> **方法**：只用仓库里的真实代码（节选处用 `…` 标出），每段给出 `文件:行号`。行号以写作时的 `Main` 为准；若将来漂移，按引用的函数名查找。
> **配套**：原则与边界见 [`rust-safety.md`](rust-safety.md)、逐条证据见 [`rust-safety-by-example.md`](rust-safety-by-example.md)、
> 验证器模型见 [`gpu-semantics-verifier.md`](gpu-semantics-verifier.md) 与 [`gpu-semantic-protocol.md`](gpu-semantic-protocol.md)、
> 当前数字见 [`current-status.md`](current-status.md) §2。

> **路线 B 之后的更新（必读）**：出厂 kernel 现在是 `fmha_fwd_hd128_mxfp8_v3_*`，写在**布局层**上
> （`amdgpu-layout` → `amdgpu-isa` → `amdgpu-tile`）。本文 §2 逐段拆解的 IR 语义（kernarg 合同、LDS 计划、DMA、MFMA、softmax、
> barrier、验证器）**全部不变**；变化的是"地址从哪里来"：以前每个 lane 的 LDS/全局地址是手写公式（`RowsReader` 的 `kfx`、`TrReader`
> 的 `trx`、`RowsCopy` 的列 swizzle），现在由 `Layout`/`Swizzle`/TV 布局推导，再由同一个编译器与验证器处理。
> 对照阅读 v3 `_vs128`：
>
> ```rust
> // crates/amdgpu-kernels/src/fmha/mod.rs:79-90（KvRing3::new，节选）
> let mma = make_tiled_mma(MmaAtom::mfma_scale_32x32x64_f8(), false);
> // K: the A operand (32 keys x 64 bytes) at rows 32 s, k-group c
> let kview = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![32, 64], tup![128, 1]));
> let placements: Vec<i64> = (0..4).flat_map(|s| (0..2).map(move |c| 4096 * s + 64 * c)).collect();
> let kpart = make_tiled_copy_a(DS_READ_B128X2, &mma).partition_lds(b, &ctx, &l, &ks, &kview, &placements);
> let vpart = make_tiled_copy(DS_READ_TR_B8X4, vt_tr_source(), 128 * 128, 8).partition_lds(b, &ctx, &l, &vs, &kv_view(), &[0]);
> …
> let kdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, ksrc, &ks, &kv_view());
> ```
>
> ```rust
> // crates/amdgpu-kernels/src/fmha/mod.rs:179-187（qk_block：两条 K 片段读、K-scale 字、两次带缩放 MFMA）
> let k0 = k.read(b, 2 * s);
> let k1 = k.read(b, 2 * s + 1);
> let kw = kd.read(b, s);
> let (sk0, sk1) = blk.sel.pair(b, kw);
> …
> gemm(b, &self.ring.mma, t.accs[s].var, k0, q[0].v, sk0, blk.sq[0], 0, 0);
> gemm(b, &self.ring.mma, t.accs[s].var, k1, q[1].v, sk1, blk.sq[1], 0, 0);
> ```
>
> kernel 函数本身（`fmha.rs:283-303`）与 dsl 版结构相同：`declare_io` → `block3` → `Roles::split` → `KvRing3::new` → `Pipeline` → `store_out3`。
> 布局层从零讲起的入门见 [`layout-tour.md`](layout-tour.md)；实施记录见 [`dsl-layout-refactor-design.md`](dsl-layout-refactor-design.md) §9。
> 下文 §2 的节选仍来自 legacy/dsl 版本（行号以写作时为准），用来解释 IR 层的每个特性。

---

## 目录

- §0 背景：给没写过 kernel 的人
- §1 最小可运行样例（逐行）
- §2 真实 FMHA 的逐段拆解（a–i 九段，每段三问）
- §3 safety 到底由什么保证（L1 / L2 / L3 分层）
- §4 演示"写错"：编译器/验证器拒绝什么
- §5 从 DSL 到 `.s` 的实际流程
- §6 新手最容易误解的 6 个点
- 如果只记三件事

---

## §0 背景：给没写过 kernel 的人

**GPU kernel 是什么**：一段由成千上万个"线程"同时执行的函数。你在 CPU 上写 `for i in 0..N { out[i] = a[i] + b[i]; }`，
在 GPU 上则是"启动 N 个线程，每个线程只算自己那一个 `i`"。

**AMD 的几个名词（括号里是 CUDA 的近似叫法）**：

| 名词 | 白话 | CUDA 近似 |
|---|---|---|
| 线程 / lane（workitem） | 一个"算一份数据"的执行者 | thread |
| 波（wave / wavefront） | **64 个 lane 绑在一起同步执行同一条指令**（像 64 人方阵齐步走） | warp（32 线程） |
| 工作组（workgroup） | 若干 wave 组成一组，共享一块片上内存，可以互相等（barrier） | thread block |
| LDS（Local Data Share） | 工作组内共享的片上小内存（gfx950 上每组最多 160 KiB） | shared memory |
| VGPR / SGPR | 向量寄存器（每个 lane 一份）/ 标量寄存器（整个 wave 共用一份） | register / uniform |
| MFMA | 矩阵乘加指令：一条指令让整个 wave 合作算一个小矩阵块（如 32×32×64） | Tensor Core `mma` |

**为什么"安全"必须由工具链给**：

1. **LDS 没有边界检查**。你往 LDS 第 200000 字节写，硬件不会报错——要么悄悄丢掉，要么写到别人的数据上。
2. **没有生命周期保护**。一个 wave 发出"从显存搬数据到 LDS"的请求（DMA）后，指令立刻返回；数据**过一会儿**才到。
   你若提前读，读到的是旧数据——不会崩溃，只会**算错**。这就像在图书馆预约了一个座位（发出请求），
   但还没确认"座位已经空出来"（等待完成）就一屁股坐下去。
3. **barrier 只管"人到齐"，不管"货到齐"**：`s_barrier` 让工作组里所有 wave 在某处集合，但它不等任何访存完成。

所以本仓库的 "safe" 不可能来自硬件，只能来自**写 kernel 的语言 + 编译器 + 验证器**。这正是本文要讲的。

---

## §1 最小可运行样例（逐行）

先看一个最小的真实 kernel：每个 lane 计算 `c[tid] = a[tid] + b[tid] + s`。

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:19135-19165（probe_llvm_kernarg，节选）
pub fn probe_llvm_kernarg() -> Function {
    let mut b = FnBuilder::new("probe_llvm_kernarg", 64);   // ①
    b.arg("a", DType::Ptr, 0, 8);                             // ②
    b.arg("b", DType::Ptr, 8, 8);
    b.arg("n", DType::I32, 16, 4);
    b.arg("s", DType::F32, 20, 4);
    b.arg("c", DType::Ptr, 24, 8);
    …
    let pa = b.let_(Op::KernargPtr { offset: 0 });            // ③
    let pb = b.let_(Op::KernargPtr { offset: 8 });
    …
    let s = b.let_(Op::KernargF32 { offset: 20 });
    let pc = b.let_(Op::KernargPtr { offset: 24 });
    …
    let tid = b.let_(Op::WorkitemId);                          // ④
    let c4 = b.c_i32(4);                                       // ⑤
    let off = b.mul_i32(tid, c4);                              // ⑥
    let qa = b.ptr_add(pa, off);                               // ⑦
    let qb = b.ptr_add(pb, off);
    let qc = b.ptr_add(pc, off);
    let x = b.load_f32(qa, 0);                                 // ⑧
    let y = b.load_f32(qb, 0);
    let xy = b.add_f32(x, y);
    let r = b.add_f32(xy, s);
    b.store_f32(qc, 0, r);                                     // ⑨
    …
    b.finish()                                                 // ⑩
}
```

| 行 | 做了什么 | 对应的 GPU 概念 |
|---|---|---|
| ① `FnBuilder::new(name, 64)` | 建一个"kernel 构造器"，工作组大小 64 个 lane（= 1 个 wave） | launch 时的 block size |
| ② `b.arg(名字, 类型, 偏移, 大小)` | 声明**参数区（kernarg）的字节布局**：`a` 是 8 字节指针放在偏移 0 | host 传进来的参数结构体 |
| ③ `b.let_(Op::KernargPtr{offset})` | "从参数区偏移 0 读出指针"——产生一个新的**值编号（ValueId）** | 读 kernel 参数 |
| ④ `Op::WorkitemId` | 本 lane 在工作组里的编号 0..63 | `threadIdx.x` |
| ⑤ `b.c_i32(4)` | 常量 4 | 立即数 |
| ⑥ `b.mul_i32` | 整数乘：`tid*4` 字节偏移 | `v_mul_lo_u32` |
| ⑦ `b.ptr_add` | "指针 + 字节偏移"，结果仍是指针 | 地址计算 |
| ⑧ `b.load_f32(ptr, 0)` | 从显存读 4 字节浮点 | `global_load_dword` |
| ⑨ `b.store_f32` | 写回显存 | `global_store_dword` |
| ⑩ `b.finish()` | 交出构造好的 `Function`（一份数据结构） | — |

**关键认识：这些都是普通的 Rust 函数调用，不是内联汇编。** 每次调用只是往一张"计算图"里加一个节点：

```rust
// crates/amdgpu-dsl/src/lib.rs:20 与 :970-976
pub type ValueId = u32;
…
pub fn let_(&mut self, op: Op) -> ValueId {
    let ty = op.result_ty();
    let dst = self.f.value_tys.len() as ValueId;
    self.f.value_tys.push(ty);
    self.f.stmts.push(Stmt::Let { dst, op });
    dst
}
```

- `ValueId` 就是一个**下标（u32）**，好比餐厅取号单上的号码——它不是指针、不指向任何内存，丢了、复制了都不会造成内存错误。
- 每个值都有类型（`DType`：`I32/F32/Ptr/Frag8/Frag16/Rsrc…`，`lib.rs:35-55`），构造时就记下来，之后由验证器检查。
- 整个 crate 顶部写着 `#![forbid(unsafe_code)]`（`lib.rs:15`）：这份 DSL 代码里**不可能**出现裸指针、`unsafe`、FFI。
- 真正的 GPU 寄存器、指令编码，只在编译器后端才出现（第 §5 节）。

---

## §2 真实 FMHA 的逐段拆解

### 2.0 先认识它的整体结构

出厂默认路径叫 `fmha_fwd_hd128_mxfp8_auto`——这不是一个单独 kernel，而是**主机侧按形状挑选**
`_vs128` / `_t128` / `_kvsplit`（今天是 v3 版本 `fmha_fwd_hd128_mxfp8_v3_*`；见 `tests/gpu/run_ref.sh` 的 `*_auto` 分支与 `docs/current-status.md` §4.1）。
本文拆解单 kernel 快路径 **`_vs128`**：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:20824-20829
pub fn fmha_fwd_hd128_mxfp8_vs128() -> Function {
    VS_MODE.store(true, std::sync::atomic::Ordering::SeqCst);
    let f = fmha_t_build("fmha_fwd_hd128_mxfp8_vs128", false, true);
    VS_MODE.store(false, std::sync::atomic::Ordering::SeqCst);
    f
}
```

`fmha_t_build`（`kernels.rs:20951` 起）负责"公共前奏"（参数、合同、Q 载入、地址），然后把 8 个 wave 分成两组
**角色 A（wave 0–3）与角色 B（wave 4–7）**，各自调用 `emit_fmha_vs128_role(b, &cx, role_a)`（`kernels.rs:9602` 起）。
一句话算法：每次处理 128 个 key（一个 tile）——**A 组搬 K、B 组搬 V**（DMA 到 LDS 的两槽环形缓冲），两组错开半拍，
交替做 "QK 矩阵乘 → softmax → PV 矩阵乘"。

> 小提示：`VS_MODE` 是一个 `AtomicBool`（安全的原子变量，不是 `static mut`），只在**构建 kernel 时**选择代码形状。

每段下面都分三问：**用到的特性 / Rust safe 怎么保证 / 如果写错了会怎样**。

---

### (a) 前奏：656 字节参数区 + 主机合同

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:20951-21010（fmha_t_build，节选）
let mut b = FnBuilder::new(name, if name.ends_with("_t128") { 256 } else { 512 });
…
b.arg("ptr_o", DType::Ptr, 0x000, 8);
b.arg("ptr_q", DType::Ptr, 0x010, 8);
b.arg("ptr_k", DType::Ptr, 0x020, 8);
b.arg("ptr_v", DType::Ptr, 0x030, 8);
b.arg("s_seq_len", DType::I32, 0x060, 4);
…
b.arg("s_descale_k_Hs", DType::I32, 0x260, 4);
…
    b.arg("abi_tail", DType::I32, 0x280, 16);        // 0x280 + 16 = 0x290 = 656 字节
…
fmha_abi_contract(&mut b, is_kvs, if name.ends_with("_t128") { 128 } else { 256 });
```

`arg` 只描述布局；**合同**才告诉编译器"这些数之间有什么关系"：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:20880-20937（fmha_abi_contract，节选）
let hk = Ext::sym("HK");
b.ka_range(0x0B0, 1, 16);                               // gqa ∈ [1, 16]
b.ka_is(0x140, hk.clone() * Ext::ka(0x0B0));            // H = HK * gqa
b.grid_is(1, Ext::ka(0x140));                           // grid.y = H
b.grid_is(0, Ext::ka(0x060).ceil_div(rows_per_block));  // grid.x = ceil(S / 256)
b.ka_range(0x090, 128, 128);                            // head stride 固定 128
…
b.ka_is(0x0A0, Ext::ka(0x060) * Ext::ka(0x070));        // Q batch stride = S * (H*128)
…
b.arg_tensor("ptr_q", batch.clone() * Ext::ka(0x0A0));  // Q 分配了 B * batch_stride 字节
b.arg_tensor("ptr_k", batch.clone() * Ext::ka(0x0E0));
b.arg_tensor("ptr_v", batch.clone() * Ext::ka(0x170));
b.arg_tensor("ptr_o", batch.clone() * Ext::ka(0x1A0));
…
b.arg_tensor_slack("ptr_q_descale", qs.clone(), qs + Ext::c(512 * 4) * h.clone());
```

前奏里还有两处**显式信任**（`assume_trusted`）和一处**夹紧（clamp）**：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:21112-21120 —— 超出序列末尾的行，改用最后一个合法行（不越界，算出相同结果）
let s_last = b.add_i32(s_len, cm1);
let row_ok = b.cmp_lt_i32(row_raw, s_len);
…
    b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(row_ok), a: Opnd::Val(row_raw), b: Opnd::Val(s_last) })

// crates/amdgpu-kernels/src/probes/mod.rs:21184-21186 —— GQA：kv_head = head >> log2(gqa)
let kvh = if ablate_pre.contains("nogqa") { hidx } else { b.let_(Op::LshrI32(Opnd::Val(hidx), Opnd::Val(gshift))) };
b.assume_trusted(kvh, Ext::c(0), Ext::sym("HK") - Ext::c(1), "kv head = head >> log2(gqa) with head < H = HK * gqa (gqa a power of two <= 16): variable shift, no interval rule");
```

**用到的特性**：kernarg 布局声明（`arg`）；主机合同 API——`arg_tensor`（指针背后分配了多少字节）、`arg_tensor_slack`
（逻辑大小 + 已清零的尾部余量：读可用余量、写不可以）、`ka_range` / `ka_is`（参数取值范围 / 参数等式）、`grid_is`（网格维度等式）、
`Ext`（符号表达式：`Ext::ka(off)`、`Ext::sym("HK")`、`Ext::c(n)`）、`assume_trusted`（带理由的信任条目）。
合同**不生成任何指令**（`lib.rs:863-890`、`:956-958`），只给验证器当"前提"。

**Rust safe 怎么保证**：这一段在 Rust 层面只是往 `Vec` 里 push 结构体，没有任何危险操作。真正的价值在 L2：
验证器把每个参数当成**符号**（`ka(0x60)` 之类），用合同做**符号区间分析**：例如 Q 的读地址上界被化简成
`row*ka(0x70)+…`，再与 `grid_z * ka(0xA0)` 比较，证明"不越过分配"。`assume_trusted` 不是偷偷放行——
它**一定**出现在报告的信任账本里（见 §3 的 certificate）。

**如果写错了会怎样**：
- 某个指针没写 `arg_tensor` → `GV-GLB-NOEXT`："kernarg pointer \`p\` has no declared extent"（`gpu_verify.rs:854-857`）。
- 少一条形状关系（比如删掉 `ka_is(0x0A0, …)`）→ `GV-GLB-OOB`，并**打印符号残差**告诉你差在哪
  （测试 `global_access_without_the_shape_bound_is_unprovable_and_prints_the_expression`，`crates/amdgpu-kernels/tests/gpu_semantics.rs:101-107`）。
- 合同本身写错（声明了比主机真实分配更大的尺寸）→ 验证器**查不出**（这是假设 A1，`gpu-semantics-verifier.md` §1）；
  因此有 `crates/amdc/tests/contract_host_truth.rs` 拿合同和 host 公式对账。

---

### (b) LDS 布局：tile / ring / 私有区 / 别名

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:20861-20878（fmha_vs_tiles）
fn fmha_vs_tiles(b: &mut FnBuilder) {
    b.lds_ring("K", 0, 32768, 2);          // 两槽环：槽 k = [32768k, +32768)，K 在前 16 KB
    b.lds_ring("V", 16384, 32768, 2);      //            V 在后 16 KB
    b.lds_ring("KD", 65536, 512, 2);       // K 的缩放因子环
    b.lds_tile("ones", 67584, 2048);       // 常量块（fp8 的 1.0）
    b.lds_tile("Q1", 69632, 16384);
    b.lds_tile("Q0", 86016, 16384);
    // epilogue: every wave transposes through its own 4 KB scratch …
    b.lds_private("epi", 69632, 4096, 8);  // 8 个 wave 各自私有 4 KB
    b.lds_alias("epi", "Q1");
    b.lds_alias("epi", "Q0");
    b.lds_alias("epi", "V");
    b.lds_alias("K", "V");
    b.lds_alias("V", "KD");
    …
}
```

总大小由 `b.lds(VS128_LDS_BYTES)` 声明（`kernels.rs:20961`），`VS128_LDS_BYTES = 2*32768 + 2048 + 2048 + 16384 + 16384 = 102400`
（`kernels.rs:19536`）。

**用到的特性**：
- `lds_tile(name, base, size)`：一块有名字的 LDS 区域。
- `lds_ring(name, base, slot_size, slots)`：**环形缓冲**（ring）——像传送带上的两个托盘：第 t 个 tile 放 `t & 1` 号托盘，
  一个托盘在被读时，另一个托盘在装下一批货。
- `lds_private(name, base, size_per_wave, waves)`：**每 wave 私有**的切片（自己写、自己读，不需要 barrier）。
- `lds_alias(a, b)`：允许两块区域**时间复用**同一段存储（例如 epilogue 时 Q 的暂存区已经没用了，可以当转置草稿纸）。

**Rust safe 怎么保证**：依然只是声明数据。保证来自验证器的两类检查：
(1) **静态布局**：所有 tile 必须在 `lds()` 段内、两两不相交，除非显式 `lds_alias`；
(2) **生命周期协议**：后面每次读写都用 `tile_produce` / `tile_consume` 标注"我在写/读哪个 tile 的哪个槽、第几代"，
验证器模拟 barrier 序列证明"读到的就是期望的那一代"。

**如果写错了会怎样**：
- 两个 tile 重叠却没 `lds_alias` → `GV-TILE-OVERLAP`："live tiles … overlap and are not declared as aliases"。
- 标注了 tile，但地址算出了 tile 外 → `GV-TILE-ESCAPE`。
- 访问私有区却用了别的 wave 的切片 → `GV-TILE-PRIVATE-ESCAPE`（`gpu_verify.rs:740-773`）。

---

### (c) 搬运 K/V：`global_load_lds` DMA（以及 MUBUF 形式）

每个 wave 负责搬 tile 的 32 行，一条 DMA 指令搬 8 行 = 64 lane × 16 字节 = 1 KB，共 4 条：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:9807-9833（emit_fmha_vs128_role 内的 dma_issue_global，节选）
let dma_issue_global = |b: &mut FnBuilder, key0: ValueId, buf_off: ValueId| {
    let rb = b.add_i32(key0, rowb);
    let lbase = b.add_i32(lds_tile, buf_off);
    for q in 0..4i32 {
        let cq = b.c_i32(8 * q);
        let row_raw = b.add_i32(rb, cq);
        let row = if fmha_t_opt("dmin") {
            b.min_i32(row_raw, cx.kv_last)
        } else {
            let ok = b.cmp_lt_i32(row_raw, cx.kv_len);
            b.let_(Op::Select { ty: DType::I32, cond: Opnd::Val(ok), a: Opnd::Val(row_raw), b: Opnd::Val(cx.kv_last) })
        };                                            // ← 行号夹紧到最后一个合法 key
        let o0 = b.mul_i32(row, seqs);
        …
        let sp = b.ptr_add(ptr, o3);
        b.load_global_lds(sp, lbase, -(1024 * (3 - q)), 4);   // ← global → LDS 直达 DMA
    }
};
```

DMA 必须"包"在 tile 标注里，并在之后等待完成：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:9913-9943（stage_start / stage_end，节选）
let stage_start = |b: &mut FnBuilder, key0: ValueId, slot_off: ValueId| -> Option<ValueId> {
    …
    let (tslot, tver) = stage_tag.get();
    let kd = if role_a && !kdglob {
        if kddma {
            b.tile_produce("KD", tslot, tver);
            kd_dma(b, key0, slot_off);                // K 缩放因子也走 DMA（1 dword/lane）
            b.tile_end();
            …
    b.tile_produce(if role_a { "K" } else { "V" }, tslot, tver);
    dma_issue(b, key0, slot_off);
    b.tile_end();
    kd
};
let stage_end = |b: &mut FnBuilder, kd: Option<ValueId>, kd_slot: ValueId| {
    if !ablate.contains("novmwait") {
        b.wait_vm(0);                                 // ← 等本 wave 的 DMA 全部落地
    }
    …
};
```

**MUBUF 形式**（`buffer_load … lds`，带硬件范围检查的描述符）当时作为**可选实验**写在同一个函数里（`FMHA_OPT=mubuf/mubufd`，默认关闭；该实验与开关已在 2026-11 整理中删除，下面的片段是历史记录）：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:9781-9790 与 :9869-9873（节选）
let rsrc: Option<ValueId> = if mubufd {
    let kvb = b.mul_i32(cx.kv_len, seqs);
    let nrec0 = b.add_i32(kvb, base_off);
    let nrec = b.read_first_lane(nrec0);
    Some(b.buffer_desc_dyn(ptr, nrec))         // 运行时 num_records：越界读由硬件返回 0
} else if mubuf { Some(b.buffer_rsrc(ptr)) } else { None };
…
b.begin_if(Opnd::Val(full));
for q in 0..4usize {
    b.load_buffer_lds(rsrc.expect("rsrc"), voffq[q], soff, m0v, (1024 * q) as u32, 4);
}
b.end_if();
```

**用到的特性**：`load_global_lds`（`Stmt::LoadGlobalLds`：lane i 把 16 字节写到 `lds_addr + 16i`，**完成只由 `vmcnt` 跟踪**，
`lib.rs:597-599`）；`load_buffer_lds` + `buffer_rsrc/buffer_desc_dyn`（MUBUF 描述符，类型是专门的 `DType::Rsrc`）；
`tile_produce/tile_end` 标注；`wait_vm(0)`；Rust 闭包（closure）把重复逻辑封装起来。

**Rust safe 怎么保证**：
- 注意每个闭包都把 `b: &mut FnBuilder` 作为**参数**接收，而不是"捕获"它。这是借用检查器逼出来的写法：
  同一时刻只能有一个 `&mut` 写者，若好几个闭包都长期捕获 `&mut b`，编译就会报 `E0499/E0502`。
  于是"谁在往 kernel 里加指令"在任何时刻都是唯一的。
- 闭包之间共享的 `(slot, version)` 标签用 `std::cell::Cell` 存（`kernels.rs:9909-9911`），这是标准库提供的**安全**内部可变性，不需要 `unsafe`。
- 地址安全在 L2：验证器把 DMA 的**64 个 lane 全部**计入目的区间 `[M0+imm, M0+imm+64*16)`，证明它在 `K`/`V` 环的当前槽内、
  也在 `lds()` 段内；源地址则靠夹紧（`Select`/`min`）+ 合同证明不越过 `ptr_k/ptr_v` 的分配。
- 描述符：`buffer_desc*` 在构造点检查 `num_records ≤ 分配`（"越界读返回 0"只有在描述符不夸大时才成立）。

**如果写错了会怎样**：
- DMA 目的地超出声明段 → `GV-LDS-OOB`；超出所标 tile → `GV-TILE-ESCAPE`。
- 去掉夹紧（例如去掉 Q 行夹紧）→ 行号上界变成 `256·grid_x − 1`，可能超过 `S − 1`，地址证不出上界 → `GV-GLB-OOB`。
- 描述符 `num_records` 夸大 → `GV-BUF-OVER`（`gpu_verify.rs:1193-1212`；攻击样例 `gv_attack_desc_over`，`kernels.rs:22395-22408`）。
- 忘了 `wait_vm(0)` 就靠 barrier 发布 → `GV-TILE-STALE|GV-PROTO-NO-COMPLETION`（§4 有完整对照）。
- 诚实说明：opt-in 的 `mubuf/mubufd` 变体**目前未认证**（会报 `GV-BUF-OVER`/`GV-GLB-OOB`），它们不在出厂路径上
  （`gpu-semantics-verifier.md` 中 13.6 节的说明）。

---

### (d) QK 矩阵乘：带缩放的 fp8 MFMA

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:10255-10310（qk 闭包，节选）
for s in 0..4usize {                                    // 128 个 key = 4 个 32-key 的 S 块
    if biasf_once.is_none() && !fmha_t_opt("biasv") && bacc.is_none() {
        b.var_mfma_f16_z(s_vars[s], ones_f16, Opnd::Val(bias_v));   // S 先放入参考偏移（f16 MFMA）
    }
    …
    let (qs_slot, qs_ver) = qk_tag.get();
    b.tile_consume("K", qs_slot, qs_ver);               // 我要读 K 环的这个槽、这一代
    let k8_0 = … b.load_lds_b128x2(Opnd::Val(a0), Opnd::Val(a1)) …;
    let k8_1 = … b.load_lds_b128x2(Opnd::Val(a2), Opnd::Val(a3)) …;
    b.tile_end();
    …
    let sk0 = b.let_(Op::LshrI32(Opnd::Val(kd), Opnd::Val(cx.g8)));   // K 的 E8M0 缩放字节
    …
    } else if let Some(ba) = bacc {
        b.var_mfma_scale_c(s_vars[s], k8_0, qb0, Opnd::Var(ba), sk0, sq0m);  // C 操作数形式（opt-in）
    …
    } else {
        b.var_mfma_scale(s_vars[s], k8_0, qb0, sk0, sq0m);           // 默认：原地累加
    }
    let qb1 = if let Some((_, q1)) = q_pre { q1 } else { load_q1(b) };
    b.var_mfma_scale(s_vars[s], k8_1, qb1, sk1, sq1m);               // head dim 后 64 维
}
```

**用到的特性**：
- **片段类型（fragment）**：`Frag8` = 每 lane 8 个 dword（MFMA 的 A/B 操作数），`Frag16` = 每 lane 16 个 dword
  （32×32 fp32 累加器，分摊到 64 个 lane）。`load_lds_b128x2` 用两次 16 字节 LDS 读拼出一个 `Frag8`。
- **循环变量（`VarId`）**：`s_vars[s]` 是循环里跨迭代保存的累加器；`var_mfma_scale(var, a, b, scale_a, scale_b)` 表示
  `var = scaled_mfma(a, b, var)`，原地更新（`lib.rs:1525-1541`）。`var_mfma_scale_c` 是"C 操作数与目的寄存器分开"的形式（`lib.rs:1471-1473`），
  在 `_vs128` 里属于 opt-in（`biasacc`/`biasonce`）。
- **MX 缩放**：`scale_a/scale_b` 是每块的 2 的幂指数（E8M0），硬件在乘时应用。
- `tile_consume("K", slot, version)`：声明这次读期望 K 环某槽的某一代。

**Rust safe 怎么保证**：
- `ValueId`（u32）和 `VarId`（`pub struct VarId(pub u32)`，`lib.rs:22-25`）是**不同类型**：把普通值误当成循环变量传给
  `var_mfma_scale` 会被 rustc 拒绝（`E0308 mismatched types: expected VarId, found u32`）。这是"新类型（newtype）"手法。
- 操作数宽度由 IR 验证器检查：`scaled mfma: operand a is frag4, expected frag8`（`lib.rs:1970-1985`），编译返回 `Err`，不会生成坏代码。
- 在 MIR 层（寄存器分配之后）验证器还检查：MFMA 的 A/B 与目的 D **不重叠**（`GV-REG-MFMA`）、多 dword 元组从**偶数** VGPR 开始（`GV-REG-ALIGN`，`gpu_verify.rs:1655-1657`，规则经 gfx950 汇编器实测）。

**如果写错了会怎样**：
- 把 `Frag4` 当 `Frag8` 喂 → IR 验证器报 "operand … expected frag8"，编译失败。
- 读 K 时版本标错（比如读了 `it` 而不是 `it+1` 的那一代）→ `GV-TILE-STALE|GV-PROTO-STALE-GEN`。
- 寄存器分配产出奇数起点元组（编译器 bug）→ `GV-REG-ALIGN`，**默认就让编译失败**（MIR 级错误在默认策略下即阻断）。

---

### (e) online softmax：最大值、回滚、`pk` 运算、`exp2`

这个 kernel 的 softmax 不是"每个元素调一次 `exp`"的教科书写法：S 里已经加了一个"参考偏移"（上一段的 f16 MFMA），
只有当新的行最大值**明显**超过参考时才"回滚"（rebase），把累计结果按 `2^(512·delta)` 缩放。

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:10368-10378（valu 闭包：一次求 64 个分数的最大值）
let mut mloc = b.max3_f32(flat[0], flat[1], flat[2]);
…
let mt = b.partner_max_f32(mloc);          // 和搭档 lane（l ^ 32）交换取 max
let grow = b.cmp_lt_f32(c_thr, mt);        // 超过阈值才需要回滚

// crates/amdgpu-kernels/src/probes/mod.rs:10448-10495（回滚分支，节选）
let nb = b.frag2(w_new, w_new);
b.set_var(bias2, nb);
b.begin_if(Opnd::Val(grow));
{
    …
    // the branch is taken when ANY lane rolls back and runs without an exec mask: lanes that do not roll back must see delta = 0
    let dupd = b.frag2_dup(delta);
    …
        for s in 0..4usize { b.var_add_pk(s_vars[s], dupd); }      // S += delta（v_pk_add_f32）
    let ex = b.mul_f32(delta, c512f);
    let corr = b.exp2_f32(ex);                                       // 2^(512·delta)
    let cd = b.frag2_dup(corr);
    …
        for e in 0..4usize { b.var_scale_pk(o_vars[e], cd); }      // O *= corr（v_pk_mul_f32）
    if smfv {
        b.var_scale_lane(la, 2, corr);                               // 行和也缩放
    …
}
b.end_if();
```

然后把 S 转成 fp8 的 P（`cvt_pknorm_*`，`kernels.rs:10516-10531`）：短序列那份代码副本带 LUT 校正
（`cvt_pknorm_pair` → `perm_b32` → `pk_sub_u16` → `cvt_p8x4_from_u16`），长序列副本直接 `cvt_pknorm_p8x4`。

**用到的特性**：`max3_f32`、`partner_max_f32`（跨 lane 交换）、`Frag2` 打包对 + `var_add_pk/var_scale_pk`（一次处理两个 fp32 的 packed 运算）、
`exp2_f32`（超越函数）、`set_var`（写循环变量）、`begin_if/end_if`。

**Rust safe 怎么保证**：
- **`begin_if` 是 wave 级分支**（没有 exec 掩码）：只要有任一 lane 需要回滚，整个 wave 都进分支。代码注释明确要求
  "不回滚的 lane 必须看到 `delta = 0`"——这是**数值**正确性，不是 Rust 能保证的，而是作者用 `Select` 保证（`kernels.rs:10425`）。
- **循环变量规则**（`var_lint`，`lib.rs:2485-2492`）：同一迭代里 `set_var` 之后再读这个变量，DSL 不保证读到新值还是旧值，
  因此 lint 报告；上面代码把新值先存在 SSA 值 `nb` 里再 `set_var`，后续都用 `nb`（`Some(nb)` 被返回），就是遵守这条规则。
- 超越函数结果需要等待若干周期才能读：在 DSL 层由编译器的 hazard pass（`hazards.rs:134` `fix_value_hazards`）自动补等待；
  在 safe-asm 层则由 `Trans` token 在类型上强制（§2(i)）。

**如果写错了会怎样**：
- 在 `set_var(bias2, nb)` 之后又读 `Opnd::Var(bias2)` → lint："loop variable var… is read after set_var … use the SSA value passed to set_var"。
- 在 `if` 外使用 `if` 内定义的值 → IR 验证器："operand v… is not visible here"（`lib.rs:1680-1682`）。
- 算法写错（例如 `delta` 没对不回滚的 lane 置零）→ **没有任何验证器会报**，只能靠精度门禁（15 形状对拍、确定性测试）发现。

---

### (f) PV 矩阵乘与行和（smfmac）

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:10610-10627（load_v：转置读 V，标注消费 V 环）
let load_v = |b: &mut FnBuilder, hh: usize, e: usize| -> ValueId {
    …
        let (pv_slot, pv_ver) = pv_tag.get();
        b.tile_consume("V", pv_slot, pv_ver);
        let vv = b.load_lds_tr_b8x4(addrs[0], addrs[1], addrs[2], addrs[3]);   // 4× ds_read_b64_tr_b8
        b.tile_end();
        vv
};

// crates/amdgpu-kernels/src/probes/mod.rs:10650-10669（PV 主体，节选）
for hh in 0..2usize {
    let p8 = match &p_pre { Some(ps) => ps[hh], None => b.frag8(w[o], …) };
    …
    for e in 0..4usize {
        let v8 = … load_v(b, hh, e) …;
        b.var_mfma_scale(o_vars[e], v8, p8, c127, c123);      // O[e] += V^T · P
    }
}

// crates/amdgpu-kernels/src/probes/mod.rs:10561-10567（行和：4 条稀疏 MFMA 原地累加）
if smfv {
    b.var_smfmac_fp8(la, a_rs, pa, idx01);
    b.var_smfmac_fp8(la, a_rs, pb, idx01);
    b.var_smfmac_fp8(la, a_rs, pa, idx23);
    b.var_smfmac_fp8(la, a_rs, pb, idx23);
    return;
}
```

**用到的特性**：`load_lds_tr_b8x4`（转置读：读出来就是 MFMA 需要的布局）、`frag8` 把 8 个 dword 组成操作数、
`var_mfma_scale`（O 的 4 个 32×32 块累加）、`var_smfmac_fp8`（`v_smfmac_f32_16x16x128_fp8_fp8`，用"全 1"稀疏矩阵算出每行 P 的和；
结果的 dword 2 = 16 × 行和，所以 epilogue 乘 `0.0625`，`kernels.rs:10866-10870`）。

**Rust safe 怎么保证**：与 (d) 相同——类型化片段 + IR 验证 + MIR 寄存器合法性；
`la` 是 `Frag4` 循环变量，`var_smfmac_fp8` 只接受 `VarId`。V 的读被 `tile_consume("V", idx, it)` 标注，
验证器证明它读到的恰是"本轮 B 组搬来、已被 barrier 发布"的那一代，并且下一代 DMA 不会在读完之前落到同一槽。

**如果写错了会怎样**：
- 把 V 的读挪到 barrier 之前 → `GV-TILE-STALE|GV-PROTO-UNPUBLISHED`（"finds NO published version"）。
- 下一轮的 DMA 提前到 V 读完之前 → `GV-TILE-RACE|GV-PROTO-RACE`（读窗口与别代写窗口相交）。
- 读完没 `wait_lds(0)` 就过 barrier → `GV-TILE-PENDING`（Warn，"barrier 不能把它和下一个写者隔开"）。

---

### (g) epilogue：归一化、bf16、8 字节存储；KV-split 的私有转置草稿纸

`_vs128` 单 kernel 路径：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:11259-11291（节选）
let o_ptr0 = b.ptr_add(cx.ptr_o, ob4);
let wave_live = b.cmp_lt_i32(cx.row0, cx.s_len);
b.begin_if(Opnd::Val(wave_live));                     // 整个 wave 都在序列外就不写
for e in 0..4usize {
    for q in 0..4usize {
        …
            v4.push(b.mul_f32(lanev, fin));           // O * (V_descale / l)
        let w0 = b.cvt_pk_bf16_f32(v4[0], v4[1]);
        let w1 = b.cvt_pk_bf16_f32(v4[2], v4[3]);
        let off = (64 * e + 16 * q) as u32;
        if fmha_t_opt("ost1") { … } else {
            // one 8-byte store (`global_store_dwordx2`) instead of two 4-byte stores (`ost1` restores them)
            let f2 = b.frag2(w0, w1);
            b.store_f32x2(o_ptr0, off, f2);
        }
    }
}
b.end_if();
```

同一个角色函数在 `_kvsplit` 下走另一条 epilogue：每个 wave 先把结果写进**自己私有的 4 KB LDS 草稿纸**、转置后再合并写出：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:10947-10951 与 :11024-11041（节选）
let wid_e = if role_a { cx.wave4 } else { b.add_i32(cx.wave4, c4) };
let c4096e = b.c_i32(4096);
let wb0 = b.mul_i32(wid_e, c4096e);
…
let lbase = b.add_i32(wb0, c69632);                   // 地址 = 69632 + wid_e*4096 + …
…
b.tile_produce("epi", Opnd::Val(wid_e), Opnd::I32(0)); // 私有区：槽 = 本 wave 的 wid_e
b.store_lds_b32(Opnd::Val(ad), Opnd::Val(w0));
…
b.tile_end();
…
b.wait_lds(0);                                        // 自己写完 → 自己读，不需要 barrier
for i in 0..4usize {
    b.tile_consume("epi", Opnd::Val(wid_e), Opnd::I32(0));
    let v = b.load_lds_b128(Opnd::Val(raddr[i]));
    b.tile_end();
    … b.store_f32x4(gptr[i], (128 * pr) as u32, v); …
}
```

而在进入这段之前，有一个专门修过真实 bug 的 barrier：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:10858-10864（节选）
// … a wave that finishes first could overwrite a partner wave's still-live Q staging during its last QK/PV …
if !no_bar && cx.kvs.is_some() {
    b.barrier();
}
```

**用到的特性**：`cvt_pk_bf16_f32`、`frag2` + `store_f32x2`（`global_store_dwordx2`）、`begin_if(wave_live)`、
`lds_private` 配合 `tile_produce/consume(name, wave_key, version)`。

**Rust safe 怎么保证**：私有区的规则写在 `lib.rs:899-904`：`wave_key` **必须就是地址计算所用的同一个值**。
验证器把地址线性化，证明 `地址 − wave_key×4096 − 69632 ∈ [0, 4096 − 访问宽度]`——也就是"只碰自己的切片"；
并且读必须读到**本 wave 自己已完成**（`wait_lds(0)`）的版本。私有区与 `Q1/Q0/V` 时间复用，所以声明了 `lds_alias`，
验证器据此要求它们的事件按"同一 tile 的先后版本"排序。全局写同样在合同下证明不越过 `ptr_o`（或 `ptr_ohat`）的分配。

**如果写错了会怎样**：
- 用 `wid ^ 1`（邻居的切片）算地址 → `GV-TILE-PRIVATE-ESCAPE`："another wave's slice may be touched"。
- 写完没 `wait_lds(0)` 就读 → `GV-TILE-STALE`（"not completed by this wave"）。
- 上面那个 barrier 是修真实 bug 时加的（代码注释：先结束的 wave 会覆盖搭档 wave 仍在用的 Q 暂存区，fp16 Ohat 转置偶发出错）。
  它提醒我们：**时间复用（`lds_alias`）的区域，"谁先用完"必须由 barrier 排序**，这正是协议检查要覆盖的那类错误。（注：`_kvsplit` 还有 160 处 epilogue 访问是 `GV-TILE-UNTAGGED` 的 Warn——界已证、生命周期未证，属于诚实清单。）

---

### (h) barrier、waitcnt 与 A/B 角色分工

角色的分裂发生在 `fmha_t_build`：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:21280 与 :21528-21542（节选）
_ => b.cmp_lt_i32(wave, c4),              // is_a：wave 0-3 是角色 A
…
for pass in 0..2u8 {
    if pass == 0 {
        b.begin_if(Opnd::Val(short));     // 短序列 / 长序列两份代码（工作组一致的条件）
        …
    } else {
        b.else_();
        …
    }
    b.begin_if(is_a);                     // 角色分裂（依赖 wave 号 → 不同 wave 走不同分支）
    vs_role_fn(&name)(&mut b, &cx, true);
    b.else_();
    vs_role_fn(&name)(&mut b, &cx, false);
    b.end_if();
}
b.end_if();
```

每轮迭代的"节拍"写在 `iter` 闭包里：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:10675-10832（iter，节选）
let iter = |b: &mut FnBuilder, masked: bool, stage: bool, last_barrier: bool| {
    let (tag_i1, slot_cur, slot_nxt, slot_n2, kds_nxt, key_n2, _kds_cur, tag_it_p1) = ring(b);
    stage_tag.set((Opnd::Val(tag_i1), Opnd::Val(tag_it_p1)));   // 下一 tile：槽 idx^1，第 it+1 代
    qk_tag.set((Opnd::Val(tag_i1), Opnd::Val(tag_it_p1)));
    pv_tag.set((Opnd::Var(idx), Opnd::Var(it)));                 // 当前 tile：槽 idx，第 it 代
    let skd = if stage && !(dsplit && role_a) { stage_start(b, key_n2, slot_n2) } else { None };  // 发 DMA
    …
    let mut w = … valu(b, masked, true, 0, 4) …;                 // softmax（V 相位）
    …
    if stage && !dsplit { stage_end(b, skd, kds_nxt); }          // wait_vm(0)：DMA 完成
    …
        b.wait_lds(0);
    if !no_bar && !nobar1 { b.barrier(); }                       // 屏障 1：发布刚落地的 tile
    …
        pv_w(b, slot_cur, &w);                                   // M 相位：PV(t)
        …
        if stage { qk(b, slot_nxt, kds_nxt, kdg.as_deref(), bias_now); }   // + QK(t+1)
    …
    b.wait_lds(0);
    if !no_bar && !nobar2 && last_barrier { b.barrier(); }       // 屏障 2：读完才允许覆盖
};
```

两组错开半拍：B 组在开头多一个 barrier（`kernels.rs:10075-10077`："P1 (group A computed QK(0) before its P1)"），
A 组在最后的剥离迭代多一个（`iter(b, true, false, role_a)`，`kernels.rs:10849-10855`），所以**总数相等**。

**用到的特性**：`begin_if/else_/end_if`、`begin_loop/break_if/end_loop`、`barrier()`、`wait_vm(n)`（`s_waitcnt vmcnt`）、
`wait_lds(n)`（`s_waitcnt lgkmcnt`）、`set_prio`。

**Rust safe 怎么保证**：这是"四轴协议"的主场（`gpu-semantic-protocol.md` §1）：
**到达**（barrier 只表示人到齐）、**完成**（每个 wave 自己的 `wait_*`）、**可见**（LDS：完成之后的第一个 barrier 发布）、
**代际**（槽/版本是循环计数器 `it`、`idx` 的仿射/周期函数）。编译器静态证明四条义务：
O1 完成先于发布、O2 发布先于读、O3 读完先于覆写、O4 代际一致。另外：
- 依赖 wave 号的 `if/else` 两侧动态 barrier 数必须相等，否则死锁（`GV-BAR` / `GV-PROTO-ARRIVAL-MISMATCH`）；
  `_vs128`（旧手写版与 v3 版相同）两侧各 18 个（`current-status.md` §2.1）。
- MIR 层还有**路径敏感**的"用前等待"检查：任一路径（含分支一侧、含跨迭代回边）读了在途加载的寄存器 → `GV-WAIT`。
- 旧手写 `_vs128` 的协议模拟检查了 **58,880 个读事件**（v3 `_vs128` 为 73,600 个），0 错误（`gpu-semantics-verifier.md` §3.1）。

**如果写错了会怎样**：
- 删掉 B 组开头那个 barrier → `GV-BAR|GV-PROTO-ARRIVAL-MISMATCH`："wave-divergent if/else … executes a different number of barriers"。
- 删掉 `stage_end` 的 `wait_vm(0)` → `GV-TILE-STALE|GV-PROTO-NO-COMPLETION`。
- 删掉"屏障 2" → 下一轮 DMA 与本轮读重叠 → `GV-TILE-RACE|GV-PROTO-RACE`。
  （变异测试：逐个删除 `_vs128` 的 barrier，30 个中 24 个被发现，其余 6 个的说明见 `gpu-semantics-verifier.md` §3.1。）

---

### (i) 可选：safe-asm 窗口桥（`Op::Asm` / `Stmt::VarAsm`）

DSL 也能嵌入一段"手写汇编"，但手写部分同样是**安全 Rust**（`crates/amdgpu-safeasm`，同样 `#![forbid(unsafe_code)]`）。
行和那 4 条 smfmac 可以换成 safe-asm 写的等价块（当时由 `FMHA_OPT=safeasm` 打开，非默认；该开关已删除，桥本身 `Op::Asm` / `Stmt::VarAsm` 仍在）：

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:10555-10560
if smfv && fmha_t_opt("safeasm") {
    // the same four in-place SMFMACs, authored in safe Rust (crates/amdgpu-safeasm) and embedded through Stmt::VarAsm
    use amdgpu_safeasm::examples::rowsum_var_block;
    b.var_asm(la, rowsum_var_block(), &[Opnd::Val(a_rs), Opnd::Val(pa), Opnd::Val(pb), Opnd::Val(idx01), Opnd::Val(idx23)]);
    return;
}

// crates/amdgpu-safeasm/src/examples.rs:67-83（rowsum_var_block）
pub fn rowsum_var_block() -> AsmBlock {
    let mut s = Seq::new();
    let acc = s.inout::<4>();
    let a_rs = s.input::<4>();
    let pa = s.input::<8>();
    …
    let (ra, rpa, rpb, ri01, ri23) = (s.ready(&a_rs), s.ready(&pa), s.ready(&pb), s.ready(&idx01), s.ready(&idx23));
    let t = s.v_smfmac_f32_16x16x128_fp8_fp8(acc, ra, rpa, ri01);
    let t = t.chain(&mut s, ra, rpb, ri01);
    …
    let ex = t.exit();
    s.output_exit(ex);
    s.finish().expect("rowsum_var_block is a verified sequence")
}
```

（更大的窗口 `pv_window_block` / `pv_qk_window_block` 曾在已删除的 `_vs5` 实验变体中接入，历史记录。）

**用到的特性**：`Seq`（安全的指令序列构造器）、`Vgpr<N>`（宽度是类型参数）、`Ready<'a, N>`（"这个寄存器现在可被矩阵指令读"的**借用证明**）、
`Mfma/Smfmac<'a, OP>`（在途矩阵运算的 token）、`Pending<T>`（在途加载的 token）。

**Rust safe 怎么保证**：这里是 Rust **生命周期/所有权真正上岗**的地方（`rust-safety-by-example.md` Part D）：
- `Ready<'a, N>` 内含 `PhantomData<&'a ()>`（`regs.rs:85-88`），只要在途的 `Mfma<'a, …>` 还活着，操作数的共享借用就在，
  **任何覆盖操作数的 `&mut` 都被借用检查器拒绝**（E0502）。
- `Pending<T>` **拥有**目的寄存器（`seq.rs:53-63`）：`wait` 之前寄存器已"被移走"，读它就是 E0382；`wait` 消费 token，等两次也是 E0382。
- 编译器再验一次：`var_asm` 的端口类型由 IR 验证器核对，块内容 lowering 后照常跑 hazard/wait/`pin_check`。

**如果写错了会怎样**：见 §4 的 E0382 / E0308 / E0502 / E0080 示例。另外请记住：
**safe-asm 块内部的访存不在 `gpu_verify` 的越界证明范围内**（报告里会有 `GV-ASM` Info），这是诚实清单里的一项。

---

## §3 safety 到底由什么保证（分层解释）

| 层 | 保证什么 | 靠什么 | 什么时候检查 |
|---|---|---|---|
| **L1 Rust 语言安全** | 编译器/DSL 这个**宿主机程序**本身没有未定义行为 | `#![forbid(unsafe_code)]`、零第三方依赖、所有权/借用、类型系统 | `cargo build` 时 |
| **L2 GPU 语义（模型内证明）** | 生成的 GPU 程序**不越界、不读未完成/未发布的数据、寄存器合法** | IR 验证器 + `gpu_verify`（区间分析 + 协议模拟）+ MIR 检查 | 编译 kernel 时 / `amdc gpu-verify` |
| **L3 不保证的东西** | 数值对不对、跨工作组竞争、i32 回绕…… | 硬件门禁：精度对拍、确定性、1200 次启动压力 | 测试时 |

### L1：Rust 语言安全

- **全部 crate 禁止 `unsafe`**：`amdgpu-dsl`、`amdgpu-compiler`、`amdgpu-safeasm`、`amdc` 的根文件都写着 `#![forbid(unsafe_code)]`，
  测试 `crates/amdgpu-compiler/tests/rust_safety.rs::every_crate_forbids_unsafe_code` 机器检查；另有逐行扫描 16 类危险构造（`transmute`、`static mut`、裸指针转换……），0 命中。
- **零第三方依赖**：只有工作区内的 path 依赖（`no_third_party_dependencies` 测试）。
- **DSL 层是"索引 IR"，不是借用管理寄存器**：`ValueId` 是 u32 下标。这里 Rust 提供的是"宿主程序安全 + 类型区分（`ValueId` vs `VarId` vs `Opnd`）+ 唯一写者（`&mut FnBuilder`）"，
  寄存器生命周期则交给编译器验证器（L2）。**不要把 DSL 层说成"借用检查器管 GPU 寄存器"**——那是 safe-asm 层的事。
- **safe-asm 层：生命周期真正在工作**：`Ready<'a, N>` / `Mfma<'a, OP>` 的借用阻止 WAR（E0502）；
  `Pending<T>` 的 move 阻止"等待前读"和"等两次"（E0382）；`Pw<L, H>` typestate 阻止读半写的 dword；
  const 断言阻止非法立即数/对齐（E0080）。28 个 `crates/amdgpu-safeasm/tests/ui/fail_*.rs` 是负例目录。
  "忘了 wait" 这种**遗漏**编译期抓不到（Rust 没有线性类型），由 token 的 `Drop` 记录、`Seq::finish()` 返回 `Err`。

### L2：已经被检查的 GPU 语义

1. **extent 合同 + 符号区间分析**（`gpu_verify.rs` 开头注释）：每个全局/LDS/buffer 访问证明 `下界 ≥ 0` 且 `上界 + 宽度 ≤ 上限`；
   多项式判定是"充分"的——证不出 ≠ 越界，但证出来一定在合同下成立。循环变量做不动点；计数器写 `assume`，且必须**归纳成立**，否则 `GV-ASSUME-VIOLATED`。
2. **LDS tile 协议**：发布 = "完成之后的第一个 barrier"；代际 = 计数器的仿射/周期函数，模拟轮数覆盖周期（周期门）；
   多生产者（同角色多个 wave 协作填一个 tile）/ 多消费者（每个 wave 独立读）上的 **O1–O4** 双向义务全部静态 discharge。
3. **路径敏感等待**（`GV-WAIT`）：MIR 上的 scoreboard 数据流，覆盖分支一侧与循环回边。
4. **寄存器合法性**：保留寄存器（v0、s0–s7）、元组对齐、上限、MFMA 操作数与目的不重叠。

### L3：没有保证的（诚实清单）

- **数值正确性**：验证器不知道你算的是不是 softmax。
- **跨工作组的数据竞争 / 别名**、跨 kernel 的可见性（除显式 last-arriver 模式外）。
- **i32 回绕**（假设 A3：地址算术不溢出 2^31）。
- **从内存读出来的指针/索引**、**safe-asm 块内部的访存**。
- **合同本身的真实性**（声明的分配是否等于 host 真正传的大小）——靠 `contract_host_truth` 测试对账。

### 可计数的信任账本

所有"逃生口"都会被**数出来并打印**（`amdc/src/main.rs:187-196`）：

```
certificate: N trusted assumes (a assume_trusted, p tile_trust_periodic, r region_trust), M trusted contracts, K unproven
```

出厂 v3 四个 kernel `--shipped --strict` 的合计：**24 trusted assumes / 100 trusted contracts / 0 unproven**（`current-status.md` §2.1）。
"认证" = **K = 0，且评审接受这 24 + 100 条**。评审只需要读这些行（例如上文 `kvh` 的那条 `assume_trusted` 及其理由）。

---

## §4 演示"写错"：编译器/验证器拒绝什么

### 4.1 等待之前就读加载结果 → `E0382`（safe-asm，编译期）

```rust
// crates/amdgpu-safeasm/tests/ui/fail_use_load_destination_before_wait.rs
// error-contains: borrow of moved value: `dst`
let dst = s.vgpr::<4>();
let pending = s.ds_read_b128::<0, _>(dst, &addr); // the destination register moves into the Pending token
let _r = s.ready(&dst); // using it before `wait`: refused
let _v = pending.wait(&mut s);
```

改对：先 `let d = pending.wait(&mut s);`，再用 `d`。同理 `fail_double_wait.rs`：等两次 → `use of moved value: \`pending\``。

### 4.2 拿裸寄存器喂矩阵指令 → `E0308`（类型不匹配）

```rust
// crates/amdgpu-safeasm/tests/ui/fail_matrix_operand_without_ready.rs
// error-contains: mismatched types
let rb = s.ready(&b);
let rsa = s.ready(&sa);
// a plain `&Vgpr` is not accepted by a matrix instruction: only `Ready` (the 2-wait VALU->matrix rule)
let _t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, &a, rb, rsa, rsa);
```

rustc 的提示是 "expected `Ready<'_, 8>`, found `&Vgpr<8>`"——矩阵指令只接受"已满足等待态"的借用证明。
（DSL 层的同类错误：把 `ValueId` 传给要 `VarId` 的 `var_mfma_scale`，同样是 E0308。）

### 4.3 改写在途 MFMA 的操作数 → `E0502`

```rust
// crates/amdgpu-safeasm/tests/ui/fail_write_operand_of_inflight_mfma.rs
let (ra, rb, rsa) = (s.ready(&a), s.ready(&b), s.ready(&sa));
let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsa);
let _w = s.lane_mut::<8, 0>(&mut a); // overwriting an operand the MFMA is still reading: refused
let _o = t.retire(&mut s);
```

### 4.4 非法立即数 → `E0080`（编译期常量求值失败）

```rust
// crates/amdgpu-safeasm/tests/ui/fail_inline_constant_range.rs
// error-contains: inline constants are -16..=64
let _i = Imm::<100>::new();
```

### 4.5 LDS 写出声明段 → `GV-LDS-OOB`

```rust
// crates/amdgpu-kernels/tests/gpu_semantics.rs:23-48（节选）
fn lds_kernel(lds_bytes: u32, offset: i32) -> Function {
    let mut b = FnBuilder::new("lds_probe", 64);
    b.lds(lds_bytes);
    let t = tid(&mut b);
    let c16 = b.c_i32(16);
    let a = b.mul_i32(t, c16);                 // lane 63 写 [1008, 1024)
    …
    b.store_lds_b128(a, v);
    b.finish()
}
// lds_kernel(1024, 0) → 证明通过；lds_kernel(512, 0) → GV-LDS-OOB
```

消息形如（`gpu_verify.rs:839`）：
`stmt N: ds_write_b128: LDS access [addr, addr+16) not provably inside the declared 512-byte segment: upper bound not provably <= limit - 16: address [0, 1008] vs limit [512]`。
负偏移（`lds_kernel(4096, -16)`）会报 "lower bound not provably >= 0"——**下溢也抓**。

### 4.6 DMA 没等就靠 barrier 发布 → `GV-TILE-STALE|GV-PROTO-NO-COMPLETION`

```rust
// crates/amdgpu-compiler/tests/gpu_tile_attacks.rs:29-56（节选）
b.tile_produce("t", Opnd::I32(0), Opnd::I32(0));
b.load_global_lds(q, Opnd::I32(0), 0, 4);        // 角色 0 发 DMA
b.tile_end();
…
if wait_vm {
    b.wait_vm(0);                                // ← 有这一行：干净；没有：被拒
}
b.barrier();
…
b.tile_consume("t", Opnd::I32(0), Opnd::I32(0)); // 角色 1 读
let _ = b.load_lds_b32(Opnd::I32(0));
```

`dma_without_wait_vm_is_stale` / `dma_with_wait_vm_is_clean` 两个测试正是这对 before/after。
报告文字（`gpu_protocol.rs:876`）："the DMA of version 0 (stmt …) is never completed (no wait_vm(0) / wait_lds(0) / fence_release between the write and the next barrier), so no barrier can publish it"。
**这就是"barrier ≠ 等待访存完成"的机器化表述。**

### 4.7 全局访问证不出来 → `GV-GLB-OOB`，并打印符号残差

```rust
// crates/amdgpu-kernels/tests/gpu_semantics.rs:61-107（节选）
b.arg("p", DType::Ptr, 0, 8);
b.arg("n", DType::I32, 8, 4);
b.arg_tensor("p", Ext::ka(8) * Ext::c(4));       // p 有 n*4 字节
// 有 b.ka_range(8, 64, i64::MAX)（n ≥ 64）→ lane 63 读 [252, 256) ≤ 4n，证明通过
// 没有这条范围 → GV-GLB-OOB，消息里必须出现 "ka(0x8)"
let off = b.mul_i32(t, c4);
let q = b.ptr_add(p, off);
let v = b.load_f32(q, 0);
```

验证器不是"猜"，而是告诉你**还差哪条合同**：补 `ka_range`，或者像 FMHA 那样写显式 `min`/`Select` 夹紧
（`global_min_clamp_is_understood_but_only_with_a_nonempty_tensor`：夹紧在 n ≥ 1 时可证，n 可能为 0 时仍被拒）。

### 4.8 `assume` 骗不过去 → `GV-ASSUME-VIOLATED`

```rust
// crates/amdgpu-kernels/src/probes/mod.rs:22355-22378（gv_attack_assume_loop，节选）
let vars = b.begin_loop(&[(DType::I32, Opnd::Val(c0))]);
let i = vars[0];
b.assume(Opnd::Var(i), Ext::c(0), Ext::c(0));    // 声称 i 永远是 0……
let ni = b.add_i32(Opnd::Var(i), c1);
b.set_var(i, ni);                                // ……但每轮 +1
```

报 `GV-ASSUME-VIOLATED`（"set_var var0 = [1, 1] is not provably inside the assumed range [0, 0]: the assumption is not inductive"）。
若真有分析器看不见的守卫，只能改写成 `assume_trusted(.., "理由")`——它会进账本，被评审看到。

---

## §5 从 DSL 到 `.s` 的实际流程

```
DSL(Rust) ─► typed IR（Op/Stmt）─► IR 验证器 ─► lowering ─► MIR
          ─► 自研调度 ─► 自研线性扫描寄存器分配 ─► post-RA MIR
          ─► 后端（llvm-tail 或 own）─► .s ─► ROCm clang 只做汇编 ─► .co
```
（`current-status.md` §2。DMA、`buffer_load … lds` 是 own-only，用到它们的 kernel 自动选 own 后端。）

**命令**（容器内；不涉及 GPU 的步骤）：

```bash
# 1) 发射汇编（own 后端：hazard/wait 全部自研）
bash tools/container/cargo.sh run --bin amdc -- emit fmha_fwd_hd128_mxfp8_vs128 --backend=own --out vs128.s

# 2) 用官方汇编器校验编码并生成 code object
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o vs128.co vs128.s

# 3) 严格的 GPU 语义证书（出厂 4 个 kernel；任一 Error → 退出码 1）
bash tools/container/cargo.sh run --bin amdc -- gpu-verify --shipped --strict
#    单个 kernel、连 Info（含每条 TRUSTED）一起看：
bash tools/container/cargo.sh run --bin amdc -- gpu-verify fmha_fwd_hd128_mxfp8_v3_vs128 -v
```

`gpu-verify` 的输出形状（`amdc/src/main.rs:181-207`）：每个 kernel 一行
`== <name>: X finding(s), Y error(s); proven {…}`，随后列出非 Info 发现（`[主名|协议别名] 严重度: 消息`），
最后一行 `certificate: …`，全部结束后再打印合计行 `certificate (all kernels listed): …`。

**一个段落最终长什么样**：以 (c) 的 DMA 为例，own 后端的发射模板是（`crates/amdgpu-compiler/src/emit.rs:605-613`）：

```rust
MOp::GlobalLoadLdsB128 | MOp::GlobalLoadLdsB32 => {
    let m0 = sr(a, reg_of(&inst.srcs[1]));
    let mn = if inst.op == MOp::GlobalLoadLdsB128 { "global_load_lds_dwordx4" } else { "global_load_lds_dword" };
    vec![
        format!("s_mov_b32 m0, {m0}"),
        "s_nop 0".to_string(),
        format!("{mn} {}", mem_op(a, inst.srcs[0], inst.imm)),
    ]
}
```

也就是说一次 `b.load_global_lds(sp, lbase, -(1024 * (3 - q)), 4)` 会变成三行：先把 LDS 目的基址（wave 一致，所以在 SGPR）放进 `m0`，
一个 `s_nop`，然后 `global_load_lds_dwordx4 <地址>, off offset:<负偏移>`；`b.wait_vm(0)` 变成 `s_waitcnt vmcnt(0)`；
`b.barrier()` 变成 `s_barrier`（`emit.rs:736`）；(g) 的 `store_f32x2` 变成 `global_store_dwordx2`（`emit.rs:649-654`）；
(d) 的 `var_mfma_scale` 变成 `v_mfma_scale_f32_32x32x64_f8f6f4 v[..], v[..], v[..], v[..], vX, vY`（`crates/amdgpu-kernels/tests/backend.rs:116-126` 检查此助记符）。
**具体寄存器号由我们自己的寄存器分配决定**，想看真实片段就运行第 1 步再 `grep global_load_lds_dwordx4 vs128.s`。

---

## §6 新手最容易误解的 6 个点

1. **"DSL 就是内联汇编的语法糖"——不是。** 你调用的是普通 Rust 函数，构造的是一张带类型的值图（`ValueId` 只是编号）。
   指令选择、调度、寄存器分配、等待插入都是编译器做的；手写汇编只能通过 safe-asm 的 `Op::Asm`/`Stmt::VarAsm` 进来，并且照样被验证。
2. **"safe 了就算得对"——不对。** L1/L2 保证"不 UB、不越界、不读到未完成/未发布的数据、寄存器合法"；
   softmax 公式写错、`delta` 没对不回滚的 lane 置零，验证器一个字都不会说。数值靠精度门禁和确定性测试。
3. **"有 barrier 就等于数据到了"——不对。** barrier 只是"到达"。DMA/LDS 写要先 `wait_vm(0)`/`wait_lds(0)` **完成**，
   完成之后的**第一个** barrier 才把它**发布**给其他 wave（§4.6 的测试就是这件事）。
4. **"LDS 是我这个 wave 的"——不是。** LDS 是整个工作组共享的；同角色的其他 wave 也在写同一个 tile 的其他部分，
   所以"我自己写完、自己等完"不足以让我读整个共享 tile（`GV-TILE-STALE` 的消息里专门写了这一点）。只有 `lds_private` 声明的私有切片才能"自产自销"。
5. **"编译器能自动知道不越界"——不能凭空知道。** 它需要你声明合同（`arg_tensor`、`ka_is`、`grid_is`……）；证不出来时会打印残差告诉你缺什么。
   合同本身写错，验证器查不出（所以有 `contract_host_truth` 对账）；`assume_trusted` 都会进账本。
6. **铁律："改了 kernel 就必须重跑 verifier"。** 默认编译策略下，只有 MIR 级错误（`GV-REG-*`、`GV-WAIT`）会让编译失败；
   IR 级的边界/协议发现**默认只收集不阻断**（`gpu-semantics-verifier.md` §10）。所以每次改动后跑
   `amdc gpu-verify --shipped --strict`（或 `AMDC_GPU_VERIFY=strict`），确认仍是 `0 unproven`、账本条数没有悄悄变多。

---

## 如果只记三件事

**第一**，这个 DSL 写 kernel 就是写普通的安全 Rust：`FnBuilder` 的每次调用往一张类型化的计算图里加节点，`ValueId` 只是编号，
全仓库 `#![forbid(unsafe_code)]`、零第三方依赖——宿主程序本身不可能有 UB；而在手写汇编（safe-asm）里，Rust 的所有权与生命周期真的在把关
（`Pending` 防提前读/重复等、`Ready<'a>`/`Mfma<'a>` 防覆盖在途操作数）。**第二**，GPU 侧的安全（不越界、读到已完成且已发布的正确那一代、
barrier 两侧数目相等、寄存器合法、用前等待）是编译器验证器在**你声明的合同之下**证明的，证不出就报带符号残差的 `GV-*` 错误，
所有被信任的东西都在 `certificate` 账本里可数（出厂 v3：24 / 100 / 0 unproven）。**第三**，"safe" 不等于"算得对"，也不覆盖跨工作组竞争、
i32 回绕和 safe-asm 块内部访存；改了 kernel 一定重跑 `amdc gpu-verify --shipped --strict` 并跑精度/确定性门禁。
