# FMHA v3 导读：一个 128-key tile 从 DMA 到输出的完整旅程

> **读者**：想读懂（或修改）出厂 FMHA kernel 的人。要求会 Rust 语法、知道 GPU wave/lane/LDS/MFMA 是什么；不要求读过仓库里别的文档。
> **范围**：`Main c9c698a2` 上的四个出厂 kernel `fmha_fwd_hd128_mxfp8_v3_{vs128, t128, kvsplit, kvsplit_combine8h}`，
> 源码在 `crates/amdgpu-kernels/src/fmha/mod.rs` 与 `crates/amdgpu-kernels/src/fmha/io.rs`。
> **约定**：每个代码引用都写成 `文件:行号`（区间为闭区间），对应 `Main c9c698a2`；如果行号漂移，按引用中的符号名查找。
> 节选里省略的部分用 `…` 标出。
> **配套**：布局层从零入门见 [`layout-tour.md`](layout-tour.md)；验证器模型见 [`gpu-semantics-verifier.md`](gpu-semantics-verifier.md)；
> 与厂商汇编的逐条对照见 [`fmha-vs-vendor-asm-diff.md`](fmha-vs-vendor-asm-diff.md)；当前数字见 [`current-status.md`](current-status.md)。

---

## 目录

- §0 全景：四层 crate 与一个 tile 的数据流
- §1 三条出厂路径与自动派发
- §2 动手：构建、emit、验证、gate
- §3 kernel 入口：`_vs128` builder 逐行
- §4 一个 tile 的完整旅程（a–j 十步）
- §5 `_t128` 与 `_kvsplit` 哪里不同
- §6 布局层与 ISA 原子：地址是怎么"推导"出来的
- §7 正确性由什么保证
- §8 改代码前先看这里
- 附录：符号索引

---

## §0 全景：四层 crate 与一个 tile 的数据流

```
amdgpu-isa      指令原子：CopyAtom（ds_read_b128 x2、ds_read_b64_tr_b8 x4、global_load_lds_dwordx4 …）
                          MmaAtom（v_mfma_scale_f32_32x32x64_f8f6f4、v_smfmac_f32_16x16x128_fp8_fp8 …）的 TV 布局
amdgpu-layout   Layout / Swizzle / ComposedLayout、TV 布局 → 每个 lane 的 (base, imm) 地址计划、构建期证明
amdgpu-tile     TiledCopy / TiledMma / partition_*、DMA 分区、流水线（Pipeline + PipeBody）、
                OnlineSoftmax、两条 smfmac 的行和方案、epilogue、KV-split combine
amdgpu-kernels  fmha/mod.rs + fmha/io.rs：把上面的部件组装成四个 kernel（本文主角）
        ↓  amdgpu-dsl 的 Function（结构化 IR）
amdgpu-compiler 下降、调度、寄存器分配、GPU 语义验证器、gfx950 汇编
```

FMHA 计算的是 `O = softmax(Q K^T) V`，head dim = 128，Q/K/V 为 MXFP8（E4M3 + 每 32 元素一个 E8M0 缩放），O 为 bf16。
kernel 把 key 维切成 128 个 key 一个 tile；每个 workgroup 处理一段 query 行（`_vs128` 256 行、`_t128` 128 行），
每个 wave 负责 32 行 query，依次处理所有 KV tile。一个 tile 的旅程：

```
 全局内存               LDS（2 或 3 槽环形缓冲）         寄存器（每 wave）
 ─────────               ─────────────────────         ─────────────────
 K[128 keys][128 d] ──DMA──▶ K 槽 ──ds_read_b128 x2──▶ K 片段 ─┐
 K 的 E8M0 scale  ──DMA──▶ KD 槽 ──ds_read_b32────────▶ scale ─┤ QK：S^T = K·Q^T（每 32 key 一个 32x32 累加器 ×4）
 Q（常驻寄存器，prologue 读一次）──────────────────────────────┘     ↓
                                                    online softmax：行最大值、参考点 bias、回滚
                                                                    ↓
                                                    P = fp8(2^S')：pknorm + scalef32 打包
                                                                    ↓
 V[128 keys][128 d] ──DMA──▶ V 槽 ──ds_read_b64_tr_b8 x4──▶ V^T 片段 ─▶ PV：O^T += V^T·P^T（4 个 32x32 累加器）
                                                    行和 l += rowsum(P)（2 条 smfmac）
                                                                    ↓（所有 tile 之后）
                                                    epilogue：O·(V_descale / l) → bf16 → 全局
```

---

## §1 三条出厂路径与自动派发

| kernel | 每 block | 结构 | 何时用 |
|---|---|---|---|
| `fmha_fwd_hd128_mxfp8_v3_vs128` | 256 行 query、512 线程（8 wave，两组各 4） | `Cadence::TwoPhaseOffset`：A/B 两组 wave 错开一个相位，2 槽环形缓冲；MUBUF K/V DMA；厂商式优先级 ping-pong；两条 smfmac 行和；K-scale op_sel 字节选择；≥ 4 个 tile 用厂商参考配方 + 懒更新；LDS 中转 epilogue | 填满的大 shape（grid 超过一波 CU） |
| `fmha_fwd_hd128_mxfp8_v3_t128` | 128 行、256 线程（4 wave，一组） | `Cadence::OneWindow`：3 槽环形缓冲，每 tile 一个"窗口"；K-scale 走 VGPR；≥ 4 个 tile 用厂商参考配方 + 懒更新 | 小 shape（grid 在一波内放得下） |
| `fmha_fwd_hd128_mxfp8_v3_kvsplit` + `…_kvsplit_combine8h` | 256 行、512 线程 + 合并 kernel | `_vs128` 的结构跑在 key 区间上（每个 split 一段 key），写 fp16 部分输出 `Ohat` 与 `lse`，再由 combine kernel 合并 | 长 KV、block 很少的 shape（例如 B1H8S2048） |

四个 builder：`crates/amdgpu-kernels/src/fmha/mod.rs:366-400`（`fmha_fwd_hd128_mxfp8_v3_vs128`）、
`crates/amdgpu-kernels/src/fmha/mod.rs:404-432`（`…_kvsplit`）、`crates/amdgpu-kernels/src/fmha/mod.rs:434-439`（`…_kvsplit_combine8h`）、
`crates/amdgpu-kernels/src/fmha/mod.rs:443-465`（`…_t128`）；注册在 `crates/amdgpu-kernels/src/registry.rs:38-41`。

**自动派发在 host 侧**（kernel 本身不选路径）。参考 host `tests/gpu/ref_host.cpp` 的规则：

1. KV-split 候选（`tests/gpu/ref_host.cpp:122-143`）：在 split 数 {2, 4} 中挑通过守卫（`N256 ≤ 128 && N256·s ≤ 256`、每个 split 至少一个 tile）
   且模型时间 `T_kv < 0.95·T_current` 的最优者；
2. 否则按 grid 大小（`tests/gpu/ref_host.cpp:56-62`、`tests/gpu/ref_host.cpp:148-155`）：`2·ceil(S/256)·H·B ≤ 256`（一波能放下）用 `_t128`，否则 `_vs128`。

三条路径共用同一个 656 字节的厂商 kernarg ABI（KV-split 多 64 字节扩展），所以 host 只换 code object 与 grid。

---

## §2 动手：构建、emit、验证、gate

```bash
# 列出 / 导出 IR / 编译成汇编（自研后端）
cargo run --bin amdc -- list
cargo run --bin amdc -- print-ir fmha_fwd_hd128_mxfp8_v3_vs128 --stage dsl
cargo run --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --backend own --out vs128.s
# GPU 语义验证（严格模式：任何 error 退出码 1；打印证书 = 可信假设台账）
cargo run --bin amdc -- gpu-verify fmha_fwd_hd128_mxfp8_v3_vs128 --strict
# 单元 / 证明测试
cargo test -p amdgpu-tile --test rowsum --test orient
cargo test -p amdgpu-compiler --test functional_proofs
# GPU 节点：完整验收 gate（构建、严格验证、精度对齐厂商、确定性、压力、性能）
HIP_VISIBLE_DEVICES=5 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3 mytag
# GPU 节点：同一次运行内交替 A/B 两个 kernel（按名字）
HIP_VISIBLE_DEVICES=5 bash tools/ab_fmha.sh fmha_fwd_hd128_mxfp8_v3_vs128 <另一个变体> -- "4 2048" "1 8192"
```

- `amdc` 的子命令见 `crates/amdc/src/main.rs:11-21`（usage）与 `crates/amdc/src/main.rs:198`（`gpu-verify`）。
- `tools/port_gate.sh` 的判定：精度按"厂商对齐"（同一次运行、同一输入，我们的 mean_err ≤ 1.10×厂商，坏点数 ≤ 厂商）；
  性能按"与同次运行的厂商之比 ≥ 0.97 × 记录的出厂比值"。各节说明在脚本头部注释。
- 汇编对照工具：`tools/asm_diff.py`（与厂商 `.co` 逐段对照）、`tools/mfma_reuse.py`（MFMA 操作数复用 / 链分析）。
- 编译策略（下降选项、调度预设）由 builder 里 `fb.compile_profile(CompileProfile::Vs128)` 声明，
  枚举在 `crates/amdgpu-dsl/src/lib.rs:971-985`，映射在 `crates/amdgpu-compiler/src/lib.rs:31-41`（`policy_name`）——与 kernel 名字无关。

---

## §3 kernel 入口：`_vs128` builder 逐行

```rust
// crates/amdgpu-kernels/src/fmha/mod.rs:366-400（fmha_fwd_hd128_mxfp8_v3_vs128，节选）
let mut fb = FnBuilder::new("fmha_fwd_hd128_mxfp8_v3_vs128", 512);       // 512 线程
fb.compile_profile(CompileProfile::Vs128);
let io = declare_io(b, 256, false);              // ABI + 主机合同（§4a）
let blk = block3(&io, b, 8);                     // 每 block 状态：行号、Q 片段、scale（§4a）
let roles = Roles::split(b, &blk.lane, 4);       // wave 0..3 = 组 A，4..7 = 组 B
let mut ring = KvRing3::new(b, &io, &blk, &roles, 2);   // LDS 计划 + 所有拷贝分区（§4b、§4c）
ring.enable_buffer_dma(b);                       // K/V DMA 改用 MUBUF 形式（§4c）
ring.enable_buffer_kd(b);                        // K-scale DMA 也用 MUBUF 形式
pipe.key_unclamped = true;                       // 所有 DMA 都有描述符越界保护：下一 tile 的 key 不必夹取
let pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, …);   // §4d
let lut = blk.lut_copy(b);                       // 少于 4 个 tile → 短序列副本
versioned(b, lut, |b, lut| {                     // 两份代码：短 / 长序列
    roles.each(b, |b, role| {                    // 两份代码：组 A / 组 B
        let mut body = AttnBody3::new(b, &ring, &blk, wid, lut);
        body.kscale_sel = true;                  // §4e
        let p = amdgpu_tile::rowsum::plan();
        body.sm.use_rows2(b, &blk.lane, p.dwords, p.idx);     // §4h
        if !lut { body.sm.set_vendor_ref(b); body.lazy_bias = true; }    // §4f：厂商配方 + 懒更新
        body.setprio = true;
        pipe.run(b, role, &mut body);            // 整个 KV 循环
        let fin = body.out_scale(b);
        store_out3_lds(&io, b, &blk, &ring.epi, wid, &body.o.expect("O"), fin);   // §4i
    });
});
```

两个"代码副本"维度都是 workgroup 一致的条件，用 `versioned` / `Roles::each` 生成 `if/else`
（`crates/amdgpu-tile/src/lane.rs:103-109`、`crates/amdgpu-tile/src/lane.rs:86-92`）。所以 `_vs128` 里共有 2×2 = 4 份内层循环。

`AttnBody3`（`crates/amdgpu-kernels/src/fmha/mod.rs:187-207`）是这个 kernel 给流水线提供的"每个相位做什么"：
它实现 `PipeBody` trait（`crates/amdgpu-tile/src/pipeline.rs:72-102`），流水线负责"什么时候做"（§4d）。

---

## §4 一个 tile 的完整旅程

### §4a ABI 与每 block 状态

- **ABI**：`declare_io`（`crates/amdgpu-kernels/src/fmha/mod.rs:469-539`）用 `KArgs::tensor` 把 kernarg 声明成 BSHD 张量：
  Q/K/V 为 fp8、O 为 bf16、Q/K 的 E8M0 descale 为 `[B, S, H, 4]`、V 的 descale 为一个标量。每个张量的形状/步长来自 kernarg 偏移，
  同时**生成主机合同**（`arg_tensor`、`ka_is`、`grid_is`、`contract::compact_tensor`）——验证器就是拿这些合同去证明每一次访存在界内。
  KV-split 多出 `0x270 Ohat、0x278 lse、0x280 splits、0x284 rows_total`（`crates/amdgpu-kernels/src/fmha/mod.rs:524-537`）。
  对应的类型 `FmhaIo` / `KvsArgs` 在 `crates/amdgpu-kernels/src/fmha/io.rs:16-39`。
- **每 block 状态**：`block3`（`crates/amdgpu-kernels/src/fmha/mod.rs:543-598`）产出 `FmhaBlock`（`crates/amdgpu-kernels/src/fmha/io.rs:141-164`）：
  - query 行：`row0 = qtile·rows + 32·wave`，`row = row0 + lane % 32`，越界行夹到 `S-1`（只读不写，写回时由 `row0 < S` 守卫）；
  - KV-split 时由 grid z 解出 (batch, split) 和本 split 的 key 区间（`components::kv_split_decode` / `kv_split_range`，
    `crates/amdgpu-tile/src/components.rs:20`、`crates/amdgpu-tile/src/components.rs:32`）；GQA 的 KV head 由 `gqa_kv_head` 算出（`crates/amdgpu-tile/src/components.rs:8`）；
  - **Q 常驻寄存器**：Q 是 QK MFMA 的 B 操作数，用 `make_tiled_copy_b(GLOBAL_LOAD_X4, &mma)` 推导出每个 lane 的 4 条 `global_load_dwordx4`
    （`crates/amdgpu-kernels/src/fmha/mod.rs:567-585`），整个 kernel 只读一次；
  - **Q 的 scale**：E8M0 字节按 lane 半区选出（`ScaleSel::pair`，`crates/amdgpu-tile/src/frag.rs:139-152`），再 **减 9**
    （`crates/amdgpu-kernels/src/fmha/mod.rs:591`）——即 S 以 1/512 为单位，`x = S·2^-9`，后面 softmax 的指数都在这个单位上；
  - `nkv = ceil(kv_len/128)` 个 tile，最后一个可能不满（masked）。

### §4b LDS 计划

```rust
// crates/amdgpu-kernels/src/fmha/mod.rs:80-88（KvRing3::new 开头）
let [ks, vs] = lds.ring_group(slots, [("K", 16384, Layout::FMHA_SWZ), ("V", 16384, Layout::FMHA_SWZ)]);
let kd = lds.ring("KD", 512, slots, Layout::Linear);
lds.at(32768 * slots as i64 + 2048);
let ones = lds.tile("ones", 2048, Layout::Linear);
let q1 = lds.tile("Q1", 16384, Layout::Linear);
let q0 = lds.tile("Q0", 16384, Layout::Linear);
let epi = lds.private("epi", 4096, 8, &[&q1, &q0]);
```

| 区域 | 大小 | 用途 |
|---|---|---|
| K / V 环 | 每槽 16 KB + 16 KB，`slots` 槽（`_vs128` 2，`_t128` 3） | 128 key × 128 字节，行内 16 字节块做 FMHA swizzle（`Layout::FMHA_SWZ`，`crates/amdgpu-tile/src/shared.rs:31`） |
| KD 环 | 每槽 512 B | 128 个 key 的 4 字节 E8M0 scale 字 |
| ones、Q0/Q1 | 2 KB、16 KB×2 | prologue 写一次的常量（`store_constants`，`crates/amdgpu-kernels/src/fmha/mod.rs:170-184`） |
| epi | 每 wave 4 KB 私有区 | epilogue 转置中转，复用（overlay）Q0/Q1 的空间 |

`LdsPlan`（`crates/amdgpu-tile/src/shared.rs:199`）是声明式的：`commit` 时把布局写进 Function，验证器据此检查 LDS 访问不越界、
环形缓冲每个 (槽, 代) 的读写顺序符合协议。

### §4c K/V DMA：从全局直接写进 LDS

gfx950 的 LDS-direct DMA（`global_load_lds_dwordx4` / `buffer_load_dwordx4 … lds`）有个硬约束：**目的地址固定**——第 q 条指令的 lane i 写
`M0 + imm + 16·i`。所以拷贝要"从目的端反推"：`partition_dma`（`crates/amdgpu-tile/src/dma.rs:37-76`）对每个 lane / 指令算出它落在 LDS 槽内的字节 d，
用视图的逆映射得到它是 tile 的哪个元素（行 = d / 128，swizzle 只动行内 16 字节块，于是 **swizzle 被搬到了源地址上**），
再生成源地址。每个 lane 的行号和不同的列偏移只绑定一次，每个 tile 只剩 `key0 + row`、夹取和乘步长。

```rust
// crates/amdgpu-kernels/src/fmha/mod.rs:107-108
let kdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, ksrc, &ks, &kv_view());
let vdma = partition_dma(b, &ctx, GLOBAL_LDS_X4, &l, &w, &wv, 4, vsrc, &vs, &kv_view());
```

- 每个组 4 个 wave，每 wave 4 条 16 字节 DMA，4×4×64×16 B = 16 KB = 一个 K（或 V）tile。
- **MUBUF 形式**（`_vs128`，`enable_buffer_dma` → `DmaPart::buffer_form` / `issue_buffer`，`crates/amdgpu-tile/src/dma.rs:103-149`）：
  每条指令一个循环不变的 voffset（VGPR），每 tile 的推进 `key0 × stride` 放进 SGPR soffset，M0 每 tile 一次 `readfirstlane`。
  越过序列末尾的行不夹取，而是落在 buffer 描述符的 record range 之外——硬件读回 0（这些 key 之后会被 mask 掉，V 行乘以 0 的 P）。
- **K 的 scale**：`_vs128` / `_kvsplit` 用 `GLOBAL_LDS_X1` 的 DMA（wave 2、3 重复 wave 0、1 的那一半，`crates/amdgpu-kernels/src/fmha/mod.rs:111-115`）；
  `_t128` 走 VGPR 路径（读进寄存器，`stage_complete` 时再 `ds_write`，`crates/amdgpu-kernels/src/fmha/mod.rs:116-119`）。
- 谁拷什么：`KvRing3::stage`（`crates/amdgpu-kernels/src/fmha/mod.rs:136-163`）——组 A 拷 K 和 K-scale，组 B 拷 V；`_t128` 一组全拷。

### §4d 流水线：谁在什么时候做什么

`Pipeline::run`（`crates/amdgpu-tile/src/pipeline.rs:176-313`）把节奏写成数据（`Cadence`，`crates/amdgpu-tile/src/pipeline.rs:24-43`）；
kernel 只提供每个相位的工作。`_vs128` 的 `TwoPhaseOffset`（文档在 `crates/amdgpu-tile/src/pipeline.rs:5-16`，循环体在
`crates/amdgpu-tile/src/pipeline.rs:365-406`）：

```
                 ┌──────── 一个 tile t ────────┐
组 A:  stage(t+1) | V 相位(t)  | wait vm | barrier | M 相位(t)        | barrier
组 B:            （比 A 晚一个相位：当 A 在 M 相位时，B 在 V 相位）
V 相位 = softmax（VALU 为主）：行最大值、回滚、P 转换          （AttnBody3::v_phase，mod.rs:292-306）
M 相位 = 矩阵（MFMA 为主）：PV(t) + 行和、然后 QK(t+1)         （AttnBody3::m_phase，mod.rs:307-318）
```

- 两组错开一个相位，使得任意时刻总有一组在用 MFMA、另一组在用 VALU（厂商同样的做法）；`setprio` 在 M 相位开头 `s_setprio 1`、V 相位开头 `s_setprio 0`，
  让矩阵相位优先发射（`crates/amdgpu-kernels/src/fmha/mod.rs:305-307`、`crates/amdgpu-kernels/src/fmha/mod.rs:327-329`）。
- 注意 QK(t+1) 是在 M 相位末尾算的，所以进入 V 相位(t+1) 时 S 已经就绪。prologue 先算 QK(0)（`prologue_compute`，`crates/amdgpu-kernels/src/fmha/mod.rs:287-289`）。
- 循环携带的状态（`loop_state` / `bind_state`，`crates/amdgpu-kernels/src/fmha/mod.rs:290-303`）：参考点 bias（Frag2）、4 个 O 累加器、4 个 S 累加器、行和累加器。
- `TileIt`（`crates/amdgpu-tile/src/pipeline.rs:47-62`）告诉 body 当前 tile 的 (槽, 代)、下一个 tile 的 (槽, 代)、是否最后一个不满 tile（`masked`）。
  最后一个 tile 被剥离出来（`masked` 与非 masked 两份），所以循环体里没有 mask 开销。

### §4e QK：S^T = K·Q^T

**方向是推导出来的**：`attention_orientation`（`crates/amdgpu-kernels/src/fmha/mod.rs:26-32`）调用 `orient::search_attention`
（`crates/amdgpu-tile/src/orient.rs:121`）在两个带缩放的 fp8 MFMA 原子上搜索 S→P 的交接方式，断言最优解是"S^T = K·Q^T（A = K，B = Q），
P 直接作为 PV MFMA（O^T = V^T·P^T）的 B 操作数，转换零代价"。这就是为什么 S 的累加器布局恰好能就地打包成 P。

```rust
// crates/amdgpu-kernels/src/fmha/mod.rs:97-103（KvRing3::new）
let mma = make_tiled_mma(o.0, false);                                       // v_mfma_scale_f32_32x32x64_f8f6f4
let kview = ComposedLayout::new(Swizzle::fmha128(), make_layout(tup![32, 64], tup![128, 1]));
let placements: Vec<i64> = (0..4).flat_map(|s| (0..2).map(move |c| 4096 * s + 64 * c)).collect();
let kpart = make_tiled_copy_a(DS_READ_B128X2, &mma).partition_lds(b, &ctx, &l, &ks, &kview, &placements);
…
let kdpart = make_tiled_copy_bcast(DS_READ_B32, kd_read_tv(), 128, 32).partition_lds(…);
```

- `make_tiled_copy_a(atom, mma)`（`crates/amdgpu-tile/src/copy.rs:54-57`）：用 MMA 原子 A 操作数的 TV 布局当拷贝的 TV 布局——
  "lane l 的寄存器里应放 A 的哪些元素"由原子决定，拷贝只负责从 LDS 把它们读出来。
- `partition_lds`（`crates/amdgpu-tile/src/copy.rs:78-98`）在构建期证明每条 `ds_read_b128` 读的 16 字节在 swizzle 后仍连续对齐（`proofs::check_contiguous`），
  并把每个 lane 的地址拆成"绑定一次的 base 寄存器 + 指令立即数"。8 个 placement = 4 个 32-key 块 × 2 个 64 字节 k 组。
- 每 32 key 一个块（`qk_block`，`crates/amdgpu-kernels/src/fmha/mod.rs:218-250`）：
  1. **参考点 bias**：`var_mfma_f16_z(S_s, ones_f16, bias)` 一条 f16 32x32x8 MFMA 把累加器初始化成 `ones × bias`，
     即每个 query 列加上同一个参考值（§4f 的"减最大值"就这样折叠进了 MFMA，不需要逐元素 VALU）；
  2. 两个 K 片段（各 `ds_read_b128 x2`）、一个 K-scale 字（`ds_read_b32`）；
  3. **K-scale 字节选择**（`kscale_sel`）：scale 字里 4 个字节对应 4 个 32 维 k 组；每个 lane 先按自己的半区右移 `8·(lane/32)` 位一次，
     然后两条带缩放 MFMA 分别用 op_sel 选字节 0 与字节 2（`crates/amdgpu-kernels/src/fmha/mod.rs:231-237`），
     这是厂商的写法（比"每块提取两个字"少一半移位）；
  4. 两条 `v_mfma_scale_f32_32x32x64_f8f6f4`，经 `gemm`（`crates/amdgpu-tile/src/mma.rs:68-82`）发出。
- 4 个块 = 4 个 32x32 S 累加器 = 128 key × 32 query，每个 lane 16 个 f32。

### §4f online softmax：参考点、回滚、厂商配方

`OnlineSoftmax`（`crates/amdgpu-tile/src/softmax.rs:34-52`）不维护"运行最大值"本身，而是维护一个**参考点** `b`（以两个 f16 字 `b = 4·(hi + lo)`
存储，精确到约 2^-22），S 在 QK 时就已经加上了它：`x = S + b`。每个 tile 做**一次**决定（`decide_impl`，`crates/amdgpu-tile/src/softmax.rs:129-231`）：

1. 行最大值 `m`：`v_max3_f32` 链 + 与伙伴 lane（`l ^ 32`，同一 query 的另一半 key）交换（`row_max`，`crates/amdgpu-tile/src/reduce.rs:52-64`）；
2. 若 `m > thr`：参考点移到 `b_new = b + (target − m)`，重新量化成两个 f16 字，`delta = b_q − b_old`；
3. 回滚分支（`crates/amdgpu-tile/src/softmax.rs:215-229`）：`S += delta`（`v_pk_add_f32`）、`O *= 2^(512·delta)`、`l *= 2^(512·delta)`。
   分支没有 exec mask——不回滚的 lane 拿到的 `delta = 0`，所以对它们是恒等。
4. 新参考字写回循环变量，并作为下一个 tile 的 bias MFMA 的输入。

**两套参数**（`versioned` 选出的两份代码各用一套）：

| | 短序列副本（< 4 个 tile，`lut = true`） | 长序列副本（≥ 4 个 tile；三条路径都一样） |
|---|---|---|
| 配方（`_vs128`） | `Recipe::vsx(true)`（`crates/amdgpu-tile/src/softmax.rs:27-31`） | 厂商参考配方 `set_vendor_ref`（`crates/amdgpu-tile/src/softmax.rs:75-88`） |
| 目标 / 阈值 | 131/512、阈值 120/512（阈值低于目标：几乎每个 tile 都重新对齐） | `s91 = 0.24799132` / `s92 = 0.26135647`（阈值比目标高 6.84/512：懒回滚，只有行最大值越过阈值才移动参考点） |
| 第一个 tile | — | 初始参考点 b = 64（bias 字 0x4800）让第一个 tile 的最大值远超阈值，必然对齐 |
| 快速路径 | 每 tile 计算新参考字 | **懒更新**（`decide_lazy`，`crates/amdgpu-tile/src/softmax.rs:236-287`；`crates/amdgpu-kernels/src/fmha/mod.rs:310-316`）：参考字重新量化与重缩放全部放进回滚分支，不回滚的 tile 只付出行最大值 + 一次比较 + 分支 |
| P 转换 | pknorm + LUT 修正 | pknorm，无修正，转换 scale = 0.25（厂商的 `s89`） |
| 误差 | 精确（坏点 0） | 落在厂商的误差统计上（坏点数、均值与厂商一致） |

为什么短序列不用厂商配方：tile 太少时懒回滚的误差平均不掉，过不了厂商对齐 gate（从 2 个 tile 起用厂商配方时，S=300 SK=129 是厂商均值的 1.31 倍；从 4 个 tile 起全部通过）。
懒更新只在真实数据上体现：harness 的调色板数据（`REF_DATA=lcg`）让行最大值几乎每个 tile 都变，两边都几乎每 tile 回滚；正态分布数据（`REF_DATA=normal`）上回滚很少，快速路径决定速度——`tools/port_gate.sh` 的 PERF 因此用 `normal` 数据。
切换条件 `lut_copy` 在 `crates/amdgpu-kernels/src/fmha/io.rs:213-217`。参数的来龙去脉见 [`vendor-spec.md`](vendor-spec.md) §4.7 与 [`current-status.md`](current-status.md)。

最后一个不满的 tile 用 `KeyMask`（`crates/amdgpu-tile/src/reduce.rs:13-32`）把越界 key 换成 `-1e30`（求最大值时）或 0（求 P 时）。

### §4g P 转换：fp8 打包

```rust
// crates/amdgpu-tile/src/softmax.rs:240-249（OnlineSoftmax::convert）
for i in t0..t1 {
    let xs = masked_values(b, s, i, mask, self.zero_f);
    words.extend(to_p8(b, &xs, self.conv_s, conv, lut.as_ref()));
}
```

`to_p8`（`crates/amdgpu-tile/src/reduce.rs:94-113`）每 4 个值出一个 fp8 字：`v_cvt_pknorm_u16_f32` 把 x ∈ [0, 1] 量化成 u16，
**把 u16 当成 bf16 看就是 `2^(512x − 127)` 的近似**（Schraudolph 式指数），再由 `v_cvt_scalef32_pk_fp8_bf16` 转成两个 E4M3（op_sel 合并高低半）。
所以 P = exp(S − max) 不需要 `v_exp_f32`——这正是 S 以 1/512 为单位（§4a 的 −9）的原因。

4 个 S 累加器 → 16 个 P 字 = 两个 PV 的 B 片段（`Frag::from_words`，`crates/amdgpu-tile/src/frag.rs:76-81`）。
因为方向是推导出来的零代价交接（§4e），P 字按累加器顺序打包即可，不需要任何 lane 间重排；key 的排列差异由 V 的转置读取吸收（§4h）。

### §4h PV 与行和

```rust
// crates/amdgpu-kernels/src/fmha/mod.rs:246-258（AttnBody3::pv）
let v = r.vpart.at(b, &r.vs.at(slot));
let p = [Frag::<B32x32x64Fp8>::from_words(b, &w[0..8]), Frag::<B32x32x64Fp8>::from_words(b, &w[8..16])];
self.sm.row_sum(b, p[0].v, p[1].v);
for (hh, pf) in p.iter().enumerate() {
    for e in 0..4usize {
        let vf = v.read(b, hh + 2 * e);
        gemm(b, &r.mma, o.accs[e].var, vf, pf.v, self.c127, self.c123, 0, 0);
    }
}
```

- **V^T 片段**：PV 计算 `O^T = V^T·P^T`，需要 V 的转置。`ds_read_b64_tr_b8` 在读的同时做 8×8 字节转置；
  它的源地址布局 `vt_tr_source`（`crates/amdgpu-kernels/src/fmha/mod.rs:42-44`）配上同一个 swizzle 视图 `kv_view`，
  由 `make_tiled_copy(DS_READ_TR_B8X4, …)` 分区（`crates/amdgpu-kernels/src/fmha/mod.rs:104`）。硬件的转置映射是实测出来的
  （`amdgpu_isa::tr_b8`，`crates/amdgpu-isa/src/lib.rs:294`；探针记录见 [`isa-probes-smfmac-tr.md`](isa-probes-smfmac-tr.md)）。
- **8 条 PV MFMA**：2 个 P 片段（各 64 key）× 4 个 O 累加器（各 32 维）。scale 固定：V 用 127（×1），P 用 123（×2^-4）；行和 l 不带这个因子，所以 `l_total` 末尾乘 1/16 与之对应（§4i）。
- **行和 l：两条 smfmac**（`crates/amdgpu-tile/src/rowsum.rs`）。`v_smfmac_f32_16x16x128_fp8_fp8` 的 A 是 2:4 压缩的稀疏操作数、B 是稠密的 P。
  P 片段里 lane l 的 32 个字节全属于 query `l % 32`；方案 `RowSum2`（`crates/amdgpu-tile/src/rowsum.rs:20-24`）：
  - A 的第 m 行只在 `m % 4 ∈ {0, 1}` 时非零，非零的"全 1"放在偶数或奇数 k 组（决定它求和的是 query n 还是 n+16）；
  - 同一 lane 的两行用互补的索引字 `0x44444444`（每个 B dword 的字节 0、1）与 `0xeeeeeeee`（字节 2、3）；
  - 两个 P 片段累加进同一个 D：lane l 的行和 = D 的 dword 0 + dword 1，**只含自己的 query**，所以回滚时逐 lane 缩放这两个 dword 仍是精确的
    （`scale_l`，`crates/amdgpu-tile/src/softmax.rs:311-319`）。
  - `rowsum::prove`（`crates/amdgpu-tile/src/rowsum.rs:79-92`）在实测的 smfmac 语义（`amdgpu_isa::smfmac::k_of`，`crates/amdgpu-isa/src/lib.rs:264`）上
    枚举 64 lane × 16 个存储值，证明每个属于 query `l % 32` 的 P 字节恰好被求和一次、没有别的；`rowsum::search`（`crates/amdgpu-tile/src/rowsum.rs:95-112`）
    枚举所有 dword 对 × 互补索引对，出厂方案 `plan()`（`crates/amdgpu-tile/src/rowsum.rs:115-119`）是其中之一。
  - 生成 A 操作数与索引字的代码：`SmfmacOnes::new_rows2`（`crates/amdgpu-tile/src/reduce.rs:143-161`）；两条 smfmac：`crates/amdgpu-tile/src/reduce.rs:163-168`。

### §4i Epilogue：归一化与写回

所有 tile 结束后：

- `l_total`（`crates/amdgpu-tile/src/softmax.rs:321-336`）= (dword 0 + dword 1) / 16（与 PV 中 P 的 2^-4 缩放约定一致），
  `out_scale = V_descale / l`（`crates/amdgpu-kernels/src/fmha/io.rs:219-222`）。
- `_vs128`：`store_out3_lds`（`crates/amdgpu-kernels/src/fmha/mod.rs:636-685`）。O^T 的累加器布局里一个 lane 持有一个 query 的 4 个连续维度 × 若干组，
  直接写全局只能是 8 字节的 `global_store_dwordx2`。这里先用 `ds_write_b32` 按累加器布局写进每 wave 私有的 LDS 区（16 字节块按 query 做 `Swizzle<3,4,3>`），
  再以行连续的方式 `ds_read_b128` 读回，最后 8 条 16 字节 `global_store_dwordx4`（厂商的条数）。写/读两个方向都是 `make_tiled_copy` + 显式 TV 布局。
- `_t128`：`store_out3`（`crates/amdgpu-kernels/src/fmha/mod.rs:603-630`）直接按 C 布局 `global_store_dwordx2`。
- 写回整体由 `row0 < S` 守卫；越界的 query 行在 Q 读取时已被夹取，结果不会被写出。

### §4j KV-split：部分结果与合并

- 主 kernel `_kvsplit` 与 `_vs128` 结构相同（同样 `TwoPhaseOffset` + 两条 smfmac 行和），区别是每个 block 只处理自己 split 的 key 区间，
  且 epilogue 写的是部分结果：`store_partial3`（`crates/amdgpu-kernels/src/fmha/mod.rs:690-756`）把 `Ohat = O·fin` 存成 fp16，
  同时写 `lse = log2(l) − 512·b`（`OnlineSoftmax::lse`，`crates/amdgpu-tile/src/softmax.rs:339-353`）——与 split 无关的对数和，各 split 可直接比较。
- 合并 kernel `_kvsplit_combine8h`：`combine_kernel`（`crates/amdgpu-kernels/src/fmha/io.rs:228-249`）→ `combine_splits`（`crates/amdgpu-tile/src/epilogue.rs:93-171`），
  每个线程负责 8 个输出元素，在 split 上做在线最大值 / 重缩放（`m`、`den = Σ 2^(lse − m)`、`acc = Σ 2^(lse − m)·Ohat`），最后输出 bf16。
- 短 / 长序列副本按**整个序列**的 tile 数选（`lut_copy` 用 `ntiles_total`），与 split 数无关；长序列副本（≥ 4 个 tile）与 `_vs128` 一样用厂商参考配方 + 懒更新。

---

## §5 `_t128` 与 `_kvsplit` 哪里不同

| | `_vs128` | `_t128` | `_kvsplit` |
|---|---|---|---|
| 线程 / 行 | 512 / 256 | 256 / 128 | 512 / 256 |
| 节奏 | `TwoPhaseOffset`（两组错相） | `OneWindow`（一组，3 槽，每 tile 一个窗口） | `TwoPhaseOffset` |
| body 入口 | `v_phase` + `m_phase` | `window`（`crates/amdgpu-kernels/src/fmha/mod.rs:336-359`） | `v_phase` + `m_phase` |
| K/V DMA | MUBUF | global LDS-direct（`issue_rows_shared`） | global LDS-direct |
| K-scale | MUBUF DMA，op_sel 字节选择 | VGPR 路径，`ScaleSel::pair` 提取 | DMA，`ScaleSel::pair` 提取 |
| 行和 | 2 条 smfmac | 4 条 smfmac（`SmfmacOnes::new`，`crates/amdgpu-tile/src/reduce.rs:125-139`） | 2 条 smfmac |
| 长序列副本的配方（≥ 4 tile） | 厂商参考配方 + 懒更新 | 厂商参考配方 + 懒更新 | 厂商参考配方 + 懒更新 |
| epilogue | LDS 中转 | 直接 `dwordx2` | 部分结果 + `lse` |

`_t128` 的窗口（`OneWindow` 循环体：`crates/amdgpu-tile/src/pipeline.rs:462-490`）：先发起 tile t+2 的 DMA，再在一个窗口里交错完成
"softmax 决定 → 每 32 key 转换 P 的同时算下一 tile 的 QK 块 → PV"，最后一次 wait + barrier。它的参考字在窗口**末尾**才写回
（`decide_deferred` + `commit_bias`，`crates/amdgpu-tile/src/softmax.rs:109-118`），因为提前写会拉长旧字的活跃区间，使调度器退化。

---

## §6 布局层与 ISA 原子：地址是怎么"推导"出来的

本仓库借用 CuTe 的思路：**布局（Layout）是从坐标到偏移的函数**，形如 `shape : stride`；**TV 布局**是从 (线程, 值) 到 tile 元素的布局。

- `make_layout(shape, stride)`：`crates/amdgpu-layout/src/layout.rs:25`。例如 K/V tile 的 LDS 视图 `[128 key, 128 字节]`、行步长 128：
  `make_layout(tup![128, 128], tup![128, 1])`（`crates/amdgpu-kernels/src/fmha/mod.rs:35-37`）。
- `Swizzle`（`crates/amdgpu-layout/src/swizzle.rs:14-23`）：把源位的奇偶 XOR 进目的位。`Swizzle::fmha128()`（`crates/amdgpu-layout/src/swizzle.rs:35-37`）
  用行号的位打散行内 16 字节块，使 `ds_read_b128` 与 `ds_read_b64_tr_b8` 都无 bank 冲突；`Swizzle::cute(3, 4, 3)` 是 CuTe 的 `Swizzle<3,4,3>`（epilogue 用）。
- `ComposedLayout`（`crates/amdgpu-layout/src/swizzle.rs:106`）= swizzle ∘ layout，即"带 swizzle 的视图"。
- 地址计划：`IssuePlan`（`crates/amdgpu-layout/src/tv.rs:41`）把 TV 布局按指令宽度切成指令，`layout_to_offsets`（`crates/amdgpu-layout/src/tv.rs:86`）
  把每条指令的地址表达式拆成"每 lane 一次的 base + 立即数"，立即数受原子的 `ImmRange` 约束。
- 构建期证明（`crates/amdgpu-layout/src/proofs.rs`）：`check_tv` 覆盖性（每个元素恰好一次，`crates/amdgpu-layout/src/proofs.rs:22`）、
  `check_contiguous` 连续对齐（`crates/amdgpu-layout/src/proofs.rs:48`）、`check_swizzle_bijective` swizzle 是双射（`crates/amdgpu-layout/src/proofs.rs:135`）等。
  证明失败时 builder 直接 panic——错的地址根本生成不出 kernel。

ISA 原子（`crates/amdgpu-isa/src/lib.rs`）：

| 原子 | 定义 | 在 FMHA 里的用途 |
|---|---|---|
| `GLOBAL_LOAD_X4` | `crates/amdgpu-isa/src/lib.rs:66` | Q 常驻片段 |
| `GLOBAL_LDS_X4` / `GLOBAL_LDS_X1` | `crates/amdgpu-isa/src/lib.rs:73-82` | K/V DMA / K-scale DMA |
| `DS_READ_B128X2` | `crates/amdgpu-isa/src/lib.rs:89` | K 片段 |
| `DS_READ_TR_B8X4` | `crates/amdgpu-isa/src/lib.rs:92-96` | V^T 片段（转置读，实测映射） |
| `DS_READ_B32` | `crates/amdgpu-isa/src/lib.rs:91` | K-scale 字（广播 TV 布局） |
| `DS_WRITE_B32` / `DS_READ_B128` | `crates/amdgpu-isa/src/lib.rs:99`、`crates/amdgpu-isa/src/lib.rs:88` | epilogue 中转 |
| `GLOBAL_STORE_X2` | `crates/amdgpu-isa/src/lib.rs:70` | `_t128` 输出 |
| `MmaAtom::mfma_scale_32x32x64_f8` | `crates/amdgpu-isa/src/lib.rs:165-186` | QK、PV（A/B/C 的 TV 布局写在原子里） |
| `MmaAtom::mfma_32x32x8_f16` | `crates/amdgpu-isa/src/lib.rs:208` | 参考点 bias（ones × bias） |
| `MmaAtom::smfmac_16x16x128_fp8` + `mod smfmac` | `crates/amdgpu-isa/src/lib.rs:234`、`crates/amdgpu-isa/src/lib.rs:264` | 行和（稀疏 A 的索引语义为实测） |

每个原子带 `provenance`：`Shipped`（出厂 kernel 已验证的用法）或 `Probe`（单热探针实测，`tests/gpu/isa_probe_host.cpp`，记录的映射有回放测试）。

---

## §7 正确性由什么保证

| 层次 | 机制 | 位置 |
|---|---|---|
| 构建期 | TV 覆盖性、连续对齐、swizzle 双射、S→P 方向搜索断言、行和方案证明 | `crates/amdgpu-layout/src/proofs.rs`、`crates/amdgpu-kernels/src/fmha/mod.rs:26-32`、`crates/amdgpu-tile/src/rowsum.rs:115-119` |
| 编译期 | GPU 语义验证器：每次访存对着 ABI 合同证明在界内、LDS 环形缓冲的 (槽, 代) 读写协议、wait/barrier 完整性；`--strict` 只允许证书里列出的可信假设 | `crates/amdgpu-compiler/src/gpu_verify.rs:1687`（`analyse`）、[`gpu-semantics-verifier.md`](gpu-semantics-verifier.md) |
| 单元/证明测试 | 方向重推导、行和方案、厂商配方的递推与朴素和一致（E2） | `crates/amdgpu-tile/tests/orient.rs`、`crates/amdgpu-tile/tests/rowsum.rs`、`crates/amdgpu-compiler/tests/functional_proofs.rs:37` |
| GPU | 精度与厂商对齐、确定性、压力、性能 | `tools/port_gate.sh`、`tests/gpu/ref_host.cpp` |

可信假设（`assume_trusted`）都写明理由，例如 tile 计数器的上界（`crates/amdgpu-tile/src/pipeline.rs:254-259`）、`nkv` 的界（`crates/amdgpu-kernels/src/fmha/mod.rs:595`）。
`gpu-verify --strict` 打印的证书统计它们，gate 记录证书，数量变化即需解释。

---

## §8 改代码前先看这里

- **不要按 kernel 名字分支**：编译策略走 `CompileProfile`，kernel 特性走 builder 里的显式字段（`AttnBody3` 的 `setprio` / `kscale_sel` / `lazy_bias`）。
- **改地址就改布局**：不要手写 lane 地址公式；换视图、TV 布局或 placement，让 `partition_*` 推导，构建期证明会替你兜底。
- **改节奏就改 `Pipeline`**：body 只描述相位里的工作，wait / barrier / 槽与代由 `Pipeline::run` 统一生成，验证器按同一协议检查。
- **改 softmax 先看两份副本**：短序列副本与长序列副本的配方不同（§4f）；改一处要分别跑 gate。
- **验收**：同一次运行内交替 A/B（`tools/ab_fmha.sh`，≥ 7 轮）+ 完整 `tools/port_gate.sh`；出厂 kernel 的 md5 变化需要说明原因。
- **对照厂商**：`tools/asm_diff.py` 与 [`fmha-vs-vendor-asm-diff.md`](fmha-vs-vendor-asm-diff.md) 列出了与厂商汇编的差异与已试过的改动。

---

## 附录：符号索引

| 符号 | 位置 |
|---|---|
| `fmha_fwd_hd128_mxfp8_v3_vs128` / `_kvsplit` / `_kvsplit_combine8h` / `_t128` | `crates/amdgpu-kernels/src/fmha/mod.rs:366-465` |
| `declare_io`、`block3` | `crates/amdgpu-kernels/src/fmha/mod.rs:469-539`、`crates/amdgpu-kernels/src/fmha/mod.rs:543-598` |
| `FmhaIo`、`KvsArgs`、`FmhaBlock` | `crates/amdgpu-kernels/src/fmha/io.rs:16-39`、`crates/amdgpu-kernels/src/fmha/io.rs:141-164` |
| `attention_orientation`、`kv_view`、`vt_tr_source`、`kd_read_tv` | `crates/amdgpu-kernels/src/fmha/mod.rs:26-55` |
| `KvRing3`（`new`、`enable_buffer_dma`、`stage`、`store_constants`） | `crates/amdgpu-kernels/src/fmha/mod.rs:58-185` |
| `AttnBody3`（`qk_block`、`qk`、`pv`、`out_scale`） | `crates/amdgpu-kernels/src/fmha/mod.rs:187-275` |
| `impl PipeBody for AttnBody3`（`v_phase`、`m_phase`、`window`） | `crates/amdgpu-kernels/src/fmha/mod.rs:277-360` |
| `store_out3`、`store_out3_lds`、`store_partial3` | `crates/amdgpu-kernels/src/fmha/mod.rs:603-756` |
| `combine_kernel`、`lut_copy`、`FmhaBlock::out_scale` | `crates/amdgpu-kernels/src/fmha/io.rs:213-249` |
| `Pipeline`、`Cadence`、`TileIt`、`PipeBody` | `crates/amdgpu-tile/src/pipeline.rs:24-157` |
| `OnlineSoftmax`、`Recipe`、`set_vendor_ref`、`decide_impl` | `crates/amdgpu-tile/src/softmax.rs:16-287` |
| `to_p8`、`row_max`、`KeyMask`、`SmfmacOnes` | `crates/amdgpu-tile/src/reduce.rs:13-173` |
| `RowSum2`、`prove`、`search`、`plan` | `crates/amdgpu-tile/src/rowsum.rs:20-119` |
| `partition_dma`、`BufForm`、`issue_buffer` | `crates/amdgpu-tile/src/dma.rs:37-149` |
| `make_tiled_copy*`、`partition_lds` | `crates/amdgpu-tile/src/copy.rs:31-98` |
| `make_tiled_mma`、`gemm` | `crates/amdgpu-tile/src/mma.rs:15`、`crates/amdgpu-tile/src/mma.rs:68-82` |
| `combine_splits` | `crates/amdgpu-tile/src/epilogue.rs:93-171` |
| `search_attention` | `crates/amdgpu-tile/src/orient.rs:121` |
| `Swizzle`、`ComposedLayout`、`make_layout` | `crates/amdgpu-layout/src/swizzle.rs:14-37`、`crates/amdgpu-layout/src/swizzle.rs:106`、`crates/amdgpu-layout/src/layout.rs:25` |
| ISA 原子 | `crates/amdgpu-isa/src/lib.rs:66-99`、`crates/amdgpu-isa/src/lib.rs:165-264` |
| `CompileProfile`、`policy_name` | `crates/amdgpu-dsl/src/lib.rs:971-985`、`crates/amdgpu-compiler/src/lib.rs:31-41` |
| host 自动派发 | `tests/gpu/ref_host.cpp:56-62`、`tests/gpu/ref_host.cpp:122-155` |
