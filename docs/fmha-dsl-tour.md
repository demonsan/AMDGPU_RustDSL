# FMHA v3 导读：从算子图到一个 128-key tile 的完整旅程

> **读者**：想读懂（或修改）出厂 FMHA kernel 的人。要求会 Rust 语法，知道 GPU wave / lane / LDS / MFMA 是什么；不要求读过仓库里别的文档。
>
> **范围**：四个出厂 kernel `fmha_fwd_hd128_mxfp8_v3_{vs128, t128, kvsplit, kvsplit_combine8h}`。源码在：
> - `crates/amdgpu-kernels/src/fmha/`：`mod.rs`、`io.rs`、`graph.rs`、`skeleton_*.rs`；
> - `crates/amdgpu-tile/src/`：`stage.rs`、`weave.rs`、`pipeline.rs`、`softmax.rs` 等。
>
> **约定**：引用写成 `文件` + 符号名（不写行号；用 grep 找符号）。节选里省略的部分用 `…` 标出。
>
> **先读**：[`fmha-case-study.md`](fmha-case-study.md)（以 FMHA 为例的构建与优化全景、方法与路程、TIRx 对照）；本文是它的逐 tile 细节篇。
>
> **配套文档**：
> - stage / weaver / 后端三层设计：[`fusion-stage-library-design.md`](fusion-stage-library-design.md) §7–8；
> - 算子图与骨架生成：[`opgraph.md`](opgraph.md)；
> - 布局层入门：[`layout-tour.md`](layout-tour.md)；
> - 验证器：[`gpu-semantics-verifier.md`](gpu-semantics-verifier.md)；
> - 与厂商汇编的对照：[`fmha-vs-vendor-asm-diff.md`](fmha-vs-vendor-asm-diff.md)；
> - 当前数字：[`current-status.md`](current-status.md)。

---

## 目录

- §0 全景：crate 分层与一个 tile 的数据流
- §1 三条出厂路径与自动派发
- §2 动手：构建、emit、验证、gate
- §3 kernel 的形态：算子图 → 骨架 → slot → stage 实例 → 发射单元
- §4 一个 tile 的完整旅程（a–j）
- §5 `_t128` 与 `_kvsplit` 哪里不同
- §6 布局层与 ISA 原子
- §7 正确性由什么保证
- §8 改代码前先看这里

---

## §0 全景：crate 分层与一个 tile 的数据流

```
amdgpu-isa      指令原子：CopyAtom（ds_read_b128 x2、ds_read_b64_tr_b8 x4、buffer/global_load … lds）、MmaAtom 的 TV 布局
amdgpu-layout   Layout / Swizzle / ComposedLayout，TV 布局 → 每个 lane 的 (base, imm) 地址计划，构建期证明
amdgpu-tile     部件：TiledCopy / TiledMma / partition_*、DMA 分区、OnlineSoftmax、行和、epilogue
                编排：stage.rs（StageInst、registry、merge）、weave.rs（Units / Weaver）、pipeline.rs（Pipeline + PipeBody）
                opgraph/：算子图、切分、骨架生成器
amdgpu-kernels  fmha/：graph.rs（算子图）→ skeleton_*.rs（生成的骨架）→ mod.rs（slot 实现、stage 实例、部件组装）
        ↓  amdgpu-dsl 的 Function（结构化 IR；Node::Stage 区域 + weave 区域标记，均不产生代码）
amdgpu-codegen 下降、调度（按区域的策略）、寄存器分配、GPU 语义验证器（含 GV-STAGE）、gfx950 汇编
```

FMHA 计算 `O = softmax(Q K^T) V`：
- head dim = 128；Q/K/V 为 MXFP8（E4M3，每 32 个元素一个 E8M0 缩放）；O 为 bf16。
- key 维切成每 128 个 key 一个 tile。
- 每个 workgroup 处理一段 query 行（`_vs128` 256 行，`_t128` 128 行）；每个 wave 负责 32 行，依次处理所有 KV tile。

```
 全局内存               LDS（2 或 3 槽环形缓冲）         寄存器（每 wave）
 K[128 keys][128 d] ──DMA──▶ K 槽 ──ds_read_b128 x2──▶ K 片段 ─┐
 K 的 E8M0 scale  ──DMA──▶ KD 槽 ──ds_read_b32────────▶ scale ─┤ QK：S^T = K·Q^T（4 个 32x32 累加器）
 Q（常驻寄存器，prologue 读一次）──────────────────────────────┘     ↓
                                                     online softmax：行最大值、参考点、懒回滚
                                                                     ↓
                                                     P = fp8(2^S')：pknorm + scalef32 打包
                                                                     ↓
 V[128 keys][128 d] ──DMA──▶ V 槽 ──ds_read_b64_tr_b8 x4──▶ V^T 片段 ─▶ PV：O^T += V^T·P^T，行和 l（2 条 smfmac）
                                                                     ↓（所有 tile 之后）
                                                     epilogue：O·(V_descale / l) → bf16 → 全局
```

---

## §1 三条出厂路径与自动派发

| kernel | 每 block | 结构 | 何时用 |
|---|---|---|---|
| `…_v3_vs128` | 256 行、512 线程（两组各 4 wave） | `Cadence::TwoPhaseOffset`：两组错开一个相位，2 槽环；MUBUF K/V/K-scale DMA；优先级 ping-pong；两条 smfmac 行和；K-scale 字节选择；≥ 4 个 tile 用厂商参考配方 + 懒更新；LDS 中转 epilogue | 填满的大 shape |
| `…_v3_t128` | 128 行、256 线程（一组） | `Cadence::OneWindow`：3 槽环，每 tile 一个窗口；K-scale 走 VGPR | 小 shape（grid 一波放得下） |
| `…_v3_kvsplit` + `…_combine8h` | 256 行、512 线程 + 合并 kernel | `_vs128` 的结构跑在一个 key 区间上，写 fp16 部分输出与 `lse`，再合并 | 长 KV、block 很少（如 B1H8S2048） |

**自动派发在 host 侧**（`tests/gpu/ref_host.cpp`），按真实数据测出的规则：
1. 先看是否用 KV-split：`ntiles ≥ 16 && N256 ≤ 64` 时切 4 份，`ntiles ≥ 32 && N256 ≤ 128` 时切 2 份。
2. 不切分时按 grid 大小：`N256 ≤ 64` 用 `_t128`，否则用 `_vs128`。

三条路径共用同一个 656 字节的厂商 kernarg ABI（KV-split 多 64 字节扩展），host 只换 code object 与 grid。

---

## §2 动手：构建、emit、验证、gate

```bash
cargo run --bin amdc -- list
cargo run --bin amdc -- print-ir fmha_fwd_hd128_mxfp8_v3_vs128 --out vs128.ir      # IR 文本（可 round-trip，见 docs/ir-text.md）
cargo run --bin amdc -- emit fmha_fwd_hd128_mxfp8_v3_vs128 --backend own --out vs128.s
cargo run --bin amdc -- gpu-verify fmha_fwd_hd128_mxfp8_v3_vs128 --strict           # 0 error；打印证书
cargo run --bin amdc -- regplan fmha_fwd_hd128_mxfp8_v3_vs128                       # 每个 weave / stage 区域的真实寄存器峰值
cargo test -p amdgpu-tile --test weave --test stage_registry --test opgraph
HIP_VISIBLE_DEVICES=5 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3 mytag          # GPU 验收 gate
HIP_VISIBLE_DEVICES=5 bash tools/ab_fmha.sh fmha_fwd_hd128_mxfp8_v3_vs128 <变体> -- "4 2048" "1 8192"
```

- **gate 判定**：
  - 精度与同一次运行的厂商对齐：mean_err ≤ 1.10 × 厂商，坏点数 ≤ 厂商；
  - 性能 ≥ 0.97 × 记录的出厂比值；
  - 用真实数据（`REF_DATA=normal`）。
- **编译策略**：用 `CompileProfile` 声明，与 kernel 名字无关。调度器 token 写在 `Function::sched_opts`，例如 `_vs128` 的 `ldsink=1,wsink,ldahead32=1`。

---

## §3 kernel 的形态：算子图 → 骨架 → slot → stage 实例 → 发射单元

出厂 kernel 分五层写成。前三层决定"有哪些 stage、谁连谁"；后两层决定"每个 stage 做什么、按什么顺序发射"。

### §3a 算子图（`fmha/graph.rs`）

`graph_vs128()` 描述的内容：
- **节点**：stage 实例，含 registry key、参数、资源估计、所在相位与 pipeline hook；
- **边**：张量，含布局、作用域、存储，以及交接代价类 C0–C4；
- **weave group**：哪些 stage 由同一个 weaver 交织。

```
kv_stage [M: move.kv_stage] (slots=2)          hook stage
softmax  [O: online.softmax] (vendor two-word reference, lazy rollback)   hook v_phase
pv       [A: mma.pv]                             hook m_phase  ┐ weave group m_phase
qk       [A: mma.qk] (K-scale byte select)       hook m_phase  ┘
out      [E: map.store_bf16] (wave-private LDS transpose)   epilogue
on-chip: kv kv_stage→qk,pv C3 (block) | s qk→softmax C0 (loop-carried) | p softmax→pv C0 | o pv→out C2
```

图同时以 JSON 存在（`fmha/vs128.graph.json`）。测试保证 JSON 与 `graph_vs128()` 一致、切分与代价类一致。

### §3b 骨架（`fmha/skeleton_vs128.rs`，生成文件，勿手改）

由 `amdgpu_graph::opgraph::skeleton` 从图生成，内容包括：
- trait `Vs128Slots`，三个 slot：`prelude` / `pipeline` / `dispatch`；
- `build_vs128(name, slots)`：先检查图（节点、代价类），再依次调用三个 slot。

**每个形状的编排全部在 slot 里，图和骨架不规定发射顺序**。

### §3c slot 实现（`fmha/mod.rs`，`impl skeleton_vs128::Vs128Slots for Vs128Slots`）

```rust
fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
    b.compile_profile(CompileProfile::Vs128);
    let io = declare_io(b, 256, false);                  // ABI + 主机合同（§4a）
    let blk = block3(&io, b, 8);                         // 行号、Q 片段、scale（§4a）
    let roles = Roles::split(b, &blk.lane, 4);           // wave 0..3 = 组 A，4..7 = 组 B
    let mut ring = KvRing3::new(b, &io, &blk, &roles, 2); // LDS 计划 + 拷贝分区（§4b、§4c）
    ring.enable_buffer_dma(b); ring.enable_buffer_kd(b);
    let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, …);
    pipe.key_unclamped = true;                           // 所有 DMA 经描述符越界保护
    let lut = blk.lut_copy(b);                           // 少于 4 个 tile → 短序列副本
    FmhaCx { io, blk, roles, ring, pipe: Some(pipe), lut }
}
fn dispatch(&mut self, b: &mut FnBuilder, cx: &mut FmhaCx, pipe: &Pipeline) {
    versioned(b, lut, |b, lut| roles.each(b, |b, role| {      // 2 (短/长) × 2 (组 A/B) 份代码
        let mut body = AttnBody3::new(b, ring, blk, wid, lut);
        body.kscale_sel = true; body.setprio = true; body.biasonce = true;
        body.sm.use_rows2(b, &blk.lane, …);                   // §4h
        if !lut { body.sm.set_vendor_ref(b); body.lazy_bias = true; }   // §4f
        pipe.run(b, role, &mut body);                         // 整个 KV 循环（§4d）
        store_out3_lds(io, b, blk, &ring.epi, wid, &body.o.expect("O"), body.out_scale(b));   // §4i
    }));
}
```

### §3d stage 实例（`AttnBody3` 的 `PipeBody` 相位）

`Pipeline::run` 决定"什么时候做"：wait、barrier、槽与代、剥离的末 tile。`AttnBody3` 在每个 hook 里构造 stage 实例（`amdgpu_stage::stage::StageInst`），每个实例 = registry key + `StageContract` + 发射单元。

| hook | stage 实例 | contract 里的交接 |
|---|---|---|
| `stage` | `kv.stage`（`move.kv_stage`）：组 A 发 K-scale + K，组 B 发 V；每条 DMA 一个单元 | slot 偏移（`SlotIdx::offset_scalars`）作为标量交出 |
| `v_phase` | `v_phase`（`online.softmax`）：`softmax.decide`（含回滚分支）+ `softmax.convert` | 交出 `P`（16 个字）与参考字 `ref` |
| `m_phase` | `m_phase` = `stage::merge` 合并 `pv`（`mma.pv`）与 `qk`（`mma.qk`），一个区域 | 声明接收 `P`、`ref` 与 slot 偏移 |
| `window`（`_t128`） | `window`（`online.softmax` + 下一 tile 的 QK 块） | slot 偏移 |

每个实例发射在 `Node::Stage` 区域里，GV-STAGE 检查以下内容：
- 跨区域读取的值必须声明为交接；
- 布局 / 存储 / owner 一致；
- 在途访存不超过容忍；
- barrier 数。

`_vs128` / `_kvsplit` 证明了 12 个交接，`_t128` 证明了 4 个。

### §3e 发射单元与 weaver（`amdgpu_codegen::weave`）

stage 实例返回发射单元（`Units`），而不是直接发射。`m_phase` 的单元如下：

```
pv.setup  pv.rowsum  pv.read.h0e0 pv.mma.h0e0 … pv.read.h1e3 pv.mma.h1e3      (mma.pv)
qk.setup(bias MFMA)  qk.read0 qk.mma0 … qk.read3 qk.mma3  qk.end            (mma.qk)
```

`StageInst::emit` 用 `Weaver::new(units).phase(name).seq()` 按声明顺序发射，`emit_with(|w| w.order(…))` 可换成显式顺序。每次发射前都会检查拓扑序。

出厂顺序就是声明顺序。改动单元顺序、区域策略或寄存器放置的实验都测成负收益，结果记在 [`fmha-weaver-parity.md`](fmha-weaver-parity.md)。

---

## §4 一个 tile 的完整旅程

### §4a ABI 与每 block 状态

- **`declare_io`**：用 `KArgs::tensor` 把 kernarg 声明成 BSHD 张量。
  - Q/K/V 为 fp8，O 为 bf16，Q/K 的 E8M0 descale 为 `[B, S, H, 4]`，V 的 descale 是一个标量。
  - 同时生成主机合同（`arg_tensor`、`ka_is`、`grid_is`），验证器用它证明每次访存都在界内。
  - KV-split 多出 `Ohat`、`lse`、`splits`、`rows_total` 四个参数。
- **`block3`**：产出 `FmhaBlock`（`io.rs`）。
  - query 行：`row0 = qtile·rows + 32·wave`，越界行夹到 `S-1`；写回时由 `row0 < S` 守卫。
  - KV-split 的 (batch, split) 与 key 区间：`components::kv_split_decode` / `kv_split_range`。
  - GQA：`gqa_kv_head`。
  - Q 常驻寄存器：它是 QK 的 B 操作数，`make_tiled_copy_b(GLOBAL_LOAD_X4, &mma)` 推导出每 lane 4 条 16 字节读。
  - Q 的 scale **减 9**：S 以 1/512 为单位，后面 softmax 的指数都在这个单位上。
  - tile 数 `nkv = ceil(kv_len / 128)`，最后一个可能不满（masked）。

### §4b LDS 计划（`KvRing3::new`）

| 区域 | 大小 | 用途 |
|---|---|---|
| K / V 环 | 每槽 16 KB + 16 KB，`_vs128` 2 槽、`_t128` 3 槽 | 128 key × 128 字节，行内 16 字节块按 `Layout::FMHA_SWZ` 打散 |
| KD 环 | 每槽 512 B | 128 个 key 的 E8M0 scale 字 |
| ones、Q0/Q1 | 2 KB、16 KB × 2 | prologue 写一次的常量（`store_constants`） |
| epi | 每 wave 4 KB 私有区 | epilogue 转置中转，复用 Q0/Q1 的空间 |

`LdsPlan::commit` 把布局写进 Function。验证器据此检查越界，以及环形缓冲每个 (槽, 代) 的读写协议。

### §4c K/V DMA（`move.kv_stage`）

LDS-direct DMA 的目的地址是固定的：lane i 写 `M0 + imm + 16·i`。因此 `partition_dma` 从目的端反推：
- 用视图的逆映射求出每个 lane 写的是哪个元素；
- swizzle 因此被搬到了源地址上。

`_vs128` 用 MUBUF 形式（`DmaPart::issue_buffer_units`）：
- 每条指令一个循环不变的 voffset；
- 每 tile 的推进放进 SGPR soffset，M0 每 tile 一次 `readfirstlane`；
- 越过序列末尾的行落在描述符的 record range 之外，读回 0，之后被 mask 掉。

K 的 scale：`_vs128` / `_kvsplit` 用 DMA；`_t128` 先读进 VGPR，在 `stage_complete` 时再 `ds_write`。

### §4d 流水线（`Pipeline::run`，`Cadence::TwoPhaseOffset`）

```
组 A:  stage(t+1) | V 相位(t) | wait vm | barrier | M 相位(t) | barrier
组 B:  比 A 晚一个相位：A 在 M 相位时，B 在 V 相位
V 相位 = v_phase 实例（softmax，VALU 为主）；M 相位 = m_phase 实例（PV(t) + 行和，然后 QK(t+1)）
```

- **两组错相**：任意时刻总有一组在用 MFMA、另一组在用 VALU。`setprio` 让矩阵相位优先发射。
- **QK 的时机**：QK(t+1) 在 M 相位末尾计算，所以进入 V 相位 (t+1) 时 S 已就绪；QK(0) 在 `prologue_compute` 里算。
- **循环携带状态**：参考字、4 个 O 累加器、4 个 S 累加器、行和；循环变量的顺序为 [it, 状态…, idx]。
- **末 tile 剥离**：masked 与非 masked 各一份，循环体里没有 mask 开销。

### §4e QK：S^T = K·Q^T（`mma.qk`）

方向是推导出来的：`attention_orientation` 调用 `orient::search_attention`，断言最优解为：
- 计算 S^T = K·Q^T（A = K，B = Q）；
- P 直接作为 PV（O^T = V^T·P^T）的 B 操作数，交接是零代价的 C0。

每 32 个 key 一个块（`qk_block` / `qk.read{s}` + `qk.mma{s}`）：
1. **参考点 bias**：`biasonce` 形式下，一条 f16 32x32x8 MFMA 写 S 块 3（`ones × bias`），块 0–2 把它当作第一条缩放 MFMA 的 C 输入。
2. **读操作数**：两个 K 片段（`ds_read_b128 x2`）和一个 K-scale 字（`ds_read_b32`）。
3. **K-scale 字节选择**：每个 lane 按所在半区右移 `8·(lane/32)`，两条缩放 MFMA 用 op_sel 分别取字节 0 和字节 2。
4. **两条 MFMA**：`v_mfma_scale_f32_32x32x64_f8f6f4`。

### §4f online softmax（`online.softmax`，`OnlineSoftmax`）

softmax 不维护运行最大值，而是维护参考点 `b`（两个 f16 字，`b = 4·(hi + lo)`）。S 在 QK 时已经加上了它。

每个 tile 做一次决定：
1. **行最大值**：`v_max3` 链，再与伙伴 lane 交换；
2. **是否越界**：若最大值超过阈值，参考点移动并重新量化，得到 `delta`；
3. **回滚分支**：`S += delta`，`O`、`l *= 2^(512·delta)`。分支没有 exec mask，不回滚的 lane 拿到的 `delta = 0`。

两份副本（`versioned`）用不同的配方：

| | 短序列（< 4 tile） | 长序列（≥ 4 tile） |
|---|---|---|
| 配方 | `Recipe::vsx(true)`：目标 131/512、阈值 120/512 | 厂商参考配方 `set_vendor_ref`：`s91` / `s92` |
| 决定 | `decide`（每 tile 计算新参考字） | `decide_lazy`：重新量化与重缩放全部放进回滚分支 |
| P 转换 | pknorm + LUT 修正 | pknorm，scale 0.25 |

短序列不用厂商配方的原因：tile 太少时懒回滚的误差平均不掉，过不了厂商对齐 gate。懒更新的收益只在真实数据上体现，所以 gate 的 PERF 用 `REF_DATA=normal`。

最后一个不满的 tile 用 `KeyMask`：越界 key 求最大值时取 `-1e30`，求 P 时取 0。

### §4g P 转换

`to_p8` 每 4 个值出一个 fp8 字：
1. `v_cvt_pknorm_u16_f32` 把 x 量化成 u16；把这个 u16 当作 bf16 看，就近似 `2^(512x − 127)`；
2. `v_cvt_scalef32_pk_fp8_bf16` 再转成 E4M3。

所以 P 不需要 `v_exp_f32`。

4 个 S 累加器产出 16 个 P 字，即两个 PV 的 B 片段。按累加器顺序打包即可；key 的排列差异由 V 的转置读取吸收。

### §4h PV 与行和（`mma.pv`）

- **V^T 片段**：`ds_read_b64_tr_b8` 读的同时做 8×8 转置。源布局 `vt_tr_source` 配同一个 swizzle 视图；硬件映射是实测的（`amdgpu_isa::tr_b8`）。
- **8 条 PV MFMA**：2 个 P 片段 × 4 个 O 累加器。V 的 scale 为 127，P 的 scale 为 123（×2^-4），epilogue 中的 `l` 相应除以 16。
- **行和**：两条 `v_smfmac_f32_16x16x128_fp8_fp8`（`rowsum.rs` 的 `RowSum2`，构建期 `prove` 证明每个 P 字节恰好被求和一次）。
  - 每个 lane 的行和 = dword 0 + dword 1，只含它自己那一行 query；
  - 因此回滚时逐 lane 缩放仍然精确（`scale_l`）。

### §4i Epilogue

- `out_scale = V_descale / l`。
- `_vs128`：`store_out3_lds`。按累加器布局 `ds_write_b32` 进每 wave 的私有区（`Swizzle<3,4,3>`），再按行 `ds_read_b128` 读回，最后 8 条 16 字节全局写。
- `_t128`：`store_out3` 直接按 C 布局写 `global_store_dwordx2`。

### §4j KV-split

- **主 kernel**：结构同 `_vs128`，只处理自己 split 的 key 区间。`store_partial3` 写 fp16 的 `Ohat` 和 `lse = log2(l) − 512·b`。
- **合并 kernel**：`combine_kernel` → `combine_splits`，在各 split 上做在线最大值与重缩放。
- **副本选择**：按整个序列的 tile 数选短 / 长副本，与 split 数无关。

---

## §5 `_t128` 与 `_kvsplit` 哪里不同

| | `_vs128` | `_t128` | `_kvsplit` |
|---|---|---|---|
| 线程 / 行 | 512 / 256 | 256 / 128 | 512 / 256 |
| 节奏 | `TwoPhaseOffset` | `OneWindow`（3 槽，每 tile 一个窗口） | `TwoPhaseOffset` |
| stage 实例 | kv.stage、v_phase、m_phase | window（softmax 转换块与下一 tile 的 QK 块交替），PV 在窗口之后 | 同 `_vs128` |
| K/V DMA | MUBUF | global LDS-direct | global LDS-direct |
| K-scale | MUBUF DMA，字节选择 | VGPR 路径 | DMA，`ScaleSel::pair` |
| 行和 | 2 条 smfmac | 4 条 smfmac | 2 条 smfmac |
| epilogue | LDS 中转 | 直接 `dwordx2` | 部分结果 + `lse` |

`_t128` 的窗口参考字在窗口**末尾**才写回（`decide_deferred` + `commit_bias`）：提前写会拉长旧字的活跃区间，使调度器退化。

---

## §6 布局层与 ISA 原子

**布局层**（借用 CuTe 的思路）：
- **Layout**：从坐标到偏移的函数（`shape : stride`）。
- **TV 布局**：从 (线程, 值) 到 tile 元素。
- **`Swizzle::fmha128()`**：用行号的位打散行内 16 字节块，使 `ds_read_b128` 与 `ds_read_b64_tr_b8` 都没有 bank 冲突。
- **`ComposedLayout`** = swizzle ∘ layout。
- **`IssuePlan` / `layout_to_offsets`**：把每条指令的地址拆成"每 lane 一次的 base + 立即数"。
- **构建期证明**（`amdgpu-layout/src/proofs.rs`）：覆盖性、连续对齐、swizzle 双射。证明失败时 builder 直接 panic。

**ISA 原子**（`amdgpu-isa`）：

| 原子 | 用途 |
|---|---|
| `GLOBAL_LOAD_X4`、`GLOBAL_LDS_X4` / `X1`、MUBUF 形式 | Q、K/V、K-scale |
| `DS_READ_B128X2`、`DS_READ_TR_B8X4`、`DS_READ_B32` | K 片段、V^T 片段、K-scale 字 |
| `MmaAtom::mfma_scale_32x32x64_f8`、`mfma_32x32x8_f16`、`smfmac_16x16x128_fp8` | QK / PV、参考点、行和 |

每个原子带 `provenance`：`Shipped`（出厂已验证）或 `Probe`（探针实测）。

---

## §7 正确性由什么保证

| 层次 | 机制 |
|---|---|
| 构建期 | TV 覆盖性、连续对齐、swizzle 双射、S→P 方向搜索断言、行和方案证明；算子图检查（`check_kernel`：节点、代价类） |
| 编译期 | GPU 语义验证器：访存在界内、LDS 环的 (槽, 代) 协议、wait / barrier；GV-STAGE 检查 stage 交接；`--strict` 只允许证书里列出的可信假设（当前全部出厂 kernel 21 / 100 / 0） |
| 测试 | 方向、行和、厂商配方递推（functional proofs）、weave / stage registry / opgraph、IR 文本 round-trip（全部 kernel round-trip 并发射相同汇编） |
| GPU | `tools/port_gate.sh`：精度与厂商对齐、确定性、压力、性能 |

---

## §8 改代码前先看这里

- **不要按 kernel 名字分支**：编译策略走 `CompileProfile`；kernel 特性走 `AttnBody3` 的显式字段（`setprio` / `kscale_sel` / `biasonce` / `lazy_bias`）。
- **改结构就改图**：stage 的增删与连接在 `graph.rs` 里改，然后重新生成骨架（`tools/` 里的 opgraph 生成命令，见 [`opgraph.md`](opgraph.md)）。
- **改编排就改 slot 或 weave 顺序**：发射顺序在 `StageInst::emit_with` 里，区域策略用 `declare_region`。不要改 stage 本身来换顺序。
- **改地址就改布局**：不要手写 lane 地址公式。
- **改节奏就改 `Pipeline`**：wait / barrier / 槽与代都由 `Pipeline::run` 统一生成。
- **验收**：
  - 出厂 kernel 的汇编 md5 不变；若变化，跑完整 `tools/port_gate.sh`；
  - 性能结论用同一次运行内交替 A/B（≥ 7 轮，真实数据）。
