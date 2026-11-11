# 用 FMHA 讲清楚：在 rust_dsl 里怎样构建、怎样优化一个 kernel

> **读者**：新的 kernel 作者（人或 agent）。
>
> **本文回答三个问题**：
> 1. 今天一个 kernel 从描述到汇编、到主机调用经过哪些层；
> 2. 我们在 FMHA 上实际用的优化方法，以及每一步背后的证据；
> 3. 从 TIRx 借鉴了什么、哪里有意不同。
>
> **对象**：出厂 FMHA `fmha_fwd_hd128_mxfp8_v3_{vs128, t128, kvsplit, kvsplit_combine8h}`（MI355X / gfx950，MXFP8，head dim 128）。
>
> **约定**：代码引用写"文件 + 符号名"，不写行号。数字都是同一次运行内对厂商 kernel 的比值、真实数据（`REF_DATA=normal`），出处随文给出。
>
> **延伸阅读**：
> - 逐个 tile 的细节：[`fmha-dsl-tour.md`](fmha-dsl-tour.md)；
> - stage 库设计：[`fusion-stage-library-design.md`](fusion-stage-library-design.md)；
> - 算子图：[`opgraph.md`](opgraph.md)；
> - 与 TIRx 的逐项对照：[`tirx-comparison.md`](tirx-comparison.md)。

---

## 1. 自顶向下：FMHA 是怎么"写出来"的

```
算子图 (fmha/vs128.graph.json + graph.rs)
  └─ amdc opgraph ──► 生成的骨架 (fmha/skeleton_vs128.rs：trait Vs128Slots + build_vs128)
        └─ slot 实现 (fmha/mod.rs：impl Vs128Slots — prelude / pipeline / dispatch)
              └─ stage 实例 (move.kv_stage / online.softmax / mma.pv + mma.qk，带 StageContract 与交接)
                    └─ 发射单元 + weaver（顺序、区域；编排留在 kernel：Pipeline 节奏、角色、ping-pong、XCD 顺序）
                          └─ 区域 IR (amdgpu-dsl Function：Node::Stage / weave 区域 / tile 协议标记)
                                └─ 编译器：lowering → 调度（预设 + token + 区域策略）→ 寄存器分配（+ 寄存器规划）→ hazard / wait → gfx950 汇编
                                      ├─ 严格验证器：证书（可信假设台账）
                                      └─ 主机合同：kernarg ABI、网格、张量范围 → ref_host / Python JIT
```

### 1.1 算子图：有哪些 stage、谁连谁

`crates/amdgpu-kernels/src/fmha/graph.rs::graph_vs128()` 与它的 JSON 镜像 `fmha/vs128.graph.json` 描述：
- **节点**：stage 实例，含 registry key、参数、资源估计、所在相位与 pipeline hook；
- **边**：张量，含布局、作用域、存储，以及交接代价类 C0–C4；
- **weave group**：哪几个 stage 由同一个 weaver 交织。

```json
{ "name": "fmha_vs128", "attrs": { "threads": 512, "pipeline": "TwoPhaseOffset", "dispatch": "roles + versions" },
  "nodes": [ { "id": "kv_stage", "prim": "M", "stage": "move.kv_stage", "params": { "slots": 2 }, "phase": "loop", "hook": "stage", … },
             { "id": "softmax",  "prim": "O", "stage": "online.softmax", "hook": "v_phase", … },
             { "id": "pv", "prim": "A", "stage": "mma.pv", "hook": "m_phase", … }, { "id": "qk", … }, { "id": "out", "stage": "map.store_bf16", … } ],
  … edges: kv(C3, block)  s: qk→softmax (C0, loop-carried)  p: softmax→pv (C0)  o: pv→out (C2) … }
```

几条交接的含义：
- **S → P 为 C0**（零代价）：方向是 `orient::search_attention` 推导出来的——S^T = K·Q^T，P 按累加器顺序打包后直接作为 PV 的 B 操作数；
- **K/V 为 C3**：经 LDS 环，需要 barrier；
- **O → epilogue 为 C2**：wave 私有 LDS 转置。

### 1.2 骨架：生成的、不手改的结构

`amdc opgraph fmha/vs128.graph.json` 生成 `fmha/skeleton_vs128.rs`，测试会重生成并逐字比较。生成内容：
- 一个评审头（stage、片上边及代价类、资源估计）；
- trait `Vs128Slots`，三个 slot：`prelude` / `pipeline` / `dispatch`；
- `build_vs128(name, slots)`：先 `check_kernel`（节点集合、代价类）对照图，再依次调用三个 slot。

**骨架只管结构，不管顺序与节奏。**

### 1.3 slot：每个形状的编排写在这里（`fmha/mod.rs`，`impl skeleton_vs128::Vs128Slots for Vs128Slots`）

```rust
fn prelude(&mut self, b: &mut FnBuilder) -> FmhaCx {
    b.compile_profile(CompileProfile::Vs128);                 // 编译策略由 profile 决定，与名字无关
    let io = declare_io(b, 256, false);                       // ABI + 主机合同（张量形状 / 步长来自 kernarg）
    let blk = block3_x(&io, b, 8, true);                      // 行号、Q 常驻寄存器、XCD 感知的 block 顺序（§2 的最后一项优化）
    let roles = Roles::split(b, &blk.lane, 4);                // 两组各 4 wave
    let mut ring = KvRing3::new(b, &io, &blk, &roles, 2);     // LDS 计划 + 所有拷贝 / 读取分区（布局层推导）
    ring.enable_buffer_dma(b); ring.enable_buffer_kd(b);      // MUBUF 形式的 DMA
    let mut pipe = Pipeline::new(Cadence::TwoPhaseOffset, blk.nkv, blk.kv_len, 128, …);   // 节奏是数据
    pipe.key_unclamped = true;
    …
}
fn dispatch(&mut self, b: &mut FnBuilder, cx: &mut FmhaCx, pipe: &Pipeline) {
    versioned(b, lut, |b, lut| roles.each(b, |b, role| {     // 短 / 长序列 × 组 A / B = 4 份代码
        let mut body = AttnBody3::new(b, ring, blk, wid, lut);
        body.kscale_sel = true; body.setprio = true; body.biasonce = true;
        if !lut { body.sm.set_vendor_ref(b); body.lazy_bias = true; }   // ≥ 4 个 tile：厂商配方 + 懒更新
        pipe.run(b, role, &mut body);                         // 整个 KV 循环
        store_out3_lds(io, b, blk, &ring.epi, wid, &body.o.expect("O"), body.out_scale(b));
    }));
}
```

**编排**全部在这一层，库与 IR 里都没有：
- ping-pong 的相位、`setprio`；
- 短 / 长序列两份代码；
- block 顺序；
- 调度 token（`f.sched_opts = "ldsink=1,wsink,ldahead32=1"`）。

### 1.4 stage 实例与契约（`AttnBody3` 的 `PipeBody` 相位）

`Pipeline::run` 生成 wait、barrier、槽与代、剥离的末 tile。在每个 hook 里，`AttnBody3` 构造 `amdgpu_stage::stage::StageInst`，即 registry key + `StageContract` + 发射单元：

| hook | 实例 | 契约里的交接（GV-STAGE 检查） |
|---|---|---|
| `stage` | `kv.stage`（`move.kv_stage`，每条 DMA 一个单元） | slot 偏移（`SlotIdx::offset_scalars`，owner = wave） |
| `v_phase` | `v_phase`（`online.softmax`：`softmax.decide` 带回滚分支 + `softmax.convert`） | 交出 `P`（16 字）与参考字 `ref`（owner = row） |
| `m_phase` | `stage::merge("m_phase", …)` 合并 `mma.pv` 与 `mma.qk` | 接收 `P`、`ref` 与 slot 偏移 |

`StageContract` 的字段：
- 每个交接值的 TV 布局与存储；
- 标量的 owner；
- 允许在途的 vm / lgkm 数；
- barrier 相位（版本 / 角色分支里用 `ANY_PHASE`）；
- 资源估计。

验证器在 stage 边界检查：跨区域读必须声明为交接、布局 / 存储 / owner 一致、在途数不超、barrier 数一致。`_vs128` / `_kvsplit` 各证明 12 个交接。

### 1.5 发射单元与 weaver

stage 返回 `Units`（`amdgpu_stage::weave`），而不是直接发射：

```
m_phase：pv.setup  pv.rowsum  pv.read.h0e0 pv.mma.h0e0 …  qk.setup(bias MFMA)  qk.read0 qk.mma0 … qk.end
```

每个单元带依赖、资源类、提示（skeleton / filler / latency-sensitive / anchor）。发射方式：
- `StageInst::emit` 用 `Weaver::new(units).phase(name).seq()`；
- `emit_with(|w| w.order(…))` 换成手写顺序；
- 发射前都做拓扑检查。

输出是普通的区域 IR，带 weave 区域标记，可以：
- 按名字声明调度策略：`b.declare_region("m_phase", authored, tokens)`，实验时用 `AMDC_SCHED_REGION`；
- 设寄存器上限：`b.reg_cap(region, n)`；
- 用 `amdc regplan` 查看每个区域的真实寄存器峰值。

### 1.6 区域 IR → 编译器 → 证书 → 汇编

**IR**：`amdgpu-dsl::Function` 是区域树，包括：
- 循环 / if；
- `Node::Stage`（≈ TIRx 的块级区域）；
- tile 协议标记（`TileProduce` / `TileConsume`：LDS 槽的读写代）；
- `Assume`（区间事实）。

IR 有文本形式，可以打印、编辑后再编译（[`ir-text.md`](ir-text.md)）：

```
amdc print-ir fmha_fwd_hd128_mxfp8_v3_vs128 --stage dsl --out vs128.ir
amdc emit --from-ir vs128.ir --out vs128.s        # 不改 Rust，直接拿编辑过的 IR 做实验
```

**编译器**（`amdgpu-codegen`）：
- **lowering**：由 profile 选择选项。
- **调度**：profile 预设 + kernel 的 `sched_opts` + 按区域的策略（见 [`weaver-sched.md`](weaver-sched.md)、[`lgkm-lookahead.md`](lgkm-lookahead.md)）。
- **寄存器分配**：VGPR / AGPR，可用 pin / reg_cap，见 [`weaver-regs.md`](weaver-regs.md)。
- **hazard / waitcnt**：包括 `hzfill`。
- **发射**：gfx950 汇编。

**严格验证器**（`amdc gpu-verify --strict`）证明以下内容：
- 每次访存对 ABI 合同在界内；
- LDS 环的读写协议；
- wait / barrier 完整；
- stage 交接。

它不能证明的事实必须写成 `assume_trusted`（带理由），并列入证书。当前出厂四个 kernel 合计 **21 trusted assumes / 100 trusted contracts / 0 unproven**（逐项见 [`trust-inventory.md`](trust-inventory.md)）；其中 2 条是 XCD 顺序重映射后 tile / head 的范围事实。

### 1.7 主机侧

- **主机合同**：`declare_io` 用 `KArgs::tensor` 声明 BSHD 张量，同时生成合同：`arg_tensor`（分配大小）、`ka_is`（kernarg 关系）、`grid_is`（网格）。`amdc gen-host` 生成 C 头，`amdc jit-info` 输出 JSON。
- **ref_host**（`tests/gpu/ref_host.cpp`）：同一进程里对比我们的、厂商的、CPU 参考，并按真实数据拟合出的规则自动派发：
  1. `ntiles ≥ 16 && N256 ≤ 64` → KV-split；
  2. 否则 `N256 ≤ 64` → `_t128`；
  3. 其余 → `_vs128`。
- **Python JIT**（[`python-jit.md`](python-jit.md)）：`amdgpu_jit.fmha_fwd_mxfp8(q, k, v, …)` 调用 `amdc emit --specialize`、clang、ctypes HIP，带磁盘缓存，并在 launch 前校验主机合同。

---

## 2. 优化方法：我们在 FMHA 上实际怎么做

### 2.1 方法（按顺序）

1. **用真实数据测。** harness 的调色板数据（`lcg`）让 softmax 几乎每个 tile 都回滚，会掩盖快速路径的成本——厂商在真实数据上快约 40 %。所以从那一轮起，矩阵和门禁都用 `REF_DATA=normal`。
2. **先归因，再动手。**
   - **ATT**（`tools/att_fmha.sh`）：拿到 MFMA 流的间隙归因，例如 barrier 37 %、相位末尾的 `vmcnt` 36 %。
   - **计数器**（rocprofv3）：LDS 指令与忙周期、bank 冲突、L2 命中 / 缺失、DRAM credit 停顿。
   - **汇编对照**：`tools/asm_diff.py` 对照厂商汇编，记录在 [`fmha-vs-vendor-asm-diff.md`](fmha-vs-vendor-asm-diff.md)。
3. **用消融估上界，再决定值不值得做。** 例如：
   - 去掉 barrier 反而慢 16–29 %：barrier 维持着 ping-pong，不是开销；
   - 去掉 vmcnt 等待约 0；
   - 去掉所有拷贝 +11–20 %（≥ 厂商）。

   由此得出：问题在拷贝本身，再往下才找到了 L2 局部性（§2.2 最后一行）。
4. **同一次运行内交替 A/B。**
   - `tools/ab_fmha.sh`，≥ 5–7 轮取中位数；
   - 对布局敏感的形状（per-CU `_vs128` 前置 0–7 条 `s_nop` 有 ±2 %）看 8 种布局的中位数；
   - 只采纳测出来的收益。
5. **门禁。** `tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3`：

   | 段 | 检查 |
   |---|---|
   | BUILD | 构建 |
   | VERIFY | 严格验证 0 错误 |
   | ACC16 / ACC64 | 与厂商同轮同输入，mean_err ≤ 1.10× 且 bad ≤ 厂商 |
   | DET | 确定性 |
   | STRESS | 压力 |
   | PERF | 同轮厂商比 ≥ 0.97 × 记录的出厂比 |

   最后用 `tools/perf_matrix.sh` 重测 25 形状矩阵（[`perf-matrix.md`](perf-matrix.md)）。
6. **出厂代码不留环境变量开关或实验变体。** 实验在分支上，结论写进文档，负结果同样记录。

### 2.2 FMHA 的路程：采纳的改动

| 改动 | 证据 / 收益（真实数据，同轮对厂商） |
|---|---|
| 懒参考点更新 + 厂商参考配方（≥ 4 个 tile；更短的序列保留精确 LUT 配方） | 采用时 B1H8S8192 +5.5 %，其余 +0.7…0.9 %；精度与厂商同分布。见 [`current-status.md`](current-status.md) §4.3 |
| bias 单次 MFMA 写进 S tile 3、零累加器（不再有零片段拷贝） | 修掉 M 相位前的零片段拷贝，见 [`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md) §1 |
| `ldsink=1`：`ds_read` 下沉到首个消费者前至多 1 条 MFMA | `_vs128`：per-CU +1.3 %、S8192 +0.6 %；`_kvsplit` +5.5…+6.8 %（[`lgkm-lookahead.md`](lgkm-lookahead.md)） |
| `wsink` + `ldahead32=1`：两处 M 相位 `lgkmcnt(0)` 变成计数等待 | +1.8 % / +1.1 %，per-CU +1.0 %，VGPR 248 → 240 |
| `_t128` 预设修复：kernel 的 token 叠加在 profile 预设之上 | 恢复了 `_t128` 被 `ldsink` 覆盖掉的跨 barrier ALU 调度与 VGPR 上限 248 |
| 用真实数据重新拟合派发规则 | `_vs128` 从 N256 = 128 起领先 `_t128`（B1H32S1024 +13 %）；split4 在 B1H8S2048 +11 % |
| **XCD 感知的 block 顺序**（厂商 `label_xcd_skip`），X > 16 时改为每块 16 个 | L2 命中率 20 % → 77 %；S ≤ 4096 的形状 +3.5…+8.3 %，S = 8192 +0.3…+3.3 %；25 形状几何平均 0.980 → **0.992**（[`fmha-wave-org.md`](fmha-wave-org.md)） |

当前门禁（真实数据）：

| 项 | ×厂商 |
|---|--:|
| KV-split B1H8S2048 | 1.025× |
| per-CU | 0.95–0.96× |
| B4H8S2048 | 0.95–0.98× |
| B1H8S8192 | 0.99× |

### 2.3 重要的负结果（都有分支与数据）

| 方向 | 结论 |
|---|---|
| 每 SIMD 1 wave 的 4-wave 设计（64 行 / 32 行 / K-V 复用） | 0.43–0.75× 厂商：单 wave 发射受限，复用变体寄存器溢出（[`fmha-4wave-design.md`](fmha-4wave-design.md)） |
| weaver 手写顺序、authored 区域、区域 token | list 调度器胜过任何手写顺序（−4…−12 %）；token 只是在 per-CU 与大形状之间互换（[`fmha-weaver-parity.md`](fmha-weaver-parity.md)） |
| 拷贝路径变体：3 槽环、提前 / 推迟发射、厂商 K 组织方式、slot padding、拷贝分散 | 每一种都让拷贝落在 M 相位、拖慢 LDS 读（[`fmha-vendor-stream-design.md`](fmha-vendor-stream-design.md) §4、§6） |
| MFMA 成对发射（`mfmapair`）、更深的 LDS 预读（`ldahead=D`） | ≤ 基线；寄存器已在 256，lgkmcnt 只有 4 位 |
| 去掉 bias MFMA 的各种方式 | −0.8…−15 %：那段 MFMA 时间正好遮盖 barrier 之后的 LDS 读延迟 |
| 生产者 wave、每组独占 ring | gfx950 的 VGPR 按 kernel 统一分配（3 × 240 > 512）；没有 named barrier |

### 2.4 工具

| 工具 | 用途 |
|---|---|
| `tools/ab_fmha.sh A B -- "<shape>" …` | 同一次运行内交替 A/B |
| `tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3 [tag]` | 完整验收门禁 |
| `tools/perf_matrix.sh fmha` + `tools/perf_matrix_md.py` | 25 形状矩阵 |
| `tools/att_fmha.sh <tag> B H S <kernel>` + `tools/att_gaps.py` / `tools/att_classes.py` | ATT 采集、MFMA 流间隙归因、按指令类的停顿统计 |
| rocprofv3 `--pmc`（`SQ_*`、`TCC_*`） | LDS、L2、DRAM 计数器 |
| `amdc regplan <kernel>` | 每个 weave / stage 区域的真实寄存器峰值 |
| `amdc print-ir … --out x.ir` / `amdc emit --from-ir x.ir` | 改 IR 文本做实验，不必改 Rust |
| `AMDC_SCHED_ADD=<tokens>`、`AMDC_SCHED_REGION=<区域>:<authored\|free>[:tokens]` | 不改代码试调度 token / 区域策略（只用于实验） |
| `tools/asm_diff.py`、`tools/mfma_reuse.py` | 与厂商汇编逐段对照 |

---

## 3. 从 TIRx 借鉴了什么，哪里有意不同

逐项对照与现状见 [`tirx-comparison.md`](tirx-comparison.md)，其中：
- [§3 继承并已实现](tirx-comparison.md#3-继承并已实现)；
- [§5 有意不同](tirx-comparison.md#5-有意不同)；
- [§8 TIRx 没有、我们有的](tirx-comparison.md#8-tirx-没有我们有的)。

下面只用 FMHA 的例子说明。

### 3.1 借鉴的

| 思路（TIRx） | 在 FMHA 里的样子 |
|---|---|
| 布局代数 / TV 布局（CuTe 式） | `KvRing3::new` 用 `make_tiled_copy_a(DS_READ_B128X2, &mma)` 与 swizzle 视图 `Swizzle::fmha128()` 推导出每个 lane 的 (base, imm)，构建期证明连续性与双射；V 的转置读、epilogue 转置同理 |
| 指令 / 原子的属性表 | `amdgpu-isa` 的 CopyAtom / MmaAtom 带 TV 布局与 `provenance`（出厂验证 / 探针实测）；调度器的指令属性表（[`op-attributes.md`](op-attributes.md)） |
| 区域 IR（RegionStmt / 块） | `Node::Stage` + `StageContract`（m_phase、v_phase、kv.stage）、weave 区域、tile 协议区域 |
| 结构相等、α-规范化与文本 round-trip（TVMScript） | `print.rs` 的结构相等、α-规范形式；IR 文本 `print-ir` / `ir-check` / `--from-ir`；测试要求全部 kernel round-trip 并发射相同汇编 |
| 带事实的符号分析器 | `amdgpu-sym` 区间 / 多项式与 `Assume` / `assume_trusted`，例如 tile 计数器的界、XCD 重映射后 tile / head 的范围 |
| 编排在源码里显式写出 | slot 实现里的 `Pipeline` 节奏、`Roles`、`versioned`、weaver 顺序、`declare_region`；库与 IR 不替 kernel 决定顺序 |

### 3.2 有意不同的

| 我们 | 原因（FMHA 的例子） |
|---|---|
| 自研调度器与寄存器分配，不走 LLVM 的 | `ldsink` / `wsink` / `ldahead32` / `hzfill` 这类 FMHA 专用的后处理，与 AGPR pin / reg_cap 需要直接控制 MIR；LLVM 尾部 pass 只用于部分 kernel（`--backend llvm-tail`） |
| 静态 GPU 语义验证器 + 证书 | 拷贝协议（LDS 环的槽与代）、越界、wait 完整性在编译期证明；改动若引入新的可信假设，证书数会变，评审时可见（例如 XCD 顺序 24 → 26） |
| 全 Rust、`forbid(unsafe_code)`、零第三方依赖 | 构建与审计简单，可以在节点容器里复现 |
| 没有 FFI 到 TVM / Python 编译栈 | Python 只做 JIT 入口（子进程调用 `amdc` + ctypes HIP），编译器本身不依赖 Python |

---

## 4. 写一个新 kernel 时照这个顺序

1. 在算子图里写 stage 与边（`graph.rs` + JSON），用 `amdc opgraph` 生成骨架。
2. 实现 slot：
   - `prelude`：ABI 与合同、LDS 计划；
   - `pipeline`：节奏；
   - `dispatch`：角色 / 版本、stage 实例。
3. stage 用库里的实例与原语（`amdgpu-tile::stage` / `prims`）；交接写进契约。
4. 用 `gpu-verify --strict` 拿到证书：0 错误，所有 `assume_trusted` 都写明理由。
5. 用真实数据、同轮 A/B 对照参考实现；先用计数器 / ATT / 消融找瓶颈，再改编排。
6. 过门禁、记录 md5 与数字；负结果写进文档。
