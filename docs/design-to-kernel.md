# 设计如何落到 kernel 上：四层 + 两个 Weaver 的实际路径

本文回答一个问题：**今天的设计（四层 + 前端 / 后端两个 Weaver）在一个具体 kernel 上到底怎么落地**——作者写哪些文件、调用哪些函数、系统替作者决定什么、每一步能看到什么输出、过哪些门禁。

- 依据：Main `0fb4df13`（文中代码行号都对应这个提交）。命令输出在 `ed666ee4` 上取得，两者之间只有文档与 pass 说明文字的变化。
- 所有命令输出都是在 `mi355-gpu-53` 上用该提交的 release `amdc` 实跑的结果，只做了截断，没有改写。门禁数字来自 GPU 0。
- 设计背景见 [`two-weavers-plan.md`](two-weavers-plan.md)、[`two-weavers.md`](two-weavers.md)、[`refactor-4layer-plan.md`](refactor-4layer-plan.md)，分层的英文摘要见 [`architecture.md`](architecture.md)。

---

## 1. 一页总览

| 部分 | crate / 模块 | kernel 作者接触的代码 | 作者决定 | 系统决定 |
|---|---|---|---|---|
| **L1 图** | `amdgpu-graph` | 分派规则 [`dispatch.rs`](../crates/amdgpu-graph/src/dispatch.rs)（如 [`mxfp8_prefill_path`](../crates/amdgpu-graph/src/dispatch.rs#L494)、[`gemm_kernels`](../crates/amdgpu-graph/src/dispatch.rs#L460)、[`split64`](../crates/amdgpu-graph/src/dispatch.rs#L113)、[`mla_config`](../crates/amdgpu-graph/src/dispatch.rs#L169)）；融合规划 [`fusion::plan`](../crates/amdgpu-graph/src/fusion/mod.rs#L385)；生成的 [`compose_facts.rs`](../crates/amdgpu-graph/src/compose_facts.rs#L5) | 每个形状用哪个 kernel、split 多少（写成带证据注释的规则，不做运行时搜索） | 模型图怎样切成 launch（融合规则匹配）、workspace 大小、每个 kernel 的 plan 事实（由 kernel 自己的 Weaver 计划生成） |
| **L2 stage** | `amdgpu-stage` | stage 实例与合并 [`StageInst`](../crates/amdgpu-stage/src/stage.rs#L38)、[`merge`](../crates/amdgpu-stage/src/stage.rs#L125)、[`emit_with`](../crates/amdgpu-stage/src/stage.rs#L83)；流水线模板 [`Pipeline::new`](../crates/amdgpu-stage/src/pipeline.rs#L134) / [`Cadence`](../crates/amdgpu-stage/src/pipeline.rs#L52)；族模板如 [`AttnBody3`](../crates/amdgpu-stage/src/attention/attn3.rs#L260)、`gemm::body::Body` | cadence（`TwoPhaseOffset` / `OneWindow` / `RegStaged`）、环槽数、每个 phase 里放哪些 stage | phase 的屏障 / 等待协议、槽位与代数、角色分工（模板固定） |
| **前端 Weaver** | `amdgpu-stage::weave` + `amdgpu-dsl::units` | 单元捕获 `Units::capture`（[`units.rs:171`](../crates/amdgpu-dsl/src/units.rs#L171)）；计划 API [`Weaver`](../crates/amdgpu-stage/src/weave.rs#L13)：[`free`](../crates/amdgpu-stage/src/weave.rs#L35)、[`budget_vgpr`](../crates/amdgpu-stage/src/weave.rs#L40)、[`interleave`](../crates/amdgpu-stage/src/weave.rs#L45)、[`seq`](../crates/amdgpu-stage/src/weave.rs#L68)、[`emit_phase`](../crates/amdgpu-stage/src/weave.rs#L134)，以及自由函数 [`declare_ring`](../crates/amdgpu-stage/src/weave.rs#L197)（这两个 API 存在，但 Main 上还没有 kernel 调用；见 §6） | 单元的切分与命名、phase 的单元顺序（或声明为 Free 让后端排）、交织提示、寄存器预算、环声明 | 把单元与约束写进 `KernelPlan`（[`plan.rs`](../crates/amdgpu-dsl/src/plan.rs#L15)：[`Phase`](../crates/amdgpu-dsl/src/plan.rs#L80)、[`RingDecl`](../crates/amdgpu-dsl/src/plan.rs#L67)、[`UnitDecl`](../crates/amdgpu-dsl/src/plan.rs#L145)） |
| **L3 功能块** | `amdgpu-tile`、`amdgpu-layout`、`amdgpu-isa` | [`TiledCopy::partition_lds`](../crates/amdgpu-tile/src/copy.rs#L78)、[`dma_plan`](../crates/amdgpu-tile/src/dma_plan.rs#L52) / [`smem_block`](../crates/amdgpu-tile/src/dma_plan.rs#L14)、`TiledMma`、softmax / 归约 / epilogue 件；指令 atom 表 [`gfx950.rs`](../crates/amdgpu-isa/src/gfx950.rs) | 用哪个 atom、哪种布局（tile 坐标 → 字节偏移的 `Layout`） | 每个 lane 的地址、立即数拆分、对齐与连续性证明（构建期检查） |
| **L4 IR + 后端 Weaver** | `amdgpu-dsl`、`amdgpu-codegen`、`amdgpu-target` | 一般不碰。kernel 只在 `Function` 上声明策略：`compile_profile`、`sched_opts`（如 vs128 的 [`ldsink=1,wsink,ldahead32=1`](../crates/amdgpu-kernels/src/fmha/mod.rs#L45)、MLA 的 [`reorder`](../crates/amdgpu-kernels/src/fmha/mla.rs#L314)）、`lower_opts`、寄存器标记（`reg_ring` / [`reg_cap`](../crates/amdgpu-dsl/src/lib.rs#L2287) / pin；`reg_rotate` 不在 Main，存档在标签 `archive/b6b-reg-rotate`） | 要不要打开哪些调度 / lowering 选项（以 perf_ab 的测量为准） | lowering；单元图上的最终顺序：B3 按作者顺序（[`bweave::order`](../crates/amdgpu-codegen/src/bweave.rs#L450)），Free phase 内的交织（B4，含跨迭代单元，[`bweave::interleave_with`](../crates/amdgpu-codegen/src/bweave.rs#L317)）；环槽依赖的等待（B5，等待 pass 里的 `RingModel::of_function` + `GV-PLAN-RING`）；Free phase 的 VGPR 预算作为列表调度器的压力上限（B6a）与 B6c 回退档；寄存器分配（遵守 `reg_ring` 不相交与 pin）；等待（[`insert_waits`](../crates/amdgpu-codegen/src/sched/waits.rs#L481)）；冒险填充；发射；`gpu_verify` 证明 |

**入口。** kernel 按名字注册：[`registry.rs`](../crates/amdgpu-kernels/src/registry.rs#L8) 里的 `kernel_by_name`，例如 [`"fmha_fwd_hd128_mxfp8_v3_vs128"`](../crates/amdgpu-kernels/src/registry.rs#L88)。编译入口是 [`amdgpu_codegen::compile`](../crates/amdgpu-codegen/src/lib.rs#L37)。

**依赖方向。** 依赖只能向下：kernels → L1 → L2 → L3 → L4。由 [`layer_check.rs`](../crates/amdc/tests/layer_check.rs) 检查，下层也不允许按 kernel 名字特判。

---

## 2. 例子 (a)：出厂 MXFP8 `fmha_fwd_hd128_mxfp8_v3_vs128`

### 2.1 L1：分派规则选中它

```text
$ amdc dispatch mxfp8 1 8 2048 2048
{"path":"kvsplit","splits":4,"kernels":["fmha_fwd_hd128_mxfp8_v3_kvsplit","fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h"]}
$ amdc dispatch mxfp8 1 8 8192 8192
{"path":"vs128","splits":0,"kernels":["fmha_fwd_hd128_mxfp8_v3_vs128"]}
$ amdc dispatch mxfp8 4 8 2048 2048
{"path":"vs128","splits":0,"kernels":["fmha_fwd_hd128_mxfp8_v3_vs128"]}
```

- 规则在 [`mxfp8_prefill_path`](../crates/amdgpu-graph/src/dispatch.rs#L494)：小网格走 KV-split，铺满网格走 `_vs128`。
- `amdc compose` 给出同一 launch 的计划事实（来自生成的 `compose_facts`）：

```text
$ amdc compose mxfp8 1 8 8192
{"family":"mxfp8","kernels":["fmha_fwd_hd128_mxfp8_v3_vs128"],"splits":1,"tile":null,"rule":"dispatch::mxfp8_prefill_path",
 "plan":{"threads":512,"waves":8,"phases":72,"units":712,"lds_bytes":102400,
  "rings":[{"name":"K","slots":2,"bytes_per_slot":32768,"owner":"cta"},{"name":"V","slots":2,"bytes_per_slot":32768,"owner":"cta"},
           {"name":"KD","slots":2,"bytes_per_slot":512,"owner":"cta"},{"name":"epi","slots":8,"bytes_per_slot":4096,"owner":"wave"}]}}
```

### 2.2 骨架与 stage 组合

**骨架。** [`skeleton_vs128.rs`](../crates/amdgpu-kernels/src/fmha/skeleton_vs128.rs) 由算子图 `vs128.graph.json` 生成（`@generated`，不要手改）。它固定了 5 个 stage 和它们之间的片上边：

```text
// kernel vs128: 5 stage(s), 512 threads, pipeline TwoPhaseOffset, dispatch roles + versions
//   kv_stage [M: move.kv_stage] (slots=2)  phase loop, hook stage
//   softmax [O: online.softmax] (recipe="vendor two-word reference, lazy rollback")  phase loop, hook v_phase
//   pv [A: mma.pv] (atom="mfma_scale_32x32x64_f8")  phase loop, hook m_phase
//   qk [A: mma.qk] (atom="mfma_scale_32x32x64_f8", scale="K-scale byte select")  phase loop, hook m_phase
//   out [E: map.store_bf16] (n=128, via="wave-private LDS transpose")  phase epilogue
//   weave group m_phase (phase label m_phase): pv + qk
```

[`build_vs128`](../crates/amdgpu-kernels/src/fmha/skeleton_vs128.rs#L42) 先用图检查 stage 名与边的类别，然后依次调用三个 slot：`prelude`、`pipeline`、`dispatch`。

**作者写的 slot**（[`fmha/mod.rs:81`](../crates/amdgpu-kernels/src/fmha/mod.rs#L81) 的 `impl Vs128Slots`）：
- 参数与块号：`declare_io` / [`block3_x`](../crates/amdgpu-stage/src/attention/block3.rs#L57)（256 行一块、XCD 感知顺序）；
- K/V 环：[`KvRing3::new`](../crates/amdgpu-stage/src/attention/attn3.rs#L81)（2 槽）；
- 流水线：`Pipeline::new(Cadence::TwoPhaseOffset, …)`（[`mod.rs:95`](../crates/amdgpu-kernels/src/fmha/mod.rs#L95)）；
- 族模板：[`AttnBody3::new`](../crates/amdgpu-stage/src/attention/attn3.rs#L323)；
- `Vs128Slots { replica: true, … }`：W5 复刻配置，即无缩放 PV、两 tile 一趟、跳转线程化、冷回滚块（[`fmha-replica.md`](fmha-replica.md) §4.9）；lowering 选项 `cmpneg`、`movb64`、`jthread`、`coldlayout` 在 `prelude` 里用 `b.lower_opts` 声明（[`mod.rs:87`](../crates/amdgpu-kernels/src/fmha/mod.rs#L87)）。

### 2.3 前端 Weaver：attn3 里的调用

矩阵 phase 由 [`AttnBody3::m_phase`](../crates/amdgpu-stage/src/attention/attn3.rs#L626) 构建，分三步：
1. [`pv_units`](../crates/amdgpu-stage/src/attention/attn3.rs#L460) 与 [`qk_units`](../crates/amdgpu-stage/src/attention/attn3.rs#L432) 用 `Units::capture` 把每条 LDS 读和每个 MFMA 捕获成命名单元：`pv.read.h{h}e{e}`、`pv.mma.h{h}e{e}`、`qk.mma{blk}` …。
2. 两组单元交给 [`stage::merge("m_phase", …)`](../crates/amdgpu-stage/src/attention/attn3.rs#L699)。
3. 由 [`emit_with`](../crates/amdgpu-stage/src/attention/attn3.rs#L707) 写进计划。出厂路径用 `w.seq()`，即作者给定的顺序。只有实验形式 `ahead2`（ring2）才会走 `w.seq().free()` 加上 `interleave(mfma, dma)` 提示，把 phase 交给后端 B4 / B5 去排。

softmax 是 `v_phase`（[`attn3.rs:579`](../crates/amdgpu-stage/src/attention/attn3.rs#L579)），K/V 拷贝是 `stage`（[`attn3.rs:535`](../crates/amdgpu-stage/src/attention/attn3.rs#L535)）。

**后端看到的单元图**（`amdc weave-report`，节选）：

```text
$ amdc weave-report fmha_fwd_hd128_mxfp8_v3_vs128
unit graph of fmha_fwd_hd128_mxfp8_v3_vs128: 1203 units, 4020 dependences (688 plan units)
     #  unit                          insts    in   out  per pipe (insts / issue cycles)
     1  dma.KD.setup                      7     3     2  Valu 6/24, Salu 1/4
    13  dma.K.0                           1     3     0  Vmem 1/14
    30  v_phase                         332    10    12  Valu 331/1460, Ctrl 1/4
    45  m_phase                           1     0     0  Ctrl 1/4
   ...
  1193  pv.read.h1e2                      4     1     1  Lds 4/24
  1195  pv.mma.h1e1                       1     3     1  Mfma 1/64
  1200  pv.mma.h1e3                       1     3     1  Mfma 1/64
  1201  end_sync                          1     0     0  Ctrl 1/4
dependences: 2389 register, 273 memory, 1358 order
```

计划单元（688）是前端捕获的单元。`glue*` 是单元之间没有被捕获的指令。依赖分三类：寄存器、内存、顺序。

### 2.4 L3：布局与 atom

- **MMA**：`mfma_scale_32x32x64_f8`（[`gfx950.rs`](../crates/amdgpu-isa/src/gfx950.rs) 的 atom 表），K 缩放按字节选择。
- **地址**：每个 LDS / 全局地址、DMA 立即数与 epilogue 存储，都由 `amdgpu-layout` 的 `Layout` 推导；构建期证明连续性与对齐（`partition_lds` 内的 `proofs::check_contiguous`）。
- **扩展阅读**：L3 入门见 [`layout-tour.md`](layout-tour.md)。

### 2.5 后端：pass 流水线与它做了什么

```text
$ amdc passes          (节选，共 63 步，scheduled 路径；取自 ed666ee4)
    4  front       lower   lower-mir                  gfx950-only    always; tokens = policy + ladder rung + Function::lower_opts
   18  scheduled   sched   waits-strip-auto           generic        scheduler
   19  scheduled   mir     bweave-order               generic        always (B3: every phase in today's order)
   24  scheduled   sched   sched-list                 gfx950-param   tokens reorder / rewait / xbar / limit=N (per region)
   25  scheduled   sched   sched-wsink                gfx950-only    token wsink
   26  scheduled   sched   sched-ldahead              gfx950-only    tokens ldsink=D / ldahead32=D
   27  scheduled   sched   waitcnt-recompute          gfx950-only    scheduler
   34  scheduled   ra      regalloc                   gfx950-param   always (scan -> scan-top -> two-phase; regplan-check when the kernel has register marks)
   40  scheduled   post    hazard-valu-sgpr           gfx950-only    always (not switchable)
   62  back        verify  verify-gpu                 gfx950-param   AMDC_GPU_VERIFY (default: MIR register / wait errors fail)
retry ladders
  outer (lowering tokens, from lower-mir; retry on `out of VGPRs`): -, remat1, remat2, remat3, remat4, imm+ldsspill8, ...
```

**后端做了什么。**
1. `waits-strip-auto`（18）先去掉 lowering 按程序顺序插入的等待，这样它们不会钉住后面的合并。
2. `bweave-order`（19）按计划顺序排单元（B3，[`bweave::order`](../crates/amdgpu-codegen/src/bweave.rs#L450)）。对 Free phase，同一个 pass 按交织提示做合并（B4，[`bweave::interleave_with`](../crates/amdgpu-codegen/src/bweave.rs#L317)）；`iter ≠ 0` 的单元作为填充者参与，这就是跨迭代放置。Main `0fb4df13` 上这一步的说明是 “always (B3: authored order; B4: the interleave hints of Free phases, ring-aware)”；上面的输出取自更早的 `ed666ee4`，那时还是旧文字。
3. 列表调度按 kernel 自己声明的选项工作：vs128 用 `ldsink=1,wsink,ldahead32=1`。
4. `waitcnt-recompute`（27）重新推导等待，环槽依赖（B5）也算在内。然后分配寄存器、插入冒险填充。
5. 寄存器不够（out of VGPRs）时：
   - 先走内层阶梯：压力上限 16 / 32 / 48、hoist 上限、launch 预算；
   - 然后是 B6c：只对 Free phase，留下一档不排序的 Free phase；
   - 然后退到不调度的 body；
   - 这些都失败后，外层阶梯才重新 lowering（remat1..4、ldsspill8..64）。

**结果**（`amdc summary` 与 `amdc regplan`，节选）：

```text
$ amdc summary fmha_fwd_hd128_mxfp8_v3_vs128
{"kernel":"fmha_fwd_hd128_mxfp8_v3_vs128","workgroup_size":512,"vgpr":248,"agpr":0,"sgpr":50,"lds_bytes":102400,"blocks":158,
 "instructions":14789,"waits":418,"nops":237,"nop_wait_states":778,
 "regions":[{"name":"kv.stage","mfma":0,"vmem":54,...},{"name":"m_phase","mfma":312,"vmem":0,"lds":928,...},{"name":"v_phase",...,"valu":3972}],
 "certificate":{"errors":0,"warnings":0,"trusted_assumes":4,"trusted_contracts":29}}
$ amdc regplan fmha_fwd_hd128_mxfp8_v3_vs128
fmha_fwd_hd128_mxfp8_v3_vs128: arch VGPRs 248 AGPRs 0
region kv.stage   insts   230  vgpr footprint 244  live dwords 197
region m_phase    insts  1272  vgpr footprint 248  live dwords 223
region v_phase    insts  3972  vgpr footprint 244  live dwords 216
```

### 2.6 门禁

**严格证书：**

```text
$ amdc gpu-verify fmha_fwd_hd128_mxfp8_v3_vs128 --strict
== fmha_fwd_hd128_mxfp8_v3_vs128: 11 finding(s), 0 error(s); proven {"barrier-balanced-ifs": 30, "buffer(hw-guarded)": 54, "global-write-disjoint": 32,
   "lds": 1242, "stage-handoff": 12, "tile-cover": 1242, "tile-reads": 1628160, "wait-paths": 1, ...}
   certificate: 4 trusted assumes (4 assume_trusted, ...), 29 trusted contracts, 0 unproven
$ amdc gpu-verify --shipped --strict | tail -1
certificate (all kernels listed): 21 trusted assumes, 100 trusted contracts, 0 unproven
```

**`port_gate`**（`SKIP_STRESS=1 bash tools/port_gate.sh fmha_fwd_hd128_mxfp8_v3 dd`，GPU 0，节选）：

```text
[BUILD] candidate: 4 kernels emitted+assembled (ref -) -> PASS
[VERIFY] fmha_fwd_hd128_mxfp8_v3_vs128: 0 errors; 4 trusted assumes ..., 29 trusted contracts, 0 unproven (ref 0 errors) -> PASS
[ACC64] vs: VENDOR PARITY mean<=1.10x & bad<=vendor (64 rows): 15/15 (ref 15/15) -> PASS
[ACC64] split: VENDOR PARITY mean<=1.10x & bad<=vendor (64 rows): 13/13 (ref 13/13) -> PASS
[DET] bit-identical 3x (auto): 5/5 (ref 5/5) -> PASS
[PERF] b1h8s2048 auto normal -> fmha_fwd_hd128_mxfp8_v3_kvsplit [KV-split (main+combine)]: 755.27 TF/s (1.034x vendor 730.12; ...)
[PERF] percu64 vs normal -> fmha_fwd_hd128_mxfp8_v3_vs128 [REF_ROWS_PER_BLOCK override]: 710.73 TF/s (0.975x vendor 729.18; ...)
[PERF] b4h8s2048 auto normal -> fmha_fwd_hd128_mxfp8_v3_vs128 [auto: ceil(S/256)*H*B > 64]: 1949.89 TF/s (1.014x vendor 1922.49; ...)
[PERF] b1h8s8192 auto normal -> fmha_fwd_hd128_mxfp8_v3_vs128 [auto: ceil(S/256)*H*B > 64]: 2515.43 TF/s (0.990x vendor 2541.43; ...)
[PERF] b2h8s2048 auto normal -> fmha_fwd_hd128_mxfp8_v3_vs128 [auto: ceil(S/256)*H*B > 64]: 1302.03 TF/s (0.986x vendor 1320.59; ...)
```

**md5**：清单中 `fmha_fwd_hd128_mxfp8_v3_vs128` 为 `ff51fd09`。

---

## 3. 例子 (b)：MLA decode `mla_decode_bf16_p16`

**L1 规则** [`mla_config`](../crates/amdgpu-graph/src/dispatch.rs#L169) 决定两件事：
- **形式**：head-parallel 形式 `mla_decode_bf16_hp_p*`，还是每 wave 16 头的形式 `mla_decode_bf16_p*`。
- **split 数**：取使 CTA ≥ 256 的最小 2 的幂，最多 16。

上下文长度 `--ctx` 是规则的一部分：B ≤ 16 且 ctx ≤ 4096 时走 head-parallel；不知道 ctx 时按长上下文处理。

```text
$ amdc dispatch mla 1 16 --ctx 512
{"hp":true,"splits":16}
$ amdc dispatch mla 1 16 --ctx 8192
{"hp":false,"splits":16}
$ amdc dispatch mla 64 16 --ctx 4096
{"hp":false,"splits":4}
$ amdc dispatch mla 1 16
{"hp":false,"splits":16}
$ amdc compose mla 1 16
{"family":"mla","kernels":["mla_decode_bf16_p64","attn_combine_bf16_hd512"],"splits":16,"tile":null,"rule":"dispatch::mla_config","plan":null}
```

**kernel 声明自己的策略。** [`mla_decode_bf16`](../crates/amdgpu-kernels/src/fmha/mla.rs#L37) 用 `CompileProfile::Gemm`，并在 [`mla.rs:314`](../crates/amdgpu-kernels/src/fmha/mla.rs#L314) 写上 `f.sched_opts = Some("reorder")`，即带重排的列表调度器（pass 24 `sched-list`）。
- 这是 kernel 级声明，后端不按名字特判（`layer_check` 禁止）。
- 测量（[`w6-laggards.md`](w6-laggards.md) §6）：perf_ab mla 4/4 IMPROVED，+7…+23 %，几何 1.139。

```text
$ amdc summary mla_decode_bf16_p16
{"kernel":"mla_decode_bf16_p16","workgroup_size":256,"vgpr":256,"agpr":0,"sgpr":81,"lds_bytes":147456,"blocks":6,"instructions":2071,
 "waits":110,"nops":3,"nop_wait_states":10,"regions":[{"name":"(kernel)","mfma":68,"vmem":89,"lds":136,...}],
 "certificate":{"errors":0,"warnings":0,"trusted_assumes":1,"trusted_contracts":14}}
$ amdc weave-report mla_decode_bf16_p16
unit graph of mla_decode_bf16_p16: 154 units, 511 dependences (7 plan units)
     1  mla.issue                       675    10    37  Valu 561/3564, Lds 36/144, Vmem 37/508, Salu 40/160, Ctrl 1/4
```

MLA 的计划单元只有 7 个（[`w6-laggards.md`](w6-laggards.md) §4 的单元化），所以收益来自列表调度器，而不是计划级的重排。

**SCC 的教训。** 打开 `reorder` 之前，MLA 出现过一次错误编译：描述符伪指令写 SCC 却没有声明，调度器把一个 `s_cmp` 和它的 `s_cselect` 拆开了（[`w6-laggards.md`](w6-laggards.md) §6）。后端修复（伪指令声明 SCC 写入）合入后，才在 MLA 和 decode16 `_c` 上打开调度器。

这个错误能藏住，是因为每个独立检查（调度验证器、冒险 / 等待 pass）读的都是同一张 `op_attrs` 表。修复同时加了守卫 `tests/emit_effects.rs`：逐个比较每种发射展开的隐式 SCC / VCC 效果与表中的行。问题出现在默认调度器（`AMDC_SCHED=1`）下，也就是在 MLA 打开 `reorder` 之前。

启示：**一个 kernel 打开更激进的调度，等于把后端的指令语义描述暴露给更多重排**。所以：
- 每次打开都要过 perf_ab 和正确性标记；
- 指令定义里漏掉的隐式寄存器读写，要先在后端修正。

---

## 4. 例子 (c)：小 M GEMM，经过图 / 组合路径

**L1 图规划。** 一个小 MLP 图：M = 4，rmsnorm → 量化 → up GEMM(fp8 per-token × per-channel, 8192 × 4096) → silu_mul → 量化 → down GEMM(4096 × 4096)。

```text
$ amdc graph plan mlp_small.json        (节选)
{ "launches": [
    { "rule": "rmsnorm+quant_token.fp8", "covers": ["norm", "quant"], "kernels": ["rmsnorm_quant_fp8_n4096"], ... },
    { "rule": "gemm", "covers": ["up"], "kernels": ["gemm_a8w8_rowcol_fp8_64_splitk", "gemm_splitk_reduce_s4_rc"],
      "params": { "kind": "fp8_rowcol", "kernel": "gemm_a8w8_rowcol_fp8_64_splitk", "splits": 4, "K": 4096, "reduce": "gemm_splitk_reduce_s4_rc" },
      "workspaces": [ { "name": "gemm_splitk", "bytes": 16777216 } ],
      "steps": [ { "kernel": "gemm_a8w8_rowcol_fp8_64_splitk", "reads": ["q", "wu", "s", "swu"], "writes": ["gemm_splitk"] },
                 { "kernel": "gemm_splitk_reduce_s4_rc", "reads": ["gemm_splitk", "s", "swu"], "writes": ["u"] } ] },
    { "rule": "act_mul.silu+quant_token.fp8", "covers": ["act", "aquant"], "kernels": ["silu_mul_quant_fp8_d4096"], ... },
    { "rule": "gemm", "covers": ["down"], "kernels": ["gemm_a8w8_rowcol_fp8_64_splitk", "gemm_splitk_reduce_s8_rc"],
      "params": { ..., "splits": 8, "reduce": "gemm_splitk_reduce_s8_rc" }, ... } ],
  "rejected": [], "errors": [] }
```

- 融合规则把 norm + 量化、act + 量化各合成一个 kernel。
- 两个 GEMM 由 [`gemm_kernels`](../crates/amdgpu-graph/src/dispatch.rs#L460) 选择。
- 计划同时给出 workspace（fp32 部分和）以及每一步的读写。

**同一规则的单独查询：**

```text
$ amdc dispatch gemm fp8_rowcol 4 8192 4096
{"kernel":"gemm_a8w8_rowcol_fp8_64_splitk","splits":4,"reduce":"gemm_splitk_reduce_s4_rc"}
$ amdc dispatch gemm fp8_rowcol 4 4096 4096
{"kernel":"gemm_a8w8_rowcol_fp8_64_splitk","splits":8,"reduce":"gemm_splitk_reduce_s8_rc"}
$ amdc dispatch gemm bf16 1 4096 4096
{"kernel":"gemm_a16w16_bf16_64_splitk","splits":8,"reduce":"gemm_splitk_reduce_s8"}
```

**为什么 N = 4096 是 S8。** [`split64`](../crates/amdgpu-graph/src/dispatch.rs#L113) 先用拟合模型在 S ∈ {1, 2, 4} 里选（一轮网格：`c S ≤ CUs`）。然后有一条测量得来的 S8 规则：
- 条件：模型选了 S4，且 M ≤ 16、`c · 8 ≤ CUs`、N 是 1024 的倍数、K 是 2048 的倍数。
- 含义：N = 4096 时 S4 只用了一半 CU，S8 把网格补满。
- 证据：注释里的 `compose_search check-gemm` 扫描（[`compose-gemm-sweeps.md`](compose-gemm-sweeps.md)）。模型本身看不到“填满 CU”这个效应。

N = 8192 时 S4 已经铺满 256 个 CU，所以保持 S4。

**`compose-search` 列出全部候选、模型时间、剪枝和规则的选择（节选）：**

```text
$ amdc compose-search gemm fp8_rowcol 4 4096 4096
{"kind":"fp8_rowcol","m":4,"n":4096,"k":4096,"cus":256,
 "rule":{"kernel":"gemm_a8w8_rowcol_fp8_64_splitk","splits":8,"reduce":"gemm_splitk_reduce_s8_rc"},
 "candidates":[{"kernel":"gemm_a8w8_rowcol_fp8_128","splits":1,"tile":[128,128],"ctas":32,"model_us":12.92,"kept":false},
               {"kernel":"gemm_a8w8_rowcol_fp8_128_splitk","splits":8,"tile":[128,128],"ctas":256,"model_us":6.47,"kept":true},
               {"kernel":"gemm_a8w8_rowcol_fp8_64_splitk","splits":4,"tile":[64,128],"ctas":128,"ring_slots":3,"model_us":7.10,"kept":true}, ...]}
$ amdc compose gemm fp8_rowcol 4 4096 4096
{"family":"gemm","kernels":["gemm_a8w8_rowcol_fp8_64_splitk","gemm_splitk_reduce_s8_rc"],"splits":8,"tile":[64,128],
 "plan":{"threads":256,"waves":4,"phases":16,"units":119,"lds_bytes":80640,
  "rings":[{"name":"A","slots":3,"bytes_per_slot":8448},{"name":"B","slots":3,"bytes_per_slot":16384},{"name":"SC","slots":3,"bytes_per_slot":2048},{"name":"EPI","slots":4,"bytes_per_slot":8704,"owner":"wave"}]}}
```

**kernel 侧：只写真实行的部分和。** `_64_splitk` 是 GEMM v4 结构的 64 × 128 形式：`Cadence::RegStaged`，3 槽环，分 S 份做 K。

它的 split-K epilogue（[`gemm/mod.rs:1003`](../crates/amdgpu-kernels/src/gemm/mod.rs#L1003)）只存含真实行的 16 行块：每块一个 wave 一致的分支。M = 1 时，64 行 tile 只写 1 块而不是 4 块。归约 kernel 本来就只读 < M 的行。

测量见 [`w6-laggards.md`](w6-laggards.md) §7.1：perf_ab gemv 几何 1.060；全行检查 0 bad。

```text
$ amdc summary gemm_a8w8_rowcol_fp8_64_splitk
{"kernel":"gemm_a8w8_rowcol_fp8_64_splitk","workgroup_size":256,"vgpr":16,"agpr":80,"sgpr":37,"lds_bytes":80640,"instructions":516,"waits":22,"nops":13,
 "regions":[{"name":"gemm.ab_stage","vmem":36,...},{"name":"gemm.tile","mfma":20,"lds":36,...}],
 "certificate":{"errors":0,"warnings":0,"trusted_assumes":3,"trusted_contracts":23}}
```

**同一位置还有一个不分派的构件**：kernel 内 last-arriver 归约 [`lastc_epilogue`](../crates/amdgpu-kernels/src/gemm/mod.rs#L1069)（`*_64_splitk_lc`，一次 launch）。它正确且通过验证，但同轮测量只在 M ≤ 4 快 0–5 %，M ≥ 16 慢 6–16 %，所以没有接进分派（[`w6-laggards.md`](w6-laggards.md) §7.2）。

**小 M 的剩余差距**是每次调用约 8.8 µs 的固定部分，不是流带宽（§7.3）。

---

## 5. 加一个新 kernel 的清单

1. **选层与模板。** 先在 L2 找现成模板：`Pipeline` 的 cadence、`AttnBody3`、GEMM `Body`、访存族模板。
   - 能复用就只写 kernel slot（`amdgpu-kernels`）。
   - 新的通用功能块放 L3（`amdgpu-tile` / `amdgpu-layout`，新指令先进 `amdgpu-isa` 的 atom 表并过“属性表 → 双后端 → hazard oracle → 硬件探针”）。
   - 下层不能出现 kernel 名字。
2. **结构。** 有算子图的族先写 `*.graph.json`，`amdc opgraph` 审阅切分，`amdc skeleton <graph.json> <kernel> --out …` 生成 `skeleton_*.rs`，作者只实现 slot。
3. **单元与计划。** 需要后端编排的 phase 用 `Units::capture` 捕获单元，通过 `merge(...).emit_with(b, |w| ...)` 交给 Weaver：
   - 默认 `seq()`；
   - 需要后端交织时 `free()` + `interleave` 提示；
   - 跨迭代的单元标上 `iter`（环默认由 contract 的多槽 tile 推出；`declare_ring` 显式声明会取代推导）。
4. **注册。** 在 [`registry.rs`](../crates/amdgpu-kernels/src/registry.rs) 加名字。会被启动的形状要接进分派规则（`dispatch.rs`，带证据注释），并在 Python 算子层 / 图融合规则里登记。
5. **门禁**（全部要过）：
   - `cargo test`：含 [`layer_check`](../crates/amdc/tests/layer_check.rs)，以及可达 kernel 严格验证 [`verify_reachable`](../crates/amdgpu-kernels/tests/verify_reachable.rs)。
   - `amdc gpu-verify <k> --strict`：0 错误，证书里的信任项要有说明。
   - md5 清单：`amdc md5-manifest --check`；新增或有意的变化用 `--out` 更新，并在提交里写明。
   - 如果 kernel 声明了计划（ring / phase）：`bash tools/regen_compose_facts.sh` 重新生成 `compose_facts.rs`。
   - 正确性：族的主机检查（全行或标记）；`python/tests/run_all.sh` OVERALL PASS。
   - 性能：`bash tools/perf_ab.sh --old <base> --new . --families <族>`，同轮 A/B 没有族变慢；出厂 FMHA 另过 `tools/port_gate.sh`。
   - 改变行为的合并只接受测得的收益。

---

## 6. 默认与 opt-in；测得为负的方向

| 机制 | 默认 | 打开方式 | 现状 |
|---|---|---|---|
| 计划顺序（B3） | **默认**：每个 phase 按作者 / 计划顺序 | — | 所有 kernel |
| Free phase + 交织（B4） | opt-in，按 phase | `Weaver::free()` + `interleave(skeleton, fillers)` | 只在实验形式（ring2、bf16 v3 `pv_early`）上用过；出厂 kernel 没有 |
| 跨迭代放置 / 环槽等待（B5） | opt-in | Free phase 中单元带 `iter`。`RingDecl` 默认在 `FnBuilder::finish` 时由 [`plan_derive`](../crates/amdgpu-dsl/src/plan_derive.rs#L202) 从 contract 的多槽 LDS tile 推出；只有在计划里还没有任何环时才推导（[`plan_derive.rs:203`](../crates/amdgpu-dsl/src/plan_derive.rs#L203)），所以显式的 [`declare_ring`](../crates/amdgpu-stage/src/weave.rs#L197) 会取代推导（Main 上没有 kernel 这样用；ring2 用的是推导出的环）。等待 pass 读环的模型是 [`RingModel::of_function`](../crates/amdgpu-codegen/src/sched/waits.rs#L96) | ring2 实验；`GV-PLAN-RING` 验证 |
| phase 寄存器预算（B6a / B6c） | opt-in | `Phase::budget` 有两个来源：[`Weaver::budget_vgpr`](../crates/amdgpu-stage/src/weave.rs#L40) 在 `emit_phase` 时写进本 phase（[`weave.rs:158`](../crates/amdgpu-stage/src/weave.rs#L158)）；[`reg_cap`](../crates/amdgpu-dsl/src/lib.rs#L2287)（`RegMark::Cap`，区域的硬上限，分配器检查）由 [`plan_derive`](../crates/amdgpu-dsl/src/plan_derive.rs#L108) 并入同名 phase，两者取较小值。只有 Free phase 把预算交给列表调度器作压力上限 | Main 上没有出厂 kernel 调用 `budget_vgpr`；`reg_cap` 只出现在测试里 |
| 寄存器环 / 轮转 | opt-in | `reg_ring`（缓冲两两不相交，Main）；`reg_rotate`（每个余数类共用一个寄存器组）不在 Main，存档在标签 `archive/b6b-reg-rotate` | 实验；ring2r 需要 263 > 256 VGPR |
| 调度选项 | 按 kernel 声明（`sched_opts`） | 如 `ldsink=1,wsink,ldahead32=1`（vs128）、`reorder`（MLA、decode16 `_c`） | 出厂 |
| lowering 选项 | 按 policy / `lower_opts` | `cmpfold`、`dce` 默认开；`movb64`、`jthread`、`coldlayout`、`kabatch` 等按 kernel 打开；`m0reuse` 是 GEMM profile 的默认 | vs128 复刻配置声明 `cmpneg`、`movb64`、`jthread`、`coldlayout`（[`fmha/mod.rs:87`](../crates/amdgpu-kernels/src/fmha/mod.rs#L87)） |

**测得为负、已关闭（原因写在文档里）：**
- **厂商 FMHA 复刻的差异分类**：[`fmha-replica.md`](fmha-replica.md) §5，包括钉住寄存器、厂商 M 阶段顺序、距离 2 拷贝环 + 寄存器轮转（§4.10）。
- **落后族**：[`w6-laggards.md`](w6-laggards.md)：
  - lowering 选项（§1）；
  - hd192 分派变体（§2）；
  - bf16 v3 的 B4 交织与单组流水（§5.2、§5.3）；
  - 更深的 GEMV 权重流（§7.3）；
  - MXFP4（§8 与 [`mxfp4-gemm-profile.md`](mxfp4-gemm-profile.md) 的 W6 节）。

共同的结论：厂商的指令编排需要配套的寄存器计划。只把调度或拷贝距离搬过来，在我们的寄存器布局下不会更快。

**这些实验的代码**保存在 `archive/*` 标签上，例如 `archive/replica-frag-ring-morder`、`archive/gemv-deep-ring4`、`archive/mxfp4-b-direct`。
