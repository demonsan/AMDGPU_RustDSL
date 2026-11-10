# 算子图与骨架生成器（stage 库 P3）

状态：已实现（`crates/amdgpu-graph/src/opgraph/`）。设计见 [`fusion-stage-library-design.md`](fusion-stage-library-design.md) §3 / §5。算子图**不是新 IR**：它只是骨架生成器的输入，不进入编译器。

## 1. 数据模型（可序列化为 JSON，无 serde）

| 对象 | 内容 |
|---|---|
| `Graph` | 名字、属性（`threads`，流水线图还有 `pipeline` / `hooks` / `dispatch`）、节点、边、weave 组 |
| `Node` | stage 实例：原语（M/A/E/R/O/Q/S/P）、注册表键（`amdgpu_tile::stage::registry()`）、参数、资源估计（VGPR/AGPR/LDS/barrier）、roles、barrier 相位、所在阶段（prologue/loop/epilogue）、流水线 hook |
| `Edge` | 张量：dtype / 形状 / 逻辑布局、生产者与消费者的 fragment 元素表（都有时**由布局代数计算**边界类）、作用域（wave/block/grid/none）、存储（VGPR/AGPR/LDS 区域/global）、声明的类、kernarg ABI、是否跨迭代（`carried`，不参与依赖序） |
| `Group` | weave 组：一个 weaver 交织的若干节点（名字、weave 相位标签、成员） |

边界类 C0–C4：两端元素表都在时由 `plan_convert_tables` 算（寄存器改名 → C0，quad 内 → C1，半波 / 跨 lane → C2），声明的类与计算结果不一致即报错；block 作用域的边至少是 C3；被切断的边是 C4。

**切分**（`Cut`）把节点分到各 kernel；`Graph::evaluate_cut` 检查：每个节点恰在一个 kernel、不能留在片上的张量（scope none / grid）必须被切、每个 kernel 的资源估计（取各 stage 的最大占用，stage 依次运行）在预算内。`CutReport::text` 给出评审文本（每个 kernel 的 stage 列表、片上边及类、HBM 张量、估计）。

## 2. 命令行

```
amdc opgraph <graph.json> [--cut k1=n1+n2,k2=n3]
amdc skeleton <graph.json> <kernel> [--nodes n1,n2|all] [--graph-file <path>] [--out <file.rs>]
```

## 3. 骨架的三种形式

生成的是确定性的 Rust 源码（测试会重新生成并与提交的文件逐字比较；`UPDATE_SKELETONS=1 bash tools/container/cargo.sh test -p amdgpu-kernels --test opgraph_<x>` 重写）。每个骨架有：评审头（stage、片上边及类、HBM 张量、估计）、`<Kernel>Slots` trait、`build_<kernel>`（先 `check_kernel`：片上边的类必须与生成时一致，再按图的依赖序调用各槽位）。槽位里放的是图**不**决定的东西：流水节奏、发射顺序、weave 顺序与区域、调度 token、lane / role。

| 形式 | 触发 | 槽位 |
|---|---|---|
| 顺序 | 默认 | `prelude`；prologue 各 stage；`loop_open`、loop 各 stage、`loop_close`；epilogue 各 stage |
| 流水线 | 图属性 `pipeline` | `prelude`、前置 stage、`pipeline`（返回配置好的 `Pipeline`）、图用到的 `PipeBody` hook（`stage` / `m_phase` / ...）、后置 stage；生成的 `<Kernel>Body` 把 `Pipeline` 的回调转发到槽位 |
| 分派的流水线 | 图属性 `dispatch`（如 FMHA 的 `roles + versions`） | `prelude`、`pipeline`、`dispatch(pipe)`（每个分支建 body、`pipe.run`、epilogue；文档列出 hook → stage / weave 组） |

- **weave 组**：成员节点必须在依赖序里连续；骨架给整个组**一个**槽位（名字 = 组名，默认体 = `stage::compose` 成员按声明序），成员的构造函数 `<node>_stage` 是 trait 的必需方法；被切断的组报错。
- **切断的边**：生产 kernel 里生成 `store_<tensor>`（紧跟生产者之后），消费 kernel 里生成 `load_<tensor>`（紧挨第一个消费者之前）；默认体是 `todo!`，提示该用的注册表 `move.*` stage（agent 可覆盖）。

## 4. 验收（P3）

出货 kernel 现在**经由生成的骨架构建**，`amdc emit` 的汇编对全部 41 个 kernel 与 `origin/stages` 逐字节相同：

| kernel | 图 | 骨架 | 形式 |
|---|---|---|---|
| `mlp_fp8_ws_fused` | `mlp/mlp_fp8.graph.json` | `mlp/skeleton_fused.rs` | 顺序（`h` 为 C0；以 `h` 切开得 g1 / g2，`h` 是 C4） |
| `mlp_fp8_ws_g1` / `mlp_fp8_ws_g2` | `mlp/mlp_fp8_unfused.graph.json`（一个图、两个 kernel，`h` 是 C4 切边） | `mlp/skeleton_g1.rs` / `mlp/skeleton_g2.rs` | 顺序（切边槽位 `store_h` / `load_h` 在这里是空操作：H 的存取由 stage 自己做） |
| `rmsnorm_quant_fp8_n<N>` | `norm/rmsnorm.graph.json` | `norm/skeleton_rmsnorm.rs` | 顺序（load → rms → quant → store，5 条片上边全是 C0；同一个骨架服务所有 N） |
| `gemm_mxfp8_bs_v4` | `gemm/gemm_v4.graph.json` | `gemm/skeleton_v4.rs` | 流水线（`RegStaged`） |
| `fmha_fwd_hd128_mxfp8_v3_{vs128, kvsplit, t128}` | `fmha/*.graph.json` | `fmha/skeleton_*.rs` | 分派的流水线（S→P 为 C0，O→epilogue 为 C2） |

测试：`amdgpu-tile/tests/opgraph.rs`（JSON、类、切分、weave 组、切断边槽位、生成确定性）、`amdgpu-kernels/tests/opgraph_{mlp,gemm,fmha}.rs`（图 / 骨架与提交文件一致、边界类、注册表键与原语一致）。

## 5. 已知限制

- `mlp_fp8_ws_g1q_*`（带 bias / scale 张量的 GEMM1 + epilogue 链）仍直接用 g1 的槽位片段构建，它自己的图还没有生成。\n- RoPE + KV-cache 写（设计 §4.5）没有骨架：它需要带索引的 `M`（gather / scatter，P4），库里还没有对应的 stage 与 kernel。
- 流水线 / 分派形式里，weave 组只作文档与 hook 映射，默认的 `compose` 体只在顺序形式里生成。
- 资源估计是 agent 规划用的上界，寄存器分配与 GV 仍是最终裁判。