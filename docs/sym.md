# 共享符号分析（`amdgpu-sym`）

> 代码：`crates/amdgpu-sym`（零依赖，`#![forbid(unsafe_code)]`）；设计依据：[`ir-review-tirx-sftensor.md`](ir-review-tirx-sftensor.md) C 项
> （TIRx 的 `arith::Analyzer` + `ConstraintContext`）。

## 1. 以前

三套各自的符号算术：

| 用户 | 表示 | 分析 |
|---|---|---|
| 合同（`amdgpu_dsl::Ext`） | 表达式树 | 验证器私有的 `ext_poly` / `ext_itv` |
| 布局（`amdgpu_layout::Ix`） | 规范化的和式 / 除法 / 取余 / xor / 掩码 / 移位 | `Ix::range`、`Ix::bits`、`BitForm` |
| GPU 语义验证器（`gpu_verify`） | 私有的 `Q` / `Poly` / `Itv` / `Ctx` | 多项式非负判定（移位系数 + 盒子）、候选界区间、可能置位位 |

## 2. 现在

`amdgpu-sym` 是唯一的符号算术实现：

| 模块 | 内容 |
|---|---|
| `Q` | 精确有理数 |
| `Sym` | 合同符号：kernarg 标量（字节偏移）、网格维度、具名符号（`sym_range`；布局的动态符号以 `ix<id>:<name>` 进入） |
| `Poly` | 合同符号上的有理系数多项式（单项式 = 排序的符号多重集） |
| `Ctx`（约束上下文） | 事实：符号下界 / 上界（`lower` / `upper`）、常量（`constant`）、代换（`define`：`ka_is`、`grid_is`）；判定 `nonneg(p)` / `le(a, b)`（移位系数检验 + 盒子检验，都只给出可靠的"是"） |
| `Itv` | 候选界区间（每侧最多 4 个多项式候选；add / scale / mul / join / min / meet / within / norm） |
| `bits` | 位支撑分析：`upto`（[0, h] 的可能置位位）、`add32` / `mul32` / `itv_bits32`（验证器的 i32 可能位规则）、`BitForm`（单符号 GF(2) 线性位场范式及其移位分组） |

各用户：

- **合同**：`Ext::to_poly(&Ctx)` / `Ext::to_itv(&Ctx)`（`amdgpu-dsl`）。
- **布局**：`Ix::bits` 用 `bits::upto`；`BitForm` 从布局 crate 移入 `amdgpu_sym::bits`（布局 crate 重新导出，`Ix::bitform` 仍在布局侧构造它）；
  `Ix::to_poly(&Dyns)` + `Dyns::ctx()`（声明的范围作为事实）+ `Ix::can_prove_le` 在同一个约束上下文上做可靠判定。
  `Ix` 的规范化与 `range` 规则本身不变（它们决定发射的指令，改动会改变 md5）。
- **验证器**：私有算术全部删除；`contract_ctx(f)` 用合同（`ka_range`、`ka_set`、`sym_range`、`ka_is`、`grid_is`）填事实；
  可能置位位用 `bits::{itv_bits32, add32, mul32}`。

## 3. 区域树上的作用域事实

验证器现在递归遍历区域树（`walk` / `walk_loop` / `exec_stmt`；线性化的语句下标照旧用于 `stmt N` 诊断与证书）：

- **循环**：循环变量的区间由不动点给出（初值与每次 `set_var` 的并，5 轮后放宽为无界），`assume` / `assume_trusted` 收窄；
- **分支**：`if a < b`（或其否定、`>=`）在 then 分支里给出 `a <= b.hi - 1`，在 else 分支里给出 `a >= b.lo`——**仅当两个操作数在波内一致**
  （常量、kernarg、workgroup id、`readfirstlane`、一致操作数的整数运算、所有赋值都一致的循环变量，循环变量按余归纳判定）。
  `begin_if` 是波级分支（任一 lane 条件成立即整个 wave 执行），逐 lane 的条件对同一 wave 的其他 lane 不成立，所以不用。
  事实只在分支内有效（退出分支即撤销）；原有候选界排在前面，不会被新界挤出。

## 4. 验证

| 检查 | 结果 |
|---|---|
| `cargo test` | 全绿（新增：`gv_bits`、布局 `can_prove_le`、`sym_branch`：一致条件给出界、`n < 65` 不足以证明、逐 lane 条件不被采信） |
| 严格证书（`--shipped` / `--legacy` / GEMM v3 / v4）与每个 kernel 的 proven 计数 | 与 Main `41a32e5` 逐行相同（32 行） |
| `gpu-verify --all`（101 个 kernel 的错误数与 proven 计数） | 与 Main 逐行相同：分支事实没有削弱任何证明；现有 kernel 没有依赖它的证明，因此也没有新增 |
| 全部 kernel 的汇编 md5 | 与 Main 逐字节相同 |

## 5. 范围说明

- `gpu_protocol` 的协议模拟器仍按线性化下标切片工作（它的语义按"发布事件之间的语句区间"定义），递归化是后续工作；
- `Ix` 的除法 / 取余 / xor 不进入 `Poly`（`to_poly` 返回 `None` 时 `can_prove_le` 退回区间比较）。
