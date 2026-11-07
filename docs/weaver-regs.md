# 寄存器规划（weaver 第 3 步）：DSL 注解 → 分配器遵守 → 分配后检查

Stage / Weaver 架构里，weaver 决定"哪些值同时活着、谁复用谁的寄存器、每个编织区域最多用多少寄存器"。本层把这些决定写成
**DSL 级注解**（`Function::reg_plan: Vec<RegMark>`），lowering 把它们解析到 vreg（`MFunction::reg_hints`），分配器
（`crates/amdgpu-compiler/src/regalloc.rs` 的 `Plan`）遵守，分配后由 `crates/amdgpu-compiler/src/regplan.rs` 检查并报告。
**没有注解时什么都不变**（49 个 kernel 的 asm md5 逐个相同）。

## API（`FnBuilder`）

| 注解 | 语义 | 强度 |
|---|---|---|
| `pin_vgpr_at(r, base)` | `r`（值或循环变量，整个元组）放在 arch VGPR `base`（按宽度对齐） | 硬：与同时活着的值冲突 → 编译错误 |
| 现有 `pin_agpr / pin_agpr_at / agpr` | AGPR 放置（见 [`agpr.md`](agpr.md)） | 硬 |
| `reg_share(&[r..])` | 生命期不相交时，后定义的成员放进先定义成员的寄存器 | 提示：报告"shared / not shared" |
| `reg_separate(&[r..])` | 成员两两永不共用物理寄存器（生命期不相交也不行；P-in-S 串行化的教训） | 硬 |
| `reg_ring(name, &[r..])` | 一个片段的 N 个寄存器缓冲（双/三缓冲）：两两分离，报告位置 | 硬 |
| `reg_cap(region, vgprs)` / `reg_cap_agpr(region, agprs)` | 区域（及其嵌套区域）内每条指令处，分配后的 arch VGPR 足迹（活着的最高物理 VGPR + 1）/ AGPR 足迹 ≤ 上限 | 硬：超限时按 "out of VGPRs" 处理（调度器先收紧压力重试，最后回退到未调度代码），仍超限则编译错误并列出峰值处活着的值 |
| `weave_region_begin(name)` / `weave_region_end()` / `weave_region(name, f)` | 编织区域标记（`Stmt::RegionMark`，不生成代码，可嵌套，可在 Stage 内）；begin/end 必须在同一结构块（IR 验证器检查） | — |

`RegRef` = `Val(ValueId)` 或 `Var(VarId)`。区域既可以是编织区域，也可以是 Stage（`stage(name, ..)` 的名字）：lowering 给区域内生成的每条
MIR 指令打上区域号（`MInst::region`，对代码生成无影响）。

## 分配器如何遵守

线性扫描与两阶段打包都经过同一个 `Plan`：给某个 vreg 选位置时，禁止（a）其他**时间重叠**的固定放置所在的寄存器，（b）已分配的 separate /
ring 伙伴所在的寄存器；有固定放置的 vreg 只能取它的 base，否则立即报错；share 伙伴的寄存器空闲时优先取。被 copy coalescing 合并掉的值
（生产者直接写进循环变量的寄存器）按合并后的寄存器规划。分配成功后 `regplan::check` 再逐条核对（固定放置、分离、上限），任一不满足即失败。

## 测量：`region_report` / `amdc regplan`

`regplan::region_report(&m, &alloc)` 按区域给出**分配器真值**：指令数、活着的 arch VGPR dword 峰值、VGPR 足迹峰值（真正需要的寄存器文件大小）、
AGPR 足迹峰值、峰值位置以及峰值处活着的值（DSL 名 `%vN` / `%rN` 与所在寄存器，高位优先）。`amdc regplan <kernel> [--specialize ..]` 打印它，
外加计划的执行报告（每个固定放置 honoured / VIOLATED、分离组与缓冲环的实际位置、share 是否共用、无寄存器的目标）。weaver 的单元估计
（RegProfile）与调度器的压力模型都应以它为准。

测试：`crates/amdgpu-compiler/tests/regplan.rs`（固定放置与冲突、share / separate / ring、区域上限与报告、未闭合区域）。
