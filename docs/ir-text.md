# DSL IR 文本（irv1）：打印 → 编辑 → 解析 → 编译

设计见 [`ir-text-roundtrip-design.md`](ir-text-roundtrip-design.md)；本文是格式参考与实验指南。实现：`crates/amdgpu-dsl/src/ir_text.rs`
（打印器、词法、通用 Debug 值解析、结构重建、校验、α 规范化）、`crates/amdgpu-dsl/src/ir_codec.rs`（**生成的**解码器 + id 访问器，
`tools/gen_ir_codec.py`）、`crates/amdc/src/main.rs`（CLI）；测试 `crates/amdgpu-kernels/tests/ir_text.rs`，验收脚本 `tools/ir_roundtrip_check.sh`。

## 1. 工作流

```bash
amdc print-ir fmha_fwd_hd128_mxfp8_v3_t128 --stage dsl > t128.ir     # 完整、可解析
# 编辑 t128.ir（改一个常量 / 换一个 op / 改 sched_opts / 加 reg_plan …）
amdc ir-check t128.ir [--verify]                                     # 解析 + 校验 + 重打印不动点；--verify 加严格证书
amdc emit --from-ir t128.ir --out t128.s [--backend own|llvm-tail] [--specialize ..]
amdc gpu-verify --from-ir t128.ir --strict
amdc jit-info --from-ir t128.ir [--specialize ..]                    # JSON 带 "source":"ir:<fnv>"
amdc print-ir --from-ir t128.ir --stage mir                          # 看改后的 MIR
amdc regplan --from-ir t128.ir                                       # 寄存器区域报告
bash tools/ab_layout.sh fmha fmha_fwd_hd128_mxfp8_v3_t128 t128.ir    # A/B：变体可以是 .ir（也可 x.ir@spec）
```

- `--from-ir` 的 `emit` / `jit-info` **默认先跑严格 GPU 语义验证**（证书打在 stderr，有未证明项 → 退出码 1；`--no-verify` 关闭）。
- kernel 名取文件里的 `dsl "name"`；`profile None` 时编译策略由**名字**推导（`ir-check` 会提示），想改名又保持策略请写显式 `profile Some(..)`。
- Python JIT：`amdgpu_jit._compile.get_kernel(label, specialize, ir_path="t128.ir")`，缓存键含文件文本的 SHA-256。
- `ab_fmha.sh` / `ab_gemm.sh` / `ab_layout.sh` 的变体参数接受 `x.ir` 与 `x.ir@name=v,..`。

## 2. 格式

整个文件是一个 token 流（换行、缩进无意义，`#` 起到行尾为注释）。**字段值一律是 Rust `Debug` 的拼写**（`DType` 写 `I32`、操作数写
`Val(6)` / `I32(4)` / `F32(1.0)` / `Var(VarId(0))`、字符串带引号、`None` / `Some(..)`、列表 `[..]`），所以新增的 IR 字段/变体自动出现在输出里。

```
irv1
dsl "probe_expr_kernarg"
kernarg_size 36  workgroup 64  lds 0  value_count 18
profile None  lower_opts []  sched_opts None
var_tys []
arg Arg { name: "a", ty: Ptr, kernarg_offset: 0, size: 8 }
agpr AgprMark { .. }            # 每个 AGPR 请求一行
reg_plan PinV { .. }            # 寄存器规划（docs/weaver-regs.md），每条一行
region_decl RegionDecl { .. }   # 编织区域的调度策略，每条一行
contract Contract { .. }        # 多行 Debug（{:#?}）
values { %v40: I32 }            # 不由 Let 定义的值的类型（通常为空）
body {
  %v0: Ptr = KernargPtr { offset: 0 }                  # Let：%vN: 类型 = Op
  StoreF32 { base: Val(9), byte_off: 0, value: Val(15) }  # 其他语句：Stmt 的 Debug
  loop vars [VarId(0), VarId(1)] init [I32(0), F32(0.0)] { .. }
  if Val(3) { .. } else { .. }
  region Produce tile "K" slot I32(0) version I32(0) { .. }  # tile / buffer 协议区域
  stage "name" StageContract { .. } { .. }
  RegionMark { name: "w", begin: true }                  # 编织区域标记（无代码）
}
```

结构语句（`Begin*/End*`、`Tile*`/`TileEnd`、`StageBegin/End`、`Else`、`Let`）只能用上面的块语法写，作为普通语句出现会报错。
值 id 与循环变量 id **原样保留**（不重新编号），所以 `parse(print(f))` 与 `f` 逐字段相等、生成相同的汇编；手加新值时用尚未使用的 id 即可。

## 3. 校验

解析时：语法与解码（错误带行号，例如 `line 812: unknown Op variant `MulI3``）、`%vN` 只定义一次、每个 `Let` 的标注类型等于该 op 的
结果类型、所有 id 在范围内；随后跑 IR 验证器（作用域、操作数类型、循环变量、区域标记配对）。`--from-ir` 的编译再加严格 GPU 语义验证（见 §1）。

## 4. 保证与验收

- 文本不动点：`print(parse(print(f))) == print(f)`；`Function` 逐字段相等；α 结构相等（`dsl_structural_eq`，现在对 op 内的值 id 也做 α 改名）。
- 全部已注册 kernel + v3 ×4 + GEMM v4 + MLP ws ×3：从 `.ir` 编译的 asm 与原 kernel 逐字节相同（own 与 llvm-tail），严格证书相同
  （`cargo test -p amdgpu-kernels --test ir_text`，`tools/ir_roundtrip_check.sh`）。
- 编织层新增内容（`RegionMark`、`reg_plan`、`region_decl`、Stage 合同）与 safe-asm 块都往返（测试 `weaver_annotations_stages_and_safe_asm_round_trip`）。
  safe-asm 块经 `AsmBlock::from_parts` 重建，编译器在 lowering 前照常复核。

## 5. 改 IR 类型时

`ir_codec.rs` 是生成的：给 `Op` / `Stmt` / 合同类型加删变体或字段后运行 `python3 tools/gen_ir_codec.py`（`--check` 检查是否过期）。
生成的穷举 `match` 没有通配分支，忘了重新生成会**编译失败**，不会静默丢字段。
