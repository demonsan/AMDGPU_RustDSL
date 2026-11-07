# DSL IR 文本往返（print → 编辑 → parse → 编译）设计

> 状态：**已实现 P0–P3**（2026-11，格式参考与用法见 [`ir-text.md`](ir-text.md)；实现与本稿的差异：Op 内的值引用保持 `Val(N)` 拼写，α 规范化改为在结构上改名）。目标：让智能体（和人）不改 Rust 就能做 kernel 实验——`amdc print-ir --stage dsl` 打出文本，手改，
> `amdc emit --from-ir` 编译；文本与 `Function` 双向无损。零第三方依赖，`#![forbid(unsafe_code)]`。MIR 文本不在本期范围（§7）。

---

## 1. 现状：打印器离"可解析"还差什么

打印器在 `crates/amdgpu-dsl/src/print.rs`（`print_dsl`）。它是给人看的：头部、参数、合同、按区域树缩进的语句。大部分 Op/Stmt 行直接用
Rust 的 `Debug` 拼写（再把 `ValueId(..)`/`VarId(..)` 换成 `%v`/`%r`），所以**字段是齐全的**，但有以下缺口，不补齐就无法无损解析：

| # | 缺口 | 现状 | 后果 |
|---|---|---|---|
| G1 | 值类型不全 | 只有 `Let` 行带类型（`%v12: f32 = …`）；`Function::value_tys` 按 id 稠密存储，非 `Let` 定义的 id（若有）不出现 | 无法重建 `value_tys` |
| G2 | 循环变量类型 | `loop vars: [...], init: [...]` 不带 `var_tys` | 无法重建 `var_tys` |
| G3 | AGPR 放置请求 | `Function::agpr`（`AgprMark`）完全不打印 | GEMM v4 等 kernel 丢 AGPR pin |
| G4 | 合同 tile 行不全 | `tile NAME [private/global] base= slot= slots=` 是手写子集 | `TileDecl` 其余字段丢失 |
| G5 | `assume` 丢理由 | `assume_trusted%v135 in [lo, hi]`：缺空格，`trusted` 的理由字符串丢失 | 证书里的信任理由丢失 |
| G6 | Op 内的值引用拼写不一致 | `ValueId` 是 `u32` 别名，Op 里打成 `Val(6)`，Let 目标打成 `%v6` | 可解析但不统一；α 规范化（`canon_text`）其实没改名 Op 里的值 id |
| G7 | 无版本/格式标记 | 只有 `dsl @name` | 以后改格式无法兼容 |
| G8 | 结构标记是 Debug 拼写 | `BeginIf { cond: Val(285) }` / `else` / `EndIf`、`TileProduce {…}` / `TileEnd`、`loop …` / `EndLoop` | 可解析，但与区域树（`Node::Loop/If/Region/Stage`）的对应要靠配对 |

不缺的：函数名（编译策略在 `compile_profile = None` 时由名字推导，名字必须保留——已打印）、kernarg/workgroup/LDS、参数（声明顺序）、
`compile_profile` / `lower_opts` / `sched_opts`、合同其余部分（`ka_is`、`ka_one_of`、`ka_ranges`、`grid_defs`、`sym_ranges`、`buf_*`、
`last_arrivers`，均为 `Debug` 形式）、所有 Op/Stmt 字段（`Debug` 天然完整，新增变体自动出现在输出里）。

`FnBuilder` 内部的缓存（常量表、LdsPlan 偏移等）不进入 `Function`，编译只看 `Function`，所以**补齐 G1–G7 后文本就是 `Function` 的完整表示**。

## 2. 文本格式（irv1）

### 2.1 决定：保留"Debug 拼写"作为 Op/Stmt 的语法，不另起一套

理由：(a) 打印端零维护——新增 Op/Stmt 变体自动被打印；(b) 现有文本已经是这个样子，智能体/人已在读它；(c) 解析端用**一个通用的
Debug 值解析器**（下文 §3）把任何一行解析成值树，再按变体名分派到解码函数。代价是解码端每个变体要写一个解码分支（§3.3 用宏和覆盖测试兜住）。

### 2.2 语法（EBNF，`#` 起为注释，缩进只为可读，**结构由显式标记决定**）

```
file      := "irv1" NL header body
header    := "dsl" "@" NAME NL
             "kernarg_size=" INT "workgroup=" INT "lds=" INT NL
             "profile=" debug "lower_opts=" debug "sched_opts=" debug NL
             { "arg" NAME ":" DTYPE "@" HEX "+" INT NL }              # 声明顺序
             { "agpr" debug NL }                                       # G3：每个 AgprMark 一行
             "values:" { VAL ":" DTYPE } NL                            # G1：仅列非 Let 定义的值 id（通常为空）
             "contract:" NL { contract_line NL }
contract_line := "tile" debug                                          # G4：完整 TileDecl 的 Debug
               | "tile_aliases=" debug | "tile_trust=" debug
               | "arg_tensor" NAME "alloc=" ext [ "logical=" ext ]
               | "ka_ranges=" debug | "grid_defs=" debug
               | "ka_is" HEX "=" ext | "ka_one_of" HEX "=" debug
               | "sym_ranges=" debug | "buf_regions=" debug | "last_arrivers=" debug | "buf_trusts=" debug
body      := "body:" NL { stmt NL }
stmt      := VAL ":" DTYPE "=" debug                                   # Stmt::Let
           | VAR "<-" opnd                                             # Stmt::SetVar
           | "assume" [ "_trusted" STRING ] opnd "in" "[" ext "," ext "]"    # G5：理由串
           | "loop" "vars:" "[" { VAR ":" DTYPE } "]" "init:" debug      # G2：带类型
           | "EndLoop" | "BeginIf {" … "}" | "else" | "EndIf"
           | "TileProduce {…}" | "TileConsume {…}" | "TileEnd"            # buf_produce / buf_consume 也落在这两个标记上（global tile）
           | "StageBegin {…}" | "StageEnd"
           | debug                                                     # 其他所有 Stmt：Debug 拼写
VAL := "%v" INT     VAR := "%r" INT     # id 原样保留（不重新编号，见 §4）
debug := 通用 Rust Debug 值：Ident | Ident "(" list ")" | Ident "{" field ":" debug, … "}" | "[" list "]" | "(" list ")"
         | INT | HEX | FLOAT | STRING | "true" | "false" | VAL | VAR
ext   := debug（Ext 的 Debug：Const(..) | Ka(..) | Grid(..) | Sym("..") | Add(..) | Sub(..) | Mul(..) | CeilDiv(.., k) | FloorDiv(.., k)）
opnd  := VAL | VAR | INT | FLOAT "f"      # 打印端统一：Op 内的 Val(6) 改打成 %v6（G6），Var(VarId(2)) 改打成 %r2
```

打印端要做的改动（P0，§6）：头部加 `irv1`；补 `agpr`、`values:`、loop 变量类型、完整 `tile` Debug、`assume_trusted "理由"`；Op 内值引用统一成
`%vN`（解析端同时接受 `Val(N)`，兼容旧文本）。f32 用 `{:?}`（Rust 保证最短可往返表示），字符串用 Debug 转义——都能精确往返。

### 2.3 一个片段（irv1）

```
irv1
dsl @probe_expr_kernarg
  kernarg_size=36 workgroup=64 lds=0
  profile=None lower_opts=[] sched_opts=None
  arg a: ptr @0x0 +8
  arg n: i32 @0x10 +4
  values:
  contract:
    arg_tensor a alloc=Const(256)
  body:
    %v0: ptr = KernargPtr { offset: 0 }
    %v2: i32 = KernargI32 { offset: 16 }
    %v6: u32 = WorkitemId
    %v7: i32 = ConstI32(4)
    %v8: i32 = MulI32(%v6, %v7)
    %v9: ptr = PtrAdd { base: %v0, byte_off: %v8 }
    loop vars: [%r0: i32, %r1: f32] init: [I32(0), F32(0.0)]
      …
      BreakIf { cond: %v20 }
    EndLoop
    StoreF32 { base: %v9, byte_off: 0, value: %v15 }
```

## 3. 解析器

### 3.1 分层

1. **词法**：`%vN`、`%rN`、标识符、整数（十进制/`0x`/负号）、浮点（含 `inf`/`NaN` 的 Debug 拼写）、字符串（Debug 转义）、标点。
2. **通用 Debug 值解析** → `IrVal` 树：`Atom(Ident)`、`Tuple(Ident?, Vec)`、`Struct(Ident, Vec<(field, IrVal)>)`、`List`、`Int`、`Float`、`Str`、`Val(u32)`、`Var(u32)`。
   与具体 Op 无关，约 250 行。
3. **解码**：`trait FromIr: Sized { fn from_ir(v: &IrVal) -> Result<Self, IrErr> }`，实现给 `DType`、`Opnd`、`Ext`、`Op`、`Stmt`、合同各结构、
   `AgprMark`、`StageContract`、safe-asm 块（`AsmBlock`，若其 Debug 可往返；否则见 §7）等。
4. **结构重建**：按 `loop/EndLoop`、`BeginIf/else/EndIf`、`TileProduce|TileConsume/TileEnd`、`StageBegin/StageEnd` 配对成 `Vec<Node>`，用新的
   `Function::from_parts(name, args, sizes, contract, body, var_tys, value_tys, agpr, profile, lower_opts, sched_opts)` 构造（`flat` 由 `body` 派生，
   与 `set_body` 同一路径）。

### 3.2 解码分派与"新变体忘了写解码"的防护

- `Op`/`Stmt` 的解码按变体名分派。用一个小 `macro_rules!`（例如 `dec!(Op::AddI32(opnd, opnd))`、`dec!(Op::PtrAdd { base: opnd, byte_off: opnd })`）
  把每个变体写成一行，字段解码器按类型选取。89 个 Op + 50 个 Stmt ≈ 140 行宏调用。
- **防漏**（编译器能强制的部分）：新增 `fn variant_name(&Op) -> &'static str` / `(&Stmt)`，穷举 `match`（新变体不写就编译不过）；
  **测试**：解码表的键集合 == `variant_name` 能产生的名字集合（用一份测试内的"每个变体一个样例实例"清单 + 全部已注册 kernel 实际出现的变体），
  以及 §4 的全 kernel 往返测试。这样新增变体时：打印自动有，解码缺失会让测试失败而不是静默丢字段。
- 远期可选（P4）：把 `Op`/`Stmt` 的定义改为一张声明式宏表，同时生成 enum、`Debug` 无关的打印器与解析器——单一事实来源，但改动面大，本期不做。

### 3.3 错误信息

每个错误带 `行:列` 和上下文（"line 812: unknown op `MulI3`; did you mean `MulI32`?"、"`%v40` used before definition (defined at line 900)"）。
面向智能体：第一条错误即停，信息足以自修复。

## 4. 往返保证

三层，从强到弱，全部做成测试：

1. **文本不动点**：`print(parse(print(f))) == print(f)`（逐字节）。
2. **`Function` 精确相等**：`parse(print(f))` 与 `f` 所有字段相等（`body`、`value_tys`、`var_tys`、`agpr`、合同……），**包括 ValueId/VarId 编号**——
   决定：**解析保留原始 id，不重新编号**。理由：编号进入 lowering 的 vreg 分配顺序等确定性路径，保留编号使"md5 相同"成为平凡推论，
   不必证明编译器对编号不敏感。允许编号有空洞（`value_tys` 按最大 id 补齐，空洞的类型由 `values:` 行给出或标为未用）。
   另提供 α 等价检查（`dsl_structural_eq`；顺带修 G6：让 `canon_text` 也规范化 Op 内的值 id），用于"手改后与原函数结构是否相同"的判断。
3. **验收测试（md5）**：对 `kernel_names()` 全部 kernel + 出厂 v3 四个 + GEMM v4 + MLP ws 三个，`emit(f)` 与 `emit(parse(print(f)))` 的 asm md5 逐个相同
   （own 与 llvm-tail 两个后端各一遍），`gpu_verify --strict` 的证书逐项相同。这是合入门槛。

## 5. 解析时与解析后的校验

**解析时（`parse` 返回 `Err`，不产生 `Function`）**：
- 类型：每个 `Let` 的标注类型 == `Op::result_ty`（已有）推导的类型；操作数类型由现有 `verify` 检查。
- SSA：每个 `%vN` 只定义一次；使用点必须被定义点支配——区域树内即"定义在同一或外层作用域、且在前"（`If` 分支内定义的值不能在 `EndIf` 之后用，
  与现有 DSL 规则一致）；`%rN` 只在其循环内使用（与现有 DSL 作用域规则一致；循环变量的顺序语义见 [`region-ir.md`](region-ir.md)）。
- 结构：`loop/EndLoop`、`BeginIf/else/EndIf`、`TileProduce|TileConsume/TileEnd`、`StageBegin/StageEnd` 严格配对、正确嵌套；`BreakIf` 只在循环内。
- 头部/合同：参数 offset+size 不越过 `kernarg_size`、不重叠；合同里引用的 kernarg offset 必须是参数；tile 名唯一；`ka_is` 等表达式语法正确。

**解析后**：照常跑 `amdgpu_dsl::verify`（IR 验证器）；`--from-ir` 的 `emit`/`jit-info` 默认再跑 `gpu_verify` 并在 stderr 打印严格证书
（与 `--specialize` 相同的约定：有未证明项 → 退出码 1，`--no-verify` 显式关闭）。手改文本最容易破坏的正是协议/边界，所以默认强制。

## 6. CLI 与工具

```
amdc print-ir <kernel> --stage dsl > k.ir          # irv1 文本（P0 起带版本头）
amdc ir-check k.ir                                   # 解析 + 校验 + 往返自检（print(parse) 不动点），打印摘要
amdc emit --from-ir k.ir [--out k.s] [--backend ..] [--specialize ..]
amdc gpu-verify --from-ir k.ir [--strict] [-v]
amdc jit-info  --from-ir k.ir [--specialize ..]      # JSON 里 "source": {"ir_sha": ...}，纳入缓存键
amdc print-ir --from-ir k.ir --stage mir|mir-ra      # 看手改后的 MIR
```

- `--from-ir` 时 kernel 名取文件里的 `dsl @name`（决定编译策略），`--name NEW` 可改名（改名会改变按名推导的策略：文件里显式写 `profile=Some(..)` 才稳定，`ir-check` 对 `profile=None` 给出提示）。
- `tools/ab_layout.sh` / `ab_fmha.sh` / `ab_gemm.sh`：变体参数接受 `path/to/k.ir`（与现有 `kernel@spec` 并列，可组合 `k.ir@spec`）。
- Python JIT（kernel-opt 的包）：`compile(ir_path=...)` 走 `amdc emit --from-ir`，缓存键加 IR 文本的 sha。
- 典型实验循环（智能体）：`print-ir` → 改一行（换 Op、调常量、删一条 wait、改 `sched_opts`）→ `ir-check` → `emit --from-ir` → `ab_layout.sh base k.ir`。

## 7. 范围与限制

- **MIR 文本不做**（本期）：MIR 依赖分配器/调度器内部状态（vreg 类、alias、pin、auto wait 标记），往返代价高、实验价值低于 DSL 层；以后若需要，
  `print_mir` 已有，可按同一 Debug-值框架做（P5）。
- **safe-asm 块**（`Op::Asm` / `Stmt::VarAsm{,N}` 内的 `AsmBlock`）：其 `Debug` 是否完整可往返需在 P1 核查；若不完整，P1 先让含 safe-asm 块的 kernel
  在 `ir-check` 报"不支持"，P2 再补 `AsmBlock` 的文本形式（它本身是指令序列 + 端口，结构简单）。
- **不做的事**：不提供从文本到 Rust 构建器代码的反向生成；不保证手改文本的语义正确（那是验证器的事）；不支持宏/包含/参数化（实验需要的话
  用 `--specialize` 或外部脚本生成文本）。
- **文件大小**：v3 `_t128` 约 6 000 行、`_vs128` 约 1 万行——解析需 O(n)，目标 < 100 ms（release）。

## 8. 规模估算与分期

| 阶段 | 内容 | 估算 |
|---|---|---|
| **P0** | 打印端补齐 G1–G7（版本头、`agpr`、`values:`、循环变量类型、完整 tile、assume 理由、Op 内 `%vN`）；`canon_text` 规范化 Op 内值 id；出厂 md5 不变（只改打印） | ~200 行 |
| **P1** | 词法 + 通用 Debug 值解析（~300）、`FromIr` 解码（Op/Stmt 宏表 ~350、Ext/Opnd/DType/合同/StageContract/AgprMark ~300）、结构重建 + `Function::from_parts`（~150）、校验（~250）；测试：全 kernel 文本不动点 + `Function` 相等 + 变体覆盖 | ~1 350 行 + 测试 ~300 |
| **P2** | CLI（`ir-check`、`--from-ir` 于 emit / gpu-verify / jit-info / print-ir，~150）；**验收**：全部已注册 kernel 双后端 md5 相同、严格证书相同；safe-asm 块文本（如需要） | ~250 行 |
| **P3** | 工具与 JIT 接入（ab 脚本接受 `.ir`、Python 包 `ir_path`、缓存键）；文档 docs/ir-text.md（待 P3 新建）（格式参考 + 智能体实验指南） | ~100 行 + 文档 |
| P4（可选） | Op/Stmt 声明式宏表（单一事实来源，生成 enum + 打印 + 解析） | 大，另行评估 |
| P5（以后） | MIR 文本往返 | — |

合计 P0–P3 约 2 000 行 Rust（含测试约 2 300），无新依赖，全部 safe Rust。

## 9. 关键决定（摘要）

1. **语法沿用现有打印器的 Debug 拼写**，打印端零维护；解析端 = 通用 Debug 值解析 + 按变体名的解码表（宏一行一个变体）+ 防漏测试。
2. **补齐打印器是第一步（P0）**：今天的文本不完整（AGPR 请求、值/循环变量类型、tile 字段、assume 理由），不补齐就谈不上往返。
3. **保留原始 ValueId/VarId 编号**：往返得到逐字段相等的 `Function`，md5 相同是平凡推论；α 等价另作工具。
4. **结构由显式配对标记决定，缩进只为可读**：手改时缩进错不致命，配对错立即报错。
5. **`--from-ir` 默认跑严格验证**（与 `--specialize` 一致），手改最容易破坏的协议/边界由 `gpu_verify` 把关。
6. **验收 = 全部注册 kernel 的 asm md5 与证书逐项相同**（双后端），作为合入门槛。
7. **MIR 文本不在本期**；safe-asm 块按 P1 核查结果决定是否单独给文本形式。
