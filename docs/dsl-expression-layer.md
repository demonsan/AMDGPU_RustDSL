# DSL 运算符表达式层（`amdgpu_dsl::expr`）

> 起因（用户评论）：“kernel 近乎手写 asm，地址计算和数学计算不能直接用 Rust 运算符描述吗？IR 为什么不能处理？”
> 结论：**能，而且已经做了**——在**不改 IR、不改 MIR、不改任何 shipped kernel** 的前提下，加一层纯数据的运算符表达式（`crates/amdgpu-dsl/src/expr.rs`）。但它**只**覆盖“地址/整数/朴素浮点”这一类逐点标量表达式；下面第 3 节解释为什么“只靠 IR”不够、哪些东西必须保持显式。

## 1. 数据：现状（`crates/amdgpu-kernels/src/probes.rs`，22,000+ 行）

| builder 调用 | 出现次数 |
|---|---|
| `c_i32` | 1,855 |
| `add_i32` | 1,826 |
| `mul_i32` | 673 |
| `shl_i32` | 432 |
| （以上四类“地址/整数算术”合计） | **4,786** |
| `bitcast_i32` / `bitcast_f32` | 570 / 435 |
| `ptr_add` | 365 |
| `cmp_lt_i32` | 236 |
| `mul_f32` / `add_f32` / `sub_f32` / `c_f32` | 291 / 148 / 92 / 248 |

此前**只有 `Ext`（合同层的符号表达式）实现了 `+ - *`**；真正生成 IR 的整数/浮点值都只能 `b.add_i32(b.mul_i32(a, c4), x)` 一层层手写，常量还要先 `let c4 = b.c_i32(4)`。地址算术因此占了 builder 调用的大头。

## 2. 现在能做什么

**表达式是纯数据**：`I32`、`F32`、`Cond`、`Ptr` 只持有 `Rc` 表达式树，**不持有 `&mut FnBuilder`**——这正是 `a * 4 + b` 在 Rust 里合法的原因（运算符只拼树，不碰 builder）。通过 `FnBuilder::bind_i32/bind_f32/bind_cond/bind_ptr`（或返回叶子的 `let_i32/let_f32/let_ptr`）才降低成 IR：

* **后序、确定性**：先左操作数、再右操作数、再运算；同一次 `bind` 里共享的子树（同一个 `Rc`）只降低一次。
* **1:1 映射到现有 `Op`，不引入新 Op**：

| 表达式 | 生成的 Op |
|---|---|
| `a + b`、`a * b`、`a << b`、`a >> b`（**逻辑右移**）、`a & b`、`a ^ b`、`a.min(b)`（I32） | `AddI32`、`MulI32`、`ShlI32`、`LshrI32`、`AndI32`、`XorI32`、`MinI32` |
| `a + b`、`a - b`、`a * b`、`a.max(b)`（F32） | `AddF32`、`SubF32`、`MulF32`、`MaxF32` |
| `a.fma(b, c)`、`a.exp2()`、`a.log2()`、`a.rcp()`、`a.exp()` | `FmaF32`、`Exp2F32`、`Log2F32`、`RcpF32`、`ExpF32` |
| `a.lt(b)`、`a.ge(b)`（I32）、`a.lt(b)`（F32）、`!c` | `CmpLtI32`、`CmpGeI32`、`CmpLtF32`、`NotI1` |
| `c.select_i32(a, b)`、`c.select_f32(a, b)` | `Select` |
| `ptr + off` | `PtrAdd` |
| `i.bitcast_f32()`、`x.bitcast_i32()` | `BitcastF32`、`BitcastI32` |

* **IR 里没有单个 Op 的运算符，定义为固定组合**（文档化，模 2^32 / IEEE 符号处理正确，`tests/expr_layer.rs` 用主机求值在 13×13 个边界样本上逐一验证）：`a - b` = `a + b * -1`（字面量右值为 `a + (-k)`）、`-a` = `a * -1`、`!a`（按位）= `a ^ -1`、`a | b` = `(a ^ b) + (a & b)`、`max(a,b)`（I32）= `select(a < b, b, a)`、`a > b` = `b < a`、`a <= b` = `b >= a`、`ne` = `min(d, -d) < 0`（`d = a - b`）、`eq` = `!ne`、`-x`（F32）= `x * -1.0`。
* **操作数顺序按书写顺序保留**（`4 * x` 与 `x * 4` 是不同的树）。
* **常量**：字面量在**首次使用处**物化为 `ConstI32/ConstF32`，之后在同一或内层作用域复用（`if`/`else`/循环作用域关闭时清空）；`b.hoist_i32(&[2, 3, 4, …])` 把常量按给定顺序提前物化——手写 kernel 习惯把常量集中写在最前面，**物化顺序决定寄存器分配**，所以要逐字节一致就用 `hoist_*` 对齐。
* **验收（已通过）**：`probe_expr_ids`（`probe_llvm_ids` 的运算符改写）与 `probe_expr_kernarg`（`probe_llvm_kernarg` 的改写）编译出的 `.s`（仅归一化 kernel 符号名）**逐字节相同**，post-RA MIR 与寄存器分配也相同，且 `amdc gpu-verify <k> --strict` 干净；见 `crates/amdgpu-kernels/tests/expr_layer.rs`。

## 3. 不能做什么，以及为什么“只靠 IR”不够

1. **指令选择不是表达式的属性**。`a * b + c` 可以是 `v_mul_f32` + `v_add_f32`，也可以是一条 `v_fma_f32`——**舍入不同**。shipped kernel 的数值是按逐位一致认证的，所以运算符**绝不**自动融合：`a * b + c` 恒为 `MulF32` + `AddF32`，要 fma 必须写 `a.fma(b, c)`。同理没有 `/`（`rcp` 近似 ≠ IEEE 除法），要写 `x * y.rcp()`。
2. **packed `v_pk_*` 的语义**：`PkAddF32/PkMulF32` 作用在 2 元素寄存器对上（`Frag2`），通道布局、`op_sel` 与“哪两个标量打包”是**调度/布局决策**，标量表达式语言无法推断，必须显式。
3. **MFMA fragment / lane 布局**：`Frag8/Frag16` 编码了“哪个 lane 持有哪个元素”；这个映射来自矩阵指令的硬件布局，不是点乘表达式能推出的。矩阵、fragment、转置读（`ds_read_b64_tr_b8`）保持显式。
4. **协议语句不是表达式**：`tile_produce/consume`、`buf_*`、`barrier`、`wait_vm/lds`、`fence_*` 是有副作用、有顺序的**语句**，验证器按事件序列建模；把它们藏进表达式会破坏“可审计的顺序”。
5. **没有表达式级优化器 / CSE**：`bind` 之间不共享（除常量复用）；“写了什么就发射什么”。这是刻意的——任何重写都可能改变 shipped kernel 的指令序列和寄存器分配。
6. 因此“IR 能不能处理”的答案是：IR 已经能表达这些**逐点标量 Op**（所以表达式层不需要改 IR）；缺的只是**书写层**。需要新信息（融合、布局、协议、调度）的地方，IR 同样没有，必须由作者显式给出。

## 4. 设计与示例

手写（`probe_llvm_ids` 的地址/数学部分，22 行，12 个临时变量）：

```rust
let c2 = b.c_i32(2);  let c3 = b.c_i32(3);  let c4 = b.c_i32(4);  let c10 = b.c_i32(10);
let c64 = b.c_i32(64); let c100 = b.c_i32(100); let c1000 = b.c_i32(1000);
let z2 = b.mul_i32(z, c2);   let zy = b.add_i32(z2, y);
let zy3 = b.mul_i32(zy, c3); let lin = b.add_i32(zy3, x);
let blk = b.mul_i32(lin, c64); let idx = b.add_i32(blk, tid);
let z100 = b.mul_i32(z, c100); let y10 = b.mul_i32(y, c10);
let t1 = b.add_i32(z100, y10); let t2 = b.add_i32(t1, x);
let t3 = b.mul_i32(t2, c1000); let val = b.add_i32(t3, tid);
let off = b.mul_i32(idx, c4); let q = b.ptr_add(o, off);
let vf = b.bitcast_f32(val);
```

表达式层（同一份算术，4 行，生成同样的指令）：

```rust
b.hoist_i32(&[2, 3, 4, 10, 64, 100, 1000]);                   // 常量按手写顺序提前物化 => 寄存器分配一致
let idx = b.let_i32(((&z * 2 + &y) * 3 + &x) * 64 + &tid);    // 一行
let val = b.bind_i32(((&z * 100 + &y * 10 + &x) * 1000) + &tid);
let q   = b.bind_ptr(b.ptrv(o) + idx * 4);
let vf  = b.bind_f32(b.i32v(val).bitcast_f32());
```

行数（`kernels.rs` 中的两个 demo，含空行外的代码行）：

| kernel | 手写 | 表达式 | 说明 |
|---|---|---|---|
| `probe_llvm_ids` → `probe_expr_ids` | 33 | 18 | 其中算术/地址部分 22 行 → 4 行；另有 4 行合同 + 1 行 `hoist` 是新增的（不产生代码） |
| `probe_llvm_kernarg` → `probe_expr_kernarg` | 31 | 31 | 算术部分 9 行 → 5 行；新增 3 行 `arg_tensor` 与元组绑定，净持平——这个 kernel 本来就几乎没有地址算术 |

叶子：已绑定的值用 `b.i32v(v)`/`f32v`/`condv`/`ptrv` 变成表达式；`let_i32` 绑定并返回叶子，便于后续表达式复用而不重复降低。

## 5. 采用路线

1. **先做地址算术**（约 4,800 个 `c_i32/add_i32/mul_i32/shl_i32` 调用点）：风险最低，因为只有整数、没有舍入问题。
2. **再做浮点数学**：只在显式区分 `fma`/packed 的地方迁移（`a * b + c` 保持两条指令，除非作者写 `.fma`）。
3. **fragment、矩阵、协议语句保持显式。**
4. **验证流程（每个 kernel 一次迁移一个函数）**：迁移前后 asm md5 相等（`emit … --backend own` 后 `md5sum`）；`cargo test`；`amdc gpu-verify <k> --strict`；shipped md5（`_vs128`/`_t128`/`_kvsplit`/`_combine8h`）不变。常量顺序不同导致 md5 变化时用 `hoist_*` 对齐。

## 6. 诚实的限制

* `>>` 是**逻辑**右移（IR 只有 `LshrI32`），与 Rust 对 `i32` 的算术右移不同；这里按 DSL 语义定义，假设值非负。
* `|`、`-`（非字面量右值）、`max`、`ne`、`eq` 会展开成多条 Op（2–5 条），比对应的“假想单条”更贵；热点路径里要有意识地使用。
* 表达式层不做类型推断：给 `i32v` 传 f32 值，由 DSL 验证器（`verify`）在 `finish` 之后报错（`tests/expr_layer.rs` 有负例）。
* 没有 `/`、没有整数除法/取模、没有 `min` 的 F32 形式（IR 里没有对应 Op）。
* `bind` 的常量复用表只在作用域关闭时整体清空（保守、安全，可能多发射一个常量）。
* 只覆盖逐点标量表达式；循环变量读写（`Opnd::Var`）、条件分支、内存访存仍然是显式语句。