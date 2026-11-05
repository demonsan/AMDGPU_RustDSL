# Region IR（结构化 kernel 体）

> 代码：`crates/amdgpu-dsl/src/lib.rs`（`Node`、`RegionKind`、`Function::{body, stmts, set_body, parse_flat, set_stmts, edit_stmts}`、
> `FnBuilder` 的区域栈与闭包形式）；测试：`crates/amdgpu-compiler/tests/region_ir.rs`。设计依据：[`ir-review-tirx-sftensor.md`](ir-review-tirx-sftensor.md) B 项。

## 1. 表示

kernel 体是一棵树（`Function::body()`）：

```rust
pub enum Node {
    S(Stmt),                                                        // 直线语句（从不是结构标记）
    Loop { vars, init, body },                                      // begin_loop .. end_loop（do-while；BreakIf 在 body 内）
    If { cond, then, els: Option<Vec<Node>> },                      // begin_if .. [else_ ..] end_if
    Region { kind: Produce | Consume, tile, slot, version, body },   // tile_produce / tile_consume / buf_produce / buf_consume .. tile_end
}
```

"开了不关 / 关了没开"在树里无法表示。构建时的不匹配（`end_loop` 碰到打开的 `if`、`tile_end` 没有打开的区域、`tile_produce` 嵌在另一个区域里、
`finish()` 时还有未关闭的区域）在 `FnBuilder` 里立刻 panic，并给出 kernel 名与最内层打开的区域。

迁移前的普查（111 个 kernel：`amdc list` + v3 / dsl / GEMM）：1495 个 tile 标记全部由显式 `tile_end` 关闭，没有一个跨越循环/分支边界，
所以 `Region` 节点与原来的标记语义一一对应；所有 kernel 共 99 个循环、159 个 if、1111 个区域。

## 2. 构建 API

`begin_*/end_*` 不变（它们维护区域栈），另有闭包形式：

```rust
b.loop_(&[(DType::I32, init)], |b, vars| { ...; b.break_if(done); });
b.if_(cond, |b| { ... });
b.if_else(cond, |b| { ... }, |b| { ... });
b.produce_region("K", slot, gen, |b| { ...stores... });
b.consume_region("K", slot, gen, |b| { ...reads... });
```

## 3. 线性视图

`Function::stmts()` 是这棵树的线性化（带 `BeginLoop/EndLoop`、`BeginIf/Else/EndIf`、`TileProduce/TileConsume/TileEnd` 标记），由树派生，因而永远配对。
`parse_flat` 是它的逆（不配对则报错）；故障注入测试用 `edit_stmts` / `set_stmts` 改线性视图，改完重建树（不配对即 panic）。

## 4. 消费者

| 消费者 | 读法 |
|---|---|
| lowering（`lower.rs`） | **遍历树**：`walk` 递归节点，在区域边界调用同一个 `lower_stmt`（标记处理器维护标签 / 快照栈），输出与线性遍历逐字节相同 |
| `gpu_verify`、`gpu_protocol`、DSL `verify` / `lint`、表达式层计数 | 读线性视图（语句下标就是证书和诊断里的 `stmt N`；协议模拟器按下标切片） |

线性视图与树同源，所以这些消费者看到的也是配对的结构；把它们改成直接递归树是后续工作，不改变语义。

## 5. 验证

- `amdc list` 的 101 个 kernel（own 后端）与 v3 / dsl / 旧手写 / GEMM v3 / v4 的汇编 md5 与 Main `1c3c9cf` 逐字节相同；
- 严格证书不变（`--shipped` 24 / 100 / 0，`--legacy`（已移除） 44 / 216 / 0）；
- `cargo test` 69 个目标 / 325 项全绿（新增 `region_ir`：所有 kernel 的线性视图往返一致、闭包形式与 begin/end 生成相同的树、不配对在构建期 panic、不配对的线性编辑被拒绝）。
