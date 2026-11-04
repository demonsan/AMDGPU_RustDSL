# Rust 安全逐项说明：从 kernel 写法到编译器，再到手写 asm

> 配套文档：`docs/rust-safety.md`（原则与边界）、`docs/safe-asm-design.md`（safe-asm 完整设计）。
> 本文的回答方式是**逐项 + 真实代码**：每一项给出 ①规则 ②仓库里的真实写法（文件路径与节选）③安全由什么保证（编译期/运行期）
> ④违反时的行为。全部代码取自当前 `Main`，可 `cargo test` 复现。

---

## 目录

- **Part A — 编译器层面（写 Rust DSL kernel 时）**：A1–A10
- **Part B — 手写 asm 时（`amdgpu-safeasm`）**：B1–B16
- **Part C — 证据与复现**

---

# Part A — 从 kernel 写法到编译器：Rust safe 如何保证

## A1. crate 级：`#![forbid(unsafe_code)]` + 零第三方依赖

**规则**：整个工作区的 Rust 代码不允许出现任何 `unsafe`；也不允许引入无法审计的外部 crate。

**真实写法**（4 个 crate 的 root 都声明）：

```rust
// crates/amdgpu-dsl/src/lib.rs:15
#![forbid(unsafe_code)]
// crates/amdgpu-compiler/src/lib.rs:3
#![forbid(unsafe_code)]
// crates/amdgpu-safeasm/src/lib.rs  (safe-asm 同样)
#![forbid(unsafe_code)]
// crates/amdc/src/main.rs:3
#![forbid(unsafe_code)]
```

**保证**：编译期。`forbid` 让后续任何 `unsafe {}` / `unsafe fn` / `unsafe impl` 直接编译失败，不依赖 review。
依赖只有 path 依赖（`amdgpu-dsl → amdgpu-safeasm`、`amdgpu-compiler → dsl+safeasm`、`amdc → dsl+compiler`），
没有 crates.io 包，因此不存在"第三方 `unsafe` 无法审计"的问题。

**违反时**：编译失败。机器检查：`crates/amdgpu-compiler/tests/rust_safety.rs`
（`every_crate_forbids_unsafe_code`、`no_third_party_dependencies`）。

---

## A2. kernel 写作本身就是纯安全 Rust：`FnBuilder` API

**规则**：kernel 不是字符串拼接、也不是内联汇编，而是用**类型化 builder** 描述数据流；所有值都是 `ValueId`，
所有操作都是 `Op` 枚举的一个变体。

**真实写法**（`crates/amdgpu-dsl/src/kernels.rs`，`probe_llvm_kernarg` 节选）：

```rust
let mut b = FnBuilder::new("probe_llvm_kernarg", 64);
b.arg("a", DType::Ptr, 0, 8);          // kernarg 布局是声明式的（偏移+大小）
b.arg("n", DType::I32, 16, 4);
let pa = b.let_(Op::KernargPtr { offset: 0 });
let n  = b.let_(Op::KernargI32 { offset: 16 });
let tid = b.let_(Op::WorkitemId);
let off = b.mul_i32(tid, b.c_i32(4));
let qa  = b.ptr_add(pa, off);          // 地址运算也是 IR 节点，不是裸指针算术
let x   = b.load_f32(qa, 0);
let xy  = b.add_f32(x, y);
```

**保证**：语言层。Rust 里没有裸指针、没有 `transmute`、没有 FFI：`Op`/`Stmt`/`Function` 都是普通 `enum`/`struct`
（`Vec`/`Option`/`Box`），builder 方法返回 `ValueId`（`u32`）。**"生成 GPU 代码"这件事只发生在编译器内部的文本发射阶段**，
DSL 层只是构造数据。

---

## A3. 类型化 IR 让"非法状态"不可表达

**规则**：IR 的每个值都有 `DType`（`V2/V4/V8/V16`、`f32`、`i32`、`Rsrc`、`Ptr`、`Frag…`），
`Op`/`Stmt` 的变体决定了合法的操作数组合；`Frag16` 的 lane 访问用 const generic 校验下标。

**真实写法**：

```rust
// crates/amdgpu-dsl/src/lib.rs：frag / lane 访问
let s0 = b.frag16_build(&lanes);            // 16 个 dword 的 fragment
let lane = b.frag16_lane(s0, 3);            // 常量下标，越界是编译错误
// const 检查（crates/amdgpu-safeasm/src/seq.rs:527）
const OK: () = assert!(I < N, "view index out of range for this register tuple");
```

**保证**：编译期（类型 + const 断言）+ 编译器 verifier（dtype/作用域/循环变量）。
非法组合（例如把 `f32` 值给整数操作、在 loop 外使用 loop 内定义的值）在 lowering 或 MIR verifier 返回 `Err`。

---

## A4. 用户输入一律走 `Result`：编译错误 ≠ 未定义行为

**规则**：编译器把"非法程序"当作**普通错误**返回，而不是 panic 或生成错误代码。

**真实写法**：

```rust
// crates/amdgpu-compiler/src/lib.rs:23
pub fn compile(f: &amdgpu_dsl::Function) -> Result<Compiled, String> { … }

// crates/amdc/src/main.rs：CLI 只返回 ExitCode
Ok(()) => ExitCode::SUCCESS,
Err(e) => { eprintln!("{e}"); ExitCode::from(1) }
```

**保证**：运行期错误模型。任何 `Err(String)` 都是可读诊断；`amdc` crate 的 `unwrap/expect/panic` 计数为 **0**。

---

## A5. 验证器而不是 UB：IR / MIR / pin / lite-contract 四道门

**规则**：safe Rust 只能保证"程序本身没有 UB"，不能保证"描述的指令序列在硬件语义上合法"。因此编译器在每个阶段都有 verifier：

| 门 | 位置 | 检查什么 | 失败 |
|---|---|---|---|
| IR verifier | `crates/amdgpu-dsl`（`Function`/`Op` 校验） | dtype、作用域、循环变量单赋值、算子合法性 | `Err` |
| lowering | `crates/amdgpu-compiler/src/lower.rs` | 操作数类别（`as_v4`/`as_v2`）、常量范围、frag 宽度 | `Err` |
| MIR verifier | `crates/amdgpu-compiler/src/mir.rs` | 物理寄存器、指令形式、描述符、部分写 | `Err` |
| `pin_check` | `crates/amdgpu-compiler/src/pin_check.rs:14` | **pinned 段**的显式 padding 是否弱于编译器规则（重跑 value-hazard + wait pass 于副本） | `Vec<String>` |
| `asm_check` | `crates/amdgpu-compiler/src/asm_check.rs:12` | last-arriver lite 契约：共享访问必须 `sc1`、ticket `sc0`、barrier 前排空 | `Result<(), Vec<String>>` |
| hazard pass | `crates/amdgpu-compiler/src/hazards.rs:134,414` | 值级、物理 WAR/WAW、宽存储数据 WAR、`vmcnt`/`lgkmcnt` | 插入等待或报错 |

**真实写法**（`pin_check` 的语义：pinned 序列不许比编译器自己更"随意"）：

```rust
// crates/amdgpu-compiler/src/pin_check.rs:14
pub fn check(m: &MFunction) -> Vec<String> {
    // 在副本上重跑 value-hazard pass 与 wait pass；若它们要在 pinned 指令前插入 nop/wait，即报错
}
```

---

## A6. 无共享可变状态、无并发、确定性

**规则**：编译器是单线程纯函数流水线；外部输入只有启动时读取的 `AMDC_*` 环境变量。

**保证**：结构上排除数据竞争（没有 `Sync` 共享可变对象、没有线程、没有锁）。
因此 Rust 侧的 "数据竞争" 不存在；GPU 侧的 wave/barrier 并发不是 Rust 数据竞争（见 Part C 边界）。

---

## A7. 无 unchecked 索引、无原始指针、无未初始化内存

**规则**：禁止 16 类危险构造：`unsafe`、`transmute`、`static mut`、`as *const`、`as *mut`、`from_raw`、
`extern "C"`、`extern "system"`、`mem::uninitialized`、`mem::zeroed`、`get_unchecked`、`unreachable_unchecked`、
`asm!`、`global_asm!`、`std::ptr`、`core::ptr`。

**机器检查**（逐行扫描全部 `src/**/*.rs`，跳过注释）：

```rust
// crates/amdgpu-compiler/tests/rust_safety.rs::no_forbidden_constructs_in_any_source_line
const FORBIDDEN: &[&str] = &["unsafe", "transmute", "static mut", "as *const", …];
```

**当前结果**：全部 **0 命中**。索引一律 `Vec`/切片的 bounds-checked 访问；`match` 对枚举穷尽。

---

## A8. panic 面：只在内部不变量上，且是 crash 不是 UB

**规则**：用户输入路径不 panic；库内 `unwrap/expect/panic!` 只用于"编译器 bug"级别的内部不变量。

**真实数据**（`cargo test --test rust_safety -- --nocapture` 每次都打印）：

| crate | `.unwrap()` | `.expect(…)` | `panic!/unreachable!` |
|---|---:|---:|---:|
| `amdgpu-dsl` | 22 | 21 | 1 |
| `amdgpu-compiler` | 35 | 25 | 6 |
| `amdgpu-safeasm` | 0 | 13 | 0 |
| `amdc` | 0 | 0 | 0 |

**语义**：panic 是可诊断的进程退出（安全），`forbid(unsafe_code)` 保证它不会退化成 UB。

---

## A9. 编译器的输出只有文本：不接触设备、不生成机器码

**规则**：Rust 侧只做 `DSL → MIR → .s` 文本生成；`.co` 由外部 `clang` 汇编、由 C++ host 加载。

**保证**：没有设备句柄、没有 ioctl、没有共享内存；`.s` 的编码合法性由**官方汇编器**验证。
Rust 程序里唯一的外部动作是 `std::fs::write` 与 `std::process::Command`（错误一律 `Result`）。

---

## A10. 编译器内部的"安全"与"正确"分工

**规则**：编译器保证 **Rust safe**（无 UB）与 **程序合法**（verifier 通过）；它**不**保证生成的 kernel 数学正确。
后者由精度门禁、确定性、1200/1200 启动压力、硬件探针保证（见 Part C）。

---

# Part B — 手写 asm 时：Rust safe 如何保证（`amdgpu-safeasm`）

手写 asm 通常意味着一堆 `unsafe`；本项目的做法是：**把 ISA 规则编码进类型系统**，让"非法的硬件状态"在安全 Rust 里
不可表达，剩下的用验证器兜底。整个 crate 同样 `#![forbid(unsafe_code)]`。

## B1. 入口：安全的 asm 序列构造器

```rust
let mut s = Seq::new();
let mut x = s.splat(3);        // Vgpr<1>，常量 3
let y = s.splat(4);
s.v_add_u32(&mut x, &y, &y);   // 生成 `v_add_u32 x, y, y`
assert!(!s.finish().unwrap().insts().is_empty());
```

`Seq` 是**纯数据结构构造器**（`Vec<AInst>`），不写文件、不调汇编器；`finish()` 校验后返回 `AsmBlock`。

## B2. 寄存器类与宽度是类型

```rust
// crates/amdgpu-safeasm/src/regs.rs
pub struct Vgpr<const N: usize> { … }   // N = dword 数
pub struct Sgpr<const N: usize> { … }
pub struct Ready<'a, const N: usize> { … }

// crates/amdgpu-safeasm/src/ops.rs:164-195
pub fn ds_read_b128<const OFF: u32, D: Dst<4>>(&mut self, dst: D, addr: &impl Readable<1>) -> Pending<D>;
pub fn ds_read_b64_tr_b8<const OFF: u32, D: Dst<2>>(&mut self, dst: D, addr: &impl Readable<1>) -> Pending<D>;
pub fn ds_write_b32<const OFF: u32>(&mut self, addr: &impl Readable<1>, data: &impl Readable<1>);
```

**保证**：编译期。宽传错（例如把 `Vgpr<4>` 给需要 2 dword 的转置读）是类型错误
（ui 用例 `fail_ds_tr_wrong_width.rs`、`fail_wrong_fragment_width.rs`）。

## B3. 偏移/对齐/立即数是 const generic + 编译期断言

```rust
// ds_read_b128 的文档与实现：OFF 必须 16 字节对齐
/// `ds_read_b128 dst, vaddr offset:OFF`: LDS read (lgkmcnt queue). OFF is 16-byte aligned (compile-time).

// 常量总线规则也用 const assert 表达（crates/amdgpu-safeasm/src/ops.rs:17-30）
struct Bus1<A, B>(PhantomData<(A, B)>);
impl<A: Operand, B: Operand> Bus1<A, B> {
    const OK: () = assert!(
        A::BUS + B::BUS <= 1 && A::VOP3_OK && B::VOP3_OK,
        "gfx9 VOP3: at most one SGPR/literal source per instruction (constant bus limit 1) and no 32-bit literal"
    );
}
```

**保证**：编译期（const eval）。ui 用例：`fail_ds_misaligned_offset.rs`、`fail_inline_constant_range.rs`、
`fail_second_compare_while_mask_live.rs`。

## B4. 线性 token：漏等 / 重复等 / 用未等的值 = 编译错误

```rust
// crates/amdgpu-safeasm/src/seq.rs:58 / ops.rs:164
pub struct Pending<T> { … }                 // 已发出、未等待的 DS/VMEM 读，owns 目的地

// ui 用例 fail_use_load_destination_before_wait.rs（真实节选）：
let addr = s.splat(0);
let dst  = s.vgpr::<4>();
let pending = s.ds_read_b128::<0, _>(dst, &addr);  // dst 被移入 token
let _r = s.ready(&dst);                            // ← 编译错误：borrow of moved value: `dst`
let _v = pending.wait(&mut s);                     // 必须先 wait
```

**保证**：编译期（Rust move 语义 + 无 `Clone`）。`Pending` 只能被 `wait`/`ready` 消费一次 → 漏等/重复等在类型层不可能。

## B5. 矩阵运算 token：20 等待态 + 操作数借用

```rust
// crates/amdgpu-safeasm/src/seq.rs:576
pub struct Mfma<'a, const OP: u32> { … }     // 'a 借用输入 Ready，直到 retire

// crates/amdgpu-safeasm/src/examples.rs（pv_rowsum_phase 节选）：
let o = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, rpa, rv0, rsa, rsb).retire(s);
let t = s.v_mfma_scale_f32_32x32x64_f8f6f4(o, rpb, rv1, rsa, rsb);   // 同 opcode 原地链
let _o = t.retire(s);                                                   // 结果只在 retire 后可读
```

**保证**：编译期。`retire` 之前结果不可读（ui：`fail_read_inflight_mfma_d.rs`、`fail_write_operand_of_inflight_mfma.rs`）；
`Ready` 借用让矩阵读期间的操作数不可被覆盖（ui：`fail_matrix_operand_without_ready.rs`）。

## B6. `Trans` token：transcendental 的 4 个等待态

```rust
// crates/amdgpu-safeasm/src/ops.rs:257
pub struct Trans { … }            // v_exp/v_log/v_rcp 的结果，必须 settle()
```

**保证**：编译期 + 序列验证器。`Trans` 不能被当作普通寄存器使用（ui：`fail_trans_token_as_register.rs`）。

## B7. 部分写用 typestate：未写的半边读不出来

```rust
// crates/amdgpu-safeasm/src/partial.rs
pub struct Unset;  pub struct Set;
pub struct Pw<L, H> { … }                      // 哪些半 dword 已写，是类型参数
pub fn v_cvt_scalef32_pk_fp8_bf16_lo<H, A, B>(&mut self, pw: Pw<Unset, H>, …) -> Pw<Set, H>;
pub fn v_cvt_scalef32_pk_fp8_bf16_hi<L, A, B>(&mut self, pw: Pw<L, Unset>, …) -> Pw<L, Set>;
impl Pw<Set, Set> { pub fn done(self) -> Vgpr<1> { … } }   // 只有两半都写完才能取回 dword

// ui 用例 fail_partial_unwritten_half.rs（真实节选）：
let pw = s.partial(d);
let pw = s.v_cvt_scalef32_pk_fp8_bf16_lo(pw, &data, &scale);   // 只写低半
let _r = pw.done();                                            // ← 编译错误：no method named `done`
```

**保证**：编译期。对应真实场景：fp8 转换确实只写半个 dword（`lo`/`hi` 两条指令），typestate 精确建模。

## B8. 输入端口只读

```rust
let x = s.input::<4>();        // In<4>：没有 Dst 实现，不能被 write
// ui 用例 fail_write_input_port.rs：尝试把 In 当目标 → 编译错误
```

**保证**：编译期（`In<N>` 不实现 `Dst`，block 的 `lower_asm` 也会再检查一次）。

## B9. 操作数类别（SGPR/VGPR/常量/字面量）是类型

```rust
// crates/amdgpu-safeasm/src/ops.rs:1-10（模块注释即规则）
// * e32 形式：src0 可为 VGPR/SGPR/inline 常量/literal（Operand）；src1 必须是 VGPR（VgprOnly）。
// * VOP3/e64：不允许 literal；常量总线最多一个标量源（const assert 检查）。
```

**保证**：编译期。ui：`fail_sgpr_in_vgpr_slot.rs`、`fail_two_sgprs_in_vop3.rs`、`fail_literal_in_vop3.rs`、
`fail_vopc_src1_sgpr.rs`、`fail_cndmask_e64_sgpr_data_operand.rs`、`fail_global_store_scalar_data.rs`、
`fail_user_implements_operand.rs`。

## B10. 序列级验证：`verify_block` + `finish()`

```rust
// crates/amdgpu-safeasm/src/verify.rs:45
pub(crate) fn verify_block(b: &AsmBlock) -> Result<(), Vec<String>> { … }
// crates/amdgpu-safeasm/src/seq.rs:499
pub fn finish(self) -> Result<AsmBlock, String> { … }     // 未兑现的 token / 未满足的规则 → Err
```

检查内容：等待态距离（VALU→matrix 2、matrix 结果 20、transcendental 4）、DS/VMEM 部分等待计数、
WAR/WAW、部分写半边、寄存器所有权、op 白名单（`op_allowed`，raw-only 算子被拒）。
**保证**：运行期（构建期），失败为 `Err`。

## B11. 循环体契约：drain-closed 边界

```rust
// crates/amdgpu-safeasm/src/seq.rs:492
pub fn finish_loop_body(mut self) -> Result<AsmBlock, String>
```

规则：作为循环体的序列必须在**边界处完全排空**——没有在途矩阵结果（20 态已 retire）、没有未完成的 DS/VMEM 读、
禁止矩阵 `output_exit`。这样跨迭代的 hazard 不会漏掉，循环体可以安全地只验证"一次迭代"。
（背景：cross-iteration 的窗口无法用直线验证器证明，见 `docs/safe-asm-design.md`。）

## B12. `pin()` + `pin_check`：手写顺序不被调度器打乱，且等待不许比编译器弱

```rust
s.pin();                    // 该序列被调度器视作屏障：不重排、不跨入跨出
// 编译后：
// crates/amdgpu-compiler/src/pin_check.rs:14
//   check(m: &MFunction) -> Vec<String>
// 在副本上重跑 value-hazard pass 与 wait pass；若它们要在 pinned 指令前插入 nop/wait，则报错
```

**保证**：编译期（pin 语义）+ 编译后检查（`pin_check`）。这使手写窗口既保持作者意图，又保证"不会比编译器更不安全"。

## B13. 桥接进 kernel：编译器再验一次

```rust
// DSL 侧：把 AsmBlock 作为一条 Stmt 嵌入
b.var_asm(vars, AsmBlock)         // 或 Op::Asm（单值形式）
// 编译器侧：lower_asm → MIR → 既有 verifier + hazard pass + pin_check 全部再跑一遍
```

**保证**：两道门。safe-asm 自己验证一次（构建时），编译器再验证一次（lowering 后），两者都不通过就 `Err`。

## B14. 硬件验证的参考块

两个真实窗口（`crates/amdgpu-safeasm/src/examples.rs`）：
`cvt_chain_block()`（pk_add/pk_mul/pknorm/svcorr/perm/pk_sub/scalef32 lo+hi/ds_write/ds_read_b64_tr_b8）与
`pv_rowsum_phase()`（scaled MFMA + 4 SMFMAC 行和）。
两者都与 DSL 路径做过 50 组数据集的**逐位相同**验证与最终 asm 的**指令序列一致**验证（`tools/cvtchain_probe.sh`、`probe_cvtchain_*`）。

## B15. 伪造路径只在测试可达

```rust
// forge_for_tests 仅 `cfg(test)` / `test-support` feature
// cargo test -p amdgpu-compiler --features amdgpu-safeasm/test-support
```

生产构建中无法构造"绕过验证器"的序列；测试里伪造的序列必须被 `verify_block` 拒绝（`tests/ui/fail_*` 的机制）。

## B16. 真机验证的窗口实现（可选）

gen5 阶段用 safe-asm 手写了两个真实窗口并落地验证：
Block C（PV+行和，`winc`，−2.5%）与 Block M（整个 M 相，`winm`，中性 +0.15–0.5%），两者都 **bit-identical**、通过全部门禁。
它们证明了"手写窗口 + 类型/验证器保证"在真实 kernel 上可用（性能结论另见 `docs/archive/gen5-window-spec.md`）。

---

# Part C — 证据、边界与复现

## C1. 机器化证据（`cargo test`）

| 检查 | 位置 | 当前结果 |
|---|---|---|
| 各 crate 都 forbid unsafe | `tests/rust_safety.rs::every_crate_forbids_unsafe_code` | ✅ 8/8 |
| 16 类危险构造逐行扫描 | `tests/rust_safety.rs::no_forbidden_constructs_in_any_source_line` | ✅ 0 命中 |
| 零第三方依赖 | `tests/rust_safety.rs::no_third_party_dependencies` | ✅ 全 path |
| panic 面报告 | `tests/rust_safety.rs::panic_surface_report` | ✅ 每次打印 |
| safe-asm crate 无 unsafe | `tests/safeasm.rs::safeasm_crate_forbids_unsafe_and_contains_none` | ✅ |
| 编译期拒绝的 29 个用例 | `crates/amdgpu-safeasm/tests/ui/`（`compile_fail.rs` 驱动，`// error-contains:`） | ✅ 28 fail + 1 pass |
| 全量测试 | `cargo test` | ✅ **67 个目标 / 314 项** |

## C2. 29 个 ui 用例（编译期拒绝的"负例目录"）

共 29 个文件 = **28 个必须编译失败 + 1 个通过对照**（`compile_fail.rs` 驱动，每个负例用 `// error-contains:` 标注期望错误类别）：

```
fail_chain_after_zero                       fail_cndmask_e64_sgpr_data_operand       fail_double_wait
fail_ds_misaligned_offset                   fail_ds_tr_wrong_width                   fail_global_store_scalar_data
fail_inline_constant_range                  fail_lane_out_of_range                   fail_literal_in_vop3
fail_matrix_operand_without_ready           fail_mfma_c_dst_not_owned                fail_mfma_c_still_in_flight
fail_partial_half_twice                     fail_partial_unwritten_half              fail_pending_as_operand
fail_readfirstlane_without_ready            fail_read_inflight_mfma_d                fail_second_compare_while_mask_live
fail_sgpr_in_vgpr_slot                      fail_trans_token_as_register             fail_two_sgprs_in_vop3
fail_unsupported_register_width             fail_user_implements_operand             fail_use_load_destination_before_wait
fail_vopc_src1_sgpr                         fail_write_input_port                    fail_write_operand_of_inflight_mfma
fail_wrong_fragment_width                   pass_sequence（通过对照）
```

## C3. 诚实边界（Rust 安全 ≠ 一切安全）

1. **不保证语义正确**：类型系统与验证器保证"序列合法、等待正确、无 UB"，**不保证**算得对（数学/精度由 15 形状对拍、
   1200/1200 启动压力、硬件探针保证）。
2. **验证器覆盖限制**（记录在 `docs/safe-asm-design.md`）：部分写追踪到 dword 级而非 lane 级；51/102 raw-only 算子
   不在 hazard 模型内（靠 op 白名单拒绝进入 safe 窗口）；越界地址/别名正确性不建模。
3. **GPU 内存安全不是 Rust 属性**：靠 MUBUF `num_records` 越界零填充、行 clamp、硬件探针（`buffer_probe.sh`）。
4. **`.s` 由外部 clang 验证编码**：我们只生成文本。
5. **未做**：Miri（无 unsafe 时代价/收益有限）、fuzz（输入是内部 DSL，不是不可信外部输入）。

## C4. 复现命令

```bash
bash tools/container/cargo.sh test -p amdgpu-compiler --test rust_safety -- --nocapture   # 4 项 + panic 面
bash tools/container/cargo.sh test -p amdgpu-compiler --test safeasm | grep forbids      # safe-asm 无 unsafe
bash tools/container/cargo.sh test -p amdgpu-compiler --test compile_fail                # 29 个 ui 用例
bash tools/container/cargo.sh test                                                       # 67 targets / 314 tests
```

---

## 一句话总结

**Part A**：kernel 是纯安全 Rust 的**类型化数据流**（`FnBuilder`/`Op`/`Stmt`），编译器把非法程序变成 `Err` 而不是 UB，
全部 8 个 crate 用 `#![forbid(unsafe_code)]` + 零第三方依赖 + 16 类构造扫描把"不安全代码"挡在编译期之外。

**Part B**：手写 asm 不是 `unsafe`，而是**把 ISA 规则编码进类型**——寄存器宽度是类型参数、等待是线性 token、
矩阵结果必须 `retire`、部分写是 typestate、常量总线是 const assert、偏移对齐是 const generic；类型系统判不了的由
`verify_block`/`finish`/`pin_check` 以 `Err` 兜底。29 个编译失败用例是这套规则的负例目录。

---

# Part D — 澄清："这是不是假 safe？"

> 这一节回应一个尖锐但正确的质疑：**Rust 的 safe 是靠编译期生命周期/借用检查管理的，而前面是不是只是"让 `unsafe` 不报错"？**

## D1. 先修正一个措辞

`#![forbid(unsafe_code)]` **不是"让 `unsafe` 不报错"**——它让 `unsafe` **无法存在**：任何 `unsafe {}`/`unsafe fn`/
`unsafe impl` 都会让整个 crate **编译失败**。它的作用是证明"类型系统/借用检查器的规则对全部代码都生效、没有被绕过"。
这是 Rust 安全性的**必要条件**，不是充分条件。真正的保证来自**检查器本身做了什么**——所以问题应该是：
**lifetime/借用检查在我们这里到底在哪些地方干活？** 下面用真实 rustc 诊断回答。

## D2. 机制分工表（每类保证的真实来源与检查时机）

| 保证（我们声称的） | Rust 机制 | 检查时机 | 违反时的真实诊断 |
|---|---|---|---|
| 在途矩阵运算期间**操作数不被改写**（WAR） | **借用 + 生命周期**：`Ready<'a,N>` 由 `&'a` 共享借用产生，`Mfma<'a,OP>` 持有到 `retire()` | **编译期** | `error[E0502]: cannot borrow 'a' as mutable because it is also borrowed as immutable`（`fail_write_operand_of_inflight_mfma`） |
| 矩阵结果**retire 前不可读** | **所有权（move）**：目的地 `Vgpr<16>` 被 move 进 `Mfma` token | **编译期** | `error[E0382]: borrow of moved value: 'o'`（`fail_read_inflight_mfma_d`） |
| load 目的寄存器**wait 前不可读、不可重复 wait** | **所有权（move）**：目的地 move 进 `Pending<T>` | **编译期** | `error[E0382]: borrow of moved value: 'dst'` / `use of moved value: 'pending'` |
| 矩阵操作数必须是 `Ready`（不能拿裸寄存器喂矩阵） | 类型 + 生命周期 | **编译期** | `expected 'Ready<'_, 8>', found '&Vgpr<8>'`（`fail_matrix_operand_without_ready`） |
| `vcc` 掩码**未消费前不能再比较** | typestate：`Vcc<Live>` → `Vcc<Free>` | **编译期** | `expected 'Vcc<Free>', found 'Vcc<Live>'`（`fail_second_compare_while_mask_live`） |
| 部分写**未写的半边不可读** | typestate：`Pw<Set,Unset>` 没有 `done()` | **编译期** | `no method named 'done' found for struct 'Pw<Set, Unset>'`（`fail_partial_unwritten_half`） |
| SGPR 不能进 VGPR-only 槽 / 输入端口只读 / 用户不能伪造操作数 | trait 约束（sealed） | **编译期** | `the trait bound '&Sgpr<1>: VgprOnly' is not satisfied` 等（E0277） |
| 对齐/常量总线/立即数范围/lane 下标 | **const eval 断言** | **编译期** | `error[E0080]: evaluation of 'Bus1::<…>::OK' failed … "constant bus limit 1"`（E0080） |
| **忘记 wait / 忘记 retire** | `Drop` 记录 poison，`Seq::finish()` 汇总 | **构建期（运行期）** | `finish() -> Err("… abandoned without waiting")` |
| 等待态距离、scoreboard 计数、WAR/WAW 组合、资源上限 | 序列验证器 `verify_block` | **构建期（运行期）** | `Err(Vec<String>)` |
| DSL 层 IR 合法（dtype/作用域/循环变量） | 类型化构造 + 编译器 verifier | **编译期 + 编译期（lowering/MIR）** | `Err(String)`（编译失败） |
| kernel 数值正确 / 不越界 | **不是 Rust**：硬件门禁 | 测试期 | 精度/确定性/1200 启动压力 |

**结论**：lifetime/借用检查**确实在关键规则上工作**（WAR、在途操作数），不是"假 safe"；但**并非全部保证都在编译期**，
也并非都来自 lifetime。诚实的分工是上面这张表。

## D3. lifetime 的真实工作点（源码 + 诊断）

```rust
// crates/amdgpu-safeasm/src/regs.rs
pub struct Ready<'a, const N: usize> {   // 读权限的"借用证明"
    pub(crate) id: VReg,
    pub(crate) _p: PhantomData<&'a ()>,  // ← 绑定到产生它的 &'a 借用
}
// crates/amdgpu-safeasm/src/seq.rs
pub fn ready<'a, const N: usize>(&mut self, r: &'a impl Readable<N>) -> Ready<'a, N> { … }

// crates/amdgpu-safeasm/src/seq.rs
pub struct Mfma<'a, const OP: u32> {
    …
    pub(crate) _p: PhantomData<&'a ()>,  // ← 持有输入的 Ready，直到 retire
}
// crates/amdgpu-safeasm/src/ops.rs
pub fn v_mfma_scale_f32_32x32x64_f8f6f4_zero<'a>(
    &mut self, acc: Vgpr<16>,
    a: Ready<'a, 8>, b: Ready<'a, 8>, sa: Ready<'a, 1>, sb: Ready<'a, 1>,
) -> Mfma<'a, OP_MFMA_SCALE>
```

只要 `Mfma<'a, …>` 还活着，`a/b/sa/sb` 的 `&'a` 共享借用就还在，**任何 `&mut a` 都会被借用检查器拒绝**：

```rust
// crates/amdgpu-safeasm/tests/ui/fail_write_operand_of_inflight_mfma.rs
let (ra, rb, rsa) = (s.ready(&a), s.ready(&b), s.ready(&sa));
let t = s.v_mfma_scale_f32_32x32x64_f8f6f4_zero(o, ra, rb, rsa, rsa);
let _w = s.lane_mut::<8, 0>(&mut a);   // ← E0502：cannot borrow `a` as mutable …
let _o = t.retire(&mut s);
```

**这就是你理解的"编译期生命周期管理"，它在这里真实存在并且真的拦住了 WAR。** 类似地还有 `Ready` 作为矩阵操作数的强制
（`fail_matrix_operand_without_ready`）、`Vcc<Live>` 的掩码生命周期（`fail_second_compare_while_mask_live`）。

## D4. 编译期抓不到的三类（必须诚实）

1. **"忘记"而不是"用错"**：move 语义只能防"重复使用/提前使用"；**忘了调用 `wait`/`retire`** 时，token 会被 `Drop`，
   `Drop` 把问题写进 `poison`，由 `Seq::finish()` 返回 `Err` —— 这是**构建期**检查，不是编译期
   （Rust 没有线性类型，无法在编译期禁止"泄漏 token"；`#[must_use]` 只是警告）。
2. **硬件规则的组合**：等待态距离（`lgkmcnt`/`vmcnt` 计数）、WAR/WAW 组合、DS 部分计数、资源上限——这些需要看到
   **整条序列**，由 `verify_block` 在构建期检查并返回 `Err`。
3. **DSL 层不是 lifetime 管理**：`FnBuilder`/`Op`/`Stmt` 是 **SSA 索引 IR**（`ValueId` 是 `u32` 下标，不是借用）。
   DSL 的安全性来自"类型化构造 + 编译器 verifier"，**不是**借用检查器管寄存器生命周期。说"lifetime 管理 DSL 寄存器"
   是不准确的——这里只有"安全的宿主机代码 + 编译器拒绝非法 IR"。

## D5. 最根本的边界：host 安全 ≠ target 安全

Rust 的 lifetime/借用检查保护的是**宿主机程序的内存安全**（无悬垂、无数据竞争、无越界）。它**对 GPU 指令的语义一无所知**：
即使整条序列 100% 通过借用检查与验证器，它仍可能算错数。因此本项目的 "safe" 必须分三层读：

| 层 | 保证 | 机制 | 本文档哪节 |
|---|---|---|---|
| L1 | 宿主机程序无 UB | `forbid(unsafe_code)` + 借用检查 + move + 类型 | 本文 Part A/B、`rust-safety.md` §1 |
| L2 | 生成的 GPU 序列**合法**（按我们的硬件模型） | 类型 + const-eval + verifier（构建期 `Err`） | Part B §B10、`rust-safety.md` §3 |
| L3 | 生成的 kernel **算得对**、不越界 | **不是 Rust**：精度/确定性/1200 启动压力/硬件探针 | `rust-safety.md` §6 |

**所以：L1 是"真 safe"（lifetime 在关键处真的工作），L2 是"模型内验证"，L3 不是 Rust 能保证的。** 任何把 L3 也说成
"Rust safe" 的表述都是过度声明——本文档与 `rust-safety.md` 已明确区分。

## D6. 结论

- **不是假 safe**：不安全代码在编译期被禁止；lifetime/借用检查在"在途操作数被改写"这类关键规则上**真实拦截**
  （E0502 诊断可复现，28 个负例覆盖）。
- **但也不是"全编译期"**：忘记 wait、等待态组合、DSL 的索引 IR 由构建期 verifier 以 `Err` 兜底；这一点之前的文档
  表述不够精确，本节修正。
- **更不是"kernel 因此正确"**：Rust 保证的是宿主机程序的 memory safety，不是 GPU 语义。