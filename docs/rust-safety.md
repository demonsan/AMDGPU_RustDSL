# Rust 安全性：这个 DSL + 编译器设计如何保证

> 对应代码：`crates/amdgpu-{dsl,compiler,safeasm}`、`crates/amdc`；机器化证据：`crates/amdgpu-compiler/tests/rust_safety.rs`。
> 本文件回答一个问题：**在这套 "Rust DSL + 自研编译器" 里，Rust 的 safe 是怎么保证的、保证到什么程度、边界在哪。**

---

## 0. 先明确 "safe" 的三层含义

| 层 | 含义 | 本项目的保证方式 |
|---|---|---|
| **L1 语言级（Rust UB 安全）** | 不会出现 Rust 意义上的未定义行为：无悬垂引用、无数据竞争、无越界、无 `unsafe` | 全部 crate `#![forbid(unsafe_code)]` + 零第三方依赖 + 禁用构造清单 + 安全 Rust 数据结构 |
| **L2 抽象级（类型系统有效性）** | 用错寄存器宽度/漏等 scoreboard/写一半 dword/非法立即数等**在编译期被拒绝** | 把 ISA 规则编码进 Rust 类型：寄存器类、线性 token、借用证明、const generic 断言、typestate |
| **L3 语义级（生成的 GPU 代码正确）** | 生成的指令序列**算得对**、在硬件上不越界/不竞态 | **不属于 Rust 安全性**：由 verifier + MIR 检查 + 硬件探针 + 精度/确定性门禁保证，见 §6 边界 |

本文重点讲 L1/L2（Rust 的安全性），并明确 L3 的分界，避免把 "safe Rust" 误读成 "生成的 kernel 一定正确"。

> **关于 "这是不是假 safe"（`#![forbid(unsafe_code)]` 只是必要条件）：** 逐项的真实机制与真实 rustc 诊断见
> [`rust-safety-by-example.md`](rust-safety-by-example.md) **Part D**。要点：
> (a) lifetime/借用检查**确实在关键规则上工作**（在途矩阵操作数被改写 → `error[E0502]: cannot borrow … as mutable
> because it is also borrowed as immutable`），不是"假 safe"；
> (b) 但**不是全部保证都在编译期**——"忘记 wait/retire"由 `Drop` + `finish()` 以 `Err` 兜底（构建期），等待态组合由
> `verify_block` 兜底（构建期），**DSL 层（`FnBuilder`/`ValueId` 索引 IR）不是 lifetime 管理的**（靠类型化构造 + 编译器
> verifier）；
> (c) 最根本的边界：Rust 的 lifetime 保护的是**宿主机程序的内存安全**，它不知道 GPU 指令的语义——L3 永远不由 Rust 保证。

---

## 1. L1：语言级安全（可编译期强制、可机器检查）

### 1.1 每个 crate 都 `#![forbid(unsafe_code)]`

| crate | 属性位置 |
|---|---|
| `amdgpu-dsl` | `crates/amdgpu-dsl/src/lib.rs` |
| `amdgpu-compiler` | `crates/amdgpu-compiler/src/lib.rs` |
| `amdgpu-safeasm` | `crates/amdgpu-safeasm/src/lib.rs` |
| `amdc` | `crates/amdc/src/main.rs` |

`forbid` 是**编译期硬约束**：任何后续出现的 `unsafe {}`、`unsafe fn`、`unsafe impl`、`unsafe trait` 都会直接编译失败，
而不是靠 code review 拦住。`crates/amdgpu-compiler/tests/rust_safety.rs::every_crate_forbids_unsafe_code`
会检查各 crate root 都声明了该属性（防止有人删掉）。仓库现有 8 个 crate（`amdgpu-dsl`、`amdgpu-layout`、`amdgpu-isa`、`amdgpu-tile`、`amdgpu-kernels`、`amdgpu-compiler`、`amdgpu-safeasm`、`amdc`），全部在 crate root 写了 `#![forbid(unsafe_code)]`，该测试逐个检查全部 8 个。

### 1.2 零第三方依赖

所有 crate 的 `[dependencies]` 只有互为 `path` 依赖（`amdgpu-dsl` → `amdgpu-safeasm`；`amdgpu-isa` → layout；`amdgpu-tile` → dsl + layout + isa；`amdgpu-kernels` → dsl + layout + isa + tile；
`amdgpu-compiler` → dsl + safeasm + tile + kernels；`amdc` → dsl + compiler；`amdgpu-layout` 无依赖）。**没有任何 crates.io 依赖**，因此不存在 "外部 crate 里的 `unsafe` 我们无法审计" 的问题
（`tests/rust_safety.rs::no_third_party_dependencies` 强制）。

### 1.3 无 FFI、无裸指针、无危险构造

Rust 侧**不调用** HIP/ROCm：编译器只做 `DSL → MIR → .s` 的文本生成，`.co` 的加载/启动由 `tests/gpu/*.cpp` 的 C++ host 完成。
所以 Rust 进程里没有设备句柄、没有驱动回调、没有跨语言的指针传递。

机器检查的禁用构造清单（`tests/rust_safety.rs::no_forbidden_constructs_in_any_source_line`，逐行扫描全部 `src/**/*.rs`，
跳过注释）：

```
unsafe  transmute  static mut  as *const  as *mut  from_raw
extern "C"  extern "system"  mem::uninitialized  mem::zeroed
get_unchecked  unreachable_unchecked  asm!(  global_asm!(  std::ptr  core::ptr
```

当前实测：上述构造在源码中 **0 处**（唯一含 "unsafe" 字样的两处是 `amdc` 的诊断字符串 `"UNSAFE TAIL: …"`
和测试里的断言，均非代码）。

### 1.4 并发与全局状态

- 编译器是**单线程、纯函数式流水线**：`Function → MIR → 物理分配 → asm`，没有共享可变状态、没有线程、没有锁。
- 唯一的外部输入是 `std::env` 的编译器开关（`AMDC_*`），在启动时读取，不参与跨调用状态。
- 输出只有 `std::fs::write` 的文本文件与 `std::process::Command` 调用外部 `llc`/`clang`（错误一律 `Result` 上抛）。
- 因此 Rust 侧的 "数据竞争" 在结构上不存在（没有 `Sync` 共享可变对象）；GPU 侧的并发（wave/barrier）不是 Rust 数据竞争。

### 1.5 错误处理模型

- 用户可触发的失败（非法 IR、非法汇编序列、缺少工具链、I/O 错误）一律 `Result<_, String>` / `ExitCode`，
  **不 panic、不 abort、不 UB**；`amdc` 的 panic 面为 **0**。
- 内部的 "不可能分支" 用 `panic!/unreachable!` 或 `expect` 标注（见 §5 统计），语义是 "编译器 bug"，不是未定义行为。

---

## 2. L2：类型层——把 ISA 规则搬进 Rust 类型系统

核心思想：**让 "非法的硬件状态" 在 Rust 里无法表达**，从而用零运行时开销换到编译期保证。主要机制（全部在 safe Rust 内实现）：

### 2.1 寄存器类与宽度

- `Vgpr<N>` / `Sgpr<N>` / `Ready<'a, N>` / `Frag8` / `Frag16`：宽度是类型参数。把 `Vgpr<4>` 传给需要 `Vgpr<2>` 的操作
  （例如 `ds_read_b64_tr_b8` 的转置读）**编译不过**——ISA 的宽度约束变成了类型错误。
- `Vcc<Free>` / `Vcc<Live>`：`Vcc` 被 out-of-line 序列占用时类型改变，重复占用/未释放占用会被借用检查器拒绝。
- 编译器侧的 `Op`/`MOp` 也带 `DType`（V2/V4/V8/V16、f32、i32、Rsrc…），lowering/MIR verifier 双向检查，
  类型不匹配返回 `Err` 而不是生成错误指令。

### 2.2 线性类型 / 所有权 token（防漏等、防重复等、防忘等）

硬件 scoreboard 的等待是**状态机**：load 发出后必须 `s_waitcnt` 才能读结果；MFMA 结果有 20 个等待态；
transcendental 有 4 个等待态。safe-asm 把这些状态编码成**只能移动、不能复制**的 token：

| token | 语义 |
|---|---|
| `Pending<Vgpr<N>>` | 一次已发出但尚未等待的 DS/VMEM 读；只有 `wait_lgkm_cnt(k)` / `.ready()` 能"兑现"它 |
| `Mfma<'a, OP>` | 在途矩阵运算；`retire()` 后才能读目的地；`.chain()` 消费并再产生一个 token |
| `Trans` | 在途 transcendental（4 等待态）；`settle()` 兑现 |
| `Pw<L, H>` | 部分写状态（见 2.4） |

因为 token 是**线性类型**（Rust 的 move 语义 + 无 `Clone`）：
- 漏等 → token 未兑现，`Seq::finish()` 返回 `Err`（编译期/运行期都拦住）；
- 重复等 → token 已被 move，二次使用是编译错误；
- 用未等的值 → 类型不是 `Ready`，API 不接受。

### 2.3 借用证明 `Ready<'a, N>`

`Ready` 是对一个值 "当前可安全读取" 的借用证明：它在生命周期 `'a` 内**借住**该寄存器，其他操作不能覆写它。
矩阵运算 token 持有 `Ready` 输入，因此 "MFMA 在读 A 的同时被 VALU 覆盖" 这类 WAR 在类型层被拒绝。

### 2.4 typestate：部分写

fp8 的 `v_cvt_scalef32_pk_fp8_bf16` 只写 dword 的一半（lo/hi）。safe-asm 用 `Pw<Unset, Unset>` 表示"两半都没写"，
lo 形式要求 low 是 `Unset` 并返回 `Pw<Set, Unset>`，hi 形式同理；只有 `Pw<Set, Set>::done()` 才交还 dword。
**读未写的半边不会编译**（`tests/ui/fail_partial_unwritten_half.rs` 等）。

### 2.5 const generics + 编译期断言（对齐/范围）

DS 偏移是 const generic（`ds_read_b128::<OFF>`），构建期断言：
- 对齐：b128 必须 16B 对齐、`tr_b8` 8B、b32 4B；
- 范围：偏移 ≤ 65535；
- 立即数：12-bit 偏移、E8M0 scale 等非法值直接编译失败。

### 2.6 DSL IR 层的类型化

- 纯值 SSA（`ValueId`）+ 显式循环变量（`VarId`），作用域检查（loop/if 内定义的值不能外泄）。
- 循环变量**单赋值**约束（`var_lint` 诊断；`_t` 的 14 条 wave 级分支警告是已知潜伏项，见 `status.md`）。
- 内存/MTBUF/MFMA 等算子都带 dtype/宽度/描述符类型，lowering 与 MIR verifier 双重检查。

### 2.7 证据：29 个 ui 编译失败用例

`crates/amdgpu-safeasm/tests/ui/` 下有 29 个用例（28 个必须编译失败 + 1 个通过对照），覆盖：
读未等的 `Pending`、重复等待、双写半 dword、非法对齐/偏移、悬垂 `Ready`、token 重复使用、`finish()` 未兑现、
裸 op 白名单外的操作、`dst` 非 owned 等。`cargo test -p amdgpu-compiler --test compile_fail`（`compile_fail.rs` 驱动）。

---

## 3. 运行时验证：类型系统判不了的，用验证器兜底（纵深防御）

safe Rust 只能保证 "程序本身没有 UB"，不能保证 "你描述的指令序列在硬件语义上安全/合法"。因此在类型层之外还有一层验证器，
**两者是互补的**：

| 层 | 位置 | 检查内容 | 失败行为 |
|---|---|---|---|
| safe-asm `verify` | `crates/amdgpu-safeasm/src/verify.rs` | 操作数合法性、2/20/4 等待态规则、DS/VMEM 部分等待计数、WAR/WAW、部分写半边、寄存器所有权、`finish()` 未兑现 | `Result::Err` |
| `forge_for_tests` | safe-asm（`test-support` feature / `cfg(test)`） | 仅测试可构造"伪造"序列，验证器必须拒绝 | 编译期不可达生产代码 |
| IR verifier | `crates/amdgpu-dsl` / compiler `lower.rs` | 类型/作用域/循环变量/算子合法性 | `Err` |
| MIR verifier | `crates/amdgpu-compiler/src/mir.rs` | 物理寄存器、指令形式、描述符、部分写 | `Err` |
| `asm_check::check_lite_contract` | compiler | last-arriver 的 lite 契约（全部访问 sc1、ticket sc0、barrier 前排空） | `Err` |
| `pin_check` | compiler | pinned 段的显式 padding 弱于编译器规则 → 拒绝 | `Err` |
| hazard/wait pass | compiler | 值级 hazard、物理 WAR/WAW、宽存储数据 WAR、vmcnt/lgkmcnt | 插入等待或报错 |

**所有验证失败都是 `Result::Err`，不会 UB、不会生成"看起来对"的汇编。**

---

## 4. 编译期约束 vs 运行期检查的分工

| 约束 | 何时检查 |
|---|---|
| 寄存器宽度/类别、token 所有权、部分写、对齐/偏移/立即数范围 | **编译期**（Rust 类型系统，零开销） |
| 等待态距离、scoreboard 计数、WAR/WAW、部分写跨指令流、契约 | **构建序列时**（safe-asm `finish()` / 编译器 verifier） |
| 生成的汇编编码是否合法 | **外部 `clang` 汇编时**（我们只生成文本，由官方汇编器验证编码） |
| 生成的代码在硬件上是否算对/不越界/不竞态 | **硬件门禁**（精度、确定性、1200/1200 启动压力、探针），见 §6 |

---

## 5. panic 面统计与策略

`tests/rust_safety.rs::panic_surface_report` 每次 `cargo test` 都会打印（数字随代码变化）：

| crate | `.unwrap()` | `.expect(…)` | `panic!/unreachable!/todo!` |
|---|---:|---:|---:|
| `amdgpu-dsl` | 22 | 21 | 1 |
| `amdgpu-compiler` | 35 | 25 | 6 |
| `amdgpu-safeasm` | 0 | 13 | 0 |
| `amdc` | 0 | 0 | 0 |
| **合计** | 57 | 59 | 7 |

策略：
- **用户输入路径**（kernarg/IR/asm/文件/工具链）→ `Result` / `ExitCode`，不 panic；`amdc` 全 0。
- 库内 `unwrap/expect` 集中在**内部不变量**（如已由 verifier 检查过的下标、固定长度的 frag 拆分、常量表查找），
  语义是 "编译器缺陷"；它们是 **crash（可诊断的进程退出）而非 UB**，不会破坏内存安全。
- 没有 `todo!/unimplemented!` 进入默认路径（7 个 panic 类中 `todo/unimplemented` 为 0，均为 `unreachable!` 或内部断言）。

---

## 6. 安全边界（诚实声明）

1. **不保证语义正确性**：safe Rust 与类型层保证 "序列合法、等待正确、不会 UB"，**不保证** 它算出的注意力结果正确。
   数学/精度由参考实现对照（15 形状、CPU/厂商三方对拍、mean err 8.3e-4）与 1200/1200 启动压力验证保证。
2. **验证器的覆盖边界**（已记录在 `docs/safe-asm-design.md`）：
   - 部分写的追踪粒度是 **dword 级**，不是 lane 级视图；
   - `raw-only` 算子（51/102 行）不在 hazard 模型内，靠 op 白名单拒绝进入 safe 窗口；
   - null/越界地址、别名正确性（除显式检查的 WAR/WAW 外）不建模。
3. **GPU 内存安全不是 Rust 属性**：kernel 的越界访问不会被 Rust 编译器发现。我们的对策是设计层（MUBUF 描述符
   `num_records` 越界零填充、行 clamp）、硬件探针（`buffer_probe.sh` 验证越界读为 0 / 越界写被丢弃）、
   以及真实形状的反复启动门禁。**这属于 L3，不宣称由 Rust 保证。**
4. **`.s` 文本由外部汇编器验证**：我们不改 LLVM、不直接编码机器码；非法编码会被 `clang` 拒绝（编译期失败，无 UB）。
5. **未做**：Miri（无 `unsafe` 时代价/收益有限）、fuzzing（编译器输入是内部 DSL，不是外部不可信输入）、
   并发安全证明（单线程结构上已排除）。

---

## 7. 未来若必须引入 `unsafe` 的规则（当前为零，故为硬门槛）

若某天必须写 `unsafe`（例如直接 mmap 设备内存），必须同时满足：

1. 放进**独立 crate**（例如 `amdgpu-ffi`），其余 crate 继续 `#![forbid(unsafe_code)]`，且主流水线**不依赖**它；
2. 每个 `unsafe` 块配 `// SAFETY:` 注释，说明前置条件与不变量；
3. 边界类型必须封装成**安全 API**（不把裸指针/切片泄漏给安全代码）；
4. 该 crate 单独跑 Miri + fuzz（或其他动态检查），并在 CI 里作为独立门禁；
5. 在 `docs/rust-safety.md`（本文件）与 `docs/design.md` 更新安全论证，并新增机器检查（类似 `tests/rust_safety.rs`）。

---

## 8. 证据与回归（全部可复现）

| 检查 | 位置 | 当前结果 |
|---|---|---|
| 每个 crate forbid unsafe | `tests/rust_safety.rs::every_crate_forbids_unsafe_code` | ✅ 8/8 |
| 禁用构造逐行扫描 | `tests/rust_safety.rs::no_forbidden_constructs_in_any_source_line` | ✅ 0 命中 |
| 零第三方依赖 | `tests/rust_safety.rs::no_third_party_dependencies` | ✅ 全 path 依赖 |
| panic 面报告 | `tests/rust_safety.rs::panic_surface_report` | ✅ 每次打印数字 |
| safe-asm crate 无 unsafe + 含 forbid | `tests/safeasm.rs::safeasm_crate_forbids_unsafe_and_contains_none` | ✅ |
| ui 编译失败用例 | `crates/amdgpu-safeasm/tests/ui/` + `compile_fail.rs` | ✅ 29 个（28 fail + 1 pass） |
| 全量测试 | `cargo test` | ✅ **67 个目标 / 314 项** |
| 禁用构造静态 grep | 见 §1.3 清单 | 全部 **0** |

复现：

```bash
bash tools/container/cargo.sh test -p amdgpu-compiler --test rust_safety -- --nocapture   # 4 项 + panic 面数字
bash tools/container/cargo.sh test -p amdgpu-compiler --test safeasm | grep forbids      # safe-asm 无 unsafe
bash tools/container/cargo.sh test -p amdgpu-compiler --test compile_fail                # 29 个 ui 用例
bash tools/container/cargo.sh test                                                       # 67 targets / 314 tests
```
