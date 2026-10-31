# LLVM AMDGPU CodeGen 测试覆盖审计：我们的 Rust DSL + 编译器能构建并正确编译多少？

> 审计对象：ROCm 10.1.0 的 LLVM 源码树 `llvm/test/CodeGen/AMDGPU`（AMD LLVM 24.0.0git）。工具：`tools/llvm_feature_audit.py`；逐测试记录：`tools/llvm_feature_audit.json`（4996 条）；汇总：`tools/llvm_feature_audit_summary.json`。
> **这是静态审计**（解析测试文本并与我们的指令全集比对），没有运行任何测试，也没有把任何 LLVM 测试翻译成 DSL kernel。第 6 节明确说明“用 Rust 构建一个测试”对我们意味着什么、不意味着什么。

## 1. 结论（先给数字）

| 项 | 数量 |
|---|---|
| 测试文件总数 | **4996**（`.ll` 3450 + `.mir` 1546；顶层 3831、`GlobalISel/` 1130、`NextUseAnalysis/` 35）|
| 其中 codegen（`llc` 生成汇编） | 2844（`.ll` 2825，`.mir` 19）|
| MIR pass 测试（`llc -run-pass/-stop-after/...`） | 1807 |
| IR pass 测试（`opt`） | 322 |
| 无 codegen RUN 行（`llvm-mc`/二进制工具/仅 FileCheck 等） | 23 |
| **gfx950 级** codegen 测试（含 gfx950 或 gfx94x/gfx942） | **350**（其中 gfx950 本身 97；其余为 gfx94x）|
| **COVERED** | **16**（全部 gfx950 级 codegen；其中**真正检查了指令的仅 2 个**，另 14 个是“无操作码检查”的空洞覆盖）|
| **PARTIAL** | **68**（缺操作码 65，仅缺特性 3）|
| **OUT-OF-SCOPE** | **4912** |

OUT-OF-SCOPE 的原因分布（每个测试按第一条原因计）：

| 原因 | 测试数 |
|---|---|
| 子目标不是 gfx950 级 | 2494 |
| MIR pass 测试 | 1807 |
| IR pass 测试（opt） | 322 |
| IR/ABI 特性不被 DSL 建模 | 266 |
| 无 codegen RUN 行 | 23 |

**诚实的一句话**：LLVM 的 AMDGPU CodeGen 测试集约 93% 针对其它子目标或其它层次（pass/汇编器/MIR），在 gfx950 级的 350 个 codegen 测试里，266 个（76%）使用了我们的 kernel DSL 不建模的 IR/ABI 特性（绝大多数是**非 kernel 函数**，即普通可调用函数），仅 2 个测试的全部指令检查落在我们的指令全集之内且没有这类特性。我们的 DSL 是“指令/特性级 kernel 描述”，不是 LLVM IR 前端。

## 2. 方法

1. **枚举**：递归遍历测试目录的全部 `*.ll` 与 `*.mir`（无其它扩展名；`README`/`lit.local.cfg` 不计）。
2. **RUN 行解析**：合并以 `\` 续行的 `; RUN:` / `# RUN:`，按管道拆分命令；`llc`（无 `-run-pass/-stop-after/-stop-before/-start-before/-start-after/-passes=`）= codegen，带这些旗标 = mir-pass；`opt` = pass；`llvm-mc` = assembler；`llvm-objdump/readobj/lld` = 二进制工具。文件的 `kind` 取优先级最高者（codegen > mir-pass > pass > assembler > 二进制工具 > other）。同时提取 `-mcpu=`、`-mtriple=`/`-march=`、`-mattr=`、`-global-isel`、`-O0`，以及 `REQUIRES/UNSUPPORTED/XFAIL`。
3. **目标类别**：每条 RUN 行由 `-mcpu`（缺省时由**带版本号的 triple**推出，如 `amdgpu9.42` = gfx942、`amdgpu10.30` = gfx1030、`amdgpu6.00` = gfx600）映射到 `gfx950 / gfx94x / gfx9x / gfx10 / gfx11 / gfx12 / pre-gfx9 / r600 / unknown`。**没有 `-mcpu` 的 `amdgcn` 视为 SI 通用目标（pre-gfx9）。** gfx950 级 = gfx950 ∪ gfx94x（MI300 系，ISA 近亲；单独统计见下）。一个文件可含多条不同目标的 RUN 行，目标类别是集合。
4. **所需操作码**：对每条 CHECK 行（`CHECK/GCN/GFX9/GFX942-SDAG-NEXT/...`，含带连字符的前缀；`-NOT`、`-LABEL` 行不计），**只统计其前缀属于某条 llc→FileCheck 管道的行**，并把操作码记到该管道的目标类别下——因此 gfx950 级的必需指令只来自 gfx950 级 RUN 行的 CHECK。汇编测试取每行第一个 token（须以 `v_/s_/buffer_/global_/ds_/flat_/scratch_/image_/tbuffer_` 开头）；`.mir` 测试取 MIR 指令操作码（`body:` 里的输入指令以及 CHECK 里的 MIR 行；排除寄存器类、FileCheck 变量引用与 `G_*`/`SI_*` 伪指令）。操作码**归一化**：转小写，去掉 `_e32/_e64/_sdwa/_dpp/_vi/_gfx*` 与 MIR 寻址后缀（`_offen/_offset/_saddr/_vgprcd/...`）。
5. **IR/ABI 特性**（对 `.ll` 文本用正则，对 `.mir` 用 MIR 关键字）：调用（非内建/间接）、`invoke/landingpad`（异常）、`alloca`/动态 alloca/`stacksave`、原子（`atomicrmw/cmpxchg/fence/atomic load|store`）、`llvm.amdgcn.*` 内建函数全表、调试信息、可变参数、调用约定、函数属性、kernarg、工作组/工作项 id、LDS（addrspace 3）、barrier/waitcnt 内建、溢出/scratch、`-mattr`、地址空间、类型。
6. **我们的指令全集**：`crates/amdgpu-compiler/src/emit.rs`（逐 `MOp` 分支内的字面量助记符，加上终结符/`s_nop/s_waitcnt/s_barrier` 等）并上 `backend/llvm{20,24}.tbl` 的 MIR 操作码归一化名；共 **95 个助记符**（`MOp` 枚举 114 个变体，两个表各 112 行）。完整清单在 JSON 的 `universe.universe`。
7. **分类规则**（与任务书一致）：
   * **OUT-OF-SCOPE**：非 codegen 种类（pass/mir-pass/assembler/other）；或没有任何 RUN 行落在 gfx950 级；或含**硬不支持特性**：`call_nonintrinsic、indirect_call、exceptions、alloca、dynamic_alloca、stacksave、debug_info、varargs、非 kernel 函数、栈对象、addrspace(5) 私有内存`。
   * **PARTIAL**：gfx950 级 codegen、无硬不支持特性，但缺操作码，或使用只部分建模的特性（原子、fence、f64、超宽整数、scratch 指令）。
   * **COVERED**：gfx950 级 codegen、无硬不支持特性、所需操作码全在我们的全集内。若该测试在 gfx950 级 RUN 行下**没有任何操作码检查**（只检查 ELF 头、诊断信息、元数据等），标记 `vacuous` ——它在文字意义上满足规则但没有证明任何指令支持，本文所有“真正覆盖”的计数都已把它们剔除。

**方法的局限**（务必按此理解数字）：(a) 纯文本启发式，CHECK 行里被正则捕获的“第一个 token”可能是伪指令、宏或自动生成的名字；(b) 不理解 `{{...}}` 正则里藏的助记符，也不做跨目标助记符别名（如 gfx10 的 `v_add_nc_u32`）；(c) gfx94x 与 gfx950 共用一类，仅 ISA 近似；(d) COVERED/PARTIAL 都只回答“指令是否存在于我们的编译器”，不回答“能否从 DSL 写出等价程序”（第 6 节）；(e) `.mir` 测试大多检验 LLVM 的机器 pass，按任务书归入 OUT-OF-SCOPE。

## 3. 逐类统计

### 3.1 测试种类 × 状态

| 种类 | 数量 | COVERED | PARTIAL | OUT-OF-SCOPE |
|---|---|---|---|---|
| codegen | 2844 | 16 | 68 | 2760 |
| mir-pass | 1807 | 0 | 0 | 1807 |
| pass | 322 | 0 | 0 | 322 |
| other | 23 | 0 | 0 | 23 |

### 3.2 目标类别（按 RUN 行；一个文件可计入多类）

| 目标类别 | 含该类的测试文件 | 其中 codegen |
|---|---|---|
| gfx950 | 125 | 97 |
| gfx94x | 357 | 272 |
| gfx9x | 1988 | 1285 |
| gfx10 | 1047 | 666 |
| gfx11 | 1199 | 839 |
| gfx12 | 1031 | 698 |
| pre-gfx9 | 2595 | 1484 |
| r600 | 229 | 224 |
| unknown | 4 | 4 |

任务书所说“只有约 10 个测试用 gfx950”是 `-mcpu=gfx950` 的**出现次数**；按文件、并把 `amdgpu9.50` 版本 triple 与 `-mcpu=` 都算进去，含 gfx950 RUN 行的测试文件是 **125** 个（其中 codegen 97 个）。gfx942 同理（出现 357 个文件）。

### 3.3 gfx950 级 codegen 测试的细分

| 状态 | 数量 |
|---|---|
| COVERED | 16 |
| PARTIAL | 68 |
| OUT-OF-SCOPE | 266 |

* **COVERED（16）**：真正检查了指令的 **2** 个：`fold-redundant-sgpr-vgpr-rw-across-bb.ll`（s_endpgm+v_readfirstlane_b32）、`remat-subrange-full-and-partial-use.ll`（s_endpgm）。其余 14 个是空洞覆盖（ELF/诊断/元数据类测试）。
* **PARTIAL（68）**：缺操作码个数的分布——缺 0 个 3（仅特性：`fence`）、缺 1 个 11、缺 2 个 9、缺 3–5 个 23、缺 6 个及以上 22；中位数 4。**最接近 COVERED 的**：`coexec-rewrite-mfma.ll`（缺 `v_mfma_f32_16x16x32_f16`）、`lds-dma-workgroup-release.ll`（缺 `s_load_dwordx4`）、`llvm.amdgcn.prng.ll`（缺 `v_prng_b32`）、`llvm.amdgcn.udot2.ll`（缺 `v_dot2_u32_u16`）、`llvm.amdgcn.udot4.ll`（缺 `v_dot4_u32_u8`）、`llvm.amdgcn.udot8.ll`（缺 `v_dot8_u32_u4`）、`lower-buffer-fat-pointers-nontemporal-metadata.ll`（缺 `s_load_dwordx4`）、`lsr-cost-model-vector-iv.ll`（缺 `s_cbranch`） 等。
* **如果撤掉 IR 特性闸门**（只看操作码）：350 个 gfx950 级 codegen 测试里，315 个缺操作码、27 个没有操作码检查、仅 8 个所有操作码都在我们的全集内。
* **OUT-OF-SCOPE（266）的主因是“非 kernel 函数”**：251 个测试含不是 `amdgpu_kernel` 的 `define`（LLVM 的 ISel 测试普遍用 `define void @f(...)` 或 `amdgpu_ps` 等做被测函数）；其中 236 个只有这一条硬特性。其余：私有内存/alloca 24 个、非内建调用 11 个（间接 4）、调试信息 2 个。
* **MFMA 类测试**（gfx950 级，检查了 `v_mfma*/v_smfmac*`）：PARTIAL 14、OUT-OF-SCOPE 12、COVERED 0。我们只有 6 族 MFMA/稀疏 MFMA（scaled/unscaled 32×32×64 f8f6f4、fp8 16×16×32 与 32×32×16、f16 16×16×16 与 32×32×8、`smfmac` 16×16×128 fp8）；LLVM 测试覆盖 bf16/xf32/i8/f64/bf8 混合、16×16×128 scaled 等数十种变体。

### 3.4 gfx9 家族放宽视图（不设目标闸门，只看 gfx9x/gfx94x/gfx950 的 codegen 测试）

1416 个 codegen 测试含 gfx9 家族 RUN 行（gfx90a/908/900/906 等 + gfx94x + gfx950）：硬不支持特性 1055、缺操作码 285、操作码全在全集内（含无检查）76。gfx9 家族的 ISA 与 gfx950 重叠最多，但 gfx90a/908 没有 gfx950 的 `v_cvt_scalef32*`/`global_load_lds_dwordx4`/`ds_read_b64_tr_*` 等，所以这仍然只是上界。

## 4. 缺失操作码 Top-30

### 4.1 gfx950 级 codegen 测试（统计的是需要该操作码的**测试数**）

| # | 操作码 | 测试数 | 示例测试 |
|---|---|---|---|
| 1 | `s_setpc_b64` | 210 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`amdgcn-av-scopes.ll` |
| 2 | `v_mov_b64` | 86 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`atomicrmw-expand.ll` |
| 3 | `s_mov_b64` | 74 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`amdgpu-cs-chain-fp-nosave.ll` |
| 4 | `scratch_load_dword` | 45 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`attributor-flatscratchinit-undefined-behavior2.ll` |
| 5 | `s_load_dwordx4` | 45 | `av-split-dead-valno-crash.ll`、`buffer-fat-pointers-memcpy.ll`、`build_vector.ll` |
| 6 | `v_cmp_eq_u32` | 44 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`bf16-conversions.ll` |
| 7 | `v_alignbit_b32` | 44 | `bitop3-shared-operand.ll`、`div-rem-by-constant-64.ll`、`i1-to-bf16.ll` |
| 8 | `s_movk_i32` | 40 | `bf16-conversions.ll`、`buffer-fat-pointer-atomicrmw-fadd.ll`、`buffer-fat-pointer-atomicrmw-fmax.ll` |
| 9 | `s_or_b64` | 39 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`atomicrmw-expand.ll` |
| 10 | `v_bfi_b32` | 37 | `bf16.ll`、`bitop3-shared-operand.ll`、`bitop3.ll` |
| 11 | `s_add_i32` | 34 | `amdgpu-cs-chain-fp-nosave.ll`、`attributor-flatscratchinit-undefined-behavior2.ll`、`bf16-conversions.ll` |
| 12 | `s_pack_ll_b32_b16` | 34 | `bf16-conversions.ll`、`build_vector.ll`、`fptrunc.f16.ll` |
| 13 | `v_lshl_add_u64` | 33 | `a-v-flat-atomicrmw.ll`、`atomicrmw-expand.ll`、`dagcombine-select.ll` |
| 14 | `s_and_b64` | 32 | `a-v-flat-atomicrmw.ll`、`atomicrmw-expand.ll`、`av-split-dead-valno-crash.ll` |
| 15 | `s_cbranch_execnz` | 29 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`atomicrmw-expand.ll` |
| 16 | `v_bfe_u32` | 28 | `bf16-conversions.ll`、`buffer-fat-pointer-atomicrmw-fadd.ll`、`buffer-fat-pointer-atomicrmw-fmax.ll` |
| 17 | `v_and_or_b32` | 28 | `bitop3-shared-operand.ll`、`bitop3.ll`、`buffer-fat-pointer-atomicrmw-fadd.ll` |
| 18 | `s_andn2_b64` | 27 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`atomicrmw-expand.ll` |
| 19 | `v_add3_u32` | 27 | `bf16-conversions.ll`、`buffer-fat-pointer-atomicrmw-fadd.ll`、`buffer-fat-pointer-atomicrmw-fmax.ll` |
| 20 | `scratch_store_dword` | 23 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`amdgpu-cs-chain-fp-nosave.ll` |
| 21 | `v_cmp_u_f32` | 23 | `bf16-conversions.ll`、`bf16.ll`、`buffer-fat-pointer-atomicrmw-fadd.ll` |
| 22 | `v_pk_mov_b32` | 23 | `rewrite-vgpr-mfma-to-agpr.ll`、`shufflevector-physreg-copy.ll`、`shufflevector.v2f32.v2f32.ll` |
| 23 | `v_pk_max_f16` | 22 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`buffer-fat-pointer-atomicrmw-fmax.ll` |
| 24 | `s_and_saveexec_b64` | 21 | `a-v-flat-atomicrmw.ll`、`buffer-fat-pointer-atomicrmw-fadd.ll`、`buffer-fat-pointer-atomicrmw-fmax.ll` |
| 25 | `v_accvgpr_write_b32` | 21 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`av-split-dead-valno-crash.ll` |
| 26 | `v_accvgpr_read_b32` | 20 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`av-split-dead-valno-crash.ll` |
| 27 | `flat_load_dword` | 19 | `a-v-flat-atomicrmw.ll`、`back-off-barrier-subtarget-feature.ll`、`flat-atomicrmw-fadd.ll` |
| 28 | `v_not_b32` | 19 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`bf16.ll` |
| 29 | `s_cselect_b32` | 18 | `a-v-flat-atomicrmw.ll`、`attributor-flatscratchinit-undefined-behavior2.ll`、`av-split-dead-valno-crash.ll` |
| 30 | `v_min_f32` | 17 | `a-v-flat-atomicrmw.ll`、`a-v-global-atomicrmw.ll`、`bf16.ll` |

### 4.2 全部 codegen 测试（含其它子目标）

| # | 操作码 | 测试数 |
|---|---|---|
| 1 | `s_setpc_b64` | 929 |
| 2 | `s_mov_b64` | 771 |
| 3 | `s_delay_alu` | 662 |
| 4 | `s_load_dwordx4` | 471 |
| 5 | `v_dual_mov_b32` | 467 |
| 6 | `s_wait_kmcnt` | 431 |
| 7 | `s_add_i32` | 374 |
| 8 | `s_clause` | 357 |
| 9 | `s_cselect_b32` | 320 |
| 10 | `s_wait_loadcnt_dscnt` | 315 |
| 11 | `flat_store_dword` | 310 |
| 12 | `s_or_b32` | 310 |
| 13 | `s_load_b64` | 303 |
| 14 | `s_or_b64` | 299 |
| 15 | `s_and_b64` | 292 |
| 16 | `global_store_b32` | 292 |
| 17 | `s_wait_loadcnt` | 289 |
| 18 | `v_cmp_eq_u32` | 273 |
| 19 | `s_movk_i32` | 272 |
| 20 | `s_cmp_lg_u32` | 255 |
| 21 | `v_add_i32` | 251 |
| 22 | `s_load_b128` | 249 |
| 23 | `s_load_b32` | 247 |
| 24 | `s_wait_samplecnt` | 246 |
| 25 | `s_wait_bvhcnt` | 236 |
| 26 | `v_mov_b16` | 236 |
| 27 | `s_setreg_imm32_b32` | 229 |
| 28 | `v_nop` | 210 |
| 29 | `s_wait_expcnt` | 208 |
| 30 | `s_and_saveexec_b64` | 207 |

**读法**：4.1 的头部是 **`s_setpc_b64`（函数返回）**、`s_mov_b64/s_and_b64/s_or_b64/s_andn2_b64/s_and_saveexec_b64/s_cbranch_execnz`（**exec 掩码/发散控制流**）、`scratch_*`（私有内存）、`v_mov_b64/v_pk_mov_b32/v_accvgpr_*`（64 位搬运与 AGPR）、`v_cmp_*/v_bfi/v_bfe/v_alignbit/v_add3/v_and_or`（通用整数 ALU）。这些恰好对应我们**有意不做**的特性：kernel 内无发散（`begin_if` 是 wave 级 `vcc` 分支，不用 exec 掩码）、无函数调用、无私有内存。

## 5. 函数级特性类别：DSL 是否建模

计数是含该特性的**测试文件数**（全部 / gfx950 级 codegen）。

| 类别 | 特性 | 测试数 | 我们的 DSL / 编译器 |
|---|---|---|---|
| ABI | kernel 入口（`amdgpu_kernel`）+ kernarg | 1720 / 129 | **是**：`FnBuilder` 定义 kernel，类型化 kernarg（`Op::Kernarg*`），与厂商 656 字节 ABI 兼容 |
| ABI | 非 kernel 函数（可调用函数） | 2393 / 251 | **否**：没有函数/栈帧/调用约定；每个 DSL `Function` 就是一个 kernel |
| 调用约定 | `amdgpu_ps/vs/gs/hs/es/ls/cs/gfx/cs_chain`、`fastcc` | 999 / 48 | **否**：只有计算 kernel（图形着色器、链式调用均不支持；也没有 `s_setpc/s_swappc`） |
| 调用 | 非内建调用 / 间接调用 / `invoke` | 433 / 8、97 / 4、1 / 0 | **否**（异常 `invoke/landingpad` 同样不支持） |
| 可变参数 | `...` / `va_arg` | 42 / 0 | **否** |
| 栈/帧 | `alloca`、动态 alloca、`stacksave`、addrspace(5) | 217 / 12、7 / 1、3 / 0、339 / 24 | **否**：没有 scratch/私有内存与栈指针；我们的寄存器溢出走 **LDS**（`ldsspill*`），不是 scratch |
| 栈/帧 | scratch 指令 / 溢出检查（CHECK 含 `scratch_*`、`Spill`） | 245 / 59、222 / 16 | **否**（无 `scratch_*` 操作码） |
| 内存 | 全局 addrspace(1) / 常量(4) / LDS(3) | 1812 / 278 / 552 | **是**：`global_*`、`s_load_*`（kernarg）、`ds_*` + 全局→LDS DMA；`global_load/store` 的宽度受限于我们用到的 dword/x2/x4/ushort/short |
| 内存 | buffer 资源 / 胖指针 addrspace(7,8) | 273 / 26 | **部分**：MUBUF raw 形式（描述符含运行时 `num_records`、`soffset`、`imm12`、LDS 直达）；无 struct/idxen/format 变体，无胖指针 lowering |
| 内存 | flat 指针 addrspace(0) | 25 | **否**（无 `flat_*` 操作码；我们只发 `global_*`） |
| 原子 | `atomicrmw` / `cmpxchg` / `fence` / atomic load|store | 171 / 53、54 / 25、35 / 7、55 / 25 | **部分**：只有 `global_atomic_add`（u32，可带返回值）与 release/acquire 栅栏原语（`buffer_wbl2/buffer_inv` + `s_waitcnt`）；没有 cmpxchg/swap/min/max/fadd、没有 flat/LDS 原子 |
| 线程模型 | workitem id x/y/z，workgroup id x/y/z | 467 / 45 | **部分**：`WorkitemId`（一维）与 `WorkgroupIdX/Y/Z`；无 y/z 维 workitem id、无 dispatch/queue/implicitarg 指针（52 个测试用到 ABI 指针） |
| 同步 | `s.barrier` / `s.waitcnt` / `s.sleep`/`s.setprio` | 84 / 23 | **是**：`barrier()`；waitcnt 由编译器按 scoreboard 管理（不作为用户 op 暴露除 `wait_vm/wait_lds`）；`s_setprio` 有；无 `s_sleep` |
| 调试 | `!dbg` / `llvm.dbg.*` / DI 元数据 | 78 / 2 | **否** |
| 属性 | `amdgpu-flat-work-group-size`、`amdgpu-waves-per-eu`、`amdgpu-num-vgpr`、`amdgpu-ieee`、`target-cpu/features`、`uniform-work-group-size` … | 见下 | **部分**：工作组大小由 `FnBuilder::new(name, wg)` 给出、VGPR/SGPR 预算由 RA 自己决定（以 256 个架构 VGPR 为界）；`waves-per-eu`/`num-vgpr`/IEEE/denorm 模式等不可配置；目标固定 gfx950 |
| 子目标特性 | `-mattr`：`real-true16`、`wavefrontsize32/64`、`cumode`、`flat-for-global`、`unaligned-access-mode`、`xnack/sramecc` … | 1005 个测试带 -mattr | **否**：固定 gfx950 默认特性（wave64、无 true16、无 xnack 配置） |
| 类型 | f64 / i64 运算、i128 及更宽整数、i8/i16 访存 | f64 486、i64 1126、i128 82、wide 25、i8 632、i16 739 | **部分**：`f32/i32/f16/bf16(cvt)/fp8` 与 u16 访存；**无** f64、64 位整数 ALU（指针对加法除外）、i8/i16 访存除 `ushort/short`、宽整数 |
| 向量/碎片 | `<N x T>` 向量类型 | 1808 | **部分**：只有不透明碎片 `Frag2/4/8/16`（MFMA 操作数与累加器），没有一般的向量 SSA 值与 `shufflevector`/`insertelement` 语义 |
| 内建函数 | `llvm.amdgcn.*`（不同名 2381 个；gfx950 级 269 个） | — | **部分**：我们建模的只有 mfma（6 族）、`smfmac` fp8、`readfirstlane`、`permlane32.swap`、`ds.read.tr8.b64`、`global.load.lds`、`raw.buffer.load/store(.lds)`、`workitem/workgroup id`、`s.barrier`、`exp2/log/rcp`、`cvt.pknorm`、`cvt.scalef32.pk.fp8.bf16`；其余（含 `image.*`、`interp.*`、`mbcnt`、`ds.bpermute`、`dpp`、`wmma`…）均未建模 |
| MIR | `.mir` 输入（1546 个） | — | **否**：我们的 MIR 与 LLVM MIR 不同；仅导出 post-RA MIR 给 llvm-tail。`.mir` 测试几乎全是 LLVM 机器 pass 的单元测试 |

**常见函数属性**（出现次数）：`amdgpu-flat-work-group-size` 103、`amdgpu-waves-per-eu` 77、`amdgpu-no-workitem-id-y` 57、`amdgpu-no-workitem-id-z` 57、`amdgpu-no-dispatch-ptr` 53、`amdgpu-no-dispatch-id` 51、`amdgpu-no-queue-ptr` 51、`amdgpu-no-implicitarg-ptr` 49、`amdgpu-no-workgroup-id-y` 48、`amdgpu-no-workgroup-id-z` 48、`amdgpu-no-workgroup-id-x` 45、`amdgpu-no-workitem-id-x` 44。**调用约定**：`amdgpu_kernel` 1720、`amdgpu_ps` 762、`amdgpu_cs` 127、`amdgpu_vs` 66、`amdgpu_gs` 60、`amdgpu_gfx` 45、`amdgpu_cs_chain` 26、`amdgpu_hs` 24、`fastcc` 19、`amdgpu_es` 10。

## 6. “用 Rust 构建一个 LLVM 测试”到底是什么意思（边界声明）

* 我们的 DSL 表达的是 **指令/特性级的 kernel**：用户用 `FnBuilder` 逐个写出 MFMA、VALU、`global_load`、`ds_read`、MUBUF 等操作及其数据流，编译器做调度、寄存器分配、hazard/wait 插入并发射汇编。**它不是 LLVM IR 前端**：不能读取 `.ll`/`.mir`，没有 LLVM 的类型系统、没有 `getelementptr`、没有 phi/select 的一般形式、没有函数与调用。
* 因此“能否从 Rust DSL 构建某个 LLVM 测试”只可能有三种含义：
  1. **指令级**：该测试 CHECK 的所有指令，我们的编译器是否**能发射**（本审计的 COVERED/PARTIAL 即回答此问题）；
  2. **特性级**：该测试想表达的**语义 kernel**，能否在 DSL 里写出来（需要人工翻译；本审计只在第 5 节给出特性是否被建模）；
  3. **逐字节等价**：编译结果与 `llc` 对该 IR 的输出一致——**这不是目标，也基本不可能**：LLVM 的指令选择、合并、调度与我们的流水线是不同的算法；我们保证的是“我们的两个后端彼此一致”（`tail_invariance`）和“数值/语义正确”（硬件验证），而不是和 `llc -O2` 一致。
* **COVERED ≠ 能通过该 LLVM 测试**：COVERED 只表示该测试在 gfx950 级 RUN 行里检查的助记符都在我们的指令全集中，且测试本身没有用到我们明确不建模的 IR/ABI 特性。很多 LLVM 测试检查的是 ISel/合并的**选择结果**（例如用 `v_add3_u32` 而不是两个 `v_add_u32`），这类测试即使助记符都在全集里，用 DSL 写出等价程序也未必会得到同一条指令。真正“把测试翻成 DSL 并得到相同/等价汇编”的工作量没有做，也不属于本审计。
* 本审计的实际用途：(1) 给出我们指令全集与 LLVM gfx950 测试的**差距清单**（第 4 节），可用来排优先级（例如若要支持通用 compute kernel，先补 `exec` 掩码与 64 位 ALU）；(2) 明确我们**有意不支持**的特性类别（第 5 节）；(3) 在没有 GPU 的情况下，用 LLVM 自己的测试作为“还差哪些指令”的参照。

## 7. 复现

```bash
python3 tools/llvm_feature_audit.py \
  --tests /shared/amdgpu/home/blyu_qle/llvm-src/rocm-10.1.0/llvm/test/CodeGen/AMDGPU \
  --repo . --out tools/llvm_feature_audit.json          # 同目录写出 *_summary.json，约 1 分钟，仅需 python3（无第三方依赖）
```

JSON 每条记录的字段：`path, ext, dir, kind(s), globalisel, n_run, mcpus, target_classes, triples, os, mattr, requires/unsupported/xfail, opcodes_by_class, opcodes, opcodes_gfx950_class, missing_opcodes, intrinsics, amdgcn_intrinsics, intrinsics_unmodelled, calling_conventions, attributes, address_spaces, types, features, status, reasons, vacuous`。
