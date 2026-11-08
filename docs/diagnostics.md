# Structured diagnostics

Agents (and humans) read compiler results as JSON instead of parsing asm or prose. The implementation is in
[diag.rs](../crates/amdgpu-compiler/src/diag.rs) and the tests in [diagnostics.rs](../crates/amdgpu-kernels/tests/diagnostics.rs).

## Commands

| command | output |
|---|---|
| `amdc emit <kernel> [--out f.s] --json` | `{"kernel","ok":true,"backend","vgpr","sgpr","out","diagnostics":[]}`; on failure `{"kernel","ok":false,"diagnostics":[...]}` (exit 1) |
| `amdc gpu-verify <kernel or --all or --shipped> --json [--strict]` | `[{"kernel","diagnostics":[...]}...]`: every GPU-semantics finding; with `--strict`, exit 1 on an error |
| `amdc ir-check <file.ir> --json [--verify]` | `{"file","kernel","ok","statements","values","diagnostics"}`: IR-verifier findings (and the certificate's with `--verify`); a parse error is `IR-PARSE` |
| `amdc regplan <kernel> --json` | `{"kernel","vgpr","agpr","regions":[{"name","insts","vgpr_footprint_peak","vgpr_live_peak","agpr_footprint_peak","peak_at","live_at_peak"}]}` |
| `amdc srcmap <kernel> [--line N]` | `[{"line","block","inst","region","value","src","asm"}]`: every own-backend asm line mapped to the IR value whose lowering emitted it and that value's Rust builder call (waits / nops / copies added after lowering have `value: null`) |
| `amdc summary <kernel> [--json]` | `{"kernel","workgroup_size","vgpr","agpr","sgpr","lds_bytes","blocks","instructions","waits","nops","nop_wait_states","regions":[{"name","mfma","vmem","lds","smem","sync","salu","valu"}],"certificate":{"errors","warnings","trusted_assumes","trusted_contracts"}}` |

## A diagnostic

```json
{"code":"GV-BUF-WRAP","severity":"error","message":"stmt 6: buffer store: ...","kernel":"gv_attack_buf_wrap",
 "stmt":6,"region":null,"src":"crates/amdgpu-kernels/src/verifier_cases/mod.rs:227:18","src_exact":false,
 "values":[{"value":"v5","src":"crates/amdgpu-kernels/src/verifier_cases/mod.rs:227:18"}],
 "fix":"use an out-of-range sentinel that stays >= num_records after the instruction offset ..."}
```

| field | meaning |
|---|---|
| `code` | a stable identifier: the GPU-verifier codes (`GV-*`), `IR-VERIFY`, `IR-UNIFORM-BRANCH`, `IR-CVT-SCALE`, `IR-PARSE`, `RA-OUT-OF-REGISTERS`, `COMPILE-REG-BUDGET`, `COMPILE-ERROR` |
| `severity` | `error`, `warn` or `info` |
| `stmt` | the IR node: an index into `Function::stmts()` (the same numbering as `stmt N` in messages and `amdc print-ir`) |
| `region` | the innermost weave region (`RegionMark`) enclosing the statement |
| `src` | `file:line:col` of the Rust builder call that created the node. It is exact (`src_exact: true`) for a value-defining statement. For other statements (stores, branches) it is the nearest preceding value's location, and `values` then lists the statement's operands with their own locations. |
| `values` | the values involved (`vN`; for an allocation failure `rN=vM`, the register and the value it was lowered from) with their builder-call locations |
| `fix` | a suggested fix, where one is known (wrap sentinel, descriptor size, provenance, out-of-bounds, uniform branch, cvt scale, register pressure) |

## Source locations

Every `FnBuilder` method is `#[track_caller]`, so `let_` records the location of the call into the builder: the kernel's own line, or the
tile-library line when a library helper built the value. It is stored by `ValueId` in `Function::src_locs`. That is a side table for
diagnostics only: it is not part of the IR (no codec, no printing, and `Debug` prints nothing), so the generated code is unchanged. A test
compiles a kernel with and without the table, and the shipped kernels' md5s are identical.

## Compile failures

* IR-verifier errors: one diagnostic per line, localized by their `stmt N`.
* Running out of registers: the allocator's message names virtual registers (`rN`). They are mapped back to the values they were lowered
  from (`lower::last_vreg_origins`), so `values` lists the live values at the failing point with their builder calls.
* The launch register budget (`COMPILE-REG-BUDGET`) and other errors come through with the message and, where present, the statement.

## Asm source map

Every machine instruction carries `MInst::origin`: the IR value whose lowering emitted it (for a store or a branch, the last value-defining
statement before it). It has no codegen effect. `diag::asm_map` matches the emitted lines in order against the instructions, so an asm
line (from a profiler, a hazard or a wait report, or a disassembly) maps to an IR value and its builder call. Weave-region names come from
the same table (`summary` and `srcmap` use the real names, not `regionN`).
