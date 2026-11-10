# IR printing and structural equality

`amdc print-ir <kernel>|--from-ir <file> [--stage dsl|mir|mir-ra] [--out <file>]` prints the IR of a kernel.

## DSL stage (`--stage dsl`, default): the `irv1` text

The DSL stage prints the complete, parseable `irv1` text (`amdgpu_dsl::ir_text::print`); it parses back to the same `Function`
(`ir_text::parse`), so a kernel can be edited and compiled again (`amdc emit --from-ir`). Format, workflow and checks:
[`ir-text.md`](ir-text.md). Use `--out <file>` for large kernels (a container's stdout pipe can cut long output short).

```
irv1
dsl "probe_expr_kernarg"
kernarg_size 36  workgroup 64  lds 0  value_count 18
profile None  lower_opts []  sched_opts None
var_tys []
arg Arg { name: "a", ty: Ptr, kernarg_offset: 0, size: 8 }
contract Contract { .. }
values { }
body {
  %v0: Ptr = KernargPtr { offset: 0 }
  %v8: I32 = MulI32(Val(6), Val(7))
  loop vars [VarId(0)] init [I32(0)] { .. }
  if Val(3) { .. } else { .. }
}
```

## MIR stages

- `--stage mir`: `lower::lower(&Function)` — virtual registers `vN:<RClass>`.
- `--stage mir-ra`: `compile_mir_alloc(&Function)` — physical `vN`/`sN` plus the original class.

```
mir @kernel
  kernarg_size=... workgroup=... lds=... blocks=... vregs=...
  arg ...
  b0:
    s8:S2 = SLoadDwordx2
    v6:V1 = VReadPhys
    v8:V1 = VMulLoU32 (v6:V1, v7:V1)
    SWaitLgkm [auto]
    ...
    goto b1 / if Scc0 goto b2 else b3 / endpgm
```

MOp names are the enum names; waits, `auto` waits and `pin` instructions are tagged, blocks print their label/predicate/terminator. The MIR
text is a debugging format (not parsed back).

## Structural equality / hash

`amdgpu_dsl::print` (re-exported by `amdgpu_codegen::ir_print`):

- `dsl_structural_eq(a, b)` / `dsl_structural_hash(f)`: full structural identity including args, contract, tiles/LDS plan, AGPR marks,
  register plan, region strategies and tuning fields; alpha-equivalent over `ValueId` / `VarId`. The function **name is not part** of the identity.
- `dsl_body_structural_eq/hash` and `canonical_dsl_body`: the same without contract metadata, for comparing a builder before/after
  annotations are added.

Canonicalisation is structural: `ir_text::alpha_canonical` renumbers every ValueId / VarId by first occurrence in program order through the
generated id visitor (`ir_codec.rs`, `tools/gen_ir_codec.py`) — in statements, op operands and annotations — and permutes the type tables
accordingly; the canonical text is the `Debug` spelling of the renamed function.

### Tests

`crates/amdgpu-kernels/tests/ir_print.rs` (expression-layer twins structurally equal, a mutated constant changes the hash, printer content,
MIR printing) and `crates/amdgpu-kernels/tests/ir_text.rs` (every kernel round-trips through the text exactly and compiles to identical
assembly; edits and located errors; alpha equivalence).
