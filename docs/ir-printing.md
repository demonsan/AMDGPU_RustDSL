# IR printing and structural equality

`amdc print-ir <kernel> [--stage dsl|mir|mir-ra]` prints the closed IR in a readable, typed form.

## DSL stage (`--stage dsl`, default)

```
dsl @kernel
  kernarg_size=... workgroup=... lds=...
  profile=... lower_opts=... sched_opts=...
  arg ptr_q: ptr @0x0 +8
  ...
  contract:
    tile A base=0 slot=32768 slots=3
    arg_tensor ptr_q alloc=...
    grid_defs=[(0, ...), (1, ...)]
  body:
    %v0: ptr = KernargPtr { offset: 0 }
    %v1: u32 = WorkitemId
    loop BeginLoop { vars: [%r0], init: [...] }
      %v2: i32 = ConstI32(4)
      %v3: i32 = MulI32(Val(%v2), Val(%v1))
      if BeginIf { cond: Val(%v4) }
        ...
      else
        ...
      end
    endloop
```

The printer reconstructs a small scope tree (`Node::Loop/If/Tile`) from the current flat statement
list and prints that tree recursively. **When item B replaces the begin/end markers with nested
regions, only `ir_print::parse_scope` (and the node arms) need to change; the recursive printer,
the statement formatting and the canonical form stay.** Tile markers are printed as indented tile
regions. `Let` lines carry the result type; other statements print their operands with `%vN` (SSA
value) and `%rN` (loop variable), so the text is faithful to the IR rather than assembly.

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
    v7:V1 = VMovB32 imm=4
    v8:V1 = VMulLoU32 (v6:V1, v7:V1)
    SWaitLgkm [auto]
    ...
    goto b1 / if Scc0 goto b2 else b3 / endpgm
```

MOp names are the enum names; waits, `auto` waits and `pin` instructions are tagged, blocks print
their label/predicate/terminator. The printer is deliberately not a parser/round-trip format (the
optional parser was not needed).

## Structural equality / hash

`amdgpu_compiler::ir_print` exposes:

- `dsl_structural_eq(a, b)` and `dsl_structural_hash(f)`: full structural identity including
  args, contract, tiles/LDS plan and tuning fields; alpha-equivalent over `ValueId`/`VarId`
  (ids are renamed by first occurrence). The function **name is not part** of the identity.
- `dsl_body_structural_eq/hash` and `canonical_dsl_body`: the same without contract metadata,
  for comparing a builder before/after annotations are added.
- `canonical_dsl_body`: the alpha-canonical text form of the body (useful as a golden/diff artifact).

Canonicalisation is exhaustive over all `Op`/`Stmt` variants because it starts from their `Debug`
spelling and rewrites `ValueId(n)`/`VarId(n)`; adding a new op does not require a new match arm.

### Tests

`crates/amdgpu-kernels/tests/ir_print.rs`:

1. `probe_llvm_ids` vs `probe_expr_ids` and `probe_llvm_kernarg` vs `probe_expr_kernarg` are
   structurally equal (body form), hash equal, and canonical text equal; the canonical text does
   not contain the kernel names. (The manual twins predate the expression-layer contract lines,
   hence the body form.)
2. Mutating one `ConstI32` changes the hash and makes the two functions unequal.
3. The DSL printer emits header/contract/tiles/body and typed `%vN: <ty> = ...` lines.
4. MIR printing shows blocks, virtual classes, and physical `vN`/`sN` registers after RA.

Run: `cargo test -p amdgpu-kernels --test ir_print`.

## Limitations

- The text is a debugging/agent format, not a stable serialization format (the canonical form is
  only for equality/hash within this build).
- Op-specific pretty-printing is the enum's Debug name; a future attribute table can provide
  per-op human syntax without touching equality.
- Scope reconstruction assumes well-formed begin/end markers; malformed lists print markers as
  leaves instead of panicking.
