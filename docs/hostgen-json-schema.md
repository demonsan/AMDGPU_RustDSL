# `amdc gen-host` JSON metadata schema (version 1)

Every generated `<kernel>_abi.json` (and the `shipped_abi.json` aggregate, which is a JSON array of
the same objects) is versioned by `schema_version` and intended for Python/aiter loaders, ABI diffs
and host-side kernarg packing. The C++ header is generated from the same facts, so the two cannot
disagree about offsets/sizes.

## Top-level object

| field | type | meaning |
|---|---|---|
| `schema_version` | integer | currently **1**; reject other values |
| `generated_by` | string | `"amdc gen-host"` |
| `kernel` | string | exact `Function.name` / `hipModuleGetFunction` symbol |
| `kernarg_size` | integer | bytes the kernel expects (the launch buffer size) |
| `workgroup_size` | integer | required block x dimension (block y=z=1) |
| `lds_bytes` | integer | static LDS group-segment size (informational; **do not pass as dynamic shared memory** if you use `hipModuleLaunchKernel` — static LDS is part of the kernel) |
| `compile_profile` | string | `CompileProfile` debug name (`Vs128`, `T128`, `KvSplit`, `Gemm`, ...) |
| `args[]` | array | see below |
| `extents[]` | array | `arg_tensor` allocation contracts |
| `ka_is[]` | array | scalar kernarg equality relations |
| `ka_range[]` | array | scalar inclusive ranges |
| `ka_one_of[]` | array | scalar allowed-value sets |
| `grid_is[]` | array | launch-grid equality relations |
| `symbols[]` | array | free-symbol names and lower bounds |

## `args[]`

One entry per declared kernarg argument, sorted by byte offset:

```json
{"name":"ptr_a","offset":0,"size":8,"type":"Ptr"}
```

`type` is a DSL `DType` name: `Ptr` (8), `I32`/`U32`/`I1` (4), `F32` (4), `F16` (2), or one of the
opaque `Frag2/4/8/16`/`Rsrc` (fixed byte size). The arguments are non-overlapping and
`offset + size <= kernarg_size`; the C++ header additionally emits `offsetof`/`sizeof` static
asserts. Padding bytes not declared by the kernel are zero in a freshly initialized buffer.

## Contracts

`Ext` expressions (the same algebra the verifier uses) are rendered as infix strings:

```
const | ka(0xOFFSET) | grid(0|1|2) | SYMBOL
      | (A+B) | (A-B) | (A*B) | ceil(A/K) | floor(A/K)
```

where `ka(0x..)` is the 32-bit scalar arg at that byte offset (unsigned value) and `SYMBOL` is a free
symbol supplied by the host (e.g. `B`, `HK`, `KL`, `MT`, `NT`, `KT`).

- `extents[]`: `{"arg": "ptr_a", "alloc": "<Ext>", "logical": "<Ext>|null"}`. Host validation must
  ensure the buffer size for `arg` is at least `alloc`; `logical` is the useful (non-slack) extent
  when the kernel overreads a zeroed tail, and is null when equal.
- `ka_is[]`: `{"offset": 24, "expr": "(MT*256)"}` — the scalar at `offset` **equals** `expr`; this
  is how strides/tile counts are tied to the shape.
- `ka_range[]`: `{"offset": 80, "lo": 1, "hi": 9223372036854775807}` — inclusive range (hi may be
  `i64::MAX` for unbounded).
- `ka_one_of[]`: `{"offset": 0, "values": [1, 2]}` — membership.
- `grid_is[]`: `{"axis": 0, "expr": "NT"}` — launch grid dimension axis 0/1/2 equals `expr`.
- `symbols[]`: `{"name": "MT", "lo": 1}` — host must provide symbols in this range; the header
  defaults them to `lo` (or 1), but real launches should set `MT=M/256`, etc.

The C++ `validate(Args, BufferSizes, dim3 grid, Symbols, string*)` implements exactly these checks
and returns a `GV-HOST-*` code/message; the Python example does not reimplement the contract logic,
only the layout packing.

## Example (host only, no GPU)

```bash
# pack a GEMM kernarg buffer from the generated JSON
python3 tests/hostgen/json_kernarg.py \
    --out /tmp/gemm_v3.karg \
    --set ptr_a=0x1000 --set ptr_b=0x2000 --set ptr_c=0x3000 \
    --set ptr_sfa=0x4000 --set ptr_sfb=0x5000 \
    --set m=1280 --set n=6144 --set k=7168
```

`tests/hostgen/json_kernarg.py` is a ~90-line stdlib-only example: it checks `schema_version`, checks
non-overlap/bounds, packs `Ptr`/`I32`/`U32`/`F32`/`F16` with `struct.pack_into`, and leaves missing
arguments zero. `tests/hostgen/test_json_kernarg.py` is its host test: it packs every shipped/gemm
JSON, reads each field back at the declared offset and asserts the declared `kernarg_size`; run it
with `python3 tests/hostgen/test_json_kernarg.py` (no GPU).

## Compatibility

Adding a kernel/contract changes only generated artifacts; the JSON shape is stable within
`schema_version: 1`. Consumers should:
1. reject unknown/missing `schema_version`;
2. use `args[].offset/size`, never a hand-coded struct;
3. call the generated `validate` (C++) or evaluate the contract lists (Python) before launch;
4. keep the packed kernarg buffer alive until kernel completion.
