# Golden outputs

## `md5_manifest.tsv`

The own-backend asm md5 of every reachable kernel: `amdc list`, the instantiations the dispatch rules produce over the shape grid of
`amdgpu_kernels::reachable::dispatched_names`, and the Python-op instantiations (`reachable::python_op_names`). It is the regression oracle of
the four-layer refactor ([`docs/refactor-4layer-plan.md`](../../docs/refactor-4layer-plan.md)): a refactoring commit must leave it identical,
which `crates/amdc/tests/md5_manifest.rs` checks in `cargo test`.

Check by hand (prints the changed / added / removed kernels):

```bash
bash tools/container/cargo.sh run -q --release --bin amdc -- md5-manifest --check tests/golden/md5_manifest.tsv
```

Regenerate **only** for an intended code change, in the same commit, after the change passed its gate (`tools/port_gate.sh`,
`tools/gemm_gate.sh`, the family's host checks, `python/tests/run_all.sh`); name the gate and the changed kernels in the commit message:

```bash
bash tools/container/cargo.sh run -q --release --bin amdc -- md5-manifest --out tests/golden/md5_manifest.tsv
```

A kernel that does not compile is recorded as `ERROR` (and one that is not registered as `MISSING`), so a breakage shows up as a diff too.
