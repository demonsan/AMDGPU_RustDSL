#!/usr/bin/env bash
# All Python op checks on a GPU node, with a summary table (docs/python-jit.md):
#   python/tests/ops_check.py   every amdgpu_jit op against its Python reference (stdlib only)
#   python/tests/decode_check.py the paged decode (both dispatch forms)
#   python/tests/test_unit.py    host-only unit tests
#   crates/amdgpu-kernels/tests/dispatch_compile.rs (--ignored, release) every reachable kernel compiles
#   python/tests/node_check.py   (--full) bit identity vs the C++ harnesses, cache behaviour, launch overhead
# usage (repo root, GPU node): HIP_VISIBLE_DEVICES=6 bash python/tests/run_all.sh [--full] [ops_check name filters ...]
#   env: AMDGPU_JIT_AMDC (default: build the release amdc with tools/container/cargo.sh and use it), OUT (default build/py_checks)
set -uo pipefail
FULL=0
if [ "${1:-}" = "--full" ]; then FULL=1; shift; fi
OUT=${OUT:-build/py_checks}
mkdir -p "$OUT"
if [ -z "${AMDGPU_JIT_AMDC:-}" ]; then
  bash tools/container/cargo.sh build -q --release --bin amdc > "$OUT/build.log" 2>&1 || { echo "amdc build failed (see $OUT/build.log)"; exit 2; }
  export AMDGPU_JIT_AMDC="${AMD_CARGO_TARGET:-/tmp/amd-rust-dsl-target}/release/amdc"
fi
t0=$(date +%s)
# every kernel a launcher can reach compiles (the dispatch rules over a shape grid, the Python-op instantiations, `amdc list`)
bash tools/container/cargo.sh test -q --release -p amdgpu-kernels --test dispatch_compile -- --include-ignored > "$OUT/compile.log" 2>&1; comp=$?
python3 -m unittest discover -s python/tests -p 'test_unit.py' > "$OUT/unit.log" 2>&1; unit=$?
python3 python/tests/ops_check.py "$@" > "$OUT/ops.log" 2>&1; ops=$?
python3 python/tests/decode_check.py > "$OUT/decode.log" 2>&1; dec=$?
nc=skipped
if [ $FULL = 1 ]; then python3 python/tests/node_check.py --skip-perf > "$OUT/node.log" 2>&1 && nc=PASS || nc=FAIL; fi
python3 - "$OUT" "$unit" "$ops" "$dec" "$nc" "$(( $(date +%s) - t0 ))" "$comp" <<'PY'
import re, sys
out, unit, ops, dec, nc, secs, comp = sys.argv[1:]
rows = []
for f in ("ops.log", "decode.log"):
    for line in open(f"{out}/{f}", encoding="utf-8", errors="replace"):
        m = re.match(r"^(ok|FAIL)\s+(.*)$", line.rstrip())
        if m:
            st, rest = m.groups()
            name, _, detail = rest.partition(":")
            rows.append((name.strip(), "PASS" if st == "ok" else "FAIL", detail.strip()[:90]))
w = max([len(r[0]) for r in rows] + [10])
print(f"| {'check'.ljust(w)} | result | detail |")
print(f"|{'-' * (w + 2)}|--------|--------|")
for n, s, d in rows:
    print(f"| {n.ljust(w)} | {s:6} | {d} |")
npass = sum(1 for r in rows if r[1] == "PASS")
cmp_ok = comp == "0"
print(f"\nop checks: {npass} / {len(rows)} PASS;  kernel compile sweep: {'PASS' if cmp_ok else 'FAIL (see compile.log)'};  unit tests: {'PASS' if unit == '0' else 'FAIL'};  node_check: {nc};  {secs} s;  logs: {out}/")
ok = npass == len(rows) and rows and unit == "0" and nc != "FAIL" and cmp_ok
print("OVERALL:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
