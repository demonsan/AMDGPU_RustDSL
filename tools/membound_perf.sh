#!/usr/bin/env bash
# Consolidated memory-bound perf matrix (docs/membound-perf.md): runs tools/b2_matrix.sh REPS times (correctness + bandwidth of every batch-2 kernel), takes the median GB/s per case, compares it with the
# committed baseline (docs/membound_baseline.tsv) and the stream roofline (STREAM_GBS, default 6500), flags regressions (> TOL, default 10 %) and any correctness failure, and writes the markdown table.
# usage (on a gfx950 node, repo root, amdc built):  bash tools/membound_perf.sh [check|update]     env: REPS (3), TOL (0.10), STREAM_GBS (6500), AMDC, plus b2_matrix.sh's
#   check  (default): run, print + write docs/membound-perf.md, exit 1 on a regression / correctness failure
#   update: also rewrite docs/membound_baseline.tsv from this run (do it deliberately, with the change that moves the numbers)
# the vendor (aiter) side of the comparison is tools/vendor_membound.sh; the numbers are same-node, same-run only (the node noise is +-3-10 %: TOL is above it).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build/mp
REPS="${REPS:-3}"; MODE="${1:-check}"
for i in $(seq 1 "$REPS"); do bash tools/b2_matrix.sh > build/mp/run$i.txt 2>&1; done
python3 tools/membound_perf_md.py build/mp/run*.txt --baseline docs/membound_baseline.tsv --out docs/membound-perf.md --tol "${TOL:-0.10}" --stream "${STREAM_GBS:-6500}" $([ "$MODE" = update ] && echo --update)