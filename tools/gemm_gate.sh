#!/usr/bin/env bash
# MXFP8 blockscale GEMM gate (our tile-library kernel vs the aiter PR #6696 4-wave pin-AGPR code object, same run):
#   BUILD    the gemm cargo test (every variant builds with its layout proofs + compiles; strict verification of v3/v4)
#   VERIFY   amdc gpu-verify --strict: 0 errors, 0 warnings, ledger printed
#   ACC      GEMM_OK (CPU double reference on sampled rows, every column) on 3 shapes, ours and the vendor's
#   DET      3 launches (separate processes, same inputs): identical output md5
#   PERF     same-run median of 5 (x100 launches) ours vs vendor on 4096 x 4096 x 8192
# Usage: HIP_VISIBLE_DEVICES=0 bash tools/gemm_gate.sh [kernel=gemm_mxfp8_bs_v4]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/gate
K="${1:-gemm_mxfp8_bs_v4}"; export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
FAIL=0; note() { echo "$*"; }; bad() { echo "FAIL: $*"; FAIL=1; }
echo "==== BUILD"
out=$(bash tools/container/cargo.sh test -q -p amdgpu-kernels --test gemm_mxfp8 2>&1); echo "$out" | grep -E 'test result' || bad "cargo test"; echo "$out" | grep -q 'test result: ok' || bad "gemm tests"
echo "==== VERIFY"
for v in "$K"; do
  out=$(bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify "$v" --strict 2>&1)
  ne=$(echo "$out" | grep -cE '^\s+\[GV-[A-Z-]+\] Error'); nw=$(echo "$out" | grep -cE '^\s+\[GV-[A-Z-]+\] Warn')
  echo "$v: errors=$ne warnings=$nw $(echo "$out" | grep -m1 -E '^\s+certificate')"; [ "$ne" = 0 ] && [ "$nw" = 0 ] || bad "verify $v"
done
echo "==== ACC (GEMM_OK, ours + vendor)"
for shp in "4096 4096 8192" "8192 7168 7168" "1280 6144 7168"; do
  set -- $shp
  out=$(GEMM_M=$1 GEMM_N=$2 GEMM_K=$3 REF_ROWS=48 REF_ITERS=0 OURS_KERNEL="$K" OURS_TILE_M=256 OURS_TILE_N=256 bash tools/gemm_ref.sh 2>&1)
  r=$(echo "$out" | grep -oE '^REF .*-> GEMM_(OK|FAIL)' | grep -oE 'GEMM_(OK|FAIL)'); o=$(echo "$out" | grep -E '^OURS ' | grep -oE 'bad=[0-9]+.*GEMM_(OK|FAIL)')
  echo "M=$1 N=$2 K=$3: vendor $r | ours $o"; echo "$o" | grep -q GEMM_OK || bad "acc $shp"
done
echo "==== DET (3 launches)"
sums=""
for i in 1 2 3; do
  rm -f build/gate/det_$i.bin
  SKIP_REF=1 REF_ROWS=1 REF_ITERS=0 OURS_DUMP=build/gate/det_$i.bin OURS_KERNEL="$K" OURS_TILE_M=256 OURS_TILE_N=256 bash tools/gemm_ref.sh >/dev/null 2>&1
  sums="$sums $(md5sum build/gate/det_$i.bin 2>/dev/null | cut -c1-12)"
done
echo "md5:$sums"; [ "$(echo $sums | tr ' ' '\n' | sort -u | wc -l)" = 1 ] && [ -n "$(echo $sums | tr -d ' ')" ] || bad "determinism"
echo "==== PERF (interleaved: 7 processes x (vendor, ours) x 3 reps x 100, median of the per-process medians)"
RS=""; OS=""
for i in 1 2 3 4 5 6 7; do
  out=$(REF_ROWS=1 REF_REPS=3 OURS_KERNEL="$K" OURS_TILE_M=256 OURS_TILE_N=256 bash tools/gemm_ref.sh 2>&1)
  RS="$RS $(echo "$out" | grep -E '^timing REF' | grep -oE 'median [0-9.]+' | awk '{print $2}')"
  OS="$OS $(echo "$out" | grep -E '^timing OURS' | grep -oE 'median [0-9.]+' | awk '{print $2}')"
done
med() { echo $* | tr ' ' '\n' | grep . | sort -g | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
r=$(med $RS); o=$(med $OS); echo "vendor runs:$RS"; echo "ours runs:  $OS"
echo "4096x4096x8192: ours $o TF/s vs vendor $r TF/s -> $(awk -v a="$o" -v b="$r" 'BEGIN{printf "%.3f", a/b}')"
[ -n "$o" ] || bad "perf"
echo "==== GATE $([ $FAIL = 0 ] && echo PASS || echo FAIL)"
