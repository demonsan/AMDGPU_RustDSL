#!/usr/bin/env bash
# Count-matched s_barrier probe (hang check + arrival-skew cost) for the gen3 rotated-phase design.
# Two wave groups in one 8-wave block: A = waves 0-3, B = waves 4-7.
#   aligned : both groups:  [BARRIER] read V M V M write          (same arrival phase)
#   rotated : A groups:     [BARRIER] read V M V M write
#             B group:      V [BARRIER] read M V M write          (barrier one phase later; count-matched)
#   none    : no barrier (baseline for the barrier cost)
#   skewX   : rotated with B held back X extra cycles before its barrier (X = 100/500/1000/2000)
#   asym    : rotated, A runs n iterations, B runs 2n (terminated-wave semantics: must not hang)
# Each iteration every wave writes its own LDS slot (value = iteration) and, after the barrier, reads its partner
# wave's slot (same lane); a mismatch increments a per-group LDS error counter. The last iteration of group A uses
# n = 100000 for the no-hang evidence.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ASM="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"; HIPROOT="${HIPROOT:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-5}"
OUT=build/gen3b; mkdir -p "$OUT"
python3 tools/gen3_barrier_gen.py || exit 1
for k in aligned rotated none skew100 skew500 skew1000 skew2000 asym; do
  if ! "$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$OUT/$k.co" "$OUT/$k.s" 2>"$OUT/$k.err"; then
    echo "ASSEMBLE FAIL $k :: $(head -4 "$OUT/$k.err")"; exit 1
  fi
done
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tools/gen3_barrier_host.cpp -o "$OUT/gen3b_host" -I"$HIPROOT/include" -L"$HIPROOT/lib" -lamdhip64 -Wl,-rpath,"$HIPROOT/lib" || exit 1
run() { timeout -s KILL 40 "$OUT/gen3b_host" "$OUT/$1.co" "$1" "$2"; local rc=$?; [ $rc -ne 0 ] && echo "KERNEL $1 n=$2 ** HANG/FAIL rc=$rc **"; return 0; }
run_wrap() {
  if command -v gpu-lock >/dev/null 2>&1; then gpu-lock bash -c "$(declare -f run); cd '$PWD'; run $1 $2"; else run "$1" "$2"; fi
}
echo "=== count-matched s_barrier probe (GPU $HIP_VISIBLE_DEVICES) ==="
run_wrap aligned 2000
run_wrap rotated 2000
run_wrap none 2000
for x in 100 500 1000 2000; do run_wrap skew$x 2000; done
echo "--- terminated-wave / imbalance check: A=n, B=2n (must terminate) ---"
run_wrap asym 1000
echo "--- no-hang evidence: rotated, n = 100000 ---"
run_wrap rotated 100000
