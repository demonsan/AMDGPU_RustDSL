#!/usr/bin/env bash
# MFMA dependency / operand-reuse probe (tools/mfma_dep.py): cycles per MFMA (one CU, in-kernel s_memtime), then the sustained full-chip effective clock and the SMU power per pattern.
# usage: HIP_VISIBLE_DEVICES=4 bash tools/mfma_dep.sh        env: ASM_ROC, ROCM_PATH, SMI (amd-smi), SECS (sustained seconds per pattern, default 3)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build/dep
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; ASM="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"; SMI="${SMI:-$ROC/bin/amd-smi}"; SECS="${SECS:-3}"
KS=$(python3 tools/mfma_dep.py build/dep/dep.s 2048 | sed 's/kernels: //')
"$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/dep/dep.co build/dep/dep.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/mfma_ceiling_host.cpp -o build/dep/host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" MPI=32
if [ -z "${ONLY_SUSTAINED:-}" ]; then
echo "=== one CU (BLOCKS=1): cycles per MFMA, ITERS=2048, 8 launches"
BLOCKS=1 ITERS=2048 REPS=8 ./build/dep/host build/dep/dep.co $KS | grep -E "^(s32|s16)"
echo "=== full chip (BLOCKS=256), cold launches: first 8 launches"
BLOCKS=256 ITERS=2048 REPS=8 ./build/dep/host build/dep/dep.co $KS | grep -E "^(s32|s16)"
fi
[ -n "${REV:-}" ] && KS=$(echo $KS | tr " " "\n" | tac | tr "\n" " ")
# which amd-smi index is the loaded GPU: scan while a pattern runs
BLOCKS=256 ITERS=2048 REPS=1200 ./build/dep/host build/dep/dep.co s32_indep_v0 > /dev/null 2>&1 &
sleep 1; BEST=0; BI=0
for g in 0 1 2 3 4 5 6 7; do p=$("$SMI" metric -g $g --power 2>/dev/null | grep -m1 SOCKET_POWER | grep -oE '[0-9]+' | head -1); p=${p:-0}; [ "$p" -gt "$BEST" ] && { BEST=$p; BI=$g; }; done; wait
echo "=== loaded GPU: amd-smi index $BI (${BEST} W while running)"
echo "=== sustained full chip: SECS=$SECS per pattern; clock = in-kernel s_memtime/s_memrealtime over the last quarter of the launches; power = SMU SOCKET_POWER mean over the second half of the run"
printf "%-16s %9s %8s %8s %8s %8s\n" kernel cyc/MFMA ns/MFMA clkGHz powerW samples
for K in $KS; do
  ( BLOCKS=256 ITERS=2048 REPS=100000 timeout $SECS ./build/dep/host build/dep/dep.co $K > /dev/null 2>&1 ) &
  sleep 0.5; : > build/dep/smi.log; t0=$(date +%s.%N)
  while [ "$(awk -v a="$t0" -v b="$(date +%s.%N)" -v s="$SECS" 'BEGIN{print (b - a < s - 0.8)}')" = 1 ]; do
    p=$("$SMI" metric -g $BI --power --clock 2>/dev/null | grep -E "SOCKET_POWER|CLK:" | head -2 | tr -s ' ' | tr '\n' ' '); echo "$(date +%s.%N) $p" >> build/dep/smi.log; done
  wait
  # the timing run itself (fixed number of launches, tail statistics), after the power sample
  R=$(BLOCKS=256 ITERS=2048 REPS=1200 TAIL=1 ./build/dep/host build/dep/dep.co $K | grep -E "^$K ")
  pw=$(awk '{n++; for (i = 1; i <= NF; i++) if ($i == "SOCKET_POWER:") v[n] = $(i + 1)} END {s = 0; c = 0; for (i = int(n / 2) + 1; i <= n; i++) if (v[i] > 0) {s += v[i]; c++} if (c) printf "%.0f %d", s / c, c; else printf "0 0"}' build/dep/smi.log)
  echo "$R" | awk -v pw="$pw" '{split(pw, a, " "); printf "%-16s %9s %8s %8s %8s %8s\n", $1, $3, $7, $6, a[1], a[2]}'
done