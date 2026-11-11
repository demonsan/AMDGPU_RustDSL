#!/usr/bin/env bash
# Model-vs-hardware table for the sched_* microkernels: bash tools/sched_model_vs_hw.sh "<kernels>" ; env HIP_VISIBLE_DEVICES (default 1)
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
printf "%-20s %-6s | %8s %8s %6s | %8s %8s %6s\n" kernel variant "mod1w" "hw1w" "err%" "mod2w" "hw2w" "err%"
for K in $1; do for V in plain sched swp; do
  case $V in plain) OPTS=""; FL="";; sched) OPTS=""; FL="--sched";; swp) OPTS="swp"; FL="--sched";; esac
  SIMLINE=$(AMDC_SCHED_OPTS=$OPTS AMD_ROCM_MOUNT="$ROC" bash tools/container/cargo.sh run -q --bin amdc -- simmodel $K $FL 2>/dev/null | grep "^SIM" | tail -1)
  W1=$(echo "$SIMLINE" | sed -n 's/.*waves1=\([0-9.]*\).*/\1/p'); W2=$(echo "$SIMLINE" | sed -n 's/.*waves2=\([0-9.]*\).*/\1/p')
  DIV=1; [ "$V" = swp ] && DIV=2
  mkdir -p build/hw/mv_$V
  AMDC_SCHED_OPTS=$OPTS AMD_ROCM_MOUNT="$ROC" bash tools/container/cargo.sh run -q --bin amdc -- emit $K $FL --out /work/build/hw/mv_$V/$K.s >/dev/null 2>&1
  "$ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/hw/mv_$V/$K.co build/hw/mv_$V/$K.s 2>/dev/null || continue
  H1=$(HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} ./build/sched_host build/hw/mv_$V/$K.co $K 64 1 4000 10 | sed -n 's/.*cycles_per_iter=\([0-9.]*\).*/\1/p')
  H2=$(HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-1} ./build/sched_host build/hw/mv_$V/$K.co $K 512 1 4000 10 | sed -n 's/.*cycles_per_iter=\([0-9.]*\).*/\1/p')
  awk -v k=$K -v v=$V -v w1=$W1 -v w2=$W2 -v d=$DIV -v h1=$H1 -v h2=$H2 'BEGIN{ m1=w1/d; m2=w2/d; printf "%-20s %-6s | %8.1f %8.1f %5.0f%% | %8.1f %8.1f %5.0f%%\n", k, v, m1, h1, (m1-h1)/h1*100, m2, h2, (m2-h2)/h2*100 }'
done; done