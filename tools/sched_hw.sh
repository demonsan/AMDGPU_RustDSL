#!/usr/bin/env bash
# Run sched_* microkernels on hardware: bash tools/sched_hw.sh "<kernels>" "<blocks>" "<n>" ; env AMDC_SCHED/AMDC_SCHED_OPTS/EMIT_FLAGS apply to the build.
# Prints RESULT lines (cycles_per_iter at 2.4 GHz) -- one co per invocation in build/hw/$TAG.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
TAG="${TAG:-run}"
cd "$REPO"; mkdir -p build/hw/$TAG
[ -x build/sched_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/sched_host.cpp -o build/sched_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
for K in $1; do
  AMD_ROCM_MOUNT="$ROC" bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" ${EMIT_FLAGS:-} --out "/work/build/hw/$TAG/$K.s" 2>&1 | grep -E "^error|compile error|falling" | cut -c1-200
  "$ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/hw/$TAG/$K.co" "build/hw/$TAG/$K.s" || continue
  for B in $2; do for N in $3; do
    echo -n "[$TAG] "; HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-1}" ./build/sched_host "build/hw/$TAG/$K.co" "$K" "$B" 1 "$N" 10 | tr '\n' ' '; echo
  done; done
done