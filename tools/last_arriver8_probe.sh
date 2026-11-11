#!/usr/bin/env bash
# 8-wave last-arriver memory-order test. Usage: RUNS=1000000 THRASH=1 STAGGER=4 bash tools/last_arriver8_probe.sh [kernel...]
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/last_arriver8_host.cpp -o build/last_arriver8_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
for K in probe_la_verify; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit $K --backend "${BACKEND:-own}" --out /work/build/$K.s 2>&1 | grep -E "wrote|error" || exit 1
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/$K.co build/$K.s || exit 1
done
KERNELS=("$@"); [ ${#KERNELS[@]} -gt 0 ] || KERNELS=(probe_la8 probe_la8_lite probe_la8_plain_fenced probe_la8_plain probe_la8_nodrain)
for K in "${KERNELS[@]}"; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit $K --backend "${BACKEND:-own}" --out /work/build/$K.s 2>&1 | grep -E "wrote|error" || exit 1
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/$K.co build/$K.s || exit 1
  HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/last_arriver8_host build/$K.co $K build/probe_la_verify.co "${RUNS:-100000}" "${THRASH:-0}" "${STAGGER:-4}" | tail -13
done