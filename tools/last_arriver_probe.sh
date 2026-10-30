#!/usr/bin/env bash
# Memory-order test of the fused-combine (last-arriver) pattern on the GPU. Usage: REPS=100 bash tools/last_arriver_probe.sh [kernel...]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/last_arriver_host.cpp -o build/last_arriver_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
KERNELS=("$@"); [ ${#KERNELS[@]} -gt 0 ] || KERNELS=(probe_last_arriver probe_last_arriver_lite probe_last_arriver_nofence)
for K in "${KERNELS[@]}"; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit $K --backend "${BACKEND:-own}" --out /work/build/$K.s 2>&1 | grep -E "wrote|error" || exit 1
  "${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/$K.co build/$K.s || exit 1
  HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" ./build/last_arriver_host build/$K.co $K "${REPS:-100}" | tail -14
done