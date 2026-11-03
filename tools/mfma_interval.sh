#!/usr/bin/env bash
# MFMA issue-interval micro-probe: bash tools/mfma_interval.sh   (GPU node; HIP_VISIBLE_DEVICES, ASM_ROC, ROCM_PATH)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; ASM="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"; ITERS="${ITERS:-4096}"
KS=$(python3 tools/mfma_interval.py build/mfma_interval.s $ITERS | sed 's/kernels: //')
"$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/mfma_interval.co build/mfma_interval.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/mfma_interval_host.cpp -o build/mfma_interval_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
for r in 1 2; do echo "--- run $r"; ITERS=$ITERS HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" ./build/mfma_interval_host build/mfma_interval.co $KS; done