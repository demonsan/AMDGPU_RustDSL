#!/usr/bin/env bash
# MFMA per-instruction ceiling probe: bash tools/mfma_ceiling.sh   (GPU node; HIP_VISIBLE_DEVICES, ASM_ROC, ROCM_PATH, ITERS, BLOCK_SETS="256 1")
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; ASM="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"; ITERS="${ITERS:-4096}"
KS=$(python3 tools/mfma_ceiling.py build/mfma_ceiling.s $ITERS | sed 's/kernels: //')
"$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/mfma_ceiling.co build/mfma_ceiling.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/mfma_ceiling_host.cpp -o build/mfma_ceiling_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
for B in ${BLOCK_SETS:-256 1}; do echo "--- blocks=$B"; BLOCKS=$B ITERS=$ITERS ./build/mfma_ceiling_host build/mfma_ceiling.co $KS; done