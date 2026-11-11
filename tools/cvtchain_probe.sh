#!/usr/bin/env bash
# Conversion chain: DSL path vs safe-asm path on the GPU (bit-identical outputs). Usage: bash tools/cvtchain_probe.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
for K in probe_cvtchain_dsl probe_cvtchain_asm; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit $K --backend own --out /work/build/$K.s 2>&1 | grep -E "wrote|error|rror" || exit 1
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/$K.co build/$K.s || exit 1
done
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/cvtchain_host.cpp -o build/cvtchain_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/cvtchain_host build/probe_cvtchain_dsl.co build/probe_cvtchain_asm.co