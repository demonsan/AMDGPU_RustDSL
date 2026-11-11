#!/usr/bin/env bash
# Function / ABI feature probes on the GPU (probe_llvm_*): emit (own backend), assemble, run, report FEATURE_PASS / FEATURE_FAIL per feature.
# Usage: bash tools/llvm_feature_probe.sh      (HIP_VISIBLE_DEVICES defaults to 0)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/llvm_feature_host.cpp -o build/llvm_feature_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
emit() { bash tools/container/cargo.sh run -q --bin amdc -- emit "$1" --backend own --out "/work/build/$1.s" 2>&1 | grep -E "compile error" -A4; "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/$1.co" "build/$1.s" || echo "ASSEMBLE_FAIL $1"; }
rc=0
for K in kernarg ids roles lds branch loop atomic maskge k1 k2; do emit probe_llvm_$K; done
for C in kernarg ids roles lds branch loop atomic maskge; do ./build/llvm_feature_host $C build/probe_llvm_$C.co || rc=1; done
./build/llvm_feature_host twokernel build/probe_llvm_k1.co build/probe_llvm_k2.co || rc=1
echo "LLVM_FEATURE_PROBE rc=$rc"; exit $rc