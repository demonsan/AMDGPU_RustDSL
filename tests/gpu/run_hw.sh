#!/usr/bin/env bash
# Run a hand-written assembly kernel through the same harness as generated ones.
# Usage: KERNEL=hw_roles ATTN_BLOCK=128 tests/gpu/run_hw.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
KERNEL="${KERNEL:-hw_roles}"
SRC="${SRC:-tools/${KERNEL}.s}"
cd "$REPO"
mkdir -p build

echo "== assemble $KERNEL =="
"$ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
    -mcode-object-version=6 -o "build/${KERNEL}.co" "$SRC"

echo "== host =="
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/attn_host.cpp -o build/attn_host \
    -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"

echo "== run =="
export ATTN_KERNEL="${KERNEL}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/attn_host "build/${KERNEL}.co"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/gpu_guard.sh"   # GPU tests only on GPUs 0-3
