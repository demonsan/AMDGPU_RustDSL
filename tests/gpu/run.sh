#!/usr/bin/env bash
# End-to-end check: DSL -> assembly -> code object -> gfx950 -> CPU reference.
# Run on the GPU node. Requires ROCm and one visible GPU.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
KERNEL="${KERNEL:-attention_fwd}"
cd "$REPO"
mkdir -p build

echo "== emit =="
AMD_ROCM_MOUNT="${AMD_ROCM_MOUNT:-$ROC}" bash tools/container/cargo.sh run -q --bin amdc -- emit "$KERNEL" --backend "${BACKEND:-own}" ${AMDC_EXTRA_FLAGS:-} --out "/work/build/${KERNEL}.s"

echo "== assemble =="
"${ASM_ROC:-$ROC}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
    -mcode-object-version=6 -o "build/${KERNEL}.co" "build/${KERNEL}.s"

echo "== host =="
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/attn_host.cpp -o build/attn_host \
    -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"

export ATTN_KERNEL="$KERNEL"
echo "== run =="
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/attn_host "build/${KERNEL}.co"
