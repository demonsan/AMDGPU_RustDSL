#!/usr/bin/env bash
# probe_store_x2 (global_store_dwordx2) on the GPU with both backends. Usage: bash tools/store_x2_probe.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/store_x2_host.cpp -o build/store_x2_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
for BE in own llvm-tail; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit probe_store_x2 --backend $BE --out /work/build/probe_store_x2_$BE.s 2>&1 | grep -E "wrote|rror" || exit 1
  "${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_store_x2_$BE.co build/probe_store_x2_$BE.s || exit 1
  echo "[$BE] $(HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/store_x2_host build/probe_store_x2_$BE.co | tr '\n' ' ')"
  grep -E "global_store" build/probe_store_x2_$BE.s
done