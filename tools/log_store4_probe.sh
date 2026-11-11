#!/usr/bin/env bash
# Run probe_log_store4 (v_log_f32 + global_store_dwordx4) on the GPU with BOTH backends. Usage: bash tools/log_store4_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/log_store4_host.cpp -o build/log_store4_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
for BE in own llvm-tail; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit probe_log_store4 --backend $BE --out /work/build/probe_log_store4_$BE.s 2>&1 | grep -E "wrote|error" || exit 1
  "${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_log_store4_$BE.co build/probe_log_store4_$BE.s || exit 1
  echo "[$BE] $(HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/log_store4_host build/probe_log_store4_$BE.co | tr '\n' ' ')"
done