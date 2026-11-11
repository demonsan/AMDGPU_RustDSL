#!/usr/bin/env bash
# probe_mfma_scale_opsel on the GPU with both backends: the op_sel / op_sel_hi scale byte select of v_mfma_scale_f32_32x32x64_f8f6f4.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/mfma_opsel_host.cpp -o build/mfma_opsel_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
for BE in own llvm-tail; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit probe_mfma_scale_opsel --backend $BE --out /work/build/probe_mfma_scale_opsel_$BE.s 2>&1 | grep -E "wrote|rror|compile error" -A3 || exit 1
  "${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_mfma_scale_opsel_$BE.co build/probe_mfma_scale_opsel_$BE.s || exit 1
  echo "[$BE] $(HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/mfma_opsel_host build/probe_mfma_scale_opsel_$BE.co | tr '\n' ' ')"
  grep -E "v_mfma_scale" build/probe_mfma_scale_opsel_$BE.s | head -8
done