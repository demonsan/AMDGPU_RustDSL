#!/usr/bin/env bash
# Emit (own backend), assemble and run the safe-asm end-to-end probe kernel on the GPU (global load -> MFMA/VALU -> global store).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
bash tools/container/cargo.sh run -q --bin amdc -- emit probe_safeasm_e2e --backend own --out /work/build/probe_safeasm_e2e.s 2>&1 | grep -E "wrote|error" || exit 1
"${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_safeasm_e2e.co build/probe_safeasm_e2e.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/safeasm_e2e_host.cpp -o build/safeasm_e2e_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" ./build/safeasm_e2e_host build/probe_safeasm_e2e.co