#!/usr/bin/env bash
# Hardware check of the global -> LDS DMA op (probe_dma): lane i must land at byte 16 i (4 i) of the M0 base.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
bash tools/container/cargo.sh run -q --bin amdc -- emit probe_dma --backend "${BACKEND:-llvm-tail}" --out /work/build/probe_dma.s >/dev/null 2>&1
"${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_dma.co build/probe_dma.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/dma_host.cpp -o build/dma_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/dma_host build/probe_dma.co