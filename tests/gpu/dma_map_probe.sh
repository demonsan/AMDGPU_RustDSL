#!/usr/bin/env bash
# Which source bytes land in which LDS slot of one global_load_lds_dwordx4, for a lane address pattern (PD_MAP=0..4)?
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
bash tools/container/cargo.sh run -q --bin amdc -- emit probe_dma_map --backend "${BACKEND:-llvm-tail}" --out /work/build/probe_dma_map.s >/dev/null 2>&1
"${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_dma_map.co build/probe_dma_map.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/dma_map_host.cpp -o build/dma_map_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/dma_map_host build/probe_dma_map.co