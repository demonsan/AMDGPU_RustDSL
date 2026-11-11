#!/usr/bin/env bash
# Hardware check of multi-wave global -> LDS DMA staging (probe_dma_tile).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
bash tools/container/cargo.sh run -q --bin amdc -- emit probe_dma_tile --backend "${BACKEND:-llvm-tail}" --out /work/build/probe_dma_tile.s >/dev/null 2>&1
"${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_dma_tile.co build/probe_dma_tile.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/dma_tile_host.cpp -o build/dma_tile_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/dma_tile_host build/probe_dma_tile.co probe_dma_tile 256
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/gpu_guard.sh"   # GPU tests only on GPUs 0-3
# single-wave sequence of 16 DMAs
bash tools/container/cargo.sh run -q --bin amdc -- emit probe_dma_seq --backend "${BACKEND:-llvm-tail}" --out /work/build/probe_dma_seq.s >/dev/null 2>&1
"${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_dma_seq.co build/probe_dma_seq.s || exit 1
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/dma_tile_host build/probe_dma_seq.co probe_dma_seq 64