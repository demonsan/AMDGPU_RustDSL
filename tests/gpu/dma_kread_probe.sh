#!/usr/bin/env bash
# Hardware check of DMA staging + K-fragment style reads, plain (default) and XOR-swizzled (PD_SWZ=1) layouts.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
bash tools/container/cargo.sh run -q --bin amdc -- emit probe_dma_kread --backend "${BACKEND:-llvm-tail}" --out /work/build/probe_dma_kread.s >/dev/null 2>&1
"${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/probe_dma_kread.co build/probe_dma_kread.s || exit 1
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/dma_kread_host.cpp -o build/dma_kread_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/dma_kread_host build/probe_dma_kread.co