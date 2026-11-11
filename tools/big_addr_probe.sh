#!/usr/bin/env bash
# B1 hardware check beyond 4 GiB (docs/op-coverage-tileops.md): HIP_VISIBLE_DEVICES=0 bash tools/big_addr_probe.sh [own|llvm-tail]
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/tileop
BE=${1:-own}
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/big_addr_host.cpp -o build/big_addr_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
${AMDC:-bash tools/container/cargo.sh run -q --bin amdc --} emit probe_big_addr --backend "$BE" --out "/work/build/tileop/probe_big_addr.$BE.s" > build/tileop/probe_big_addr.emit.log 2>&1 || { tail -5 build/tileop/probe_big_addr.emit.log; exit 1; }
"$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/tileop/probe_big_addr.$BE.co" "build/tileop/probe_big_addr.$BE.s" || exit 1
./build/big_addr_host "build/tileop/probe_big_addr.$BE.co"