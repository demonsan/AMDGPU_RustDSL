#!/usr/bin/env bash
# MUBUF probes on the GPU. Usage: bash tools/buffer_probe.sh [rw|lds|both]   (own backend; the lds forms are own-only)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/buffer_host.cpp -o build/buffer_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
for M in ${1:-both}; do
  for T in $( [ "$M" = both ] && echo "rw lds" || echo "$M" ); do
    K=probe_buffer_$T; BE="${BACKEND:-own}"
    bash tools/container/cargo.sh run -q --bin amdc -- emit $K --backend $BE --out /work/build/$K.s 2>&1 | grep -E "wrote|compile error" -A4 || exit 1
    "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/$K.co build/$K.s || exit 1
    echo "== $K ($BE)"; grep -E "buffer_(load|store)" build/$K.s | head -12
    HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" ./build/buffer_host $T build/$K.co
  done
done