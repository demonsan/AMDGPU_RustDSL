#!/usr/bin/env bash
# MFMA result latency per operand format (tools/mfma_lat_probe.py).  Usage: HIP_VISIBLE_DEVICES=0 bash tools/mfma_lat_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/mlat
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/tileop_probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
${AMDC:-bash tools/container/cargo.sh run -q --bin amdc --} emit probe_store --backend own --out /work/build/mlat/template.s > build/mlat/emit.log 2>&1 || { tail -5 build/mlat/emit.log; exit 1; }
read -r NAME BYTES < <(python3 tools/mfma_lat_probe.py gen build/mlat/template.s build/mlat/lat.s)
"$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/mlat/lat.co build/mlat/lat.s || exit 1
head -c 256 /dev/zero > build/mlat/in.bin
for rep in 1 2 3; do
  ./build/tileop_probe_host build/mlat/lat.co "$NAME" 64 build/mlat/in.bin "$BYTES" build/mlat/out.bin > /dev/null || exit 1
  echo "== run $rep"; python3 tools/mfma_lat_probe.py check build/mlat/out.bin
done
