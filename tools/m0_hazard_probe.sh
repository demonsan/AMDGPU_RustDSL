#!/usr/bin/env bash
# SALU M0 write -> LDS-DMA hazard probe (tools/m0_hazard_probe.py).  Usage: [THREADS=512] [REPS=5] HIP_VISIBLE_DEVICES=2 bash tools/m0_hazard_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/m0
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/tileop_probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
${AMDC:-bash tools/container/cargo.sh run -q --bin amdc --} emit probe_store --backend own --out "$( [ -n "${AMDC:-}" ] && echo "$REPO" || echo /work )/build/m0/template.s" > build/m0/emit.log 2>&1 || { tail -5 build/m0/emit.log; exit 1; }
read -r NAME BYTES < <(python3 tools/m0_hazard_probe.py gen build/m0/template.s build/m0/m0.s)
"$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/m0/m0.co build/m0/m0.s || exit 1
python3 tools/m0_hazard_probe.py inputs build/m0/in.bin
for rep in $(seq 1 "${REPS:-3}"); do
  ./build/tileop_probe_host build/m0/m0.co "$NAME" "${THREADS:-64}" build/m0/in.bin "$BYTES" build/m0/out.bin > /dev/null || exit 1
  echo "== run $rep"; python3 tools/m0_hazard_probe.py check build/m0/out.bin
done
