#!/usr/bin/env bash
# VALU-writes-lane-mask -> v_cndmask hazard probe (tools/lanemask_hazard_probe.py).  Usage: HIP_VISIBLE_DEVICES=0 bash tools/lanemask_hazard_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/lanemask
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/tileop_probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
${AMDC:-bash tools/container/cargo.sh run -q --bin amdc --} emit probe_store --backend own --out /work/build/lanemask/template.s > build/lanemask/emit.log 2>&1 || { tail -5 build/lanemask/emit.log; exit 1; }
read -r NAME BYTES < <(python3 tools/lanemask_hazard_probe.py gen build/lanemask/template.s build/lanemask/lm.s)
"$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/lanemask/lm.co build/lanemask/lm.s || exit 1
head -c 256 /dev/zero > build/lanemask/in.bin
for rep in 1 2 3; do
  ./build/tileop_probe_host build/lanemask/lm.co "$NAME" 64 build/lanemask/in.bin "$BYTES" build/lanemask/out.bin > /dev/null || exit 1
  echo "== run $rep"; python3 tools/lanemask_hazard_probe.py check build/lanemask/out.bin
done
