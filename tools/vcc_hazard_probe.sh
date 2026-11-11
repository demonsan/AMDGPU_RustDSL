#!/usr/bin/env bash
# VALU-writes-VCC -> VALU-reads-VCC hazard probe (tools/vcc_hazard_probe.py).  Usage: HIP_VISIBLE_DEVICES=0 bash tools/vcc_hazard_probe.sh
set -uo pipefail
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2}"; case ",$HIP_VISIBLE_DEVICES," in *,[4-7],*) echo "GPUs 4-7 are not to be used (user rule): HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 2;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/vcc
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/tileop_probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
${AMDC:-bash tools/container/cargo.sh run -q --bin amdc --} emit probe_store --backend own --out /work/build/vcc/template.s > build/vcc/emit.log 2>&1 || { tail -5 build/vcc/emit.log; exit 1; }
read -r NAME BYTES < <(python3 tools/vcc_hazard_probe.py gen build/vcc/template.s build/vcc/vcc.s)
"$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/vcc/vcc.co build/vcc/vcc.s || exit 1
head -c 256 /dev/zero > build/vcc/in.bin
for rep in 1 2 3; do
  ./build/tileop_probe_host build/vcc/vcc.co "$NAME" 64 build/vcc/in.bin "$BYTES" build/vcc/out.bin > /dev/null || exit 1
  echo "== run $rep"; python3 tools/vcc_hazard_probe.py check build/vcc/out.bin
done
