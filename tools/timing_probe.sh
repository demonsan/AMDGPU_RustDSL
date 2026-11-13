#!/usr/bin/env bash
# LDS timing-reliance probe (tools/timing_probe.py): X reads left in flight across GAP instructions and two newer reads, consumed without a
# wait, 8 waves of LDS traffic. Usage: HIP_VISIBLE_DEVICES=2 bash tools/timing_probe.sh [gaps] [iters] [reps]
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/timing
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
GAPS="${1:-0 2 8 16}"; ITERS="${2:-4000}"; REPS="${3:-5}"
[ -x build/tileop_probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
if [ -n "${AMDC:-}" ]; then W="$REPO"; else W=/work; fi
${AMDC:-bash tools/container/cargo.sh run -q --release --bin amdc --} emit probe_store --backend own --out "$W/build/timing/template.s" > build/timing/emit.log 2>&1 || { tail -5 build/timing/emit.log; exit 1; }
head -c 4096 /dev/zero > build/timing/in.bin
rc=0
for g in $GAPS; do
  read -r NAME BYTES < <(python3 tools/timing_probe.py gen build/timing/template.s build/timing/g$g.s "$g" "$ITERS")
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/timing/g$g.co build/timing/g$g.s || exit 1
  for r in $(seq 1 "$REPS"); do
    ./build/tileop_probe_host build/timing/g$g.co "$NAME" 512 build/timing/in.bin "$BYTES" build/timing/out.bin > /dev/null || exit 1
    echo -n "gap $g rep $r: "; python3 tools/timing_probe.py check build/timing/out.bin "$ITERS" || rc=1
  done
done
exit $rc