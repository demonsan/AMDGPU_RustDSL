#!/usr/bin/env bash
# Hardware probe of the P1 base ops (docs/op-coverage-tileops.md §6.5).  Usage: HIP_VISIBLE_DEVICES=6 bash tools/tileop_probe.sh <probe> [own|llvm-tail]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/tileop
P=${1:?probe name}; BE=${2:-own}
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/tileop_probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
${AMDC:-bash tools/container/cargo.sh run -q --bin amdc --} emit "$P" --backend "$BE" --out "/work/build/tileop/$P.$BE.s" > build/tileop/$P.emit.log 2>&1 || { tail -5 build/tileop/$P.emit.log; exit 1; }
"$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/tileop/$P.$BE.co" "build/tileop/$P.$BE.s" || exit 1
OUT=$(python3 tools/tileop_probe_check.py gen "$P" build/tileop/$P.in) || exit 1
./build/tileop_probe_host "build/tileop/$P.$BE.co" "$P" 64 build/tileop/$P.in "$OUT" build/tileop/$P.$BE.out || exit 1
python3 tools/tileop_probe_check.py check "$P" build/tileop/$P.in build/tileop/$P.$BE.out
