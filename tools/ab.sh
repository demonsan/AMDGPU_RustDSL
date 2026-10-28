#!/usr/bin/env bash
# A/B timing of build-time kernel variants on the benchmark shape.
# Usage: AB_VARIANTS="FMHA_PITCH=128 FMHA_PITCH=144;FMHA_OPT=a,b" [AB_SHAPE="1 8 2048 2048"] bash tools/ab.sh
# Each space-separated variant is a semicolon-separated list of VAR=value build-time environment settings.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
K="${OUR_KERNEL:-fmha_fwd_hd128_mxfp8_t}"
[ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host \
    -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
set -- ${AB_SHAPE:-1 8 2048 2048}
i=0
for V in ${AB_VARIANTS:-FMHA_PITCH=128}; do
    i=$((i+1))
    ( export $(echo "$V" | tr ';' ' ')
      bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" --out "/work/build/ab$i.s" 2>&1 | grep -E "^error|out of VGPR" -A5
      "$ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/ab$i.co build/ab$i.s )
    echo "### variant $i: $V  ($(grep -m1 -E 'vgpr_count' build/ab$i.s))"
    for rep in 1 2; do
    REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$4 REF_ROWS=${REF_ROWS:-4} REF_ITERS=${REF_ITERS:-30} REF_CO2=build/ab$i.co REF_KNAME2="$K" \
        HIP_VISIBLE_DEVICES=0 ./build/ref_host "$CO" 2>&1 | grep -E "vs|timing|REF_|TFLOP" || true
    done
done