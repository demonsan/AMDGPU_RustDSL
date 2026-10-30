#!/usr/bin/env bash
# Determinism reproducer for the biasacc,vpre scheduler/RA race: emit _vs128 with the given env, run ref_host N times, print the md5 of each output.
# Usage: FMHA_OPT=... [AMDC_SCHED=..] [AMDC_SCHED_OPTS=..] RUNS=6 TAG=x bash tools/race_repro.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
TAG="${TAG:-x}"; K="${KERNEL:-fmha_fwd_hd128_mxfp8_vs128}"
bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" --backend own --out "/work/build/race_$TAG.s" 2>&1 | grep -E "wrote|sched:|rror"
"${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/race_$TAG.co" "build/race_$TAG.s" || exit 1
[ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
for i in $(seq 1 "${RUNS:-6}"); do
  rm -f build/race_out.bin
  REF_B=${REF_B:-4} REF_H=${REF_H:-8} REF_S=${REF_S:-2048} REF_SK=${REF_SK:-2048} REF_HK=${REF_HK:-8} REF_ROWS=16 REF_ITERS=1 \
    REF_CO2="build/race_$TAG.co" REF_KNAME2="$K" REF_DUMP2=build/race_out.bin HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" \
    ./build/ref_host "${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}" > /dev/null 2>&1
  echo "$TAG run$i md5=$(md5sum build/race_out.bin 2>/dev/null | cut -c1-12)"
done