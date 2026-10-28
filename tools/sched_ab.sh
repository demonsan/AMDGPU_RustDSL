#!/usr/bin/env bash
# Scheduler A/B: build variants of one kernel, run 15 shapes vs the vendor kernel, compare every output
# bit-for-bit with the first variant (REF_DUMP2), and print timing of 1x8x2048x2048.
# Usage: REF_CO=<vendor .co> VARIANTS="base: full:AMDC_SCHED=1 xbar:AMDC_SCHED=1,AMDC_SCHED_OPTS=xbar" bash tools/sched_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
REF_CO="${REF_CO:?}"; K="${OUR_KERNEL:-fmha_fwd_hd128_mxfp8_t}"
cd "$REPO"; mkdir -p build/ab
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
SHAPES=("1 1 256 256 1" "1 2 512 512 1" "2 4 1024 1024 1" "1 8 2048 2048 1" "1 2 512 2048 1" "3 3 768 1280 1" "1 4 300 512 1" "2 8 700 640 2" "1 16 256 1024 4" "1 2 256 1000 1" "2 4 300 129 1" "1 8 512 100 2" "1 2 1024 1088 1" "1 4 77 1 1" "1 2 200 1337 2")
FIRST=""
for V in $VARIANTS; do
  name="${V%%:*}"; envs="${V#*:}"
  if [ -n "$envs" ]; then EV=(${envs//,/ }); else EV=(); fi
  env "${EV[@]}" AMD_ROCM_MOUNT="$ROC" bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" ${EMIT_FLAGS:-} --out "/work/build/ab/$name.s" 2>&1 | grep -E "wrote|^error|sched:" | cut -c1-200
  "$ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/ab/$name.co" "build/ab/$name.s" || exit 1
  [ -z "$FIRST" ] && FIRST="$name"
  i=0
  for CFG in "${SHAPES[@]}"; do
    set -- $CFG
    REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$4 REF_HK=$(( $2 / $5 )) REF_ROWS=16 REF_ITERS=30 REF_CO2="build/ab/$name.co" REF_KNAME2="$K" \
      REF_DUMP2="build/ab/${name}_$i.bin" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/ref_host "$REF_CO" > "build/ab/${name}_$i.log" 2>&1
    if [ "$name" != "$FIRST" ]; then
      if cmp -s "build/ab/${name}_$i.bin" "build/ab/${FIRST}_$i.bin"; then r=BITEXACT; else r=DIFFERENT; fi
    else r=-; fi
    ok=$(grep -c REF_KERNEL_OK "build/ab/${name}_$i.log")
    echo "[$name] shape $CFG: $r kernel_ok=$ok"
    if [ "$i" = 3 ]; then grep -E "timing|TFLOP" "build/ab/${name}_$i.log"; fi
    i=$((i+1))
  done
done