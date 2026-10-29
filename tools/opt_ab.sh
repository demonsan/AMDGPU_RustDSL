#!/usr/bin/env bash
# Bit-exactness + timing of compiler-optimisation variants of fmha_t against the first variant.
#   AB_VARIANTS="none imm cmpfold imm,cmpfold" bash tools/opt_ab.sh      (each variant = value of AMDC_LOWER_OPTS)
# For every variant: emit (default LLVM-tail backend, optional AB_EXTRA_ENV like "FMHA_OPT=..."), assemble with ROCm 10.1,
# dump the bf16 output for the 15 run_ref shapes and compare md5 with the first variant, then time B1H8S2048 (3 reps).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/ab
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; ASM="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
GPU="${HIP_VISIBLE_DEVICES:-4}"; K="${OUR_KERNEL:-fmha_fwd_hd128_mxfp8_t}"
CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
[ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
SHAPES=("1 1 256 256 1" "1 2 512 512 1" "2 4 1024 1024 1" "1 8 2048 2048 1" "1 2 512 2048 1" "3 3 768 1280 1" "1 4 300 512 1" "2 8 700 640 2" "1 16 256 1024 4" "1 2 256 1000 1" "2 4 300 129 1" "1 8 512 100 2" "1 2 1024 1088 1" "1 4 77 1 1" "1 2 200 1337 2")
first=""
for V in ${AB_VARIANTS:-none}; do
  T="v_$(echo $V | tr ',' '_')"
  ( export AMDC_LOWER_OPTS=$V ${AB_EXTRA_ENV:-}
    bash tools/container/cargo.sh run -q --bin amdc -- emit $K --out /work/build/ab/$T.s 2>&1 | grep -E "compile error|error:" -A3
    "$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/ab/$T.co build/ab/$T.s ) || continue
  echo "### variant $V: static valu=$(grep -cE '^\s+v_' build/ab/$T.s) mov=$(grep -cE '^\s+v_mov_b32' build/ab/$T.s) nop=$(grep -cE 's_nop' build/ab/$T.s) $(grep -m1 vgpr_count build/ab/$T.s | tr -s ' ')"
  diffs=0; i=0
  for CFG in "${SHAPES[@]}"; do
    i=$((i+1)); set -- $CFG
    REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$4 REF_HK=$(( $2 / $5 )) REF_ROWS=4 REF_ITERS=1 REF_DUMP2=build/ab/$T.$i.bin REF_CO2=build/ab/$T.co REF_KNAME2=$K HIP_VISIBLE_DEVICES=$GPU ./build/ref_host "$CO" >/dev/null 2>&1
    if [ -z "$first" ]; then :; else cmp -s build/ab/$T.$i.bin build/ab/$first.$i.bin || { diffs=$((diffs+1)); echo "   DIFF shape $i ($CFG)"; }; fi
  done
  [ -z "$first" ] && first=$T && echo "   (reference variant)" || echo "   bit-exact vs $first: $((15-diffs))/15 shapes identical"
  for rep in 1 2 3; do
    echo -n "   rep$rep: "; REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=4 REF_ITERS=30 REF_CO2=build/ab/$T.co REF_KNAME2=$K HIP_VISIBLE_DEVICES=$GPU ./build/ref_host "$CO" 2>&1 | grep -E "timing +ours" | tr -s ' '
  done
done