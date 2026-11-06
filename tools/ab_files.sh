#!/usr/bin/env bash
# Layout-median A/B of two prebuilt .s files of the SAME kernel (e.g. two compiler builds / flags): for each code shift (LAYOUT_PADS leading
# `s_nop 0` in the entry block) both are assembled and timed interleaved; prints the medians over the pads.
#   FMHA: HIP_VISIBLE_DEVICES=6 B=4 S=2048 RPB=256 bash tools/ab_files.sh fmha <kernel> a.s b.s
#   GEMM: HIP_VISIBLE_DEVICES=6 bash tools/ab_files.sh gemm <kernel> a.s b.s
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p build/abf
MODE=$1; K=$2; FA=$3; FB=$4
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang; V=${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}
[ -x build/ff/ref_host ] || { mkdir -p build/ff; g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ff/ref_host -I$R/include -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib; }
[ -x build/gemm_ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/gemm_ref_host.cpp -o build/gemm_ref_host -I"$R/include" -Itests/gpu -I. -L"$R/lib" -lamdhip64 -Wl,-rpath,"$R/lib" || exit 1
: > build/abf/a.t; : > build/abf/b.t
for p in ${LAYOUT_PADS-0 1 2 3 4 5 6 7}; do
  for v in a b; do
    f=$FA; [ $v = b ] && f=$FB
    awk -v p=$p 'BEGIN{d=0} {print} /^BB0:$/ && !d {for (j = 0; j < p; j++) print "    s_nop 0"; d=1}' "$f" > build/abf/p.s
    $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/abf/$v.co build/abf/p.s || exit 1
    if [ "$MODE" = fmha ]; then
      t=$(env REF_B=${B:-4} REF_H=8 REF_S=${S:-2048} REF_SK=${S:-2048} REF_HK=8 REF_CO2=build/abf/$v.co REF_KNAME2=$K REF_ROWS_PER_BLOCK=${RPB:-256} REF_ROWS=1 REF_WARMUP=150 REF_ITERS=100 ./build/ff/ref_host $V 2>&1 | awk '/^timing ours/ {for(i=1;i<=NF;i++) if($i ~ /^TFLOP/) print $(i-1)}')
    else
      t=$(env REF_ITERS=100 REF_WARMUP=100 REF_REPS=1 REF_ROWS=8 OURS_TILE_M=256 OURS_TILE_N=256 OURS_CO=build/abf/$v.co OURS_KNAME=$K ./build/gemm_ref_host 2>&1 | grep -E 'timing OURS' | sed 's/.*median \([0-9.]*\) TFLOP.*/\1/')
    fi
    echo "$t" >> build/abf/$v.t
  done
done
med() { sort -n "$1" | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
echo "$K: A $(med build/abf/a.t) [$(tr '\n' ' ' < build/abf/a.t)] | B $(med build/abf/b.t) [$(tr '\n' ' ' < build/abf/b.t)]"
