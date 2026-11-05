#!/usr/bin/env bash
# Layout-median A/B of kernel variants (`kernel` or `kernel@name=v,..` = `--specialize`, docs/jit.md): every variant is emitted once, then for
# each code shift (LAYOUT_PADS leading `s_nop 0` in the entry block) all variants are timed interleaved; prints per-variant per-pad TF/s and the
# median over pads (single-layout comparisons move by ~2%, docs/current-status.md §4.5).
#   FMHA: HIP_VISIBLE_DEVICES=6 B=4 S=2048 RPB=256 bash tools/ab_layout.sh fmha <variant>...
#   GEMM: HIP_VISIBLE_DEVICES=6 bash tools/ab_layout.sh gemm <variant>...        (gemm_ref_host default shape 4096x4096x8192)
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p build/abl
MODE=$1; shift
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
A=${AMDC:-${AMD_CARGO_TARGET:-/tmp/amd-rust-dsl-target}/debug/amdc}; V=${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}
PADS=${LAYOUT_PADS-0 1 2 3 4 5 6 7}
[ -x build/ff/ref_host ] || { mkdir -p build/ff; g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ff/ref_host -I$R/include -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib; }
[ -x build/gemm_ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/gemm_ref_host.cpp -o build/gemm_ref_host -I"$R/include" -Itests/gpu -I. -L"$R/lib" -lamdhip64 -Wl,-rpath,"$R/lib" || exit 1
i=0; for v in "$@"; do
  k=${v%%@*}; sp=""; case "$v" in *@*) sp="--specialize ${v#*@}";; esac
  $A emit "$k" --backend=own $sp --out build/abl/v$i.s 2>build/abl/v$i.log || { echo "EMIT FAIL $v"; tail -3 build/abl/v$i.log; exit 1; }
  echo "v$i = $v  md5 $(md5sum < build/abl/v$i.s | cut -c1-12) insts $(grep -cE '^\s+(v_|s_|ds_|buffer_|global_)' build/abl/v$i.s)"; : > build/abl/v$i.t; i=$((i+1))
done
n=$i
for p in $PADS; do
  for i in $(seq 0 $((n-1))); do
    awk -v p=$p 'BEGIN{d=0} {print} /^BB0:$/ && !d {for (j = 0; j < p; j++) print "    s_nop 0"; d=1}' build/abl/v$i.s > build/abl/p.s
    $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/abl/v$i.co build/abl/p.s || exit 1
  done
  for i in $(seq 0 $((n-1))); do
    v=${@:$((i+1)):1}; k=${v%%@*}
    if [ "$MODE" = fmha ]; then
      t=$(env REF_B=${B:-4} REF_H=8 REF_S=${S:-2048} REF_SK=${S:-2048} REF_HK=8 REF_CO2=build/abl/v$i.co REF_KNAME2=$k REF_ROWS_PER_BLOCK=${RPB:-256} REF_ROWS=1 REF_WARMUP=150 REF_ITERS=100 ./build/ff/ref_host $V 2>&1 | awk '/^timing ours/ {for(i=1;i<=NF;i++) if($i ~ /^TFLOP/) print $(i-1)}')
    else
      t=$(env REF_ITERS=100 REF_WARMUP=100 REF_REPS=1 REF_ROWS=8 OURS_TILE_M=256 OURS_TILE_N=256 OURS_CO=build/abl/v$i.co OURS_KNAME=$k ./build/gemm_ref_host 2>&1 | grep -E 'timing OURS' | sed 's/.*median \([0-9.]*\) TFLOP.*/\1/')
    fi
    echo "$t" >> build/abl/v$i.t
  done
done
for i in $(seq 0 $((n-1))); do
  echo "v$i median $(sort -n build/abl/v$i.t | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')  per-pad: $(tr '\n' ' ' < build/abl/v$i.t)"
done
