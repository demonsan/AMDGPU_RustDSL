#!/usr/bin/env bash
# Warm ATT (rocprofv3 thread trace, one SIMD, the 5th dispatch) of one of our attention kernels and of an aiter fmha_v3 .co on the same data, through tests/gpu/attn_host.cpp (aiter fmha_v3 656-byte ABI), then the per-class
# table (tools/att_classes.py) and the MFMA-gap attribution (tools/att_gaps.py) of each. The CU that hosts a wave of the grid is not predictable: CUS (default "1 2 0 3 4 5 6 7 8 9") are tried until a trace has instructions.
# usage (repo root, GPU node, amdc built; HIP_VISIBLE_DEVICES=N):
#   bash tools/att_attn.sh <tag> <our_kernel> <vendor_co> B H HK SQ SK causal [D]
#   env: OURS_ENV / VENDOR_ENV (strings of VAR=value passed to attn_host for that side: ROWS BLOCK PAIR DV KS ...; defaults of the host otherwise), AMDC, AITER_HSA, CUS, ATT_DECODER_DIR,
#        KREGEX (our kernel-name regex for the trace filter, default the kernel name), VREGEX (default "aiter")
# example (hd192 / hd128 MLA prefill, causal, B1 H16 2048): OURS_ENV="ROWS=128 BLOCK=256 PAIR=0 DV=128" VENDOR_ENV="ROWS=128 BLOCK=256 DV=128" \
#   bash tools/att_attn.sh hd192c fmha_fwd_hd192_hd128_bf16_v3_causal_np_q128 $AITER_HSA/fmha_v3_fwd/fwd_hd192_hd128_bf16_causal.co 1 16 16 2048 2048 1 192
set -u
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
TAG=$1; K=$2; VCO=$3; B=$4; H=$5; HK=$6; SQ=$7; SK=$8; CAUSAL=$9; D=${10:-128}
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
DEC=${ATT_DECODER_DIR:-/shared/apps/ubuntu/opt/rocm-10.1.0/lib}
AMDC=${AMDC:-/tmp/target-kfmha/debug/amdc}
O=build/att/$TAG; rm -rf "$O"; mkdir -p "$O"
$AMDC emit "$K" --backend own --out "$O/k.s" > /dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$O/k.co" "$O/k.s" || { echo "emit/assemble of $K failed"; exit 1; }
{ [ -x build/ff/attn_host ] && [ build/ff/attn_host -nt tests/gpu/attn_host.cpp ]; } || { mkdir -p build/ff; $R/bin/amdclang++ -x hip -O2 --offload-arch=gfx950 --rocm-path=$R -std=c++17 tests/gpu/attn_host.cpp -o build/ff/attn_host -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib || exit 1; }
VK=$(strings "$VCO" | grep -m1 '^_ZN')
att() { who=$1; regex=$2; envs=$3; shift 3
  for cu in ${CUS:-1 2 0 3 4 5 6 7 8 9}; do
    rm -rf "$O/$who"; env $envs ITERS=10 WARMUP=4 CHECK=0 LSE=0 $R/bin/rocprofv3 --att --att-library-path "$DEC" --att-target-cu $cu --att-simd-select 0x1 --kernel-include-regex "$regex" --kernel-iteration-range "[5]" -d "$O/$who" -o run -- ./build/ff/attn_host "$@" > "$O/$who.log" 2>&1
    st=$(find "$O/$who" -name "stats*.csv" | head -1)
    n=$([ -n "$st" ] && python3 tools/att_classes.py "$st" 2>&1 | head -1 | sed -E 's/.*instructions executed ([0-9]+).*/\1/')
    [ -n "$n" ] && [ "$n" -gt 0 ] 2>/dev/null && { echo "$who: CU $cu ($n instructions)"; return 0; }
  done; echo "$who: no wave found"; return 1; }
att ours "${KREGEX:-$K}" "${OURS_ENV:-}" ours "$O/k.co" "$K" $B $H $HK $SQ $SK $CAUSAL $D
att vendor "${VREGEX:-aiter}" "${VENDOR_ENV:-}" vendor "$VCO" "$VK" $B $H $HK $SQ $SK $CAUSAL $D
for who in ours vendor; do
  st=$(find "$O/$who" -name "stats*.csv" | head -1); ui=$(dirname "$(find "$O/$who" -name code.json | head -1)")
  echo "######## $who"; [ -n "$st" ] && python3 tools/att_classes.py "$st" | head -24; [ -n "$ui" ] && [ "$ui" != "." ] && python3 tools/att_gaps.py "$ui" 12 | head -28
done