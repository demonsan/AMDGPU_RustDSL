#!/usr/bin/env bash
# Warm ATT (rocprofv3 thread trace, CU 1 SIMD 0) of our FMHA kernel and the vendor kernel on the same inputs, one dispatch each
# (the 5th of the run: warm). Output: build/att/<tag>/{ours,vendor}/... + the per-class table (tools/att_classes.py) and the MFMA-gap
# attribution (tools/att_gaps.py) of each.
# usage (repo root, GPU node, amdc built): HIP_VISIBLE_DEVICES=5 bash tools/att_fmha.sh <tag> [B H S] [kernel]
#   env: REF_DATA (default normal), ROWS_PER_BLOCK (default 256), REF_CO (vendor .co), ATT_DECODER_DIR (librocprof-trace-decoder)
set -u
TAG=${1:-att}; B=${2:-1}; H=${3:-8}; S=${4:-2048}; K=${5:-fmha_fwd_hd128_mxfp8_v3_vs128}
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
DEC=${ATT_DECODER_DIR:-/shared/apps/ubuntu/opt/rocm-10.1.0/lib}
AMDC=${AMDC:-/tmp/target-kfmha/debug/amdc}; V=${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}
O=build/att/$TAG; rm -rf "$O"; mkdir -p "$O"
$AMDC emit "$K" --backend own --out "$O/k.s" >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$O/k.co" "$O/k.s" || exit 1
{ [ -x build/ff/ref_host ] && [ build/ff/ref_host -nt tests/gpu/ref_host.cpp ]; } || { mkdir -p build/ff; g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ff/ref_host -I$R/include -Itests/gpu -I. -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib 2>/dev/null; }
VK=$(strings "$V" | grep -m1 -E '^_ZN|fmha|fwd' | head -1)
run() {  # $1 = ours|vendor, $2 = kernel regex
    env REF_DATA=${REF_DATA:-normal} REF_B=$B REF_H=$H REF_S=$S REF_SK=$S REF_HK=$H REF_ROWS=1 REF_WARMUP=4 REF_ITERS=2 REF_TIME_ONLY=$1 \
        REF_CO2="$O/k.co" REF_KNAME2="$K" REF_ROWS_PER_BLOCK=${ROWS_PER_BLOCK:-256} ${OURS_BLOCK_THREADS:+REF_OURS_BLOCK_THREADS=$OURS_BLOCK_THREADS} \
        $R/bin/rocprofv3 --att --att-library-path "$DEC" --att-target-cu 1 --att-simd-select ${ATT_SIMD:-0x1} --kernel-include-regex "$2" \
        --kernel-iteration-range "[5]" -d "$O/$1" -o run -- ./build/ff/ref_host "$V" >"$O/$1.log" 2>&1
}
run ours "$K"
run vendor "${VENDOR_REGEX:-fmha_fwd}"
for who in ours vendor; do
    st=$(find "$O/$who" -name "stats*.csv" | head -1); ui=$(dirname "$(find "$O/$who" -name "code.json" | head -1)")
    echo "######## $who ($st)"
    [ -n "$st" ] && python3 tools/att_classes.py "$st" | head -24
    [ -n "$ui" ] && [ "$ui" != "." ] && python3 tools/att_gaps.py "$ui" 12 | head -30
done
