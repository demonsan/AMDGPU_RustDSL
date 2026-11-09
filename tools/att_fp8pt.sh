#!/usr/bin/env bash
# Warm ATT (rocprofv3 thread trace, CU 1 SIMD 0) of the per-tensor FP8 prefill (ours) and aiter's fwd_hd128_fp8[_causal].co on the same inputs
# (tests/gpu/fp8pt_host.cpp, SCALAR_DS=1), one dispatch each (the 5th: warm), then the per-class table (tools/att_classes.py) and the MFMA-gap
# attribution (tools/att_gaps.py) of each - as tools/att_fmha.sh does for the MXFP8 kernels.
# usage (repo root, GPU node, amdc built): HIP_VISIBLE_DEVICES=5 bash tools/att_fp8pt.sh <tag> [B H S causal]
set -u
TAG=${1:-att}; B=${2:-2}; H=${3:-32}; S=${4:-8192}; C=${5:-0}
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
DEC=${ATT_DECODER_DIR:-/shared/apps/ubuntu/opt/rocm-10.1.0/lib}
AMDC=${AMDC:-/tmp/target-kfmha/release/amdc}
SFX=""; [ "$C" = 1 ] && SFX=_causal
K=fmha_fwd_hd128_fp8pt_v3$SFX
V=${AITER_HSA:-/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950}/fmha_v3_fwd/fwd_hd128_fp8$SFX.co
VK=$(strings "$V" | grep -m1 '^_ZN')
O=build/att/$TAG; rm -rf "$O"; mkdir -p "$O"
$AMDC emit "$K" --backend own --out "$O/k.s" >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$O/k.co" "$O/k.s" || exit 1
$R/bin/hipcc -O2 -std=c++17 tests/gpu/fp8pt_host.cpp -o "$O/fp8pt_host" || exit 1
run() {  # $1 = ours|vendor, $2 = kernel regex
    env SCALAR_DS=1 CHECK=1 ITERS=3 WARMUP=2 ROUNDS=1 timeout -s KILL 600 \
        $R/bin/rocprofv3 --att --att-library-path "$DEC" --att-target-cu 1 --att-simd-select 0x1 --kernel-include-regex "$2" \
        --kernel-iteration-range "[5]" -d "$O/$1" -o run -- "$O/fp8pt_host" "$V" "$VK" "$O/k.co" "$K" $B $H $H $S $S $C >"$O/$1.log" 2>&1
}
run ours "$K"
run vendor "fp8(_causal)?_gfx950"
for who in ours vendor; do
    st=$(find "$O/$who" -name "stats*.csv" | head -1); ui=$(dirname "$(find "$O/$who" -name "code.json" | head -1)")
    echo "######## $who ($st)"
    [ -n "$st" ] && python3 tools/att_classes.py "$st" | head -24
    [ -n "$ui" ] && [ "$ui" != "." ] && python3 tools/att_gaps.py "$ui" 12 | head -30
done
