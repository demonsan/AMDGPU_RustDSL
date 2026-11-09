#!/usr/bin/env bash
# Warm ATT (rocprofv3 thread trace, CU 1 SIMD 0) of one of our bf16 prefill kernels and the matching aiter fmha_v3 kernel on the same shape
# (tests/gpu/attn_host.cpp, one process each), the 5th dispatch, then the per-class table (tools/att_classes.py) and the MFMA-gap
# attribution (tools/att_gaps.py) of each - as tools/att_fmha.sh / att_fp8pt.sh.
# usage (repo root, GPU node, amdc built): HIP_VISIBLE_DEVICES=5 bash tools/att_attn.sh <tag> <our kernel> <vendor .co (fmha_v3_fwd/...)> B H HK SQ SK causal [D]
#   env: OURS_ENV / VENDOR_ENV = attn_host env for each side (e.g. OURS_ENV="ROWS=128 BLOCK=256 PAIR=0", VENDOR_ENV="ROWS=128 BLOCK=256"); DV
set -u
TAG=$1; K=$2; VCO=$3; shift 3
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
DEC=${ATT_DECODER_DIR:-/shared/apps/ubuntu/opt/rocm-10.1.0/lib}
AMDC=${AMDC:-/tmp/target-kfmha/release/amdc}
V=${AITER_HSA:-/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950}/$VCO
VK=$(strings "$V" | grep -m1 '^_ZN')
O=build/att/$TAG; rm -rf "$O"; mkdir -p "$O"
$AMDC emit "$K" --backend own --out "$O/k.s" >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$O/k.co" "$O/k.s" || exit 1
$R/bin/hipcc -O2 -std=c++17 tests/gpu/attn_host.cpp -o "$O/attn_host" || exit 1
run() {  # ours|vendor, co, kernel, env, then the attn_host shape arguments
    local who=$1 co=$2 kn=$3 envs=$4; shift 4
    # shellcheck disable=SC2086
    env LSE=0 CHECK=1 ITERS=3 WARMUP=2 $envs timeout -s KILL 600 \
        $R/bin/rocprofv3 --att --att-library-path "$DEC" --att-target-cu 1 --att-simd-select 0x1 --kernel-include-regex "$kn" \
        --kernel-iteration-range "[5]" -d "$O/$who" -o run -- "$O/attn_host" "$who" "$co" "$kn" "$@" >"$O/$who.log" 2>&1
}
ARGS=("$@")
run ours "$O/k.co" "$K" "${OURS_ENV:-}" "${ARGS[@]}"
run vendor "$V" "$VK" "${VENDOR_ENV:-}" "${ARGS[@]}"
for who in ours vendor; do
    st=$(find "$O/$who" -name "stats*.csv" | head -1); ui=$(dirname "$(find "$O/$who" -name "code.json" | head -1)")
    echo "######## $who ($st)"
    [ -n "$st" ] && python3 tools/att_classes.py "$st" | head -24
    [ -n "$ui" ] && [ "$ui" != "." ] && python3 tools/att_gaps.py "$ui" 12 | head -30
done
