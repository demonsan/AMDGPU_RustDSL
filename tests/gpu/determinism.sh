#!/usr/bin/env bash
# Determinism check: every shape is run 3 times and the dumped bf16 output must be bit-identical.
# Usage: REF_CO=/path/vendor.co bash tests/gpu/determinism.sh   (OUR_KERNEL, ROCM_PATH, HIP_VISIBLE_DEVICES optional)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
REF_CO="${REF_CO:?set REF_CO to the vendor fwd_hd128_mxfp8.co}"
OUR_KERNEL="${OUR_KERNEL:-fmha_fwd_hd128_mxfp8_t}"
cd "$REPO"; mkdir -p build
build_one() { # kernel, out.co
    bash tools/container/cargo.sh run -q --bin amdc -- emit "$1" --backend "${BACKEND:-auto}" ${AMDC_EXTRA_FLAGS:-} --out "/work/build/$1.s" >/dev/null 2>&1
    "${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 \
        -o "$2" "build/$1.s"
}
# OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto: _vs128 + _t128, ref_host picks the tiling per shape (see run_ref.sh)
KNAME="$OUR_KERNEL"; SMALL_ENV=""; SPLIT_ENV=""; COMBINE_ENV=""; SPLIT_N="0"; SPLIT_KNAME=""; COMBINE_KNAME=""
if [[ "$OUR_KERNEL" == *_auto_split ]]; then
    # KV-split auto path: build _vs128 + _t128 + _kvsplit + _kvsplit_combine; ref_host applies the
    # dispatch guard (docs/gen4-tiling-design.md section 11) and falls back to _t128/_vs128 when rejected.
    P="${OUR_KERNEL%_auto_split}"
    KNAME="${P}_vs128"
    build_one "$KNAME" build/fmha_ours.co
    build_one "${P}_t128" build/fmha_ours_small.co
    build_one "${P}_kvsplit" build/fmha_ours_split.co
    build_one "${P}_kvsplit_combine4h" build/fmha_combine.co
    SMALL_ENV="build/fmha_ours_small.co"
    SPLIT_ENV="build/fmha_ours_split.co"
    COMBINE_ENV="build/fmha_combine.co"
    SPLIT_N="${REF_SPLIT:-2}"
    SPLIT_KNAME="${P}_kvsplit"
    COMBINE_KNAME="${P}_kvsplit_combine4h"
elif [[ "$OUR_KERNEL" == *_auto ]]; then
    # default auto dispatch: _vs128 / _t128 / KV-split (_kvsplit + _kvsplit_combine4h); ref_host applies the guard per shape (REF_SPLIT=s forces a
    # split candidate, REF_NO_AUTO_SPLIT=1 disables the split, REF_ROWS_PER_BLOCK overrides the tiling)
    P="${OUR_KERNEL%_auto}"
    KNAME="${P}_vs128"
    build_one "$KNAME" build/fmha_ours.co
    build_one "${P}_t128" build/fmha_ours_small.co
    build_one "${P}_kvsplit" build/fmha_ours_split.co
    build_one "${P}_kvsplit_combine4h" build/fmha_combine.co
    SMALL_ENV="build/fmha_ours_small.co"
    SPLIT_ENV="build/fmha_ours_split.co"
    COMBINE_ENV="build/fmha_combine.co"
    SPLIT_N="${REF_SPLIT:-0}"
    SPLIT_KNAME="${P}_kvsplit"
    COMBINE_KNAME="${P}_kvsplit_combine4h"
else
    build_one "$OUR_KERNEL" build/fmha_ours.co
fi
[ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host \
    -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
rc=0
for CFG in "2 8 700 640 2" "1 2 200 1337 2" "1 4 300 512 1" "2 4 300 129 1" "1 8 2048 2048 1"; do
    set -- $CFG
    sums=""
    for run in 1 2 3; do
        REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$4 REF_HK=$(( $2 / $5 )) REF_ROWS=4 REF_ITERS=5 REF_DUMP2=build/det_$run.bin \
            REF_CO2=build/fmha_ours.co REF_KNAME2="$KNAME" REF_CO2_SMALL="$SMALL_ENV" \
            REF_CO2_SPLIT="$SPLIT_ENV" REF_KNAME2_SPLIT="$SPLIT_KNAME" REF_CO2_COMBINE="$COMBINE_ENV" REF_KNAME2_COMBINE="$COMBINE_KNAME" REF_SPLIT="$SPLIT_N" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
            ./build/ref_host "$REF_CO" >/dev/null 2>&1
        sums="$sums $(md5sum build/det_$run.bin | cut -c1-12)"
    done
    n=$(echo $sums | tr ' ' '\n' | sort -u | wc -l)
    if [ "$n" = 1 ]; then echo "DET_OK   B=$1 H=$2 S=$3 SK=$4 gqa=$5 :$sums"; else echo "DET_FAIL B=$1 H=$2 S=$3 SK=$4 gqa=$5 :$sums"; rc=1; fi
done
exit $rc