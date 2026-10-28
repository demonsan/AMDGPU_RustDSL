#!/usr/bin/env bash
# Determinism check: every shape is run 3 times and the dumped bf16 output must be bit-identical.
# Usage: REF_CO=/path/vendor.co bash tests/gpu/determinism.sh   (OUR_KERNEL, ROCM_PATH, HIP_VISIBLE_DEVICES optional)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
REF_CO="${REF_CO:?set REF_CO to the vendor fwd_hd128_mxfp8.co}"
OUR_KERNEL="${OUR_KERNEL:-fmha_fwd_hd128_mxfp8_t}"
cd "$REPO"; mkdir -p build
bash tools/container/cargo.sh run -q --bin amdc -- emit "$OUR_KERNEL" --out "/work/build/$OUR_KERNEL.s" >/dev/null 2>&1
"$ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 \
    -o build/fmha_ours.co "build/$OUR_KERNEL.s"
[ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host \
    -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
rc=0
for CFG in "2 8 700 640 2" "1 2 200 1337 2" "1 4 300 512 1" "2 4 300 129 1" "1 8 2048 2048 1"; do
    set -- $CFG
    sums=""
    for run in 1 2 3; do
        REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$4 REF_HK=$(( $2 / $5 )) REF_ROWS=4 REF_ITERS=5 REF_DUMP2=build/det_$run.bin \
            REF_CO2=build/fmha_ours.co REF_KNAME2="$OUR_KERNEL" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
            ./build/ref_host "$REF_CO" >/dev/null 2>&1
        sums="$sums $(md5sum build/det_$run.bin | cut -c1-12)"
    done
    n=$(echo $sums | tr ' ' '\n' | sort -u | wc -l)
    if [ "$n" = 1 ]; then echo "DET_OK   B=$1 H=$2 S=$3 SK=$4 gqa=$5 :$sums"; else echo "DET_FAIL B=$1 H=$2 S=$3 SK=$4 gqa=$5 :$sums"; rc=1; fi
done
exit $rc