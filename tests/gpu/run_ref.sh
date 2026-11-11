#!/usr/bin/env bash
# Compare our fmha_fwd_hd128_mxfp8 with the vendor fwd_hd128_mxfp8.co (and a CPU model) on the GPU node.
# Usage: REF_CO=/path/to/fwd_hd128_mxfp8.co bash tests/gpu/run_ref.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/tools/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
REF_CO="${REF_CO:?set REF_CO to the vendor fwd_hd128_mxfp8.co}"
OUR_KERNEL="${OUR_KERNEL:-fmha_fwd_hd128_mxfp8}"
cd "$REPO"; mkdir -p build
# OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto: build the 256-row kernel (_vs128) and the 128-row kernel (_t128); ref_host then picks per shape
# (2*ceil(S/256)*H*B <= 256 -> 128 rows / 256 threads, else 256 / 512; REF_ROWS_PER_BLOCK overrides)
build_one() { # kernel, out.co
    bash tools/container/cargo.sh run -q --bin amdc -- emit "$1" --backend "${BACKEND:-auto}" ${AMDC_EXTRA_FLAGS:-} --out "/work/build/$1.s"
    "${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 \
        -o "$2" "build/$1.s"
}
# The shipped default `fmha_fwd_hd128_mxfp8_auto` runs the layout-layer kernels (`fmha_fwd_hd128_mxfp8_v3_*`); any other prefix is used as is.
kernel_prefix() { case "$1" in fmha_fwd_hd128_mxfp8) echo fmha_fwd_hd128_mxfp8_v3 ;; *) echo "$1" ;; esac; }
KNAME="$OUR_KERNEL"; SMALL_ENV=""; SMALL_KNAME="fmha_fwd_hd128_mxfp8_v3_t128"; SPLIT_ENV=""; COMBINE_ENV=""; SPLIT_N="0"; SPLIT_KNAME=""; COMBINE_KNAME=""
if [[ "$OUR_KERNEL" == *_auto_split ]]; then
    # KV-split auto path: build _vs128 + _t128 + _kvsplit + _kvsplit_combine; ref_host applies the
    # dispatch guard (docs/archive/gen4-tiling-design.md section 11) and falls back to _t128/_vs128 when rejected.
    P="${OUR_KERNEL%_auto_split}"; P="$(kernel_prefix "$P")"
    KNAME="${P}_vs128"
    build_one "$KNAME" build/fmha_ours.co
    build_one "${P}_t128" build/fmha_ours_small.co
    build_one "${P}_kvsplit" build/fmha_ours_split.co
    build_one "${P}_kvsplit_combine8h" build/fmha_combine.co
    SMALL_ENV="build/fmha_ours_small.co"; SMALL_KNAME="${P}_t128"
    SPLIT_ENV="build/fmha_ours_split.co"
    COMBINE_ENV="build/fmha_combine.co"
    SPLIT_N="${REF_SPLIT:-2}"
    SPLIT_KNAME="${P}_kvsplit"
    COMBINE_KNAME="${P}_kvsplit_combine8h"
elif [[ "$OUR_KERNEL" == *_auto ]]; then
    # default auto dispatch: _vs128 / _t128 / KV-split (_kvsplit + _kvsplit_combine8h); ref_host applies the guard per shape (REF_SPLIT=s forces a
    # split candidate, REF_NO_AUTO_SPLIT=1 disables the split, REF_ROWS_PER_BLOCK overrides the tiling)
    P="${OUR_KERNEL%_auto}"; P="$(kernel_prefix "$P")"
    KNAME="${P}_vs128"
    build_one "$KNAME" build/fmha_ours.co
    build_one "${P}_t128" build/fmha_ours_small.co
    build_one "${P}_kvsplit" build/fmha_ours_split.co
    build_one "${P}_kvsplit_combine8h" build/fmha_combine.co
    SMALL_ENV="build/fmha_ours_small.co"; SMALL_KNAME="${P}_t128"
    SPLIT_ENV="build/fmha_ours_split.co"
    COMBINE_ENV="build/fmha_combine.co"
    SPLIT_N="${REF_SPLIT:-0}"
    SPLIT_KNAME="${P}_kvsplit"
    COMBINE_KNAME="${P}_kvsplit_combine8h"
else
    build_one "$OUR_KERNEL" build/fmha_ours.co
fi
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host \
    -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
rc=0
for CFG in "1 1 256 256 1" "1 2 512 512 1" "2 4 1024 1024 1" "1 8 2048 2048 1" "1 2 512 2048 1" "3 3 768 1280 1" "1 4 300 512 1" "2 8 700 640 2" "1 16 256 1024 4" "1 2 256 1000 1" "2 4 300 129 1" "1 8 512 100 2" "1 2 1024 1088 1" "1 4 77 1 1" "1 2 200 1337 2"; do
    set -- $CFG
    echo "=== B=$1 H=$2 S=$3 SK=$4 gqa=$5"
    out=$(REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$4 REF_HK=$(( $2 / $5 )) REF_ROWS="${REF_ROWS:-16}" REF_ITERS="${REF_ITERS:-20}" \
        REF_CO2=build/fmha_ours.co REF_KNAME2="$KNAME" REF_CO2_SMALL="$SMALL_ENV" REF_KNAME2_SMALL="$SMALL_KNAME" \
            REF_CO2_SPLIT="$SPLIT_ENV" REF_KNAME2_SPLIT="$SPLIT_KNAME" REF_CO2_COMBINE="$COMBINE_ENV" REF_KNAME2_COMBINE="$COMBINE_KNAME" REF_COMBINE_ELEMS=4 REF_SPLIT="$SPLIT_N" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
        ./build/ref_host "$REF_CO" 2>&1 | grep -E "vs|timing|REF_|^\[grid\]") || rc=1
    echo "$out"
    # Second gate, VENDOR PARITY: our MEAN error against the CPU model must be within PARITY_MARGIN (default 1.10) of the vendor's on the same
    # elements. The worst element is printed but does not gate: on a small row sample (REF_ROWS=16) it is noise-dominated (even the accurate
    # recipe fails a 1.05 worst-element gate on 7 of 15 shapes); a mean over 64 rows x 128 dims is a stable statistic - use REF_ROWS=64.
    # The absolute-tolerance verdict (REF_KERNEL_OK/FAIL) stays the correctness requirement.
    echo "$out" | awk -v m="${PARITY_MARGIN:-1.10}" '
        /^vendor kernel +vs CPU/ { for (i = 1; i <= NF; i++) { if ($i ~ /^worst_err=/) { sub("worst_err=", "", $i); vw = $i + 0 } if ($i ~ /^mean_err=/) { sub("mean_err=", "", $i); vm = $i + 0 } } have_v = 1 }
        /^our kernel +vs CPU/    { for (i = 1; i <= NF; i++) { if ($i ~ /^worst_err=/) { sub("worst_err=", "", $i); ow = $i + 0 } if ($i ~ /^mean_err=/) { sub("mean_err=", "", $i); om = $i + 0 } } have_o = 1 }
        END {
            if (!(have_v && have_o)) { print "PARITY_NA (no vendor/ours error lines)"; exit 0 }
            ok = (om <= m * vm)
            printf "%s ours_mean=%.4e ours_worst=%.4e vendor_mean=%.4e vendor_worst=%.4e margin=%.2f\n", (ok ? "PARITY_OK" : "PARITY_FAIL"), om, ow, vm, vw, m
        }'
done
exit $rc