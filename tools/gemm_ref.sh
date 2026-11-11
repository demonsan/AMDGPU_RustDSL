#!/usr/bin/env bash
# MXFP8 blockscale GEMM reference: runs the aiter PR #6696 4-wave pin-AGPR code object (and optionally ours, OURS_CO/OURS_KNAME) on identical
# inputs, checks sampled rows against a double-precision CPU reference and times both (median of REF_REPS).
# Usage: HIP_VISIBLE_DEVICES=0 [GEMM_M=4096 GEMM_N=4096 GEMM_K=8192] [REF_VARIANT=pm0_sp64_u2_gm16_wide1_vf1] bash tools/gemm_ref.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
CO_DIR="${GEMM_CO_DIR:-/shared/amdgpu/home/blyu_qle/sessions/aiter_pr6696_co}"
VAR="${REF_VARIANT:-pm0_sp64_u2_gm16_wide1_vf1}"
IMG="$CO_DIR/opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_${VAR}.co"
[ -f "$IMG" ] || { echo "missing $IMG (copy csrc/opus_gemm/gen_co/gfx950/*.co of the PR there)"; exit 1; }
# the code-object symbol carries the default sn/sl/sched fields: read it from the ELF
SYM="${REF_KNAME:-$(strings -a "$IMG" | grep -m1 -E '^opus_gemm_gfx950_4wave_pin_agpr_co_.*_vf[01]$')}"
[ -x build/gemm_ref_host ] && [ build/gemm_ref_host -nt tests/gpu/gemm_ref_host.cpp ] || \
    g++ -O3 -march=native -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/gemm_ref_host.cpp -o build/gemm_ref_host \
        -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
echo "aiter co: $(basename "$IMG")  symbol: $SYM"
# ours: OURS_KERNEL=<amdc kernel name> (OURS_TILE = its CTA tile, default 128)
OURS_ENV=()
if [ -n "${OURS_KERNEL:-}" ]; then
    rm -f "build/$OURS_KERNEL.s"
    bash tools/container/cargo.sh run -q --bin amdc -- emit "$OURS_KERNEL" ${AMDC_EXTRA_FLAGS:-} --out "/work/build/$OURS_KERNEL.s" 2>&1 | tail -3
    [ -f "build/$OURS_KERNEL.s" ] || exit 1
    "${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 \
        -o "build/$OURS_KERNEL.co" "build/$OURS_KERNEL.s" || exit 1
    OURS_ENV=(OURS_CO="build/$OURS_KERNEL.co" OURS_KNAME="$OURS_KERNEL" OURS_TILE_M="${OURS_TILE_M:-128}" OURS_TILE_N="${OURS_TILE_N:-128}" OURS_DUMP="${OURS_DUMP:-}")
fi
[ -n "${SKIP_REF:-}" ] && IMG=""
env REF_CO="$IMG" REF_KNAME="$SYM" "${OURS_ENV[@]}" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/gemm_ref_host
