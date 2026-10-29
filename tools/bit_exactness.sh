#!/usr/bin/env bash
# OWN-backend vs LLVM-tail *bit-exactness* on hardware, for every kernel the repo can drive.
#
# For each kernel: emit + assemble the same lowered MIR through both tails, run the existing
# per-kernel harness on the same deterministic inputs, and compare the outputs bit-for-bit:
#   fmha  : the 15 run_ref.sh shapes, full BF16 output dump (ref_host REF_DUMP2, cmp)
#   sched : sched_host output hash (FNV of the first 4096 dwords)
#   probes: attn_host / probe_host printed results (PASS/FAIL plus bad/max_abs_err numbers)
# Any DIFF means the two tails do not produce the same values -> red flag (the own backend can be
# silently corrupted; see docs/verification.md).
#
# Usage: HIP_VISIBLE_DEVICES=5 bash tools/bit_exactness.sh [fmha|sched|probes|all]
# Env:   REF_CO      vendor .co (default: the session copy)
#        BX_FILTER   only kernels whose name contains this string
#        BX_ROWS     fmha CPU-check rows (default 2; the dump is always complete)
#        BX_WORK     work dir (default build/bx)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
ASM="${ASM_ROC:-$ROC}"
CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
GPU="${HIP_VISIBLE_DEVICES:-5}"
WORK="${BX_WORK:-build/bx}"
WHICH="${1:-all}"
cd "$REPO"; mkdir -p "$WORK"
DIFFS=0

emit_both() { # $1 = kernel name
    local K="$1" BK
    for BK in own llvm-tail; do
        bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" --backend "$BK" \
            ${AMDC_EXTRA_FLAGS:-} --out "/work/$WORK/$K.$BK.s" >/dev/null 2>&1 ||
            { echo "BX_EMIT_FAIL $K $BK"; return 1; }
        "$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
            -mcode-object-version=6 -o "$WORK/$K.$BK.co" "$WORK/$K.$BK.s" 2>/dev/null ||
            { echo "BX_ASM_FAIL $K $BK"; return 1; }
    done
    return 0
}

skip() { [[ -n "${BX_FILTER:-}" && "$1" != *"$BX_FILTER"* ]]; }

# ---------------------------------------------------------------------------------------------
# FMHA kernels: full output dumps on all 15 reference shapes (same shapes as tests/gpu/run_ref.sh)
# ---------------------------------------------------------------------------------------------
SHAPES=("1 1 256 256 1" "1 2 512 512 1" "2 4 1024 1024 1" "1 8 2048 2048 1" "1 2 512 2048 1"
        "3 3 768 1280 1" "1 4 300 512 1" "2 8 700 640 2" "1 16 256 1024 4" "1 2 256 1000 1"
        "2 4 300 129 1" "1 8 512 100 2" "1 2 1024 1088 1" "1 4 77 1 1" "1 2 200 1337 2")

fmha() {
    [ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp \
        -o build/ref_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || return 1
    local K cfg b h s sk g
    for K in fmha_fwd_hd128_mxfp8_t fmha_fwd_hd128_mxfp8_sv fmha_fwd_hd128_mxfp8_pp fmha_fwd_hd128_mxfp8; do
        skip "$K" && continue
        emit_both "$K" || { DIFFS=$((DIFFS + 1)); continue; }
        for cfg in "${SHAPES[@]}"; do
            set -- $cfg; b=$1; h=$2; s=$3; sk=$4; g=$5
            local tag="$b-$h-$s-$sk-$g" bk
            for bk in own llvm-tail; do
                REF_B=$b REF_H=$h REF_S=$s REF_SK=$sk REF_HK=$((h / g)) REF_ROWS="${BX_ROWS:-2}" \
                    REF_ITERS=0 REF_CO2="$WORK/$K.$bk.co" REF_KNAME2="$K" \
                    REF_DUMP2="$WORK/$K.$tag.$bk.bin" HIP_VISIBLE_DEVICES=$GPU \
                    ./build/ref_host "$CO" >/dev/null 2>&1
                [ -s "$WORK/$K.$tag.$bk.bin" ] || { echo "BX_FMHA_NOOUT $K B=$b H=$h S=$s SK=$sk gqa=$g $bk"; DIFFS=$((DIFFS + 1)); }
            done
            if cmp -s "$WORK/$K.$tag.own.bin" "$WORK/$K.$tag.llvm-tail.bin"; then
                echo "BX_FMHA_OK $K B=$b H=$h S=$s SK=$sk gqa=$g md5=$(md5sum "$WORK/$K.$tag.own.bin" | cut -c1-12)"
            else
                echo "BX_FMHA_DIFF $K B=$b H=$h S=$s SK=$sk gqa=$g"
                DIFFS=$((DIFFS + 1))
            fi
        done
    done
}

# ---------------------------------------------------------------------------------------------
# sched_* microkernels: output hash from sched_host (deterministic per-thread results)
# ---------------------------------------------------------------------------------------------
SCHED_K=(sched_valu_mul sched_valu_max3 sched_valu_exp sched_mfma_chain sched_mfma_valu16
         sched_mfma_valu4 sched_diamond sched_occ_chain)

sched() {
    [ -x build/sched_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/sched_host.cpp \
        -o build/sched_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || return 1
    local K BLK G N cfg ho hl
    for K in "${SCHED_K[@]}"; do
        skip "$K" && continue
        emit_both "$K" || { DIFFS=$((DIFFS + 1)); continue; }
        for cfg in "256 1 2000" "512 1 2000" "256 256 2000"; do
            set -- $cfg; BLK=$1; G=$2; N=$3
            ho=$(HIP_VISIBLE_DEVICES=$GPU ./build/sched_host "$WORK/$K.own.co" "$K" "$BLK" "$G" "$N" 10 2>&1 | sed -n 's/.*hash=//p')
            hl=$(HIP_VISIBLE_DEVICES=$GPU ./build/sched_host "$WORK/$K.llvm-tail.co" "$K" "$BLK" "$G" "$N" 10 2>&1 | sed -n 's/.*hash=//p')
            if [ -n "$ho" ] && [ "$ho" = "$hl" ]; then
                echo "BX_SCHED_OK $K block=$BLK grid=$G hash=$ho"
            else
                echo "BX_SCHED_DIFF $K block=$BLK grid=$G own='$ho' llvm='$hl'"
                DIFFS=$((DIFFS + 1))
            fi
        done
    done
}

# ---------------------------------------------------------------------------------------------
# probe kernels driven by attn_host (same launch parameters for both tails)
# ---------------------------------------------------------------------------------------------
PROBES=(attention_fwd probe_math probe_mfma probe_lds probe_mfma_wide probe_mfma_fp8
        probe_mfma_scale probe_roles probe_waves2 probe_lds_self2 probe_roles_loop
        probe_fmha_datapath probe_ab_loops probe_nested_acc probe_loop_mfma probe_fmha_core
        probe_pack probe_fp8_matmul probe_fp8_load probe_fp8_min probe_fp8_minload
        probe_fp8_softmax probe_fp8_fmha_tile probe_fp8_fmha_2tile probe_fp8_fmha_online
        probe_fp8_fmha_oreg probe_fp8_fmha_multiwave probe_fp8_fmha_pingpong probe_fp8_fmha_abloops
        probe_lds_tr_b8 probe_fp8_matmul_tr probe_fp8_fmha_tr probe_fp8_fmha_trab
        probe_fp8_fmha_hd128 probe_fp8_fmha_hd128b probe_fp8_fmha_hd128c probe_fp8_matmul_blk)

status_of() { # $1 = filtered output file
    if grep -qE "PROBE_[A-Z0-9_]*_OK|ATTENTION_FWD OK" "$1"; then echo OK
    elif grep -qE "PROBE_[A-Z0-9_]*_FAIL|ATTENTION_FWD FAIL|HIP error|failed" "$1"; then echo FAIL
    else echo OTHER; fi
}

probes() {
    [ -x build/attn_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/attn_host.cpp \
        -o build/attn_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || return 1
    local K bk st_own st_ll
    for K in "${PROBES[@]}"; do
        skip "$K" && continue
        emit_both "$K" || { DIFFS=$((DIFFS + 1)); continue; }
        for bk in own llvm-tail; do
            ATTN_KERNEL="$K" ATTN_BLOCK="${BX_BLOCK:-64}" HIP_VISIBLE_DEVICES=$GPU \
                ./build/attn_host "$WORK/$K.$bk.co" >"$WORK/$K.$bk.out" 2>&1
            grep -E "PROBE_|ATTENTION_FWD|bad=|max_abs_err|mean_abs_err|rows_with_bad|HIP error|failed" \
                "$WORK/$K.$bk.out" >"$WORK/$K.$bk.filt"
        done
        st_own=$(status_of "$WORK/$K.own.filt")
        st_ll=$(status_of "$WORK/$K.llvm-tail.filt")
        if cmp -s "$WORK/$K.own.filt" "$WORK/$K.llvm-tail.filt"; then
            echo "BX_PROBE_SAME $K own=$st_own llvm=$st_ll"
        else
            echo "BX_PROBE_DIFF $K own=$st_own llvm=$st_ll"
            DIFFS=$((DIFFS + 1))
        fi
    done
    # probe_store has its own minimal ABI harness
    if ! skip probe_store; then
        if emit_both probe_store; then
            [ -x build/probe_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/probe_host.cpp \
                -o build/probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || return 1
            for bk in own llvm-tail; do
                HIP_VISIBLE_DEVICES=$GPU ./build/probe_host "$WORK/probe_store.$bk.co" >"$WORK/probe_store.$bk.out" 2>&1
            done
            if cmp -s "$WORK/probe_store.own.out" "$WORK/probe_store.llvm-tail.out"; then
                echo "BX_PROBE_SAME probe_store $(grep -oE 'PROBE_(OK|FAIL)' "$WORK/probe_store.own.out" | head -1)"
            else
                echo "BX_PROBE_DIFF probe_store"; DIFFS=$((DIFFS + 1))
            fi
        else
            DIFFS=$((DIFFS + 1))
        fi
    fi
}

case "$WHICH" in
    fmha) fmha ;;
    sched) sched ;;
    probes) probes ;;
    all) fmha; sched; probes ;;
    *) echo "usage: $0 [fmha|sched|probes|all]"; exit 2 ;;
esac
echo "BX_SUMMARY diffs=$DIFFS"
[ "$DIFFS" = 0 ]
