#!/usr/bin/env bash
# Same-run perf sweep vs the vendor (docs/perf-matrix.md): FMHA fwd hd128 mxfp8 (shipped `_auto` set vs ref_mxfp8.co) and MXFP8 blockscale GEMM v4 (vs the aiter 4wave .co).
# usage (repo clone on the GPU node, amdc built): HIP_VISIBLE_DEVICES=4 bash tools/perf_matrix.sh [fmha|gemm|all]     env: REPS (5), AMDC, REF_CO, GEMM_CO_DIR
# Raw output: build/pm/raw_fmha.txt, build/pm/raw_gemm.txt (one line per run); tools/perf_matrix_md.py turns them into the markdown tables.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build/pm
AMDC="${AMDC:-$(ls target/debug/amdc 2>/dev/null || echo amdc)}"; R="${ROCM_RT:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
REPS="${REPS:-5}"; export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}"; WHAT="${1:-all}"
asm() { "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$2" "$1" || exit 1; }

if [ "$WHAT" = fmha ] || [ "$WHAT" = all ]; then
  REF_CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"; P=fmha_fwd_hd128_mxfp8_v3
  for k in vs128 t128 kvsplit kvsplit_combine8h; do "$AMDC" emit ${P}_$k --backend auto --out build/pm/$k.s >/dev/null || exit 1; asm build/pm/$k.s build/pm/$k.co; done
  g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/pm/ref_host -I"$R/include" -L"$R/lib" -lamdhip64 -Wl,-rpath,"$R/lib" || exit 1
  rh() { # B H S SK gqa [extra env...]
    env REF_B=$1 REF_H=$2 REF_S=$3 REF_SK=$4 REF_HK=$(( $2 / $5 )) REF_COMBINE_ELEMS=4 REF_CO2=build/pm/vs128.co REF_KNAME2=${P}_vs128 REF_CO2_SMALL=build/pm/t128.co REF_KNAME2_SMALL=${P}_t128 \
        REF_CO2_SPLIT=build/pm/kvsplit.co REF_KNAME2_SPLIT=${P}_kvsplit REF_CO2_COMBINE=build/pm/kvsplit_combine8h.co REF_KNAME2_COMBINE=${P}_kvsplit_combine8h REF_SPLIT=0 "${@:6}" ./build/pm/ref_host "$REF_CO" 2>&1; }
  SHAPES=(); for B in 1 2 4 8; do for S in 512 1024 2048 4096 8192; do SHAPES+=("$B 8 $S $S 1"); done; done
  SHAPES+=("1 8 2048 2048 2" "1 8 2048 2048 4" "1 8 1000 1000 1" "1 8 2048 1337 1" "1 8 1000 1337 1")
  : > build/pm/raw_fmha.txt
  for rep in $(seq 1 $REPS); do
    for cfg in "${SHAPES[@]}"; do set -- $cfg
      out=$(rh $1 $2 $3 $4 $5 REF_ROWS=1 REF_WARMUP=150 REF_ITERS=100 REF_SPLIT_ULP=0)
      ours=$(echo "$out" | awk '/^timing ours total:/ {for (i = 1; i <= NF; i++) if ($i == "TFLOP/s") print $(i-1)} /^timing ours +:/ {for (i = 1; i <= NF; i++) if ($i ~ /^TFLOP\/s/) print $(i-1)}' | tail -1)
      vend=$(echo "$out" | awk '/^timing vendor/ {for (i = 1; i <= NF; i++) if ($i ~ /^TFLOP\/s/) print $(i-1)}' | head -1)
      path=$(echo "$out" | sed -nE 's/^\[grid\].*ours \(([^)]*)\).*/\1/p' | head -1 | sed "s/${P}_//")
      echo "FMHA $1 $2 $3 $4 $5 rep$rep ours=${ours:-0} vendor=${vend:-0} path=${path:-?}" >> build/pm/raw_fmha.txt
    done
  done
  for cfg in "${SHAPES[@]}"; do set -- $cfg   # 64-row accuracy vs the CPU model (one run per shape)
    out=$(rh $1 $2 $3 $4 $5 REF_ROWS=64 REF_ITERS=0)
    v=$(echo "$out" | sed -nE 's/^vendor kernel +vs CPU: checked=([0-9]+) bad=([0-9]+) worst_err=([^ ]+) mean_err=([^ ]+).*/bad=\2 worst=\3 mean=\4/p' | head -1)
    o=$(echo "$out" | sed -nE 's/^our kernel +vs CPU: checked=([0-9]+) bad=([0-9]+) worst_err=([^ ]+) mean_err=([^ ]+).*/bad=\2 worst=\3 mean=\4/p' | head -1)
    ok=$(echo "$out" | grep -oE 'REF_KERNEL_(OK|FAIL)' | head -1)
    echo "ACC $1 $2 $3 $4 $5 ours: ${o:-NA} vendor: ${v:-NA} $ok" >> build/pm/raw_fmha.txt
  done
fi

if [ "$WHAT" = gemm ] || [ "$WHAT" = all ]; then
  CO_DIR="${GEMM_CO_DIR:-/shared/amdgpu/home/blyu_qle/sessions/aiter_pr6696_co}"; VAR="${REF_VARIANT:-pm0_sp64_u2_gm16_wide1_vf1}"
  IMG="$CO_DIR/opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_${VAR}.co"; SYM=$(strings -a "$IMG" | grep -m1 -E '^opus_gemm_gfx950_4wave_pin_agpr_co_.*_vf[01]$')
  "$AMDC" emit gemm_mxfp8_bs_v4 --backend=own --out build/pm/gemm_v4.s >/dev/null || exit 1; asm build/pm/gemm_v4.s build/pm/gemm_v4.co
  g++ -O3 -march=native -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/gemm_ref_host.cpp -o build/pm/gemm_ref_host -I"$R/include" -Itests/gpu -I. -L"$R/lib" -lamdhip64 -Wl,-rpath,"$R/lib" || exit 1
  : > build/pm/raw_gemm.txt
  GSHAPES=("4096 4096 8192" "8192 7168 7168" "1280 6144 7168" "2048 2048 4096" "8192 8192 8192" "16384 4096 4096")
  for rep in $(seq 1 $REPS); do
    for shp in "${GSHAPES[@]}"; do set -- $shp
      out=$(GEMM_M=$1 GEMM_N=$2 GEMM_K=$3 REF_ROWS=48 REF_ITERS=100 REF_WARMUP=300 REF_REPS=3 REF_CO="$IMG" REF_KNAME="$SYM" OURS_CO=build/pm/gemm_v4.co OURS_KNAME=gemm_mxfp8_bs_v4 OURS_TILE_M=256 OURS_TILE_N=256 ./build/pm/gemm_ref_host 2>&1)
      o=$(echo "$out" | awk '/^timing OURS/ {for (i = 1; i <= NF; i++) if ($i == "median") print $(i+1)}' | head -1)
      v=$(echo "$out" | awk '/^timing REF/ {for (i = 1; i <= NF; i++) if ($i == "median") print $(i+1)}' | head -1)
      ok=$(echo "$out" | grep -E '^OURS ' | grep -oE 'GEMM_(OK|FAIL)' | head -1); vok=$(echo "$out" | grep -E '^REF ' | grep -oE 'GEMM_(OK|FAIL)' | head -1)
      ob=$(echo "$out" | grep -E '^OURS ' | sed -nE 's/.*bad=([0-9]+).*worst_rel=([^ ]+).*/bad=\1 worst_rel=\2/p' | head -1)
      echo "GEMM $1 $2 $3 rep$rep ours=${o:-0} vendor=${v:-0} ours_check=${ok:-NA} $ob vendor_check=${vok:-NA}" >> build/pm/raw_gemm.txt
    done
  done
fi
echo "done: build/pm/raw_fmha.txt build/pm/raw_gemm.txt"