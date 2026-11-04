#!/usr/bin/env bash
# Alternating A/B of GEMM kernels (same process per kernel, rounds interleaved): KERNELS="a b c" ROUNDS=7 bash tools/ab_gemm.sh
# env: AMDC (amdc binary), HIP_VISIBLE_DEVICES (default 4), REF_ITERS (100), REF_WARMUP (100), SKIP_CHECK=1 to skip the CPU row check
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build/ab
AMDC="${AMDC:-$(ls target/debug/amdc 2>/dev/null || echo amdc)}"; R="${ROCM_RT:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; C="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/gemm_ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/gemm_ref_host.cpp -o build/gemm_ref_host -I"$R/include" -Itests/gpu -I. -L"$R/lib" -lamdhip64 -Wl,-rpath,"$R/lib" || exit 1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" REF_ITERS="${REF_ITERS:-100}" REF_WARMUP="${REF_WARMUP:-100}" REF_REPS=1 REF_ROWS="${REF_ROWS:-8}" OURS_TILE_M=256 OURS_TILE_N=256
for K in $KERNELS; do
  "$AMDC" emit $K --backend=own --out build/ab/$K.s > build/ab/$K.emit.log 2>&1 || { echo "$K: emit failed"; tail -3 build/ab/$K.emit.log; continue; }
  "$C" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/ab/$K.co build/ab/$K.s 2>&1 | head -2
  echo "$K: $(grep -cE '^\s+(v_mfma|s_nop)' build/ab/$K.s) mfma+nop lines, vgpr $(grep -m1 -E 'vgpr_count' build/ab/$K.s | tr -s ' ')"
done
for r in $(seq 1 ${ROUNDS:-7}); do
  for K in $KERNELS; do
    [ -f build/ab/$K.co ] || continue
    OURS_CO=build/ab/$K.co OURS_KNAME=$K ./build/gemm_ref_host > build/ab/$K.run.log 2>&1
    echo "$K $r $(grep -E 'timing OURS' build/ab/$K.run.log | sed 's/.*median \([0-9.]*\) TFLOP.*/\1/') $(grep -c GEMM_OK build/ab/$K.run.log)"
  done
done | awk '{ if ($1 in n) {} ; v[$1]=v[$1] " " $3; ok[$1]+=$4; c[$1]++ } END { for (k in v) { split(v[k], a, " "); n=asort(a); printf "%-34s median %8.1f  min %8.1f  max %8.1f  GEMM_OK %d/%d\n", k, a[int((n+1)/2)], a[1], a[n], ok[k], c[k] } }' | sort