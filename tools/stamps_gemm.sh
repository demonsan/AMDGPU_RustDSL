#!/usr/bin/env bash
# GEMM cycle split with the DSL timing stamps: for each kernel, build with GEMM_STAMP=ends and tiles, run, summarise (cycles per wave, medians over 1024 waves).
# usage (repo clone on the node, amdc built): KERNELS="gemm_mxfp8_bs_v4 ..." bash tools/stamps_gemm.sh   (env HIP_VISIBLE_DEVICES, AMDC=<amdc binary>, GEMM_CO_DIR)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build/stamps
AMDC="${AMDC:-$(ls target/debug/amdc 2>/dev/null || echo amdc)}"; R="${ROCM_RT:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; C="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
[ -x build/gemm_ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/gemm_ref_host.cpp -o build/gemm_ref_host -I"$R/include" -Itests/gpu -I. -L"$R/lib" -lamdhip64 -Wl,-rpath,"$R/lib" || exit 1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" REF_ITERS="${REF_ITERS:-100}" REF_WARMUP="${REF_WARMUP:-200}" REF_REPS="${REF_REPS:-3}" REF_ROWS=8 STAMPS=1 OURS_TILE_M=256 OURS_TILE_N=256
printf "%-30s %-5s %9s %9s %10s %9s %9s %10s\n" kernel mode setup prologue loop epilogue total "cyc/tile"
for K in ${KERNELS:-gemm_mxfp8_bs_v4}; do
  for M in ends tiles; do
    GEMM_STAMP=$M "$AMDC" emit $K --backend=own --out build/stamps/${K}_$M.s > /dev/null 2>&1 || { echo "$K $M: emit failed"; continue; }
    "$C" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/stamps/${K}_$M.co build/stamps/${K}_$M.s 2>&1 | head -2
    OURS_CO=build/stamps/${K}_$M.co OURS_KNAME=$K STAMP_DUMP=build/stamps/${K}_$M.u32 ./build/gemm_ref_host > build/stamps/${K}_$M.log 2>&1
    python3 tools/stamps.py build/stamps/${K}_$M.u32 --phases 1:setup,2:prologue,3:loop,4:epilogue | awk -v k=$K -v m=$M '/setup|prologue|loop|epilogue/ {v[$5]=$7} END {t=v["setup"]+v["prologue"]+v["loop"]+v["epilogue"]; printf "%-30s %-5s %9d %9d %10d %9d %9d %10.0f\n", k, m, v["setup"], v["prologue"], v["loop"], v["epilogue"], t, v["loop"]/64}'
  done
done