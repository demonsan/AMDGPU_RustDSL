#!/usr/bin/env bash
# Same-process fused-MoE stage 1 vs aiter's 2-stage ASM stage 1 (docs/op-coverage-design.md section 7.6 shapes): moe_host runs ours
# (gemm_moe1 + gemm_moe2 + moe_sum) and, with VENDOR1, aiter's fmoe_stage1 ..._128x128_pf2 on the same sorted routes; both at BLOCK_M 128.
#   HIP_VISIBLE_DEVICES=0 bash tools/vs_aiter_moe.sh [out_dir]
set -u
OUT=${1:-build/vs_aiter_moe}
AMDC=${AMDC:-/tmp/target-kfmha/release/amdc}
CLANG=${CLANG:-/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang}
ROCM=${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}
HSA=${AITER_HSA:-/shared/amdgpu/home/blyu_qle/sessions/aiter_src/hsa/gfx950}
mkdir -p "$OUT"
emit() { [ -s "$OUT/$1.co" ] || { "$AMDC" emit "$1" --backend own --out "$OUT/$1.s" >"$OUT/$1.log" 2>&1 && "$CLANG" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$OUT/$1.co" "$OUT/$1.s"; } || echo "EMIT FAIL $1"; }
vname() { strings "$1" | grep -m1 '^_ZN'; }
"$ROCM/bin/amdclang++" -x hip -O2 --offload-arch=gfx950 --rocm-path="$ROCM" -std=c++17 -I tests/gpu tests/gpu/moe_host.cpp -o "$OUT/moe_host" -L"$ROCM/lib" -lamdhip64 -Wl,-rpath,"$ROCM/lib" || { echo "host build failed"; exit 1; }
for dt in fp8 fp8bs; do
    case $dt in fp8) vco="$HSA/fmoe_2stages/fmoe_stage1_bf16_pertokenFp8_g1u1_128x128_pf2.co";; fp8bs) vco="$HSA/fmoe_2stages/fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1_128x128_pf2.co";; esac
    for sh in "1024 4096 1024 8 2" "4096 7168 512 32 8" "256 7168 2048 64 8"; do
        set -- $sh
        ks="gemm_moe1_${dt}_silu gemm_moe2_${dt} moe_sum_bf16_k$5"
        [ $dt != bf16 ] && ks="$ks quant_token_bf16_fp8_n$3"
        for k in $ks; do emit "$k"; done
        out=$(BLOCK_M=128 CHECK=2 ITERS=50 WARMUP=10 VENDOR1="$vco" VENDOR1_KN="$(vname "$vco")" timeout 900 "$OUT/moe_host" "$dt" "$OUT" $1 $2 $3 $4 $5 silu 2>&1)
        o=$(echo "$out" | grep -oE 'stage1 [0-9.]+ us' | head -1 | grep -oE '[0-9.]+')
        v=$(echo "$out" | grep -E '^vendor stage1' | grep -oE ': [0-9.]+ us' | grep -oE '[0-9.]+')
        ok=$(echo "$out" | grep -oE 'MOE_(OK|FAIL)' | head -1)
        r=$(python3 -c "print(f'{$v/$o:.2f}')" 2>/dev/null || echo -)
        echo "MOE $dt M$1 D$2 I$3 E$4 K$5 ours_stage1=$o us aiter_stage1=$v us ratio=$r $ok"
        [ -z "$o" ] || [ -z "$v" ] && echo "$out" | tail -5
    done
done
