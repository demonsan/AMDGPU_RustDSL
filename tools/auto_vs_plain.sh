#!/usr/bin/env bash
# Measure `--sched=auto` against the unscheduled build on hardware (default LLVM-tail backend).
#   micro: sched_* microkernels, cycles per iteration (1 wave / 2 waves per SIMD)   bash tools/auto_vs_plain.sh micro
#   fmha : B1H8S2048 TFLOP/s for own / llvm-tail / llvm-tail+auto                    bash tools/auto_vs_plain.sh fmha
# env: HIP_VISIBLE_DEVICES (default 4), ASM_ROC (default rocm-10.1.0)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/hw
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; ASM="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
GPU="${HIP_VISIBLE_DEVICES:-4}"
asm() { "$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$2" "$1"; }
case "${1:-micro}" in
micro)
  [ -x build/sched_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/sched_host.cpp -o build/sched_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
  printf "%-18s %-6s | %9s %9s | %9s %9s\n" kernel variant 1wave 2waves "vs plain 1w" "vs plain 2w"
  for K in sched_valu_mul sched_valu_max3 sched_valu_exp sched_mfma_chain sched_mfma_valu16 sched_mfma_valu4 sched_diamond; do
    for V in plain auto; do
      FL=""; [ $V = auto ] && FL="--sched=auto"
      bash tools/container/cargo.sh run -q --bin amdc -- emit $K $FL --out /work/build/hw/$K.$V.s >/dev/null 2>&1
      asm build/hw/$K.$V.s build/hw/$K.$V.co || continue
      H1=$(HIP_VISIBLE_DEVICES=$GPU ./build/sched_host build/hw/$K.$V.co $K 64 1 4000 10 | sed -n 's/.*cycles_per_iter=\([0-9.]*\).*/\1/p')
      H2=$(HIP_VISIBLE_DEVICES=$GPU ./build/sched_host build/hw/$K.$V.co $K 512 1 4000 10 | sed -n 's/.*cycles_per_iter=\([0-9.]*\).*/\1/p')
      if [ $V = plain ]; then P1=$H1; P2=$H2; printf "%-18s %-6s | %9s %9s |\n" $K $V $H1 $H2
      else awk -v k=$K -v a=$H1 -v b=$H2 -v p1=$P1 -v p2=$P2 'BEGIN{printf "%-18s %-6s | %9s %9s | %+8.1f%% %+8.1f%%\n", k, "auto", a, b, (a/p1-1)*100, (b/p2-1)*100}'; fi
    done
  done ;;
fmha)
  K=fmha_fwd_hd128_mxfp8_t; CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
  [ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
  for V in own llvm-tail llvm-tail+auto; do
    FL="--backend ${V%%+*}"; [ "$V" = llvm-tail+auto ] && FL="--backend llvm-tail --sched=auto"
    bash tools/container/cargo.sh run -q --bin amdc -- emit $K $FL --out /work/build/hw/fmha.$V.s >/dev/null 2>&1
    asm build/hw/fmha.$V.s build/hw/fmha.$V.co || continue
    for rep in 1 2 3; do
      echo -n "$V rep$rep: "
      REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=4 REF_ITERS=30 REF_CO2=build/hw/fmha.$V.co REF_KNAME2=$K HIP_VISIBLE_DEVICES=$GPU ./build/ref_host "$CO" 2>&1 | grep -E "timing +ours|timing ours" | tr -s ' '
    done
  done ;;
esac