#!/usr/bin/env bash
# Paged decode the way a host launches it: ask `amdc dispatch decode16` for the plan (crates/amdgpu-graph/src/dispatch.rs `decode16_plan`, the single
# rule), emit + assemble that kernel, run tests/gpu/pa_host.cpp with the matching mode (fused / last-arriver) and split count.
# usage (repo root, GPU node): HIP_VISIBLE_DEVICES=0 bash tools/pa_auto.sh B H HK CTX [PAGE]   env: AMDC (default: the container cargo run), O (build/pa_auto)
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
B=$1; H=$2; HK=$3; CTX=$4; P=${5:-16}
R=${ROCM:-/shared/apps/ubuntu/opt/rocm-7.0.2}; CL=${CLANG:-/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang}; O=${O:-build/pa_auto}; mkdir -p "$O"
amdc() { if [ -n "${AMDC:-}" ]; then "$AMDC" "$@"; else bash tools/container/cargo.sh run -q --bin amdc -- "$@"; fi; }
plan=$(amdc dispatch decode16 "$B" "$HK" "$CTX" --page "$P")
k=$(echo "$plan" | sed -E 's/.*"kernel":"([^"]+)".*/\1/'); form=$(echo "$plan" | sed -E 's/.*"form":"([^"]+)".*/\1/'); S=$(echo "$plan" | sed -E 's/.*"splits":([0-9]+).*/\1/')
echo "plan: $plan"
[ -x "$O/pa_host" ] || $R/bin/amdclang++ -x hip -O2 --offload-arch=gfx950 --rocm-path=$R -std=c++17 tests/gpu/pa_host.cpp -o "$O/pa_host" -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib
for kk in "$k" attn_combine_bf16_hd128; do
  if [ -n "${AMDC:-}" ]; then amdc emit "$kk" --backend=own --out "$O/$kk.s" >/dev/null; else amdc emit "$kk" --backend=own --out "/work/$O/$kk.s" >/dev/null; fi
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$O/$kk.co" "$O/$kk.s"
done
case "$form" in
  fused) export FUSED=1 KSUFFIX=_f ;;
  lastc) export LASTC=1 KSUFFIX=_c ;;
esac
KPREFIX=pa_decode16_bf16_hd128_p "$O/pa_host" ours "$O/$k.co" "$O/attn_combine_bf16_hd128.co" "$B" "$H" "$HK" "$CTX" "$P" "$S"
