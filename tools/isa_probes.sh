#!/usr/bin/env bash
# Re-record the ISA one-hot probe maps (docs/isa-probes-smfmac-tr.md) on a gfx950 node and compare them with the committed ones.
# usage (repo clone, amdc built): HIP_VISIBLE_DEVICES=0 bash tools/isa_probes.sh [--update]
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; mkdir -p build/isa/out
AMDC="${AMDC:-$(ls target/debug/amdc 2>/dev/null || echo amdc)}"; R="${ROCM_RT:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; C="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
for k in probe_smfmac_gen probe_lds_tr_b8_gen; do
  "$AMDC" emit $k --backend=own --out build/isa/$k.s >/dev/null || exit 1
  "$C" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/isa/$k.co build/isa/$k.s || exit 1
done
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/isa_probe_host.cpp -o build/isa/isa_probe_host -I"$R/include" -L"$R/lib" -lamdhip64 -Wl,-rpath,"$R/lib" || exit 1
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" ./build/isa/isa_probe_host build/isa build/isa/out || exit 1
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
for f in smfmac_16x16x128_fp8.map lds_tr_b8.map; do
  if cmp -s build/isa/out/$f crates/amdgpu-isa/data/$f; then echo "$f: identical to the committed map"; else echo "$f: DIFFERS from the committed map"; [ "${1:-}" = "--update" ] && cp build/isa/out/$f crates/amdgpu-isa/data/$f && echo "  updated"; fi
done