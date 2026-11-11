#!/usr/bin/env bash
# Emit, assemble and run the sparse-fp8 MFMA layout probe (v_smfmac_f32_16x16x128_fp8_fp8).
# This is the reproduction of the smfmac recipe in docs/vendor-spec.md ("smfmac sparse layout").
#
# Usage:
#   bash tools/vendor_smfmac_probe.sh            # emit + assemble + run, print the decoded slots
#   SLOTS=12 bash tools/vendor_smfmac_probe.sh   # kernel currently dumps 12 slots
#
# Env: ROCM_PATH (clang/assemble, default ROCm 10.1.0), AMD_LLVM_BIN (own-backend llc dir when the
# default tail is used - this probe is own-backend only), HIP_VISIBLE_DEVICES (default 5),
# ROCM_PROF/HIP runtime for the host (default /shared/apps/ubuntu/opt/rocm-7.2.4 for HIP).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
HIPROOT="${HIPROOT:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
OUT="${VENDOR_WORK:-$REPO/build/vendor}"
cd "$REPO"
mkdir -p "$OUT"
unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
export AMD_CARGO_TARGET="${AMD_CARGO_TARGET:-/tmp/amd-target-occ}"

echo "[smfmac] emit (own backend)"
bash tools/container/cargo.sh run -q --bin amdc -- emit probe_smfmac_ones --backend own \
    --out /work/build/probe_smfmac_ones.s 2>&1 | grep -E "wrote|error" || exit 1
cp build/probe_smfmac_ones.s "$OUT/probe_smfmac_ones.s"
"$ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
    -mcode-object-version=6 -o "$OUT/probe_smfmac_ones.co" "$OUT/probe_smfmac_ones.s" || exit 1
grep -E "amdhsa_next_free|v_smfmac" "$OUT/probe_smfmac_ones.s" | head -8

echo "[smfmac] build host"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/smfmac_host.cpp -o build/smfmac_host \
    -I"$HIPROOT/include" -L"$HIPROOT/lib" -lamdhip64 -Wl,-rpath,"$HIPROOT/lib" || exit 1

echo "[smfmac] run"
SLOTS="${SLOTS:-16}"
./build/smfmac_host "$OUT/probe_smfmac_ones.co" "$SLOTS"
