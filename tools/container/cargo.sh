#!/usr/bin/env bash
# Run cargo inside the Rust container. Usage: tools/container/cargo.sh build
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AMD_PODMAN_ROOT:-/tmp/blyu-podman}"
TARGET="${AMD_CARGO_TARGET:-/tmp/amd-rust-dsl-target}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" "$TARGET"
chmod 700 "$XDG_RUNTIME_DIR"

# Default: mount the pinned LLVM (ROCm 10.1.0 + the llc tool dir) so the LLVM tail (the default backend) runs in the container.
DEF_ROCM=/shared/apps/ubuntu/opt/rocm-10.1.0; DEF_TOOLS=/shared/amdgpu/home/blyu_qle/llvm-tools/rocm-10.1.0
if [ -z "${AMD_ROCM_MOUNT:-}" ] && [ -d "$DEF_ROCM" ] && [ -d "$DEF_TOOLS" ]; then
    AMD_ROCM_MOUNT="$DEF_ROCM $DEF_TOOLS"; AMD_LLVM_BIN="${AMD_LLVM_BIN:-$DEF_TOOLS/bin}"
fi
# Optional read-only mount of the ROCm tree so the LLVM-tail tests can run llc in the container.
EXTRA=()
# AMD_ROCM_MOUNT may list several directories; AMD_LLVM_BIN overrides the tool directory.
if [ -n "${AMD_ROCM_MOUNT:-}" ]; then
    for m in $AMD_ROCM_MOUNT; do EXTRA+=(-v "$m:$m:ro"); done
    EXTRA+=(-e ROCM_LLVM_BIN="${AMD_LLVM_BIN:-${AMD_ROCM_MOUNT%% *}/llvm/bin}")
fi

podman --root "$ROOT" --runroot "$ROOT-run" run --rm --network=host "${EXTRA[@]}" \
    -v "$REPO:/work:Z" \
    -v "$TARGET:/target:Z" \
    -e CARGO_TARGET_DIR=/target \
    -e FMHA_ABLATE="${FMHA_ABLATE:-}" \
    -e FMHA_PITCH="${FMHA_PITCH:-}" \
    -e FMHA_OPT="${FMHA_OPT:-}" \
    -e PD_LDS="${PD_LDS:-}" \
    -e AMDC_ORACLE_OUT="${AMDC_ORACLE_OUT:-}" \
    -e AMDC_TEST_KERNELS="${AMDC_TEST_KERNELS:-}" \
    -e AMDC_KEEP_MIR="${AMDC_KEEP_MIR:-}" \
    -e AMDC_LLC_ARGS="${AMDC_LLC_ARGS:-}" \
    -e AMDC_DIALECT="${AMDC_DIALECT:-}" \
    -e AMDC_DROP_AUTO_WAITS="${AMDC_DROP_AUTO_WAITS:-}" \
    -e AMDC_SCHED="${AMDC_SCHED:-}" \
    -e AMDC_SCHED_OPTS="${AMDC_SCHED_OPTS:-}" \
    -w /work \
    amd-rust-dsl:1 cargo "$@"
