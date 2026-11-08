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
    -e FMHA_OPT_SCOPE="${FMHA_OPT_SCOPE:-}" \
    -e AMDC_RA_DEBUG_FILE="${AMDC_RA_DEBUG_FILE:-}" \
    -e AMDC_COALESCE_DEBUG="${AMDC_COALESCE_DEBUG:-}" \
    -e AMDC_WAW_DEBUG="${AMDC_WAW_DEBUG:-}" \
    -e AMDC_NO_INFLIGHT_WAW="${AMDC_NO_INFLIGHT_WAW:-}" \
    -e AMDC_NO_PHYS_STORE="${AMDC_NO_PHYS_STORE:-}" \
    -e AMDC_NO_PHYS_WAW="${AMDC_NO_PHYS_WAW:-}" \
    -e AMDC_RA_DEBUG_BLOCKS="${AMDC_RA_DEBUG_BLOCKS:-}" \
    -e PD_LDS="${PD_LDS:-}" \
    -e PD_SWZ="${PD_SWZ:-}" \
    -e PD_NOPS="${PD_NOPS:-}" \
    -e PD_B2="${PD_B2:-}" \
    -e PD_NODMA="${PD_NODMA:-}" \
    -e PD_STRIDE="${PD_STRIDE:-}" \
    -e PD_MAP="${PD_MAP:-}" \
    -e PD_NEW="${PD_NEW:-}" \
    -e PD_ROWSTRIDE="${PD_ROWSTRIDE:-}" \
    -e PD_HEADOFF="${PD_HEADOFF:-}" \
    -e PD_MAPSTRIDE="${PD_MAPSTRIDE:-}" \
    -e AMDC_M0_NOP="${AMDC_M0_NOP:-}" \
    -e AMDC_DMA_NOP="${AMDC_DMA_NOP:-}" \
    -e FMHA_LDSPAD="${FMHA_LDSPAD:-}" \
    -e AMDC_ORACLE_OUT="${AMDC_ORACLE_OUT:-}" \
    -e AMDC_TEST_KERNELS="${AMDC_TEST_KERNELS:-}" \
    -e AMDC_KEEP_MIR="${AMDC_KEEP_MIR:-}" \
    -e AMDC_LLC_ARGS="${AMDC_LLC_ARGS:-}" \
    -e AMDC_RA_FORCE_TWOPHASE="${AMDC_RA_FORCE_TWOPHASE:-}" \
    -e AMDC_RA_LDS_SPILL="${AMDC_RA_LDS_SPILL:-}" \
    -e AMDC_RA_LDS_SPILL_TARGET="${AMDC_RA_LDS_SPILL_TARGET:-}" \
    -e AMDC_HOIST_DEBUG="${AMDC_HOIST_DEBUG:-}" \
    -e AMDC_EXP_NO_NOPS="${AMDC_EXP_NO_NOPS:-}" \
    -e AMDC_M0_REUSE="${AMDC_M0_REUSE:-}" \
    -e AMDC_WAITS_SKIP_FENCE_OPERANDS="${AMDC_WAITS_SKIP_FENCE_OPERANDS:-}" \
    -e AMDC_NO_PHYS_WAR="${AMDC_NO_PHYS_WAR:-}" \
    -e AMDC_VCC_FILL="${AMDC_VCC_FILL:-}" \
    -e AMDC_LOOP_ALIGN="${AMDC_LOOP_ALIGN:-}" \
    -e AMDC_LOWER_OPTS="${AMDC_LOWER_OPTS:-}" \
    -e AMDC_DIALECT="${AMDC_DIALECT:-}" \
    -e AMDC_DROP_AUTO_WAITS="${AMDC_DROP_AUTO_WAITS:-}" \
    -e AMDC_SCHED="${AMDC_SCHED:-}" \
    -e AMDC_SCHED_OPTS="${AMDC_SCHED_OPTS:-}" \
    -e AMDC_SCHED_ADD="${AMDC_SCHED_ADD:-}" \
    -e AMDC_SCHED_REGION="${AMDC_SCHED_REGION:-}" \
    -e AMDC_SCHED_VERBOSE="${AMDC_SCHED_VERBOSE:-}" \
    -e AMDC_PIN_VMWAIT="${AMDC_PIN_VMWAIT:-}" \
    -e SCHED_WG="${SCHED_WG:-}" \
    -e AMDC_TEST_LLVM="${AMDC_TEST_LLVM:-}" \
-e UPDATE_SKELETONS="${UPDATE_SKELETONS:-}" \
    -w /work \
    amd-rust-dsl:1 cargo "$@"
