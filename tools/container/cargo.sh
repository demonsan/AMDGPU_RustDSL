#!/usr/bin/env bash
# Run cargo inside the Rust container. Usage: tools/container/cargo.sh build
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AMD_PODMAN_ROOT:-/tmp/blyu-podman}"
TARGET="${AMD_CARGO_TARGET:-/tmp/amd-rust-dsl-target}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" "$TARGET"
chmod 700 "$XDG_RUNTIME_DIR"

# Optional read-only mount of the ROCm tree so the LLVM-tail tests can run llc in the container.
EXTRA=()
if [ -n "${AMD_ROCM_MOUNT:-}" ]; then EXTRA=(-v "$AMD_ROCM_MOUNT:$AMD_ROCM_MOUNT:ro" -e ROCM_LLVM_BIN="$AMD_ROCM_MOUNT/llvm/bin"); fi

podman --root "$ROOT" --runroot "$ROOT-run" run --rm --network=host "${EXTRA[@]}" \
    -v "$REPO:/work:Z" \
    -v "$TARGET:/target:Z" \
    -e CARGO_TARGET_DIR=/target \
    -e FMHA_ABLATE="${FMHA_ABLATE:-}" \
    -w /work \
    amd-rust-dsl:1 cargo "$@"
