#!/usr/bin/env bash
# Build (or rebuild) the Rust toolchain image. Run on the GPU node.
# Images live under /tmp so the shared home stays clean.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AMD_PODMAN_ROOT:-/tmp/blyu-podman}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

podman --root "$ROOT" --runroot "$ROOT-run" build \
    --network=host \
    -t amd-rust-dsl:1 \
    -f "$REPO/tools/container/Containerfile" \
    "$REPO"
echo "IMAGE_OK amd-rust-dsl:1"
