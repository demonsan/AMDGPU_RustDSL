#!/usr/bin/env bash
# Run cargo inside the Rust container. Usage: tools/container/cargo.sh build
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AMD_PODMAN_ROOT:-/tmp/blyu-podman}"
TARGET="${AMD_CARGO_TARGET:-/tmp/amd-rust-dsl-target}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
mkdir -p "$XDG_RUNTIME_DIR" "$TARGET"
chmod 700 "$XDG_RUNTIME_DIR"

podman --root "$ROOT" --runroot "$ROOT-run" run --rm --network=host \
    -v "$REPO:/work:Z" \
    -v "$TARGET:/target:Z" \
    -e CARGO_TARGET_DIR=/target \
    -w /work \
    amd-rust-dsl:1 cargo "$@"
