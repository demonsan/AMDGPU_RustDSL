#!/usr/bin/env bash
# Regenerate crates/amdgpu-kernels/src/ladder_hints.rs: the retry-ladder rung each kernel compiles at (docs/compile-time.md). Run after a change that moves where a kernel's register allocation
# succeeds (scheduler / lowering / allocator changes, a kernel that gets heavier); tests/ladder_hints.rs fails with a stale hint.
#   bash tools/regen_ladder_hints.sh        (container cargo as the other tools; AMD_CARGO_TARGET = target dir, default /tmp/target-compose)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="${AMD_CARGO_TARGET:-/tmp/target-compose}"
AMD_CARGO_TARGET="$T" bash tools/container/cargo.sh build --release --bin amdc
"$T/release/amdc" ladder-hints --out crates/amdgpu-kernels/src/ladder_hints.rs