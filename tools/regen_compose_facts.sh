#!/usr/bin/env bash
# Regenerate crates/amdgpu-graph/src/compose_facts.rs from the kernels' own plans (docs/compose.md). Run it after ANY change that moves a kernel's plan
# (stage structure, rings, CTA shape, weave phases / units) and commit the result with the change; tests/compose_facts.rs fails until you do.
#   bash tools/regen_compose_facts.sh        (container cargo as the other tools; AMD_CARGO_TARGET = target dir, default /tmp/target-compose)
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="${AMD_CARGO_TARGET:-/tmp/target-compose}"
AMD_CARGO_TARGET="$T" bash tools/container/cargo.sh build --bin amdc
"$T/debug/amdc" compose-facts --out crates/amdgpu-graph/src/compose_facts.rs
echo "regenerated crates/amdgpu-graph/src/compose_facts.rs ($(grep -c 'KernelFacts {' crates/amdgpu-graph/src/compose_facts.rs) kernels)"