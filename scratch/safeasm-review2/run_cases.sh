#!/usr/bin/env bash
# Adversarial-review driver v2 (hardened safe-asm API): compiles every attack case with rustc against the freshly built
# crate rlibs inside the pinned rust container, classifies pass_/fail_, and runs the pass binaries (they print their findings).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AMD_PODMAN_ROOT:-/tmp/blyu-podman}"
TARGET="${AMD_CARGO_TARGET:-/tmp/amd-target-occ}"
D="$REPO/scratch/safeasm-review2"
cd "$REPO"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
export AMD_ROCM_MOUNT="${AMD_ROCM_MOUNT:-/shared/apps/ubuntu/opt/rocm-10.1.0 /shared/amdgpu/home/blyu_qle/llvm-tools}"
export AMD_LLVM_BIN="${AMD_LLVM_BIN:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
mkdir -p "$XDG_RUNTIME_DIR" "$TARGET" "$D/out"
echo "[review2] cargo build -p amdgpu-safeasm -p amdgpu-codegen -p amdgpu-dsl"
bash tools/container/cargo.sh build -q -p amdgpu-safeasm -p amdgpu-codegen -p amdgpu-dsl 2>&1 | tail -3
deps="$TARGET/debug/deps"
newest() { ls -1t "$deps"/lib$1-*.rlib 2>/dev/null | head -1; }
SAFEASM="$(newest amdgpu_safeasm)"; COMPILER="$(newest amdgpu_codegen)"; DSL="$(newest amdgpu_dsl)"
echo "[review2] rlibs: $(basename "$SAFEASM") $(basename "$COMPILER") $(basename "$DSL")"
crustc() {
  podman --root "$ROOT" --runroot "$ROOT-run" run --rm --network=host \
    -v "$REPO:/work:Z" -v "$TARGET:/target:Z" -e CARGO_TARGET_DIR=/target \
    -v /shared/apps/ubuntu/opt/rocm-10.1.0:/shared/apps/ubuntu/opt/rocm-10.1.0:ro \
    -w /work amd-rust-dsl:1 rustc "$@"
}
compile_case() {
  local extra=(--edition 2021 --crate-type bin --crate-name uicase -L dependency=/target/debug/deps
               --extern "amdgpu_safeasm=/target/debug/deps/$(basename "$SAFEASM")")
  grep -q "amdgpu_codegen" "$1" && extra+=(--extern "amdgpu_codegen=/target/debug/deps/$(basename "$COMPILER")")
  grep -q "amdgpu_dsl" "$1" && extra+=(--extern "amdgpu_dsl=/target/debug/deps/$(basename "$DSL")")
  crustc "${extra[@]}" -o "/target/uicase-$2" "scratch/safeasm-review2/cases/$(basename "$1")" 2>&1
}
echo
echo "================ ATTACK MATRIX v2 ================"
for f in "$D"/cases/*.rs; do
  n="$(basename "$f")"; b="${n%.rs}"
  err="$(compile_case "$f" "$b")"; rc=$?
  first="$(printf '%s\n' "$err" | grep -E '^error' | head -3 | tr '\n' ' | ')"
  case "$n" in
    fail_*) if [ $rc -ne 0 ]; then echo "CASE $n: REJECTED ✓  :: ${first}"; else echo "CASE $n: ** COMPILED (expected reject) **"; fi ;;
    pass_*) if [ $rc -ne 0 ]; then echo "CASE $n: ** REJECTED (expected compile) **  :: ${first}"; else
               echo "CASE $n: COMPILED ✓"
               out="$("$TARGET/uicase-$b" 2>&1 | head -60)"
               printf '%s\n' "$out" | sed 's/^/  /'
             fi ;;
  esac
done
echo "=================================================="
