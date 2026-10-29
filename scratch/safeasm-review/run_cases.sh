#!/usr/bin/env bash
# Adversarial review driver for crates/amdgpu-safeasm. Compiles each case with rustc inside the pinned rust
# container (links against the just-built crate rlibs), classifies pass_/fail_, runs the pass binaries if they
# print something, then emits both the safe-DSL phase asm and the kernel-path asm for the POC identity check.
# Not part of the cargo workspace: it only reads the crate.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${AMD_PODMAN_ROOT:-/tmp/blyu-podman}"
TARGET="${AMD_CARGO_TARGET:-/tmp/amd-target-occ}"
D="$REPO/scratch/safeasm-review"
cd "$REPO"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/xdg-$USER}"
export AMD_ROCM_MOUNT="${AMD_ROCM_MOUNT:-/shared/apps/ubuntu/opt/rocm-10.1.0 /shared/amdgpu/home/blyu_qle/llvm-tools}"
export AMD_LLVM_BIN="${AMD_LLVM_BIN:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
mkdir -p "$XDG_RUNTIME_DIR" "$TARGET" "$D/out"
echo "[review] cargo build -p amdgpu-safeasm"
bash tools/container/cargo.sh build -q -p amdgpu-safeasm 2>&1 | tail -3

deps="$TARGET/debug/deps"
newest() { ls -1t "$deps"/lib$1-*.rlib 2>/dev/null | head -1; }
SAFEASM="$(newest amdgpu_safeasm)"; COMPILER="$(newest amdgpu_compiler)"; DSL="$(newest amdgpu_dsl)"
echo "[review] rlibs: $SAFEASM $COMPILER $DSL"
[ -n "$SAFEASM" ] || { echo "no safeasm rlib"; exit 1; }

# run one rustc invocation inside the container (same mounts as cargo.sh)
crustc() {
  podman --root "$ROOT" --runroot "$ROOT-run" run --rm --network=host \
    -v "$REPO:/work:Z" -v "$TARGET:/target:Z" -e CARGO_TARGET_DIR=/target \
    -v /shared/apps/ubuntu/opt/rocm-10.1.0:/shared/apps/ubuntu/opt/rocm-10.1.0:ro \
    -w /work amd-rust-dsl:1 rustc "$@"
}

compile_case() { # $1 = file, $2 = output binary
  local extra=(--edition 2021 --crate-type bin --crate-name uicase -L "dependency=/target/debug/deps"
               --extern "amdgpu_safeasm=/target/debug/deps/$(basename "$SAFEASM")")
  if grep -q "amdgpu_compiler" "$1"; then
    extra+=(--extern "amdgpu_compiler=/target/debug/deps/$(basename "$COMPILER")")
  fi
  if grep -q "amdgpu_dsl" "$1"; then
    extra+=(--extern "amdgpu_dsl=/target/debug/deps/$(basename "$DSL")")
  fi
  crustc "${extra[@]}" -o "/target/uicase-$2" "scratch/safeasm-review/cases/$(basename "$1")" 2>&1
}

echo
echo "================ ATTACK MATRIX ================"
for f in "$D"/cases/*.rs; do
  n="$(basename "$f")"; bin="$D/out/${n%.rs}"
  err="$(compile_case "$f" "${n%.rs}" )"; rc=$?
  first="$(printf '%s\n' "$err" | grep -E '^(error|warning: unused)' | head -3 | tr '\n' ' | ')"
  case "$n" in
    fail_*) if [ $rc -ne 0 ]; then echo "CASE $n: REJECTED ✓  :: ${first}"; else echo "CASE $n: ** COMPILED (expected reject) **"; fi ;;
    pass_*) if [ $rc -ne 0 ]; then echo "CASE $n: ** REJECTED (expected compile) **  :: ${first}"; else
               echo "CASE $n: COMPILED ✓"
               out="$("$TARGET/uicase-${n%.rs}" 2>&1 | head -80)"
               echo "  ---- run output ----"; printf '%s\n' "$out" | sed 's/^/  /'
             fi ;;
  esac
done
echo "=============================================="
