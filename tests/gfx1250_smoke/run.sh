#!/usr/bin/env bash
# gfx1250 assembler smoke checks: every snippet in this directory must assemble with the rocm-10.1 clang for -mcpu=gfx1250, and every
# `# EXPECT-FAIL` snippet must be rejected (`*.asm`: `*.s` is git-ignored in this repo). `probe_*` snippets record what this toolchain
# accepts for the open questions of docs/target-gfx1250.md. They pin the instruction syntax the document refers to (phase G of
# docs/refactor-4layer-plan.md: specification only - assembler, no code generation, no hardware runs). The instruction forms come from
# the public upstream LLVM MC tests (llvm/test/MC/AMDGPU/gfx1250_asm_*.s).
# usage (any node with /shared mounted): bash tests/gfx1250_smoke/run.sh [-v]
#   env: ROCM (default /shared/apps/ubuntu/opt/rocm-10.1.0)
set -u
ROCM=${ROCM:-/shared/apps/ubuntu/opt/rocm-10.1.0}
CL=$ROCM/llvm/bin/clang
DIS=$ROCM/llvm/bin/llvm-objdump
D=$(cd "$(dirname "$0")" && pwd)
V=0; [ "${1:-}" = "-v" ] && V=1
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
echo "toolchain: $($CL --version | head -1)"
pass=0; fail=0
for s in "$D"/*.asm; do
    n=$(basename "$s" .asm)
    if grep -q '^# EXPECT-FAIL' "$s"; then
        if $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -c -o "$T/$n.o" "$s" 2>"$T/$n.err"; then
            echo "FAIL $n: expected the assembler to reject it"; fail=$((fail + 1))
        else
            echo "ok   $n (rejected as expected: $(grep -m1 error "$T/$n.err" | sed 's/.*error: //'))"; pass=$((pass + 1))
        fi
        continue
    fi
    if $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -c -o "$T/$n.o" "$s" 2>"$T/$n.err"; then
        k=$(grep -cvE '^\s*(#|//|$|\.|[A-Za-z_0-9]+:)' "$s")
        echo "ok   $n ($k instructions)"; pass=$((pass + 1))
        [ $V = 1 ] && $DIS -d --mcpu=gfx1250 "$T/$n.o" | sed -n '/<.text>/,$p' | tail -n +2
    else
        echo "FAIL $n"; sed 's/^/     /' "$T/$n.err" | head -20; fail=$((fail + 1))
    fi
done
echo "gfx1250 smoke: $pass passed, $fail failed"
[ $fail = 0 ]