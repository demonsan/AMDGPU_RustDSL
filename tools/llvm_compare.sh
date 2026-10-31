#!/usr/bin/env bash
# Comparison evidence: for each case, the instruction `llc` (the pinned AMD LLVM 24 tail, gfx950) selects for an LLVM IR feature vs the instruction our DSL probe emits.
# Usage: bash tools/llvm_compare.sh [sweep_dir]    (sweep_dir = output of tools/llvm_opcode_sweep.sh, default build/llvm_sweep; the sweep must have run so the .s files exist)
# Verdicts: IDENTICAL (same text after register-number normalisation), SAME_MNEMONIC (mnemonic equal, operands / modifiers differ: printed), MISSING_OURS / LLC_ERROR.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
SW="${1:-build/llvm_sweep}"; OUT=build/llvm_compare; mkdir -p "$OUT"
LLC="${LLC:-/shared/amdgpu/home/blyu_qle/llvm-tools/rocm-10.1.0/bin/llc}"
norm() { sed -E 's/;.*$//; s/[[:space:]]+/ /g; s/^ //; s/ $//; s/v\[[0-9]+:[0-9]+\]/v[N:N]/g; s/s\[[0-9]+:[0-9]+\]/s[N:N]/g; s/\bv[0-9]+\b/vN/g; s/\bs[0-9]+\b/sN/g; s/\bvcc_lo\b/vcc/g'; }
n=0; ident=0; same=0; diff=0; miss=0
printf "%-26s %-9s | %s\n" CASE VERDICT "llc  ||  ours"
while IFS=$'\t' read -r ID IR RX EX LAB; do
  case "$ID" in \#*|"") continue;; esac
  n=$((n+1))
  "$LLC" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O2 "tools/llvm_compare/$IR" -o "$OUT/$ID.s" 2> "$OUT/$ID.err" || { printf "%-26s %-9s | %s\n" "$ID" LLC_ERROR "$(head -1 "$OUT/$ID.err" | cut -c1-100)"; miss=$((miss+1)); continue; }
  pick() { f="$1"; if [ "$EX" = "-" ]; then grep -E "$RX" "$f" 2>/dev/null | head -1; else grep -E "$RX" "$f" 2>/dev/null | grep -v -E "$EX" | head -1; fi; }
  L=$(pick "$OUT/$ID.s" | norm)
  O=""
  if [ "$LAB" != "-" ] && [ -f "$SW/s/$LAB.s" ]; then O=$(pick "$SW/s/$LAB.s" | norm); fi
  if [ -z "$O" ]; then for f in "$SW"/s/*.s; do O=$(pick "$f" | norm); [ -n "$O" ] && break; done; fi
  if [ -z "$L" ]; then printf "%-26s %-9s | llc did not emit %s\n" "$ID" LLC_ERROR "$RX"; miss=$((miss+1)); continue; fi
  if [ -z "$O" ]; then printf "%-26s %-9s | %s  ||  (none in the sweep outputs)\n" "$ID" MISSING_OURS "$L"; miss=$((miss+1)); continue; fi
  if [ "$L" = "$O" ]; then v=IDENTICAL; ident=$((ident+1)); else
    ml=$(echo "$L" | awk '{print $1}'); mo=$(echo "$O" | awk '{print $1}')
    if [ "$ml" = "$mo" ]; then v=SAME_MNEM; same=$((same+1)); else v=DIFFER; diff=$((diff+1)); fi
  fi
  printf "%-26s %-9s | %s  ||  %s\n" "$ID" "$v" "$L" "$O"
done < tools/llvm_compare/cases.tsv
echo "COMPARISON: $n cases: $ident IDENTICAL, $same SAME_MNEMONIC (operand/modifier text differs), $diff DIFFER, $miss missing/llc-error"