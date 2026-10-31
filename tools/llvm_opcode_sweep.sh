#!/usr/bin/env bash
# Opcode emitability sweep (executable side of the LLVM-test coverage audit).
#
# For every kernel known to `amdc list` (existing probe_* kernels, the FMHA families, and the probe_llvm_* family added for this sweep):
#   (a) `amdc mops <kernel>`  - the distinct machine opcodes (MOp variants) of its final MIR,
#   (b) `amdc emit <kernel> --backend own`   - emit,
#   (c) `clang -x assembler -mcpu=gfx950`    - assemble.
# and then maps every MOp variant of crates/amdgpu-compiler/src/mir.rs to the kernels that exercise it.
#
# Usage:  bash tools/llvm_opcode_sweep.sh [out_dir]      (default build/llvm_sweep; needs the cargo.sh container environment like the other tools)
# Output: out_dir/kernels.tsv   kernel  mops_ok  emit_ok  asm_ok  n_mops
#         out_dir/ops.tsv       MOp  n_kernels  first_kernels  emit+asm status of the best kernel
#         out_dir/unreachable.txt  MOp variants no kernel in the registry exercises
#         stdout: summary (N/total opcodes emitted + assembled, unreachable list)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT="${1:-build/llvm_sweep}"; mkdir -p "$OUT/s" "$OUT/m"
CARGO="bash tools/container/cargo.sh"
CLANG="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
$CARGO build -q --bin amdc 2>&1 | grep -E "^error" -A8 | head -20
amdc() { $CARGO run -q --bin amdc -- "$@"; }

# 1) every MOp variant (the `pub enum MOp` of mir.rs)
awk '/^pub enum MOp/{f=1;next} f&&/^}/{exit} f&&/^    [A-Z][A-Za-z0-9]*,/{gsub(/[ ,]/,"");print}' crates/amdgpu-compiler/src/mir.rs | sort -u > "$OUT/all_mops.txt"
NALL=$(wc -l < "$OUT/all_mops.txt")
# MOp keys that have a row in the LLVM table (post-RA MIR rows): the "101" of the audit
awk -F'|' '!/^[ \t]*#/ && NF>=3 {k=$1; gsub(/[ \t]/,"",k); sub(/:.*/,"",k); print k}' crates/amdgpu-compiler/src/backend/llvm24.tbl | sort -u > "$OUT/tbl_keys.txt"

# 2) per-kernel mops / emit / assemble
amdc list > "$OUT/kernel_list.txt"
: > "$OUT/kernels.tsv"
while read -r K; do
  [ -z "$K" ] && continue
  mok=0; eok=0; aok=0; n=0
  if amdc mops "$K" > "$OUT/m/$K.txt" 2> "$OUT/m/$K.err"; then mok=1; n=$(wc -l < "$OUT/m/$K.txt"); fi
  if amdc emit "$K" --backend own --out "/work/$OUT/s/$K.s" > "$OUT/s/$K.log" 2>&1; then eok=1
    if $CLANG -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$OUT/s/$K.co" "$OUT/s/$K.s" > "$OUT/s/$K.clang.log" 2>&1; then aok=1; fi
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "$K" "$mok" "$eok" "$aok" "$n" >> "$OUT/kernels.tsv"
done < "$OUT/kernel_list.txt"

# 3) opcode -> kernels (only kernels that emit AND assemble count as proof)
: > "$OUT/ops.tsv"; : > "$OUT/unreachable.txt"
OKK="$OUT/ok_kernels.txt"; awk -F'\t' '$2==1 && $3==1 && $4==1{print $1}' "$OUT/kernels.tsv" > "$OKK"
while read -r OP; do
  ks=$(while read -r K; do grep -qx "$OP" "$OUT/m/$K.txt" 2>/dev/null && echo "$K"; done < "$OKK" | head -50)
  n=$(printf "%s\n" "$ks" | grep -c .)
  if [ "$n" -gt 0 ]; then printf "%s\t%s\t%s\tOK\n" "$OP" "$n" "$(printf "%s\n" "$ks" | head -3 | tr '\n' ',')" >> "$OUT/ops.tsv"; else printf "%s\t0\t-\tNO_KERNEL\n" "$OP" >> "$OUT/ops.tsv"; echo "$OP" >> "$OUT/unreachable.txt"; fi
done < "$OUT/all_mops.txt"

NOK=$(awk -F'\t' '$4=="OK"' "$OUT/ops.tsv" | wc -l)
NK=$(wc -l < "$OUT/kernels.tsv"); NKOK=$(wc -l < "$OKK")
echo "KERNELS: $NK registered, $NKOK emit+assemble OK, $((NK-NKOK)) not"; awk -F'\t' '!($2==1 && $3==1 && $4==1){print "  NOT_OK " $1 " mops=" $2 " emit=" $3 " asm=" $4}' "$OUT/kernels.tsv"
echo "OPCODES (MOp variants of mir.rs): $NOK / $NALL emitted by at least one kernel that emits + assembles"
TB=$(wc -l < "$OUT/tbl_keys.txt"); TOK=$(grep -c -x -F -f "$OUT/tbl_keys.txt" <(awk -F'\t' '$4=="OK"{print $1}' "$OUT/ops.tsv"))
echo "LLVM-table keys (llvm24.tbl): $TOK / $TB covered"
echo "UNREACHABLE (no registered kernel exercises them):"; cat "$OUT/unreachable.txt" | sed 's/^/  /'
