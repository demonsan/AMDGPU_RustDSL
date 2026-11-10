#!/usr/bin/env bash
# Opcode emitability sweep (executable side of the LLVM-test coverage audit).
#
# For every kernel `amdc` can build (all `kernel_by_name` arms, incl. the FMHA families and the probe_llvm_* family) and for a list of option combinations
# (FMHA_OPT / AMDC_LOWER_OPTS variants that select other instruction forms):
#   (a) `amdc mops <kernel>`                 - the distinct machine opcodes (MOp variants) of its final MIR,
#   (b) `amdc emit <kernel> --backend own`   - emit,
#   (c) `clang -x assembler -mcpu=gfx950`    - assemble,
# then maps every MOp variant of crates/amdgpu-codegen/src/mir.rs to the kernels that exercise it.
#
# Usage:  bash tools/llvm_opcode_sweep.sh [out_dir]      (default build/llvm_sweep; run it where tools/container/cargo.sh works)
# Output: out_dir/kernels.tsv      label  mops_ok  emit_ok  asm_ok  n_mops
#         out_dir/ops.tsv          MOp  n_kernels  first_kernels  status (OK | NO_KERNEL)
#         out_dir/unreachable.txt  MOp variants no kernel / combo exercises
#         stdout: summary (N/total opcodes emitted + assembled, unreachable list)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT="${1:-build/llvm_sweep}"; mkdir -p "$OUT/s" "$OUT/m"
CARGO="bash tools/container/cargo.sh"
CLANG="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
$CARGO build -q --bin amdc 2>&1 | grep -E "^error" -A8 | head -20

# 1) every MOp variant (the `pub enum MOp` of mir.rs) and the keys of the LLVM table
awk '/^pub enum MOp/{f=1;next} f&&/^}/{exit} f&&/^    [A-Z][A-Za-z0-9]*,/{gsub(/[ ,]/,"");print}' crates/amdgpu-codegen/src/mir.rs | sort -u > "$OUT/all_mops.txt"
NALL=$(wc -l < "$OUT/all_mops.txt")
awk -F'|' '!/^[ \t]*#/ && NF>=3 {k=$1; gsub(/[ \t]/,"",k); sub(/:.*/,"",k); print k}' crates/amdgpu-codegen/src/target/gfx950/llvm/llvm24.tbl | sort -u > "$OUT/tbl_keys.txt"

# 2) the run list: label|kernel|ENV   (ENV = space separated VAR=VAL, may be empty)
{
  # all names accepted by kernel_by_name
  awk '/pub fn kernel_by_name/{f=1} f' crates/amdgpu-codegen/src/lib.rs | grep -oE '"[a-z0-9_]+"[[:space:]]*=>[[:space:]]*Some' | sed -E 's/"([a-z0-9_]+)".*/\1/' | sort -u | while read -r K; do echo "$K|$K|"; done
  # option combinations that select other instruction forms
  V=fmha_fwd_hd128_mxfp8_v3_vs128; T=fmha_fwd_hd128_mxfp8_v3_t128
  echo "vs128+lower-sgpr|$V|AMDC_LOWER_OPTS=dce,cmpfold,scalarize,saddr,sgpr,imm"
  echo "vs128+lower-addrfold|$V|AMDC_LOWER_OPTS=dce,cmpfold,addrfold,shl"
  echo "maskge+maskreg|probe_llvm_maskge|AMDC_LOWER_OPTS=dce,cmpfold,maskreg"
  echo "vs128+maskreg|$V|AMDC_LOWER_OPTS=dce,cmpfold,maskreg"
  echo "vs128+maskreg2|$V|AMDC_LOWER_OPTS=dce,cmpfold,maskreg2"
  echo "vs128+sbound|$V|AMDC_LOWER_OPTS=dce,cmpfold,sbound,scalarize,imm"
  echo "t128+maskreg|$T|AMDC_LOWER_OPTS=dce,cmpfold,maskreg,sbound,scalarize,imm"
  echo "t128+lower-sgpr|$T|AMDC_LOWER_OPTS=dce,cmpfold,scalarize,saddr,sgpr,imm"
} > "$OUT/run_list.txt"

: > "$OUT/kernels.tsv"
while IFS='|' read -r LABEL K ENVV; do
  [ -z "$K" ] && continue
  F=$(printf "%s" "$LABEL" | tr -c 'A-Za-z0-9_.+-' '_')
  mok=0; eok=0; aok=0; n=0
  if env $ENVV $CARGO run -q --bin amdc -- mops "$K" > "$OUT/m/$F.txt" 2> "$OUT/m/$F.err"; then mok=1; n=$(wc -l < "$OUT/m/$F.txt"); fi
  if env $ENVV $CARGO run -q --bin amdc -- emit "$K" --backend own --out "/work/$OUT/s/$F.s" > "$OUT/s/$F.log" 2>&1; then eok=1
    if $CLANG -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$OUT/s/$F.co" "$OUT/s/$F.s" > "$OUT/s/$F.clang.log" 2>&1; then aok=1; fi
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "$F" "$mok" "$eok" "$aok" "$n" >> "$OUT/kernels.tsv"
done < "$OUT/run_list.txt"

# 3) opcode -> kernels (only runs that emit AND assemble count as proof)
OKK="$OUT/ok_kernels.txt"; awk -F'\t' '$2==1 && $3==1 && $4==1{print $1}' "$OUT/kernels.tsv" > "$OKK"
: > "$OUT/ops.tsv"; : > "$OUT/unreachable.txt"
while read -r OP; do
  ks=$(while read -r F; do grep -qx "$OP" "$OUT/m/$F.txt" 2>/dev/null && echo "$F"; done < "$OKK" | head -60)
  n=$(printf "%s\n" "$ks" | grep -c .)
  if [ "$n" -gt 0 ]; then printf "%s\t%s\t%s\tOK\n" "$OP" "$n" "$(printf "%s\n" "$ks" | head -3 | tr '\n' ',')" >> "$OUT/ops.tsv"; else printf "%s\t0\t-\tNO_KERNEL\n" "$OP" >> "$OUT/ops.tsv"; echo "$OP" >> "$OUT/unreachable.txt"; fi
done < "$OUT/all_mops.txt"

NOK=$(awk -F'\t' '$4=="OK"' "$OUT/ops.tsv" | wc -l)
NK=$(wc -l < "$OUT/kernels.tsv"); NKOK=$(wc -l < "$OKK")
echo "RUNS: $NK (kernels + option combinations), $NKOK emit+assemble OK, $((NK-NKOK)) not"; awk -F'\t' '!($2==1 && $3==1 && $4==1){print "  NOT_OK " $1 " mops=" $2 " emit=" $3 " asm=" $4}' "$OUT/kernels.tsv"
echo "OPCODES (MOp variants of mir.rs): $NOK / $NALL emitted by at least one run that emits + assembles"
TB=$(wc -l < "$OUT/tbl_keys.txt"); TOK=$(awk -F'\t' '$4=="OK"{print $1}' "$OUT/ops.tsv" | grep -c -x -F -f "$OUT/tbl_keys.txt")
echo "LLVM-table keys (llvm24.tbl): $TOK / $TB covered"
echo "UNREACHABLE (no run exercises them):"; sed 's/^/  /' "$OUT/unreachable.txt"

# 4) mnemonic inventory of everything that emitted and assembled
cat "$OUT"/s/*.s 2>/dev/null | sed -E 's/;.*$//' | awk '/^[ \t]+[a-z][a-z0-9_]*[ \t]*/{print $1}' | grep -v "^\." | sort -u > "$OUT/mnemonics.txt"
echo "DISTINCT ASM MNEMONICS emitted (all assembled by clang): $(wc -l < "$OUT/mnemonics.txt")"
