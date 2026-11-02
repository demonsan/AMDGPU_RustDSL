#!/usr/bin/env bash
# Quick port loop on the node (no GPU needed): build amdc, show compile errors of the given source file only, gpu-verify --strict and a mnemonic histogram
# diff of a candidate kernel against a reference kernel.
# Usage: bash tools/port_quick.sh <candidate_kernel> <reference_kernel> [src_filter]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/pq
C="$1"; R="$2"; F="${3:-fmha_tile}"
out=$(bash tools/container/cargo.sh build -q --bin amdc 2>&1)
if echo "$out" | grep -q '^error'; then echo "$out" | awk '/^error/{p=1} /^warning/{p=0} p' | head -150; exit 1; fi
echo "$out" | grep -A8 "warning.*" | grep -B3 -A5 "$F" | head -40
bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify "$C" --strict 2>&1 | grep -vE 'Info' | tail -25
for k in "$C" "$R"; do bash tools/container/cargo.sh run -q --bin amdc -- emit "$k" --out "/work/build/pq/$k.s" 2>&1 | tail -3; done
hist() { grep -E '^\s+[a-z_0-9]+' "build/pq/$1.s" | awk '{print $1}' | sort | uniq -c | awk '{print $2, $1}'; }
join -a1 -a2 -e0 -o 0,1.2,2.2 <(hist "$C" | sort) <(hist "$R" | sort) | awk '$2 != $3 {printf "%-40s cand=%-6s ref=%-6s\n", $1, $2, $3}' | head -60
for k in "$C" "$R"; do echo "$k: $(grep -cE '^\s+[a-z]' build/pq/$k.s) instrs; $(grep -E '\.vgpr_count|\.sgpr_count|next_free_vgpr|group_segment_fixed_size' build/pq/$k.s | tr -s ' \n' ' ')"; done
