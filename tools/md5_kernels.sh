#!/usr/bin/env bash
# md5 of the emitted assembly of every kernel (own backend): bash tools/md5_kernels.sh <out.txt>   (cwd = repo clone; uses tools/container/cargo.sh)
set -uo pipefail
OUT="${1:-/dev/stdout}"; : > "$OUT"
mkdir -p build/md5
for k in $(bash tools/container/cargo.sh run -q --bin amdc -- list 2>/dev/null); do
  bash tools/container/cargo.sh run -q --bin amdc -- emit "$k" --backend=own --out "/work/build/md5/$k.s" >/dev/null 2>&1
  if [ -f "build/md5/$k.s" ]; then echo "$k $(md5sum < build/md5/$k.s | cut -d' ' -f1)" >> "$OUT"; else echo "$k FAILED" >> "$OUT"; fi
done