#!/usr/bin/env bash
# Per-kernel differential oracle: OwnTail vs LlvmTail for every kernel (or the named ones).
# Needs amdc built (via cargo.sh) and the pinned ROCm on the node. Output: one line per kernel.
#   bash tools/hazard_oracle.sh [kernel ...]        # writes build/oracle/<kernel>.{own,llvm-tail}.s
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build/oracle
export AMD_ROCM_MOUNT="$ROC"
KERNELS=("$@")
if [ ${#KERNELS[@]} -eq 0 ]; then
  mapfile -t KERNELS < <(bash tools/container/cargo.sh run -q --bin amdc -- list 2>/dev/null)
fi
printf "%-28s %8s %8s %6s %6s %5s %5s %s\n" kernel own_nop llvm_nop own_wc llvm_wc dalu mism "(core instr)"
for K in "${KERNELS[@]}"; do
  for be in own llvm-tail; do
    bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" --backend $be --out "/work/build/oracle/$K.$be.s" >/dev/null 2>&1 || echo "EMIT FAILED $K $be"
  done
  python3 tools/asmdiff.py "build/oracle/$K.own.s" "build/oracle/$K.llvm-tail.s" | python3 -c '
import sys,re
t=sys.stdin.read()
m=re.search(r"core instructions: own=(\d+) llvm=(\d+) mismatches=(\d+)",t)
n=re.search(r"nop cycles : own=(\d+) llvm=(\d+)",t)
f=re.search(r"filler ops : own=(\{.*?\}) llvm=(\{.*\})",t)
own=eval(f.group(1)); ll=eval(f.group(2))
print("%8s %8s %6d %6d %5d %5s %s" % (n.group(1),n.group(2),own.get("s_waitcnt",0),ll.get("s_waitcnt",0),ll.get("s_delay_alu",0),m.group(3),m.group(1)))' | sed "s/^/$(printf '%-28s ' $K)/"
done