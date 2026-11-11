#!/usr/bin/env bash
# AGPR probes: emit, assemble, run, compare outputs of the pinned variants with the arch-VGPR reference.
# usage (repo clone on the node): bash tools/agpr_probe.sh   (env HIP_VISIBLE_DEVICES, ASM_ROC)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/agpr
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-10.1.0}"; ASM="${ASM_ROC:-$ROC}"
[ -x build/sched_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/sched_host.cpp -o build/sched_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
for K in ${KERNELS:-probe_agpr_ref probe_agpr_acc probe_agpr_full probe_agpr_pinned}; do
  bash tools/container/cargo.sh run -q --bin amdc -- emit $K --backend=own --out /work/build/agpr/$K.s 2>&1 | grep -E "^error|compile error|wrote" | cut -c1-300
  "$ASM/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/agpr/$K.co build/agpr/$K.s 2>&1 | head -5
  for N in ${COUNTS:-1 2 3 7}; do
    echo -n "$K n=$N: "; SH_DUMP=4 HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} ./build/sched_host build/agpr/$K.co $K ${BLOCK:-512} 1 $N 3 | tr '\n' ' ' | sed 's/ms=[0-9.]* cycles_per_iter=[0-9.]*//'; echo
  done
done
grep -E "accum_offset|next_free_vgpr|agpr_count|vgpr_count" build/agpr/probe_agpr_full.s build/agpr/probe_agpr_pinned.s