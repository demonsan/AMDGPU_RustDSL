#!/usr/bin/env bash
# gen3 synchronization capability probe.
#   Part 1 (toolchain/ISA): the gfx12 named/split barrier opcodes vs gfx950 (CDNA4).
#   Part 2 (hardware, GPU 5): block s_barrier vs a software 4-wave group barrier (LDS atomic counter + spin) vs no sync,
#   balanced and asymmetric (group B has an extra scalar delay), plus a wrong-count / timeout case.
# Two wave groups: A = waves 0-3, B = waves 4-7 of one 8-wave block. The host prints per-wave elapsed cycles, error count
# and timeout flag (row = wave id).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ASM="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"; HIPROOT="${HIPROOT:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
OUT=build/gen3; mkdir -p "$OUT"
LM="$ASM/llvm/bin/llvm-mc"; CLANG="$ASM/llvm/bin/clang"

echo "=== 1) toolchain / ISA: named-barrier opcodes per mcpu ==="
try() { local mcpu="$1" inst="$2"; if printf '\t%s\n' "$inst" | "$LM" -triple=amdgcn-amd-amdhsa -mcpu=$mcpu -show-encoding 2>/tmp/g3.err | grep -q "encoding:"; then echo "ACCEPT $mcpu : $inst"; else echo "REJECT $mcpu : $inst :: $(grep -o 'error:.*' /tmp/g3.err | head -1)"; fi; }
for mc in gfx950 gfx942 gfx1200 gfx1250; do
  for inst in "s_barrier" "s_barrier_signal 1" "s_barrier_wait 1" "s_barrier_leave" "s_barrier_init m0" "s_barrier_signal_isfirst 1" "s_get_barrier_state s0, 1"; do try $mc "$inst"; done
done
echo "--- clang (real assembler path) for gfx950 ---"
for inst in "s_barrier" "s_barrier_signal 1" "s_barrier_wait 1" "s_barrier_init m0"; do printf '\t%s\n' "$inst" > /tmp/g3one.s; if "$CLANG" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -c -o /tmp/g3one.o /tmp/g3one.s 2>/tmp/g3one.err; then echo "CLANG_ACCEPT gfx950: $inst"; else echo "CLANG_REJECT gfx950: $inst :: $(grep -o 'error:.*' /tmp/g3one.err | head -1)"; fi; done
echo "--- extra wave-sync candidates on gfx950 ---"
for inst in "ds_gws_barrier" "s_barrier_join" "s_wakeup_barrier"; do try gfx950 "$inst"; done

echo; echo "=== 2) hardware probe (GPU $HIP_VISIBLE_DEVICES) ==="
python3 tools/gen3_sync_gen.py || exit 1
for k in block_bal soft_bal block_asym soft_asym none_bal soft_badcnt; do
  if ! "$CLANG" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$OUT/$k.co" "$OUT/$k.s" 2>"$OUT/$k.err"; then
    echo "ASSEMBLE FAIL $k :: $(head -4 "$OUT/$k.err")"; exit 1
  fi
done
echo "[gen3] assembled; building host"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tools/gen3_sync_host.cpp -o "$OUT/gen3_host" -I"$HIPROOT/include" -L"$HIPROOT/lib" -lamdhip64 -Wl,-rpath,"$HIPROOT/lib" || exit 1
run_one() { "$OUT/gen3_host" "$OUT/$1.co" "$1"; }
if command -v gpu-lock >/dev/null 2>&1; then
  echo "[gen3] using gpu-lock"
  gpu-lock bash -c "$(declare -f run_one); cd '$PWD'; for k in block_bal soft_bal block_asym soft_asym none_bal soft_badcnt; do run_one \$k; done"
else
  echo "[gen3] gpu-lock not installed; running directly"
  for k in block_bal soft_bal block_asym soft_asym none_bal soft_badcnt; do run_one $k; done
fi
