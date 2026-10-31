#!/usr/bin/env bash
# gen9 acceptance gates for `_vs9` (docs/gen9-vendor-mirror-design.md). gen9 output is NOT bit-identical to `_vs128` (different reference math): the gate is
# accuracy / vendor parity (tools/gen9_gate.py), determinism, and per-CU throughput milestones, not an md5.
#   REF_CO=/path/vendor.co [GEN9_KERNEL=fmha_fwd_hd128_mxfp8_vs9] [GEN9_MD5=<md5>  (step 0 only: output md5 of B1H8S2048 via REF_DUMP2)] bash tests/gpu/gen9_gate.sh
# Gates: (1) 15 shapes at REF_ROWS=64 (and 16): mean err <= 2.6e-3, bad <= the vendor's bad on the same shape, no NaN; (2) determinism 3 x 5 shapes;
# (3) per-CU at 64 blocks (B1H8S2048, >= 8 interleaved rounds vs `_vs128`): milestones M2 >= 460, M3 >= 480, M4 >= 510, M5 >= 537.6 TFLOP/s (8.4 TF/CU);
# (4) filled shapes B4H8S2048 / B1H8S8192 must not regress vs `_vs128` (same-run ratio printed).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"; mkdir -p build
REF_CO="${REF_CO:?set REF_CO}"; K="${GEN9_KERNEL:-fmha_fwd_hd128_mxfp8_vs9}"; export BACKEND="${BACKEND:-own}" HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
rc=0
for R in 64 16; do
    REF_ROWS=$R OUR_KERNEL=$K bash tests/gpu/run_ref.sh > build/gen9_ref_$R.log 2>&1
    python3 tools/gen9_gate.py build/gen9_ref_$R.log | tail -17 | sed "s/^/[rows $R] /" || rc=1
    python3 tools/gen9_gate.py build/gen9_ref_$R.log >/dev/null || rc=1
done
det=$(for r in 1 2 3; do OUR_KERNEL=$K bash tests/gpu/determinism.sh 2>&1 | grep -c DET_OK; done | tr '\n' ' '); echo "determinism (5 shapes x3 runs): $det"; [[ "$det" == "5 5 5 " ]] || rc=1
bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_vs128 --backend own --out /work/build/g9_base.s >/dev/null 2>&1
bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" --backend own --out /work/build/g9_new.s >/dev/null 2>&1
CLANG="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
for T in base new; do $CLANG -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/g9_$T.co build/g9_$T.s; done
tf() { REF_B=$1 REF_H=$2 REF_HK=$2 REF_S=$3 REF_SK=$3 REF_ROWS=1 REF_ITERS=200 REF_CO2=build/g9_$4.co REF_KNAME2=$( [ $4 = base ] && echo fmha_fwd_hd128_mxfp8_vs128 || echo $K ) ./build/ref_host "$REF_CO" 2>&1 | grep -E "^timing ours" | sed -E 's/.*ms.iter, ([0-9.]+) TFLOP.*/\1/'; }
med() { sort -n | awk '{a[NR]=$1} END{printf "%.1f", a[int((NR+1)/2)]}'; }
for CFG in "1 8 2048" "4 8 2048" "1 8 8192"; do set -- $CFG; b=$(for r in 1 2 3 4 5 6 7 8; do tf $1 $2 $3 base; done | med); n=$(for r in 1 2 3 4 5 6 7 8; do tf $1 $2 $3 new; done | med)
    echo "B$1H$2S$3: _vs128 $b  $K $n TFLOP/s  ratio $(python3 -c "print(round($n/$b,3))")  per-CU $(python3 -c "print(round($n/256 if $1*$2*(($3+255)//256)>=256 else $n/($1*$2*(($3+255)//256)),2))") TF/CU"; done
if [ -n "${GEN9_MD5:-}" ]; then REF_DUMP2=build/g9.bin REF_B=1 REF_H=8 REF_HK=8 REF_S=2048 REF_SK=2048 REF_ROWS=16 REF_ITERS=1 REF_CO2=build/g9_new.co REF_KNAME2=$K ./build/ref_host "$REF_CO" >/dev/null 2>&1; m=$(md5sum build/g9.bin | cut -c1-32); echo "step-0 md5 $m (want $GEN9_MD5)"; [[ "$m" == "$GEN9_MD5" ]] || rc=1; fi
echo "GEN9_GATE rc=$rc"; exit $rc
