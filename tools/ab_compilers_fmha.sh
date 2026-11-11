#!/usr/bin/env bash
# Same-run A/B of two COMPILERS on the MXFP8 FMHA kernels (vendor ABI harness tests/gpu/ref_host.cpp): each kernel is emitted by
# AMDC_A and AMDC_B, accuracy-checked, then timed in interleaved rounds; prints the medians per compiler.
#   AMDC_A=build/amdc_old AMDC_B=build/amdc_new HIP_VISIBLE_DEVICES=2 bash tools/ab_compilers_fmha.sh <kernel>... [-- "B S" ...]
set -u
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-2}"; case ",$HIP_VISIBLE_DEVICES," in *,[4-7],*) echo "GPUs 4-7 are not to be used (user rule): HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"; exit 2;; esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
V=${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}
KS=(); SH=()
while [ $# -gt 0 ]; do [ "$1" = -- ] && { shift; SH=("$@"); break; }; KS+=("$1"); shift; done
[ ${#SH[@]} = 0 ] && SH=("4 2048" "1 8192")
REPS=${REPS:-7}
mkdir -p build/abc build/ff
{ [ -x build/ff/ref_host ] && [ build/ff/ref_host -nt tests/gpu/ref_host.cpp ]; } || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ff/ref_host -I$R/include -Itests/gpu -I. -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib 2>/dev/null
for k in "${KS[@]}"; do for c in A B; do
  bin=AMDC_$c; "${!bin}" emit "$k" --backend=own --out "build/abc/$k.$c.s" > /dev/null 2>&1 || { echo "EMIT FAIL $k $c"; exit 1; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/abc/$k.$c.co" "build/abc/$k.$c.s" || exit 1
  echo "$k $c: md5 $(md5sum < build/abc/$k.$c.s | cut -c1-12) s_nop $(grep -c s_nop build/abc/$k.$c.s)"
done; done
run() { env HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-2} REF_B=$1 REF_H=8 REF_S=$2 REF_SK=$2 REF_HK=8 REF_CO2=build/abc/$3.$4.co REF_KNAME2=$3 REF_ROWS_PER_BLOCK=${ROWS_PER_BLOCK:-256} "${@:5}" ./build/ff/ref_host $V 2>&1; }
for s in "${SH[@]}"; do set -- $s
  for k in "${KS[@]}"; do for c in A B; do echo "acc B$1 S$2 $k $c: $(run $1 $2 $k $c REF_ROWS=64 REF_WARMUP=2 REF_ITERS=2 | grep -E 'kernel +vs CPU: ' | sed -E 's/checked=[0-9]+ //;s/ kernel +vs CPU://' | tr '\n' ' ' | cut -c1-120)"; done; done
  for k in "${KS[@]}"; do for c in A B; do : > "/tmp/abc_$k.$c"; done; done
  for r in $(seq 1 $REPS); do for k in "${KS[@]}"; do for c in A B; do run $1 $2 $k $c REF_ROWS=1 REF_WARMUP=150 REF_ITERS=200 | awk '/^timing ours/ {for(i=1;i<=NF;i++) if($i ~ /^TFLOP/) print $(i-1)}' >> "/tmp/abc_$k.$c"; done; done; done
  echo "== perf B$1 S$2 (TFLOP/s, median of $REPS)"
  for k in "${KS[@]}"; do for c in A B; do echo "  $k $c $(sort -n /tmp/abc_$k.$c | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')  [$(tr '\n' ' ' < /tmp/abc_$k.$c)]"; done; done
done
