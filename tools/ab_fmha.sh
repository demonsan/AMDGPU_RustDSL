#!/usr/bin/env bash
# Same-run A/B of FMHA kernel variants (by name) on the filled shapes: emit + assemble each kernel, a 64-row accuracy check per shape
# (ours vs CPU, vendor vs CPU), then REPS interleaved timing rounds; prints the medians.
#   HIP_VISIBLE_DEVICES=5 bash tools/ab_fmha.sh <kernel>... [-- "B S" ...]   (defaults: shapes "4 2048" "1 8192"; REPS=7; ROWS_PER_BLOCK=256)
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
A=${AMDC:-/tmp/target-kfmha/debug/amdc}; V=${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}
KS=(); SH=()
while [ $# -gt 0 ]; do [ "$1" = -- ] && { shift; SH=("$@"); break; }; KS+=("$1"); shift; done
[ ${#SH[@]} = 0 ] && SH=("4 2048" "1 8192")
REPS=${REPS:-7}; RPB=${ROWS_PER_BLOCK:-256}
mkdir -p build/ab
[ -x build/ff/ref_host ] || { mkdir -p build/ff; g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ff/ref_host -I$R/include -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib; }
for k in "${KS[@]}"; do
  $A emit "$k" --backend=own --out "build/ab/$k.s" >/tmp/ab_e.log 2>&1 || { echo "EMIT FAIL $k"; tail -4 /tmp/ab_e.log; exit 1; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/ab/$k.co" "build/ab/$k.s" || exit 1
  v=$($A gpu-verify "$k" --strict 2>&1 | grep -E '^== ' | sed -E 's/proven.*//')
  echo "$k: md5 $(md5sum < build/ab/$k.s | cut -c1-12) $(grep -E '\.vgpr_count:' build/ab/$k.s | tr -s ' ') | $v"
done
run() { HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-5} REF_B=$1 REF_H=8 REF_S=$2 REF_SK=$2 REF_HK=8 REF_CO2=build/ab/$3.co REF_KNAME2=$3 REF_ROWS_PER_BLOCK=$RPB "${@:4}" ./build/ff/ref_host $V 2>&1; }
for s in "${SH[@]}"; do set -- $s
  for k in "${KS[@]}"; do echo "acc B$1 S$2 $k: $(run $1 $2 $k REF_ROWS=64 REF_WARMUP=2 REF_ITERS=2 | grep -E 'kernel +vs CPU: ' | sed -E 's/checked=[0-9]+ //;s/ max\|ref\|=[0-9.]+//;s/ kernel +vs CPU://;s/worst_err=[^ ]+ //' | tr '\n' ' ')"; done
done
for s in "${SH[@]}"; do set -- $s
  for k in "${KS[@]}"; do : > "/tmp/ab_$k"; done
  for r in $(seq 1 $REPS); do for k in "${KS[@]}"; do run $1 $2 $k REF_ROWS=1 REF_WARMUP=150 REF_ITERS=200 | awk '/^timing ours/ {for(i=1;i<=NF;i++) if($i ~ /^TFLOP/) print $(i-1)}' >> "/tmp/ab_$k"; done; done
  echo "== perf B$1 S$2 (median of $REPS)"
  for k in "${KS[@]}"; do echo "  $k $(sort -n /tmp/ab_$k | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')  [$(tr '\n' ' ' < /tmp/ab_$k)]"; done
done
