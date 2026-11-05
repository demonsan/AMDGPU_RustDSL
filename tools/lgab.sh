#!/usr/bin/env bash
# tools/ab_fmha.sh with per-variant EMIT environment: same-run A/B of FMHA kernel variants (emit + assemble each, a 64-row accuracy check per shape, REPS interleaved timing rounds, medians).
#   REF_DATA=normal HIP_VISIBLE_DEVICES=4 AMDC=<amdc> bash tools/lgab.sh <kernel>[@spec][%VAR=val;VAR=val]... [-- "B S" ...]   (defaults: shapes "4 2048" "1 8192"; REPS=7; ROWS_PER_BLOCK=256)
#   `%VAR=val;..` is exported only while that variant is emitted (e.g. `fmha_fwd_hd128_mxfp8_v3_vs128%AMDC_SCHED=1;AMDC_SCHED_OPTS=hoist=64`); variants are labelled by the whole string.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"; cd "$REPO"
R=/shared/apps/ubuntu/opt/rocm-7.0.2; CL=/shared/apps/ubuntu/opt/rocm-10.1.0/llvm/bin/clang
A=${AMDC:-/tmp/target-kfmha/debug/amdc}; V=${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}
KS=(); SH=()
while [ $# -gt 0 ]; do [ "$1" = -- ] && { shift; SH=("$@"); break; }; KS+=("$1"); shift; done
[ ${#SH[@]} = 0 ] && SH=("4 2048" "1 8192")
REPS=${REPS:-7}; RPB=${ROWS_PER_BLOCK:-256}
mkdir -p build/lgab
{ [ -x build/ff/ref_host ] && [ build/ff/ref_host -nt tests/gpu/ref_host.cpp ]; } || { mkdir -p build/ff; g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ff/ref_host -I$R/include -Itests/gpu -I. -L$R/lib -lamdhip64 -Wl,-rpath,$R/lib 2>/dev/null; }
base() { local x="${1%%%*}"; echo "$x"; }                 # kernel[@spec]
kn() { local x="$(base "$1")"; echo "${x%%@*}"; }
ksp() { local x="$(base "$1")"; case "$x" in *@*) echo "--specialize ${x#*@}";; esac; }
kenv() { case "$1" in *%*) echo "${1#*%}" | tr ';' ' ';; esac; }
lab() { echo "$1" | tr -c 'A-Za-z0-9_.-' '_'; }
for k in "${KS[@]}"; do
  f=$(lab "$k")
  env $(kenv "$k") $A emit "$(kn "$k")" --backend=own $(ksp "$k") --out "build/lgab/$f.s" >/tmp/lgab_e.log 2>&1 || { echo "EMIT FAIL $k"; tail -4 /tmp/lgab_e.log; exit 1; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/lgab/$f.co" "build/lgab/$f.s" || exit 1
  v=$($A gpu-verify "$(kn "$k")" --strict $(ksp "$k") 2>&1 | grep -E '^== ' | sed -E 's/proven.*//')
  echo "$k: md5 $(md5sum < build/lgab/$f.s | cut -c1-12) $(grep -E '\.vgpr_count:' build/lgab/$f.s | tr -s ' ') waits0=$(grep -c 's_waitcnt lgkmcnt(0)' build/lgab/$f.s) | $v"
done
run() { local f=$(lab "$3"); env HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-5} REF_B=$1 REF_H=8 REF_S=$2 REF_SK=$2 REF_HK=8 REF_CO2=build/lgab/$f.co REF_KNAME2=$(kn "$3") REF_ROWS_PER_BLOCK=$RPB "${@:4}" ./build/ff/ref_host $V 2>&1; }
for s in "${SH[@]}"; do set -- $s
  for k in "${KS[@]}"; do echo "acc B$1 S$2 $k: $(run $1 $2 $k REF_ROWS=64 REF_WARMUP=2 REF_ITERS=2 | grep -E 'kernel +vs CPU: ' | sed -E 's/checked=[0-9]+ //;s/ max\|ref\|=[0-9.]+//;s/ kernel +vs CPU://;s/worst_err=[^ ]+ //' | tr '\n' ' ')"; done
done
for s in "${SH[@]}"; do set -- $s
  for k in "${KS[@]}"; do : > "/tmp/lgab_$(lab "$k")"; done
  for r in $(seq 1 $REPS); do for k in "${KS[@]}"; do run $1 $2 $k REF_ROWS=1 REF_WARMUP=150 REF_ITERS=200 | awk '/^timing ours/ {for(i=1;i<=NF;i++) if($i ~ /^TFLOP/) print $(i-1)}' >> "/tmp/lgab_$(lab "$k")"; done; done
  echo "== perf B$1 S$2 (median of $REPS)"
  for k in "${KS[@]}"; do f=$(lab "$k"); echo "  $k $(sort -n /tmp/lgab_$f | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')  [$(tr '\n' ' ' < /tmp/lgab_$f)]"; done
done