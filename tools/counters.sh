#!/usr/bin/env bash
# rocprofv3 counter row for one .co of the fmha kernel (ours): bash tools/counters.sh build/ab/X.co [label]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
K="${OUR_KERNEL:-fmha_fwd_hd128_mxfp8_t}"; GPU="${HIP_VISIBLE_DEVICES:-4}"; ALL=""
for SET in "SQ_INSTS_VALU SQ_INSTS_VALU_INT32 SQ_WAIT_ANY SQ_BUSY_CYCLES" "SQ_WAVE_CYCLES SQ_LDS_BANK_CONFLICT SQ_IFETCH SQ_WAIT_INST_ANY"; do
  rm -rf /tmp/prof_c
  REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=4 REF_ITERS=3 REF_CO2="$1" REF_KNAME2=$K HIP_VISIBLE_DEVICES=$GPU timeout 300 $ROC/bin/rocprofv3 --pmc $SET --output-format csv -d /tmp/prof_c -o p -- ./build/ref_host "$CO" > /tmp/prof_c.log 2>&1
  f=$(find /tmp/prof_c -name "*counter_collection.csv" | head -1)
  ALL="$ALL $(python3 - "$f" "$K" <<'EOF'
import csv,sys,collections
agg=collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    if r['Kernel_Name'].startswith(sys.argv[2]): agg[r['Counter_Name']].append(float(r['Counter_Value']))
print(' '.join('%s=%d' % (k.replace('SQ_',''), sum(v)/len(v)) for k,v in agg.items()))
EOF
)"
done
echo "${2:-$1}: $ALL"