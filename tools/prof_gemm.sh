#!/usr/bin/env bash
# Issue-level comparison vendor vs ours: rocprofv3 SQ counters of the aiter code object and of an amdc GEMM kernel, same process, same inputs.
# usage: OURS_KERNEL=gemm_mxfp8_bs_v3 HIP_VISIBLE_DEVICES=4 bash tools/prof_gemm.sh   (repo clone on the node; env as for tools/gemm_ref.sh)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K="${OURS_KERNEL:-gemm_mxfp8_bs_v3}"; R=/shared/apps/ubuntu/opt/rocm-7.0.2
CO_DIR="${GEMM_CO_DIR:-/shared/amdgpu/home/blyu_qle/sessions/aiter_pr6696_co}"
IMG="$CO_DIR/opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_u2_gm16_wide1_vf1.co"
SYM="$(strings -a "$IMG" | grep -m1 -E '^opus_gemm_gfx950_4wave_pin_agpr_co_.*_vf[01]$')"
# build ours + host (a light run)
REF_ITERS=1 REF_WARMUP=1 REF_REPS=1 OURS_TILE_M=256 OURS_TILE_N=256 SKIP_REF=1 REF_ROWS=1 bash tools/gemm_ref.sh > /tmp/prof_build.log 2>&1
export LD_LIBRARY_PATH=$R/lib REF_CO="$IMG" REF_KNAME="$SYM" OURS_CO="build/$K.co" OURS_KNAME="$K" OURS_TILE_M=256 OURS_TILE_N=256 REF_ITERS=${PROF_ITERS:-5} REF_WARMUP=2 REF_REPS=1 REF_ROWS=1
P1="SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_MFMA SQ_INSTS_LDS SQ_INSTS_VMEM"
P2="SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS SQ_ACTIVE_INST_ANY SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_SCA"
P3="SQ_ACTIVE_INST_MISC SQ_ACTIVE_INST_FLAT SQ_VALU_MFMA_BUSY_CYCLES SQ_INSTS_SMEM SQ_INST_CYCLES_VMEM_RD SQ_LDS_BANK_CONFLICT SQ_LDS_ADDR_CONFLICT SQ_INSTS_VMEM_RD"
rm -rf /tmp/prof_$K; mkdir -p /tmp/prof_$K
i=0; for P in "$P1" "$P2" "$P3"; do i=$((i+1))
  $R/bin/rocprofv3 --pmc $P --output-format csv -d /tmp/prof_$K/p$i -o r -- ./build/gemm_ref_host > /tmp/prof_$K/p$i.log 2>&1 || echo "pass $i failed"
done
python3 - "$K" <<'PY'
import csv,glob,sys,collections
k=sys.argv[1]; agg=collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"/tmp/prof_{k}/p*/**/*counter_collection.csv", recursive=True):
    for r in csv.DictReader(open(f)):
        name=r["Kernel_Name"]; who="vendor" if name.startswith("opus_gemm") else "ours"
        agg[who][r["Counter_Name"]].append(float(r["Counter_Value"]))
cs=sorted({c for w in agg for c in agg[w]})
print(f"{'counter':28s} {'vendor':>16s} {'ours('+k[-2:]+')':>16s} ratio")
for c in cs:
    v=agg['vendor'].get(c); o=agg['ours'].get(c)
    mv=sum(v)/len(v) if v else float('nan'); mo=sum(o)/len(o) if o else float('nan')
    print(f"{c:28s} {mv:16.4g} {mo:16.4g} {mo/mv if mv else float('nan'):6.2f}")
PY