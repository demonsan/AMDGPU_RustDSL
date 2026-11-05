#!/usr/bin/env bash
# Issue-level comparison vendor vs ours: rocprofv3 SQ counters of the aiter code object and of an amdc GEMM kernel, same process, same inputs.
# usage: OURS_KERNEL=gemm_mxfp8_bs_v4 HIP_VISIBLE_DEVICES=4 bash tools/prof_gemm.sh   (repo clone on the node; env as for tools/gemm_ref.sh)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K="${OURS_KERNEL:-gemm_mxfp8_bs_v4}"; R=/shared/apps/ubuntu/opt/rocm-7.0.2
CO_DIR="${GEMM_CO_DIR:-/shared/amdgpu/home/blyu_qle/sessions/aiter_pr6696_co}"
IMG="$CO_DIR/opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_u2_gm16_wide1_vf1.co"
SYM="$(strings -a "$IMG" | grep -m1 -E '^opus_gemm_gfx950_4wave_pin_agpr_co_.*_vf[01]$')"
# build ours + host (a light run)
REF_ITERS=1 REF_WARMUP=1 REF_REPS=1 OURS_TILE_M=256 OURS_TILE_N=256 SKIP_REF=1 REF_ROWS=1 bash tools/gemm_ref.sh > /tmp/prof_build.log 2>&1
export LD_LIBRARY_PATH=$R/lib REF_CO="$IMG" REF_KNAME="$SYM" OURS_CO="build/$K.co" OURS_KNAME="$K" OURS_TILE_M=256 OURS_TILE_N=256 REF_ITERS=${PROF_ITERS:-5} REF_WARMUP=2 REF_REPS=1 REF_ROWS=1
P1="SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_MFMA SQ_INSTS_LDS SQ_INSTS_VMEM"
P2="SQ_WAVES SQ_WAVE_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS SQ_ACTIVE_INST_ANY SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_LDS"
P3="SQ_WAVES SQ_WAVE_CYCLES SQ_ACTIVE_INST_VMEM SQ_ACTIVE_INST_MISC SQ_ACTIVE_INST_SCA SQ_VALU_MFMA_BUSY_CYCLES SQ_INST_CYCLES_VMEM_RD SQ_LDS_BANK_CONFLICT"
rm -rf /tmp/prof_$K; mkdir -p /tmp/prof_$K
i=0; for P in "$P1" "$P2" "$P3"; do i=$((i+1))
  $R/bin/rocprofv3 --pmc $P --output-format csv -d /tmp/prof_$K/p$i -o r -- ./build/gemm_ref_host > /tmp/prof_$K/p$i.log 2>&1 || echo "pass $i failed"
done
python3 - "$K" <<'PY'
import csv,glob,sys,collections
k=sys.argv[1]
disp=collections.defaultdict(dict)   # (who, file, dispatch) -> counter -> value (summed over agents/dims)
for f in glob.glob(f"/tmp/prof_{k}/p*/**/*counter_collection.csv", recursive=True):
    for r in csv.DictReader(open(f)):
        name=r["Kernel_Name"]; who="vendor" if name.startswith("opus_gemm") else ("ours" if k in name else None)
        if who is None: continue
        key=(who,f,r["Dispatch_Id"]); c=r["Counter_Name"]
        disp[key][c]=disp[key].get(c,0)+float(r["Counter_Value"])
agg=collections.defaultdict(lambda: collections.defaultdict(list))
for (who,f,d),cv in disp.items():
    w=cv.get("SQ_WAVES"); cyc=cv.get("SQ_WAVE_CYCLES")
    if not w or not cyc: continue
    for c,v in cv.items():
        if c in ("SQ_WAVES",): continue
        agg[who][c].append((v/w, v/cyc))
names=sorted({c for w in agg for c in agg[w]})
print(f"per-wave counts (and share of wave cycles), vendor vs ours={k}")
print(f"{'counter':28s} {'vendor/wave':>14s} {'ours/wave':>14s} {'ratio':>6s} | {'vendor %cyc':>11s} {'ours %cyc':>10s}")
for c in names:
    v=agg['vendor'].get(c); o=agg['ours'].get(c)
    f=lambda L,i: sum(x[i] for x in L)/len(L) if L else float('nan')
    mv,mo=f(v,0),f(o,0)
    print(f"{c:28s} {mv:14.5g} {mo:14.5g} {mo/mv if mv else float('nan'):6.2f} | {100*f(v,1):10.2f}% {100*f(o,1):9.2f}%")
PY