#!/usr/bin/env bash
# Collect the same rocprofv3 counter set for the vendor FMHA kernel and for our kernel in one
# process, and print a vendor-vs-ours table (raw mean per dispatch + normalised per wave-iteration).
# This is the measurement behind section 6 of docs/vendor-spec.md and the acceptance checklist in 7.5.
#
# Usage:
#   OURS_CO=build/ours_t.co bash tools/vendor_counters.sh vendor-spec
#
# Env:
#   REF_CO    vendor .co          (default /shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co)
#   OURS_CO   our .co             (default build/ours_t.co; built here with amdc + clang if missing)
#   OURS_KERNEL  kernel name inside OURS_CO (default fmha_fwd_hd128_mxfp8_t)
#   ROCM_PATH ROCm used for clang/llvm-objdump (default 10.1.0); the profiler comes from ROCM_PROF
#             (default /shared/apps/ubuntu/opt/rocm-7.2.4 - rocprofv3 1.1.0; the 10.1.0 one aborts)
#   HIP_VISIBLE_DEVICES  default 5
#   PROFILE_SHAPE   default "1 8 2048 2048" (B H S SK); REF_ITERS default 5
#
# Notes (learned on node 53):
#  * rocprofv3 must be the ROCm-7.2.4 build; rocm-10.1.0's 1.4.1 aborts on this node.
#  * ROCr aborts with "Conflicting visibility of agent" unless GPU_DEVICE_ORDINAL /
#    CUDA_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES are unset before HIP_VISIBLE_DEVICES is set.
#  * Counter passes are split into small groups (<=6 counters): a 20-counter invocation runs for
#    >25 min on this node and starves the queue for everybody.
#  * A counter-pass duration is NOT a performance measurement; the wall numbers in docs/vendor-spec.md
#    come from ref_host's hipEvent timing (unprofiled).
#  * Counters in one --pmc invocation may come from several replays; do not build ratios that mix
#    replays (the kernels are deterministic, so per-counter values are comparable across kernels).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-run}"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
PROF="${ROCM_PROF:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
RP="$PROF/bin/rocprofv3"
REF_CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
OURS_CO="${OURS_CO:-$REPO/build/ours_t.co}"
OUT="${VENDOR_WORK:-$REPO/build/vendor}"
mkdir -p "$OUT/prof"
cd "$REPO"
unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-5}"

# ---- our code object: emit with the default (llvm-tail) build and assemble -----------------------
if [ ! -s "$OURS_CO" ]; then
    K="${OURS_KERNEL:-fmha_fwd_hd128_mxfp8_t}"
    echo "[vendor_counters] emitting $K (default llvm-tail)"
    ASM_ROC="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
    mkdir -p "$(dirname "$OURS_CO")"
    bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" \
        --out /work/build/ours_t.s 2>&1 | grep -E "^error|compile error" && exit 1
    "$ASM_ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
        -mcode-object-version=6 -o "$OURS_CO" build/ours_t.s || exit 1
fi

# ---- host: build against the same ROCm as the profiler ------------------------------------------
if [ ! -x build/ref_host72 ]; then
    g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host72 \
        -I"$PROF/include" -L"$PROF/lib" -lamdhip64 -Wl,-rpath,"$PROF/lib" || exit 1
fi
set -- ${PROFILE_SHAPE:-1 8 2048 2048}
B="$1"; H="$2"; S="$3"; SK="$4"; ITERS="${REF_ITERS:-5}"
COMMON="REF_B=$B REF_H=$H REF_S=$S REF_SK=$SK REF_HK=$H REF_ROWS=2 REF_ITERS=$ITERS \
REF_CO2=$OURS_CO REF_KNAME2=${OURS_KERNEL:-fmha_fwd_hd128_mxfp8_t}"

run_pass() { # $1 tag, rest = counters
    P="$1"; shift
    rm -rf "$OUT/prof/${TAG}_$P"; mkdir -p "$OUT/prof/${TAG}_$P"
    env $COMMON timeout 300 "$RP" --pmc "$@" --output-format csv \
        -d "$OUT/prof/${TAG}_$P" -o r -- ./build/ref_host72 "$REF_CO" \
        > "$OUT/prof/${TAG}_$P.log" 2>&1
    echo "[vendor_counters] $P exit $?  -> $OUT/prof/${TAG}_$P"
}

# 4 small passes (a single 20-counter pass is far too slow on this node)
run_pass PA SQ_INSTS SQ_INSTS_VALU SQ_INSTS_LDS SQ_INSTS_VMEM SQ_IFETCH SQ_BUSY_CYCLES
run_pass PB SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_CVT \
             SQ_INSTS_VALU_INT32 SQ_INSTS_VALU_FMA_F32 SQ_INSTS_SALU
run_pass PC SQ_WAVE_CYCLES SQ_ACTIVE_INST_VALU SQ_VALU_MFMA_BUSY_CYCLES GRBM_GUI_ACTIVE
run_pass PD SQ_WAIT_ANY SQ_LDS_BANK_CONFLICT SQ_INST_LEVEL_LDS SQ_INST_LEVEL_VMEM
rm -rf "$OUT/prof/${TAG}_TR"; mkdir -p "$OUT/prof/${TAG}_TR"
env $COMMON timeout 300 "$RP" --kernel-trace --output-format csv \
    -d "$OUT/prof/${TAG}_TR" -o r -- ./build/ref_host72 "$REF_CO" \
    > "$OUT/prof/${TAG}_TR.log" 2>&1
echo "[vendor_counters] TR exit $?  -> $OUT/prof/${TAG}_TR"

python3 - "$OUT/prof" "$TAG" "$B $H $S $SK" "$ITERS" <<'PY'
import collections, csv, glob, sys
base, tag, shape, iters = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
B, H, S, SK = (int(x) for x in shape.split())
blocks = max(1, S // 256) * H * B          # both kernels: 256 query rows per block
waves = blocks * 8                          # 512 threads = 8 waves
iters = SK // 128                           # one outer iteration per 128 keys
print(f"# shape B{B} H{H} S{S} SK{SK}: blocks={blocks} waves={waves} iterations={iters}")
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for p in ("PA", "PB", "PC", "PD"):
    for f in glob.glob(f"{base}/{tag}_{p}/*counter_collection.csv"):
        for r in csv.DictReader(open(f)):
            k = r.get("Kernel_Name", "")
            nm = "vendor" if k.startswith("_ZN5aiter") else ("ours" if k.startswith("fmha") else None)
            if nm is None:
                continue
            try:
                v = float(r["Counter_Value"])
            except ValueError:
                continue
            agg[nm][r["Counter_Name"]].append(v)
names = sorted({n for nm in agg for n in agg[nm]})
print(f"{'counter':36s} {'vendor':>12s} {'ours':>12s} {'vendor/wave-iter':>17s} {'ours/wave-iter':>15s}")
for n in names:
    vs = sum(agg['vendor'][n]) / len(agg['vendor'][n]) if agg['vendor'].get(n) else None
    os_ = sum(agg['ours'][n]) / len(agg['ours'][n]) if agg['ours'].get(n) else None
    def w(v):
        return f"{v / (waves * iters):.3f}" if v is not None else "-"
    sv = f"{vs:.0f}" if vs is not None else "-"
    so = f"{os_:.0f}" if os_ is not None else "-"
    print(f"{n:36s} {sv:>12s} {so:>12s} {w(vs):>17s} {w(os_):>15s}")
print()
# wall time (unprofiled) for the same shape: ref_host prints per-launch ms + TFLOP/s
print("# wall (unprofiled, 20 launches): run ref_host without a profiler and grep timing")
for f in glob.glob(f"{base}/{tag}_TR/*kernel_trace.csv"):
    rows = list(csv.DictReader(open(f)))
    seen = set()
    for r in rows:
        k = r.get("Kernel_Name", "")
        nm = "vendor" if k.startswith("_ZN5aiter") else ("ours" if k.startswith("fmha") else None)
        if nm is None or nm in seen:
            continue
        seen.add(nm)
        keys = [x for x in ("Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z", "Workgroup_Size_X",
                            "VGPR_Count", "SGPR_Count", "LDS_Block_Size", "Kernel_Duration_ns") if x in r]
        print(f"# trace {nm}: " + " ".join(f"{x}={r[x]}" for x in keys))
PY
