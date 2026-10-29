#!/usr/bin/env bash
# Run the docs/vendor-spec.md §7.5 acceptance checklist against one of our FMHA kernel variants.
#
# Usage:
#   bash tools/vendor_validate.sh [kernel] [--no-counters]
#     kernel: fmha_fwd_hd128_mxfp8_vs (default), ..._vs128, ..._t, ..._sv
#
# Prints machine-readable VAL key=value lines (structural checks + wall time), the ref_host timing
# lines (vendor vs candidate, same run), the 15-shape reference results and the determinism check.
# §7.5 thresholds are documented in docs/vendor-spec.md; this script only measures.
#
# Env: REF_CO (vendor .co), HIP_VISIBLE_DEVICES (default 5), ROCM_PATH/ROCM_PROF/AMD_LLVM_BIN as in
# the other vendor_* tools, SKIP_COUNTERS=1 to skip the slow profile passes, SKIP_ROWS64=1 to
# skip the 15-shape run at REF_ROWS=64.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K="${1:-fmha_fwd_hd128_mxfp8_vs}"
shift || true
RUN_COUNTERS=1
for a in "$@"; do [ "$a" = "--no-counters" ] && RUN_COUNTERS=0; done
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
ASM_ROC="${ASM_ROC:-$ROC}"
HIPROOT="${HIPROOT:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
REF_CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
OUT="${VENDOR_WORK:-$REPO/build/vendor}"
mkdir -p "$OUT"
cd "$REPO"
unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-5}"
export AMD_CARGO_TARGET="${AMD_CARGO_TARGET:-/tmp/amd-target-occ}"

echo "[validate] candidate: $K"
bash tools/container/cargo.sh run -q --bin amdc -- emit "$K" --backend own \
    --out "/work/build/$K.s" 2>&1 | grep -E "wrote|error" || exit 1
"$ASM_ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 \
    -mcode-object-version=6 -o "$OUT/$K.co" "build/$K.s" || exit 1
cp "$OUT/$K.co" "build/$K.co"

# ---- wall: vendor vs candidate, same run, unprofiled -------------------------------------------------
if [ ! -x build/ref_host72 ]; then
    g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host72 \
        -I"$HIPROOT/include" -L"$HIPROOT/lib" -lamdhip64 -Wl,-rpath,"$HIPROOT/lib" || exit 1
fi
REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=2 REF_ITERS=20 \
    REF_CO2="build/$K.co" REF_KNAME2="$K" \
    ./build/ref_host72 "$REF_CO" 2>&1 | grep -E "timing" || true

# ---- structural checks on the emitted assembly (labels are visible in our .s) -----------------------
python3 - "build/$K.s" <<'PY'
import re, sys
from collections import Counter

path = sys.argv[1]
labels, ins = {}, []
for line in open(path, errors="replace"):
    s = line.rstrip("\n")
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_.$]*):\s*$", s)
    if m:
        labels[m.group(1)] = len(ins)
        continue
    m = re.match(r"^\s+([a-z][A-Za-z0-9_.]*)", s)
    if m:
        ins.append((m.group(1), s.strip()))

print("VAL candidate=%s" % path.split("/")[-1].replace(".s", ""))
print("VAL static_insns=%d" % len(ins))
c = Counter(m for m, _ in ins)
stats = [("s_barrier", "static_barriers"), ("s_setprio", "static_setprio"),
         ("v_smfmac_f32_16x16x128_fp8_fp8", "static_smfmac"),
         ("v_mfma_scale_f32_32x32x64_f8f6f4", "static_mfma_scale"),
         ("v_mfma_f32_32x32x8_f16", "static_msum_f16"),
         ("v_cvt_pknorm_u16_f32", "static_pknorm"),
         ("v_cvt_scalef32_pk_fp8_bf16", "static_scalef32"),
         ("v_perm_b32", "static_perm_lut"), ("v_pk_sub_u16", "static_pk_sub"),
         ("global_load_lds_dwordx4", "static_lds_dma")]
for op, tag in stats:
    print("VAL %s=%d" % (tag, c.get(op, 0)))

# loops: backward branches to labels visible in the .s
backs = []
for i, (m, s) in enumerate(ins):
    if m.startswith("s_branch") or m.startswith("s_cbranch"):
        p = s.split()
        tgt = p[1] if len(p) > 1 else ""
        if tgt in labels and labels[tgt] < i:
            backs.append((labels[tgt], i, tgt))
backs.sort(key=lambda t: t[1] - t[0], reverse=True)
print("VAL loop_copies=%d" % len(backs))

def realign(body):
    n = 0
    for i, (m, s) in enumerate(body):
        if m.startswith("s_cbranch_vccnz"):
            ctx = " ".join(x[1] for x in body[max(0, i - 3):i])
            if "v_cmp_lt_f32" in ctx:
                n += 1
    return n

for k, (a, b, tgt) in enumerate(backs):
    body = ins[a:b + 1]
    cnt = lambda pred: sum(1 for m, s in body if pred(m, s))
    print("VAL loop%d_insns=%d loop%d_barriers=%d loop%d_setprio=%d loop%d_realign=%d "
          "loop%d_dma=%d loop%d_smfmac=%d loop%d_mfma_scale=%d loop%d_msum_f16=%d "
          "loop%d_pknorm=%d loop%d_scalef32=%d loop%d_lut_corr_ops=%d loop%d_wait=%d loop%d_nop=%d" % (
              k, len(body), k, cnt(lambda m, s: m == "s_barrier"),
              k, cnt(lambda m, s: m.startswith("s_setprio")), k, realign(body),
              k, cnt(lambda m, s: "load_lds" in m), k, cnt(lambda m, s: "smfmac" in m),
              k, cnt(lambda m, s: m.startswith("v_mfma_scale")),
              k, cnt(lambda m, s: m.startswith("v_mfma_f32_32x32x8_f16")),
              k, cnt(lambda m, s: "pknorm" in m), k, cnt(lambda m, s: "scalef32" in m),
              k, cnt(lambda m, s: m.startswith(("v_perm_b32", "v_pk_sub_u16"))),
              k, cnt(lambda m, s: m.startswith("s_waitcnt")), k, cnt(lambda m, s: m.startswith("s_nop"))))

# long-sequence path = the loop copies without LUT-correction ops; short copies carry them
lut = [sum(1 for m, s in ins[a:b + 1] if m.startswith(("v_perm_b32", "v_pk_sub_u16"))) for a, b, _ in backs]
if backs:
    print("VAL loop_lut_ops_min=%d loop_lut_ops_max=%d loop_copies_with_lut=%d loop_copies_without_lut=%d" % (
        min(lut), max(lut), sum(1 for x in lut if x > 0), sum(1 for x in lut if x == 0)))
PY

# ---- correctness: 15 reference shapes --------------------------------------------------------------
echo "[validate] 15-shape reference (own backend)"
REF_CO="$REF_CO" OUR_KERNEL="$K" BACKEND=own REF_ROWS="${REF_ROWS:-16}" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
    bash tests/gpu/run_ref.sh > "/tmp/validate_$K.shapes.log" 2>&1
echo "VAL ref_shapes_ok=$(grep -c REF_KERNEL_OK "/tmp/validate_$K.shapes.log") ref_shapes_fail=$(grep -c REF_KERNEL_FAIL "/tmp/validate_$K.shapes.log")"
grep -E "REF_KERNEL_FAIL" "/tmp/validate_$K.shapes.log" | head -3

# ---- determinism ------------------------------------------------------------------------------------
REF_CO="$REF_CO" OUR_KERNEL="$K" BACKEND=own HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
    bash tests/gpu/determinism.sh > "/tmp/validate_$K.det.log" 2>&1
echo "VAL det_ok=$(grep -c DET_OK "/tmp/validate_$K.det.log") det_fail=$(grep -c DET_FAIL "/tmp/validate_$K.det.log")"

# ---- correctness at 64 rows: ours bad must be 0; a shape can still report REF_KERNEL_FAIL because the
#      VENDOR has bad elements there ---------------------------------------------------------------
if [ "${SKIP_ROWS64:-0}" != "1" ]; then
    REF_CO="$REF_CO" OUR_KERNEL="$K" BACKEND=own REF_ROWS=64 HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
        bash tests/gpu/run_ref.sh > "/tmp/validate_$K.rows64.log" 2>&1 || true
    bad_total=$(awk -F"bad=" '/our kernel/ {split($2, a, " "); s += a[1]} END {print s + 0}' "/tmp/validate_$K.rows64.log")
    fail=$(grep -c REF_KERNEL_FAIL "/tmp/validate_$K.rows64.log")
    echo "VAL rows64_ours_bad_total=$bad_total rows64_shapes_ref_fail=$fail (non-zero REF fails only where the vendor has bad elements)"
fi

# ---- counters (optional, slow) ----------------------------------------------------------------------
if [ "$RUN_COUNTERS" = "1" ] && [ "${SKIP_COUNTERS:-0}" != "1" ]; then
    OURS_CO="build/$K.co" OURS_KERNEL="$K" bash tools/vendor_counters.sh "vs-$K" 2>&1 | tail -30
fi
