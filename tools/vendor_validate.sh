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
# the other vendor_* tools, SKIP_COUNTERS=1 to skip the slow profile passes.
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

# ---- structural checks on the emitted assembly -------------------------------------------------------
"$ASM_ROC/llvm/bin/llvm-objdump" -d --mcpu=gfx950 --no-show-raw-insn "build/$K.co" \
    > "$OUT/$K.objdump.txt" 2>/dev/null
python3 - "$OUT/$K.objdump.txt" <<'PY'
import re, sys
from collections import Counter

path = sys.argv[1]
LAB = re.compile(r"^([0-9a-fA-F]{6,16}) <(.+)>:")
INS = re.compile(r"^\s+([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*//\s*([0-9a-fA-F]{6,16}):")
ins, labels = [], {}
for line in open(path, errors="replace"):
    m = LAB.match(line)
    if m:
        labels[int(m.group(1), 16)] = m.group(2)
        continue
    m = INS.match(line)
    if m:
        ins.append((int(m.group(3), 16), m.group(1), m.group(2).strip()))
ins.sort()

def cls(m):
    if m.startswith(("v_mfma", "v_smfmac")):
        return "MFMA"
    if m.startswith("ds_"):
        return "LDS"
    if m.startswith(("global_", "buffer_", "scratch_", "flat_")):
        return "VMEM"
    if m.startswith("s_"):
        return "CTRL" if m in {"s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccnz",
                               "s_cbranch_vccz", "s_endpgm", "s_barrier", "s_setprio", "s_nop",
                               "s_waitcnt", "s_sleep"} else "SALU"
    return "VALU"

# loops: backward branches to a known label
loops = []
for i, (a, m, ops) in enumerate(ins):
    if m.startswith(("s_branch", "s_cbranch")):
        t = ops.split()[0].strip() if ops else ""
        if t in labels and labels[t] <= a:
            j = next(k for k, x in enumerate(ins) if x[0] == labels[t])
            loops.append((j, i))
loops.sort(key=lambda t: t[1] - t[0], reverse=True)

tuples = re.compile(r"v\[(\d+):(\d+)\]")

def regs_of(ops):
    out = []
    for m in tuples.finditer(ops):
        out.append((int(m.group(1)), int(m.group(2))))
    return out

def body(j0, j1):
    return ins[j0:j1 + 1]

print("VAL candidate=%s" % path.split("/")[-1].replace(".objdump.txt", ""))
c = Counter(cls(m) for _, m, _ in ins)
print("VAL static_insns=%d" % len(ins))
print("VAL static_mfma=%d static_smfmac=%d static_lds=%d static_vmem=%d" % (
    c["MFMA"], sum(1 for _, m, _ in ins if m.startswith("v_smfmac")), c["LDS"], c["VMEM"]))
print("VAL static_barriers=%d static_setprio=%d static_dual_mfma=%d" % (
    sum(1 for _, m, _ in ins if m == "s_barrier"),
    sum(1 for _, m, _ in ins if m == "s_setprio"),
    sum(1 for _, m, _ in ins if m.startswith("v_mfma_f32_32x32x64_f8f6f4"))))
print("VAL static_lds_dma=%d" % sum(1 for _, m, ops in ins if "load_lds" in m or (" lds" in ops and m.startswith("buffer_load"))))
print("VAL static_next_tile_qk=%d" % sum(1 for _, m, _ in ins if m.startswith("v_mfma_scale")))
print("VAL static_smfmac=%d" % sum(1 for _, m, _ in ins if m.startswith("v_smfmac")))

for li, (j0, j1) in enumerate(loops[:3]):
    b = body(j0, j1)
    cc = Counter(cls(m) for _, m, _ in b)
    names = [n for a, n in sorted(labels.items()) if ins[j0][0] <= a <= ins[j1][0]]
    print("VAL loop%d_insns=%d loop%d_barriers=%d loop%d_setprio=%d loop%d_mfma_scale=%d "
          "loop%d_mfma_unscaled=%d loop%d_smfmac=%d loop%d_lds_dma=%d" % (
              li, len(b), li, sum(1 for _, m, _ in b if m == "s_barrier"),
              li, sum(1 for _, m, _ in b if m == "s_setprio"),
              li, sum(1 for _, m, _ in b if m.startswith("v_mfma_scale")),
              li, sum(1 for _, m, _ in b if m.startswith("v_mfma_f32_32x32x64")),
              li, sum(1 for _, m, _ in b if m.startswith("v_smfmac")),
              li, sum(1 for _, m, ops in b if " lds" in ops and m.startswith("buffer_load"))))

# smfmac operand hazards: A/B/C must not have been written by a non-XDL (VALU) instruction less than
# 2 required waits (= 8 clocks, ISA table 38 p74) before the smfmac.
def cycles_between(w, sm):
    n = 0
    for k in range(w + 1, sm):
        m = ins[k][1]
        if m == "s_nop":
            ops = ins[k][2]
            try:
                n += int(ops.strip()) + 1
            except ValueError:
                n += 1
        else:
            n += 1
    return n

cviol = aviol = bviol = 0
smfmacs = 0
for i, (a, m, ops) in enumerate(ins):
    if not m.startswith("v_smfmac"):
        continue
    smfmacs += 1
    ranges = regs_of(ops)
    for ri in (0, 1, 2):  # D (implicit C), A, B
        if ri >= len(ranges):
            continue
        lo, hi = ranges[ri]
        w = None
        for k in range(i - 1, max(-1, i - 12), -1):
            mk, opsk = ins[k][1], ins[k][2]
            if mk.startswith(("v_mfma", "v_smfmac")):
                continue
            wr = regs_of(opsk)
            first = opsk.split(",")[0] if "," in opsk else opsk
            mrange = tuples.findall(first.replace(" ", ""))
            wrote = False
            for x in mrange:
                x0, x1 = int(x[0]), int(x[1])
                if not (x1 < lo or x0 > hi):
                    wrote = True
            if wrote:
                w = k
                break
        if w is not None and ins[w][1].startswith("v_"):
            if cycles_between(w, i) < 8:
                if ri == 0:
                    cviol += 1
                elif ri == 1:
                    aviol += 1
                else:
                    bviol += 1
print("VAL smfmac_count=%d" % smfmacs)
print("VAL smfmac_d_c_from_valu_closer_than_8cyc=%d" % cviol)
print("VAL smfmac_a_from_valu_closer_than_8cyc=%d" % aviol)
print("VAL smfmac_b_from_valu_closer_than_8cyc=%d" % bviol)

# chained smfmacs (same D tuple as the previous smfmac within a few instructions)
chain = 0
for i, (a, m, ops) in enumerate(ins):
    if not m.startswith("v_smfmac"):
        continue
    d = regs_of(ops)[:1]
    for k in range(i - 1, max(-1, i - 6), -1):
        if ins[k][1].startswith("v_smfmac") and regs_of(ins[k][2])[:1] == d:
            chain += 1
            break
print("VAL smfmac_chained=%d" % chain)
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

# ---- counters (optional, slow) ----------------------------------------------------------------------
if [ "$RUN_COUNTERS" = "1" ] && [ "${SKIP_COUNTERS:-0}" != "1" ]; then
    OURS_CO="build/$K.co" OURS_KERNEL="$K" bash tools/vendor_counters.sh "vs-$K" 2>&1 | tail -30
fi
