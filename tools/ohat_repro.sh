#!/usr/bin/env bash
# Launch-to-launch determinism of the forced KV-split path on B1H2S200 SK1337 (gqa2, s=4): N separate ref_host launches, md5 histogram of the dumped output.
# Usage: FMHA_OPT=... [BACKEND=own] [TAG=x] [RUNS=100] bash tools/ohat_repro.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
cd "$REPO"; mkdir -p build
TAG="${TAG:-x}"; REF_CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
P=fmha_fwd_hd128_mxfp8
CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
build_one() { bash tools/container/cargo.sh run -q --bin amdc -- emit "$1" --backend "${BACKEND:-own}" --out "/work/build/oh_${TAG}_$1.s" >/dev/null 2>&1 || { echo "emit $1 failed"; exit 1; }
  if [ "$1" = "${P}_kvsplit" ] && [ -n "${ASM_SED:-}" ]; then sed -i -E "$ASM_SED" "build/oh_${TAG}_$1.s"; fi
  if [ "$1" = "${P}_kvsplit" ] && [ -n "${ASM_PERL:-}" ]; then perl -0pi -e "$ASM_PERL" "build/oh_${TAG}_$1.s"; fi
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "build/oh_${TAG}_$2.co" "build/oh_${TAG}_$1.s" || exit 1; }
build_one ${P}_vs128 main; build_one ${P}_t128 small; build_one ${P}_kvsplit split; build_one ${P}_kvsplit_combine8h comb
[ -x build/ref_host ] || g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib"
rm -f build/oh_${TAG}.sums
for i in $(seq 1 "${RUNS:-100}"); do
  REF_B=1 REF_H=2 REF_S=200 REF_SK=1337 REF_HK=1 REF_ROWS=4 REF_ITERS=1 REF_DUMP2=build/oh_${TAG}.bin \
    REF_CO2=build/oh_${TAG}_main.co REF_KNAME2=${P}_vs128 REF_CO2_SMALL=build/oh_${TAG}_small.co \
    REF_CO2_SPLIT=build/oh_${TAG}_split.co REF_KNAME2_SPLIT=${P}_kvsplit REF_CO2_COMBINE=build/oh_${TAG}_comb.co REF_KNAME2_COMBINE=${P}_kvsplit_combine8h \
    REF_COMBINE_ELEMS=4 REF_SPLIT=4 HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4}" ./build/ref_host "$REF_CO" >/dev/null 2>&1
  md5sum build/oh_${TAG}.bin | cut -c1-12 >> build/oh_${TAG}.sums
done
echo "[$TAG] FMHA_OPT='${FMHA_OPT:-}' backend=${BACKEND:-own} runs=${RUNS:-100}: $(sort build/oh_${TAG}.sums | uniq -c | sort -rn | awk '{printf "%s x%s  ", $2, $1}')"