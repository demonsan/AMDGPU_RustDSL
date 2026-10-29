#!/usr/bin/env bash
# LLVM-tail conformance (docs/llvm-tail-conformance.md). Runs, for one pinned LLVM tree:
#   1. lit baseline of the tail-pass tests (test/CodeGen/AMDGPU, MIR tests whose RUN line uses a tail pass)
#   2. the same tests after every `body: |` was round-tripped through our MIR importer/printer
#   3. the opcode-table report against the instructions of those tests
#   4. tail idempotence + -verify-machineinstrs on the MIR exported for every kernel of ours
# Required env (defaults are the ROCm 10.1.0 pin on the node):
#   LLVM_SRC    llvm-project checkout (contains llvm/)           LLVM_BUILD  cmake build dir (lit.site.cfg.py, configure only)
#   LLVM_TOOLS  dir with llc + FileCheck/not/count/...           LLVM_ROCM   ROCm tree whose libLLVM.so llc uses
#   AMDC        path to the amdc binary                          WORK        scratch dir (default /tmp/llvm-conf)
# Usage: bash tools/llvm_conformance.sh [lit|table|idem|all]   (default all)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_SRC="${LLVM_SRC:-/shared/amdgpu/home/blyu_qle/llvm-src/rocm-10.1.0}"
LLVM_BUILD="${LLVM_BUILD:-/shared/amdgpu/home/blyu_qle/llvm-src/build-rocm-10.1.0}"
LLVM_TOOLS="${LLVM_TOOLS:-/shared/amdgpu/home/blyu_qle/llvm-tools/rocm-10.1.0/bin}"
LLVM_ROCM="${LLVM_ROCM:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
AMDC="${AMDC:-${AMD_CARGO_TARGET:-/tmp/amd-target-backend}/debug/amdc}"
WORK="${WORK:-/tmp/llvm-conf}"
OUT="${OUT:-$WORK/results}"
JOBS="${JOBS:-$(nproc)}"
WHAT="${1:-all}"
TEST_SUB=test/CodeGen/AMDGPU
TAIL_PASSES='post-RA-hazard-rec|si-insert-waitcnts|amdgpu-insert-delay-alu|amdgpu-wait-sgpr-hazards|si-memory-legalizer|si-pre-emit-peephole|si-insert-hard-clauses|si-mode-register|amdgpu-lower-vgpr-encoding|gcn-create-vopd|branch-relaxation|si-late-branch-lowering'
mkdir -p "$OUT"
LIT="python3 $LLVM_SRC/llvm/utils/lit/lit.py"

selected_tests() {  # basename list of MIR tests with a tail pass in a RUN line
    (cd "$LLVM_SRC/llvm/$TEST_SUB" && grep -lE "^# RUN:.*($TAIL_PASSES)" *.mir | sort)
}

make_mirror() {  # $1 = dir, $2 = tools dir ; a copy of llvm/test plus symlinks to the rest of llvm/, with a lit site config pointing at it
    local M="$1" TD="$2"
    rm -rf "$M"; mkdir -p "$M/llvm" "$M/build/test"
    # llvm/test is 1.2 GB: copy only the config files and CodeGen/AMDGPU, symlink the rest.
    local T="$LLVM_SRC/llvm/test"
    mkdir -p "$M/llvm/test/CodeGen"
    for e in "$T"/*; do b=$(basename "$e"); if [ -d "$e" ]; then [ "$b" = CodeGen ] || ln -s "$e" "$M/llvm/test/$b"; else cp "$e" "$M/llvm/test/$b"; fi; done
    for e in "$T"/CodeGen/*; do b=$(basename "$e"); if [ -d "$e" ]; then [ "$b" = AMDGPU ] || ln -s "$e" "$M/llvm/test/CodeGen/$b"; else cp "$e" "$M/llvm/test/CodeGen/$b"; fi; done
    cp -r "$T/CodeGen/AMDGPU" "$M/llvm/test/CodeGen/AMDGPU"
    for e in "$LLVM_SRC"/llvm/*; do b=$(basename "$e"); [ "$b" = test ] || ln -s "$e" "$M/llvm/$b"; done
    for e in "$LLVM_SRC"/*; do b=$(basename "$e"); [ "$b" = llvm ] || ln -s "$e" "$M/$b"; done
    sed -e "s#^config.llvm_src_root = .*#config.llvm_src_root = '$M/llvm'#" \
        -e "s#^config.llvm_obj_root = .*#config.llvm_obj_root = '$M/build'#" \
        -e "s#^config.llvm_tools_dir = .*#config.llvm_tools_dir = '$TD'#" \
        -e "s#^config.llvm_shlib_dir = .*#config.llvm_shlib_dir = '$LLVM_ROCM/lib/llvm/lib'#" \
        "$LLVM_BUILD/test/lit.site.cfg.py" > "$M/build/test/lit.site.cfg.py"
    mkdir -p "$M/build/test/CodeGen/AMDGPU"
}

mk_fc_tools() {  # $1 = dir, $2 = record|compare : tools dir whose FileCheck is the differential wrapper
    rm -rf "$1"; mkdir -p "$1"
    for e in "$LLVM_TOOLS"/*; do b=$(basename "$e"); [ "$b" = FileCheck ] || ln -s "$e" "$1/$b"; done
    printf '#!/bin/sh\nexport REAL_FILECHECK=%s GOLD_DIR=%s FC_MODE=%s\nexec python3 %s/tools/filecheck_diff.py "$@"\n' \
        "$(readlink -f "$LLVM_TOOLS/FileCheck")" "$WORK/gold" "$2" "$REPO" > "$1/FileCheck"
    chmod +x "$1/FileCheck"
}

run_lit() {  # $1 = mirror ; $2 = result json ; stdin = basenames
    local M="$1" J="$2" re
    re="CodeGen/AMDGPU/($(paste -sd'|' | sed 's/\./\\./g'))\$"
    (cd "$M/build" && $LIT -j"$JOBS" -q -a --filter="$re" -o "$J" "$M/build/test/CodeGen/AMDGPU" > "${J%.json}.log" 2>&1)
}

summ() { python3 - "$@" <<'EOF'
import json,sys,collections
d=json.load(open(sys.argv[1])); c=collections.Counter(t["code"] for t in d["tests"]); print(dict(c))
EOF
}

roundtrip_all() {  # $1 = mirror ; $2 = level flag ; fills $OUT/roundtrip-stats-<level>.txt
    local M="$1" LV="$2" ST="$OUT/roundtrip-stats$3.txt"
    : > "$ST"
    while read -r f; do
        p="$M/llvm/$TEST_SUB/$f"
        echo -n "$f " >> "$ST"
        "$AMDC" mir-roundtrip "$LLVM_SRC/llvm/$TEST_SUB/$f" "$p.new" $LV >> "$ST" 2>&1 && mv "$p.new" "$p"
    done < "$OUT/selected.txt"
}

if [ "$WHAT" = lit ] || [ "$WHAT" = all ]; then
    echo "== lit: baseline, then identity / flags / all round trips"
    selected_tests > "$OUT/selected.txt"; echo "selected tail-pass MIR tests: $(wc -l < "$OUT/selected.txt")"
    rm -rf "$WORK/gold"; mkdir -p "$WORK/gold"
    mk_fc_tools "$WORK/tools-record" record
    mk_fc_tools "$WORK/tools-compare" compare
    make_mirror "$WORK/base" "$WORK/tools-record"
    run_lit "$WORK/base" "$OUT/lit-baseline.json" < "$OUT/selected.txt"
    echo -n "baseline (records the goldens): "; summ "$OUT/lit-baseline.json"
    # identity round trip with the REAL FileCheck: do the original CHECK lines still hold?
    make_mirror "$WORK/rt-id-real" "$LLVM_TOOLS"; roundtrip_all "$WORK/rt-id-real" "" "-identity"
    run_lit "$WORK/rt-id-real" "$OUT/lit-identity-check.json" < "$OUT/selected.txt"
    echo -n "identity round trip, original CHECK lines: "; summ "$OUT/lit-identity-check.json"
    for LV in identity flags reflag all; do
        make_mirror "$WORK/rt-$LV" "$WORK/tools-compare"; roundtrip_all "$WORK/rt-$LV" "--level=$LV" "-$LV"
        run_lit "$WORK/rt-$LV" "$OUT/lit-diff-$LV.json" < "$OUT/selected.txt"
        echo -n "level=$LV, differential vs baseline output: "; summ "$OUT/lit-diff-$LV.json"
    done
    python3 - "$OUT" <<'EOF'
import json,sys,re
o=sys.argv[1]
def load(p): return {t["name"]:t["code"] for t in json.load(open(p))["tests"]}
b=load(o+"/lit-baseline.json")
def stats(p):
    st={}
    for l in open(p):
        m=re.match(r"(\S+) STATS files=1 bodies=(\d+) insts=(\d+) parsed=(\d+) opaque=(\d+) bundles=(\d+) changed=(\d+)",l)
        if m: st["LLVM :: CodeGen/AMDGPU/"+m.group(1)]=tuple(map(int,m.groups()[1:]))
    return st
st=stats(o+"/roundtrip-stats-identity.txt")
imp=[n for n,s in st.items() if s[0]>0 and s[3]==0]
tot=[sum(s[i] for s in st.values()) for i in range(6)]
print("tests=%d baseline PASS=%d; fully importable (bodies>0, no opaque instruction)=%d" % (len(b), sum(v=="PASS" for v in b.values()), len(imp)))
print("instructions=%d parsed=%d opaque=%d bundles=%d lines-changed-by-printing=%d" % (tot[1],tot[2],tot[3],tot[4],tot[5]))
for name,p in (("identity/original CHECK","lit-identity-check"),("identity/differential","lit-diff-identity"),("flags/differential","lit-diff-flags"),("reflag/differential","lit-diff-reflag"),("all/differential","lit-diff-all")):
    r=load(o+"/%s.json"%p)
    reg=sorted(n for n in b if b[n]=="PASS" and r.get(n)!="PASS")
    print("%-26s PASS %d/%d ; importable PASS %d/%d ; mismatching: %d" % (name, sum(v=="PASS" for v in r.values()), len(b), sum(r.get(n)=="PASS" for n in imp), len(imp), len(reg)))
    open(o+"/failing-%s.txt"%p,"w").write("\n".join(n.split("::")[-1].strip() for n in reg)+"\n")
EOF
fi
if [ "$WHAT" = census ] || [ "$WHAT" = all ]; then
    echo "== census of test/CodeGen/AMDGPU/*.mir"
    python3 "$REPO/tools/mir_census.py" "$LLVM_SRC/llvm/$TEST_SUB" | tee "$OUT/census.txt"
fi

if [ "$WHAT" = table ] || [ "$WHAT" = all ]; then
    echo "== opcode table vs LLVM's own MIR"
    selected_tests > "$OUT/selected.txt"
    (cd "$LLVM_SRC/llvm/$TEST_SUB" && "$AMDC" mir-table-report $(cat "$OUT/selected.txt")) > "$OUT/table-report.txt"
    head -3 "$OUT/table-report.txt"
fi

if [ "$WHAT" = idem ] || [ "$WHAT" = all ]; then
    echo "== tail idempotence + verifier on exported kernels (AMDC_SCHED=${AMDC_SCHED:-0} AMDC_SCHED_OPTS=${AMDC_SCHED_OPTS:-})"
    mkdir -p "$OUT/idem"; ok=0; bad=0
    LLC="$LLVM_TOOLS/llc"; ARGS="-mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O2"
    for K in $("$AMDC" list); do
        "$AMDC" emit "$K" --mir-out "$OUT/idem/$K.mir" 2>/dev/null || { echo "EXPORT FAIL $K"; bad=$((bad+1)); continue; }
        if ! $LLC $ARGS -verify-machineinstrs -start-before=gcn-create-vopd -stop-after=amdgpu-insert-delay-alu "$OUT/idem/$K.mir" -o "$OUT/idem/$K.a.mir" 2> "$OUT/idem/$K.err"; then echo "VERIFY/LLC FAIL $K"; bad=$((bad+1)); continue; fi
        $LLC $ARGS -verify-machineinstrs -start-before=si-insert-waitcnts -stop-after=amdgpu-insert-delay-alu "$OUT/idem/$K.a.mir" -o "$OUT/idem/$K.b.mir" 2> "$OUT/idem/$K.err2" || { echo "SECOND RUN FAIL $K"; bad=$((bad+1)); continue; }
        if diff -q <(grep -v "ModuleID" "$OUT/idem/$K.a.mir") <(grep -v "ModuleID" "$OUT/idem/$K.b.mir") >/dev/null; then ok=$((ok+1)); else echo "NOT IDEMPOTENT $K ($(diff <(grep -v ModuleID "$OUT/idem/$K.a.mir") <(grep -v ModuleID "$OUT/idem/$K.b.mir") | grep -c '^[<>]') lines)"; bad=$((bad+1)); fi
    done
    echo "idempotent+verified kernels: $ok  failing: $bad"
fi