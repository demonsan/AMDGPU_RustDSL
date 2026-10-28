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

make_mirror() {  # $1 = dir ; a copy of llvm/test plus symlinks to the rest of llvm/, with a lit site config pointing at it
    local M="$1"
    rm -rf "$M"; mkdir -p "$M/llvm" "$M/build/test"
    cp -r "$LLVM_SRC/llvm/test" "$M/llvm/test"
    for e in "$LLVM_SRC"/llvm/*; do b=$(basename "$e"); [ "$b" = test ] || ln -s "$e" "$M/llvm/$b"; done
    for e in "$LLVM_SRC"/*; do b=$(basename "$e"); [ "$b" = llvm ] || ln -s "$e" "$M/$b"; done
    sed -e "s#^config.llvm_src_root = .*#config.llvm_src_root = '$M/llvm'#" \
        -e "s#^config.llvm_obj_root = .*#config.llvm_obj_root = '$M/build'#" \
        -e "s#^config.llvm_tools_dir = .*#config.llvm_tools_dir = '$LLVM_TOOLS'#" \
        -e "s#^config.llvm_shlib_dir = .*#config.llvm_shlib_dir = '$LLVM_ROCM/lib/llvm/lib'#" \
        "$LLVM_BUILD/test/lit.site.cfg.py" > "$M/build/test/lit.site.cfg.py"
    mkdir -p "$M/build/test/CodeGen/AMDGPU"
}

run_lit() {  # $1 = mirror ; $2 = result json ; stdin = basenames
    local M="$1" J="$2" re
    re="CodeGen/AMDGPU/($(paste -sd'|' | sed 's/\./\\./g'))\$"
    (cd "$M/build" && $LIT -j"$JOBS" -q --filter="$re" -o "$J" "$M/build/test/CodeGen/AMDGPU" > "${J%.json}.log" 2>&1)
}

summ() { python3 - "$@" <<'EOF'
import json,sys,collections
d=json.load(open(sys.argv[1])); c=collections.Counter(t["code"] for t in d["tests"]); print(dict(c))
EOF
}

if [ "$WHAT" = lit ] || [ "$WHAT" = all ]; then
    echo "== lit: baseline vs round trip"
    selected_tests > "$OUT/selected.txt"; echo "selected tail-pass MIR tests: $(wc -l < "$OUT/selected.txt")"
    make_mirror "$WORK/base"
    run_lit "$WORK/base" "$OUT/lit-baseline.json" < "$OUT/selected.txt"
    echo -n "baseline: "; summ "$OUT/lit-baseline.json"
    make_mirror "$WORK/rt"
    : > "$OUT/roundtrip-stats.txt"
    while read -r f; do
        p="$WORK/rt/llvm/$TEST_SUB/$f"
        echo -n "$f " >> "$OUT/roundtrip-stats.txt"
        "$AMDC" mir-roundtrip "$LLVM_SRC/llvm/$TEST_SUB/$f" "$p.new" >> "$OUT/roundtrip-stats.txt" 2>&1 && mv "$p.new" "$p"
    done < "$OUT/selected.txt"
    run_lit "$WORK/rt" "$OUT/lit-roundtrip.json" < "$OUT/selected.txt"
    echo -n "round trip: "; summ "$OUT/lit-roundtrip.json"
    python3 - "$OUT" <<'EOF'
import json,sys,re,collections
o=sys.argv[1]
b={t["name"]:t["code"] for t in json.load(open(o+"/lit-baseline.json"))["tests"]}
r={t["name"]:t["code"] for t in json.load(open(o+"/lit-roundtrip.json"))["tests"]}
st={}
for l in open(o+"/roundtrip-stats.txt"):
    m=re.match(r"(\S+) STATS files=1 bodies=(\d+) insts=(\d+) parsed=(\d+) opaque=(\d+) bundles=(\d+) changed=(\d+)",l)
    if m: st["LLVM :: CodeGen/AMDGPU/"+m.group(1)]=tuple(map(int,m.groups()[1:]))
imp=[n for n,s in st.items() if s[0]>0 and s[3]==0]
print("tests: %d  baseline PASS: %d  roundtrip PASS: %d" % (len(b), sum(v=="PASS" for v in b.values()), sum(v=="PASS" for v in r.values())))
print("fully importable (bodies>0, no opaque instruction): %d ; of those PASS after round trip: %d" % (len(imp), sum(r.get(n)=="PASS" for n in imp)))
print("instructions: %d parsed: %d opaque: %d bundles: %d lines changed by printing: %d" % tuple(sum(s[i] for s in st.values()) for i in (1,2,3,4)) if False else "")
tot=[sum(s[i] for s in st.values()) for i in range(6)]
print("instructions=%d parsed=%d opaque=%d bundles=%d changed_lines=%d" % (tot[1],tot[2],tot[3],tot[4],tot[5]))
reg=[n for n in b if b[n]=="PASS" and r.get(n)!="PASS"]
print("REGRESSIONS (baseline PASS, round trip not PASS): %d" % len(reg))
for n in sorted(reg): print("  ",n,r.get(n),st.get(n))
EOF
fi

if [ "$WHAT" = table ] || [ "$WHAT" = all ]; then
    echo "== opcode table vs LLVM's own MIR"
    selected_tests > "$OUT/selected.txt"
    (cd "$LLVM_SRC/llvm/$TEST_SUB" && "$AMDC" mir-table-report $(cat "$OUT/selected.txt")) > "$OUT/table-report.txt"
    head -3 "$OUT/table-report.txt"
fi

if [ "$WHAT" = idem ] || [ "$WHAT" = all ]; then
    echo "== tail idempotence + verifier on exported kernels"
    mkdir -p "$OUT/idem"; ok=0; bad=0
    LLC="$LLVM_TOOLS/llc"; ARGS="-mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O2"
    for K in $("$AMDC" list); do
        "$AMDC" emit "$K" --mir-out "$OUT/idem/$K.mir" 2>/dev/null || { echo "EXPORT FAIL $K"; bad=$((bad+1)); continue; }
        if ! $LLC $ARGS -verify-machineinstrs -start-before=gcn-create-vopd -stop-after=amdgpu-insert-delay-alu "$OUT/idem/$K.mir" -o "$OUT/idem/$K.a.mir" 2> "$OUT/idem/$K.err"; then echo "VERIFY/LLC FAIL $K"; bad=$((bad+1)); continue; fi
        $LLC $ARGS -verify-machineinstrs -start-before=si-insert-waitcnts -stop-after=amdgpu-insert-delay-alu "$OUT/idem/$K.a.mir" -o "$OUT/idem/$K.b.mir" 2> "$OUT/idem/$K.err2" || { echo "SECOND RUN FAIL $K"; bad=$((bad+1)); continue; }
        if diff -q "$OUT/idem/$K.a.mir" "$OUT/idem/$K.b.mir" >/dev/null; then ok=$((ok+1)); else echo "NOT IDEMPOTENT $K ($(diff "$OUT/idem/$K.a.mir" "$OUT/idem/$K.b.mir" | grep -c '^[<>]') lines)"; bad=$((bad+1)); fi
    done
    echo "idempotent+verified kernels: $ok  failing: $bad"
fi