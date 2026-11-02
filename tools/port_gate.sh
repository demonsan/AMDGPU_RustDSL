#!/usr/bin/env bash
# FMHA port acceptance gate: one run on the GPU node checks a candidate kernel set against the full acceptance protocol and prints a PASS/FAIL table.
#
# Usage:  HIP_VISIBLE_DEVICES=6 bash tools/port_gate.sh <kernel_prefix> [tag]
#   The candidate set is ${prefix}_vs128, ${prefix}_t128, ${prefix}_kvsplit, ${prefix}_kvsplit_combine8h; any of them can be overridden
#   (e.g. while only _vs128 is ported): KN_VS128=... KN_T128=... KN_KVSPLIT=... KN_COMBINE=... (the combine name must contain "combine8":
#   ref_host derives the elements/thread from the name).
#   Baseline (shipped):  bash tools/port_gate.sh fmha_fwd_hd128_mxfp8
#
# Sections (each prints PASS/FAIL; SKIP_<SECTION>=1 skips one):
#   BUILD    emit (+ --backend ${BACKEND:-auto}) and assemble the 4 kernels, build ref_host
#   VERIFY   amdc gpu-verify <k> --strict for each kernel: 0 errors
#   ACC16    15-shape gate, auto dispatch, REF_ROWS=16: REF_KERNEL_OK + vendor parity (mean <= 1.10 x vendor)          [= tests/gpu/run_ref.sh]
#   ACC64    15 shapes x 4 forced paths (auto / _vs128 / _t128 / forced split), REF_ROWS=64: bad=0 (+ parity, SPLIT_ULP informational)
#   DET      5 shapes x 3 launches, auto dispatch: bit-identical md5                                                       [= tests/gpu/determinism.sh]
#   STRESS   repeated launches of split shapes (forced s=4 B1H2S200 SK1337 gqa2, auto B1H8S2048), every launch REF_KERNEL_OK and
#            all-output worst diff vs vendor <= STRESS_MAXDIFF; the md5 histogram is printed but bit identity is NOT required
#   PERF     warm protocol (REF_WARMUP=150 REF_ITERS=100), PERF_REPS (5) interleaved repetitions, median; candidate and (unless the
#            candidate IS the shipped set) the shipped set in the same run, same-run vendor:
#              B1H8S2048 auto (KV-split), per-CU 64 blocks (_vs128 forced, B1H8S2048), B4H8S2048 and B1H8S8192 (filled, _vs128), B2H8S2048 (_t128, info)
#            PASS when candidate median >= PERF_TOL (0.97) x shipped median (same run) - or, for the baseline run, >= 0.97 x the recorded numbers.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
P="${1:?usage: tools/port_gate.sh <kernel_prefix> [tag]}"
TAG="${2:-$(echo "$P" | tr -c 'A-Za-z0-9_\n' '_')}"
SHIP=fmha_fwd_hd128_mxfp8
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"
CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
REF_CO="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
GPU="${HIP_VISIBLE_DEVICES:-6}"; export HIP_VISIBLE_DEVICES="$GPU"
PERF_REPS="${PERF_REPS:-5}"; PERF_TOL="${PERF_TOL:-0.97}"; STRESS_RUNS="${STRESS_RUNS:-300}"; STRESS_MAXDIFF="${STRESS_MAXDIFF:-0.15}"   # shipped baseline: 0.066 (B1H2S200 split4) / 0.098 (B1H8S2048); recalibrate if a corruption could hide below it
KN_VS128="${KN_VS128:-${P}_vs128}"; KN_T128="${KN_T128:-${P}_t128}"; KN_KVSPLIT="${KN_KVSPLIT:-${P}_kvsplit}"; KN_COMBINE="${KN_COMBINE:-${P}_kvsplit_combine8h}"
B="build/pg_${TAG}"; mkdir -p "$B"
LOG="$B/gate.log"; : > "$LOG"
SUMMARY=()   # "section|item|value|reference|verdict"
FAILS=0
row() { SUMMARY+=("$1|$2|$3|$4|$5"); [ "$5" = FAIL ] && FAILS=$((FAILS + 1)); echo "[$1] $2: $3 (ref $4) -> $5" | tee -a "$LOG"; }
skip() { [ "${!1:-0}" = 1 ]; }
echo "port_gate: prefix=$P tag=$TAG GPU=$GPU kernels: $KN_VS128 $KN_T128 $KN_KVSPLIT $KN_COMBINE  git=$(git rev-parse --short HEAD 2>/dev/null)" | tee -a "$LOG"

# ---------------------------------------------------------------- BUILD
emit() { # kernel out.co
    bash tools/container/cargo.sh run -q --bin amdc -- emit "$1" --backend "${BACKEND:-auto}" ${AMDC_EXTRA_FLAGS:-} --out "/work/$B/$1.s" >>"$LOG" 2>&1 \
        && "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o "$2" "$B/$1.s" >>"$LOG" 2>&1
}
build_set() { # prefix-dir vs t128 kvsplit combine
    local d="$1"; mkdir -p "$d"; local ok=1
    emit "$2" "$d/vs.co" || ok=0; emit "$3" "$d/t128.co" || ok=0; emit "$4" "$d/split.co" || ok=0; emit "$5" "$d/comb.co" || ok=0
    [ $ok = 1 ]
}
bash tools/container/cargo.sh build -q --bin amdc >>"$LOG" 2>&1 || { row BUILD cargo "build failed" "-" FAIL; }
if build_set "$B/cand" "$KN_VS128" "$KN_T128" "$KN_KVSPLIT" "$KN_COMBINE"; then row BUILD candidate "4 kernels emitted+assembled" "-" PASS
else row BUILD candidate "emit/assemble failed (see $LOG)" "-" FAIL; echo "abort: candidate build failed"; tail -30 "$LOG"; exit 1; fi
for k in "$KN_VS128" "$KN_T128" "$KN_KVSPLIT"; do grep -cE '^\s*v_' "$B/$k.s" | xargs -I{} echo "  $k: {} VALU/VOP lines, $(grep -cE '^\s*v_mfma' "$B/$k.s") mfma" | tee -a "$LOG"; done
SAME_AS_SHIP=0; [ "$KN_VS128 $KN_T128 $KN_KVSPLIT $KN_COMBINE" = "${SHIP}_vs128 ${SHIP}_t128 ${SHIP}_kvsplit ${SHIP}_kvsplit_combine8h" ] && SAME_AS_SHIP=1
if [ $SAME_AS_SHIP = 0 ] && ! skip SKIP_PERF; then
    build_set "$B/ship" "${SHIP}_vs128" "${SHIP}_t128" "${SHIP}_kvsplit" "${SHIP}_kvsplit_combine8h" || { echo "abort: shipped build failed"; exit 1; }
fi
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o "$B/ref_host" -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" >>"$LOG" 2>&1 \
    || { row BUILD ref_host "g++ failed" "-" FAIL; exit 1; }
RH="$B/ref_host"

# ---------------------------------------------------------------- VERIFY
if ! skip SKIP_VERIFY; then
    for k in "$KN_VS128" "$KN_T128" "$KN_KVSPLIT" "$KN_COMBINE"; do
        out=$(bash tools/container/cargo.sh run -q --bin amdc -- gpu-verify "$k" --strict 2>&1); echo "$out" >>"$LOG"
        errs=$(echo "$out" | sed -nE 's/^== [^:]+: [0-9]+ finding\(s\), ([0-9]+) error\(s\).*/\1/p' | head -1)
        cert=$(echo "$out" | grep -m1 'certificate:' | sed -E 's/^ *certificate: //')
        [ -n "$errs" ] && [ "$errs" = 0 ] && v=PASS || v=FAIL
        row VERIFY "$k" "${errs:-?} errors; $cert" "0 errors" $v
    done
fi

# ---------------------------------------------------------------- helpers
SHAPES=("1 1 256 256 1" "1 2 512 512 1" "2 4 1024 1024 1" "1 8 2048 2048 1" "1 2 512 2048 1" "3 3 768 1280 1" "1 4 300 512 1" "2 8 700 640 2"
        "1 16 256 1024 4" "1 2 256 1000 1" "2 4 300 129 1" "1 8 512 100 2" "1 2 1024 1088 1" "1 4 77 1 1" "1 2 200 1337 2")
# run_rh <set-dir> <mode> B H S SK gqa  [extra env as VAR=val ...]; modes: auto | vs | t128 | split<s>
run_rh() {
    local d="$1" mode="$2" b="$3" h="$4" s="$5" sk="$6" g="$7"; shift 7
    local -a e=(REF_B=$b REF_H=$h REF_S=$s REF_SK=$sk REF_HK=$((h / g)) REF_COMBINE_ELEMS=4)
    local kv kt ks kc; if [ "$d" = "$B/ship" ]; then kv=${SHIP}_vs128; kt=${SHIP}_t128; ks=${SHIP}_kvsplit; kc=${SHIP}_kvsplit_combine8h
    else kv=$KN_VS128; kt=$KN_T128; ks=$KN_KVSPLIT; kc=$KN_COMBINE; fi
    case "$mode" in
        auto)  e+=(REF_CO2=$d/vs.co REF_KNAME2=$kv REF_CO2_SMALL=$d/t128.co REF_KNAME2_SMALL=$kt REF_CO2_SPLIT=$d/split.co REF_KNAME2_SPLIT=$ks
                   REF_CO2_COMBINE=$d/comb.co REF_KNAME2_COMBINE=$kc REF_SPLIT=0) ;;
        vs)    e+=(REF_CO2=$d/vs.co REF_KNAME2=$kv REF_ROWS_PER_BLOCK=256) ;;
        t128)  e+=(REF_CO2=$d/t128.co REF_KNAME2=$kt REF_ROWS_PER_BLOCK=128) ;;
        split*) e+=(REF_CO2=$d/vs.co REF_KNAME2=$kv REF_CO2_SPLIT=$d/split.co REF_KNAME2_SPLIT=$ks REF_CO2_COMBINE=$d/comb.co REF_KNAME2_COMBINE=$kc
                   REF_SPLIT=${mode#split} REF_SPLIT_FORCE=1) ;;
    esac
    env "${e[@]}" "$@" "$RH" "$REF_CO" 2>&1
}
field() { sed -nE "s/.*$1=([^ ]+).*/\1/p" | head -1; }
# a valid forced split count for SK (every split owns a tile): prefer 4, else 2, else none
split_for() { local n=$(( ($1 + 127) / 128 )); for s in 4 2; do local t=$(( (n + s - 1) / s )); if [ $n -ge 2 ] && [ $(( (s - 1) * t )) -lt $n ]; then echo $s; return; fi; done; echo 0; }
parity() { awk -v m="${PARITY_MARGIN:-1.10}" '
    /^vendor kernel +vs CPU/ { for (i = 1; i <= NF; i++) if ($i ~ /^mean_err=/) { sub("mean_err=", "", $i); vm = $i + 0 } hv = 1 }
    /^our kernel +vs CPU/    { for (i = 1; i <= NF; i++) if ($i ~ /^mean_err=/) { sub("mean_err=", "", $i); om = $i + 0 } ho = 1 }
    END { if (!(hv && ho)) { print "NA"; exit } printf "%s %.3e %.3e\n", (om <= m * vm) ? "OK" : "FAIL", om, vm }'; }

# ---------------------------------------------------------------- ACC16 (the 15-shape gate of run_ref.sh)
if ! skip SKIP_ACC; then
    nok=0; npar=0; worst_ratio=0
    for CFG in "${SHAPES[@]}"; do set -- $CFG
        out=$(run_rh "$B/cand" auto $1 $2 $3 $4 $5 REF_ROWS=16 REF_ITERS=0); echo "== ACC16 $CFG"$'\n'"$out" >>"$LOG"
        echo "$out" | grep -q REF_KERNEL_OK && nok=$((nok + 1)) || echo "  ACC16 FAIL kernel B=$1 H=$2 S=$3 SK=$4 gqa=$5" | tee -a "$LOG"
        p=$(echo "$out" | parity); [ "${p%% *}" = OK ] && npar=$((npar + 1)) || echo "  ACC16 parity $p B=$1 H=$2 S=$3 SK=$4 gqa=$5" | tee -a "$LOG"
    done
    [ $nok = 15 ] && v=PASS || v=FAIL; row ACC16 "REF_KERNEL_OK (auto, 16 rows)" "$nok/15" "15/15" $v
    [ $npar = 15 ] && v=PASS || v=FAIL; row ACC16 "vendor parity <= 1.10 (16 rows)" "$npar/15" "15/15" $v
# ---------------------------------------------------------------- ACC64 (64 rows, every path forced)
    for mode in auto vs t128 split; do
        nbad=0; nshape=0; npar=0; sumr=0; ulp_ok=0; ulp_n=0
        for CFG in "${SHAPES[@]}"; do set -- $CFG
            m=$mode; if [ $mode = split ]; then s=$(split_for $4); [ $s = 0 ] && continue; m=split$s; fi
            out=$(run_rh "$B/cand" $m $1 $2 $3 $4 $5 REF_ROWS=64 REF_ITERS=0); echo "== ACC64 $m $CFG"$'\n'"$out" >>"$LOG"
            nshape=$((nshape + 1))
            bad=$(echo "$out" | grep '^our kernel' | field bad); [ "${bad:-1}" = 0 ] || { nbad=$((nbad + 1)); echo "  ACC64 $m bad=${bad:-?} B=$1 H=$2 S=$3 SK=$4 gqa=$5" | tee -a "$LOG"; }
            p=$(echo "$out" | parity); [ "${p%% *}" = OK ] && npar=$((npar + 1)) || echo "  ACC64 $m parity $p B=$1 H=$2 S=$3 SK=$4 gqa=$5" | tee -a "$LOG"
            if echo "$out" | grep -q SPLIT_ULP_; then ulp_n=$((ulp_n + 1)); echo "$out" | grep -q SPLIT_ULP_OK && ulp_ok=$((ulp_ok + 1)); fi
        done
        [ $nbad = 0 ] && v=PASS || v=FAIL; row ACC64 "$mode: shapes with bad>0 (64 rows)" "$nbad/$nshape" "0" $v
        [ $npar = $nshape ] && v=PASS || v=WARN; row ACC64 "$mode: vendor parity (64 rows)" "$npar/$nshape" "$nshape/$nshape" $v
        [ $ulp_n -gt 0 ] && row ACC64 "$mode: SPLIT_ULP vs _vs128 (info)" "$ulp_ok/$ulp_n" "-" INFO
    done
fi

# ---------------------------------------------------------------- DET (determinism.sh: bit-identical over 3 launches)
if ! skip SKIP_DET; then
    nd=0
    for CFG in "2 8 700 640 2" "1 2 200 1337 2" "1 4 300 512 1" "2 4 300 129 1" "1 8 2048 2048 1"; do set -- $CFG
        sums=""; for r in 1 2 3; do run_rh "$B/cand" auto $1 $2 $3 $4 $5 REF_ROWS=4 REF_ITERS=5 REF_DUMP2="$B/det_$r.bin" >/dev/null; sums="$sums $(md5sum "$B/det_$r.bin" | cut -c1-12)"; done
        n=$(echo $sums | tr ' ' '\n' | sort -u | wc -l); [ "$n" = 1 ] && nd=$((nd + 1)) || echo "  DET_FAIL B=$1 H=$2 S=$3 SK=$4 gqa=$5:$sums" | tee -a "$LOG"
    done
    [ $nd = 5 ] && v=PASS || v=FAIL; row DET "bit-identical 3x (auto)" "$nd/5" "5/5" $v
fi

# ---------------------------------------------------------------- STRESS (split shapes, repeated launches)
if ! skip SKIP_STRESS; then
    for CFG in "1 2 200 1337 2 split4 $STRESS_RUNS" "1 8 2048 2048 1 auto $(( STRESS_RUNS / 3 ))"; do set -- $CFG
        rm -f "$B/st.sums"; nfail=0; wmax=0
        for i in $(seq 1 $7); do
            out=$(run_rh "$B/cand" $6 $1 $2 $3 $4 $5 REF_ROWS=8 REF_ITERS=0 REF_SPLIT_ULP=0 REF_DUMP2="$B/st.bin")
            echo "$out" | grep -q REF_KERNEL_OK || nfail=$((nfail + 1))
            wd=$(echo "$out" | grep '^our kernel vs vendor' | field worst_diff); wmax=$(awk -v a="$wmax" -v b="${wd:-99}" 'BEGIN{print (b > a) ? b : a}')
            md5sum "$B/st.bin" | cut -c1-12 >> "$B/st.sums"
        done
        hist=$(sort "$B/st.sums" | uniq -c | sort -rn | awk '{printf "%s x%s ", $2, $1}')
        echo "$out" | grep -q '\[split\].*CHOSEN\|KV-split' || echo "  STRESS WARNING: B=$1 H=$2 S=$3 SK=$4 did not take the split path" | tee -a "$LOG"
        ok=$(awk -v w="$wmax" -v t="$STRESS_MAXDIFF" -v f=$nfail 'BEGIN{print (f == 0 && w <= t) ? "PASS" : "FAIL"}')
        row STRESS "B$1H$2S$3SK$4 gqa$5 $6 x$7" "fail=$nfail max_worst_diff_vs_vendor=$wmax md5: $hist" "fail=0, diff<=$STRESS_MAXDIFF" $ok
    done
fi

# ---------------------------------------------------------------- PERF (same run, interleaved, median)
# recorded shipped numbers (docs/current-status.md, Main 8fec359; used only when the candidate IS the shipped set)
declare -A REC=([b1h8s2048]=674 [percu64]=446 [b4h8s2048]=1443 [b1h8s8192]=1999 [b2h8s2048]=1057)
median() { sort -g | awk '{a[NR]=$1} END {if (NR == 0) print "nan"; else if (NR % 2) print a[(NR + 1) / 2]; else print (a[NR / 2] + a[NR / 2 + 1]) / 2}'; }
if ! skip SKIP_PERF; then
    PERF=("b1h8s2048|1 8 2048 2048 1|auto" "percu64|1 8 2048 2048 1|vs" "b4h8s2048|4 8 2048 2048 1|auto" "b1h8s8192|1 8 8192 8192 1|auto" "b2h8s2048|2 8 2048 2048 1|auto")
    SETS=("$B/cand"); [ $SAME_AS_SHIP = 0 ] && SETS+=("$B/ship")
    for spec in "${PERF[@]}"; do
        IFS='|' read -r key cfg mode <<<"$spec"; set -- $cfg
        : > "$B/pf_cand"; : > "$B/pf_ship"; : > "$B/pf_vend"; path=""
        for rep in $(seq 1 $PERF_REPS); do
            for d in "${SETS[@]}"; do
                out=$(run_rh "$d" $mode $1 $2 $3 $4 $5 REF_ROWS=1 REF_WARMUP=150 REF_ITERS=100 REF_SPLIT_ULP=0); echo "== PERF $key $d rep$rep"$'\n'"$out" >>"$LOG"
                tf=$(echo "$out" | awk '/^timing ours total:/ {for (i = 1; i <= NF; i++) if ($i == "TFLOP/s") print $(i-1)} /^timing ours +:/ {for (i = 1; i <= NF; i++) if ($i ~ /^TFLOP\/s/) print $(i-1)}' | tail -1)
                vt=$(echo "$out" | awk '/^timing vendor/ {for (i = 1; i <= NF; i++) if ($i ~ /^TFLOP\/s/) print $(i-1)}' | head -1)
                [ "$d" = "$B/cand" ] && { echo "${tf:-0}" >> "$B/pf_cand"; path=$(echo "$out" | sed -nE 's/^\[grid\].*ours \(([^)]*)\).*\[([^;]*);.*/\1 [\2]/p' | head -1); } || echo "${tf:-0}" >> "$B/pf_ship"
                echo "${vt:-0}" >> "$B/pf_vend"
            done
        done
        mc=$(median < "$B/pf_cand"); mv=$(median < "$B/pf_vend")
        if [ $SAME_AS_SHIP = 0 ]; then ms=$(median < "$B/pf_ship"); refs="shipped $ms (same run)"; base=$ms
        else base=${REC[$key]}; refs="recorded $base"; fi
        v=$(awk -v c="$mc" -v r="$base" -v t="$PERF_TOL" 'BEGIN{print (c >= t * r) ? "PASS" : "FAIL"}'); [ "$key" = b2h8s2048 ] && v="INFO($v)"
        row PERF "$key $mode -> $path" "$(printf '%s TF/s (%.3fx vendor %s; runs: %s)' "$mc" "$(awk -v a=$mc -v b=$mv 'BEGIN{print (b > 0) ? a / b : 0}')" "$mv" "$(tr '\n' ' ' < $B/pf_cand)")" "$refs, tol $PERF_TOL" "$v"
    done
fi

# ---------------------------------------------------------------- table
echo; echo "================ PORT GATE: $P ($TAG) git=$(git rev-parse --short HEAD 2>/dev/null) GPU=$GPU ================"
printf '%-7s | %-48s | %-70s | %-34s | %s\n' SECTION ITEM VALUE REFERENCE VERDICT
for r in "${SUMMARY[@]}"; do IFS='|' read -r a b c d e <<<"$r"; printf '%-7s | %-48s | %-70s | %-34s | %s\n' "$a" "$b" "$c" "$d" "$e"; done
echo "OVERALL: $([ $FAILS = 0 ] && echo PASS || echo "FAIL ($FAILS)")   log: $LOG"
exit $([ $FAILS = 0 ] && echo 0 || echo 1)
