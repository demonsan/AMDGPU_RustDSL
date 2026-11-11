#!/usr/bin/env bash
# Warm ATT (rocprofv3 thread trace, one CU, SIMD 0) of a paged-decode kernel and of the aiter vendor kernel on the same data, one dispatch each (the 5th of the run), with the per-class table (tools/att_classes.py)
# and the MFMA-gap attribution (tools/att_gaps.py). The CU that hosts a wave of a 128-CTA grid is not predictable: the script tries CUS (default "1 2 0 3 4 5 6 7") until the trace has instructions.
# usage (repo root of a tree that has tools/att_*.py, GPU node):  HIP_VISIBLE_DEVICES=0 PAH=/tmp/pa2/pah KCO=pa_decode16_bf16_hd128_p16_f.co CCO=attn_combine_bf16_hd128.co VCO=vendor.co bash tools/att_pa.sh <tag> [B H HK CTX]
#   env: KREGEX (default pa_decode16), FUSED (1 = fused-combine kernel), KPREFIX / KSUFFIX (pa_host naming), PAGE (16), S (1), CUS, ATT_DECODER_DIR
set -u
TAG=${1:-pa}; B=${2:-16}; H=${3:-64}; HK=${4:-8}; CTX=${5:-2048}
R=/shared/apps/ubuntu/opt/rocm-7.0.2; DEC=${ATT_DECODER_DIR:-/shared/apps/ubuntu/opt/rocm-10.1.0/lib}; O=build/att/$TAG; rm -rf "$O"; mkdir -p "$O"
VN=$(strings "$VCO" | grep -m1 '^_ZN5aiter')
att() { who=$1; regex=$2; shift 2
  for cu in ${CUS:-1 2 0 3 4 5 6 7}; do
    rm -rf "$O/$who"; $R/bin/rocprofv3 --att --att-library-path "$DEC" --att-target-cu $cu --att-simd-select 0x1 --kernel-include-regex "$regex" --kernel-iteration-range "[5]" -d "$O/$who" -o run -- "$@" > "$O/$who.log" 2>&1
    st=$(find "$O/$who" -name "stats*.csv" | head -1)
    [ -n "$st" ] && [ "$(python3 tools/att_classes.py "$st" 2>&1 | head -1 | sed -E 's/.*instructions executed ([0-9]+).*/\1/')" -gt 0 ] && { echo "$who: CU $cu"; return 0; }
  done; return 1; }
att ours "${KREGEX:-pa_decode16}" env ${FUSED:+FUSED=1} KPREFIX=${KPREFIX:-pa_decode16_bf16_hd128_p} KSUFFIX=${KSUFFIX:-} "$PAH" ours "$KCO" "$CCO" $B $H $HK $CTX ${PAGE:-16} ${S:-1}
att vendor pa_bf16_noquant "$PAH" vendor "$VCO" "$VN" $B $H $HK $CTX
for who in ours vendor; do
  st=$(find "$O/$who" -name "stats*.csv" | head -1); ui=$(dirname "$(find "$O/$who" -name "code.json" | head -1)")
  echo "######## $who"; [ -n "$st" ] && python3 tools/att_classes.py "$st" | head -24; [ -n "$ui" ] && [ "$ui" != "." ] && python3 tools/att_gaps.py "$ui" 12 | head -30
done