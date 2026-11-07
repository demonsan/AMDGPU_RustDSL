#!/usr/bin/env bash
# IR text acceptance (docs/ir-text.md): for every registered kernel + the shipped set, `amdc print-ir --stage dsl` -> .ir -> `amdc emit --from-ir`
# must give byte-identical assembly on both backends, and `amdc gpu-verify --from-ir --strict` the identical certificate.
#   bash tools/ir_roundtrip_check.sh [kernel...]      (in the container: AMDC defaults to `bash tools/container/cargo.sh run -q --bin amdc --`)
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"; mkdir -p build/irrt
A=(${AMDC:-bash tools/container/cargo.sh run -q --bin amdc --})
KS=("$@")
[ ${#KS[@]} = 0 ] && KS=($("${A[@]}" list 2>/dev/null) fmha_fwd_hd128_mxfp8_v3_vs128 fmha_fwd_hd128_mxfp8_v3_t128 fmha_fwd_hd128_mxfp8_v3_kvsplit fmha_fwd_hd128_mxfp8_v3_kvsplit_combine8h gemm_mxfp8_bs_v4 mlp_fp8_ws_fused mlp_fp8_ws_g1 mlp_fp8_ws_g2)
ok=0; bad=0
for k in "${KS[@]}"; do
  "${A[@]}" print-ir "$k" --stage dsl > "build/irrt/$k.ir" 2>/dev/null || { echo "FAIL $k: print-ir"; bad=$((bad+1)); continue; }
  line="$k"
  fail=0
  for be in own llvm-tail; do
    a=$("${A[@]}" emit "$k" --backend=$be --out "/work/build/irrt/$k.$be.a.s" >/dev/null 2>&1 && md5sum < "build/irrt/$k.$be.a.s" | cut -c1-12 || echo none)
    b=$("${A[@]}" emit --from-ir "build/irrt/$k.ir" --backend=$be --no-verify --out "/work/build/irrt/$k.$be.b.s" >/dev/null 2>&1 && md5sum < "build/irrt/$k.$be.b.s" | cut -c1-12 || echo none)
    [ "$a" = "$b" ] || fail=1
    line="$line $be:$a/$b"
  done
  ca=$("${A[@]}" gpu-verify "$k" --strict 2>&1 | grep -E "certificate \(all" || echo none)
  cb=$("${A[@]}" gpu-verify --from-ir "build/irrt/$k.ir" --strict 2>&1 | grep -E "certificate \(all" || echo none)
  [ "$ca" = "$cb" ] || fail=1
  if [ $fail = 0 ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAIL $line | cert [$ca] vs [$cb]"; fi
done
echo "ir round trip: $ok kernels identical (asm own + llvm-tail, certificate), $bad failing"
[ $bad = 0 ]
