cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
$A md5-manifest --check tests/golden/md5_manifest.tsv 2>/dev/null | tail -1
$A md5-manifest --cert --check tests/golden/cert_md5_manifest.tsv 2>/dev/null | tail -1
for k in mla_g1250_qh16_ws mla_g1250_qh16_ws2 mla_g1250_qh16_ws4; do
echo "== $k: A mode 0, B mode 2 + ralru (scheduled path)"
KA=$k KB=$k B_ENV="AMDC_MODE2=1 AMDC_RA_LRU=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL|falling" | head -24
done