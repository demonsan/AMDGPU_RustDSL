cd $SRC
for k in mla_g1250_qh16_wsp mla_g1250_qh16_wsi; do
  echo "== $k: A even spread (AMDC_IDENSITY=0), B density 1"
  KA=$k KB=$k A_ENV="AMDC_IDENSITY=0" B_ENV="AMDC_IDENSITY=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -24
done
echo "== mla_g1250_qh16_wsp: A density 1, B density 2"
KA=mla_g1250_qh16_wsp KB=mla_g1250_qh16_wsp A_ENV="AMDC_IDENSITY=1" B_ENV="AMDC_IDENSITY=2" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -24