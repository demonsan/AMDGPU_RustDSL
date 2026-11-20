cd $SRC
for k in mla_g1250_qh16_ws mla_g1250_qh16_ws2 mla_g1250_qh16_ws4; do
  echo "== $k: A AMDC_NO_VOPD=1, B shipped (vopd)"
  KA=$k KB=$k A_ENV="AMDC_NO_VOPD=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL|hash" | head -24
done