cd $SRC
echo "== MLA ws: A = default allocation, B = window plan (loop-carried accumulators above v255)"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws B_ENV="AMDC_ACC_HIGH=1 AMDC_ACC_HIGH_ANY=1" bash $SRC/tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB|vendor) |STOP|MLA-PAIR-OK"