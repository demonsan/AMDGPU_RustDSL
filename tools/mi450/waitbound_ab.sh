cd $SRC
echo "== MLA ws: A = old bound, B = safe bound"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws A_ENV=AMDC_WAIT_BOUND_OLD=1 bash $SRC/tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) (4x1000|32x2048|128x4096|128x8192)|STOP|MLA-PAIR-OK"