cd $SRC
echo "== MLA ws: A = old TRANS evidence set, B = + v_sub / v_cvt_pk_fp8"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws A_ENV=AMDC_TRANS_PROVEN_OLD=1 bash $SRC/tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB|vendor) |STOP|MLA-PAIR-OK"