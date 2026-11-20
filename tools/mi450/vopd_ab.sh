cd $SRC
echo "== MLA ws: A default, B AMDC_VOPD=1"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws B_ENV="AMDC_VOPD=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB|vendor) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -30
echo "== GEMM 256x256 ks2: a default, b AMDC_VOPD=1"
A_ENV="X=1" B_ENV="AMDC_VOPD=1" KERNEL=gemm_g1250_tdm_stage_f4_256x256_ks2 BM=256 BN=256 bash tools/mi450/accwin_ab.sh 2>&1 | grep -E "^[ab] |STOP|OK|bad" | head -16