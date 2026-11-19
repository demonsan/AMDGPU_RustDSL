cd $SRC
echo "== GEMM 256x256 ks2: A = no dse / tdm_first, B = with"
A_ENV="X=1" B_ENV="AMDC_DSE=1 AMDC_TDM_FIRST=1" KERNEL=gemm_g1250_tdm_stage_f4_256x256_ks2 BM=256 BN=256 bash tools/mi450/accwin_ab.sh 2>&1 | grep -E "^[ab] |STOP|OK|bad|GPU:" | head -20
echo "== MLA ws: A = no dse / tdm_first, B = with"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws B_ENV="AMDC_DSE=1 AMDC_TDM_FIRST=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) (4x1000|32x2048|128x8192)|STOP|MLA-PAIR-OK"
