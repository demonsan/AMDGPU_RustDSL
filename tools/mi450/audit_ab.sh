cd $SRC
for k in mla_g1250_qh16_ws mla_g1250_qh16_ws2 mla_g1250_qh16_ws4; do
echo "== $k: A gfx950 constants (AMDC_T950_COMPAT), B current target"
KA=$k KB=$k A_ENV="AMDC_T950_COMPAT=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -24
done
echo "== GEMM 256x256_ks2_persist: a compat, b current"
A_ENV="AMDC_T950_COMPAT=1" B_ENV="X=1" KERNEL=gemm_g1250_tdm_stage_f4_256x256_ks2_persist BM=256 BN=256 bash tools/mi450/accwin_ab.sh 2>&1 | grep -E "^[ab] |STOP|OK|bad" | head -16