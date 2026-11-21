cd $SRC
for k in mla_g1250_qh16_ws4 mla_g1250_qh16_ws mla_g1250_qh16_ws2; do
echo "== $k: A mode 0, B mode 2"
KA=$k KB=$k B_ENV="AMDC_MODE2=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL|cos_diff" | head -24
done
echo "== GEMM ks4_s2_epit_roles: a mode 0, b mode 2"
A_ENV="X=1" B_ENV="AMDC_MODE2=1" KERNEL=gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles BM=256 BN=256 bash tools/mi450/accwin_ab.sh 2>&1 | grep -E "^[ab] |STOP|OK|bad" | head -16