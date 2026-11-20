cd $SRC
for kk in "gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles 256 256" "gemm_g1250_tdm_stage_f4_256x128_ks2_persist 256 128"; do set -- $kk
  echo "== $1: a default, b AMDC_VOPD=1"
  A_ENV="X=1" B_ENV="AMDC_VOPD=1" KERNEL=$1 BM=$2 BN=$3 bash tools/mi450/accwin_ab.sh 2>&1 | grep -E "^[ab] |STOP|OK|bad" | head -16
done