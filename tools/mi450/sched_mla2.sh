cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
AMDC_SCHED=1 $A emit gemm_g1250_tdm_stage_f4_128x128_ks2 --target gfx1250 --backend=own --out /tmp/bk_s.s 2>&1 | tail -3 | cut -c1-300
for e in "AMDC_SCHED=1 AMDC_SCHED_OPTS=rewait" "AMDC_SCHED=ld"; do
  echo "== MLA ws: A = unscheduled, B = [$e]"
  KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws B_ENV="$e" bash $SRC/tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) (32x2048|128x4096|128x8192)|STOP|MLA-PAIR-OK"
done