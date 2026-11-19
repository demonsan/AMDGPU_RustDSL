cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
for k in gemm_g1250_tdm_stage_f4_256x256_ks2 gemm_g1250_tdm_stage_f4_256x128_ks2 gemm_g1250_tdm_stage_f4_128x128_ks2; do AMDC_SCHED=1 $A emit $k --target gfx1250 --backend=own --out /tmp/bk_s.s 2>&1 | grep -A2 "compile error" | head -3 | cut -c1-200; echo "$k sched: $(grep -oE 'next_free_vgpr [0-9]+' /tmp/bk_s.s)"; done
echo "== MLA ws: A = unscheduled, B = list scheduler, gfx1250 model"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws B_ENV=AMDC_SCHED=1 bash $SRC/tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) (32x2048|128x4096|128x8192|512x8192)|STOP|MLA-PAIR-OK"
echo "== MLA ws: A = unscheduled, B = list scheduler, gfx950 model"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws B_ENV="AMDC_SCHED=1 AMDC_SCHED_MODEL950=1" bash $SRC/tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) (32x2048|128x4096|128x8192)|STOP|MLA-PAIR-OK"