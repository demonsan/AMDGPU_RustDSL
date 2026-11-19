cd $SRC
echo "== MLA ws: A = [ld, gfx950 model], B = [ld, gfx1250 model]"
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws A_ENV="AMDC_SCHED=ld AMDC_SCHED_MODEL950=1" B_ENV="AMDC_SCHED=ld" bash $SRC/tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) (32x2048|128x4096|128x8192)|STOP|MLA-PAIR-OK"
A=$CARGO_TARGET_DIR/release/amdc; for e in "AMDC_SCHED=ld AMDC_SCHED_MODEL950=1" "AMDC_SCHED=ld"; do env $e $A emit mla_g1250_qh16_ws --target gfx1250 --backend=own --out /tmp/bk_m.s >/dev/null 2>&1; echo "[$e] md5 $(md5sum < /tmp/bk_m.s | cut -c1-8)"; done