cd $SRC
for k in mla_g1250_qh16_ws mla_g1250_qh16_ws2; do
echo "== $k (mode2+ralru via the kernel's tokens): A unscheduled (= the old fallback, AMDC_SCHED=0), B scheduled (the fix)"
KA=$k KB=$k A_ENV="AMDC_SCHED=0" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -24
done