cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
for k in mla_g1250_qh16_ws mla_g1250_qh16_ws2 mla_g1250_qh16_ws4; do for e in "X=1" "AMDC_MODE2=1 AMDC_RA_LRU=1"; do env $e $A emit $k --target gfx1250 --backend=own --out /tmp/bk_l.s >/dev/null 2>&1; echo "$k [$e]: $(grep -E 'amdhsa_next_free_vgpr' /tmp/bk_l.s | head -1)"; done; done
for k in mla_g1250_qh16_ws mla_g1250_qh16_ws2 mla_g1250_qh16_ws4; do
echo "== $k: A mode 2, B mode 2 + LRU"
KA=$k KB=$k A_ENV="AMDC_MODE2=1" B_ENV="AMDC_MODE2=1 AMDC_RA_LRU=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -24
done