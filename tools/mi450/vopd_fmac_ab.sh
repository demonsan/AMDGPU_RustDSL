cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 96 --bin amdc 2>&1 | grep -E "^error" -A8 | head -20
for k in mla_g1250_qh16_ws mla_g1250_qh16_ws4; do $A emit $k --target gfx1250 --backend=own --out /tmp/bk_f.s >/dev/null 2>&1; echo "$k: dual $(grep -c v_dual_ /tmp/bk_f.s) fmac-dual $(grep -c 'v_dual_fmac' /tmp/bk_f.s) v_fma $(grep -cE '^\s+v_fma_f32' /tmp/bk_f.s)"; done
KA=mla_g1250_qh16_ws KB=mla_g1250_qh16_ws A_ENV="AMDC_NO_VOPD=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -24