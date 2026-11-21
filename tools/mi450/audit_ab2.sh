cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 96 --bin amdc 2>&1 | grep -E "^error" -A6 | head -10
for k in mla_g1250_qh16_ws; do
 rm -f /tmp/x1.s /tmp/x2.s; AMDC_T950_COMPAT=1 $A emit $k --target gfx1250 --backend=own --out /tmp/x1.s >/dev/null 2>&1; AMDC_T950_LAT=1 $A emit $k --target gfx1250 --backend=own --out /tmp/x2.s >/dev/null 2>&1; cmp -s /tmp/x1.s /tmp/x2.s && echo "$k: compat == current+gfx950 latencies (the hazard-rule change alone does not alter it)" || echo "$k: differs even with gfx950 latencies"
echo "== $k: A gfx950 constants, B current target with gfx950 latencies"
KA=$k KB=$k A_ENV="AMDC_T950_COMPAT=1" B_ENV="AMDC_T950_LAT=1" bash tools/mi450/mla_pair.sh 2>&1 | grep -E "^(KA|KB) |STOP|MLA-PAIR-OK|MLA_FAIL" | head -18
done