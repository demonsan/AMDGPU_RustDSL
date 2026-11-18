# VGPR count of every gfx1250 kernel: the default allocator vs AMDC_RA_FIRSTFIT (the previous two-phase rule); CPU only
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" -A7 | head -20
names=$(grep -oE '&\["probe_g1250_add"[^]]*\]' crates/amdgpu-kernels/src/registry.rs | grep -oE '"[a-z0-9_]+"' | tr -d '"')
extra=${EXTRA:-}
for k in $names $extra; do
  a=$(AMDC_RA_FIRSTFIT=1 $A emit $k --target gfx1250 --backend=own --out /tmp/bk_ra_a.s 2>&1 | grep -E "error|out of" | head -1 | cut -c1-60); va=$(grep -oE 'next_free_vgpr [0-9]+' /tmp/bk_ra_a.s 2>/dev/null | awk '{print $2}')
  b=$($A emit $k --target gfx1250 --backend=own --out /tmp/bk_ra_b.s 2>&1 | grep -E "error|out of" | head -1 | cut -c1-60); vb=$(grep -oE 'next_free_vgpr [0-9]+' /tmp/bk_ra_b.s 2>/dev/null | awk '{print $2}')
  [ -n "$a" ] && va="ERR"; [ -n "$b" ] && vb="ERR"
  [ "$va" != "$vb" ] && echo "DIFF $k: firstfit=$va bestfit=$vb" || echo "same $k: $va"
  rm -f /tmp/bk_ra_a.s /tmp/bk_ra_b.s
done