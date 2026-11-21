cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 96 --bin amdc 2>&1 | grep -E "^error" -A6 | head -10
names=$(grep -E '&\[".*probe_g1250_' crates/amdgpu-kernels/src/registry.rs | head -1 | grep -oE '"[a-z0-9_]+"' | tr -d '"')
d=0; for k in $names; do rm -f /tmp/x1.s /tmp/x2.s; AMDC_T950_COMPAT=1 $A emit $k --target gfx1250 --backend=own --out /tmp/x1.s >/dev/null 2>&1; $A emit $k --target gfx1250 --backend=own --out /tmp/x2.s >/dev/null 2>&1; cmp -s /tmp/x1.s /tmp/x2.s || { d=$((d+1)); echo "differs: $k"; }; done; echo "$d differ"