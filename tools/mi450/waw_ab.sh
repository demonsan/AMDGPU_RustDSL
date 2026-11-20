cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 96 --bin amdc 2>&1 | grep -E "^error" -A5 | head
names=$(grep -E '&\[".*probe_g1250_' crates/amdgpu-kernels/src/registry.rs | head -1 | grep -oE '"[a-z0-9_]+"' | tr -d '"')
rm -rf /tmp/bk_new /tmp/bk_old; mkdir -p /tmp/bk_new /tmp/bk_old
n=0; d=0
for k in $names; do n=$((n+1)); $A emit $k --target gfx1250 --backend=own --out /tmp/bk_new/$k.s >/dev/null 2>&1; AMDC_WAW_OLD_LIMITS=1 AMDC_WAIT_CHECK_WARN=1 $A emit $k --target gfx1250 --backend=own --out /tmp/bk_old/$k.s >/dev/null 2>&1
 if ! cmp -s /tmp/bk_new/$k.s /tmp/bk_old/$k.s; then d=$((d+1)); echo "DIFF $k: waits old $(grep -cE 's_wait_(dscnt|loadcnt|storecnt)' /tmp/bk_old/$k.s) new $(grep -cE 's_wait_(dscnt|loadcnt|storecnt)' /tmp/bk_new/$k.s)"; fi; done; echo "$n kernels, $d differ"