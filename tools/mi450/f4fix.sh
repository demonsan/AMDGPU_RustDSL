cd "$SRC"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
O="$W/f4fix"; mkdir -p $O
for k in gemm_g1250_tdm_stage_f4_ks2_grouped gemm_g1250_tdm_stage_f4_ks2 gemm_g1250_tdm_stage_f4 gemm_g1250_tdm_stage_f4_128x128; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "$k EMIT-FAIL: $(tail -2 $O/$k.log | cut -c1-300)"; continue; }
  echo "$k: ds_loads=$(grep -cE '^\s+ds_load' $O/$k.s) dscnt_waits=$(grep -cE '^\s+s_wait_dscnt' $O/$k.s) wmma=$(grep -c v_wmma_scale $O/$k.s)  distinct dscnt: $(grep -E '^\s+s_wait_dscnt' $O/$k.s | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
done