cd "$SRC"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
for kn in gemm_g1250_tdm_fp8 gemm_g1250_tdm_fp8_k192; do
  echo "== $kn"
  $A emit $kn --target gfx1250 --backend=own --out $W/$kn.s 2>&1 | tail -3
  [ -f $W/$kn.s ] && grep -E "next_free_sgpr|vgpr_count|sgpr_count|^\s+(tensor_|s_wait_tensorcnt|s_barrier_(signal|wait)|v_wmma|s_cbranch|s_branch)" $W/$kn.s | sed -E 's/^\s+//' | sort | uniq -c | sort -rn | head -14
done