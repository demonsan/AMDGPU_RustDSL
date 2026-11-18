# CPU only: emit every MXFP4 variant (named kernels, no env) and print its LDS / register facts; the 256x128 repro is expected to fail in the back end
set -u
cd "$SRC"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
O="$W/f4var"; mkdir -p "$O"
for k in gemm_g1250_tdm_stage_f4 gemm_g1250_tdm_stage_f4_128x128 gemm_g1250_tdm_stage_f4_256x64 gemm_g1250_tdm_stage_f4_ks2 gemm_g1250_tdm_stage_f4_ks2_grouped gemm_g1250_tdm_stage_f4_256x128_repro; do
  if $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1; then
    echo "$k: OK  $(grep -m1 amdhsa_group_segment_fixed_size $O/$k.s | tr -s ' ')  $(grep -m1 vgpr_count $O/$k.s | tr -d ' ')"
  else
    echo "$k: FAIL: $(tail -3 $O/$k.log | tr '\n' ' ' | cut -c1-420)"
  fi
done
