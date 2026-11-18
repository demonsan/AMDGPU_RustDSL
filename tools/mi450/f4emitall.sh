# every registered MXFP4 TDM kernel compiles (all checks), with its LDS size: a CPU-only sweep after a layout change
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" -A7 | head -20
for k in $(grep -oE '"gemm_g1250_tdm_stage_f4[a-z0-9_]*"' crates/amdgpu-kernels/src/registry.rs | tr -d '"' | sort -u); do
  r=$($A emit $k --target gfx1250 --backend=own --out /tmp/bk_x.s 2>&1 | tail -1 | cut -c1-150)
  echo "${k} => ${r} $(grep -oE 'group_segment_fixed_size [0-9]+' /tmp/bk_x.s)"
done
