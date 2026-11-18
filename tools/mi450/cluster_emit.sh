# every registered cluster MXFP4 kernel still compiles (CPU only), plus the exact_1250 tests
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" -A7 | head -20
cargo test -q --release -j 64 -p amdgpu-kernels --test exact_1250 2>&1 | grep -E "^error|^test result|panicked" -A4 | head -10 | cut -c1-200
for k in $(grep -oE 'gemm_g1250_tdm_stage_f4[a-z0-9_]*_c[0-9]x[0-9][a-z_]*' crates/amdgpu-kernels/src/registry.rs | sort -u); do
  $A emit $k --target gfx1250 --backend=own --out /tmp/bk_c.s 2>&1 | tail -1 | cut -c1-120
done
