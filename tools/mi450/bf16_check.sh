# the f32 -> bf16 epilogue codegen of the registered bf16 MXFP4 kernels (CPU only): conversion / store instruction histogram, and the
# gfx1250 spelling of v_cvt_pk_bf16_f32 checked with llvm-mc
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc
cargo build -q --release -j 96 --bin amdc 2>&1 | grep -E "^error" | head -3
for k in $(grep -oE '"gemm_g1250[a-z0-9_]*bf16[a-z0-9_]*"' crates/amdgpu-kernels/src/registry.rs | tr -d '"' | sort -u); do
  $A emit $k --target gfx1250 --backend=own --out /tmp/bk_b.s 2>&1 | grep -iE "error" | head -2 | cut -c1-200
  h=$(grep -oE '^\s+(v_cvt[a-z0-9_]*|v_perm[a-z0-9_]*|v_lshrrev_b32|v_and_or_b32|global_store[a-z0-9_]*|ds_store[a-z0-9_]*|tensor_store[a-z_]*)' /tmp/bk_b.s | sort | uniq -c | tr '\n' ' ' | tr -s ' ')
  echo "$k => $h"
done
printf '%s\n' 'v_cvt_pk_bf16_f32 v1, v2, v3' 'v_cvt_pk_bf16_f32 v1, s2, v3' | /opt/rocm/llvm/bin/llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -show-encoding 2>&1 | grep -E "encoding|error"
grep -n '"v_cvt_pk_bf16_f32"' crates/amdgpu-target/src/gfx1250_co/tables.rs
