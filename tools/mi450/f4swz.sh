# tile-order swizzle: epit_roles / epit_roles_relax with and without SWIZZLE=G over shapes; hard timeouts
set -u
cd "$SRC"
O="$W/f4swz"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=20 CHECK_ROWS=16
for k in gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_swz8 gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax_swz4 gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax_swz8 gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax_swz16; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || continue
  case $k in *_swz*) export SWIZZLE=${k##*swz};; *) unset SWIZZLE;; esac
  echo "== $k (SWIZZLE=${SWIZZLE:-off})"
  for sh in "2048 2048 2048 20" "4096 4096 4096 10" "4096 4096 8192 5" "8192 8192 4096 5" "8192 8192 8192 5" "8192 8192 16384 3"; do
    out=$(gpurun $O/own_host $O/$k.co $k $sh 256 256 128 2>&1 | grep -E '^\{"host"|HANG|FAIL|mismatch|SWIZZLE' | head -1)
    echo "   $sh: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 us=\3 tflops=\4/')"
    case "$out" in *'"ok":true'*) ;; *) echo "   STOP: $out"; break;; esac
  done
done