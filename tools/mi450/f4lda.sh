set -u
cd "$SRC"
O="$W/f4lda"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=20 CHECK_ROWS=16
for k in gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_lda; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k"; tail -3 $O/$k.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || exit 5
  for K in 3072 4096 5120 8192; do
    line="K=$K:"
    for pad in 0 64 256 1024 4096; do
      out=$(LDA_PAD=$pad gpurun $O/own_host $O/$k.co $k 4096 4096 $K 20 256 256 128 2>&1 | grep -E '^\{"host"|FAIL|HANG|mismatch' | head -1)
      us=$(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/\3/'); bad=$(echo "$out" | sed -E 's/.*"bad":([0-9]+).*/\1/')
      line="$line pad$pad=${us}us(bad$bad)"
    done
    echo "$line"
  done
done