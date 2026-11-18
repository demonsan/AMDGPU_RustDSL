# correctness (bad = 0) + throughput of the MXFP4 TDM kernels after a layout change, gated: per kernel the smallest grid first, then 4096^3;
# stop at the first failure. Non-cluster kernels only (the cluster class: f4clpad.sh).
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/f4sweep; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=20
# name BM BN THREADS
for spec in "gemm_g1250_tdm_stage_f4 128 64 128" "gemm_g1250_tdm_stage_f4_ks2 128 64 128" "gemm_g1250_tdm_stage_f4_ks2_grouped 128 64 128" \
            "gemm_g1250_tdm_stage_f4_128x128_ks2 128 128 128" "gemm_g1250_tdm_stage_f4_128x128_ks4 128 128 128" "gemm_g1250_tdm_stage_f4_256x128 256 128 128" \
            "gemm_g1250_tdm_stage_f4_256x128_ks2 256 128 128" "gemm_g1250_tdm_stage_f4_256x256_ks2 256 256 128" "gemm_g1250_tdm_stage_f4_8w_256x64 256 64 256"; do
  set -- $spec; k=$1; bm=$2; bn=$3; th=$4
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { tail -3 $O/$k.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
  for s in "$bm $bn 1024" "4096 4096 4096"; do
    out=$(gpurun $O/host $O/$k.co $k $s 10 $bm $bn $th 2>&1)
    echo "$out" | grep -q '"ok":true' || { echo "$out" | tail -3; echo "STOP: $k $s"; exit 3; }
    echo "$k [$s]: $(echo "$out" | grep -oE '"tflops":[0-9.]+')"
  done
done
echo SWEEP-OK
