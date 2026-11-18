# the vendor-shaped MXFP4 variant (256 x 256 per workgroup, ks = 2) vs 128 x 128 ks2 and the vendor f4gemm, gated: one workgroup first,
# then small -> large; every run checked (bad = 0), stop at the first failure. Never GPU 3 on f01-2.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/f4big; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
emit() { env $3 $A emit $1 --target gfx1250 --backend=own --out $O/$2.s > $O/$2.log 2>&1 || { tail -3 $O/$2.log; exit 4; }; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$2.co $O/$2.s || exit 5; }
emit gemm_g1250_tdm_stage_f4_256x256_ks2 big X=1
emit gemm_g1250_tdm_stage_f4_256x256_ks2 bigacc AMDC_ACCWIN=1
emit gemm_g1250_tdm_stage_f4_128x128_ks2 mid X=1
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -Itests/gpu tests/gpu/g1250_f4gemm_host.cpp -o $O/vhost -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
VCO=$BASE/vendor/gfx1250/f4gemm/f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_ps.co; VK=$(grep -a -o '_ZN[0-9A-Za-z_]*E' $VCO | head -1)
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=20
run() { out=$(gpurun $O/host $O/$1.co $2 $3 $4 $5 20 $6 $7 128 2>&1); echo "$out" | grep -q '"ok":true' || { echo "$out" | tail -3; echo "STOP $1 $3"; exit 3; }; echo "$1 ${3}x${4}x${5}: $(echo "$out" | grep -oE '"tflops":[0-9.]+')"; }
K=gemm_g1250_tdm_stage_f4_256x256_ks2; KM=gemm_g1250_tdm_stage_f4_128x128_ks2
run big $K 256 256 512 256 256
run bigacc $K 256 256 512 256 256
run big $K 1024 1024 1024 256 256
run bigacc $K 1024 1024 1024 256 256
for s in 4096 8192; do for r in 1 2; do
  run mid $KM $s $s $s 128 128; run big $K $s $s $s 256 256; run bigacc $K $s $s $s 256 256
  echo -n "vendor $s: "; gpurun $O/vhost $VCO $VK $s $s $s 1 50 20000 2>&1 | grep -oE '"tflops":[0-9.]+|FAIL' | head -1
done; done
echo BIG-OK