# cluster + TDM pad on the 256 x 256 ks2 MXFP4 kernel vs the non-cluster padded kernel, gated: one cluster at the smallest shape first,
# every run checked (bad = 0), stop at the first failure.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/f4clpad; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
for k in gemm_g1250_tdm_stage_f4_256x256_ks2_pad gemm_g1250_tdm_stage_f4_256x256_ks2_c2x2_pad gemm_g1250_tdm_stage_f4_256x256_ks2_c4x4_pad; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { tail -3 $O/$k.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=20
run() { k=gemm_g1250_tdm_stage_f4_256x256_ks2_$1; cl=$2; s=$3; n=$(( ${cl%,*} * ${cl#*,} )); out=$(EXPECT_CLUSTER=$n CLUSTER=$cl gpurun $O/host $O/$k.co $k $s $s $s 20 256 256 128 2>&1); echo "$out" | grep -q '"ok":true' || { echo "$out" | tail -3; echo "STOP $1 $s"; exit 3; }; echo "$1 $s: $(echo "$out" | grep -oE '"tflops":[0-9.]+')"; }
run c2x2_pad 2,2 512
run c4x4_pad 4,4 1024
for s in 4096 8192; do for r in 1 2; do run pad 1,1 $s; run c2x2_pad 2,2 $s; run c4x4_pad 4,4 $s; done; done
echo CLPAD-OK