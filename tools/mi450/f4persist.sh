# persistent vs non-persistent MXFP4 kernel (KA = non-persistent, KB = its persistent twin, tile BM x BN), gated: the persistent kernel at
# the smallest shape first (its smoke launch is one workgroup per tile column), every run checked (bad = 0), stop at the first failure;
# then 3 alternating pairs at 4096^3 / 8192^3 with PERSIST=${G:-256} workgroups. Never GPU 3 on f01-2.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/f4persist; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
for v in a b; do k=$([ $v = a ] && echo $KA || echo $KB); $A emit $k --target gfx1250 --backend=own --out $O/$v.s > $O/$v.log 2>&1 || { tail -3 $O/$v.log; exit 4; }; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || exit 5; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=20
G=${G:-256}
run() {
  v=$1; s=$2; k=$([ $v = a ] && echo $KA || echo $KB)
  if [ $v = b ]; then p=$(( G < (s / BN) * (s / BM) ? G : (s / BN) * (s / BM) )); out=$(PERSIST=$p gpurun $O/host $O/$v.co $k $s $s $s 20 $BM $BN 128 2>&1); else out=$(gpurun $O/host $O/$v.co $k $s $s $s 20 $BM $BN 128 2>&1); fi
  echo "$out" | grep -q '"ok":true' || { echo "$out" | tail -3; echo "STOP $v $s"; exit 3; }
  echo "$v $s: $(echo "$out" | grep -oE '"tflops":[0-9.]+')"
}
run b 512; run b 1024; run a 1024
for s in 4096 8192; do for r in 1 2 3; do run a $s; run b $s; done; done
echo PERSIST-OK
