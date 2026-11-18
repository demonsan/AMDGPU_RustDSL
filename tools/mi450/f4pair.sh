# paired A/B of two registered MXFP4 kernels KA / KB (same tile BM x BN), gated: one workgroup, then 1024^3, then 3 alternating pairs at
# 4096^3 / 8192^3; every run checked (bad = 0); stop at the first failure.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/f4pair; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
for v in a b; do k=$([ $v = a ] && echo $KA || echo $KB); $A emit $k --target gfx1250 --backend=own --out $O/$v.s > $O/$v.log 2>&1 || { tail -3 $O/$v.log; exit 4; }; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || exit 5; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=20
run() { k=$([ $1 = a ] && echo $KA || echo $KB); out=$(gpurun $O/host $O/$1.co $k $2 $3 $4 20 $BM $BN 128 2>&1); echo "$out" | grep -q '"ok":true' || { echo "$out" | tail -3; echo "STOP $1 $2"; exit 3; }; echo "$1 $2: $(echo "$out" | grep -oE '"tflops":[0-9.]+')"; }
run b $BM $BN 512; run a $BM $BN 512; run b 1024 1024 1024
for s in 4096 8192; do for r in 1 2 3; do run a $s $s $s; run b $s $s $s; done; done
echo PAIR-OK