# back-end option A/B on an MXFP4 kernel (KERNEL, BM, BN; A built with A_ENV, B with B_ENV; default B = AMDC_ACCWIN=1); smallest shape first,
# every run checked, stop at the first failure; then 3 alternating pairs at 4096 / 8192.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/f4acc; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
k=${KERNEL:-gemm_g1250_tdm_stage_f4_256x128}; BM=${BM:-256}; BN=${BN:-128}
env ${B_ENV:-AMDC_ACCWIN=1} $A emit $k --target gfx1250 --backend=own --out $O/b.s || exit 4
env ${A_ENV:-X=1} $A emit $k --target gfx1250 --backend=own --out $O/a.s || exit 4
echo "msb a: $(grep -c s_set_vgpr_msb $O/a.s) b: $(grep -c s_set_vgpr_msb $O/b.s)"
for v in a b; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || exit 5; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=20
run() { out=$(gpurun $O/host $O/$1.co $k $2 $2 $2 20 $BM $BN 128 2>&1); echo "$out" | grep -q '"ok":true' || { echo "$out" | tail -3; echo "STOP $1 $2"; exit 3; }; echo "$1 $2: $(echo "$out" | grep -oE '"tflops":[0-9.]+')"; }
run b 512; run a 512
for s in 4096 8192; do for r in 1 2 3; do run a $s; run b $s; done; done
echo AB-OK