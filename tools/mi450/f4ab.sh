# paired A/B of the MXFP4 kernel: B = this build, A = the same build with one back-end option off (AMDC_AB_OFF=<env var to set for A>);
# smallest shape first, every run checked (bad = 0), stop at the first failure; then 3 alternating pairs per large shape. Never on GPU 3.
set -u
cd "$SRC"
O="$W/f4ab"; rm -rf "$O"; mkdir -p "$O"
[ -z "${IDLE_GPU:-}" ] && { echo "NO-IDLE-GPU"; exit 251; }
[ "${IDLE_GPU:-none}" = "3" ] && { echo "REFUSE: GPU 3"; exit 9; }
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
k=${KERNEL:-gemm_g1250_tdm_stage_f4}
$A emit $k --target gfx1250 --backend=own --out $O/b.s || exit 4
env ${AMDC_AB_OFF:?set AMDC_AB_OFF}=1 $A emit $k --target gfx1250 --backend=own --out $O/a.s || exit 4
cmp -s $O/a.s $O/b.s && echo "NOTE: A and B identical"
for v in a b; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || exit 5; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"
export PROBE_WATCHDOG_S=20
run() { out=$(gpurun $O/host $O/$1.co $k $2 $3 $4 30 2>&1); echo "$out" | grep -q '"ok":true' || { echo "$out" | tail -3; echo "STOP: $1 $2"; exit 3; }; echo "$1 $2: $(echo "$out" | grep -oE '"tflops":[0-9.]+')"; }
run b 512 512 512
run a 512 512 512
for s in 4096 8192; do for r in 1 2 3; do run a $s $s $s; run b $s $s $s; done; done
echo AB-OK