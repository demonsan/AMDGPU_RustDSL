set -u
cd $SRC; O=$W/wtb; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_wmma_trans_bench.cpp -o $O/bench -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib 2>&1 | grep -E "error" | head -5
[ -x $O/bench ] || exit 6
echo "GPU: $IDLE_GPU"
for w in 4 8; do gpurun $O/bench $w 20000 2>&1 | grep -E "cycles|OK|HANG|error"; done