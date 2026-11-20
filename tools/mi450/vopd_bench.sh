# VALU issue-rate microbenchmark (tests/gpu/g1250_vopd_bench.cpp): plain v_fmac vs VOPD pairs vs a dependent chain, 1 and 2 / 4 waves per SIMD
set -u
cd $SRC; O=$W/vopd; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_vopd_bench.cpp -o $O/bench -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib 2>&1 | grep -E "error" | head -5
[ -x $O/bench ] || exit 6
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_vopd_bench.cpp --cuda-device-only -S -o $O/bench.s 2>/dev/null; grep -c "v_dual_fmac" $O/bench.s
echo "GPU: $IDLE_GPU"
for w in 4 8 16; do gpurun $O/bench $w 20000 2>&1 | grep -E "fmac|v_dual|OK|HANG|error"; done
