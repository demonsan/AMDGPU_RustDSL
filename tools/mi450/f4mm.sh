# one cluster of the 2x1 multicast kernel with the per-row mismatch map (hard timeout, stop on hang)
set -u
cd "$SRC"
O="$W/f4mm"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
k=${KERNEL:-gemm_g1250_tdm_stage_f4_128x128_ks2_c2x1}
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL"; tail -3 $O/$k.log; exit 4; }
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
kdc=$($A print-ir $k 2>/dev/null | grep -m1 '^kd "cluster_dims"' | awk '{print $3}')
echo "kd cluster_dims=$kdc"
PROBE_WATCHDOG_S=10 CHECK_ROWS=128 MISMAP=1 EXPECT_CLUSTER=$kdc CLUSTER=2,1 gpurun $O/own_host $O/$k.co $k 128 256 512 0 128 128 128 2>&1 | grep -vE "^device" | head -150