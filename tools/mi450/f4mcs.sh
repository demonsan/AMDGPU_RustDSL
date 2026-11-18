# the multicast probes on hardware (cluster of 2): which slices does each workgroup see? Variants: mask 0b11 with th 0 / 2 / 5, and mask 0 (no multicast) as the control. Hard timeouts, stops at the first hang.
set -u
cd "$SRC"
O="$W/f4mcs"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mcast_scan_host.cpp -o $O/mcast_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=10
for k in ${NAMES:-probe_g1250_mcast_scan_m0}; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL: $(tail -2 $O/$k.log | tr '\n' ' ' | cut -c1-300)"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || continue
  kdc=$($A print-ir $k 2>/dev/null | grep -m1 '^kd "cluster_dims"' | awk '{print $3}')
  echo "== $k (kd cluster_dims=$kdc): $(grep -m1 tensor_load_to_lds $O/$k.s | sed -E 's/^\s+//')"
  gpurun $O/mcast_host $O/$k.co $k ${kdc:-0} 2>&1 | grep -vE '^device'
  rc=${PIPESTATUS[0]}
  [ $rc -eq 124 ] && { echo "HANG: stopping"; exit 124; }
done