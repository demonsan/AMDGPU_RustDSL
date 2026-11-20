set -u
cd "$SRC"
O="$W/waves"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" -A8 | head -30
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_waves_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"; export PROBE_WATCHDOG_S=20
for m in ${MASKS:-01 02 03 21 22 11 12 31}; do
  k=probe_g1250_waves_$m
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "mask $m: EMIT-FAIL $(tail -2 $O/$k.log | cut -c1-300)"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || { echo "mask $m: ASM-FAIL"; continue; }
  echo "mask $m: $(gpurun $O/host $O/$k.co $k 2000 2>&1 | grep -E 'waves iters|HANG|rror' | head -1)"
done