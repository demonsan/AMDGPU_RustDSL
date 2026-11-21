# MODE 2 for the compiler's own code (token mode2 / AMDC_MODE2): the probes with the compiler's s_wait_alu waits (the prologue sets
# mode 2), 1 WG first, short watchdog, stop at the first failure; then the MXFP4 GEMM and the MLA correctness.
set -u
cd "$SRC"
O="$W/mode2c"; rm -rf "$O"; mkdir -p "$O"
[ -z "${IDLE_GPU:-}" ] && { echo "NO-IDLE-GPU"; exit 251; }
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
for k in probe_g1250_add probe_g1250_math probe_g1250_tdm probe_g1250_select probe_g1250_buffer probe_g1250_tdm2d probe_g1250_tdm_ring probe_g1250_named probe_g1250_wide; do
  AMDC_MODE2=1 $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k"; tail -3 $O/$k.log; exit 4; }
  grep -q "HW_REG_WAVE_SCHED_MODE, 0, 2), 2" $O/$k.s || { echo "NO-MODE2 $k"; exit 7; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || { echo "ASM-FAIL $k"; exit 5; }
  echo "$k: $(grep -c s_wait_alu $O/$k.s) s_wait_alu"
done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || { echo HOST-BUILD-FAIL; exit 6; }
echo "GPU: $IDLE_GPU"
export PROBE_WATCHDOG_S=5
for step in "tdm $O/probe_g1250_tdm.co probe_g1250_tdm 4" "add $O/probe_g1250_add.co probe_g1250_add 4096" "math $O/probe_g1250_math.co probe_g1250_math" "select $O/probe_g1250_select.co probe_g1250_select" "buffer $O/probe_g1250_buffer.co probe_g1250_buffer" "tdm $O/probe_g1250_tdm.co probe_g1250_tdm 64" "tdm2d $O/probe_g1250_tdm2d.co probe_g1250_tdm2d" "tdm_ring $O/probe_g1250_tdm_ring.co probe_g1250_tdm_ring" "named $O/probe_g1250_named.co probe_g1250_named" "wide $O/probe_g1250_wide.co probe_g1250_wide"; do
  out=$(gpurun $O/host $step 2>&1); rc=$?
  echo "$out" | grep -qE "PROBE_FAIL|FAIL" && rc=3
  echo "--- $step: rc=$rc $(echo "$out" | grep -E "OK|ok" | tail -1 | cut -c1-120)"
  [ $rc -ne 0 ] && { echo "$out" | tail -4; echo "STOP"; exit $rc; }
done
echo MODE2-PROBES-OK