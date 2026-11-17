# gfx1250 G1 probes on hardware: build amdc, emit probe_g1250_{add,math,tdm} for gfx1250, assemble with the node's clang, run the hosts via gpurun (smallest configuration first, hard timeouts).
# Stops at the first failure or hang (exit != 0): report, do not retry.
set -u
cd "$SRC"
O="$W/g1probe"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | tail -2
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
for k in probe_g1250_add probe_g1250_math probe_g1250_tdm probe_g1250_select probe_g1250_buffer probe_g1250_tdm2d probe_g1250_tdm_ring probe_g1250_named probe_g1250_wide; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.emit.log 2>&1 || { echo "EMIT-FAIL $k"; tail -5 $O/$k.emit.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2> $O/$k.asm.log || { echo "ASM-FAIL $k"; tail -5 $O/$k.asm.log; exit 5; }
  echo "$k: $(wc -l < $O/$k.s) asm lines, $(stat -c %s $O/$k.co) bytes"
done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/g1250_probe_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL"; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-10}
for step in "tdm $O/probe_g1250_tdm.co probe_g1250_tdm 4" "add $O/probe_g1250_add.co probe_g1250_add 4096" "math $O/probe_g1250_math.co probe_g1250_math" "select $O/probe_g1250_select.co probe_g1250_select" "buffer $O/probe_g1250_buffer.co probe_g1250_buffer" "tdm $O/probe_g1250_tdm.co probe_g1250_tdm 64" "tdm2d $O/probe_g1250_tdm2d.co probe_g1250_tdm2d" "tdm_ring $O/probe_g1250_tdm_ring.co probe_g1250_tdm_ring" "named $O/probe_g1250_named.co probe_g1250_named" "wide $O/probe_g1250_wide.co probe_g1250_wide"; do
  echo "--- g1250_probe_host $step"
  gpurun $O/g1250_probe_host $step
  rc=$?
  if [ $rc -ne 0 ]; then echo "STOP: rc=$rc on '$step'"; exit $rc; fi
done
echo "ALL-PROBES-OK"
