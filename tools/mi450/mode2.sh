# Scheduling mode 2 evidence: the compiler's code (no s_wait_alu: it targets mode 0) run with the wave put into mode 2 right after the
# kernel start sequence. AMDC_MODE2_SITES lists the dependencies mode 2 leaves to software in that code (LLVM's expert-mode model:
# va_vdst / vm_vsrc); every kernel's own host checks the results; 5 rounds under contention. Never on GPU 3.
set -u
cd "$SRC"
O="$W/mode2"; export MODE=${MODE:-2}; rm -rf "$O"; mkdir -p "$O"
echo RUNNING > "$O/STATUS"
[ -z "${IDLE_GPU:-}" ] && { echo NOGPU > "$O/STATUS"; echo "NO-IDLE-GPU"; exit 251; }
[ "${IDLE_GPU:-none}" = "3" ] && { echo "REFUSE: GPU 3"; exit 9; }
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
for k in probe_g1250_add probe_g1250_buffer probe_g1250_named probe_g1250_tdm_ring gemm_g1250_tdm_stage_f4; do
  AMDC_MODE2_SITES=1 $A emit $k --target gfx1250 --backend=own --out $O/$k.s 2> $O/$k.sites || { echo "EMIT-FAIL $k"; exit 4; }
  grep MODE2 $O/$k.sites | sed "s/^/$k: /"
  awk '{print} /global_prefetch_b8 v0, s\[6:7\] scope:SCOPE_SE/ && !done {print "    s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), " ENVIRON["MODE"]; done=1}' $O/$k.s > $O/$k.m2.s
  grep -c HW_REG_WAVE_SCHED_MODE $O/$k.m2.s | grep -q 1 || { echo "INJECT-FAIL $k"; exit 7; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.m2.s || { echo "ASM-FAIL $k"; exit 5; }
done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo HOST-BUILD-FAIL; exit 6; }
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo HOST-BUILD-FAIL; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-20}
for round in 1 2 3 4 5; do
  for step in "add $O/probe_g1250_add.co probe_g1250_add 4096" "buffer $O/probe_g1250_buffer.co probe_g1250_buffer" "named $O/probe_g1250_named.co probe_g1250_named" "tdm_ring $O/probe_g1250_tdm_ring.co probe_g1250_tdm_ring"; do
    out=$(gpurun $O/host $step 2>&1); rc=$?
    echo "$out" | grep -qE "PROBE_FAIL" && rc=3
    [ $rc -ne 0 ] && { echo "$out" | tail -3; echo "FAIL $rc" > "$O/STATUS"; echo "STOP: rc=$rc round $round on '$step'"; exit $rc; }
  done
  out=$(gpurun $O/own_host $O/gemm_g1250_tdm_stage_f4.co gemm_g1250_tdm_stage_f4 4096 4096 4096 10 2>&1); rc=$?
  echo "$out" | grep -E '^\{"host"' | head -1
  echo "$out" | grep -qE '"ok":true' || { echo "$out" | tail -3; echo "FAIL f4" > "$O/STATUS"; echo "STOP: f4 round $round"; exit 3; }
  echo "round $round: all correct"
done
grep -q "^RUNNING" "$O/STATUS" && echo DONE > "$O/STATUS"
echo MODE2-OK