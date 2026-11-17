# P2 (only after P1 passed): a 4 KiB tensor load to LDS 4096 (control), 70000 and 130000 (LDS 163840), read back. Never on GPU 3.
set -u
cd "$SRC"
O="$W/p2"; rm -rf "$O"; mkdir -p "$O"
echo RUNNING > "$O/STATUS"
[ -z "${IDLE_GPU:-}" ] && { echo NOGPU > "$O/STATUS"; echo "NO-IDLE-GPU"; exit 251; }
[ "${IDLE_GPU:-none}" = "3" ] && { echo "REFUSE: GPU 3"; exit 9; }
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
for k in probe_g1250_tdm_hi_4096 probe_g1250_tdm_hi_70000 probe_g1250_tdm_hi_130000; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.emit.log 2>&1 || { echo "EMIT-FAIL $k"; tail -5 $O/$k.emit.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2> $O/$k.asm.log || { echo "ASM-FAIL $k"; tail -5 $O/$k.asm.log; exit 5; }
done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL"; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-10}
for k in probe_g1250_tdm_hi_4096 probe_g1250_tdm_hi_70000 probe_g1250_tdm_hi_130000; do
  echo "--- tdm_hi $k"
  gpurun $O/host tdm_hi $O/$k.co $k
  rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL $rc" > "$O/STATUS"; echo "STOP: rc=$rc on $k"; exit $rc; fi
done
echo "P2-OK"
grep -q "^RUNNING" "$O/STATUS" && echo DONE > "$O/STATUS"
