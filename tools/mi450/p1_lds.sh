# P1: plain LDS round trip over the whole allocation (below / above 64 KiB, unaligned 86784 and aligned 163840), single workgroup first,
# then many concurrent workgroups. Stops at the first failure or hang. Never on GPU 3 (the hung one).
set -u
cd "$SRC"
O="$W/p1"; rm -rf "$O"; mkdir -p "$O"
echo RUNNING > "$O/STATUS"
[ -z "${IDLE_GPU:-}" ] && { echo NOGPU > "$O/STATUS"; echo "NO-IDLE-GPU"; exit 251; }
[ "${IDLE_GPU:-none}" = "3" ] && { echo "REFUSE: GPU 3"; exit 9; }
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
for k in probe_g1250_lds_61440 probe_g1250_lds_86784 probe_g1250_lds_163840; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.emit.log 2>&1 || { echo "EMIT-FAIL $k"; tail -5 $O/$k.emit.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2> $O/$k.asm.log || { echo "ASM-FAIL $k"; tail -5 $O/$k.asm.log; exit 5; }
  echo "$k: $(grep group_segment_fixed_size $O/$k.s | head -1)"
done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL"; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-10}
for step in "probe_g1250_lds_61440 1" "probe_g1250_lds_86784 1" "probe_g1250_lds_163840 1" "probe_g1250_lds_61440 1024" "probe_g1250_lds_86784 1024" "probe_g1250_lds_163840 1024"; do
  set -- $step
  echo "--- lds_big $1 x $2"
  gpurun $O/host lds_big $O/$1.co $1 $2
  rc=$?
  if [ $rc -ne 0 ]; then echo "FAIL $rc" > "$O/STATUS"; echo "STOP: rc=$rc on '$step'"; exit $rc; fi
done
echo "P1-OK"
grep -q "^RUNNING" "$O/STATUS" && echo DONE > "$O/STATUS"
