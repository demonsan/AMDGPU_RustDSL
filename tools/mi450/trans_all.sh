# TRANS co-execution evidence per reader class (after P1): the emitter's assembly of probe_g1250_trans_all, every TRANS result's first
# reader moved right behind the v_nop (trans_shape.py, register-checked), then per reader class a variant with only that class's v_nops
# removed; 5 runs x 1024 workgroups x 512 work-items each, compared bit for bit with the reference. Never on GPU 3.
set -u
cd "$SRC"
O="$W/transall"; rm -rf "$O"; mkdir -p "$O"
echo RUNNING > "$O/STATUS"
[ -z "${IDLE_GPU:-}" ] && { echo NOGPU > "$O/STATUS"; echo "NO-IDLE-GPU"; exit 251; }
[ "${IDLE_GPU:-none}" = "3" ] && { echo "REFUSE: GPU 3"; exit 9; }
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
k=probe_g1250_trans_all
$A emit $k --target gfx1250 --backend=own --out $O/ref.s > $O/emit.log 2>&1 || { echo EMIT-FAIL; tail -5 $O/emit.log; exit 4; }
python3 tools/mi450/trans_shape.py $O/ref.s $O || { echo SHAPE-FAIL; exit 7; }
for f in $O/shaped.s $O/no_*.s; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o ${f%.s}.co $f || { echo "ASM-FAIL $f"; exit 5; }; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo HOST-BUILD-FAIL; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-20}
gpurun $O/host trans $O/shaped.co $k 5 1024 $O/shaped.bin; rc=$?
[ $rc -ne 0 ] && { echo "FAIL $rc" > "$O/STATUS"; echo "STOP: rc=$rc on the reference"; exit $rc; }
for f in $O/no_*.co; do
  c=$(basename $f .co)
  gpurun $O/host trans $f $k 5 1024 $O/$c.bin; rc=$?
  [ $rc -ne 0 ] && { echo "FAIL $rc" > "$O/STATUS"; echo "STOP: rc=$rc on $c"; exit $rc; }
  cmp -s $O/shaped.bin $O/$c.bin && echo "CLASS ${c#no_}: bit-identical (5 runs)" || echo "CLASS ${c#no_}: DIFFERENT"
done
grep -q "^RUNNING" "$O/STATUS" && echo DONE > "$O/STATUS"