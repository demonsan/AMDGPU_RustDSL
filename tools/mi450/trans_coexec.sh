# TRANS co-execution evidence (after P1): the probe with the emitter's v_nops (reference) and a copy with every v_nop removed (TRANS
# result read by the very next VALU); 5 runs x 1024 workgroups of 512 each; the two dumps must be bit-identical. Never on GPU 3.
set -u
cd "$SRC"
O="$W/trans"; rm -rf "$O"; mkdir -p "$O"
[ "${IDLE_GPU:-none}" = "3" ] && { echo "REFUSE: GPU 3"; exit 9; }
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
k=probe_g1250_trans
$A emit $k --target gfx1250 --backend=own --out $O/ref.s > $O/emit.log 2>&1 || { echo EMIT-FAIL; tail -5 $O/emit.log; exit 4; }
echo "v_nop sites: $(grep -cE '^\s*v_nop\s*$' $O/ref.s)"
grep -B1 -A1 -E '^\s*v_nop\s*$' $O/ref.s | head -9
sed -E '/^\s*v_nop\s*$/d' $O/ref.s > $O/nonop.s
for v in ref nonop; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || { echo ASM-FAIL; exit 5; }; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo HOST-BUILD-FAIL; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-20}
for v in ref nonop; do
  gpurun $O/host trans $O/$v.co $k 5 1024 $O/$v.bin; rc=$?
  [ $rc -ne 0 ] && { echo "STOP: rc=$rc on $v"; exit $rc; }
done
cmp $O/ref.bin $O/nonop.bin && echo "TRANS-COEXEC: no-v_nop build bit-identical to the reference (5 runs)" || echo "TRANS-COEXEC: DIFFERENT"