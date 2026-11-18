# kernel-opt's var_scale_lane + in-place WMMA repro (probe_g1250_varscale{4,32}[_br]): 1 workgroup, N = 1, 2, 4 loop iterations; the
# host checks O against the exact reference. Gated: stop at a hang (the watchdog); a wrong result is reported and the sweep goes on.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/varscale; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=10
for nv in 4 32; do for br in "" _br; do
  k=probe_g1250_varscale${nv}${br}
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k: $(tail -1 $O/$k.log | cut -c1-150)"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || { echo ASM-FAIL; exit 5; }
  for n in 1 2 4; do
    out=$(NV=$nv N=$n gpurun $O/host varscale $O/$k.co $k 2>&1); rc=$?
    echo "$out" | grep -qE "HANG|watchdog" && { echo "$out" | tail -2; echo "STOP hang $k N=$n"; exit 3; }
    echo "$k N=$n rc=$rc: $(echo "$out" | grep -E 'PROBE_OK|PROBE_FAIL|bad=' | head -1 | cut -c1-160)"
  done
done; done
echo VARSCALE-DONE
