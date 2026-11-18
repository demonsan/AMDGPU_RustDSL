set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/wsbr; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" -A3 | head -6
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=10
for k in mla_g1250_qh16_wsbr mla_g1250_qh16_wsbrg mla_g1250_qh16_ws; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k"; tail -1 $O/$k.log; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
  for c in 320 1000 4096; do
    out=$(gpurun $O/host $O/$k.co $k 1 $c 16 2 2>&1)
    echo "$out" | grep -qE "HANG|watchdog" && { echo "STOP hang $k $c"; exit 3; }
    echo "$k B=1 ctx=$c: $(echo "$out" | grep -oE 'cos_diff=[0-9.e+-]+ (MLA_OK|MLA_FAIL)' | head -1)"
  done
done