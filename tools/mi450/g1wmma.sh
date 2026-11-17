# gfx1250 WMMA probes on hardware (own-compiled): probe_g1250_wmma_{fp8,fp8_k128,f4}; emit, assemble with the node's clang, run kernel-opt's wmma host via gpurun. Stops at the first failure or hang.
set -u
cd "$SRC"
O="$W/g1wmma"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | tail -1
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
for k in probe_g1250_wmma_fp8 probe_g1250_wmma_fp8_k128 probe_g1250_wmma_f4; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.emit.log 2>&1 || { echo "EMIT-FAIL $k"; tail -5 $O/$k.emit.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2> $O/$k.asm.log || { echo "ASM-FAIL $k"; tail -5 $O/$k.asm.log; exit 5; }
  echo "$k: $(wc -l < $O/$k.s) asm lines, $(stat -c %s $O/$k.co) bytes"
done
/opt/rocm/bin/amdclang++ -x hip -O2 -w --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_wmma_probe_host.cpp -o $O/wmma_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL"; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
for step in "probe_g1250_wmma_fp8 fp8 1" "probe_g1250_wmma_fp8_k128 fp8k128 0" "probe_g1250_wmma_f4 f4 1"; do
  set -- $step
  echo "--- wmma_host $1 $2 with_c=$3"
  gpurun $O/wmma_host $O/$1.co $1 $2 $3 10000
  rc=$?
  if [ $rc -ne 0 ]; then echo "STOP: rc=$rc on '$step'"; exit $rc; fi
done
echo "ALL-WMMA-PROBES-OK"
