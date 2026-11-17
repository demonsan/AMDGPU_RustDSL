# our MXFP4 TDM GEMM on hardware: emit + assemble, build the host, smallest shapes first (hard timeouts through gpurun); stops at the first failure or hang
set -u
cd "$SRC"
O="$W/f4hw"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
k=gemm_g1250_tdm_stage_f4
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.emit.log 2>&1 || { echo "EMIT-FAIL $k"; tail -5 $O/$k.emit.log; exit 4; }
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2> $O/$k.asm.log || { echo "ASM-FAIL $k"; tail -5 $O/$k.asm.log; exit 5; }
echo "$k: $(wc -l < $O/$k.s) asm lines"
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/g1250_f4own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL"; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-10}
for step in "128 64 128" "128 64 256" "256 128 512" "1024 1024 2048" "4096 4096 4096 20" "8192 8192 8192 10"; do
  echo "--- g1250_f4own_host $k $step"
  gpurun $O/g1250_f4own_host $O/$k.co $k $step
  rc=$?
  if [ $rc -ne 0 ]; then echo "STOP: rc=$rc on '$step'"; exit $rc; fi
done
echo "ALL-F4-OK"
