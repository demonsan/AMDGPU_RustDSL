# the TDM-ring WMMA GEMM on hardware: emit + assemble the kernels, build the host, smallest shapes first (hard timeouts through gpurun); stops at the first failure or hang
set -u
cd "$SRC"
O="$W/gemmhw"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
for k in gemm_g1250_tdm_fp8_k192 gemm_g1250_tdm_fp8; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.emit.log 2>&1 || { echo "EMIT-FAIL $k"; tail -5 $O/$k.emit.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2> $O/$k.asm.log || { echo "ASM-FAIL $k"; tail -5 $O/$k.asm.log; exit 5; }
  echo "$k: $(wc -l < $O/$k.s) asm lines"
done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_gemm_host.cpp -o $O/g1250_gemm_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || { echo "HOST-BUILD-FAIL"; exit 6; }
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=${PROBE_WATCHDOG_S:-10}
for step in "$O/gemm_g1250_tdm_fp8_k192.co gemm_g1250_tdm_fp8_k192 32 32 192" "$O/gemm_g1250_tdm_fp8.co gemm_g1250_tdm_fp8 32 32 384" "$O/gemm_g1250_tdm_fp8.co gemm_g1250_tdm_fp8 64 64 384" "$O/gemm_g1250_tdm_fp8.co gemm_g1250_tdm_fp8 256 256 1536" "$O/gemm_g1250_tdm_fp8.co gemm_g1250_tdm_fp8 1024 1024 4608 20"; do
  echo "--- g1250_gemm_host $step"
  gpurun $O/g1250_gemm_host $step
  rc=$?
  if [ $rc -ne 0 ]; then echo "STOP: rc=$rc on '$step'"; exit $rc; fi
done
echo "ALL-GEMM-OK"
