# the 8-wave 256x64 MXFP4 kernel (the one that hung a device): 1 work-group first, then the grid grows; hard timeouts, stops at the first hang or wrong result
set -u
cd "$SRC"
O="$W/f4w8"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
k=${KERNEL:-gemm_g1250_tdm_stage_f4_8w_256x64}
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k"; tail -3 $O/$k.log; exit 4; }
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
echo "GPU: ${IDLE_GPU:-none}  $(grep -m1 amdhsa_group_segment_fixed_size $O/$k.s | tr -s ' ')  dscnt waits: $(grep -c s_wait_dscnt $O/$k.s)"
export PROBE_WATCHDOG_S=10 CHECK_ROWS=64
for step in ${STEPS:-"256 64 128 0" "256 64 512 0" "256 64 1024 0" "512 128 1024 0" "1024 1024 1024 0" "4096 4096 4096 5" "8192 8192 8192 5"}; do
  echo "--- $step"
  gpurun $O/own_host $O/$k.co $k $step 256 64 256 2>&1 | grep -E '^\{"host"|HANG|FAIL|mismatch' | head -3
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then echo "STOP rc=$rc on '$step'"; exit $rc; fi
done
echo "ALL-W8-OK"
