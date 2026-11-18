# bisect of the ks=2 wrong-result case (docs/mxfp4: one work-group, hard timeouts, idle GPU only): which K range of the tiles reaches the WMMA wrongly
set -u
cd "$SRC"
O="$W/f4bis"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=10 CHECK_ROWS=128
for k in ${KERNELS:-gemm_g1250_tdm_stage_f4_ks2_grouped gemm_g1250_tdm_stage_f4_ks2}; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || continue
  for K in ${KS:-512 1024}; do
    # the whole K, then one K tile (256 elements) at a time, then the two 128-halves of a tile
    for rng in "0,$K" $(for t in $(seq 0 $((K/256-1))); do echo "$((256*t)),$((256*t+256))"; done) ${HALVES:-}; do
      out=$(ONLY_K=$rng gpurun $O/own_host $O/$k.co $k 128 64 $K 0 128 64 128 2>&1 | grep -E '^\{"host"|HANG|FAIL' | head -1 | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+).*/bad=\1 maxrel=\2/')
      echo "[$k K=$K only_k=$rng] $out"
      case "$out" in *HANG*) echo "HANG: stopping"; exit 124;; esac
    done
  done
done