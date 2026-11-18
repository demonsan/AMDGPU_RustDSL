# crossover of the relaxed tensor wait: epit_roles vs epit_roles_relax over (M N K) shapes; hard timeouts
set -u
cd "$SRC"
O="$W/f4xo"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=20 CHECK_ROWS=16
for k in gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles gemm_g1250_tdm_stage_f4_256x256_ks2_epit_roles_relax; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || continue
  echo "== $k"
  for sh in "4096 4096 3328 10" "4096 4096 3584 10" "4096 4096 3840 10" "4096 4096 4096 10" "4096 4096 4352 10" "4096 4096 4608 10" "4096 4096 5120 10" "8192 8192 3840 5" "8192 8192 4096 5" "8192 8192 4352 5"; do
    out=$(gpurun $O/own_host $O/$k.co $k $sh 256 256 128 2>&1 | grep -E '^\{"host"|HANG|FAIL|mismatch' | head -1)
    echo "   $sh: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 us=\3 tflops=\4/')"
    case "$out" in *'"ok":true'*) ;; *) echo "   STOP"; break;; esac
  done
done