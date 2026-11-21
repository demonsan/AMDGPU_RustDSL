set -u
cd "$SRC"
O="$W/f4pers4"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=25 CHECK_ROWS=16
KL=${KERNELS:-gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles exp_gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_persist}
for k in $KL exp_gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_persist_stamps gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_stamps; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL"; tail -4 $O/$k.log | cut -c1-300; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null
done
run() { k=$1; sh=$2; case $k in *_persist*) export PERSIST=${PG:-256};; *) unset PERSIST;; esac
  case $k in *stamps*) export STAMPS=1;; *) unset STAMPS;; esac
  [ -f $O/$k.co ] || return
  raw=$(gpurun $O/own_host $O/$k.co $k $sh 256 256 128 2>&1); echo "$raw" | grep '"stamps' | cut -c1-300
  out=$(echo "$raw" | grep -E '^\{"host"|HANG|REFUSED|FAIL' | head -1)
  echo "$k $sh: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 us=\3 tflops=\4/')"; }
for sh in "8192 8192 8192 5" "8192 8192 4096 5" "4096 4096 4096 10"; do
  for r in 1 2; do for k in $KL; do run $k "$sh"; done; done
done
for sh in "8192 8192 8192 3"; do for k in gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_stamps exp_gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_persist_stamps; do run $k "$sh"; done; done
echo DONE