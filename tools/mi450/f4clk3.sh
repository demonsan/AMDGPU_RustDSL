set -u
cd "$SRC"
O="$W/f4clk3"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=25 CHECK_ROWS=16
B=gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles
KS="$B ${B}_stamps ${B}_c4x4_lock ${B}_c4x4_lock_stamps"
for k in $KS; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL"; tail -3 $O/$k.log | cut -c1-300; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null
done
run() { k=$1; sh=$2; unset CLUSTER EXPECT_CLUSTER STAMPS
  case $k in *c4x4*) export CLUSTER=4,4;; esac
  case $k in *stamps*) export STAMPS=1;; esac
  [ -f $O/$k.co ] || return
  raw=$(gpurun $O/own_host $O/$k.co $k $sh 256 256 128 2>&1)
  echo "$raw" | grep -E '"clock"|"stamps_kernel":\{"wave":0' | cut -c1-260
  out=$(echo "$raw" | grep -E '^\{"host"|HANG|REFUSED|FAIL|must divide' | head -1)
  echo "$k $sh: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 us=\3 tflops=\4/')"; }
for sh in "4096 4096 4096 10" "8192 8192 8192 5" "8192 8192 4096 5" "16384 8192 4096 3"; do
  echo "--- $sh"
  for r in 1 2; do for k in $KS; do run $k "$sh"; done; done
done
echo DONE