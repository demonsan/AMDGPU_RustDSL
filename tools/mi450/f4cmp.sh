set -u
cd "$SRC"
O="$W/f4cmp"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=20 CHECK_ROWS=16
KS="gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_c2x2_lock gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_c4x4_lock"
for k in $KS; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k"; tail -3 $O/$k.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || exit 5
done
for rep in 1 2; do
for K in ${KLIST:-2048 3072 4096 5120 6144 8192}; do
  line="rep$rep K=$K:"
  for k in $KS; do
    case $k in *_c2x2*) cl=2,2;; *_c4x4*) cl=4,4;; *) cl=1,1;; esac
    out=$(EXPECT_CLUSTER=$((${cl%,*}*${cl#*,})) CLUSTER=$cl gpurun $O/own_host $O/$k.co $k ${MN:-4096 4096} $K 20 256 256 128 2>&1 | grep -E '^\{"host"|FAIL|HANG|mismatch' | head -1)
    us=$(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/\3/'); bad=$(echo "$out" | sed -E 's/.*"bad":([0-9]+).*/\1/')
    line="$line [${cl}]=${us}us(bad$bad)"
  done
  echo "$line"
done
done