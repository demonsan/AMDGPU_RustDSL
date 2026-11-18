# gfx1250 last-arriver memory-order probe (release = global_wb scope:SCOPE_DEV, acquire = global_inv scope:SCOPE_DEV, ticket =
# global_atomic_add_u32 th:TH_ATOMIC_RETURN): gated, 2 blocks first, then every block count up to 2432
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/lastarr; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/last_arriver_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: $IDLE_GPU"; export GPU_TIMEOUT=60
for k in ${KERNELS:-probe_last_arriver_plain_fenced}; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s >/dev/null 2>&1 || { echo EMIT-FAIL $k; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
  out=$(MAXN=2 gpurun $O/host $O/$k.co $k 5 2>&1); rc=$?; echo "$k MAXN=2 rc=$rc: $(echo "$out" | tail -2 | tr '\n' ' ' | cut -c1-200)"
  [ $rc -ne 0 ] && { echo STOP; exit 3; }
  out=$(gpurun $O/host $O/$k.co $k ${REPS:-20} 2>&1); rc=$?; echo "$k all rc=$rc: $(echo "$out" | tail -3 | tr '\n' ' ' | cut -c1-300)"
done
echo LASTARR-DONE