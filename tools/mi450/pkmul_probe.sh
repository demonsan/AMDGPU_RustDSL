# probe_g1250_pkmul (in-place var_pk_mul) on the GPU, gated: 1 workgroup, then 64; also its v_pk_mul count and no extra copies
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/pkmul; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" -A5 | head
k=probe_g1250_pkmul
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/emit.log 2>&1 || { tail -3 $O/emit.log; exit 4; }
echo "v_pk_mul: $(grep -c v_pk_mul_f32 $O/$k.s)  v_mov in loop body: $(awk '/^BB1:/,/s_cbranch/' $O/$k.s | grep -c v_mov)"
awk '/^BB1:/,/s_cbranch/' $O/$k.s | head -20
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=10
WGS=1 gpurun $O/host pkmul $O/$k.co $k 2>&1 | grep -E "pkmul:|got|HANG" | head -8
WGS=64 gpurun $O/host pkmul $O/$k.co $k 2>&1 | grep -E "pkmul:|HANG" | head -3