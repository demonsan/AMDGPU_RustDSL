# probe_g1250_lds_fadd on the GPU: 1 workgroup first, then 1024 x 5 (gated)
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/ldsfadd; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
python3 tools/gen_enc_tables_gfx1250.py --check 2>&1 | tail -1
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
k=probe_g1250_lds_fadd
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/emit.log 2>&1 || { tail -2 $O/emit.log; exit 4; }
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || exit 5
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=10
WGS=1 REPS=1 gpurun $O/host lds_fadd $O/$k.co $k 2>&1 | grep -E "lds_fadd|got|HANG" | head -5 || exit 3
WGS=1024 REPS=5 gpurun $O/host lds_fadd $O/$k.co $k 2>&1 | grep -E "lds_fadd|got|HANG" | head -5