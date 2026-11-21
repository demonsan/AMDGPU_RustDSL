cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/ws4m2; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || exit 6
AMDC_MODE2=1 $A emit mla_g1250_qh16_ws4 --target gfx1250 --backend=own --out $O/m2.s >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/m2.co $O/m2.s || exit 5
$A emit mla_g1250_qh16_ws4 --target gfx1250 --backend=own --out $O/m0.s >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/m0.co $O/m0.s || exit 5
export PROBE_WATCHDOG_S=10
for ctx in 64 65 128 256 1000 4113; do for v in m0 m2; do echo "ctx $ctx $v: $(CHECK_B=1 gpurun $O/host $O/$v.co mla_g1250_qh16_ws4 1 $ctx 16 2 5000 2>&1 | grep -oE 'max_abs=[0-9.e+-]+ cos_diff=[0-9.e+-]+ (MLA_OK|MLA_FAIL)|hash [0-9a-f]+' | tr '\n' ' ')"; done; done