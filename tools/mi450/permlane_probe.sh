# probe_g1250_permlane on the GPU, gated: 1 workgroup full EXEC, then 64; then the partial-EXEC variant (the same code with
# EXEC = MASK around every permute; the host model applies the mask)
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/permlane; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" -A5 | head
k=probe_g1250_permlane
$A emit $k --target gfx1250 --backend=own --out $O/full.s > $O/emit.log 2>&1 || { tail -3 $O/emit.log; exit 4; }
grep -E "v_permlane" $O/full.s | head -6
M=${MASK:-0x5555aaa5}
awk -v m=$M '/^\s+v_permlane/ && !/idx_gen/ { print "    s_mov_b32 exec_lo, " m; print "    s_nop 7"; print; print "    s_mov_b32 exec_lo, -1"; print "    s_nop 7"; next } { print }' $O/full.s > $O/part.s
for v in full part; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || exit 5; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_probe_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || exit 6
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=10
WGS=1 DUMP_IDX=1 gpurun $O/host permlane $O/full.co $k 2>&1 | grep -E "permlane:|got|HANG|idx_gen" | head -20
WGS=64 gpurun $O/host permlane $O/full.co $k 2>&1 | grep -E "permlane:|HANG" | head -3
WGS=1 MASK=$M gpurun $O/host permlane $O/part.co $k 2>&1 | grep -E "permlane:|got|HANG" | head -20