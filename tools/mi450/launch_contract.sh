set -u
cd "$SRC"
O="$W/chk"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
HC="/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu"
for h in g1250_probe_host g1250_f4own_host g1250_mla_host g1250_gemm_host g1250_f4gemm_host g1250_mcast_host g1250_mcast_scan_host g1250_launch_host g1250_wmma_probe_host g1250_waves_host; do
  $HC tests/gpu/$h.cpp -o $O/$h -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > $O/$h.log 2>&1 && echo "built $h" || { echo "BUILD-FAIL $h"; grep error $O/$h.log | head -3; }
done
emit() { $A emit $1 --target gfx1250 --backend=own --out $O/$1.s >/dev/null 2>&1; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$1.co $O/$1.s 2>/dev/null; }
emit gemm_g1250_tdm_stage_f4_256x128_ks2_persist; emit gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles; emit gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_stamps; emit mla_g1250_qh16_w1r; emit mla_g1250_qh16_w1p
export PROBE_WATCHDOG_S=20
echo "--- the incident: a persistent kernel without PERSIST (must be refused before any launch)"
$O/g1250_f4own_host $O/gemm_g1250_tdm_stage_f4_256x128_ks2_persist.co gemm_g1250_tdm_stage_f4_256x128_ks2_persist 512 512 512 0 256 128 128; echo "rc=$?"
echo "--- a persistent kernel with PERSIST=G smaller than a tile column count (existing check)"
PERSIST=3 $O/g1250_f4own_host $O/gemm_g1250_tdm_stage_f4_256x128_ks2_persist.co gemm_g1250_tdm_stage_f4_256x128_ks2_persist 512 512 512 0 256 128 128; echo "rc=$?"
echo "--- a stamped kernel without STAMPS"
$O/g1250_f4own_host $O/gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_stamps.co gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_stamps 512 512 512 0 256 256 128; echo "rc=$?"
echo "--- a plain kernel with PERSIST set"
PERSIST=2 $O/g1250_f4own_host $O/gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles.co gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles 512 512 512 0 256 256 128; echo "rc=$?"
echo "--- the 256-work-item role kernel launched with 128 work-items (default BLOCK)"
$O/g1250_mla_host $O/mla_g1250_qh16_w1r.co mla_g1250_qh16_w1r 1 128 16 5; echo "rc=$?"
echo "--- a 128-work-item kernel launched with BLOCK=256"
BLOCK=256 $O/g1250_mla_host $O/mla_g1250_qh16_w1p.co mla_g1250_qh16_w1p 1 128 16 5; echo "rc=$?"
echo "--- legal launches (1 work-group sizes, small): own GEMM plain + MLA w1r with BLOCK=256"
$O/g1250_f4own_host $O/gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles.co gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles 512 512 512 0 256 256 128 2>&1 | tail -2; echo "rc=$?"
BLOCK=256 gpurun $O/g1250_mla_host $O/mla_g1250_qh16_w1r.co mla_g1250_qh16_w1r 1 128 16 5 2>&1 | grep -E 'MLA_|LAUNCH' | head -2
echo "--- cluster contract: a cluster kernel without CLUSTER (must be refused)"
emit gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_c4x4_lock
EXPECT_CLUSTER=16 $O/g1250_f4own_host $O/gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_c4x4_lock.co gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_c4x4_lock 1024 1024 1024 0 256 256 128; echo "rc=$?"
echo "--- a plain kernel with CLUSTER=2,2"
CLUSTER=2,2 $O/g1250_f4own_host $O/gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles.co gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles 1024 1024 1024 0 256 256 128; echo "rc=$?"
grep -c cluster_workgroups $O/gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles_c4x4_lock.s
