# MLA (kernel-opt's mla_g1250_qh16) A/B: $W/mla/a.s and $W/mla/b.s prepared by `amdc emit` from the two source versions; smallest shape first,
# each run must print MLA_OK, stop at the first failure; then alternating pairs on the larger shapes, the vendor kernel for reference. Never GPU 3.
set -u
cd "$SRC"
O="$W/mla"
[ -z "${IDLE_GPU:-}" ] && { echo "NO-IDLE-GPU"; exit 251; }
[ "${IDLE_GPU:-none}" = "3" ] && { echo "REFUSE: GPU 3"; exit 9; }
CL=/opt/rocm/llvm/bin/clang; k=${KERNEL:-mla_g1250_qh16}
for v in a b; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || exit 5; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
VCO=$BASE/vendor/gfx1250/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co; VK=_ZN5aiter34mla_a8w8_qh16_1tg_16mx1_32nx4_np_3pE
echo "GPU: $IDLE_GPU"
export PROBE_WATCHDOG_S=20
run() { out=$(gpurun $O/host $1 $2 $3 $4 16 20 2>&1); echo "$out" | grep -q MLA_OK || { echo "$out" | tail -4; echo "STOP: $1 $3x$4"; exit 3; }; echo "$(basename $1) $3x$4: $(echo "$out" | grep -oE '(us|time|hash)[^,]*' | tr '\n' ' ' | head -c 200)"; }
run $O/b.co $k 4 1000
run $O/a.co $k 4 1000
for s in "32 2048" "128 4096" "128 8192"; do
  for r in 1 2; do run $O/a.co $k $s; run $O/b.co $k $s; done
  run $VCO $VK $s
done
echo MLA-AB-OK