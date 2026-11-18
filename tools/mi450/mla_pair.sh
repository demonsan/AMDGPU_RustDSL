# two MLA kernels KA / KB (registry names) paired, plus the vendor kernel, gated: every run must print MLA_OK (stop at the first
# failure); the smallest shape first. Shapes B x ctx: 4x1000, 32x2048, 128x4096, 128x8192, 512x8192.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/mlapair; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
for v in a b; do k=$([ $v = a ] && echo $KA || echo $KB); $A emit $k --target gfx1250 --backend=own --out $O/$v.s > $O/$v.log 2>&1 || { tail -2 $O/$v.log; exit 4; }; $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s || exit 5; done
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
VCO=$BASE/vendor/gfx1250/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co; VK=_ZN5aiter34mla_a8w8_qh16_1tg_16mx1_32nx4_np_3pE
echo "GPU: $IDLE_GPU"; export PROBE_WATCHDOG_S=20
run() { out=$(gpurun $O/host $1 $2 $3 $4 16 20 2>&1); echo "$out" | grep -q MLA_OK || { echo "$out" | tail -3; echo "STOP: $2 $3x$4"; exit 3; }; echo "$5 $3x$4: $(echo "$out" | grep -oE "[0-9.]+ us" | head -1)"; }
run $O/b.co $KB 1 1000 KB
run $O/a.co $KA 1 1000 KA
for s in "4 1000" "32 2048" "128 4096" "128 8192" "512 8192"; do
  set -- $s
  for r in 1 2; do run $O/a.co $KA $1 $2 KA; run $O/b.co $KB $1 $2 KB; done
  run $VCO $VK $1 $2 vendor
done
echo MLA-PAIR-OK
