set -u
cd "$SRC"
O="$W/mlaph2"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"; export PROBE_WATCHDOG_S=20
k=mla_g1250_qh16_w1i
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || exit 4
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || exit 5
VCO=$BASE/vendor/gfx1250/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co; VK=_ZN5aiter34mla_a8w8_qh16_1tg_16mx1_32nx4_np_3pE
for ctx in 16384 32768; do
  for p in "$O/$k.co $k w1i" "$VCO $VK vendor"; do set -- $p
    out=$(gpurun $O/host $1 $2 1 $ctx 16 20 2>&1 | grep -E '^mla B' | head -1)
    echo "$3 B=1 ctx=$ctx: $(echo "$out" | grep -oE '[0-9.]+ us' | head -1) $(echo "$out" | grep -oE 'MLA_[A-Z]+')"
  done
done