set -u
cd "$SRC"
O="$W/w1rp"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
sed "s/, 128, 1, 1,/, 256, 1, 1,/" tests/gpu/g1250_mla_host.cpp > $O/host256.cpp
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu $O/host256.cpp -o $O/host256 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host128 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
KS="mla_g1250_qh16_w1r mla_g1250_qh16_w1p mla_g1250_qh16_ws mla_g1250_qh16_ws2 mla_g1250_qh16_ws4"
for k in $KS; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k: $(tail -1 $O/$k.log | cut -c1-120)"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null
done
VCO=$BASE/vendor/gfx1250/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co; VK=_ZN5aiter34mla_a8w8_qh16_1tg_16mx1_32nx4_np_3pE
echo "GPU: ${IDLE_GPU:-none}"; export PROBE_WATCHDOG_S=25
run() { h=host128; [ "$1" = mla_g1250_qh16_w1r ] && h=host256
  co=$O/$1.co; kn=$1; [ "$1" = vendor ] && { co=$VCO; kn=$VK; }
  out=$(gpurun $O/$h $co $kn $2 $3 16 20 2>&1); ok=$(echo "$out" | grep -oE 'MLA_[A-Z]+' | head -1)
  echo "$1 $2x$3: $(echo "$out" | grep -oE '[0-9.]+ us' | head -1) $ok"; }
for s in "4 512" "4 1000" "32 2048" "128 4096" "128 8192" "512 8192"; do set -- $s
  for r in 1 2; do for k in $KS vendor; do [ -f $O/$k.co ] || [ $k = vendor ] || continue; run $k $1 $2; done; done
done
echo PAIR-OK