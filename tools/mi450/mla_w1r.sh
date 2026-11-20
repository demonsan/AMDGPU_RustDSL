set -u
cd "$SRC"
O="$W/w1r"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
sed "s/, 128, 1, 1,/, 256, 1, 1,/" tests/gpu/g1250_mla_host.cpp > $O/host256.cpp
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu $O/host256.cpp -o $O/host256 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host128 -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
for k in mla_g1250_qh16_w1r mla_g1250_qh16_w1p; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "EMIT-FAIL $k"; tail -3 $O/$k.log; exit 4; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || exit 5
done
echo "GPU: ${IDLE_GPU:-none}"; export PROBE_WATCHDOG_S=25
run() { # kernel host B ctx
  out=$(gpurun $O/$2 $O/$1.co $1 $3 $4 16 20 2>&1); echo "$out" | grep -q MLA_OK || { echo "$out" | tail -4; echo "STOP: $1 $3x$4"; exit 3; }
  echo "$1 $3x$4: $(echo "$out" | grep -oE '[0-9.]+ us' | head -1) $(echo "$out" | grep -oE 'output hash [0-9a-f]+' | head -1)"
}
for c in ${CTXS:-64 100 128 129 200 256 320 1000 1100 4113 8192}; do run mla_g1250_qh16_w1r host256 1 $c; done
for r in 1 2 3; do run mla_g1250_qh16_w1r host256 32 2048; done
run mla_g1250_qh16_w1p host128 32 2048
echo W1R-OK