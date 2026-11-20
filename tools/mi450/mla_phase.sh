set -u
cd "$SRC"
O="$W/mlaph"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"; export PROBE_WATCHDOG_S=20 CHECK_B=0
for m in ${MASKS:-0 10 1 2 6 8 3 7 f 1f}; do
  k=mla_g1250_qh16_ph_$m
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "mask $m: EMIT-FAIL: $(grep -m2 -E 'GV-|error|rror' $O/$k.log | cut -c1-220)"; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || { echo "mask $m: ASM-FAIL"; continue; }
  line="mask $m:"
  for ctx in 16384 32768; do
    rm -f $O/dump.bin
    out=$(DUMP=$O/dump.bin gpurun $O/host $O/$k.co $k 1 $ctx 16 20 2>&1 | grep -E '^mla B' | head -1)
    us=$(echo "$out" | grep -oE "[0-9.]+ us" | head -1 | cut -d' ' -f1)
    cyc=$(python3 - $O/dump.bin <<'EOF'
import struct,sys
try:
    d=open(sys.argv[1],'rb').read()
    print(' '.join('%d/%d'%struct.unpack_from('<II',d,256*w) for w in range(4)))
except Exception as e:
    print('nodump')
EOF
)
    line="$line  ctx$ctx: ${us}us [$cyc]"
  done
  echo "$line"
done