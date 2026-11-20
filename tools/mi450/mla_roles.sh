set -u
cd "$SRC"
O="$W/mlaroles"; rm -rf "$O"; mkdir -p "$O"
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
sed 's/, 128, 1, 1,/, 256, 1, 1,/' tests/gpu/g1250_mla_host.cpp > $O/host256.cpp
cp tests/gpu/g1250_timing.h $O/ 2>/dev/null
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu $O/host256.cpp -o $O/host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"; export PROBE_WATCHDOG_S=20 CHECK_B=0
for k in ${KS:-mla_g1250_qh16_roles}; do
$A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { tail -3 $O/$k.log; exit 4; }
$CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || exit 5
grep -E "amdhsa_next_free_vgpr|amdhsa_group_segment" $O/$k.s | tr -s ' '
for ctx in ${CTXS:-16384 32768}; do
  rm -f $O/dump.bin
  out=$(DUMP=$O/dump.bin gpurun $O/host $O/$k.co $k ${BATCH:-1} $ctx 16 20 2>&1 | grep -E '^mla B|HANG|FAIL' | head -2)
  us=$(echo "$out" | grep -oE "[0-9.]+ us" | head -1)
  cyc=$(python3 - $O/dump.bin <<'EOF'
import struct,sys
try:
    d=open(sys.argv[1],'rb').read()
    print(' '.join('%d/%d'%struct.unpack_from('<II',d,256*w) for w in range(8)))
except Exception as e:
    print('nodump', e)
EOF
)
  echo "$k ctx$ctx: $us [$cyc]"
done
done