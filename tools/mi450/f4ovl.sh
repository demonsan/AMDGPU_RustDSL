set -u
cd "$SRC"
O="$W/f4ovl"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=20 CHECK_ROWS=16 OUT_BF16=1
KL=${KERNELS:-gemm_g1250_tdm_stage_f4_256x256_ks4_s2_ovl8a_roles gemm_g1250_tdm_stage_f4_256x256_ks4_s2_ovl16a_roles}
CFGS=${CFGS:-"4096 4096 4096 10:128|8192 8192 8192 5:256|8192 8192 16384 3:256"}
for k in $KL; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL"; tail -3 $O/$k.log; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || continue
  echo "== $k"
  IFS='|'; for cfg in $CFGS; do unset IFS
    sh=${cfg%%:*}; P=${cfg##*:}
    export PERSIST=$P
    raw=$(gpurun $O/own_host $O/$k.co $k $sh 256 256 128 2>&1)
    out=$(echo "$raw" | grep -E '^\{"host"|HANG|FAIL|mismatch|PERSIST' | head -1)
    [ -n "$out" ] || echo "RAW: $(echo "$raw" | head -c 300)"
    echo "   $sh PERSIST=$P: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 us=\3 tflops=\4/')"
    case "$out" in *'"ok":true'*) ;; *) echo "   STOP: $out"; break;; esac
  IFS='|'; done; unset IFS
done