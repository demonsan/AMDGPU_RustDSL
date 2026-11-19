set -u
cd "$SRC"
O="$W/f4tile"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=20 CHECK_ROWS=16
SHAPES=${SHAPES:-"2048 2048 2048 20|4096 4096 2048 10|4096 4096 4096 10|4096 4096 8192 5|8192 8192 8192 5"}
KL=${KERNELS:-gemm_g1250_tdm_stage_f4_256x256_ks2_swapbf16_roles:256:256 gemm_g1250_tdm_stage_f4_256x256_ks2_swapbf16_roles_relax:256:256}
for kb in $KL; do
  IFS=: read k bm bn <<< "$kb"; unset IFS; echo "[parse] k=$k bm=$bm bn=$bn"
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL"; tail -3 $O/$k.log; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || continue
  case $k in *bf16*) export OUT_BF16=1;; *) unset OUT_BF16;; esac
  echo "== $k (${bm}x${bn}) OUT_BF16=${OUT_BF16:-0}"
  IFS='|'; for sh in $SHAPES; do unset IFS
    raw=$(gpurun $O/own_host $O/$k.co $k $sh $bm $bn 128 2>&1); out=$(echo "$raw" | grep -E '^\{"host"|HANG|FAIL|mismatch' | head -1); [ -n "$out" ] || echo "RAW: $(echo "$raw" | head -c 300)"
    echo "   $sh: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 us=\3 tflops=\4/')"
    case "$out" in *'"ok":true'*) ;; *) echo "   STOP: $out"; break;; esac
  IFS='|'; done; unset IFS
done