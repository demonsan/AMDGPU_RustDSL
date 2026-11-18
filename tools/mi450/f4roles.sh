# the cluster-multicast MXFP4 kernels (128x128 tile, ks 2): one cluster first (the smallest launch), then the grid grows; hard timeouts; stops at the first hang / wrong result
set -u
cd "$SRC"
O="$W/f4cl"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=10 CHECK_ROWS=${CHECK_ROWS:-256}
for k in ${NAMES:-gemm_g1250_tdm_stage_f4_256x256_ks2_pad gemm_g1250_tdm_stage_f4_256x256_ks2_pad_ss_relax gemm_g1250_tdm_stage_f4_256x256_ks2_pad_ss_roles gemm_g1250_tdm_stage_f4_256x256_ks2_pad_ss_roles_relax gemm_g1250_tdm_stage_f4_256x256_ks2_pad_roles_relax}; do
  case $k in *_c2x2*) cl=2,2;; *_c2x1*) cl=2,1;; *_c1x2*) cl=1,2;; *_c4x4*) cl=4,4;; *) cl=1,1;; esac
  case $k in *256x256*) bm=256; bn=256;; *) bm=128; bn=128;; esac
  case $k in *_bf16*) export OUT_BF16=1;; *) unset OUT_BF16;; esac
  case $k in *_ss*) export SCALE_SHUF=1;; *) unset SCALE_SHUF;; esac
  cx=${cl%,*}; cy=${cl#*,}
  kdc=$($A print-ir $k 2>/dev/null | grep -m1 '^kd "cluster_dims"' | awk '{print $3}'); [ "${kdc:-0}" = "$((cx*cy))" ] || { [ "$((cx*cy))" = 1 ] && [ -z "$kdc" ] && kdc=1; } || { echo "[$k] REFUSED: kd cluster_dims=${kdc:-none}, launch $cl"; continue; }
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL"; tail -2 $O/$k.log; continue; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || continue
  echo "== $k cluster $cl  $(grep -m1 amdhsa_group_segment_fixed_size $O/$k.s | tr -s ' ')  $(grep -m1 vgpr_count $O/$k.s | tr -d ' ')  tensor_loads=$(grep -c tensor_load_to_lds $O/$k.s) cluster_signals=$(grep -c 's_barrier_signal -3' $O/$k.s)"
  for sh in "$((bm*cy)) $((bn*cx)) 512 0" "$((bm*cy)) $((bn*cx)) 1024 0" "$((2*bm*cy)) $((2*bn*cx)) 1024 0" "1024 1024 1024 0" "4096 4096 4096 10" "8192 8192 8192 5"; do
    out=$(EXPECT_CLUSTER=$kdc CLUSTER=$cl gpurun $O/own_host $O/$k.co $k $sh $bm $bn 128 2>&1 | grep -E '^\{"host"|HANG|FAIL|mismatch|must divide' | head -1)
    echo "   $sh: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 maxrel=\2 us=\3 tflops=\4/')"
    case "$out" in *'"ok":true'*) ;; *) echo "   STOP $k"; break;; esac
  done
done