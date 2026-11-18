# correctness + perf of the registered MXFP4 variants (tile BMxBN, block threads): 1 work-group first, then the grid grows; hard timeouts; stops a kernel at its first failure, continues with the next
set -u
cd "$SRC"
O="$W/f4list"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
export PROBE_WATCHDOG_S=10 CHECK_ROWS=64
run() {  # name BM BN THR
  k=$1; bm=$2; bn=$3; thr=$4
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s > $O/$k.log 2>&1 || { echo "[$k] EMIT-FAIL: $(tail -2 $O/$k.log | tr '\n' ' ' | cut -c1-200)"; return; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s || { echo "[$k] ASM-FAIL"; return; }
  echo "== $k  $(grep -m1 amdhsa_group_segment_fixed_size $O/$k.s | tr -s ' ')  $(grep -m1 vgpr_count $O/$k.s | tr -d ' ')  set_vgpr_msb=$(grep -c s_set_vgpr_msb $O/$k.s)"
  for sh in "$bm $bn 128 0" "$bm $bn 1024 0" "1024 1024 1024 0" "4096 4096 4096 10" "8192 8192 8192 5"; do
    out=$(gpurun $O/own_host $O/$k.co $k $sh $bm $bn $thr 2>&1 | grep -E '^\{"host"|HANG|FAIL|mismatch' | head -1)
    echo "   $sh: $(echo "$out" | sed -E 's/.*"bad":([0-9]+),"maxrel":([0-9.e+-]+),"us":([0-9.]+),"tflops":([0-9.]+).*/bad=\1 maxrel=\2 us=\3 tflops=\4/')"
    case "$out" in *'"ok":true'*) ;; *) echo "   STOP $k"; return;; esac
  done
}
run gemm_g1250_tdm_stage_f4 128 64 128
run gemm_g1250_tdm_stage_f4_128x128 128 128 128
run gemm_g1250_tdm_stage_f4_256x64 256 64 128
run gemm_g1250_tdm_stage_f4_256x128 256 128 128
