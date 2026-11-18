# Where does the time go? Profiling by deletion: the emitted assembly of one kernel with a class of instructions removed (results are wrong by construction; timing only, SKIPCHECK).
# CPU assembly + GPU timing; every variant is a plain loop-free-of-hazard transformation (no new waits, no new branches); hard timeouts.
set -u
cd "$SRC"
O="$W/f4prof"; rm -rf "$O"; mkdir -p "$O"
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error|Finished" | head -3
A="$CARGO_TARGET_DIR/release/amdc"
CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
echo "GPU: ${IDLE_GPU:-none}"
k=${KERNEL:-gemm_g1250_tdm_stage_f4_256x256_ks2_pad}
$A emit $k --target gfx1250 --backend=own --out $O/src.s > $O/emit.log 2>&1 || { echo EMIT-FAIL; tail -3 $O/emit.log; exit 4; }
mk() { # name awk-condition-on-n (0-based index of the tensor load; 4 per stage: A, B, SA, SB)
  awk -v cond="$2" '/^[ \t]+tensor_load_to_lds/ { n++; k=(n-1)%4; if ((cond=="scale" && k>=2) || (cond=="ab" && k<2) || (cond=="all")) next } { print }' $O/src.s > $O/$1.s
  $A hang-scan $O/$1.s > $O/$1.scan 2>&1 || { echo "[$1] HANG-SCAN FAIL (not run): $(head -c 300 $O/$1.scan | tr "\n" " ")"; return 1; }
  $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$1.co $O/$1.s || { echo "[$1] ASM-FAIL"; return 1; }
  echo "$1: wmma=$(grep -c v_wmma $O/$1.s) tensor=$(grep -c tensor_load $O/$1.s)"
}
export PROBE_WATCHDOG_S=20 CHECK_ROWS=1 SKIPCHECK=1
mk base none
mk no_scale_loads scale
mk no_ab_loads ab
mk no_tdm all
for v in base no_scale_loads no_ab_loads no_tdm; do
  [ -f $O/$v.co ] || { echo "   $v: skipped (no verified assembly)"; continue; }
  for sh in "4096 4096 4096 10" "8192 8192 8192 5"; do
    out=$(gpurun $O/own_host $O/$v.co $k $sh 256 256 128 2>&1 | grep -E '^\{"host"|HANG' | head -1)
    echo "   $v  $sh: $(echo "$out" | sed -E 's/.*"us":([0-9.]+),"tflops":([0-9.]+).*/us=\1 tflops=\2/')"
  done
done