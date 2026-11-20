# VOPD default-flip check: every registered gfx1250 GEMM (f4 host) and MLA (mla host) kernel, A = VOPD off, B = VOPD on (AMDC_VOPD=1;
# AMDC_NO_VOPD=1 for A so the MLA kernels that carry the token are off too). Correctness first (a failing A skips the kernel: a host / launch
# mismatch, not VOPD; a failing B with a passing A stops the sweep), then alternating timings. One line per kernel.
set -u
cd $SRC; A=$CARGO_TARGET_DIR/release/amdc; CL=/opt/rocm/llvm/bin/clang; O=$W/vsweep; rm -rf $O; mkdir -p $O
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_f4own_host.cpp -o $O/f4host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || exit 6
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/mlahost -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib > /dev/null 2>&1 || exit 6
export PROBE_WATCHDOG_S=30
names=$(grep -E '&\[".*probe_g1250_' crates/amdgpu-kernels/src/registry.rs | head -1 | grep -oE '"[a-z0-9_]+"' | tr -d '"')
build() { # $1 kernel
  AMDC_NO_VOPD=1 $A emit $1 --target gfx1250 --backend=own --out $O/a.s > $O/a.log 2>&1 && AMDC_VOPD=1 $A emit $1 --target gfx1250 --backend=own --out $O/b.s > $O/b.log 2>&1 || return 1
  for v in a b; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$v.co $O/$v.s 2>/dev/null || return 1; done
  return 0
}
for k in $names; do
  case $k in
    gemm_g1250_tdm_stage_f4*)
      sh=$(echo $k | grep -oE '_[0-9]+x[0-9]+' | head -1 | tr -d '_'); BM=${sh%x*}; BN=${sh#*x}; [ -z "$sh" ] && { BM=256; BN=128; }
      build $k || { echo "$k: BUILD-SKIP"; continue; }
      nd=$(grep -c v_dual_ $O/b.s)
      g() { gpurun $O/f4host $O/$1.co $k $2 $2 $2 20 $BM $BN 128 2>&1; }
      ra=$(g a 512); echo "$ra" | grep -q '"ok":true' || { echo "$k: A-FAILS (skip)"; continue; }
      rb=$(g b 512); echo "$rb" | grep -q '"ok":true' || { echo "$k: B-FAILS with A passing -> STOP"; echo "$rb" | tail -2; exit 3; }
      t=""; for s in 4096 8192; do for r in 1 2; do for v in a b; do x=$(g $v $s | grep -oE '"tflops":[0-9.]+' | cut -d: -f2); t="$t $v$s=$x"; done; done; done
      echo "$k: dual $nd OK$t" ;;
    mla_g1250_*)
      build $k || { echo "$k: BUILD-SKIP"; continue; }
      nd=$(grep -c v_dual_ $O/b.s)
      m() { gpurun $O/mlahost $O/$1.co $k $2 $3 16 ${4:-20} 2>&1; }
      ra=$(m a 1 1000 2); echo "$ra" | grep -q MLA_OK || { echo "$k: A-FAILS (skip)"; continue; }
      rb=$(m b 1 1000 2); echo "$rb" | grep -q MLA_OK || { echo "$k: B-FAILS with A passing -> STOP"; echo "$rb" | tail -2; exit 3; }
      t=""; for s in "4 1000" "32 2048" "128 8192"; do set -- $s; for r in 1 2; do for v in a b; do o=$(m $v $1 $2); x=$(echo "$o" | grep -q MLA_OK && echo "$o" | grep -oE '"us":[0-9.]+' | cut -d: -f2 || echo FAIL); t="$t $v$1x$2=$x"; done; done; done
      echo "$k: dual $nd OK$t" ;;
  esac
done
echo VSWEEP-DONE
