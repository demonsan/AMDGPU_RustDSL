# WMMA operand reuse (matrix_a/b_reuse, AMDC_WMMA_REUSE) A/B with the clock and power: PCC (8192^3) and the default D kernel's stamped
# form (its own shader-cycle clock estimate), paired alternating, 3 reps. Clock / power: amd-smi sampled during a long run (REPS launches).
set -u
cd "$SRC"
O="$W/reuseclk"; rm -rf "$O"; mkdir -p "$O"
[ -z "${IDLE_GPU:-}" ] && { echo NO-IDLE-GPU; exit 251; }
cargo build --release --bin amdc -j 64 2>&1 | grep -E "^error" | head -3
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang
/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -I tests/gpu tests/gpu/g1250_f4own_host.cpp -o $O/own_host -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib || exit 6
B=gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles
KS=${KS:-"${B}_c4x4_lock_persist_carry ${B}_stamps"}
for k in $KS; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.a.s > /dev/null 2>&1 || { echo "EMIT-FAIL $k"; exit 4; }
  AMDC_WMMA_REUSE=1 $A emit $k --target gfx1250 --backend=own --out $O/$k.b.s > /dev/null 2>&1 || { echo "EMIT-FAIL $k"; exit 4; }
  echo "$k: reuse flags a $(grep -c '_reuse' $O/$k.a.s) b $(grep -c '_reuse' $O/$k.b.s)"
  for v in a b; do $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.$v.co $O/$k.$v.s || exit 5; done
done
echo "GPU: $IDLE_GPU"
export PROBE_WATCHDOG_S=60 CHECK_ROWS=16
REPS=${REPS:-3000}
run() { k=$1; v=$2; unset CLUSTER STAMPS
  case $k in *c4x4*) export CLUSTER=4,4;; esac
  case $k in *stamps*) export STAMPS=1;; esac
  rm -f $O/smi.txt
  ( for i in $(seq 1 4000); do amd-smi metric -g $IDLE_GPU -c -p 2>/dev/null | grep -E "SOCKET_POWER|^\s+CLK:" | head -2 | tr '\n' ' '; echo; sleep 0.05; done > $O/smi.txt ) &
  sp=$!
  raw=$(gpurun $O/own_host $O/$k.$v.co $k 8192 8192 8192 $REPS 256 256 128 2>&1)
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  out=$(echo "$raw" | grep -E '^\{"host"|HANG|REFUSED|FAIL' | head -1)
  ghz=$(echo "$raw" | grep -oE '"est_ghz":[0-9.]+' | head -1 | cut -d: -f2)
  us=$(echo "$out" | grep -oE '"us":[0-9.]+' | cut -d: -f2); bad=$(echo "$out" | grep -oE '"bad":[0-9]+' | cut -d: -f2)
  # the samples while busy (clock below the idle max and power above idle): median clock / power
  smi=$(awk '{for(i=1;i<=NF;i++){if($i=="SOCKET_POWER:")p=$(i+1); if($i=="CLK:")c=$(i+1)} if(p>0&&c>0) print c, p}' $O/smi.txt | awk '$2 > 1300' | sort -n | awk '{c[NR]=$1; p[NR]=$2} END{if(NR) printf "smi_mhz_med=%s smi_w_med=%s n=%d", c[int((NR+1)/2)], p[int((NR+1)/2)], NR}')
  cyc=""; [ -n "$ghz" ] && [ -n "$us" ] && cyc=$(awk -v g=$ghz -v u=$us 'BEGIN{printf "%.0f", g*u*1000}')
  echo "$k [$v]: bad=$bad us=$us est_ghz=${ghz:--} cycles=${cyc:--} $smi"
}
for r in 1 2 3; do for k in $KS; do run $k a; run $k b; done; done
echo DONE
