# Which of the offered counters actually count on e08 (MI450, the pinned ROCm 10.2 image): every counter in short passes (batches of
# BATCH; a batch rocprofv3 rejects is retried one by one) over a small MLA run, reporting the median value per counter for the MLA kernel.
# Each pass holds the whole-node gpu-lock for a few seconds only (pmc_run.sh).
set -u
cd "$SRC"
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang; O=$W/pmcval; mkdir -p $O
[ -f $O/ws2.co ] || { $A emit mla_g1250_qh16_ws2 --target gfx1250 --backend=own --out $O/ws2.s >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/ws2.co $O/ws2.s 2>/dev/null; }
[ -f $O/mlahost ] || /opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 tests/gpu/g1250_mla_host.cpp -o $O/mlahost -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib >/dev/null 2>&1
NAMES=/data/blyu/amdrustdsl/pmc/counters_gfx1250.txt
RES=/data/blyu/amdrustdsl/pmc/sweep_result.txt; : > $RES
BATCH=${BATCH:-8}
mapfile -t all < $NAMES
run_set() {
  PASS_TIMEOUT=60 bash tools/mi450/pmc_run.sh sweep 0 "$1" $O/mlahost $O/ws2.co mla_g1250_qh16_ws2 1 512 16 2 3 > /tmp/sw.txt 2>&1
  L=$(ls -d /data/blyu/amdrustdsl/pmc/sweep/pass_* | tail -1)
  python3 tools/mi450/pmc_sum.py $L mla_g1250 2>/dev/null | sed -E 's/.*median: //' | tr ' ' '\n' | grep '='
}
for ((i = 0; i < ${#all[@]}; i += BATCH)); do
  set=$(printf '%s ' "${all[@]:i:BATCH}")
  out=$(run_set "$set")
  if [ -z "$out" ]; then
    for c in $set; do o=$(run_set "$c"); [ -n "$o" ] && echo "$o" >> $RES || echo "$c=REJECTED" >> $RES; done
  else
    echo "$out" >> $RES
  fi
done
echo "counters: $(wc -l < $RES), nonzero: $(grep -vcE '=0$|REJECTED' $RES), zero: $(grep -cE '=0$' $RES), rejected: $(grep -c REJECTED $RES)"
grep -vE '=0$|REJECTED' $RES | tr '\n' ' '
echo
