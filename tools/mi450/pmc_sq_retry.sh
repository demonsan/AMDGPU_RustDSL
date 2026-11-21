# SQ per-instruction counters, retry once VMP is armed (2026-11-22): every pass requires VMP armed (pmc_run.sh refuses otherwise),
# counter passes only, no SPM. The add probe (known instruction counts) first, then the MLA _ws2 2-page vs 4-page merge question.
set -u
cd "$SRC"
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang; O=$W/pmcsq; mkdir -p $O
export PMC_REQUIRE_VMP=1
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
$A emit probe_g1250_add --target gfx1250 --backend=own --out $O/add.s >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/add.co $O/add.s 2>/dev/null || exit 4
$A emit mla_g1250_qh16_ws2 --target gfx1250 --backend=own --out $O/ws2.s >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/ws2.co $O/ws2.s 2>/dev/null || exit 4
H="/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -Itests/gpu -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib"
$H tests/gpu/g1250_probe_host.cpp -o $O/host 2>/dev/null && $H tests/gpu/g1250_mla_host.cpp -o $O/mla_host 2>/dev/null || exit 6
# the static instruction counts of the probe, for the plausibility check (per wave: VALU / SALU lines of the kernel body)
echo "probe_g1250_add static: $(grep -cE '^\s+v_' $O/add.s) VALU, $(grep -cE '^\s+s_' $O/add.s) SALU lines per wave"
SQ1="SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_SMEM SQ_WAVES"
SQ2="SQ_INSTS_LDS SQ_INSTS_VEC32_VALU_WMMA SQ_INST_CYCLES_VALU_WMMA SQ_INST_CYCLES_LDS"
SQ3="SQ_WAIT_ANY SQ_WAIT_INST_ANY SQ_WAVE_CYCLES SQ_BUSY_CYCLES"
pass() { tag=$1; filt=$2; shift 2
  for set in "$SQ1" "$SQ2" "$SQ3"; do
    PASS_TIMEOUT=60 bash tools/mi450/pmc_run.sh $tag ${PMC_GPU:-0} "$set" "$@" > $O/last.txt 2>&1; rc=$?
    [ $rc -eq 5 ] && { grep REFUSED $O/last.txt; echo "STOP: VMP not armed"; exit 5; }
    L=$(ls -d /data/blyu/amdrustdsl/pmc/$tag/pass_* | tail -1); head -1 $L/vmp.txt
    python3 tools/mi450/pmc_sum.py $L "$filt" | sed "s/^/[$tag] /" | cut -c1-500
  done; }
pass sq_add probe_g1250 $O/host add $O/add.co probe_g1250_add 4096
# the merge question: 2 pages (ctx 1000 at B=4) vs 4 pages per wave (the dg path's page counts follow ctx)
pass sq_ws2_1000 mla_g1250 $O/mla_host $O/ws2.co mla_g1250_qh16_ws2 4 1000 16 3
pass sq_ws2_2048 mla_g1250 $O/mla_host $O/ws2.co mla_g1250_qh16_ws2 4 2048 16 3
echo DONE