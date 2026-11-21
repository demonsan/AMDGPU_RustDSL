# L2 / memory traffic per dispatch (pmc_run.sh, the counters that count on e08): the MXFP4 GEMM kernels D / M / PCC and the vendor f4gemm
# at 8192^3 and 4096^3, the MLA _ws and the vendor MLA at 128 x 8192. Two short passes per run (the whole-node lock is held per pass only).
set -u
cd "$SRC"
A="$CARGO_TARGET_DIR/release/amdc"; CL=/opt/rocm/llvm/bin/clang; O=$W/pmctraffic; rm -rf $O; mkdir -p $O
cargo build -q --release -j 64 --bin amdc 2>&1 | grep -E "^error" | head -3
B=gemm_g1250_tdm_stage_f4_256x256_ks4_s2_epit_roles
for k in $B ${B}_c4x4_lock ${B}_c4x4_lock_persist_carry mla_g1250_qh16_ws; do
  $A emit $k --target gfx1250 --backend=own --out $O/$k.s >/dev/null 2>&1 && $CL -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=6 -o $O/$k.co $O/$k.s 2>/dev/null || { echo "EMIT-FAIL $k"; exit 4; }
done
H="/opt/rocm/bin/amdclang++ -x hip -O2 --offload-arch=gfx1250 --rocm-path=/opt/rocm -std=c++17 -Itests/gpu -L/opt/rocm/lib -lamdhip64 -Wl,-rpath,/opt/rocm/lib"
$H tests/gpu/g1250_f4own_host.cpp -o $O/own_host 2>/dev/null && $H tests/gpu/g1250_f4gemm_host.cpp -o $O/vendor_host 2>/dev/null && $H tests/gpu/g1250_mla_host.cpp -o $O/mla_host 2>/dev/null || { echo HOST-BUILD-FAIL; exit 6; }
VCO=$BASE/vendor/gfx1250/f4gemm/f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_ps.co; VK=$(grep -a -o '_ZN[0-9A-Za-z_]*E' $VCO | head -1)
MCO=$BASE/vendor/gfx1250/mla/mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co; MK=_ZN5aiter34mla_a8w8_qh16_1tg_16mx1_32nx4_np_3pE
P1="GL2C_READ_SECTORS GL2C_WRITE_SECTORS GL2C_HIT GL2C_MISS GL2C_REQ SQ_WAVES"
P2="GL2C_EA_RDREQ GL2C_EA_RDREQ_32B GL2C_EA_RDREQ_64B GL2C_EA_WRREQ GRBM_GUI_ACTIVE SQ_BUSY_CYCLES"
GPU=${PMC_GPU:-0}
pass() { tag=$1; filt=$2; shift 2
  for set in "$P1" "$P2"; do
    PASS_TIMEOUT=90 bash tools/mi450/pmc_run.sh $tag $GPU "$set" "$@" > $O/last.txt 2>&1
    grep -qE '"bad":0|MLA_OK' $O/last.txt || { echo "[$tag] RUN-FAIL: $(grep -E 'bad|FAIL|HANG|rror' $O/last.txt | head -2 | cut -c1-200)"; return; }
    L=$(ls -d /data/blyu/amdrustdsl/pmc/$tag/pass_* | tail -1)
    python3 tools/mi450/pmc_sum.py $L "$filt" | sed "s/^/[$tag] /" | cut -c1-600
  done; }
for sh in "8192 8192 8192" "4096 4096 4096"; do
  set -- $sh; t=$1
  unset CLUSTER; pass gemmD_$t $B $O/own_host $O/$B.co $B $sh 3 256 256 128
  export CLUSTER=4,4; pass gemmM_$t ${B}_c4x4_lock $O/own_host $O/${B}_c4x4_lock.co ${B}_c4x4_lock $sh 3 256 256 128
  pass gemmPCC_$t persist_carry $O/own_host $O/${B}_c4x4_lock_persist_carry.co ${B}_c4x4_lock_persist_carry $sh 3 256 256 128
  unset CLUSTER; pass gemmV_$t f4gemm $O/vendor_host $VCO $VK $sh 1 3 20000
done
pass mla_ws mla_g1250 $O/mla_host $O/mla_g1250_qh16_ws.co mla_g1250_qh16_ws 128 8192 16 3
pass mla_vendor mla_a8w8 $O/mla_host $MCO $MK 128 8192 16 3
echo DONE
