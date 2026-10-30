#!/usr/bin/env bash
# Independent validator for kernel-opt's safe-asm _vs128 inner window.
# Builds the base kernel and the window kernel from the same tree, then checks:
#   1) window flag actually changes the emitted code (not silently ignored),
#   2) output md5 bit-identical to base (arithmetic unchanged),
#   3) per-CU / filled timing vs same-run vendor (best+median),
#   4) counter direction: WAIT_INST_ANY down, XDL busy up, wave cycles down,
#   5) auto-path gate 15/15 @16 rows + determinism 5/5.
# Usage (own backend, GPU 5):
#   WINDOW_ENV='FMHA_OPT=cvt4,pksub,lazy,msum,permlane,kdlds,dbuf,lacc1,psh2,svpk,svcorr,swz,<token>' \
#     WINDOW_TAG=win bash tools/safeasm_window_validate.sh
# Env: REF_CO vendor co, REF default shared session copy; WARMUP/ITERS (200/100); GPU 5.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
REF="${REF_CO:-/shared/amdgpu/home/blyu_qle/sessions/ref_mxfp8.co}"
TAG="${WINDOW_TAG:-win}"
WARMUP="${WARMUP:-200}"; ITERS="${ITERS:-100}"
export AMD_CARGO_TARGET="${AMD_CARGO_TARGET:-/tmp/amd-target-occ}"
export AMD_ROCM_MOUNT="${AMD_ROCM_MOUNT:-/shared/apps/ubuntu/opt/rocm-10.1.0 /shared/amdgpu/home/blyu_qle/llvm-tools}"
export AMD_LLVM_BIN="${AMD_LLVM_BIN:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
export ASM_ROC="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}"
export ROCM_PATH="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
export HIPROOT="${HIPROOT:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
export ROCM_PROF="${ROCM_PROF:-/shared/apps/ubuntu/opt/rocm-7.2.4}"
unset GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-5}"
RP="$ROCM_PROF/bin/rocprofv3"
mkdir -p build out/wv
emit_asm() { # $1 tag, $2 extra env prefix
  local t="$1" e="$2"
  env $e bash tools/container/cargo.sh run -q --bin amdc -- emit fmha_fwd_hd128_mxfp8_vs128 --backend own --out /work/build/vs_$t.s > out/wv/emit_$t.log 2>&1 || { echo "EMIT FAIL $t"; tail -5 out/wv/emit_$t.log; exit 1; }
  "$ASM_ROC/llvm/bin/clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/vs_$t.co build/vs_$t.s || { echo "ASSEMBLE FAIL $t"; exit 1; }
  echo "[$t] vgpr/sgpr=$(grep -oE 'vgpr=[0-9]+ sgpr=[0-9]+' out/wv/emit_$t.log | head -1) asm_md5=$(md5sum build/vs_$t.s | cut -c1-16)"
}
echo "== build =="
emit_asm base ""
emit_asm "$TAG" "$WINDOW_ENV"
if [ "$(md5sum build/vs_base.s | cut -d' ' -f1)" = "$(md5sum build/vs_$TAG.s | cut -d' ' -f1)" ]; then
  echo "FATAL: window flag produced byte-identical asm - the flag is not active or not routed"; exit 2
fi
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/ref_host.cpp -o build/ref_host -I"$HIPROOT/include" -L"$HIPROOT/lib" -lamdhip64 -Wl,-rpath,"$HIPROOT/lib" || exit 1
run_one() { # $1 co, $2 kname, $3 shape env, $4 grep
  env $3 REF_CO2="$1" REF_KNAME2="$2" HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" ./build/ref_host "$REF" 2>&1 | grep -E "$4"
}
B64="REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=2 REF_ITERS=$ITERS REF_WARMUP=$WARMUP"
FILL="REF_B=4 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=2 REF_ITERS=50 REF_WARMUP=100"
echo "== timing (best+median of 5, same-run vendor) =="
for i in 1 2 3 4 5; do run_one build/vs_base.co fmha_fwd_hd128_mxfp8_vs128 "$B64" 'timing (vendor|ours)'; echo ---; done > out/wv/base64.log 2>&1
for i in 1 2 3 4 5; do run_one build/vs_$TAG.co fmha_fwd_hd128_mxfp8_vs128 "$B64" 'timing (vendor|ours)'; echo ---; done > out/wv/win64.log 2>&1
for i in 1 2 3; do run_one build/vs_base.co fmha_fwd_hd128_mxfp8_vs128 "$FILL" 'timing (vendor|ours)'; echo ---; done > out/wv/basefill.log 2>&1
for i in 1 2 3; do run_one build/vs_$TAG.co fmha_fwd_hd128_mxfp8_vs128 "$FILL" 'timing (vendor|ours)'; echo ---; done > out/wv/winfill.log 2>&1
echo "== md5 (B1H8S2048, bf16 REF_DUMP2 = _vs128 output; authoritative baseline faadf00c95f4eaffff94ece7d1429b3d) =="
env REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=4 REF_ITERS=1 REF_CO2=build/vs_base.co REF_KNAME2=fmha_fwd_hd128_mxfp8_vs128 REF_DUMP2=/tmp/wv_base.bin HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" ./build/ref_host "$REF" > out/wv/md5_base.log 2>&1
env REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=4 REF_ITERS=1 REF_CO2=build/vs_$TAG.co REF_KNAME2=fmha_fwd_hd128_mxfp8_vs128 REF_DUMP2=/tmp/wv_win.bin HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" ./build/ref_host "$REF" > out/wv/md5_win.log 2>&1
echo "base_md5=$(md5sum /tmp/wv_base.bin | cut -c1-32) win_md5=$(md5sum /tmp/wv_win.bin | cut -c1-32) (target faadf00c95f4eaffff94ece7d1429b3d)"
grep -E 'our kernel.*vs CPU' out/wv/md5_win.log | head -1
echo "== counters (window; compare with frozen §7.10.2 baseline) =="
VC="REF_B=1 REF_H=8 REF_S=2048 REF_SK=2048 REF_HK=8 REF_ROWS=4 REF_ITERS=10 REF_CO2=build/vs_$TAG.co REF_KNAME2=fmha_fwd_hd128_mxfp8_vs128 HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
wcp() { P=$1; shift; rm -rf out/wv/c_$P; mkdir -p out/wv/c_$P; env $VC timeout 150 "$RP" --pmc "$@" --output-format csv -d out/wv/c_$P -o r -- ./build/ref_host "$REF" > out/wv/c_$P.log 2>&1; echo "pass $P rc=$?"; }
wcp X1 SQ_WAVE_CYCLES
wcp X2 SQ_VALU_MFMA_BUSY_CYCLES
wcp X3 SQ_WAIT_ANY
wcp X4 SQ_WAIT_INST_ANY SQ_WAIT_INST_LDS
wcp X6 SQ_INSTS_VALU SQ_ACTIVE_INST_VALU SQ_INSTS_LDS SQ_INSTS_VMEM SQ_IFETCH SQ_INSTS_SALU
echo "== auto-path gate/determinism with the window build (env is inherited) =="
env $WINDOW_ENV REF_CO="$REF" OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto BACKEND=own REF_ROWS=16 HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" bash tests/gpu/run_ref.sh > out/wv/gate16.log 2>&1
echo "gate16 ok=$(grep -c REF_KERNEL_OK out/wv/gate16.log) fail=$(grep -c REF_KERNEL_FAIL out/wv/gate16.log) parity_ok=$(grep -c PARITY_OK out/wv/gate16.log) ours_bad=$(grep -c 'our kernel.*bad=[1-9]' out/wv/gate16.log)"
env $WINDOW_ENV REF_CO="$REF" OUR_KERNEL=fmha_fwd_hd128_mxfp8_auto BACKEND=own HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" bash tests/gpu/determinism.sh > out/wv/det.log 2>&1
echo "det ok=$(grep -c DET_OK out/wv/det.log) fail=$(grep -c DET_FAIL out/wv/det.log)"
python3 - <<'PY'
import collections,csv,glob,re
def runs(fn):
    out=[];cur={}
    for line in open(fn,errors='replace'):
        m=re.search(r'timing (vendor|ours)\s*:\s*([0-9.]+) ms/iter,\s*([0-9.]+) TFLOP/s',line)
        if m: cur['vendor' if m.group(1)=='vendor' else 'ours']=(float(m.group(2)),float(m.group(3)))
        if line.strip()=='---':
            if cur: out.append(cur); cur={}
    return out
def stat(fn):
    rs=runs(fn); o=[r['ours'][1] for r in rs if 'ours' in r]; v=[r['vendor'][1] for r in rs if 'vendor' in r]
    if not o: return None
    return (max(o), sorted(o)[len(o)//2], max(v) if v else None)
for name,fn in (('base64','out/wv/base64.log'),('win64','out/wv/win64.log'),('basefill','out/wv/basefill.log'),('winfill','out/wv/winfill.log')):
    s=stat(fn); print(name, 'best/med/vendor', s)
# counters
N=512*16; agg=collections.defaultdict(dict)
for tag in ('X1','X2','X3','X4','X6'):
    for f in glob.glob(f'out/wv/c_{tag}/*counter_collection.csv'):
        for r in csv.DictReader(open(f)):
            k=r.get('Kernel_Name','')
            if not k.startswith('fmha_fwd_hd128_mxfp8_vs128'): continue
            try: v=float(r['Counter_Value'])
            except: continue
            agg[r['Counter_Name']][tag]=v/N
for n in ('SQ_WAVE_CYCLES','SQ_VALU_MFMA_BUSY_CYCLES','SQ_WAIT_INST_ANY','SQ_WAIT_ANY','SQ_WAIT_INST_LDS','SQ_ACTIVE_INST_VALU','SQ_INSTS_VALU','SQ_INSTS_LDS','SQ_INSTS_VMEM','SQ_IFETCH','SQ_INSTS_SALU'):
    print('counter', n, round(agg[n][next(iter(agg[n]))],2) if agg[n] else None)
PY
echo "DONE"
