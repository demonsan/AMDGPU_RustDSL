#!/usr/bin/env bash
# Hardware check of the jump-threading pass (lowering token `jthread`, crates/amdgpu-codegen/src/cfg_thread.rs) on the two-exit test kernel
# (crates/amdgpu-kernels/tests/jthread.rs: out[tid] = 1.0 when n is odd, 2.0 when n is even). Usage: HIP_VISIBLE_DEVICES=0 bash tools/jthread_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/jthread
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
bash tools/container/cargo.sh test -q --release -p amdgpu-kernels --test jthread write_asm -- --ignored > build/jthread/emit.log 2>&1 || { tail -5 build/jthread/emit.log; exit 1; }
head -c 4 /dev/zero > build/jthread/in.bin
bad=0
for v in base jthread; do
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/jthread/$v.co build/jthread/$v.s || exit 1
  for n in 1 2 3 4 5 6 7 8; do
    PROBE_I32=$n ./build/tileop_probe_host build/jthread/$v.co probe_two_exit 64 build/jthread/in.bin 256 build/jthread/out.bin > /dev/null || exit 1
    want=$(( n % 2 == 1 ? 1065353216 : 1073741824 ))
    got=$(python3 -c "import struct;d=open('build/jthread/out.bin','rb').read();v=struct.unpack('<64I',d);print(sum(1 for x in v if x!=$want))")
    echo "$v n=$n wrong lanes: $got"; [ "$got" = 0 ] || bad=1
  done
done
[ $bad = 0 ] && echo "JTHREAD_OK" || echo "JTHREAD_FAIL"
