#!/usr/bin/env bash
# Hardware check of the E5 exact branch forms (crates/amdgpu-kernels/tests/e5_branches.rs: out[tid] = 1.0 when n is odd, 2.0 when n is
# even, carried in a scope variable; out[64 + tid] = 3.0 from the cold branch when n >= 3), placed and plain. Usage:
# HIP_VISIBLE_DEVICES=0 bash tools/e5_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/e5
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
bash tools/container/cargo.sh test -q --release -p amdgpu-kernels --test e5_branches write_asm -- --ignored > build/e5/emit.log 2>&1 || { tail -5 build/e5/emit.log; exit 1; }
head -c 4 /dev/zero > build/e5/in.bin
bad=0
for v in plain placed; do
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/e5/$v.co build/e5/$v.s || exit 1
  for n in 1 2 3 4 5 6 7 8; do
    PROBE_I32=$n ./build/tileop_probe_host build/e5/$v.co probe_e5_branches 64 build/e5/in.bin 512 build/e5/out.bin > /dev/null || exit 1
    want=$(( n % 2 == 1 ? 1065353216 : 1073741824 )); want2=$(( n >= 3 ? 1077936128 : 2779096485 ))
    got=$(python3 -c "import struct;d=open('build/e5/out.bin','rb').read();v=struct.unpack('<128I',d);print(sum(1 for x in v[:64] if x!=$want)+sum(1 for x in v[64:] if x!=$want2))")
    echo "$v n=$n wrong lanes: $got"; [ "$got" = 0 ] || bad=1
  done
done
[ $bad = 0 ] && echo "E5_PROBE_OK" || echo "E5_PROBE_FAIL"