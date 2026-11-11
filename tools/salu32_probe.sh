#!/usr/bin/env bash
# Hardware check of the `salu32` forms (s_sub_u32 / s_or_b32 / s_addk_i32 / s_cmp_ge_u32) and v_mul_i32_i24 on the test kernel of
# crates/amdgpu-kernels/tests/mops_w5.rs (salu_kernel). Usage: HIP_VISIBLE_DEVICES=0 bash tools/salu32_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/salu32
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
bash tools/container/cargo.sh test -q --release -p amdgpu-kernels --test mops_w5 write_salu_asm -- --ignored > build/salu32/emit.log 2>&1 || { tail -5 build/salu32/emit.log; exit 1; }
head -c 1024 /dev/zero > build/salu32/in.bin
bad=0
for v in base salu32; do
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/salu32/$v.co build/salu32/$v.s || exit 1
  for n in 1 2 3 7 8 20 64; do
    PROBE_I32=$n ./build/tileop_probe_host build/salu32/$v.co probe_salu32 256 build/salu32/in.bin 1024 build/salu32/out.bin > /dev/null || exit 1
    got=$(python3 - "$n" <<'EOF'
import struct, sys
n = int(sys.argv[1]); it = 0; acc = 0
while True:
    acc = (acc + ((n - it) & 0xffffffff) + (it | 1)) & 0xffffffff
    it += 3
    if it >= n: break
def s24(x):
    x &= 0xffffff
    return x - (1 << 24) if x & 0x800000 else x
v = struct.unpack('<256I', open('build/salu32/out.bin', 'rb').read())
print(sum(1 for tid in range(256) if v[tid] != (s24(tid & 255) * s24(acc)) & 0xffffffff))
EOF
)
    echo "$v n=$n wrong lanes: $got"; [ "$got" = 0 ] || bad=1
  done
done
[ $bad = 0 ] && echo "SALU32_OK" || echo "SALU32_FAIL"
