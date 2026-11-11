#!/usr/bin/env bash
# Hardware check of the `salu32` forms (s_sub_u32 / s_or_b32 / s_addk_i32 / s_cmp_ge_u32) and v_mul_i32_i24 on the test kernel of
# crates/amdgpu-kernels/tests/b2_operands.rs (uniform_operand_kernel). Usage: HIP_VISIBLE_DEVICES=0 bash tools/sgpr2_probe.sh
set -uo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_guard.sh"   # GPU tests only on GPUs 0-3
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"; mkdir -p build/sgpr2
ROC="${ROCM_PATH:-/shared/apps/ubuntu/opt/rocm-7.0.2}"; CL="${ASM_ROC:-/shared/apps/ubuntu/opt/rocm-10.1.0}/llvm/bin/clang"
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ tests/gpu/tileop_probe_host.cpp -o build/tileop_probe_host -I"$ROC/include" -L"$ROC/lib" -lamdhip64 -Wl,-rpath,"$ROC/lib" || exit 1
bash tools/container/cargo.sh test -q --release -p amdgpu-kernels --test b2_operands write_sgpr2_asm -- --ignored > build/sgpr2/emit.log 2>&1 || { tail -5 build/sgpr2/emit.log; exit 1; }
head -c 1024 /dev/zero > build/sgpr2/in.bin
bad=0
for v in base sgpr2; do
  "$CL" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -mcode-object-version=6 -o build/sgpr2/$v.co build/sgpr2/$v.s || exit 1
  for n in 0 1 5 31 32 63 64 1000; do
    PROBE_I32=$n ./build/tileop_probe_host build/sgpr2/$v.co probe_sgpr2 64 build/sgpr2/in.bin 256 build/sgpr2/out.bin > /dev/null || exit 1
    got=$(python3 - "$n" <<'EOF'
import struct, sys
n = int(sys.argv[1])
M = 0xffffffff
v = struct.unpack('<64I', open('build/sgpr2/out.bin', 'rb').read())
print(sum(1 for t in range(64) if v[t] != (((max(n, t) - min(n, t)) & M) | ((n - t) & M))))
EOF
)
    echo "$v n=$n wrong lanes: $got"; [ "$got" = 0 ] || bad=1
  done
done
[ $bad = 0 ] && echo "SGPR2_OK" || echo "SGPR2_FAIL"
